"""
PSDFS (Python) — Paired Signed-Deviation Feature Selection.

This module implements the **paper-aligned** training-fold feature ranking used
in the experiments.

Given training data:
  - X: (n, d) nonnegative features (typically min–max scaled to [0,1] per fold)
  - Y: (n, L) binary multi-label matrix in {0,1}

PSDFS learns a nonnegative weight matrix W and returns a **single feature
ranking** per fold.

Core design:
1) Signed-deviation lift (center-split):
     X -> [ (X - μ)_+ , (μ - X)_+ ]  with μ computed on the training fold.
   This represents two-sided evidence while keeping W >= 0 and multiplicative
   updates.

2) Rarity-aware instance reweighting (training-only stabilizer):
   labels with fewer positives receive higher weight, which upweights samples
   carrying rare labels when estimating W.

3) Paired group sparsity (when lifted):
   (j+, j-) are coupled so sparsity and ranking operate in the original d-space,
   avoiding lift-induced feature splitting.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PSDFSParams:
    """
    Hyperparameters for PSDFS.

    Notes:
    - `beta` controls the sparsity strength (larger => stronger shrinkage).
    - The algorithm expects **nonnegative** X. In the paper we min–max scale to
      [0,1] per fold (fit on train, applied to test).
    """

    # Sparsity strength
    beta: float = 0.10

    # Multiplicative updates
    max_iter: int = 40

    # Rarity prior strength
    rarity_gamma: float = 2.0

    # Instance reweighting (rarity-aware)
    kappa: float = 1.50
    s_max: float = 3.0

    # Reproducibility / init
    seed: int = 0
    W0: Optional[np.ndarray] = None

    # Lift + grouping
    feature_lift: str = "center_split"  # "none" | "center_split"
    lift_center: str = "mean"  # "mean" | "median" (only when feature_lift="center_split")
    paired_penalty: bool = True  # if lifted: couple (j+,j-) for sparsity and ranking


def _coerce_params(params: PSDFSParams | Dict[str, Any] | None) -> PSDFSParams:
    if params is None:
        return PSDFSParams()
    if isinstance(params, PSDFSParams):
        return params
    allowed = {f.name for f in fields(PSDFSParams)}
    return PSDFSParams(**{k: v for k, v in dict(params).items() if k in allowed})


def _validate_xy(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n,d), got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (n,L), got shape {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must share n; got X {X.shape}, Y {Y.shape}")
    return X, Y


def _lift_center_split(X: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Center-split lift:
      x^+ = max(x - μ, 0),  x^- = max(μ - x, 0)
    """
    center = np.asarray(center, dtype=np.float64)
    if center.ndim == 1:
        center = center.reshape(1, -1)
    Xpos = np.maximum(X - center, 0.0)
    Xneg = np.maximum(center - X, 0.0)
    return np.concatenate([Xpos, Xneg], axis=1)


def psdfs(
    X: np.ndarray,
    Y: np.ndarray,
    params: PSDFSParams | Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute a training-fold feature ranking with PSDFS.

    Returns:
      ranking:
        - 1-based indices of length d_orig when paired ranking is used, else
          1-based indices in the lifted space (length 2*d_orig).
      W:
        nonnegative weights, shape (d_lift, L).
      info:
        config dictionary (useful for logs and sanity checks).
    """
    p = _coerce_params(params)
    X, Y = _validate_xy(X, Y)

    epsv = np.finfo(np.float64).eps
    n, d_orig = X.shape
    L = int(Y.shape[1])

    # ---------------------------------------------------------------------
    # 1) Rarity-aware instance reweighting (training-only).
    #
    # Build a label rarity prior and an instance weight:
    #   s_i = 1 + κ <y_i, prior>, clipped, then re-centered to mean 1.
    #
    # Apply weights via sqrt(s): Xw = diag(sqrt(s)) X, Yw = diag(sqrt(s)) Y.
    # ---------------------------------------------------------------------
    freq = Y.sum(axis=0) + 1.0
    prior = freq ** (-float(p.rarity_gamma))
    prior = prior / (prior.sum() + epsv)

    s = 1.0 + float(p.kappa) * (Y @ prior)
    s = np.minimum(s, float(p.s_max))
    s = s / (s.mean() + epsv)
    sw = np.sqrt(s)
    Yw = Y * sw[:, None]

    # ---------------------------------------------------------------------
    # 2) Signed-deviation lift (optional).
    # ---------------------------------------------------------------------
    lift_mode = str(p.feature_lift).lower().strip()
    if lift_mode == "none":
        X_lift = X
        d_lift = d_orig
        lift_center = "none"
    elif lift_mode == "center_split":
        center_mode = str(p.lift_center).lower().strip()
        if center_mode == "mean":
            center = X.mean(axis=0, keepdims=True)
        elif center_mode == "median":
            center = np.median(X, axis=0, keepdims=True)
        else:
            raise ValueError("lift_center must be 'mean' or 'median'.")
        X_lift = _lift_center_split(X, center=center)
        d_lift = 2 * d_orig
        lift_center = center_mode
    else:
        raise ValueError(f"Unknown feature_lift: {p.feature_lift!r}")

    Xw = X_lift * sw[:, None]

    # ---------------------------------------------------------------------
    # 3) Multiplicative updates for nonnegative reconstruction with group sparsity.
    #
    # Schematic objective:
    #   min_{W>=0} ||Xw W - Yw||_F^2 + beta * sum_groups ||W_g||_2
    #
    # When lifted and paired_penalty=True, groups correspond to (j+, j-).
    # ---------------------------------------------------------------------
    if p.W0 is not None:
        W = np.asarray(p.W0, dtype=np.float64).copy()
        if W.shape != (d_lift, L):
            raise ValueError(f"W0 has shape {W.shape}, expected {(d_lift, L)}")
    else:
        rng = np.random.default_rng(int(p.seed))
        W = rng.random((d_lift, L), dtype=np.float64)

    XtX = Xw.T @ Xw
    XtY = Xw.T @ Yw
    paired = bool(p.paired_penalty)

    for _ in range(int(p.max_iter)):
        # ℓ2,1 surrogate weights (iteratively reweighted group-L2).
        if d_lift == d_orig:
            row_norms = np.sqrt((W * W).sum(axis=1)) + epsv
            inv2norm = 1.0 / (2.0 * row_norms)
        elif d_lift == 2 * d_orig:
            if paired:
                Wp = W[:d_orig]
                Wn = W[d_orig:]
                pair_norms = np.sqrt((Wp * Wp).sum(axis=1) + (Wn * Wn).sum(axis=1)) + epsv
                inv2 = 1.0 / (2.0 * pair_norms)
                inv2norm = np.concatenate([inv2, inv2], axis=0)
            else:
                row_norms = np.sqrt((W * W).sum(axis=1)) + epsv
                inv2norm = 1.0 / (2.0 * row_norms)
        else:
            raise ValueError("Lifted dimension incompatible.")

        # Multiplicative update: W <- W ⊙ (XtY / (XtX W + beta * G(W))).
        numer = XtY
        denom = (XtX @ W) + float(p.beta) * (inv2norm[:, None] * W)
        W = W * (numer / np.maximum(denom, epsv))

    # ---------------------------------------------------------------------
    # Feature ranking by group norm.
    # ---------------------------------------------------------------------
    if d_lift == d_orig:
        scores = np.sqrt((W * W).sum(axis=1))
        ranking0 = np.argsort(-scores, kind="mergesort")
        ranking = (ranking0 + 1).astype(np.int64)
    elif d_lift == 2 * d_orig:
        if paired:
            Wp = W[:d_orig]
            Wn = W[d_orig:]
            scores = np.sqrt((Wp * Wp).sum(axis=1) + (Wn * Wn).sum(axis=1))
            ranking0 = np.argsort(-scores, kind="mergesort")
            ranking = (ranking0 + 1).astype(np.int64)
        else:
            scores = np.sqrt((W * W).sum(axis=1))
            ranking0 = np.argsort(-scores, kind="mergesort")
            ranking = (ranking0 + 1).astype(np.int64)
    else:
        raise ValueError("Lifted dimension incompatible.")

    info: Dict[str, Any] = {
        "feature_lift": lift_mode,
        "lift_center": lift_center,
        "paired_penalty": paired,
        "beta": float(p.beta),
        "max_iter": int(p.max_iter),
        "rarity_gamma": float(p.rarity_gamma),
        "kappa": float(p.kappa),
        "s_max": float(p.s_max),
        "n": int(n),
        "d_orig": int(d_orig),
        "d_lift": int(d_lift),
        "L": int(L),
    }
    return ranking, W, info


def _make_toy_dataset(seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Small synthetic multi-label dataset for demos / sanity checks.

    Returns:
      X: (n,d) nonnegative features in [0,1]
      Y: (n,L) binary labels
      active: indices of ground-truth informative features (0-based)
    """
    rng = np.random.default_rng(int(seed))
    n, d, L = 250, 40, 3

    # Nonnegative, moderately skewed features, then scaled to [0,1].
    X = rng.gamma(shape=2.0, scale=1.0, size=(n, d)).astype(np.float64)
    X = (X - X.min(axis=0, keepdims=True)) / (X.ptp(axis=0, keepdims=True) + 1e-12)

    active = np.asarray([1, 5, 11, 19, 27], dtype=np.int64)
    # Construct labels from simple, sparse rules so the informative features are
    # recoverable by a ranking method.
    noise = 0.02 * rng.standard_normal((n,))

    s0 = 2.5 * X[:, active[0]] + 2.0 * X[:, active[1]] + 1.5 * X[:, active[2]] + noise
    s1 = 2.2 * X[:, active[2]] + 2.0 * X[:, active[3]] + 1.2 * X[:, active[4]] - noise
    s2 = 2.4 * X[:, active[0]] + 1.8 * X[:, active[3]] + 1.0 * X[:, active[4]] + 0.5 * noise

    # Choose thresholds so positives are neither too rare nor too dense.
    t0 = float(np.quantile(s0, 0.70))
    t1 = float(np.quantile(s1, 0.75))
    t2 = float(np.quantile(s2, 0.72))

    Y = np.zeros((n, L), dtype=np.float64)
    Y[:, 0] = (s0 > t0).astype(np.float64)
    Y[:, 1] = (s1 > t1).astype(np.float64)
    Y[:, 2] = (s2 > t2).astype(np.float64)
    return X, Y, active


def _load_mat_xy(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a `.mat` file containing either (X_train,Y_train) or (X,Y).
    """
    try:
        import scipy.io as sio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required to load .mat files. Install with: pip install scipy") from e

    obj = sio.loadmat(str(mat_path))
    if "X_train" in obj and "Y_train" in obj:
        X = obj["X_train"]
        Y = obj["Y_train"]
    elif "X" in obj and "Y" in obj:
        X = obj["X"]
        Y = obj["Y"]
    else:
        keys = sorted([k for k in obj.keys() if not k.startswith("__")])
        raise KeyError(f"{mat_path} does not contain X_train/Y_train or X/Y. Keys: {keys}")
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)


def _minmax_scale_01(X: np.ndarray) -> np.ndarray:
    """
    Feature-wise min–max scaling to [0,1].

    This is *not* applied automatically in `psdfs()` because callers may want
    to control preprocessing externally. It exists to keep the CLI usable on
    arbitrary inputs.
    """
    X = np.asarray(X, dtype=np.float64)
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    den = np.maximum(mx - mn, 1e-12)
    return (X - mn) / den


def main(argv: Optional[list[str]] = None) -> int:  # pragma: no cover
    import argparse
    import json

    ap = argparse.ArgumentParser(description="PSDFS CLI: demo + ranking on real datasets.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_demo = sub.add_parser("demo", help="Run PSDFS on a small synthetic dataset (no files needed).")
    ap_demo.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    ap_demo.add_argument("--top", type=int, default=10, help="How many top-ranked features to print (default: 10).")
    ap_demo.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path to save X/Y.")
    ap_demo.add_argument("--beta", type=float, default=0.10)
    ap_demo.add_argument("--max-iter", type=int, default=40)
    ap_demo.add_argument("--lift-center", type=str, default="mean", choices=["mean", "median"])

    ap_rank = sub.add_parser("rank", help="Compute a 1-based feature ranking from a dataset.")
    src = ap_rank.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=Path, default=None, help="Input .npz containing arrays X and Y.")
    src.add_argument("--mat", type=Path, default=None, help="Input .mat containing X_train/Y_train or X/Y.")
    ap_rank.add_argument("--top", type=int, default=20, help="How many top-ranked features to print (default: 20).")
    ap_rank.add_argument(
        "--out-ranking",
        type=Path,
        default=None,
        help="Optional path to write the full ranking as a CSV (1-based indices).",
    )
    ap_rank.add_argument(
        "--minmax-01",
        action="store_true",
        help="Apply feature-wise min–max scaling to [0,1] before running PSDFS.",
    )
    ap_rank.add_argument("--beta", type=float, default=0.10)
    ap_rank.add_argument("--max-iter", type=int, default=40)
    ap_rank.add_argument("--lift-center", type=str, default="mean", choices=["mean", "median"])
    ap_rank.add_argument("--print-info-json", action="store_true", help="Print the `info` dict as JSON.")

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        X, Y, active = _make_toy_dataset(seed=int(args.seed))
        print(f"[demo] Toy dataset: X={X.shape}, Y={Y.shape}, active_features(0-based)={active.tolist()}")

        if args.save_npz is not None:
            out = Path(args.save_npz)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(out), X=X, Y=Y, active_features=active)
            print(f"[demo] Wrote: {out}")

        params = PSDFSParams(
            beta=float(args.beta),
            max_iter=int(args.max_iter),
            feature_lift="center_split",
            lift_center=str(args.lift_center),
            paired_penalty=True,
            seed=int(args.seed),
        )
        ranking_1based, _W, info = psdfs(X, Y, params)

        top = int(max(1, args.top))
        print(f"Top-{top} ranking (1-based): {ranking_1based[:top].tolist()}")
        top0 = set((ranking_1based[:top] - 1).tolist())
        hit = [int(a) for a in active.tolist() if int(a) in top0]
        print(f"[demo] Hits in Top-{top} (0-based): {hit}")
        print("Info:", info)
        return 0

    if args.cmd == "rank":
        if args.npz is not None:
            obj = np.load(str(args.npz), allow_pickle=False)
            if "X" not in obj or "Y" not in obj:
                raise KeyError(f"{args.npz} must contain arrays 'X' and 'Y'. Keys: {list(obj.keys())}")
            X = np.asarray(obj["X"], dtype=np.float64)
            Y = np.asarray(obj["Y"], dtype=np.float64)
            print(f"[npz] Loaded: X={X.shape}, Y={Y.shape} from {args.npz}")
        else:
            X, Y = _load_mat_xy(Path(args.mat))
            print(f"[mat] Loaded: X={X.shape}, Y={Y.shape} from {args.mat}")

        if bool(args.minmax_01):
            X = _minmax_scale_01(X)

        params = PSDFSParams(
            beta=float(args.beta),
            max_iter=int(args.max_iter),
            feature_lift="center_split",
            lift_center=str(args.lift_center),
            paired_penalty=True,
            seed=0,
        )
        ranking_1based, _W, info = psdfs(X, Y, params)

        top = int(max(1, args.top))
        print(f"Top-{top} ranking (1-based): {ranking_1based[:top].tolist()}")

        if args.out_ranking is not None:
            out = Path(args.out_ranking)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(str(out), ranking_1based.astype(np.int64), delimiter=",", fmt="%d")
            print(f"[rank] Wrote: {out}")

        if bool(args.print_info_json):
            print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    raise SystemExit("Unknown command.")


__all__ = ["PSDFSParams", "psdfs", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
