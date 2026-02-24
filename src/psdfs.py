"""
PSDFS (Python) — Paired Signed-Deviation Feature Selection.

This is the **paper-aligned** training-fold feature-ranking implementation used
throughout the experiments. Given nonnegative training features `X` and labels
`Y`, it learns a nonnegative weight matrix `W` and returns a **single feature
ranking** (descending importance).

Key ingredients (all *training-only*, no test leakage):

1) Signed-deviation lift (center-split)
   - Expands the nonnegative feature space to represent **two-sided evidence**
     while preserving `W >= 0` and multiplicative updates.
   - `X -> [ (X - μ)_+ , (μ - X)_+ ]` where `μ` is a per-feature center statistic
     computed on the training fold (mean by default; median supported).

2) Rarity-aware instance reweighting
   - Builds an instance weight from a rarity prior over labels (labels with
     fewer positives receive higher weight).
   - Stabilizes learning when supervision density is heterogeneous.

3) Embedded nonnegative reconstruction + group sparsity
   - Optimizes a convex-in-`W` objective with multiplicative updates.
   - Ranks features by row/group norms; with lifting, ranking is performed in
     the *original* feature space via paired group norms (default).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PSDFSParams:
    beta: float = 0.10
    max_iter: int = 40
    rarity_gamma: float = 2.0

    # Instance reweighting (rarity-aware)
    kappa: float = 1.50
    s_max: float = 3.0

    # Reproducibility / init
    seed: int = 0
    W0: Optional[np.ndarray] = None

    # Lift + grouping
    feature_lift: str = "center_split"  # 'none' or 'center_split'
    lift_center: str = "mean"  # used only when feature_lift='center_split': 'mean' | 'median'
    paired_penalty: bool = True  # if lifted: use paired group sparsity and paired ranking


def _lift_center_split(X: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float64)
    if center.ndim == 1:
        center = center.reshape(1, -1)
    Xpos = np.maximum(X - center, 0.0)
    Xneg = np.maximum(center - X, 0.0)
    return np.concatenate([Xpos, Xneg], axis=1), center


def psdfs(
    X: np.ndarray,
    Y: np.ndarray,
    params: PSDFSParams | Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute a training-fold feature ranking with PSDFS.

    Returns:
      ranking: 1-based indices (length d_orig) if paired ranking is used, else
               1-based indices in the lifted space (length 2d_orig).
      W:       nonnegative weights, shape (d_lift, L).
      info:    small config dict (for logging).
    """
    if params is None:
        p = PSDFSParams()
    elif isinstance(params, PSDFSParams):
        p = params
    else:
        allowed = {f.name for f in fields(PSDFSParams)}
        p = PSDFSParams(**{k: v for k, v in dict(params).items() if k in allowed})

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    epsv = np.finfo(np.float64).eps

    n, d_orig = X.shape
    L = int(Y.shape[1])

    # --- rarity prior (for instance weights)
    freq = Y.sum(axis=0) + 1.0
    prior = freq ** (-float(p.rarity_gamma))
    prior = prior / (prior.sum() + epsv)

    # --- instance reweighting: s_i = 1 + κ <y_i, prior>, clipped and re-centered to mean 1
    s = 1.0 + float(p.kappa) * (Y @ prior)
    s = np.minimum(s, float(p.s_max))
    s = s / (s.mean() + epsv)
    sw = np.sqrt(s)
    Yw = Y * sw[:, None]

    # --- feature lift
    lift_mode = str(p.feature_lift).lower().strip()
    if lift_mode == "none":
        X_lift = X
        d = d_orig
    elif lift_mode == "center_split":
        center_mode = str(p.lift_center).lower().strip()
        if center_mode == "mean":
            center = X.mean(axis=0, keepdims=True)
        elif center_mode == "median":
            center = np.median(X, axis=0, keepdims=True)
        else:
            raise ValueError("lift_center must be 'mean' or 'median'.")
        X_lift, _center = _lift_center_split(X, center=center)
        d = 2 * d_orig
    else:
        raise ValueError(f"Unknown feature_lift: {p.feature_lift!r}")

    Xw = X_lift * sw[:, None]

    # --- init W
    if p.W0 is not None:
        W = np.asarray(p.W0, dtype=np.float64).copy()
        if W.shape != (d, L):
            raise ValueError(f"W0 has shape {W.shape}, expected {(d, L)}")
    else:
        rng = np.random.default_rng(int(p.seed))
        W = rng.random((d, L), dtype=np.float64)

    XtX = Xw.T @ Xw
    XtY = Xw.T @ Yw

    paired = bool(p.paired_penalty)

    for _it in range(int(p.max_iter)):
        # ℓ2,1 subgradient weights (paired for lifted features when requested)
        if d == d_orig:
            row_norms = np.sqrt((W * W).sum(axis=1)) + epsv
            inv2norm = 1.0 / (2.0 * row_norms)
        elif d == 2 * d_orig:
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

        Up = XtY
        Dw = (XtX @ W) + float(p.beta) * (inv2norm[:, None] * W)
        W = W * (Up / np.maximum(Dw, epsv))

    # --- ranking
    if d == d_orig:
        scores = np.sqrt((W * W).sum(axis=1))
        ranking0 = np.argsort(-scores, kind="mergesort")
        ranking = (ranking0 + 1).astype(np.int64)
    elif d == 2 * d_orig:
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
        "lift_center": str(p.lift_center) if lift_mode == "center_split" else "none",
        "paired_penalty": paired,
        "beta": float(p.beta),
        "max_iter": int(p.max_iter),
        "rarity_gamma": float(p.rarity_gamma),
        "kappa": float(p.kappa),
        "s_max": float(p.s_max),
        "n": int(n),
        "d_orig": int(d_orig),
        "d_lift": int(d),
        "L": int(L),
    }
    return ranking, W, info


# Legacy aliases kept inside this single module for convenience.
D2GFSParams = PSDFSParams
D2FSParams = PSDFSParams

d2gfs = psdfs
d2fs = psdfs

dagfs_v2 = psdfs

__all__ = [
    "PSDFSParams",
    "psdfs",
    "dagfs_v2",
    "D2GFSParams",
    "D2FSParams",
    "d2gfs",
    "d2fs",
]

