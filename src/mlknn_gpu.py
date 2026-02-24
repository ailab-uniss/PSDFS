"""
GPU-accelerated ML-kNN utilities (CuPy).

This module provides a minimal **ML-kNN** (Multi-Label k-Nearest Neighbours)
implementation where the kNN search is performed on GPU via CuPy.  It is meant
as a drop-in building block for faster evaluation pipelines — especially when
repeatedly evaluating many feature budgets on the same folds.

Typical usage
-------------
>>> from mlknn_gpu import fit_and_predict_gpu
>>> Y_pred, proba = fit_and_predict_gpu(X_train, Y_train, X_test, k=10)

The convenience function `fit_and_predict_gpu` chains all steps:
  1. GPU brute-force kNN search  (train→train and train→test),
  2. ML-kNN model fitting        (label priors + conditional tables),
  3. Prediction                   (MAP decision per label).

You can also call each step individually for more control — see the public API
listed in ``__all__``.

Notes
-----
- Requires ``cupy`` and a working CUDA stack.
- The brute-force kNN graph is O(n_train² · d).  For very large training folds,
  consider an external ANN/kNN index and feed the resulting neighbor indices
  directly into ``fit_mlknn``.

References
----------
Zhang, M.-L., & Zhou, Z.-H. (2007). ML-kNN: A lazy learning approach to
multi-label learning. *Pattern Recognition*, 40(7), 2038–2048.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CuPy availability check
# ---------------------------------------------------------------------------


def _require_cupy():
    """Import and return the ``cupy`` module, raising a clear error if missing."""
    try:
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CuPy is required for GPU ML-kNN. Install e.g. `pip install cupy-cuda12x` "
            "(choose the CUDA version matching your system)."
        ) from e
    return cp


# ---------------------------------------------------------------------------
# GPU kNN search
# ---------------------------------------------------------------------------


def knn_indices_gpu(
    X_train: np.ndarray,
    X_query: np.ndarray,
    *,
    k: int,
    chunk_size: int = 2048,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Brute-force k-nearest-neighbour search on GPU.

    For every row in *X_query*, find the *k* closest rows in *X_train*
    using squared Euclidean distance.  The computation is chunked along
    the query axis so that GPU memory stays bounded.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, d)
        Reference points (uploaded to GPU once).
    X_query : ndarray of shape (n_query, d)
        Query points.
    k : int
        Number of neighbours to return.
    chunk_size : int, default 2048
        Number of query rows processed per GPU kernel launch.
        Decrease if running out of GPU memory.
    exclude_self : bool, default False
        If ``True``, assume *X_query* ≡ *X_train* and exclude the trivial
        self-match (distance = 0) from each neighbourhood.  Useful for
        building the train→train neighbour graph required by ML-kNN.

    Returns
    -------
    indices : ndarray of shape (n_query, k), dtype int64
        Column indices into *X_train* for each query, sorted by distance
        (nearest first).
    """
    cp = _require_cupy()
    Xtr = np.asarray(X_train, dtype=np.float32, order="C")
    Xq = np.asarray(X_query, dtype=np.float32, order="C")
    n_tr = int(Xtr.shape[0])
    n_q = int(Xq.shape[0])
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k >= n_tr and not exclude_self:
        raise ValueError("k must be < n_train.")
    if exclude_self and k >= n_tr:
        raise ValueError("k must be < n_train when exclude_self=True.")

    # Upload training data once; keep on GPU for all chunks.
    Xtr_g = cp.asarray(Xtr)
    tr_norm = cp.sum(Xtr_g * Xtr_g, axis=1)  # (n_tr,)

    out = np.empty((n_q, k), dtype=np.int64)

    for i0 in range(0, n_q, int(chunk_size)):
        i1 = min(n_q, i0 + int(chunk_size))
        Xc_g = cp.asarray(Xq[i0:i1])
        q_norm = cp.sum(Xc_g * Xc_g, axis=1)  # (chunk,)

        # Squared Euclidean: ||a - b||² = ||a||² + ||b||² − 2 a·b
        d2 = q_norm[:, None] + tr_norm[None, :] - 2.0 * (Xc_g @ Xtr_g.T)

        if exclude_self:
            # Mask the diagonal (query i ↔ train i) with +∞.
            rows = cp.arange(i1 - i0, dtype=cp.int64)
            cols = cp.arange(i0, i1, dtype=cp.int64)
            d2[rows, cols] = cp.inf

        # Partial sort to select the k smallest distances, then full-sort
        # those k for deterministic ordering.
        idx = cp.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        d2_k = cp.take_along_axis(d2, idx, axis=1)
        order = cp.argsort(d2_k, axis=1)
        idx = cp.take_along_axis(idx, order, axis=1)

        out[i0:i1] = cp.asnumpy(idx).astype(np.int64, copy=False)

    return out


# ---------------------------------------------------------------------------
# ML-kNN model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLKNNModel:
    """
    Fitted ML-kNN model.

    Stores the label priors and the per-label conditional probability tables
    estimated from the training fold's neighbour graph.

    Attributes
    ----------
    k : int
        Number of neighbours used.
    smooth : float
        Laplace smoothing parameter (typically 1.0).
    prior_pos : ndarray of shape (L,)
        P(y_l = 1)  for each label l.
    prior_neg : ndarray of shape (L,)
        P(y_l = 0)  for each label l.
    cond_pos : ndarray of shape (L, k+1)
        P(C = c | y_l = 1)  — probability that exactly *c* out of *k*
        neighbours carry label *l*, given the instance is positive for *l*.
    cond_neg : ndarray of shape (L, k+1)
        P(C = c | y_l = 0)  — same, conditioned on the instance being
        negative for label *l*.
    """

    k: int
    smooth: float
    prior_pos: np.ndarray  # (L,)
    prior_neg: np.ndarray  # (L,)
    cond_pos: np.ndarray   # (L, k+1)  P(C=c | y=1)
    cond_neg: np.ndarray   # (L, k+1)  P(C=c | y=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _counts_per_label(Y: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """
    Compute the neighbour-label count matrix.

    For each instance *i* and label *l*, count how many of the *k*
    neighbours of *i* carry label *l*.

    Parameters
    ----------
    Y : ndarray of shape (n, L), values in {0, 1}
        Label matrix.
    knn_idx : ndarray of shape (n, k)
        Neighbour indices (rows of *Y*).

    Returns
    -------
    C : ndarray of shape (n, L), dtype int16
        ``C[i, l]`` = number of neighbours of instance *i* that have
        label *l* active.
    """
    Y = np.asarray(Y, dtype=np.int8)
    n, L = Y.shape
    k = int(knn_idx.shape[1])
    C = np.zeros((n, L), dtype=np.int16)
    # Sum label vectors of each neighbour (k is small, loop is efficient).
    for j in range(k):
        C += Y[knn_idx[:, j]]
    return C


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_mlknn(
    Y_train: np.ndarray,
    knn_train: np.ndarray,
    *,
    smooth: float = 1.0,
) -> MLKNNModel:
    """
    Fit ML-kNN priors and conditional tables from precomputed train→train
    neighbours.

    Parameters
    ----------
    Y_train : ndarray of shape (n_train, L), values in {0, 1}
        Training label matrix.
    knn_train : ndarray of shape (n_train, k)
        For each training instance, the indices of its *k* nearest
        neighbours **within the training set** (self excluded).
    smooth : float, default 1.0
        Laplace smoothing parameter for probability estimates.

    Returns
    -------
    model : MLKNNModel
        Fitted model containing priors and conditional tables.
    """
    Y = np.asarray(Y_train, dtype=np.int8)
    n, L = Y.shape
    k = int(knn_train.shape[1])
    smooth = float(smooth)

    C = _counts_per_label(Y, knn_train)  # (n, L), values in 0..k

    pos = (Y == 1)
    neg = ~pos
    n_pos = pos.sum(axis=0).astype(np.float64)
    n_neg = neg.sum(axis=0).astype(np.float64)

    # Label priors with Laplace smoothing.
    prior_pos = (smooth + n_pos) / (2.0 * smooth + float(n))
    prior_neg = 1.0 - prior_pos

    # Conditional tables: P(C = c | y_l = 1/0) for c in {0, …, k}.
    cond_pos = np.zeros((L, k + 1), dtype=np.float64)
    cond_neg = np.zeros((L, k + 1), dtype=np.float64)

    for l in range(L):
        c_l = C[:, l]
        for c in range(k + 1):
            cond_pos[l, c] = smooth + float(np.sum((c_l == c) & pos[:, l]))
            cond_neg[l, c] = smooth + float(np.sum((c_l == c) & neg[:, l]))
        cond_pos[l, :] /= (smooth * float(k + 1) + float(n_pos[l]))
        cond_neg[l, :] /= (smooth * float(k + 1) + float(n_neg[l]))

    return MLKNNModel(
        k=k,
        smooth=smooth,
        prior_pos=prior_pos,
        prior_neg=prior_neg,
        cond_pos=cond_pos,
        cond_neg=cond_neg,
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


def predict_proba_mlknn(
    model: MLKNNModel,
    *,
    Y_train: np.ndarray,
    knn_test: np.ndarray,
) -> np.ndarray:
    """
    Compute per-label posterior probabilities for test instances.

    For each test instance and each label *l*, compute:

    .. math::

        P(y_l = 1 \\mid C = c) = \\frac{P(y_l=1)\\,P(C=c \\mid y_l=1)}
        {P(y_l=1)\\,P(C=c \\mid y_l=1) + P(y_l=0)\\,P(C=c \\mid y_l=0)}

    where *c* is the number of test-instance neighbours carrying label *l*.

    Parameters
    ----------
    model : MLKNNModel
        A fitted ML-kNN model (from ``fit_mlknn``).
    Y_train : ndarray of shape (n_train, L)
        Training label matrix (needed to count neighbour labels).
    knn_test : ndarray of shape (n_test, k)
        For each test instance, indices of its *k* nearest neighbours
        in the training set.

    Returns
    -------
    proba : ndarray of shape (n_test, L)
        Posterior probability P(y_l = 1 | neighbours) for each label.
    """
    Y = np.asarray(Y_train, dtype=np.int8)
    C = _counts_per_label(Y, knn_test)  # (n_test, L)
    n_test, L = C.shape
    k = int(model.k)
    if k != int(knn_test.shape[1]):
        raise ValueError("model.k and knn_test.shape[1] mismatch.")

    proba = np.empty((n_test, L), dtype=np.float64)
    for l in range(L):
        c = np.asarray(C[:, l], dtype=np.int64)
        p1 = model.prior_pos[l] * model.cond_pos[l, c]  # P(y=1) P(C=c|y=1)
        p0 = model.prior_neg[l] * model.cond_neg[l, c]  # P(y=0) P(C=c|y=0)
        proba[:, l] = p1 / np.maximum(p1 + p0, np.finfo(np.float64).eps)
    return proba


def predict_mlknn(
    model: MLKNNModel,
    *,
    Y_train: np.ndarray,
    knn_test: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict binary labels and probabilities for test instances.

    Parameters
    ----------
    model : MLKNNModel
        A fitted ML-kNN model.
    Y_train : ndarray of shape (n_train, L)
        Training label matrix.
    knn_test : ndarray of shape (n_test, k)
        Test→train neighbour indices.
    threshold : float, default 0.5
        Decision threshold applied to the posterior probabilities.

    Returns
    -------
    Y_pred : ndarray of shape (n_test, L), dtype int8
        Binary predictions (1 if proba ≥ threshold, else 0).
    proba : ndarray of shape (n_test, L)
        Posterior probabilities (same as ``predict_proba_mlknn`` output).
    """
    proba = predict_proba_mlknn(model, Y_train=Y_train, knn_test=knn_test)
    Yhat = (proba >= float(threshold)).astype(np.int8)
    return Yhat, proba


# ---------------------------------------------------------------------------
# Convenience: end-to-end GPU pipeline
# ---------------------------------------------------------------------------


def fit_and_predict_gpu(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    k: int = 10,
    smooth: float = 1.0,
    chunk_size: int = 2048,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end GPU ML-kNN: kNN search → fit → predict.

    This is the main convenience entry point.  It chains:

    1. ``knn_indices_gpu(X_train, X_train, …, exclude_self=True)``
       → train neighbour graph,
    2. ``fit_mlknn(Y_train, knn_train)`` → model,
    3. ``knn_indices_gpu(X_train, X_test, …)`` → test neighbour graph,
    4. ``predict_mlknn(model, …)`` → predictions.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, d)
        Training features.
    Y_train : ndarray of shape (n_train, L)
        Training labels in {0, 1}.
    X_test : ndarray of shape (n_test, d)
        Test features.
    k : int, default 10
        Number of neighbours.
    smooth : float, default 1.0
        Laplace smoothing for ML-kNN probability estimates.
    chunk_size : int, default 2048
        GPU chunk size for the kNN search (decrease if GPU OOM).
    threshold : float, default 0.5
        Decision threshold for binary predictions.

    Returns
    -------
    Y_pred : ndarray of shape (n_test, L), dtype int8
        Binary predictions.
    proba : ndarray of shape (n_test, L)
        Posterior probabilities.
    """
    knn_tr = knn_indices_gpu(
        X_train, X_train,
        k=int(k), chunk_size=int(chunk_size), exclude_self=True,
    )
    model = fit_mlknn(Y_train, knn_tr, smooth=float(smooth))
    knn_te = knn_indices_gpu(
        X_train, X_test,
        k=int(k), chunk_size=int(chunk_size), exclude_self=False,
    )
    return predict_mlknn(
        model, Y_train=Y_train, knn_test=knn_te, threshold=float(threshold),
    )


__all__ = [
    "knn_indices_gpu",
    "MLKNNModel",
    "fit_mlknn",
    "predict_proba_mlknn",
    "predict_mlknn",
    "fit_and_predict_gpu",
]
