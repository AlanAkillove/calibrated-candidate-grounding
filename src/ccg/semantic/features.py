"""E1 handcrafted semantic-geometry statistics (protocol A7.3, frozen 16-d).

This module is pure NumPy (no torch) and fully deterministic: the same inputs
always produce bit-identical outputs, and no random or global state is read.
:func:`semantic_stats` returns a ``[n, 16]`` matrix whose columns follow exactly
the frozen order of :data:`SEMANTIC_STAT_NAMES` (see Amendment A7.3 of
``docs/research_protocol.md``).

Inputs / assumptions
--------------------
``z_q``
    ``[n, d]`` query embeddings, already L2-normalised.
``z_i``
    ``[n, K, d]`` candidate embeddings, already L2-normalised.
``scores``
    ``[n, K]`` B3 raw scores (any floating dtype); larger is better.

Because ``z_q`` and ``z_i`` are assumed L2-normalised, cosine similarity is a
plain dot product and the inputs are **never re-normalised** here. All internal
arithmetic is promoted to ``float64``.

The softmax temperature is fixed at ``T = 1`` and the density thresholds are the
frozen :data:`DENSITY_TAUS`; neither is searchable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DENSITY_TAUS",
    "SEMANTIC_STAT_NAMES",
    "b3_order",
    "descending_average_ranks",
    "semantic_stats",
    "top5_indices",
]

SEMANTIC_STAT_NAMES: Tuple[str, ...] = (
    "clip_top1",
    "clip_top2",
    "clip_margin12",
    "clip_entropy",
    "clip_normH",
    "clip_rank_top1",
    "cand_vmax",
    "cand_vmean",
    "cand_vstd",
    "cand_top12_sim",
    "cand_top15_mean",
    "density_070",
    "density_080",
    "q_top3",
    "q_margin13",
    "cand_top13_sim",
)

DENSITY_TAUS: Tuple[float, ...] = (0.7, 0.8)

#: Frozen CLIP inference temperature.  OpenCLIP ViT-B/32 (``openai``) learns
#: ``logit_scale = 100`` exactly, i.e. cosine similarities enter the CLIP
#: softmax as ``a / T`` with ``T = 0.01``.  Using ``T = 1`` saturates the
#: softmax at CLIP cosine scale (the entropy would collapse to ``log K`` for
#: every row); the revision to ``T = 0.01`` was recorded in protocol A7.3
#: *before* any Phase 1 results were produced.
CLIP_TEMPERATURE: float = 0.01

_MIN_CANDIDATES = 5


def _as_float_2d(values, name: str) -> NDArray:
    """Return ``values`` as a ``float64`` 2-D array or raise ``ValueError``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D, got shape {arr.shape}")
    return arr


def b3_order(scores) -> NDArray:
    """Return the per-row B3 descending order as stable ranking indices.

    Parameters
    ----------
    scores : array_like
        ``[n, K]`` score matrix; ``K >= 2``.

    Returns
    -------
    numpy.ndarray
        ``[n, K]`` ``int64`` index array equal to
        ``np.argsort(-scores, axis=1, kind="stable")``. Ties are broken by
        ascending column index (the smaller candidate index comes first).

    Raises
    ------
    ValueError
        If ``scores`` is not 2-D or has fewer than 2 columns.
    """
    s = _as_float_2d(scores, "scores")
    if s.shape[1] < 2:
        raise ValueError(f"`scores` needs at least 2 candidates (K >= 2), got K={s.shape[1]}")
    return np.argsort(-s, axis=1, kind="stable")


def top5_indices(scores) -> NDArray:
    """Return the indices of the 5 highest-scoring candidates per row.

    Parameters
    ----------
    scores : array_like
        ``[n, K]`` score matrix; ``K >= 5``.

    Returns
    -------
    numpy.ndarray
        ``[n, 5]`` ``int64`` array: the first 5 columns of :func:`b3_order`,
        i.e. the 5 largest scores per row with stable tie-breaking.

    Raises
    ------
    ValueError
        If ``scores`` is not 2-D or has fewer than 5 columns.
    """
    s = _as_float_2d(scores, "scores")
    if s.shape[1] < _MIN_CANDIDATES:
        raise ValueError(
            f"`scores` needs at least {_MIN_CANDIDATES} candidates (K >= {_MIN_CANDIDATES}), "
            f"got K={s.shape[1]}"
        )
    return b3_order(s)[:, :_MIN_CANDIDATES]


def descending_average_ranks(values) -> NDArray:
    """Return per-element average ranks under a *descending* ordering.

    Parameters
    ----------
    values : array_like
        ``[n, K]`` matrix of values.

    Returns
    -------
    numpy.ndarray
        ``[n, K]`` ``float64`` array of 1-based ranks where the largest value in
        each row receives rank ``1``. Equal values share their average rank,
        computed as ``#{strictly greater} + (#{equal} + 1) / 2``. The result is
        deterministic and independent of input column order.

    Examples
    --------
    >>> descending_average_ranks(np.array([[2.0, 2.0, 1.0]]))
    array([[1.5, 1.5, 3. ]])
    """
    v = _as_float_2d(values, "values")
    greater = np.sum(v[:, None, :] > v[:, :, None], axis=2)
    equal = np.sum(v[:, :, None] == v[:, None, :], axis=2)
    return greater + (equal + 1.0) / 2.0


def _softmax_rows(a: NDArray) -> NDArray:
    """Numerically stable row-wise softmax of ``a / CLIP_TEMPERATURE``."""
    shifted = a / CLIP_TEMPERATURE
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _entropy_rows(prob: NDArray) -> NDArray:
    """Row-wise Shannon entropy (natural log) of a probability matrix."""
    with np.errstate(divide="ignore"):
        logp = np.log(prob)
    terms = np.where(prob > 0.0, prob * logp, 0.0)
    return -np.sum(terms, axis=1)


def semantic_stats(z_q, z_i, scores) -> NDArray:
    """Compute the frozen E1 16-d semantic statistics.

    Parameters
    ----------
    z_q : array_like
        ``[n, d]`` L2-normalised query embeddings.
    z_i : array_like
        ``[n, K, d]`` L2-normalised candidate embeddings; ``K >= 5``.
    scores : array_like
        ``[n, K]`` B3 raw scores (any floating dtype); larger is better.

    Returns
    -------
    numpy.ndarray
        ``[n, 16]`` ``float64`` matrix. Column ``j`` corresponds to
        ``SEMANTIC_STAT_NAMES[j]``. Because the embeddings are L2-normalised,
        cosine similarities are plain dot products and are **not** re-normalised.

    Raises
    ------
    ValueError
        If the arrays are not compatible (2-D/3-D shape mismatch between
        ``z_q``/``z_i``/``scores``, differing batch size or embedding width, or
        ``K < 5``).

    Notes
    -----
    Let ``order`` be the B3 descending order (see :func:`b3_order`), ``t`` the
    winner ``order[:, 0]``, ``a[i, j] = cos(z_q_i, z_i_j)`` and
    ``v[t, j] = cos(z_i_t, z_i_j)``. The columns are, in order::

        clip_top1       = a[t]
        clip_top2       = a[order[:, 1]]
        clip_margin12   = clip_top1 - clip_top2
        clip_entropy    = H(softmax(a; T=CLIP_TEMPERATURE))   (nats, T = 0.01)
        clip_normH      = 1 - clip_entropy / log(K)
        clip_rank_top1  = descending_rank(a)[t] / K
        cand_vmax       = max_{j != t} v[t, j]
        cand_vmean      = mean_{j != t} v[t, j]
        cand_vstd       = std_{j != t} v[t, j]      (population, ddof=0)
        cand_top12_sim  = v[t, order[:, 1]]
        cand_top15_mean = mean_{r=2..5} v[t, order[:, r]]
        density_070     = mean_{j != t}(v[t, j] > 0.7)
        density_080     = mean_{j != t}(v[t, j] > 0.8)
        q_top3          = a[order[:, 2]]
        q_margin13      = clip_top1 - q_top3
        cand_top13_sim  = v[t, order[:, 2]]

    The density indicators use a **strict** greater-than comparison.
    """
    zq = np.asarray(z_q, dtype=np.float64)
    zi = np.asarray(z_i, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)

    if zq.ndim != 2:
        raise ValueError(f"`z_q` must be 2-D [n, d], got shape {zq.shape}")
    if zi.ndim != 3:
        raise ValueError(f"`z_i` must be 3-D [n, K, d], got shape {zi.shape}")
    if sc.ndim != 2:
        raise ValueError(f"`scores` must be 2-D [n, K], got shape {sc.shape}")

    n = zq.shape[0]
    if zi.shape[0] != n:
        raise ValueError(
            f"batch size mismatch: `z_q` has {n} rows but `z_i` has {zi.shape[0]}"
        )
    if sc.shape[0] != n:
        raise ValueError(
            f"batch size mismatch: `z_q` has {n} rows but `scores` has {sc.shape[0]}"
        )

    k = zi.shape[1]
    if k < _MIN_CANDIDATES:
        raise ValueError(
            f"`semantic_stats` needs at least {_MIN_CANDIDATES} candidates (K >= "
            f"{_MIN_CANDIDATES}), got K={k}"
        )
    if sc.shape[1] != k:
        raise ValueError(
            f"candidate-count mismatch: `z_i` has K={k} but `scores` has K={sc.shape[1]}"
        )
    if zi.shape[2] != zq.shape[1]:
        raise ValueError(
            f"embedding-dim mismatch: `z_q` has d={zq.shape[1]} but `z_i` has d={zi.shape[2]}"
        )

    order = b3_order(sc)
    t = order[:, 0]
    rows = np.arange(n)

    a = np.einsum("nd,nkd->nk", zq, zi)

    clip_top1 = a[rows, t]
    clip_top2 = a[rows, order[:, 1]]
    clip_margin12 = clip_top1 - clip_top2

    prob = _softmax_rows(a)
    clip_entropy = _entropy_rows(prob)
    clip_normH = 1.0 - clip_entropy / np.log(float(k))

    ranks = descending_average_ranks(a)
    clip_rank_top1 = ranks[rows, t] / float(k)

    z_t = zi[rows, t]
    v_t = np.einsum("nd,nkd->nk", z_t, zi)
    not_winner = np.arange(k)[None, :] != t[:, None]
    masked = np.where(not_winner, v_t, np.nan)
    cand_vmax = np.nanmax(masked, axis=1)
    cand_vmean = np.nanmean(masked, axis=1)
    cand_vstd = np.nanstd(masked, axis=1)

    cand_top12_sim = v_t[rows, order[:, 1]]
    cand_top13_sim = v_t[rows, order[:, 2]]
    cand_top15_mean = np.mean(v_t[rows[:, None], order[:, 1:5]], axis=1)

    density_070 = np.sum((v_t > DENSITY_TAUS[0]) & not_winner, axis=1) / float(k - 1)
    density_080 = np.sum((v_t > DENSITY_TAUS[1]) & not_winner, axis=1) / float(k - 1)

    q_top3 = a[rows, order[:, 2]]
    q_margin13 = clip_top1 - q_top3

    return np.column_stack(
        [
            clip_top1,
            clip_top2,
            clip_margin12,
            clip_entropy,
            clip_normH,
            clip_rank_top1,
            cand_vmax,
            cand_vmean,
            cand_vstd,
            cand_top12_sim,
            cand_top15_mean,
            density_070,
            density_080,
            q_top3,
            q_margin13,
            cand_top13_sim,
        ]
    )
