"""V2-G backbone-neutral semantic features (protocol V2-A1 §V2-A1.5).

This module provides **primary** semantic features that are free from any
backbone-specific assumptions.  Unlike V1's :mod:`ccg.semantic.features`, there is
no hard-coded ``CLIP_TEMPERATURE``, and fixed cosine thresholds (0.7, 0.8) are
**not** part of the primary evidence.

All primary features are functions of:

* ``a[i, j] = cos(z_q_i, z_j)`` — query-to-candidate cosine similarities;
* ``v[t, j] = cos(z_t, z_j)``   — winner-to-candidate inter-similarities;
* ``order``  — B3-score descending rank.

These quantities are representation-space invariant: for any L2-normalised
embedding set (512-d CLIP, 768-d SigLIP, or future backbones) they carry the
same geometric meaning.

**Secondary** (descriptive-only) features — normalized entropy and density@τ —
require a per-backbone calibrated temperature and are computed via
:func:`secondary_descriptive`.  They may be reported but must NOT be the sole
evidence for G2 hard-semantic replication.

Inputs (same contract as V1 ``semantic_stats``):

``z_q``
    ``[n, d]`` L2-normalised query embeddings (any ``d``).
``z_i``
    ``[n, K, d]`` L2-normalised candidate embeddings (``K >= 5``).
``scores``
    ``[n, K]`` B3 raw scores; larger = better.

All internal arithmetic is ``float64``; embeddings are never re-normalised
(assumed already unit-norm by the extraction pipeline).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "V2_PRIMARY_SEMANTIC_NAMES",
    "V2_SECONDARY_NAMES",
    "MANIPULATION_METRICS",
    "v2_primary_semantic_stats",
    "secondary_descriptive",
]

# ---------------------------------------------------------------------------
# Feature name registry (frozen order)
# ---------------------------------------------------------------------------

#: 14 backbone-neutral primary semantic features.
V2_PRIMARY_SEMANTIC_NAMES: Tuple[str, ...] = (
    # Query-side cosine rank features (from B3-order perspective)
    "q_top1_cos",
    "q_top2_cos",
    "q_top3_cos",
    "q_margin12",
    "q_margin13",
    "q_margin23",
    # Winner-to-competitor inter-candidate similarities
    "winner_competitor_max_cos",
    "winner_competitor_mean_cos",
    "winner_competitor_std_cos",
    # Winner vs B3-ranked competitors
    "winner_top2_cos",
    "winner_top3_cos",
    "winner_top5_mean_cos",
    # Normalized rank
    "winner_rank_under_query_similarity",
    # Spread / flatness of query similarity distribution
    "query_cos_spread",
)

#: Secondary descriptive features (NOT used in G2 gate).
V2_SECONDARY_NAMES: Tuple[str, ...] = (
    "norm_entropy",
    "density_070",
    "density_080",
)

#: Manipulation check metric specs: (feature_name, expected_direction).
#: At least 2/3 must match direction AND 1 must have CI excluding 0.
MANIPULATION_METRICS: Tuple[Tuple[str, str], ...] = (
    ("winner_competitor_max_cos", "up"),    # SameCat: competitors more similar to winner
    ("winner_top2_cos", "up"),             # SameCat: B3 2nd is more similar to winner
    ("q_margin12", "down"),                # SameCat: top1-top2 query cos gap shrinks
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MIN_CANDIDATES = 5


def _validate_inputs(z_q, z_i, scores) -> Tuple[NDArray, NDArray, NDArray, int, int]:
    """Validate and return float64 arrays + (n, k)."""
    zq = np.asarray(z_q, dtype=np.float64)
    zi = np.asarray(z_i, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    if zq.ndim != 2:
        raise ValueError(f"`z_q` must be 2-D [n, d], got {zq.shape}")
    if zi.ndim != 3:
        raise ValueError(f"`z_i` must be 3-D [n, K, d], got {zi.shape}")
    if sc.ndim != 2:
        raise ValueError(f"`scores` must be 2-D [n, K], got {sc.shape}")
    n = zq.shape[0]
    if zi.shape[0] != n or sc.shape[0] != n:
        raise ValueError(f"batch mismatch: z_q n={n}, z_i n={zi.shape[0]}, scores n={sc.shape[0]}")
    k = zi.shape[1]
    if k < _MIN_CANDIDATES:
        raise ValueError(f"K must be >= {_MIN_CANDIDATES}, got {k}")
    if sc.shape[1] != k:
        raise ValueError(f"candidate count mismatch: z_i K={k}, scores K={sc.shape[1]}")
    if zi.shape[2] != zq.shape[1]:
        raise ValueError(f"embedding dim mismatch: z_q d={zq.shape[1]}, z_i d={zi.shape[2]}")
    return zq, zi, sc, n, k


def _b3_order(scores: NDArray) -> NDArray:
    """[n, K] int64 descending stable rank."""
    return np.argsort(-scores, axis=1, kind="stable")


def _descending_average_ranks(values: NDArray) -> NDArray:
    """[n, K] float64; highest value gets rank 1; ties share average rank."""
    v = values
    greater = np.sum(v[:, None, :] > v[:, :, None], axis=2)
    equal = np.sum(v[:, :, None] == v[:, None, :], axis=2)
    return greater + (equal + 1.0) / 2.0


# ---------------------------------------------------------------------------
# primary computation
# ---------------------------------------------------------------------------

def v2_primary_semantic_stats(z_q, z_i, scores) -> NDArray:
    """Compute the 14-d backbone-neutral V2 primary semantic features.

    Parameters
    ----------
    z_q : array_like  [n, d]  L2-normalised query embeddings.
    z_i : array_like  [n, K, d]  L2-normalised candidate embeddings.
    scores : array_like  [n, K]  B3 raw scores (larger = better).

    Returns
    -------
    numpy.ndarray  [n, 14]  float64.

    Raises
    ------
    ValueError  On shape mismatch or K < 5.
    """
    zq, zi, sc, n, k = _validate_inputs(z_q, z_i, scores)
    order = _b3_order(sc)
    t = order[:, 0]  # winner index per row
    rows = np.arange(n)

    # --- query-to-candidate cosine similarities (a) ---
    a = np.einsum("nd,nkd->nk", zq, zi)

    q_top1 = a[rows, order[:, 0]]
    q_top2 = a[rows, order[:, 1]]
    q_top3 = a[rows, order[:, 2]]
    q_margin12 = q_top1 - q_top2
    q_margin13 = q_top1 - q_top3
    q_margin23 = q_top2 - q_top3

    # --- winner-to-all-candidate cosine similarities (v_t) ---
    z_t = zi[rows, t]  # [n, d]
    v_t = np.einsum("nd,nkd->nk", z_t, zi)  # [n, K]
    not_winner = np.arange(k)[None, :] != t[:, None]  # [n, K] bool
    masked = np.where(not_winner, v_t, np.nan)
    winner_comp_max = np.nanmax(masked, axis=1)
    winner_comp_mean = np.nanmean(masked, axis=1)
    winner_comp_std = np.nanstd(masked, axis=1)  # population (ddof=0)

    # --- winner vs B3-ranked competitors ---
    winner_top2_cos = v_t[rows, order[:, 1]]
    winner_top3_cos = v_t[rows, order[:, 2]]
    # top5_mean: mean of v[t, order[r]] for r = 1..min(4, K-1)
    n_top5 = min(4, k - 1)
    winner_top5_mean = np.mean(v_t[rows[:, None], order[:, 1 : 1 + n_top5]], axis=1)

    # --- normalized rank of winner under cosine-only ordering ---
    ranks = _descending_average_ranks(a)
    winner_rank_norm = ranks[rows, t] / float(k)

    # --- query cosine spread (std of a across K) ---
    query_cos_spread = np.std(a, axis=1)

    return np.column_stack([
        q_top1,
        q_top2,
        q_top3,
        q_margin12,
        q_margin13,
        q_margin23,
        winner_comp_max,
        winner_comp_mean,
        winner_comp_std,
        winner_top2_cos,
        winner_top3_cos,
        winner_top5_mean,
        winner_rank_norm,
        query_cos_spread,
    ])


# ---------------------------------------------------------------------------
# secondary descriptive (NOT primary gate evidence)
# ---------------------------------------------------------------------------

def secondary_descriptive(
    z_q,
    z_i,
    scores,
    *,
    temperature: Optional[float] = None,
) -> NDArray:
    """Compute secondary descriptive features (entropy, density@tau).

    These are **descriptive only** — they MUST NOT be used as the sole evidence
    for the G2 hard-semantic replication gate.

    Parameters
    ----------
    temperature : float, optional
        The **per-backbone fitted** global temperature (T*).  When ``None``,
        entropy defaults to ``T=1`` (not comparable to V1's ``0.01``).
        The caller must supply the correct calibrated T for the backbone in use;
        the native checkpoint logit_scale (100 or 117.331) is provenance only and
        must NOT be substituted here.

    Returns
    -------
    numpy.ndarray  [n, 3]  float64:
        ``norm_entropy``, ``density_070``, ``density_080``.
    """
    zq, zi, sc, n, k = _validate_inputs(z_q, z_i, scores)
    order = _b3_order(sc)
    t = order[:, 0]
    rows = np.arange(n)

    a = np.einsum("nd,nkd->nk", zq, zi)

    # Normalized entropy using the per-backbone Fitted temperature
    T = float(temperature) if temperature is not None else 1.0
    if T <= 0:
        raise ValueError(f"temperature must be positive, got {T}")
    shifted = a / T
    shifted -= np.max(shifted, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    prob = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore"):
        entropy = -np.sum(np.where(prob > 0, prob * np.log(prob), 0.0), axis=1)
    norm_entropy = 1.0 - entropy / np.log(float(k))

    # Density indicators (threshold-based, descriptive only)
    z_t = zi[rows, t]
    v_t = np.einsum("nd,nkd->nk", z_t, zi)
    not_winner = np.arange(k)[None, :] != t[:, None]
    density_070 = np.sum((v_t > 0.70) & not_winner, axis=1) / float(k - 1)
    density_080 = np.sum((v_t > 0.80) & not_winner, axis=1) / float(k - 1)

    return np.column_stack([norm_entropy, density_070, density_080])
