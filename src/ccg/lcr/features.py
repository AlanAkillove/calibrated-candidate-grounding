"""Frozen LCR relation features (protocol ``results/v2_local_competition/protocol.json``).

Per row (query ``q``, candidate set ``C_K`` with frozen B3 scores ``s``) this
module computes:

* the winner ``t = argmax_i s_i`` (stable descending order);
* the local competition set = the ``M = 4`` highest-scoring non-winner
  candidates (stable ordering, so ties resolve deterministically);
* the 7-d relation vector of every competitor ``j``::

      r_j = [ ds_norm, dp, a_t, a_j, a_t - a_j, v_tj, rank_j_over_k ]

  with ``ds_norm = (s_t - s_j) / (std(s) + eps)`` (``std`` over all K scores,
  ddof=0), ``dp = p_t - p_j`` from the stable softmax at the scorer's frozen
  corrected temperature ``T``, ``a = cos(z_q, z_.)``, ``v_tj = cos(z_t, z_j)``
  and ``rank_j_over_k`` the 1-based descending B3 rank of ``j`` over ``K``
  (winner = 1, competitors therefore carry ranks 2..M+1);

* the 3-d ambiguity-gate input ``[max_j v_tj, mean_j v_tj, ds_norm(rank-2)]``.

**Input discipline (protocol section 6).** The functions read exactly
``(scores, z_q, z_i, temperature)`` and nothing else.  Raw embedding
projections, GT category, objectness, target IoU and hard/random regime labels
are forbidden inputs and are never accepted by any signature here.

Embeddings are assumed L2-normalised by the extraction pipeline (the embedding
store validates this); cosine similarity is therefore the plain dot product,
exactly as in the V2 semantic features.  All internal arithmetic is float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EPS",
    "GATE_INPUT_NAMES",
    "M_COMPETITORS",
    "RELATION_FEATURE_NAMES",
    "RelationBatch",
    "relation_features",
    "winning_index",
]

#: Number of local competitors per row (frozen; never searched).
M_COMPETITORS = 4
#: Stability epsilon of the normalised score gap (frozen).
EPS = 1e-8
#: Column names of the 7-d relation vector, in frozen order.
RELATION_FEATURE_NAMES: tuple[str, ...] = (
    "ds_norm",
    "dp",
    "a_t",
    "a_j",
    "a_t_minus_a_j",
    "v_tj",
    "rank_j_over_k",
)
#: Column names of the 3-d ambiguity-gate input, in frozen order.
GATE_INPUT_NAMES: tuple[str, ...] = ("v_max", "v_mean", "ds_norm_top2")

#: Minimum candidate count that guarantees ``M`` competitors (K >= 5 regime).
_MIN_K = 5


def winning_index(scores: Any) -> NDArray[np.int64]:
    """``argmax`` with the frozen stable tie resolution (lower index wins)."""
    sc = np.asarray(scores, dtype=np.float64)
    if sc.ndim != 2:
        raise ValueError(f"`scores` must be 2-D [n, K], got {sc.shape}")
    return np.argsort(-sc, axis=1, kind="stable")[:, 0].astype(np.int64, copy=False)


@dataclass(frozen=True)
class RelationBatch:
    """Relation features of one ``(n, K)`` score block.

    Attributes
    ----------
    order:
        ``[n, K]`` int64 stable descending B3 order (column 0 = winner).
    winner_idx / competitor_idx:
        ``[n]`` / ``[n, M]`` local indices into the candidate set.
    r:
        ``[n, M, 7]`` float64 relation vectors (columns in
        :data:`RELATION_FEATURE_NAMES` order; ``r[:, j]`` belongs to
        ``competitor_idx[:, j]``).
    gate:
        ``[n, 3]`` float64 ambiguity-gate input (columns in
        :data:`GATE_INPUT_NAMES` order).
    k:
        candidate-set size.
    """

    order: NDArray[np.int64]
    winner_idx: NDArray[np.int64]
    competitor_idx: NDArray[np.int64]
    r: NDArray[np.float64]
    gate: NDArray[np.float64]
    k: int

    @property
    def n(self) -> int:
        return int(self.winner_idx.shape[0])

    @property
    def m(self) -> int:
        return int(self.competitor_idx.shape[1])


def _validate(scores: Any, z_q: Any, z_i: Any, temperature: Any, m: int) -> tuple:
    sc = np.asarray(scores, dtype=np.float64)
    zq = np.asarray(z_q, dtype=np.float64)
    zi = np.asarray(z_i, dtype=np.float64)
    if sc.ndim != 2:
        raise ValueError(f"`scores` must be 2-D [n, K], got {sc.shape}")
    if zq.ndim != 2:
        raise ValueError(f"`z_q` must be 2-D [n, d], got {zq.shape}")
    if zi.ndim != 3:
        raise ValueError(f"`z_i` must be 3-D [n, K, d], got {zi.shape}")
    n, k = sc.shape
    if zq.shape[0] != n or zi.shape[0] != n:
        raise ValueError(
            f"row mismatch: scores n={n}, z_q n={zq.shape[0]}, z_i n={zi.shape[0]}"
        )
    if zi.shape[1] != k or sc.shape[1] != k:
        raise ValueError(
            f"candidate count mismatch: scores K={k}, z_i K={zi.shape[1]}"
        )
    if zi.shape[2] != zq.shape[1]:
        raise ValueError(
            f"embedding dim mismatch: z_q d={zq.shape[1]}, z_i d={zi.shape[2]}"
        )
    if k < _MIN_K:
        raise ValueError(f"K must be >= {_MIN_K} (protocol primary regime), got {k}")
    if int(m) < 1 or int(m) > k - 1:
        raise ValueError(f"M must be in [1, K-1], got m={m} for K={k}")
    if not np.all(np.isfinite(sc)):
        raise ValueError("`scores` contains non-finite values")
    if not np.all(np.isfinite(zq)) or not np.all(np.isfinite(zi)):
        raise ValueError("embeddings contain non-finite values")
    t = float(temperature)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError(f"`temperature` must be finite and > 0, got {temperature}")
    return sc, zq, zi, t


def relation_features(
    scores: Any,
    z_q: Any,
    z_i: Any,
    temperature: float,
    *,
    m: int = M_COMPETITORS,
) -> RelationBatch:
    """Compute the frozen LCR relation features of one score block.

    Parameters
    ----------
    scores:
        ``[n, K]`` frozen B3 scores (never modified; the caller keeps the
        grounding decision untouched).
    z_q / z_i:
        ``[n, d]`` / ``[n, K, d]`` L2-normalised embeddings.
    temperature:
        the scorer's frozen corrected global temperature (used for ``dp``).
    m:
        number of competitors (frozen to :data:`M_COMPETITORS`; exposed for
        tests only).
    """
    sc, zq, zi, temp = _validate(scores, z_q, z_i, temperature, m)
    n, k = sc.shape
    mm = int(m)
    rows = np.arange(n)

    order = np.argsort(-sc, axis=1, kind="stable")
    t = order[:, 0]
    comp = order[:, 1 : 1 + mm]

    # --- normalised score gap and probability gap ---------------------------
    s_t = sc[rows, t][:, None]
    s_j = sc[rows[:, None], comp]
    std = sc.std(axis=1, ddof=0)[:, None]
    ds_norm = (s_t - s_j) / (std + EPS)

    shifted = sc / temp
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    p = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    dp = p[rows, t][:, None] - p[rows[:, None], comp]

    # --- cosines -------------------------------------------------------------
    a = np.einsum("nd,nkd->nk", zq, zi)
    a_t = np.broadcast_to(a[rows, t][:, None], (n, mm))
    a_j = a[rows[:, None], comp]
    z_t = zi[rows, t]
    v_t = np.einsum("nd,nkd->nk", z_t, zi)
    v_tj = v_t[rows[:, None], comp]

    # --- relative rank (1-based descending; winner = 1) ----------------------
    rank_j_over_k = np.broadcast_to(
        (np.arange(2, mm + 2, dtype=np.float64) / float(k))[None, :], (n, mm)
    )

    r = np.stack(
        [ds_norm, dp, a_t, a_j, a_t - a_j, v_tj, rank_j_over_k], axis=-1
    )

    gate = np.stack(
        [v_tj.max(axis=1), v_tj.mean(axis=1), ds_norm[:, 0]], axis=-1
    )

    return RelationBatch(
        order=order.astype(np.int64, copy=False),
        winner_idx=t.astype(np.int64, copy=False),
        competitor_idx=comp.astype(np.int64, copy=False),
        r=r,
        gate=gate,
        k=int(k),
    )
