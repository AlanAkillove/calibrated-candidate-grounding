"""Phase 1 DeepSets reliability model - PLACEHOLDER ONLY (Gate Q1/Q2 not passed).

Protocol section 17/18: this module exists so the *shape* of the candidate-aware
design is agreed before any result is seen.  Implementation and training are
forbidden until

1. Gate Q1 shows a stable candidate-set reliability failure, and
2. Gate Q2 shows that score statistics (C1-C3, :mod:`ccg.models.stats_calibrator`)
   are *not* sufficient.

If either gate says NO-GO, this file stays a stub and the project ships the
smaller "candidate-set calibration audit" result - which is an acceptable
outcome, not a failure.

Intended layer shapes (comment only, deliberately not implemented)
------------------------------------------------------------------
Per candidate, with ``z_q`` the query embedding, ``z_i`` the crop embedding,
``g_i`` the geometry block and ``s_i`` the independent ranker's logit::

    v_i = [z_q, z_i, z_q * z_i, g_i, s_i]              # dim 3*512 + geo_dim + 1
    h_i = phi(v_i)         Linear(3*512+geo_dim+1 -> 128) -> ReLU -> Linear(128 -> 128)
    u_mean = mean_i psi(h_i)                            # permutation-invariant pool
    u_max  = max_i  psi(h_i)                            # psi: Linear(128 -> 128)
    u = [u_mean, u_max]                                 # dim 256

    # reliability head (preferred over a reranker, section 18)
    p_correct = g([u, score_stats])    Linear(256+stats_dim -> 128) -> ReLU -> Linear(128 -> 1) -> sigmoid

    # optional reranker (only if candidate-aware info provably helps ranking)
    s_i' = rho([h_i, u])               Linear(128+256 -> 128) -> ReLU -> Linear(128 -> 1)

Parameter target: ``< 0.5M`` for the first version (section 17).  Both pooling
operations are the *only* place where the set is allowed to interact; everything
upstream of ``phi`` is per-candidate, exactly as in B3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["GATE_MESSAGE", "DeepSetsReliability", "DeepSetsReranker"]

GATE_MESSAGE = (
    "DeepSets is a Phase 1 model: implementing it requires Gate Q1 == GO and Gate Q2 == "
    "NOT-SUFFICIENT. Until then this placeholder is intentionally unimplemented (protocol "
    "sections 14-17); do not add a forward() or a training loop here."
)


class DeepSetsReliability:
    """Placeholder for the set-aware reliability head (``p_correct``)."""

    name = "p1_deepsets_reliability"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - gated stub
        raise NotImplementedError(GATE_MESSAGE)

    def score(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
    ) -> np.ndarray:  # pragma: no cover - gated stub
        raise NotImplementedError(GATE_MESSAGE)


class DeepSetsReranker:
    """Placeholder for the optional set-aware reranker (``s_i'``), lower priority."""

    name = "p1_deepsets_reranker"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - gated stub
        raise NotImplementedError(GATE_MESSAGE)
