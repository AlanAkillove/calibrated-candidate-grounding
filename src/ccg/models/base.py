"""Unified scorer interface for the Phase 0 baselines (Task 6).

Every decision model - cosine, independent MLP, and later the candidate-aware
models - exposes the same call:

``score(query_feature, candidate_features, geometry=None) -> scores [K]``

``scores`` are *logit-like* real numbers (one per candidate, in the order of
``candidate_features``).  Probabilities are always obtained downstream with
:func:`softmax_np` (optionally after temperature scaling) so that calibration
code never has to know which model produced the logits.

The public contract is numpy in / numpy out; a torch-backed model keeps the
tensor inside its own ``score`` implementation and converts at the boundary,
which lets the calibration and audit code stay dependency-light.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "Scorer",
    "BaseScorer",
    "softmax_np",
    "cosine_similarity",
    "as_candidate_matrix",
    "GEOMETRY_FIELDS_DEFAULT",
]

#: Default geometry layout: normalised ``x, y, w, h`` plus relative area.
GEOMETRY_FIELDS_DEFAULT = ("x", "y", "w", "h", "area")

_EPS = 1e-8


@runtime_checkable
class Scorer(Protocol):
    """Structural interface every scorer implements (duck-typing friendly)."""

    def score(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return one logit per candidate, shape ``[K]``."""
        ...  # pragma: no cover - protocol


class BaseScorer(ABC):
    """Convenience ABC: validates shapes and adds softmax helpers.

    Subclasses implement :meth:`score` only.  Nothing here may mix information
    across candidates: batching is a pure performance detail.
    """

    #: short name used in experiment records (``ccg.utils.logging``)
    name: str = "base"

    @abstractmethod
    def score(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError

    # -- shared helpers ------------------------------------------------------
    def probabilities(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Softmax over this scorer's logits (``p_i`` of the ``K``-way choice)."""
        logits = self.score(query_feature, candidate_features, geometry)
        return softmax_np(logits, temperature=temperature)

    def score_sets(
        self,
        query_features: Sequence[np.ndarray],
        candidate_features: Sequence[np.ndarray],
        geometries: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> List[np.ndarray]:
        """Score many ``(query, candidate set)`` pairs; a thin loop over :meth:`score`."""
        if geometries is None:
            geometries = [None] * len(query_features)
        if not len(query_features) == len(candidate_features) == len(geometries):
            raise ValueError("query_features, candidate_features and geometries must be aligned")
        return [
            self.score(query, candidates, geometry)
            for query, candidates, geometry in zip(query_features, candidate_features, geometries)
        ]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def as_candidate_matrix(features: np.ndarray, query_feature: Optional[np.ndarray] = None) -> np.ndarray:
    """Coerce ``candidate_features`` to a ``[K, D]`` float32 matrix (1-D -> 1 row)."""
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"candidate_features must be [K,D] or [D], got shape {matrix.shape}")
    if query_feature is not None:
        query_dim = int(np.asarray(query_feature).reshape(-1).size)
        if matrix.shape[1] != query_dim:
            raise ValueError(
                f"feature dim mismatch: query {query_dim} vs candidates {matrix.shape[1]}"
            )
    return matrix


def cosine_similarity(
    query_feature: np.ndarray, candidate_features: np.ndarray, eps: float = _EPS
) -> np.ndarray:
    """``[K]`` cosine similarities between one query and ``[K, D]`` candidates."""
    query = np.asarray(query_feature, dtype=np.float32).reshape(-1)
    matrix = as_candidate_matrix(candidate_features, query)
    if np.linalg.norm(query) <= eps:
        raise ValueError("query_feature has near-zero norm; cosine similarity is undefined")
    sims = matrix @ query / (float(np.linalg.norm(query)) * np.maximum(np.linalg.norm(matrix, axis=1), eps))
    return np.asarray(sims, dtype=np.float32)


def softmax_np(logits: np.ndarray, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax with an optional temperature divisor.

    Shared by the models and the calibration code so that "probability of the
    top choice" is defined identically everywhere (``max_softmax`` in the score
    statistics uses exactly this function on raw logits).
    """
    values = np.asarray(logits, dtype=np.float32)
    if float(temperature) <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = values / float(temperature)
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32, copy=False)
