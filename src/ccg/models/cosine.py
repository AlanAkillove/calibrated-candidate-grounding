"""B1 - frozen CLIP cosine scorer (protocol section 8).

``s_i = cos(z_q, z_i)`` and, when a temperature is attached, ``logits_i = s_i / T``.
Pure numpy: the embeddings come from the offline cache, nothing is learned here
(temperature scaling on top of this scorer is B2 / C1, see
:mod:`ccg.calibration.temperature`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import BaseScorer, cosine_similarity

__all__ = ["CosineScorer"]


@dataclass
class CosineScorer(BaseScorer):
    """Cosine-similarity scorer with an optional temperature divisor.

    Parameters
    ----------
    temperature:
        ``T > 0``.  ``score()`` returns ``cos / T`` so the same object can be
        used before (``T=1``) and after calibration (fitted ``T``).  A scorer
        fitted by :func:`ccg.calibration.temperature.fit_temperature` is
        reproduced exactly by setting this value.

    Cosine similarity is computed on L2-normalised vectors by
    :func:`ccg.models.base.cosine_similarity`, so cached embeddings do not have
    to be normalised in advance.
    """

    temperature: float = 1.0
    name: str = "b1_clip_cosine"

    def __post_init__(self) -> None:
        self.temperature = float(self.temperature)
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def score(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """``[K]`` cosine logits.

        ``geometry`` is accepted for interface parity with
        :class:`~ccg.models.independent.IndependentMLPScorer` and deliberately
        unused: B1 must stay a pure query-crop similarity so that any observed
        effect can be attributed to the frozen backbone alone.
        """
        sims = cosine_similarity(query_feature, candidate_features)
        return (sims / self.temperature).astype(np.float32, copy=False)

    def similarities(
        self, query_feature: np.ndarray, candidate_features: np.ndarray
    ) -> np.ndarray:
        """Raw ``[K]`` cosines (no temperature), handy for reliability diagrams."""
        return cosine_similarity(query_feature, candidate_features)
