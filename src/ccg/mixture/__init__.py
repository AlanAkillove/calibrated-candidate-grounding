"""Competition-Adaptive Reliability Mixture (V2-M3 amendment).

This package implements the frozen V2-M3 formulas
(``results/v2_local_competition/m3_mixture/protocol_m3.json``):

* :mod:`ccg.mixture.mixer` -- the per-backbone train-only competition index
  ``A`` of the three frozen features and the ``EqualMix`` / ``StaticMix`` /
  ``AdaptiveMix`` prediction heads (2 trainable scalars at most);
* :mod:`ccg.mixture.bootstrap` -- the macro paired image-cluster bootstrap
  with shared draws across severity.

Nothing in this package trains, modifies or even holds an expert or a
grounding scorer: ``p_R`` / ``p_C`` / ``A`` are plain arrays produced by
frozen artifacts, and the mixture only changes the reliability estimate.
"""

from __future__ import annotations

__all__ = [
    "COMPETITION_FEATURES",
    "COMPETITION_DIRECTIONS",
    "MIXER_LEVELS",
    "softplus",
    "CompetitionIndex",
    "competition_features_from_sem14",
    "EqualMix",
    "StaticMix",
    "AdaptiveMix",
    "macro_paired_cluster_bootstrap",
]

from .bootstrap import macro_paired_cluster_bootstrap
from .mixer import (
    COMPETITION_DIRECTIONS,
    COMPETITION_FEATURES,
    MIXER_LEVELS,
    AdaptiveMix,
    CompetitionIndex,
    EqualMix,
    StaticMix,
    competition_features_from_sem14,
    softplus,
)
