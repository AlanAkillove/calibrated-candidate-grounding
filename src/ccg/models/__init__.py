"""Decision models: Phase 0 baselines plus gated placeholders.

Phase 0 (implemented):
    * :class:`ccg.models.cosine.CosineScorer` - B1 frozen CLIP cosine.
    * :class:`ccg.models.independent.IndependentMLPScorer` - B3 candidate-blind MLP.
    * :func:`ccg.models.stats_calibrator.extract_score_stats` - the C3 statistic
      vector (implemented and tested even though the calibrator is untrained).

Gated placeholders (raise ``NotImplementedError`` on construction):
    * :mod:`ccg.models.deepsets` (Phase 1, needs Gate Q1 GO + Gate Q2 NOT-SUFFICIENT)
    * :mod:`ccg.models.presence` (Phase 2 target-presence / NONE heads)
"""

from __future__ import annotations

from .base import (
    GEOMETRY_FIELDS_DEFAULT,
    BaseScorer,
    Scorer,
    as_candidate_matrix,
    cosine_similarity,
    softmax_np,
)
from .cosine import CosineScorer
from .deepsets import DeepSetsReliability, DeepSetsReranker
from .independent import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_HIDDEN_DIM,
    GEOMETRY_FIELDS,
    PARAM_BUDGET,
    IndependentMLPScorer,
    ScoringExample,
    build_candidate_inputs,
    candidate_input_dim,
    geometry_features,
)
from .presence import (
    FlatNoneClassifier,
    StatsPresenceModel,
    SetAwarePresenceModel,
    FactorizedPresenceModel,
    factorized_probabilities,
)
from .stats_calibrator import (
    STATS_FIELDS,
    TOP3_FIELDS,
    StatsOnlyCalibrator,
    extract_score_stats,
    extract_score_stats_batch,
    stats_dim,
)

__all__ = [
    "Scorer",
    "BaseScorer",
    "softmax_np",
    "cosine_similarity",
    "as_candidate_matrix",
    "GEOMETRY_FIELDS_DEFAULT",
    # baselines
    "CosineScorer",
    "IndependentMLPScorer",
    "ScoringExample",
    "build_candidate_inputs",
    "candidate_input_dim",
    "geometry_features",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_HIDDEN_DIM",
    "GEOMETRY_FIELDS",
    "PARAM_BUDGET",
    # stats-only calibrator (structure) + numpy statistics
    "extract_score_stats",
    "extract_score_stats_batch",
    "stats_dim",
    "STATS_FIELDS",
    "TOP3_FIELDS",
    "StatsOnlyCalibrator",
    # gated placeholders
    "DeepSetsReliability",
    "DeepSetsReranker",
    "FlatNoneClassifier",
    "StatsPresenceModel",
    "SetAwarePresenceModel",
    "FactorizedPresenceModel",
    "factorized_probabilities",
]
