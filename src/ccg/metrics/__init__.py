"""Evaluation metrics for calibrated candidate grounding.

Sub-modules (all numpy-first, ``scikit-learn`` optional):

* :mod:`ccg.metrics.ranking` -- Top-1 / Top-k accuracy, MRR, accuracy-by-K.
* :mod:`ccg.metrics.calibration` -- Top-label ECE (equal-width & adaptive),
  binary-correctness Brier / NLL, multiclass NLL / Brier, confidence-accuracy
  gap, reliability tables.
* :mod:`ccg.metrics.selective` -- risk-coverage curve, AURC, selective accuracy,
  risk-at-coverage.
* :mod:`ccg.metrics.absence` -- AUROC, AUPRC, FPR@TPR, NONE precision/recall/F1,
  false selection rate.
* :mod:`ccg.metrics.bootstrap` -- paired (optionally image-clustered) bootstrap
  CIs and a generic bootstrap CI.

Everything is re-exported here so that analysis scripts can simply
``from ccg import metrics``.
"""

from __future__ import annotations

from ccg.metrics.absence import (
    auprc,
    auroc,
    false_selection_rate,
    fpr_at_tpr,
    none_f1,
    none_precision,
    none_recall,
    roc_points,
)
from ccg.metrics.bootstrap import bootstrap_ci, paired_bootstrap
from ccg.metrics.calibration import (
    brier_binary,
    confidence_accuracy_gap,
    confidence_from_scores,
    multiclass_brier,
    multiclass_nll,
    nll_binary,
    reliability_table,
    softmax_scores,
    top_label_ece,
    top_label_ece_adaptive,
)
from ccg.metrics.ranking import (
    accuracy_by_k,
    mrr,
    top1_accuracy,
    topk_accuracy,
    target_rank,
)
from ccg.metrics.selective import (
    aurc,
    risk_at_coverage,
    risk_coverage_curve,
    selective_accuracy,
    selective_risk,
)

__all__ = [
    "accuracy_by_k",
    "auroc",
    "auprc",
    "aurc",
    "bootstrap_ci",
    "brier_binary",
    "confidence_accuracy_gap",
    "confidence_from_scores",
    "false_selection_rate",
    "fpr_at_tpr",
    "mrr",
    "multiclass_brier",
    "multiclass_nll",
    "nll_binary",
    "none_f1",
    "none_precision",
    "none_recall",
    "paired_bootstrap",
    "reliability_table",
    "risk_at_coverage",
    "risk_coverage_curve",
    "roc_points",
    "selective_accuracy",
    "selective_risk",
    "softmax_scores",
    "top1_accuracy",
    "top_label_ece",
    "top_label_ece_adaptive",
    "topk_accuracy",
    "target_rank",
]
