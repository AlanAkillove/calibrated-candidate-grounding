"""Calibration layer: temperature scaling (B2/C1/C2) and abstention thresholds (N0/N1).

Fitted exclusively on ``val_calib`` - never on ``val_select`` (model selection)
and never on test data (protocol sections 9 and 15).
"""

from __future__ import annotations

from .temperature import (
    DEFAULT_TEMP_BOUNDS,
    KAwareTemperature,
    TemperatureScaler,
    apply_temperature,
    apply_temperature_sets,
    fit_temperature,
    fit_temperature_k_aware,
    mean_nll,
    nll_at_temperature,
)
from .thresholds import (
    ThresholdSelection,
    accept_mask,
    coverage_risk_curve,
    fit_margin_threshold,
    fit_max_conf_threshold,
    max_confidences,
    top1_top2_margins,
)

__all__ = [
    "DEFAULT_TEMP_BOUNDS",
    "fit_temperature",
    "apply_temperature",
    "apply_temperature_sets",
    "nll_at_temperature",
    "mean_nll",
    "TemperatureScaler",
    "KAwareTemperature",
    "fit_temperature_k_aware",
    "fit_max_conf_threshold",
    "fit_margin_threshold",
    "coverage_risk_curve",
    "max_confidences",
    "top1_top2_margins",
    "accept_mask",
    "ThresholdSelection",
]
