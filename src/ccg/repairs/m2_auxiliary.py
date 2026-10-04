"""Deterministic derived M2 estimate from existing shared bootstrap draws."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .bootstrap import _summarize_replicates

M2_RELATIVE_EAURC_NAME = "m8__LCR_E1b_relative_eaurc_reduction"
M2_SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
M2_ANCHOR_ATOL = 1e-9
M2_DENOMINATOR_EPSILON = 1e-12
M2_RELATIVE_EAURC_SPEC = {
    "name": M2_RELATIVE_EAURC_NAME,
    "metric": "e_aurc",
    "operation": "relative_contrast",
    "contrast_positive": "m8__E1b",
    "contrast_negative": "m8__LCR",
    "denominator_condition": "m8__E1b",
    "scale": 1.0,
    "denominator_epsilon": M2_DENOMINATOR_EPSILON,
}


def _ratio(numerator_model: float, comparator_model: float, denominator: float) -> float:
    if (not np.isfinite(numerator_model) or not np.isfinite(comparator_model)
            or not np.isfinite(denominator) or denominator <= M2_DENOMINATOR_EPSILON):
        return float("nan")
    return float((numerator_model - comparator_model) / denominator)


def derive_m2_relative_eaurc_reduction(
    *,
    e1b_draws: Mapping[str, ArrayLike],
    lcr_draws: Mapping[str, ArrayLike],
    e1b_points: Mapping[str, float],
    lcr_points: Mapping[str, float],
    old_seed_points: Mapping[str, float],
    old_mean_point: float,
    ci_level: float = 0.95,
    minimum_valid_fraction: float = 0.95,
    expected_replicates: int = 5000,
    anchor_atol: float = M2_ANCHOR_ATOL,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Form per-seed ratios at each stored draw, then average the seed ratios.

    This consumes draws already generated jointly for the M2 endpoints. It does
    not resample. A seed draw with an E-AURC denominator at or below 1e-12 is
    invalid; the aggregate draw is invalid unless all three seed effects are
    finite. The historical standalone M2 gate values anchor the observed point.
    """
    if tuple(e1b_draws) != M2_SEEDS or tuple(lcr_draws) != M2_SEEDS:
        raise ValueError(f"M2 draws must contain fixed seeds in order {M2_SEEDS}")
    if tuple(e1b_points) != M2_SEEDS or tuple(lcr_points) != M2_SEEDS:
        raise ValueError(f"M2 endpoint points must contain fixed seeds in order {M2_SEEDS}")
    if tuple(old_seed_points) != M2_SEEDS:
        raise ValueError(f"M2 historical anchors must contain fixed seeds in order {M2_SEEDS}")
    if not math.isfinite(float(ci_level)) or not 0.0 < float(ci_level) < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    if not math.isfinite(float(minimum_valid_fraction)) or not 0.0 < float(minimum_valid_fraction) <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if int(expected_replicates) != expected_replicates or expected_replicates < 1:
        raise ValueError("expected_replicates must be a positive integer")

    per_seed_draws: dict[str, np.ndarray] = {}
    point_by_seed: dict[str, float] = {}
    old_by_seed: dict[str, float] = {}
    for seed in M2_SEEDS:
        e1b = np.asarray(e1b_draws[seed], dtype=np.float64)
        lcr = np.asarray(lcr_draws[seed], dtype=np.float64)
        if e1b.ndim != 1 or lcr.ndim != 1 or e1b.shape != lcr.shape:
            raise ValueError(f"M2 draw arrays must be aligned 1D vectors for {seed}")
        if e1b.size != int(expected_replicates):
            raise ValueError(f"{seed} has {e1b.size} draws; expected {expected_replicates}")
        per_seed_draws[seed] = np.asarray(
            [_ratio(float(a), float(b), float(a)) for a, b in zip(e1b, lcr, strict=True)],
            dtype=np.float64,
        )
        current = _ratio(float(e1b_points[seed]), float(lcr_points[seed]), float(e1b_points[seed]))
        old = float(old_seed_points[seed])
        if not np.isfinite(current) or not np.isfinite(old):
            raise ValueError(f"M2 observed or historical point is undefined for {seed}")
        if not math.isclose(current, old, rel_tol=0.0, abs_tol=float(anchor_atol)):
            raise ValueError(
                f"M2 relative E-AURC anchor failed for {seed}: old={old:.12g}, "
                f"new={current:.12g}, abs_error={abs(current-old):.3g}, atol={anchor_atol}"
            )
        point_by_seed[seed] = current
        old_by_seed[seed] = old

    point = float(np.mean([point_by_seed[seed] for seed in M2_SEEDS]))
    if not math.isfinite(float(old_mean_point)) or not math.isclose(
        point, float(old_mean_point), rel_tol=0.0, abs_tol=float(anchor_atol),
    ):
        raise ValueError(
            f"M2 relative E-AURC seed-mean anchor failed: old={old_mean_point:.12g}, "
            f"new={point:.12g}, abs_error={abs(point-float(old_mean_point)):.3g}, atol={anchor_atol}"
        )

    matrix = np.stack([per_seed_draws[seed] for seed in M2_SEEDS], axis=0)
    valid_all_seeds = np.all(np.isfinite(matrix), axis=0)
    aggregate_draws = np.full(int(expected_replicates), np.nan, dtype=np.float64)
    aggregate_draws[valid_all_seeds] = np.mean(matrix[:, valid_all_seeds], axis=0)
    aggregate = _summarize_replicates(
        aggregate_draws, point, float(ci_level), float(minimum_valid_fraction),
    )
    per_seed = {
        seed: {
            key: value for key, value in _summarize_replicates(
                per_seed_draws[seed], point_by_seed[seed], float(ci_level), float(minimum_valid_fraction),
            ).items() if key != "replicates"
        }
        for seed in M2_SEEDS
    }
    seed_values = np.asarray([point_by_seed[seed] for seed in M2_SEEDS], dtype=np.float64)
    raw = {
        f"aggregate__{M2_RELATIVE_EAURC_NAME}": aggregate_draws,
        **{f"seed_{seed}__{M2_RELATIVE_EAURC_NAME}": per_seed_draws[seed] for seed in M2_SEEDS},
    }
    estimate = {
        "name": M2_RELATIVE_EAURC_NAME,
        "point": point,
        "ci_low": aggregate["ci_low"],
        "ci_high": aggregate["ci_high"],
        "bootstrap_std": aggregate["bootstrap_std"],
        "seed_standard_deviation": float(np.std(seed_values, ddof=1)),
        "seed_estimates": point_by_seed,
        "per_seed": per_seed,
        "valid_replicates": aggregate["valid_replicates"],
        "invalid_replicates": aggregate["invalid_replicates"],
        "interval_status": aggregate["interval_status"],
        "uncertainty_status": aggregate["uncertainty_status"],
        "valid_fraction": aggregate["valid_fraction"],
        "gate_eligible": aggregate["gate_eligible"],
        "ci_conditioning": aggregate["ci_conditioning"],
        "minimum_valid_replicates_for_exploratory_ci": aggregate["minimum_valid_replicates_for_exploratory_ci"],
        "old_point": float(old_mean_point),
        "old_seed_points": old_by_seed,
        "anchor_tolerance": float(anchor_atol),
        "anchor_status": "PASS",
        "raw_replicates_key": f"aggregate__{M2_RELATIVE_EAURC_NAME}",
        "raw_replicates_seed_keys": {
            seed: f"seed_{seed}__{M2_RELATIVE_EAURC_NAME}" for seed in M2_SEEDS
        },
    }
    return estimate, raw


__all__ = ["M2_ANCHOR_ATOL", "M2_DENOMINATOR_EPSILON", "M2_RELATIVE_EAURC_NAME",
           "M2_RELATIVE_EAURC_SPEC", "M2_SEEDS", "derive_m2_relative_eaurc_reduction"]
