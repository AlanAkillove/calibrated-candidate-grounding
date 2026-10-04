"""Statistics helpers and inference adapters for Research Repair v1.

The repair uses the metrics already defined by :mod:`ccg.metrics`, with a fast
single-sort bundle for the many repeated image-cluster bootstrap draws. Derived
effects are evaluated per seed inside each draw and only then averaged by the
shared bootstrap implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ccg.metrics.calibration import (
    brier_binary,
    confidence_accuracy_gap,
    nll_binary,
    top_label_ece,
    top_label_ece_adaptive,
)
from ccg.metrics.discrimination import auprc_correct, auroc_correct
from ccg.metrics.selective import aurc, e_aurc, oracle_aurc
from ccg.repairs.bootstrap import shared_image_cluster_bootstrap

__all__ = [
    "METRIC_NAMES",
    "joint_prediction_bootstrap",
    "metric_bundle",
    "relative_effect",
]

METRIC_NAMES = (
    "accuracy",
    "auroc_correct",
    "auprc_correct",
    "aurc",
    "e_aurc",
    "rer_at_50",
    "rer_at_80",
    "rer_at_90",
    "rer_at_95",
    "risk_at_50",
    "risk_at_80",
    "risk_at_90",
    "ece_adaptive",
    "ece_equal_width",
    "brier_binary",
    "nll_binary",
    "confidence_accuracy_gap",
)


def _as_pairs(confidence: Any, correctness: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    corr = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if conf.size != corr.size:
        raise ValueError(f"confidence/correctness length mismatch: {conf.size} != {corr.size}")
    if conf.size == 0:
        raise ValueError("metrics require at least one row")
    if not np.all(np.isfinite(conf)) or np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence must be finite and in [0, 1]")
    if not np.all(np.isfinite(corr)) or np.any((corr != 0) & (corr != 1)):
        raise ValueError("correctness must be finite and binary")
    return conf, corr


def _auroc_from_sorted(conf_desc: NDArray[np.float64], corr_desc: NDArray[np.float64]) -> float:
    """Tie-correct Mann–Whitney AUROC using the confidence sort already held."""
    n_pos = float(corr_desc.sum())
    n_neg = float(corr_desc.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    if conf_desc.size == 1:
        return float("nan")
    boundaries = np.flatnonzero(conf_desc[:-1] != conf_desc[1:]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [conf_desc.size]))
    positives = np.add.reduceat(corr_desc, starts)
    group_sizes = ends - starts
    negatives = group_sizes - positives
    negatives_after = n_neg - np.cumsum(negatives)
    wins = float(np.sum(positives * (negatives_after + 0.5 * negatives)))
    return wins / (n_pos * n_neg)


def _auprc_from_sorted(corr_desc: NDArray[np.float64], conf_desc: NDArray[np.float64]) -> float:
    n_pos = float(corr_desc.sum())
    if n_pos == 0.0 or n_pos == float(corr_desc.size):
        return float("nan")
    boundaries = np.flatnonzero(conf_desc[:-1] != conf_desc[1:]) + 1
    ends = np.concatenate((boundaries, [conf_desc.size]))
    tp = np.cumsum(corr_desc)[ends - 1]
    fp = ends - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def metric_bundle(
    confidence: Any,
    correctness: Any,
    *,
    metrics: Sequence[str] = METRIC_NAMES,
    zero_error_rer: Literal["nan", "legacy_zero"] = "nan",
) -> dict[str, float]:
    """Compute selected correctness, discrimination, risk, and calibration metrics.

    A single stable descending confidence sort is shared by AUROC, AUPRC,
    AURC, fixed-coverage risk, and RER. Calibration helpers retain their
    repository-defined binning and clipping conventions.
    """
    conf, corr = _as_pairs(confidence, correctness)
    requested = tuple(metrics)
    unknown = sorted(set(requested) - set(METRIC_NAMES))
    if unknown:
        raise ValueError(f"unknown metrics requested: {unknown}")

    out: dict[str, float] = {}
    accuracy = float(corr.mean())
    if "accuracy" in requested:
        out["accuracy"] = accuracy
    need_order = any(name in requested for name in (
        "auroc_correct", "auprc_correct", "aurc", "e_aurc", "rer_at_50", "rer_at_80",
        "rer_at_90", "rer_at_95", "risk_at_50", "risk_at_80", "risk_at_90",
    ))
    if need_order:
        order = np.argsort(-conf, kind="stable")
        conf_desc = conf[order]
        corr_desc = corr[order]
        n = corr.size
        if "auroc_correct" in requested:
            out["auroc_correct"] = _auroc_from_sorted(conf_desc, corr_desc)
        if "auprc_correct" in requested:
            out["auprc_correct"] = _auprc_from_sorted(corr_desc, conf_desc)
        if "aurc" in requested or "e_aurc" in requested:
            accepted = np.arange(1, n + 1, dtype=np.float64)
            coverage = accepted / float(n)
            risk = 1.0 - np.cumsum(corr_desc) / accepted
            raw_aurc = float(risk[0]) if n == 1 else float(np.trapz(risk, coverage))
            if "aurc" in requested:
                out["aurc"] = raw_aurc
            if "e_aurc" in requested:
                out["e_aurc"] = e_aurc(raw_aurc, accuracy)
        for coverage, suffix in ((0.50, "50"), (0.80, "80"), (0.90, "90"), (0.95, "95")):
            rer_name = f"rer_at_{suffix}"
            risk_name = f"risk_at_{suffix}"
            if rer_name in requested or risk_name in requested:
                n_keep = max(1, int(np.ceil(coverage * n)))
                risk_value = float(1.0 - corr_desc[:n_keep].mean())
                if risk_name in requested:
                    out[risk_name] = risk_value
                if rer_name in requested:
                    base_risk = 1.0 - accuracy
                # A zero base-error denominator is undefined for inference. The
                # legacy scalar helper keeps its documented 0 convention, but
                # the repair bootstrap must count these draws as invalid.
                if base_risk == 0.0 and zero_error_rer == "legacy_zero":
                    out[rer_name] = 0.0
                else:
                    out[rer_name] = float("nan") if base_risk == 0.0 else (base_risk - risk_value) / base_risk
    if "ece_adaptive" in requested:
        out["ece_adaptive"] = top_label_ece_adaptive(conf, corr)
    if "ece_equal_width" in requested:
        out["ece_equal_width"] = top_label_ece(conf, corr)
    if "brier_binary" in requested:
        out["brier_binary"] = brier_binary(conf, corr)
    if "nll_binary" in requested:
        out["nll_binary"] = nll_binary(conf, corr)
    if "confidence_accuracy_gap" in requested:
        out["confidence_accuracy_gap"] = confidence_accuracy_gap(conf, corr)
    return {name: out[name] for name in requested}


def relative_effect(numerator: float, denominator: float) -> float:
    """Return a signed ratio using an explicitly chosen signed denominator."""
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def _estimate_value(
    metric_values: Mapping[str, float],
    spec: Mapping[str, Any],
) -> float:
    metric = str(spec["metric"])
    operation = str(spec.get("operation", "condition"))
    if operation == "condition":
        return float(metric_values[str(spec["condition"]) + "::" + metric])
    if operation == "difference":
        a = float(metric_values[str(spec["a"]) + "::" + metric])
        b = float(metric_values[str(spec["b"]) + "::" + metric])
        return float(spec.get("scale", 1.0)) * (a - b)
    if operation == "relative_contrast":
        positive = str(spec["contrast_positive"])
        negative = str(spec["contrast_negative"])
        denominator_condition = str(spec["denominator_condition"])
        numerator = (
            float(metric_values[positive + "::" + metric])
            - float(metric_values[negative + "::" + metric])
        )
        denominator = float(metric_values[denominator_condition + "::" + metric])
        if "denominator_epsilon" in spec:
            epsilon = float(spec["denominator_epsilon"])
            if not np.isfinite(denominator) or denominator <= epsilon:
                return float("nan")
        effect = relative_effect(numerator, denominator)
        return float(spec.get("scale", 1.0)) * effect
    if operation == "relative_contrast_of_macros":
        positive_conditions = [str(value) for value in spec["contrast_positive_conditions"]]
        negative_conditions = [str(value) for value in spec["contrast_negative_conditions"]]
        if not positive_conditions or len(positive_conditions) != len(negative_conditions):
            raise ValueError("relative_contrast_of_macros requires paired non-empty condition lists")
        positive = [float(metric_values[name + "::" + metric]) for name in positive_conditions]
        negative = [float(metric_values[name + "::" + metric]) for name in negative_conditions]
        if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
            return float("nan")
        denominator = float(np.mean(positive))
        numerator = denominator - float(np.mean(negative))
        effect = relative_effect(numerator, denominator)
        return float(spec.get("scale", 1.0)) * effect
    if operation == "difference_of_differences":
        a, b, c, d = (float(metric_values[str(spec[key]) + "::" + metric]) for key in ("a", "b", "c", "d"))
        return float(spec.get("scale", 1.0)) * ((a - b) - (c - d))
    if operation == "difference_of_relative_contrasts":
        metric_key = "::" + metric
        positive_a = float(metric_values[str(spec["positive_a"]) + metric_key])
        negative_a = float(metric_values[str(spec["negative_a"]) + metric_key])
        denominator_a = float(metric_values[str(spec["denominator_a"]) + metric_key])
        positive_b = float(metric_values[str(spec["positive_b"]) + metric_key])
        negative_b = float(metric_values[str(spec["negative_b"]) + metric_key])
        denominator_b = float(metric_values[str(spec["denominator_b"]) + metric_key])
        effect_a = relative_effect(positive_a - negative_a, denominator_a)
        effect_b = relative_effect(positive_b - negative_b, denominator_b)
        return float(spec.get("scale", 1.0)) * (effect_a - effect_b)
    if operation == "macro_mean":
        values = [float(metric_values[str(condition) + "::" + metric]) for condition in spec["conditions"]]
        return float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan")
    if operation == "macro_difference":
        left = [float(metric_values[str(condition) + "::" + metric]) for condition in spec["conditions_a"]]
        right = [float(metric_values[str(condition) + "::" + metric]) for condition in spec["conditions_b"]]
        if len(left) != len(right) or not left:
            raise ValueError("macro_difference requires paired non-empty condition lists")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return float("nan")
        return float(spec.get("scale", 1.0)) * float(np.mean(np.asarray(left) - np.asarray(right)))
    raise ValueError(f"unsupported estimate operation {operation!r}")


def joint_prediction_bootstrap(
    image_ids: Sequence[Any] | NDArray[Any],
    seed_cells: Mapping[Any, Mapping[str, tuple[Any, Any]]],
    *,
    estimates: Sequence[Mapping[str, Any]] | None = None,
    metrics: Sequence[str] = METRIC_NAMES,
    auxiliary_cells: Mapping[Any, Mapping[str, Any]] | None = None,
    auxiliary_estimates: Sequence[Mapping[str, Any]] = (),
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    point_order: Sequence[int] | NDArray[np.int64] | None = None,
    point_order_by_seed_condition: Mapping[Any, Mapping[str, Sequence[int] | NDArray[np.int64]]] | None = None,
) -> dict[str, Any]:
    """Bootstrap aligned condition arrays with common image draws across seeds.

    ``seed_cells[seed][condition]`` is ``(confidence, correctness)``; every
    condition and seed must have the canonical row count of ``image_ids``.
    ``estimates`` may define per-condition metrics and replicate-internal
    differences, relative effects, difference-of-differences, or severity
    macros. ``auxiliary_cells`` adds finite continuous row fields such as
    preregistered manipulation diagnostics; ``auxiliary_estimates`` summarizes
    their means and contrasts on the very same sampled image rows. When omitted,
    every requested metric for every prediction condition is emitted.
    ``point_order`` optionally supplies one permutation for the observed point
    estimate. ``point_order_by_seed_condition`` supplies the more specific
    historical source-row permutation when seed/condition archives had distinct
    row orders. Both leave bootstrap sampling canonical.
    """
    ids = np.asarray(image_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("image_ids must be a non-empty one-dimensional canonical row array")
    if not seed_cells:
        raise ValueError("seed_cells must include at least one fixed seed/model")
    normalized: dict[Any, dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]] = {}
    for seed_key, conditions in seed_cells.items():
        if not conditions:
            raise ValueError(f"seed {seed_key!r} has no conditions")
        normalized[seed_key] = {}
        for condition, (conf, corr) in conditions.items():
            c, y = _as_pairs(conf, corr)
            if c.size != ids.size:
                raise ValueError(f"condition {condition!r} seed {seed_key!r} has {c.size} rows; expected {ids.size}")
            normalized[seed_key][str(condition)] = (c, y)
    condition_order = tuple(next(iter(normalized.values())))
    for seed_key, conditions in normalized.items():
        if tuple(conditions) != condition_order:
            raise ValueError(f"seed {seed_key!r} conditions are not identically ordered")

    normalized_aux: dict[Any, dict[str, NDArray[np.float64]]] = {}
    if auxiliary_cells:
        if set(auxiliary_cells) != set(normalized):
            raise ValueError("auxiliary_cells must have exactly the prediction seed keys")
        aux_order: tuple[str, ...] | None = None
        for seed_key in normalized:
            normalized_aux[seed_key] = {}
            raw_conditions = auxiliary_cells[seed_key]
            if aux_order is None:
                aux_order = tuple(str(key) for key in raw_conditions)
            elif tuple(str(key) for key in raw_conditions) != aux_order:
                raise ValueError(f"auxiliary conditions are not identically ordered for seed {seed_key!r}")
            for raw_condition, raw_values in raw_conditions.items():
                condition = str(raw_condition)
                values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
                if values.size != ids.size:
                    raise ValueError(f"auxiliary condition {condition!r} seed {seed_key!r} has {values.size} rows; expected {ids.size}")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"auxiliary condition {condition!r} seed {seed_key!r} must be finite")
                normalized_aux[seed_key][condition] = values
        if aux_order is None:
            aux_order = ()
    else:
        aux_order = ()

    specs = list(estimates) if estimates is not None else [
        {"name": f"{condition}::{metric}", "operation": "condition", "condition": condition, "metric": metric}
        for condition in condition_order for metric in metrics
    ]
    aux_specs = list(auxiliary_estimates)
    for spec in aux_specs:
        if not normalized_aux:
            raise ValueError("auxiliary_estimates require auxiliary_cells")
        if str(spec.get("metric", "mean")) != "mean":
            raise ValueError("auxiliary estimate metric must be 'mean'")
        referenced = [str(spec[key]) for key in ("condition", "a", "b", "c", "d",
                      "contrast_positive", "contrast_negative", "denominator_condition",
                      "positive_a", "negative_a", "denominator_a", "positive_b", "negative_b",
                      "denominator_b") if key in spec]
        referenced.extend(str(value) for key in ("conditions", "conditions_a", "conditions_b")
                          for value in spec.get(key, ()))
        referenced.extend(str(value) for key in ("contrast_positive_conditions", "contrast_negative_conditions")
                          for value in spec.get(key, ()))
        missing = sorted(set(referenced) - set(aux_order))
        if missing:
            raise ValueError(f"auxiliary estimate {spec.get('name')!r} references unknown fields: {missing}")
    specs.extend(aux_specs)
    if not specs:
        raise ValueError("at least one estimate must be requested")
    names = [str(spec["name"]) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("estimate names must be unique")
    canonical_point_order = (
        np.arange(ids.size, dtype=np.int64)
        if point_order is None else np.asarray(point_order, dtype=np.int64)
    )
    if canonical_point_order.ndim != 1 or canonical_point_order.size != ids.size:
        raise ValueError("point_order must be a permutation with one entry per canonical row")
    if not np.array_equal(np.sort(canonical_point_order), np.arange(ids.size, dtype=np.int64)):
        raise ValueError("point_order must be a permutation containing every canonical row index exactly once")
    normalized_point_orders: dict[Any, dict[str, NDArray[np.int64]]] = {}
    if point_order_by_seed_condition is not None:
        if set(point_order_by_seed_condition) != set(normalized):
            raise ValueError("point_order_by_seed_condition must have exactly the prediction seed keys")
        for seed_key, order_map in point_order_by_seed_condition.items():
            if set(order_map) != set(condition_order):
                raise ValueError(f"point-order conditions differ for seed {seed_key!r}")
            normalized_point_orders[seed_key] = {}
            for condition, raw_order in order_map.items():
                order = np.asarray(raw_order, dtype=np.int64)
                if order.ndim != 1 or order.size != ids.size or not np.array_equal(np.sort(order), np.arange(ids.size, dtype=np.int64)):
                    raise ValueError(f"point order for {seed_key!r}/{condition!r} must be a full row permutation")
                normalized_point_orders[seed_key][str(condition)] = order

    def calculate(rows: NDArray[np.int64], *, point: bool) -> dict[str, dict[Any, float]]:
        results: dict[str, dict[Any, float]] = {name: {} for name in names}
        for seed_key, conditions in normalized.items():
            values: dict[str, float] = {}
            for condition in condition_order:
                conf, corr = conditions[condition]
                condition_rows = rows
                if point and point_order_by_seed_condition is not None:
                    condition_rows = normalized_point_orders[seed_key][condition]
                bundle = metric_bundle(
                    conf[condition_rows], corr[condition_rows], metrics=metrics,
                    zero_error_rer="legacy_zero" if point else "nan",
                )
                values.update({condition + "::" + metric: value for metric, value in bundle.items()})
            for condition in aux_order:
                values[condition + "::mean"] = float(np.mean(normalized_aux[seed_key][condition][rows]))
            for spec, name in zip(specs, names, strict=True):
                results[name][seed_key] = _estimate_value(values, spec)
        return results

    def statistic(rows: NDArray[np.int64]) -> dict[str, dict[Any, float]]:
        return calculate(rows, point=False)

    def point_statistic(rows: NDArray[np.int64]) -> dict[str, dict[Any, float]]:
        return calculate(canonical_point_order, point=True)

    return shared_image_cluster_bootstrap(
        ids,
        statistic,
        point_statistic=point_statistic,
        n_replicates=n_replicates,
        seed=seed,
        ci=ci,
    )

