"""Reliability metrics, image-clustered paired bootstrap rows and the sufficiency gate.

Phase 0.5 (Amendment A6) compares the frozen grounding scorers' *score sets*
across candidate-set sizes.  This module reuses the already-landed metric
implementations instead of re-deriving any formula:

* :func:`ccg.metrics.discrimination.auroc_correct`
* :func:`ccg.metrics.selective.risk_coverage_curve` / :func:`~ccg.metrics.selective.aurc`
  / :func:`~ccg.metrics.selective.oracle_aurc` / :func:`~ccg.metrics.selective.rer_at_coverage`
* :func:`ccg.metrics.calibration.top_label_ece_adaptive` / :func:`~ccg.metrics.calibration.brier_binary`
  / :func:`~ccg.metrics.calibration.nll_binary`
* :func:`ccg.experiment.phase0a.paired_cluster_bootstrap` and
  :func:`ccg.experiment.phase0b.paired_cluster_bootstrap_ratio` (with the
  ``SampleStats`` block and the phase0b ``_e_aurc_metric`` / ``_auroc_metric`` /
  ``_make_rer_metric`` registry).

Sign conventions
----------------
* ``e_aurc_abs`` / ``aurc``: lower is better.  ``E-AURC = AURC - oracle_aurc(1 - Acc)``.
* ``rer_at_c``: risk-erosion ratio ``(R1 - Rc) / R1``; higher is better.
* Cross-``K`` rows use the **phase0b ratio convention** ``t = (metric_lo - metric_hi) / metric_hi``
  where ``lo`` = K5 and ``hi`` = K_b, i.e. ``diff_kind="relative"`` rows store
  ``t`` (positive = K_b worse).  Because a raw ratio ``t`` is awkward to read at
  the boundary, every relative row additionally carries the **unified worsening**
  ``w = -t / (1 + t)`` (positive = worsening).  ``w`` is monotone in ``t``, so the
  CI endpoints are mapped element-wise and then min/max sorted; never read ``t``
  without its ``w`` companion (the sign trap of the previous phase).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ccg.experiment.phase0a import SampleStats, paired_cluster_bootstrap
from ccg.experiment.phase0b import (
    _auroc_metric,
    _e_aurc_metric,
    _make_rer_metric,
    paired_cluster_bootstrap_ratio,
)
from ccg.metrics.calibration import brier_binary, nll_binary, top_label_ece_adaptive
from ccg.metrics.discrimination import auroc_correct
from ccg.metrics.selective import aurc, oracle_aurc, rer_at_coverage, risk_coverage_curve

__all__ = [
    "DEFAULT_THRESHOLDS",
    "PRIMARY_METRICS",
    "PROBABILITY_METRICS",
    "SECONDARY_METRICS",
    "cross_k_bootstrap_row",
    "model_vs_model_bootstrap_row",
    "point_metric_row",
    "sufficiency_verdict",
    "worsening_ci_from_ratio_ci",
    "worsening_from_ratio",
]

#: Primary cross-K metrics (Amendment A6.5).
PRIMARY_METRICS: tuple[str, ...] = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
#: Secondary metrics (RER@90/95 plus the probability-only calibration metrics).
SECONDARY_METRICS: tuple[str, ...] = (
    "rer_at_90",
    "rer_at_95",
    "ece_adaptive",
    "brier_binary",
    "nll_binary",
)
#: Metrics that require the model to expose ``P(correct)``.
PROBABILITY_METRICS: tuple[str, ...] = ("ece_adaptive", "brier_binary", "nll_binary")

#: Default sufficiency-gate thresholds (Amendment A6.6).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "auroc_drop": 0.03,
    "auroc_stable": 0.02,
    "e_aurc_worsen": 0.20,
    "e_aurc_abs": 0.03,
    "rer_drop_pp": 10.0,
    "override_auroc": 0.85,
    "override_e_aurc": 0.03,
    "override_rer": 0.70,
}


# ---------------------------------------------------------------------------
# metric registry (SampleStats -> float), reusing the frozen implementations
# ---------------------------------------------------------------------------
def _metric_fn(name: str) -> Callable[[SampleStats], float]:
    """Resolve a metric name to the shared ``SampleStats -> float`` callable."""
    if name == "auroc_correct":
        return _auroc_metric
    if name == "e_aurc":
        return _e_aurc_metric
    if name.startswith("rer_at_"):
        level = int(name[len("rer_at_") :]) / 100.0
        return _make_rer_metric(level)
    raise ValueError(f"unknown bootstrap metric {name!r}")


def _as_pair(confidence: Any, correct: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.shape[0] != corr.shape[0]:
        raise ValueError(
            f"confidence ({conf.shape[0]}) and correct ({corr.shape[0]}) differ in length"
        )
    return conf, corr


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------
def point_metric_row(
    confidence: Any,
    correct: Any,
    *,
    probability: Any = None,
) -> dict[str, Any]:
    """Point reliability metrics for one scorer ``(cell, K)``.

    Parameters
    ----------
    confidence:
        ``[n]`` per-sample confidence (higher = more trustworthy), in ``[0, 1]``
        (the selective helpers validate this range).
    correct:
        ``[n]`` binary correctness ``1[argmax == target]``.
    probability:
        Optional ``[n]`` model output ``P(correct)``.  When given, the three
        probability-only metrics are filled; otherwise they are ``None``.

    Returns
    -------
    dict
        Always ``auroc_correct``, ``aurc``, ``aurc_oracle``, ``e_aurc`` and
        ``rer_at_50/80/90/95``; plus ``ece_adaptive`` / ``brier_binary`` /
        ``nll_binary`` (``None`` without ``probability``).
    """
    conf, corr = _as_pair(confidence, correct)
    accuracy = float(np.mean(corr))
    coverage, risk = risk_coverage_curve(conf, corr)
    aurc_value = float(aurc(coverage, risk))
    aurc_oracle_value = float(oracle_aurc(1.0 - accuracy))
    row: dict[str, Any] = {
        "auroc_correct": float(auroc_correct(conf, corr)),
        "aurc": aurc_value,
        "aurc_oracle": aurc_oracle_value,
        "e_aurc": float(aurc_value - aurc_oracle_value),
        "rer_at_50": float(rer_at_coverage(conf, corr, 0.50)),
        "rer_at_80": float(rer_at_coverage(conf, corr, 0.80)),
        "rer_at_90": float(rer_at_coverage(conf, corr, 0.90)),
        "rer_at_95": float(rer_at_coverage(conf, corr, 0.95)),
        "ece_adaptive": None,
        "brier_binary": None,
        "nll_binary": None,
    }
    if probability is not None:
        prob = np.asarray(probability, dtype=np.float64).reshape(-1)
        if prob.shape[0] != corr.shape[0]:
            raise ValueError(
                f"probability ({prob.shape[0]}) and correct ({corr.shape[0]}) differ in length"
            )
        row["ece_adaptive"] = float(top_label_ece_adaptive(prob, corr))
        row["brier_binary"] = float(brier_binary(prob, corr))
        row["nll_binary"] = float(nll_binary(prob, corr))
    return row


# ---------------------------------------------------------------------------
# relative-gap -> unified worsening mapping
# ---------------------------------------------------------------------------
def worsening_from_ratio(t: float) -> float:
    """Map the phase0b relative gap ``t = (lo - hi) / hi`` to worsening ``w``.

    ``w = -t / (1 + t)``: ``w > 0`` means K_b (``hi``) is *worse* than K5 (``lo``),
    ``w = 0`` means identical, ``w < 0`` means K_b is better.  Monotone in ``t``.
    """
    value = float(t)
    if not np.isfinite(value):
        return float("nan")
    denom = 1.0 + value
    if denom == 0.0:
        return float("nan")
    return float(-value / denom)


def worsening_ci_from_ratio_ci(ratio_lo: float, ratio_hi: float) -> tuple[float, float]:
    """Map a ``t`` CI ``[ratio_lo, ratio_hi]`` to a sorted worsening CI ``(low, high)``.

    The two endpoints are mapped through :func:`worsening_from_ratio` and then
    min/max sorted, so the returned interval is always ordered regardless of the
    direction of the monotone map.
    """
    a = worsening_from_ratio(ratio_lo)
    b = worsening_from_ratio(ratio_hi)
    return (float(min(a, b)), float(max(a, b)))


# ---------------------------------------------------------------------------
# cross-K bootstrap rows
# ---------------------------------------------------------------------------
def _cross_k_common_fields(
    row: Mapping[str, Any],
    *,
    eval_split: str,
    k_lo: Optional[int],
    k_hi: Optional[int],
    metric: str,
    diff_kind: str,
) -> dict[str, Any]:
    return {
        "eval_split": eval_split,
        "K_lo": None if k_lo is None else int(k_lo),
        "K_hi": None if k_hi is None else int(k_hi),
        "metric": metric,
        "diff_kind": diff_kind,
        "diff": float(row["diff"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "mean_lo": float(row["mean_a"]),
        "mean_hi": float(row["mean_b"]),
        "n": int(row["n"]),
        "n_clusters": int(row["n_clusters"]),
        "resample_unit": str(row["resample_unit"]),
        "n_replicates": int(row["n_replicates"]),
        "ci_level": float(row["ci_level"]),
    }


def cross_k_bootstrap_row(
    confidence_lo: Any,
    correct_lo: Any,
    confidence_hi: Any,
    correct_hi: Any,
    clusters: Any,
    *,
    eval_split: str,
    k_lo: Optional[int] = None,
    k_hi: Optional[int] = None,
    metrics: Sequence[str] = PRIMARY_METRICS,
    replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
) -> list[dict[str, Any]]:
    """Image-clustered paired bootstrap rows comparing K5 (``lo``) against K_b (``hi``).

    For every requested metric an **absolute** row is emitted with
    ``diff = metric(lo) - metric(hi)`` and ``diff_kind="absolute"``.  For
    ``e_aurc`` an extra **relative** row is emitted following the phase0b
    convention ``t = (metric_lo - metric_hi) / metric_hi`` (positive = K_b worse);
    that row additionally carries the unified worsening ``w = -t / (1 + t)`` and
    its mapped, sorted CI endpoints.

    ``eval_split`` names the split the ``clusters`` belong to.  ``k_lo`` / ``k_hi``
    are recorded as ``K_lo`` / ``K_hi`` when supplied.
    """
    a = SampleStats.from_conf_correct(confidence_lo, correct_lo)
    b = SampleStats.from_conf_correct(confidence_hi, correct_hi)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        fn = _metric_fn(metric)
        absolute = paired_cluster_bootstrap(
            fn,
            a,
            b,
            clusters,
            n_replicates=replicates,
            seed=seed,
            ci=ci,
            metric_name=metric,
        )
        rows.append(
            _cross_k_common_fields(
                absolute,
                eval_split=eval_split,
                k_lo=k_lo,
                k_hi=k_hi,
                metric=metric,
                diff_kind="absolute",
            )
        )
        if metric == "e_aurc":
            relative = paired_cluster_bootstrap_ratio(
                fn,
                a,
                b,
                clusters,
                n_replicates=replicates,
                seed=seed,
                ci=ci,
                metric_name=metric,
            )
            row = _cross_k_common_fields(
                relative,
                eval_split=eval_split,
                k_lo=k_lo,
                k_hi=k_hi,
                metric=metric,
                diff_kind="relative",
            )
            t = float(relative["diff"])
            t_lo = float(relative["ci_low"])
            t_hi = float(relative["ci_high"])
            w_lo, w_hi = worsening_ci_from_ratio_ci(t_lo, t_hi)
            row["worsening"] = worsening_from_ratio(t)
            row["worsening_ci_low"] = w_lo
            row["worsening_ci_high"] = w_hi
            rows.append(row)
    return rows


def model_vs_model_bootstrap_row(
    confidence_a: Any,
    correct_a: Any,
    confidence_b: Any,
    correct_b: Any,
    clusters: Any,
    *,
    eval_split: str,
    K: int,
    model_a: str,
    model_b: str,
    metrics: Sequence[str] = PRIMARY_METRICS,
    replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
) -> list[dict[str, Any]]:
    """Image-clustered paired bootstrap of ``metric(model_a) - metric(model_b)``.

    Same universe / same draw for both models (paired design); a positive ``diff``
    means ``model_a`` is better.  Emits one row per metric with the model pair and
    the ``K`` recorded.
    """
    a = SampleStats.from_conf_correct(confidence_a, correct_a)
    b = SampleStats.from_conf_correct(confidence_b, correct_b)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        fn = _metric_fn(metric)
        result = paired_cluster_bootstrap(
            fn,
            a,
            b,
            clusters,
            n_replicates=replicates,
            seed=seed,
            ci=ci,
            metric_name=metric,
        )
        rows.append(
            {
                "eval_split": eval_split,
                "K": int(K),
                "model_a": str(model_a),
                "model_b": str(model_b),
                "metric": metric,
                "diff": float(result["diff"]),
                "ci_low": float(result["ci_low"]),
                "ci_high": float(result["ci_high"]),
                "mean_a": float(result["mean_a"]),
                "mean_b": float(result["mean_b"]),
                "n": int(result["n"]),
                "n_clusters": int(result["n_clusters"]),
                "resample_unit": str(result["resample_unit"]),
                "n_replicates": int(result["n_replicates"]),
                "ci_level": float(result["ci_level"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# sufficiency gate (Amendment A6.6)
# ---------------------------------------------------------------------------
def _route_a_cell(cell: Mapping[str, Any], th: Mapping[str, float]) -> bool:
    return (
        cell["e_aurc_worsening"] >= th["e_aurc_worsen"]
        and cell["rer50_drop"] >= th["rer_drop_pp"]
        and cell["e_aurc_ci_low"] > 0.0
        and cell["rer50_ci_low"] > 0.0
    )


def _route_b_cell(cell: Mapping[str, Any], th: Mapping[str, float]) -> bool:
    return cell["auroc_drop"] >= th["auroc_drop"] and cell["auroc_drop_ci_low"] > 0.0


def sufficiency_verdict(
    cells: Sequence[Mapping[str, Any]],
    *,
    k50_point: Mapping[str, Any],
    thresholds: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Apply the Amendment A6.6 sufficiency gate.

    Parameters
    ----------
    cells:
        One dict per OOD / pooled cell with ``split`` (``"testA"`` / ``"testB"`` /
        ``"__pooled_test__"``), ``K`` and the drop fields ``auroc_drop`` /
        ``auroc_drop_ci_low`` / ``auroc_drop_ci_high``, ``e_aurc_worsening`` /
        ``e_aurc_ci_low`` / ``e_aurc_ci_high``, ``rer50_drop`` / ``rer50_ci_low`` /
        ``rer50_ci_high`` and ``e_aurc_abs`` (all positive = degradation).
    k50_point:
        ``{"auroc": ..., "e_aurc": ..., "rer_at_50": ...}`` absolute values of the
        model on ``__pooled_test__`` at ``K=50`` (for the override rule).
    thresholds:
        Optional overrides merged onto :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    dict
        ``verdict`` (priority ``override`` > ``GO`` > ``NO_GO`` > ``inconclusive``),
        ``route_A_cells`` / ``route_B_cells`` (``"split/K…"`` labels),
        ``n_failing_ood_cells``, ``override``, ``no_go_sufficient`` and the
        effective ``thresholds``.
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update({k: float(v) for k, v in thresholds.items()})

    ood_cells = [
        c for c in cells if int(c["K"]) in (20, 50) and c["split"] in ("testA", "testB")
    ]
    route_a_cells = [f"{c['split']}/K{int(c['K'])}" for c in ood_cells if _route_a_cell(c, th)]
    route_b_cells = [f"{c['split']}/K{int(c['K'])}" for c in ood_cells if _route_b_cell(c, th)]
    n_failing_ood_cells = sum(
        1 for c in ood_cells if _route_a_cell(c, th) or _route_b_cell(c, th)
    )
    go = n_failing_ood_cells >= 2

    pooled = next(
        (
            c
            for c in cells
            if int(c["K"]) == 50 and c["split"] == "__pooled_test__"
        ),
        None,
    )
    no_go_sufficient = False
    if pooled is not None:
        no_go_sufficient = (
            abs(pooled["auroc_drop"]) < th["auroc_stable"]
            and (
                pooled["e_aurc_worsening"] < th["e_aurc_worsen"]
                or pooled["e_aurc_abs"] < th["e_aurc_abs"]
            )
            and pooled["rer50_drop"] < th["rer_drop_pp"]
        )

    override = (
        float(k50_point["auroc"]) >= th["override_auroc"]
        and float(k50_point["e_aurc"]) <= th["override_e_aurc"]
        and float(k50_point["rer_at_50"]) >= th["override_rer"]
    )

    if override:
        verdict = "practically_solved_by_score_only"
    elif go:
        verdict = "GO_candidate_embeddings"
    elif no_go_sufficient:
        verdict = "NO_GO_candidate_embeddings"
    else:
        verdict = "inconclusive"

    return {
        "verdict": verdict,
        "route_A_cells": route_a_cells,
        "route_B_cells": route_b_cells,
        "n_failing_ood_cells": int(n_failing_ood_cells),
        "override": bool(override),
        "no_go_sufficient": bool(no_go_sufficient),
        "thresholds": th,
    }
