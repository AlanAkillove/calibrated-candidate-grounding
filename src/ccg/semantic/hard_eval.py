"""Phase 1F (Amendment A8) evaluation primitives.

Two families of pure functions sit on top of the frozen Phase 0.5 / Phase 1
bootstrap machinery:

* :func:`paired_diff_of_diffs_bootstrap` — the A8.5 *primary* judgment quantity
  ``Δ^hard - Δ^rand`` with ``Δ^regime = metric(E1b) - metric(Stats)``, evaluated
  inside the *same* image-cluster draw for all four scalars (paired design);
* :func:`paired_shift_bootstrap` — the generic image-clustered paired mean-shift
  used by the A8.7 manipulation check and the grounding-difficulty table.

:func:`hard_gate` applies the frozen A8.6 gate (STRONG > CONFIRMED >
NO-CONFIRMATION > INCONCLUSIVE-HARD).  Its thresholds live in
:data:`A8_THRESHOLDS` and were frozen in ``docs/research_protocol.md``
(Amendment A8) *before* any hard-regime result existed; the function is a pure
predicate over its inputs and never raises.

Sign conventions (A8.5)
-----------------------
* ``ΔAUROC^regime = AUROC(E1b) - AUROC(Stats)``: positive = semantic better.
* ``E-AURC reduction = (E_stats - E_e1b) / E_stats``: positive = semantic better
  (reuses :func:`ccg.semantic.evaluate.reduction_from_ratio`).
* ``RER@50 gain_pp = 100 * (RER_e1b - RER_stats)``: positive = semantic better.
* ``diff_of_diffs = Δ^hard - Δ^rand``: positive = semantic increment larger
  under the hard composition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..experiment import phase0a

__all__ = [
    "A8_THRESHOLDS",
    "A8_VERDICT_LABELS",
    "hard_gate",
    "paired_diff_of_diffs_bootstrap",
    "paired_diff_of_diffs_ratio_bootstrap",
    "paired_shift_bootstrap",
    "quartile_labels",
]

#: Frozen A8.6 gate thresholds (never adjusted after results exist).
A8_THRESHOLDS: dict[str, float] = {
    "auroc_gain": 0.02,
    "auroc_gain_strong": 0.03,
    "e_aurc_reduction": 0.10,
    "e_aurc_reduction_strong": 0.20,
    "rer50_gain_pp": 5.0,
    "rer50_gain_pp_strong": 10.0,
    "diff_of_diffs_min": 0.005,
    "seed_sig_min": 2,
    "seed_total": 3,
}

#: Human-readable verdict labels of the A8.6 gate.
A8_VERDICT_LABELS: dict[str, str] = {
    "STRONG": "STRONG HARD-REGIME SEMANTIC SIGNAL",
    "CONFIRMED": "CONFIRMED HARD-REGIME SEMANTIC SIGNAL",
    "NO_CONFIRMATION": "NO-CONFIRMATION",
    "INCONCLUSIVE_HARD": "INCONCLUSIVE-HARD",
}


def _finite_or_none(value: Any) -> Optional[float]:
    """Return ``float(value)`` when finite, else ``None`` (``None``-tolerant)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


# ---------------------------------------------------------------------------
# diff-of-diffs paired cluster bootstrap (A8.5 primary quantity)
# ---------------------------------------------------------------------------
def paired_diff_of_diffs_bootstrap(
    metric_fn: Callable[[phase0a.SampleStats], float],
    hard_sem: phase0a.SampleStats,
    hard_score: phase0a.SampleStats,
    rand_sem: phase0a.SampleStats,
    rand_score: phase0a.SampleStats,
    cluster_ids: Any = None,
    *,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Image-clustered paired bootstrap of ``(sem - score)^hard - (sem - score)^rand``.

    All four prediction blocks must be aligned row-by-row on the same cohort
    (same ``ref_id`` / ``image_id`` / target): the hard and random candidate
    sets of row ``i`` share the target and the image cluster.  Every replicate
    draws one cluster resample and evaluates the four ``metric_fn`` values on
    the *same* drawn rows, so the reported interval is honest under
    within-image correlation and under the cross-regime pairing.

    Returns
    -------
    dict
        ``diff`` (``Δ^hard - Δ^rand`` point estimate), ``hard_diff`` /
        ``rand_diff`` (the two point estimates), ``ci_low`` / ``ci_high`` /
        ``ci_level`` / ``std_diff``, ``n`` / ``n_clusters`` /
        ``resample_unit`` / ``n_replicates`` and the raw ``replicates`` array.
    """
    a = hard_sem if isinstance(hard_sem, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*hard_sem)
    b = hard_score if isinstance(hard_score, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*hard_score)
    c = rand_sem if isinstance(rand_sem, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*rand_sem)
    d = rand_score if isinstance(rand_score, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*rand_score)
    n = len(a)
    for name, block in (("hard_score", b), ("rand_sem", c), ("rand_score", d)):
        if len(block) != n:
            raise ValueError(f"hard_sem has {n} rows but {name} has {len(block)}")
    if n < 2:
        raise ValueError("paired bootstrap needs at least 2 rows")
    reps = int(n_replicates)
    if reps < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    level = float(ci)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    hard_diff = float(metric_fn(a)) - float(metric_fn(b))
    rand_diff = float(metric_fn(c)) - float(metric_fn(d))
    rng = np.random.default_rng(int(seed))
    if cluster_ids is None:
        resample_unit = "sample"
        sampler = None
        n_clusters = n
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.shape[0] != n:
            raise ValueError(
                f"cluster_ids has {clusters.shape[0]} entries but predictions have {n} rows"
            )
        sampler = phase0a._ClusterSampler(clusters)
        resample_unit = "image"
        n_clusters = sampler.n_clusters

    replicates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        if sampler is None:
            draw = rng.integers(0, n, size=n)
        else:
            draw = sampler.draw(rng)
        hard_value = float(metric_fn(a.take(draw))) - float(metric_fn(b.take(draw)))
        rand_value = float(metric_fn(c.take(draw))) - float(metric_fn(d.take(draw)))
        replicates[index] = hard_value - rand_value

    alpha = (1.0 - level) / 2.0
    return {
        "metric": metric_name or getattr(metric_fn, "__name__", "metric"),
        "diff": float(hard_diff - rand_diff),
        "hard_diff": hard_diff,
        "rand_diff": rand_diff,
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "ci_level": level,
        "std_diff": float(np.std(replicates)),
        "n": int(n),
        "n_clusters": int(n_clusters),
        "resample_unit": resample_unit,
        "n_replicates": reps,
        "replicates": replicates,
    }


# ---------------------------------------------------------------------------
# diff-of-diffs of the E-AURC *relative reduction* (scale-free variant)
# ---------------------------------------------------------------------------
def paired_diff_of_diffs_ratio_bootstrap(
    metric_fn: Callable[[phase0a.SampleStats], float],
    hard_sem: phase0a.SampleStats,
    hard_score: phase0a.SampleStats,
    rand_sem: phase0a.SampleStats,
    rand_score: phase0a.SampleStats,
    cluster_ids: Any = None,
    *,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Image-clustered paired bootstrap of the reduction gain ``r^hard - r^rand``.

    Per regime ``x`` the statistic is the A8.5 relative reduction
    ``r^x = (E_score^x - E_sem^x) / E_score^x`` (positive = semantic better),
    with ``E`` produced by ``metric_fn`` (expected: the ``e_aurc`` metric; the
    function is generic for any *positive, lower-is-better* metric).  All four
    blocks share every cluster draw; replicates with a degenerate denominator
    (``E_score == 0``) are dropped from the quantile like
    :func:`ccg.experiment.phase0b.paired_cluster_bootstrap_ratio` does.

    Returns
    -------
    dict
        ``diff`` (``r^hard - r^rand`` point estimate), ``hard_reduction`` /
        ``rand_reduction``, the raw ``hard_diff`` / ``rand_diff``
        (``E_sem - E_score``), ``ci_low`` / ``ci_high`` / ``ci_level`` /
        ``std_diff``, ``n`` / ``n_clusters`` / ``resample_unit`` /
        ``n_replicates`` and the raw ``replicates`` array.
    """
    a = hard_sem if isinstance(hard_sem, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*hard_sem)
    b = hard_score if isinstance(hard_score, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*hard_score)
    c = rand_sem if isinstance(rand_sem, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*rand_sem)
    d = rand_score if isinstance(rand_score, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*rand_score)
    n = len(a)
    for name, block in (("hard_score", b), ("rand_sem", c), ("rand_score", d)):
        if len(block) != n:
            raise ValueError(f"hard_sem has {n} rows but {name} has {len(block)}")
    if n < 2:
        raise ValueError("paired bootstrap needs at least 2 rows")
    reps = int(n_replicates)
    if reps < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    level = float(ci)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    def _reduction(sub: phase0a.SampleStats, base: phase0a.SampleStats) -> float:
        denominator = float(metric_fn(base))
        if denominator == 0.0:
            return float("nan")
        return (denominator - float(metric_fn(sub))) / denominator

    hard_reduction = _reduction(a, b)
    rand_reduction = _reduction(c, d)
    hard_diff = float(metric_fn(a)) - float(metric_fn(b))
    rand_diff = float(metric_fn(c)) - float(metric_fn(d))
    rng = np.random.default_rng(int(seed))
    if cluster_ids is None:
        resample_unit = "sample"
        sampler = None
        n_clusters = n
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.shape[0] != n:
            raise ValueError(
                f"cluster_ids has {clusters.shape[0]} entries but predictions have {n} rows"
            )
        sampler = phase0a._ClusterSampler(clusters)
        resample_unit = "image"
        n_clusters = sampler.n_clusters

    replicates = np.full(reps, np.nan, dtype=np.float64)
    for index in range(reps):
        if sampler is None:
            draw = rng.integers(0, n, size=n)
        else:
            draw = sampler.draw(rng)
        hard_value = _reduction(a.take(draw), b.take(draw))
        rand_value = _reduction(c.take(draw), d.take(draw))
        if math.isfinite(hard_value) and math.isfinite(rand_value):
            replicates[index] = hard_value - rand_value
    valid = replicates[np.isfinite(replicates)]
    alpha = (1.0 - level) / 2.0
    if valid.size:
        ci_low = float(np.quantile(valid, alpha))
        ci_high = float(np.quantile(valid, 1.0 - alpha))
    else:
        ci_low = float("nan")
        ci_high = float("nan")
    return {
        "metric": metric_name or getattr(metric_fn, "__name__", "metric"),
        "diff": float(hard_reduction - rand_reduction),
        "hard_reduction": hard_reduction,
        "rand_reduction": rand_reduction,
        "hard_diff": hard_diff,
        "rand_diff": rand_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": level,
        "std_diff": float(np.nanstd(replicates)),
        "n": int(n),
        "n_clusters": int(n_clusters),
        "resample_unit": resample_unit,
        "n_replicates": reps,
        "replicates": replicates,
    }


# ---------------------------------------------------------------------------
# generic paired mean-shift (manipulation check / grounding difficulty)
# ---------------------------------------------------------------------------
def paired_shift_bootstrap(
    hard_values: Any,
    rand_values: Any,
    cluster_ids: Any = None,
    *,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Image-clustered paired bootstrap of ``mean(hard) - mean(rand)``.

    Both arrays must be aligned per row (same row universe under both regimes).
    """
    hard = np.asarray(hard_values, dtype=np.float64).reshape(-1)
    rand = np.asarray(rand_values, dtype=np.float64).reshape(-1)
    if hard.shape[0] != rand.shape[0]:
        raise ValueError(f"hard_values has {hard.shape[0]} entries but rand_values has {rand.shape[0]}")
    n = int(hard.shape[0])
    if n < 2:
        raise ValueError("paired bootstrap needs at least 2 rows")
    reps = int(n_replicates)
    if reps < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    level = float(ci)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    point = float(np.mean(hard) - np.mean(rand))
    rng = np.random.default_rng(int(seed))
    if cluster_ids is None:
        resample_unit = "sample"
        sampler = None
        n_clusters = n
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.shape[0] != n:
            raise ValueError(
                f"cluster_ids has {clusters.shape[0]} entries but values have {n} rows"
            )
        sampler = phase0a._ClusterSampler(clusters)
        resample_unit = "image"
        n_clusters = sampler.n_clusters

    replicates = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        if sampler is None:
            draw = rng.integers(0, n, size=n)
        else:
            draw = sampler.draw(rng)
        replicates[index] = float(np.mean(hard[draw]) - np.mean(rand[draw]))

    alpha = (1.0 - level) / 2.0
    return {
        "name": str(name) if name is not None else "shift",
        "diff": point,
        "mean_hard": float(np.mean(hard)),
        "mean_rand": float(np.mean(rand)),
        "ci_low": float(np.quantile(replicates, alpha)),
        "ci_high": float(np.quantile(replicates, 1.0 - alpha)),
        "ci_level": level,
        "std_diff": float(np.std(replicates)),
        "n": n,
        "n_clusters": int(n_clusters),
        "resample_unit": resample_unit,
        "n_replicates": reps,
        "replicates": replicates,
    }


# ---------------------------------------------------------------------------
# quartile labels (error-concentration diagnostic, A8.7)
# ---------------------------------------------------------------------------
def quartile_labels(values: Any) -> np.ndarray:
    """Deterministic ``Q1..Q4`` labels from row quantiles (stricter-than edges).

    Mirrors the Phase 1 ``_quartile_labels`` convention (Amendment A7.7), so the
    Phase 1F subgroup table is comparable with the Phase 1 error-subset table.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.quantile(array, [0.25, 0.50, 0.75])
    labels = np.full(array.shape[0], "Q1", dtype=object)
    labels[array > edges[0]] = "Q2"
    labels[array > edges[1]] = "Q3"
    labels[array > edges[2]] = "Q4"
    return labels


# ---------------------------------------------------------------------------
# A8.6 gate (pure predicate; None-robust; never raises)
# ---------------------------------------------------------------------------
def _cell_value(cell: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
    if not isinstance(cell, Mapping):
        return None
    return _finite_or_none(cell.get(key))


def _dod_value(diff_of_diffs: Any) -> Optional[float]:
    if isinstance(diff_of_diffs, Mapping):
        return _finite_or_none(diff_of_diffs.get("diff"))
    return _finite_or_none(diff_of_diffs)


def _count_seed_significant(values: Any) -> int:
    """Number of per-seed values that are finite and strictly positive."""
    if values is None:
        return 0
    try:
        iterator = list(values)
    except TypeError:
        return 0
    count = 0
    for value in iterator:
        number = _finite_or_none(value)
        if number is not None and number > 0.0:
            count += 1
    return count


def hard_gate(
    cell: Any,
    diff_of_diffs: Any,
    *,
    seed_delta_ci_lows: Any,
    seed_mean_delta: Any,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Apply the frozen A8.6 Hard-Competition Semantic Confirmation gate.

    Parameters
    ----------
    cell:
        The SameCat-K5 cell (3-seed means of the per-seed paired-bootstrap rows)
        with keys ``delta_auroc`` / ``delta_auroc_ci_low`` /
        ``e_aurc_reduction`` / ``e_aurc_reduction_ci_low`` / ``rer50_gain_pp`` /
        ``rer50_gain_pp_ci_low`` (same shape as the Phase 1 gate cells).
    diff_of_diffs:
        The A8.5 primary quantity ``Δ^hard - Δ^rand`` either as a mapping with a
        ``diff`` key (a :func:`paired_diff_of_diffs_bootstrap` result) or as a
        bare float.  Only the point estimate enters the gate.
    seed_delta_ci_lows:
        Per-seed SameCat-K5 ``ΔAUROC`` CI lower bounds; a seed counts as
        individually significant when its value is finite and ``> 0``.
    seed_mean_delta:
        The 3-seed mean ``ΔAUROC``; must be positive.
    thresholds:
        Optional overrides merged onto :data:`A8_THRESHOLDS`.

    Returns
    -------
    dict
        ``verdict`` (``STRONG`` > ``CONFIRMED`` > ``NO_CONFIRMATION`` >
        ``INCONCLUSIVE_HARD``), ``verdict_label``, the boolean roll-ups
        ``strong`` / ``confirmed`` / ``no_confirmation`` /
        ``stop_architecture_exploration``, the granular ``conditions`` mapping,
        ``n_seed_significant`` / ``seed_sig_required`` / ``seed_mean_delta`` /
        ``diff_of_diffs`` and the effective ``thresholds``.  Never raises.
    """
    th = dict(A8_THRESHOLDS)
    if thresholds:
        try:
            th.update({str(key): float(value) for key, value in thresholds.items()})
        except (TypeError, ValueError):
            pass

    try:
        delta = _cell_value(cell, "delta_auroc")
        delta_lo = _cell_value(cell, "delta_auroc_ci_low")
        red = _cell_value(cell, "e_aurc_reduction")
        red_lo = _cell_value(cell, "e_aurc_reduction_ci_low")
        rer = _cell_value(cell, "rer50_gain_pp")
        rer_lo = _cell_value(cell, "rer50_gain_pp_ci_low")
        dod = _dod_value(diff_of_diffs)
        n_seed_significant = _count_seed_significant(seed_delta_ci_lows)
        seed_mean = _finite_or_none(seed_mean_delta)

        cond: Dict[str, bool] = {
            "delta_auroc_ge_thr": delta is not None and delta >= float(th["auroc_gain"]),
            "delta_auroc_ci_low_gt0": delta_lo is not None and delta_lo > 0.0,
            "e_aurc_reduction_ge_thr": red is not None and red >= float(th["e_aurc_reduction"]),
            "e_aurc_reduction_ci_low_gt0": red_lo is not None and red_lo > 0.0,
            "rer50_gain_ge_thr": rer is not None and rer >= float(th["rer50_gain_pp"]),
            "rer50_gain_ci_low_gt0": rer_lo is not None and rer_lo > 0.0,
        }
        cond["branch_e_aurc"] = (
            cond["e_aurc_reduction_ge_thr"] and cond["e_aurc_reduction_ci_low_gt0"]
        )
        cond["branch_rer50"] = cond["rer50_gain_ge_thr"] and cond["rer50_gain_ci_low_gt0"]
        cond["branch"] = bool(cond["branch_e_aurc"] or cond["branch_rer50"])
        cond["seed_consistency"] = n_seed_significant >= int(th["seed_sig_min"])
        cond["seed_mean_positive"] = seed_mean is not None and seed_mean > 0.0
        cond["diff_of_diffs_ge_min"] = dod is not None and dod >= float(th["diff_of_diffs_min"])

        support = bool(
            cond["delta_auroc_ci_low_gt0"]
            and cond["branch"]
            and cond["seed_consistency"]
            and cond["seed_mean_positive"]
            and cond["diff_of_diffs_ge_min"]
        )
        core = bool(cond["delta_auroc_ge_thr"] and support)
        strong_core = bool(
            delta is not None
            and delta >= float(th["auroc_gain_strong"])
            and support
            and (
                (red is not None and red >= float(th["e_aurc_reduction_strong"]) and cond["branch_e_aurc"])
                or (rer is not None and rer >= float(th["rer50_gain_pp_strong"]) and cond["branch_rer50"])
            )
        )
        cond["confirmed_core"] = core
        cond["strong_core"] = strong_core

        no_confirmation = bool(
            delta is not None
            and red is not None
            and rer is not None
            and delta < float(th["auroc_gain"])
            and red < float(th["e_aurc_reduction"])
            and rer < float(th["rer50_gain_pp"])
        )

        if strong_core:
            verdict = "STRONG"
        elif core:
            verdict = "CONFIRMED"
        elif no_confirmation:
            verdict = "NO_CONFIRMATION"
        else:
            verdict = "INCONCLUSIVE_HARD"

        return {
            "verdict": verdict,
            "verdict_label": A8_VERDICT_LABELS[verdict],
            "strong": bool(strong_core),
            "confirmed": bool(core),
            "no_confirmation": bool(no_confirmation),
            "stop_architecture_exploration": bool(verdict == "NO_CONFIRMATION"),
            "conditions": cond,
            "n_seed_significant": int(n_seed_significant),
            "seed_sig_required": int(th["seed_sig_min"]),
            "seed_mean_delta": seed_mean,
            "diff_of_diffs": dod,
            "thresholds": th,
        }
    except Exception:  # noqa: BLE001 - the gate must never raise on bad input
        return {
            "verdict": "INCONCLUSIVE_HARD",
            "verdict_label": A8_VERDICT_LABELS["INCONCLUSIVE_HARD"],
            "strong": False,
            "confirmed": False,
            "no_confirmation": False,
            "stop_architecture_exploration": False,
            "conditions": {},
            "n_seed_significant": 0,
            "seed_sig_required": int(th["seed_sig_min"]),
            "seed_mean_delta": _finite_or_none(seed_mean_delta),
            "diff_of_diffs": _dod_value(diff_of_diffs),
            "thresholds": th,
        }
