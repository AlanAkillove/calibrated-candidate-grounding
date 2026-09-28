"""Phase 1 (Amendment A7) candidate-semantic sufficiency evaluation.

This module adds the *semantic vs score-only* comparison on top of the frozen
Phase 0.5 reliability machinery.  It never re-derives a metric: every number is
produced by :mod:`ccg.reliability.evaluate` and
:func:`ccg.experiment.phase0b.paired_cluster_bootstrap_ratio` (whose
``metric_fn`` consumes a :class:`ccg.experiment.phase0a.SampleStats`, matching
``reval._metric_fn("e_aurc")``).

Sign conventions (A7.5)
-----------------------
* ``t = (E_sem - E_score) / E_score`` is the E-AURC *relative gap* of semantic
  against score-only: positive means semantic has the larger E-AURC (worse).
* ``reduction = -t = (E_score - E_sem) / E_score`` is the unified, human-readable
  quantity (positive = semantic improves E-AURC).  ``reduction`` is monotone
  *decreasing* in ``t``, so the CI endpoints are mapped element-wise and then
  min/max sorted (never read ``t`` without its ``reduction`` companion).

CSV field discipline
--------------------
:data:`RATIO_EXTRA_FIELDS` lists the columns this module adds on top of the
"reval bootstrap schema" (see :func:`ccg.reliability.evaluate.model_vs_model_bootstrap_row`).
A driver writing the merged ``bootstrap.csv`` should declare its column order as
``<reval bootstrap fields> + list(RATIO_EXTRA_FIELDS)``; the row returned by
:func:`model_vs_model_e_aurc_ratio_row` introduces no other stat column that a
``csv.DictWriter(extrasaction="raise")`` would reject.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Optional

from ccg.experiment import phase0b
from ccg.experiment.phase0a import SampleStats
from ccg.reliability import evaluate as reval

__all__ = [
    "RATIO_EXTRA_FIELDS",
    "SEMANTIC_THRESHOLDS",
    "model_vs_model_e_aurc_ratio_row",
    "reduction_ci_from_ratio_ci",
    "reduction_from_ratio",
    "semantic_gate",
]

#: Frozen A7.6 semantic-gate thresholds.
SEMANTIC_THRESHOLDS: dict[str, float] = {
    "auroc_gain": 0.02,
    "auroc_gain_strong": 0.03,
    "e_aurc_reduction": 0.10,
    "e_aurc_reduction_strong": 0.20,
    "rer50_gain_pp": 5.0,
    "rer50_gain_pp_strong": 10.0,
    "no_go_auroc": 0.01,
    "no_go_e_aurc": 0.05,
    "no_go_rer50_pp": 3.0,
    "seed_sig_min": 2,
    "seed_total": 3,
}

#: Columns added by this module on top of the reval bootstrap schema.
RATIO_EXTRA_FIELDS: tuple[str, ...] = ("reduction", "reduction_ci_low", "reduction_ci_high")


# ---------------------------------------------------------------------------
# relative ratio -> unified reduction mapping
# ---------------------------------------------------------------------------
def reduction_from_ratio(t: float) -> float:
    """Map the E-AURC relative gap ``t = (E_sem - E_score) / E_score`` to reduction.

    ``reduction = (E_score - E_sem) / E_score = -t``: positive means semantic has
    the *lower* (better) E-AURC.  Non-finite inputs map to ``nan``.
    """
    value = float(t)
    if not math.isfinite(value):
        return float("nan")
    return -value


def reduction_ci_from_ratio_ci(ratio_low: float, ratio_high: float) -> tuple[float, float]:
    """Map a ``t`` CI ``[ratio_low, ratio_high]`` to a sorted reduction CI ``(low, high)``.

    ``reduction`` is monotonically *decreasing* in ``t``, so the endpoints are
    mapped element-wise and then min/max sorted to keep the interval ordered
    regardless of the direction of ``ratio_low`` / ``ratio_high``.
    """
    a = reduction_from_ratio(ratio_low)
    b = reduction_from_ratio(ratio_high)
    return (float(min(a, b)), float(max(a, b)))


# ---------------------------------------------------------------------------
# model-vs-model E-AURC relative-reduction bootstrap row
# ---------------------------------------------------------------------------
def model_vs_model_e_aurc_ratio_row(
    confidence_sem: Any,
    correct_sem: Any,
    confidence_score: Any,
    correct_score: Any,
    clusters: Any,
    *,
    eval_split: str,
    K: int,
    model_sem: str,
    model_score: str,
    replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Paired image-clustered bootstrap of the E-AURC relative reduction.

    Compares the *semantic* reliability scorer against the *score-only* scorer on
    a shared draw (paired design) and reports the E-AURC relative reduction
    ``(E_score - E_sem) / E_score`` with its bootstrap CI.  The computation is a
    thin wrapper over
    :func:`ccg.experiment.phase0b.paired_cluster_bootstrap_ratio` with
    :func:`ccg.reliability.evaluate._metric_fn("e_aurc")`.

    Parameters
    ----------
    confidence_sem, correct_sem:
        ``[n]`` confidence and binary correctness of the semantic scorer.
    confidence_score, correct_score:
        ``[n]`` confidence and binary correctness of the score-only scorer.
    clusters:
        ``[n]`` image ids; the cluster (image) is the resample unit.
    eval_split:
        Name of the split the rows belong to (e.g. ``"__pooled_test__"``).
    K:
        Candidate-set size label; recorded only (it does not change the maths).
    model_sem, model_score:
        Model identifiers recorded as ``model_a`` / ``model_b``.
    replicates, seed, ci:
        Bootstrap replicates, RNG seed and CI level (forwarded verbatim).

    Returns
    -------
    dict
        ``metric="e_aurc"``, ``diff_kind="relative"``, ``model_a`` / ``model_b``,
        ``eval_split``, ``K``, ``diff`` (the ratio ``t``) and its CI
        (``ci_low`` / ``ci_high``), plus ``reduction = -t`` and its sorted CI
        (``reduction_ci_low`` / ``reduction_ci_high``).  The reval bootstrap
        schema columns (``mean_a`` / ``mean_b`` / ``n`` / ``n_clusters`` /
        ``resample_unit`` / ``n_replicates`` / ``ci_level``) are carried through,
        and the echoed ``seed`` / ``replicates`` / ``ci`` provenance columns are
        added.  See :data:`RATIO_EXTRA_FIELDS`.
    """
    a = SampleStats.from_conf_correct(confidence_sem, correct_sem)
    b = SampleStats.from_conf_correct(confidence_score, correct_score)
    metric_fn = reval._metric_fn("e_aurc")
    result = phase0b.paired_cluster_bootstrap_ratio(
        metric_fn,
        a,
        b,
        clusters,
        n_replicates=int(replicates),
        seed=int(seed),
        ci=float(ci),
        metric_name="e_aurc",
    )

    t = float(result["diff"])
    reduction = reduction_from_ratio(t)
    red_lo, red_hi = reduction_ci_from_ratio_ci(
        float(result["ci_low"]), float(result["ci_high"])
    )

    return {
        "eval_split": str(eval_split),
        "K": int(K),
        "model_a": str(model_sem),
        "model_b": str(model_score),
        "metric": "e_aurc",
        "diff_kind": "relative",
        "diff": t,
        "ci_low": float(result["ci_low"]),
        "ci_high": float(result["ci_high"]),
        "mean_a": float(result["mean_a"]),
        "mean_b": float(result["mean_b"]),
        "n": int(result["n"]),
        "n_clusters": int(result["n_clusters"]),
        "resample_unit": str(result["resample_unit"]),
        "n_replicates": int(result["n_replicates"]),
        "ci_level": float(result["ci_level"]),
        "reduction": reduction,
        "reduction_ci_low": red_lo,
        "reduction_ci_high": red_hi,
        "seed": int(seed),
        "replicates": int(replicates),
        "ci": float(ci),
    }


# ---------------------------------------------------------------------------
# A7.6 semantic gate (pure function; None-robust; never raises)
# ---------------------------------------------------------------------------
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


def _as_cell(cells: Any, k: int) -> Optional[Mapping[str, Any]]:
    if not isinstance(cells, Mapping):
        return None
    cell = cells.get(k)
    return cell if isinstance(cell, Mapping) else None


def _cell_conditions(cell: Mapping[str, Any], th: Mapping[str, float]) -> dict[str, bool]:
    """Per-condition booleans of ``cell_ok`` (A7.6); missing/None => ``False``."""
    delta = _finite_or_none(cell.get("delta_auroc"))
    delta_lo = _finite_or_none(cell.get("delta_auroc_ci_low"))
    red = _finite_or_none(cell.get("e_aurc_reduction"))
    red_lo = _finite_or_none(cell.get("e_aurc_reduction_ci_low"))
    rer = _finite_or_none(cell.get("rer50_gain_pp"))
    rer_lo = _finite_or_none(cell.get("rer50_gain_pp_ci_low"))

    cond: dict[str, bool] = {
        "delta_auroc_ge_thr": delta is not None and delta >= th["auroc_gain"],
        "delta_auroc_ci_low_gt0": delta_lo is not None and delta_lo > 0.0,
        "e_aurc_reduction_ge_thr": red is not None and red >= th["e_aurc_reduction"],
        "e_aurc_reduction_ci_low_gt0": red_lo is not None and red_lo > 0.0,
        "rer50_gain_ge_thr": rer is not None and rer >= th["rer50_gain_pp"],
        "rer50_gain_ci_low_gt0": rer_lo is not None and rer_lo > 0.0,
    }
    cond["e_aurc_branch"] = (
        cond["e_aurc_reduction_ge_thr"] and cond["e_aurc_reduction_ci_low_gt0"]
    )
    cond["rer_branch"] = cond["rer50_gain_ge_thr"] and cond["rer50_gain_ci_low_gt0"]
    cond["cell_ok"] = (
        cond["delta_auroc_ge_thr"]
        and cond["delta_auroc_ci_low_gt0"]
        and (cond["e_aurc_branch"] or cond["rer_branch"])
    )
    return cond


def _strong_k50(cell: Optional[Mapping[str, Any]], th: Mapping[str, float]) -> bool:
    """A7.6 STRONG condition, evaluated on the K50 cell only."""
    if cell is None:
        return False
    delta = _finite_or_none(cell.get("delta_auroc"))
    delta_lo = _finite_or_none(cell.get("delta_auroc_ci_low"))
    red = _finite_or_none(cell.get("e_aurc_reduction"))
    red_lo = _finite_or_none(cell.get("e_aurc_reduction_ci_low"))
    rer = _finite_or_none(cell.get("rer50_gain_pp"))
    rer_lo = _finite_or_none(cell.get("rer50_gain_pp_ci_low"))

    base = (
        delta is not None
        and delta >= th["auroc_gain_strong"]
        and delta_lo is not None
        and delta_lo > 0.0
    )
    e_branch = (
        red is not None
        and red >= th["e_aurc_reduction_strong"]
        and red_lo is not None
        and red_lo > 0.0
    )
    rer_branch = (
        rer is not None
        and rer >= th["rer50_gain_pp_strong"]
        and rer_lo is not None
        and rer_lo > 0.0
    )
    return bool(base and (e_branch or rer_branch))


def _no_go(cells: Any, th: Mapping[str, float]) -> bool:
    """A7.6 NO-GO: both K20 and K50 cells are simultaneously negligible."""
    for k in (20, 50):
        cell = _as_cell(cells, k)
        if cell is None:
            return False
        delta = _finite_or_none(cell.get("delta_auroc"))
        red = _finite_or_none(cell.get("e_aurc_reduction"))
        rer = _finite_or_none(cell.get("rer50_gain_pp"))
        if delta is None or red is None or rer is None:
            return False
        if not (
            delta < th["no_go_auroc"]
            and abs(red) < th["no_go_e_aurc"]
            and abs(rer) < th["no_go_rer50_pp"]
        ):
            return False
    return True


def _count_seed_significant(values: Any) -> int:
    """Number of ``values`` that are finite and strictly positive."""
    if values is None:
        return 0
    count = 0
    try:
        iterator = list(values)
    except TypeError:
        return 0
    for value in iterator:
        number = _finite_or_none(value)
        if number is not None and number > 0.0:
            count += 1
    return count


def semantic_gate(
    cells: Any,
    seed_k50_delta_ci_lows: Any,
    *,
    seed_mean_delta: Any,
    thresholds: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Apply the frozen A7.6 Candidate Semantic GO gate.

    Parameters
    ----------
    cells:
        ``{20: cell, 50: cell}``; each cell is a mapping with the keys
        ``delta_auroc``, ``delta_auroc_ci_low``, ``e_aurc_reduction``,
        ``e_aurc_reduction_ci_low``, ``rer50_gain_pp`` and
        ``rer50_gain_pp_ci_low``.  A missing cell or a missing/``None`` field is
        treated as "condition not satisfied".
    seed_k50_delta_ci_lows:
        Per-seed K50 ``ΔAUROC`` CI lower bounds; a seed counts as significant
        when its value is finite and ``> 0``.
    seed_mean_delta:
        The 3-seed mean ``ΔAUROC``; must be positive for PASS.
    thresholds:
        Optional overrides merged onto :data:`SEMANTIC_THRESHOLDS`.

    Returns
    -------
    dict
        ``verdict`` (priority ``STRONG`` > ``PASS`` > ``NO_GO`` > ``INCONCLUSIVE``),
        ``strong`` / ``pass`` / ``no_go`` booleans, ``cell_ok`` (``{20, 50}`` to
        ``bool`` or ``None`` when the cell is absent), per-cell
        ``cell_conditions``, ``n_seed_significant``, ``seed_sig_required``,
        ``seed_mean_delta`` and the effective ``thresholds``.  This function is a
        pure predicate over its inputs and never raises.
    """
    th = dict(SEMANTIC_THRESHOLDS)
    if thresholds:
        try:
            th.update({str(key): float(value) for key, value in thresholds.items()})
        except (TypeError, ValueError):
            pass

    try:
        cell_20 = _as_cell(cells, 20)
        cell_50 = _as_cell(cells, 50)
        cond_20 = _cell_conditions(cell_20, th) if cell_20 is not None else None
        cond_50 = _cell_conditions(cell_50, th) if cell_50 is not None else None
        ok_20 = bool(cond_20["cell_ok"]) if cond_20 is not None else None
        ok_50 = bool(cond_50["cell_ok"]) if cond_50 is not None else None

        n_seed_significant = _count_seed_significant(seed_k50_delta_ci_lows)
        seed_mean = _finite_or_none(seed_mean_delta)

        strong = _strong_k50(cell_50, th)
        seed_ok = (
            n_seed_significant >= int(th["seed_sig_min"])
            and seed_mean is not None
            and seed_mean > 0.0
        )
        pass_ = bool(ok_20) and bool(ok_50) and seed_ok
        no_go = _no_go(cells, th)

        if strong:
            verdict = "STRONG"
        elif pass_:
            verdict = "PASS"
        elif no_go:
            verdict = "NO_GO"
        else:
            verdict = "INCONCLUSIVE"

        cell_conditions: dict[int, dict[str, bool]] = {}
        if cond_20 is not None:
            cell_conditions[20] = cond_20
        if cond_50 is not None:
            cell_conditions[50] = cond_50

        return {
            "verdict": verdict,
            "strong": bool(strong),
            "pass": bool(pass_),
            "no_go": bool(no_go),
            "cell_ok": {20: ok_20, 50: ok_50},
            "cell_conditions": cell_conditions,
            "n_seed_significant": int(n_seed_significant),
            "seed_sig_required": int(th["seed_sig_min"]),
            "seed_mean_delta": seed_mean,
            "thresholds": th,
        }
    except Exception:  # noqa: BLE001 - the gate must never raise on bad input
        return {
            "verdict": "INCONCLUSIVE",
            "strong": False,
            "pass": False,
            "no_go": False,
            "cell_ok": {20: None, 50: None},
            "cell_conditions": {},
            "n_seed_significant": 0,
            "seed_sig_required": int(th["seed_sig_min"]),
            "seed_mean_delta": _finite_or_none(seed_mean_delta),
            "thresholds": th,
        }


# Re-exported for convenience so drivers need a single import for the row shape.
_BOOTSTRAP_BASE_FIELDS: tuple[str, ...] = (
    "eval_split",
    "K",
    "model_a",
    "model_b",
    "metric",
    "diff_kind",
    "diff",
    "ci_low",
    "ci_high",
    "mean_a",
    "mean_b",
    "n",
    "n_clusters",
    "resample_unit",
    "n_replicates",
    "ci_level",
    "seed",
    "replicates",
    "ci",
)

#: Full ordered column set of :func:`model_vs_model_e_aurc_ratio_row`
#: (``_BOOTSTRAP_BASE_FIELDS + RATIO_EXTRA_FIELDS``); a driver may use it as its
#: ``csv.DictWriter`` ``fieldnames`` for the semantic bootstrap table.
SEMANTIC_RATIO_FIELDS: tuple[str, ...] = _BOOTSTRAP_BASE_FIELDS + RATIO_EXTRA_FIELDS
