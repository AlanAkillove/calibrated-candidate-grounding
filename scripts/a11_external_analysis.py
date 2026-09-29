"""A11 external confirmation analysis (protocol A11.11-A11.13, sections 12-31).

Reads the six raw-prediction files written by :mod:`scripts.a11_external_inference`
(3 frozen B3 seeds x {Random-K5, SameCategory-K5}) and answers the two independent
questions the amendment froze, with **zero fitting**:

* **Q1 semantic transfer** - does ``Stats -> E1b`` still help on image-disjoint
  RefCOCOg under the same frozen pipeline?
* **Q2 hard amplification** - is that increment *larger* under same-category
  competition than under random candidates (the diff-of-diffs)?

Every statistic reuses the frozen Phase-1F machinery: the image-clustered paired
bootstrap (``hard_eval`` / ``reliability.evaluate`` / ``semantic.evaluate``) with
Random and Hard sharing the *same* cluster draws, the frozen 16-d semantic
features for the manipulation check, and the gate thresholds frozen in
``docs/research_protocol.md`` Amendment A11.13 *before* any prediction existed.
Random and Hard rows are the same cohort (matched), so ``sentence_id`` alignment
is asserted, never assumed.

    conda activate deepminer
    python scripts/a11_external_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.experiment.phase0a import SampleStats  # noqa: E402
from ccg.external import refcocog as rg  # noqa: E402
from ccg.external import refcocog_external as rx  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.semantic import evaluate as seval  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import hard_eval as heval  # noqa: E402

DEFAULT_PRED = Path("results/phase1e_refcocog_external/predictions")
DEFAULT_COHORT = Path("results/phase1e_refcocog_external/cohort")
DEFAULT_OUT = Path("results/phase1e_refcocog_external")
SEEDS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
REGIMES: Tuple[str, str] = ("random", "same_category")
REGIME_TAG = {"random": "rand5", "same_category": "hard5"}
POOLED = "pooled"
K = 5
PRIMARY_METRICS: Tuple[str, ...] = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
CRITERION_FEATURES: Tuple[str, ...] = ("cand_vmax", "cand_top12_sim", "clip_margin12")
CRITERION_EXPECTED_SIGN = {"cand_vmax": +1, "cand_top12_sim": +1, "clip_margin12": -1}

#: Frozen Amendment A11.13 gate thresholds (never adjusted after results exist).
A11_THRESHOLDS: Dict[str, float] = {
    "q1_delta_auroc": 0.015,
    "q1_delta_auroc_strong": 0.025,
    "q1_e_aurc_reduction": 0.05,
    "q1_e_aurc_reduction_strong": 0.10,
    "q1_rer50_gain_pp": 3.0,
    "q1_rer50_gain_pp_strong": 5.0,
    "q2_amplification_auroc": 0.005,
    "q2_amplification_strong_full": 0.015,
    "seed_sig_min": 2,
    "seed_total": 3,
    "severe_random_acc": 0.40,
    "severe_hard_acc": 0.25,
    "category_min_n": 300,
}


# ---------------------------------------------------------------------------
# loading + alignment
# ---------------------------------------------------------------------------
def _load_cell(pred_dir: Path, scorer: str, regime: str) -> Dict[str, Any]:
    tag = REGIME_TAG[regime]
    target = pred_dir / f"external__{scorer}__{tag}.npz"
    if not target.exists():
        raise FileNotFoundError(f"{target} missing - run scripts/a11_external_inference.py first")
    with np.load(target, allow_pickle=False) as store:
        return {
            "sentence_id": np.asarray(store["sentence_id"], dtype=np.int64),
            "ref_id": np.asarray(store["ref_id"], dtype=np.int64),
            "image_id": np.asarray(store["image_id"], dtype=np.int64),
            "correct": np.asarray(store["correct"], dtype=bool),
            "conf_stats": np.asarray(store["conf_stats"], dtype=np.float64),
            "conf_e1b": np.asarray(store["conf_e1b"], dtype=np.float64),
            "conf_msp": np.asarray(store["conf_msp"], dtype=np.float64),
            "margin": np.asarray(store["margin"], dtype=np.float64),
            "entropy": np.asarray(store["entropy"], dtype=np.float64),
            "sem16": np.asarray(store["sem16"], dtype=np.float64),
            "regime": regime,
            "scorer": scorer,
        }


def _load_cohort_meta(cohort: Any) -> Dict[int, Dict[str, Any]]:
    """``{expr_id: {text, n_tokens, category, absolute}}`` from the frozen cohort.

    The sidecar ``cohort.csv`` stores the distractor arrays as raw comma-bearing
    JSON, so it is not safely ``csv``-parseable; the authoritative row
    attributes come from the rebuilt in-memory cohort (whose ``text`` and
    ``category`` are the frozen A10 values).
    """
    meta: Dict[int, Dict[str, Any]] = {}
    for row in range(len(cohort)):
        expr_id = int(cohort.expr_id[row])
        text = str(cohort.text[row])
        tokens = set(rg.whitespace_tokens(text))
        category = str(cohort.category[row])
        meta[expr_id] = {
            "text": text,
            "n_tokens": len(tokens),
            "category": category,
            "absolute": bool(tokens & set(rg.ABSOLUTE_POSITION_TOKENS)),
        }
    return meta


def _check_matched(cells: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    for scorer in SEEDS:
        hard = cells[(scorer, "same_category")]
        rand = cells[(scorer, "random")]
        if not np.array_equal(hard["sentence_id"], rand["sentence_id"]):
            raise AssertionError(f"{scorer}: Random/Hard rows are not matched by sentence_id")
        if not np.array_equal(hard["image_id"], rand["image_id"]):
            raise AssertionError(f"{scorer}: Random/Hard image_id alignment drifted")


# ---------------------------------------------------------------------------
# per-cell metrics (Stats -> E1b) with image-clustered paired bootstrap
# ---------------------------------------------------------------------------
def _cell_metrics(cell: Dict[str, Any], *, replicates: int, seed: int, ci: float) -> Dict[str, Any]:
    clusters = cell["image_id"]
    rows = reval.model_vs_model_bootstrap_row(
        cell["conf_e1b"], cell["correct"], cell["conf_stats"], cell["correct"], clusters,
        eval_split=POOLED, K=K, model_a="e1b", model_b="stats",
        metrics=PRIMARY_METRICS, replicates=replicates, seed=seed, ci=ci,
    )
    by_metric = {str(r["metric"]): r for r in rows}
    ratio = seval.model_vs_model_e_aurc_ratio_row(
        cell["conf_e1b"], cell["correct"], cell["conf_stats"], cell["correct"], clusters,
        eval_split=POOLED, K=K, model_sem="e1b", model_score="stats",
        replicates=replicates, seed=seed, ci=ci,
    )
    auroc = by_metric["auroc_correct"]
    rer50 = by_metric["rer_at_50"]
    rer80 = by_metric["rer_at_80"]
    return {
        "n": int(len(cell["correct"])),
        "n_clusters": int(auroc["n_clusters"]),
        "accuracy": float(cell["correct"].mean()),
        "msp": float(cell["conf_msp"].mean()),
        "margin": float(cell["margin"].mean()),
        "entropy": float(cell["entropy"].mean()),
        "auroc_stats": float(auroc["mean_b"]),
        "auroc_e1b": float(auroc["mean_a"]),
        "delta_auroc": float(auroc["diff"]),
        "delta_auroc_ci_low": float(auroc["ci_low"]),
        "delta_auroc_ci_high": float(auroc["ci_high"]),
        "e_aurc_stats": float(by_metric["e_aurc"]["mean_b"]),
        "e_aurc_e1b": float(by_metric["e_aurc"]["mean_a"]),
        "e_aurc_reduction": float(ratio["reduction"]),
        "e_aurc_reduction_ci_low": float(ratio["reduction_ci_low"]),
        "rer50_stats": float(rer50["mean_b"]),
        "rer50_e1b": float(rer50["mean_a"]),
        "rer50_gain_pp": float(100.0 * rer50["diff"]),
        "rer50_gain_ci_low_pp": float(100.0 * rer50["ci_low"]),
        "rer80_gain_pp": float(100.0 * rer80["diff"]),
    }


# ---------------------------------------------------------------------------
# amplification (diff-of-diffs), Random/Hard same draws
# ---------------------------------------------------------------------------
def _amplification(hard: Dict[str, Any], rand: Dict[str, Any], *, replicates: int, seed: int, ci: float) -> Dict[str, Any]:
    clusters = hard["image_id"]
    auroc = heval.paired_diff_of_diffs_bootstrap(
        reval._metric_fn("auroc_correct"),
        SampleStats.from_conf_correct(hard["conf_e1b"], hard["correct"]),
        SampleStats.from_conf_correct(hard["conf_stats"], hard["correct"]),
        SampleStats.from_conf_correct(rand["conf_e1b"], rand["correct"]),
        SampleStats.from_conf_correct(rand["conf_stats"], rand["correct"]),
        clusters, n_replicates=replicates, seed=seed, ci=ci, metric_name="auroc",
    )
    rer50 = heval.paired_diff_of_diffs_bootstrap(
        reval._metric_fn("rer_at_50"),
        SampleStats.from_conf_correct(hard["conf_e1b"], hard["correct"]),
        SampleStats.from_conf_correct(hard["conf_stats"], hard["correct"]),
        SampleStats.from_conf_correct(rand["conf_e1b"], rand["correct"]),
        SampleStats.from_conf_correct(rand["conf_stats"], rand["correct"]),
        clusters, n_replicates=replicates, seed=seed, ci=ci, metric_name="rer50",
    )
    eaurc = heval.paired_diff_of_diffs_ratio_bootstrap(
        reval._metric_fn("e_aurc"),
        SampleStats.from_conf_correct(hard["conf_e1b"], hard["correct"]),
        SampleStats.from_conf_correct(hard["conf_stats"], hard["correct"]),
        SampleStats.from_conf_correct(rand["conf_e1b"], rand["correct"]),
        SampleStats.from_conf_correct(rand["conf_stats"], rand["correct"]),
        clusters, n_replicates=replicates, seed=seed, ci=ci, metric_name="e_aurc",
    )
    return {
        "ampl_auroc": float(auroc["diff"]),
        "ampl_auroc_ci_low": float(auroc["ci_low"]),
        "ampl_auroc_ci_high": float(auroc["ci_high"]),
        "ampl_rer50_gain_pp": float(100.0 * rer50["diff"]),
        "ampl_rer50_ci_low_pp": float(100.0 * rer50["ci_low"]),
        "ampl_eaurc_reduction": float(eaurc["diff"]),
        "ampl_eaurc_ci_low": float(eaurc["ci_low"]),
    }


# ---------------------------------------------------------------------------
# manipulation + difficulty (paired mean-shift), same cluster draws
# ---------------------------------------------------------------------------
def _manipulation(hard: Dict[str, Any], rand: Dict[str, Any], *, replicates: int, seed: int, ci: float) -> Dict[str, Any]:
    index = {name: pos for pos, name in enumerate(sfeat.SEMANTIC_STAT_NAMES)}
    clusters = hard["image_id"]
    features: Dict[str, Any] = {}
    n_pass = 0
    for feature in CRITERION_FEATURES:
        shift = heval.paired_shift_bootstrap(
            hard["sem16"][:, index[feature]], rand["sem16"][:, index[feature]], clusters,
            n_replicates=replicates, seed=seed, ci=ci, name=feature,
        )
        sign = CRITERION_EXPECTED_SIGN[feature]
        point = float(shift["diff"])
        ci_low, ci_high = float(shift["ci_low"]), float(shift["ci_high"])
        # significant toward the expected direction: point on that side AND the
        # whole 95% CI on the same side of 0 (the CI does not cross 0).
        if sign > 0:
            sig = point > 0.0 and ci_low > 0.0
        else:
            sig = point < 0.0 and ci_high < 0.0
        features[feature] = {
            "diff": point, "ci_low": ci_low, "ci_high": ci_high,
            "mean_hard": float(shift["mean_hard"]), "mean_rand": float(shift["mean_rand"]),
            "expected_sign": sign, "significant_expected": bool(sig),
        }
        n_pass += int(sig)
    return {"features": features, "n_criterion_significant": n_pass,
            "n_criterion_total": len(CRITERION_FEATURES), "valid": n_pass >= A11_THRESHOLDS["seed_sig_min"]}


def _difficulty(hard: Dict[str, Any], rand: Dict[str, Any], *, replicates: int, seed: int, ci: float) -> Dict[str, Any]:
    clusters = hard["image_id"]
    out: Dict[str, Any] = {}
    pairs = {
        "accuracy": (hard["correct"].astype(np.float64), rand["correct"].astype(np.float64)),
        "msp": (hard["conf_msp"], rand["conf_msp"]),
        "margin": (hard["margin"], rand["margin"]),
        "entropy": (hard["entropy"], rand["entropy"]),
    }
    for name, (hv, rv) in pairs.items():
        shift = heval.paired_shift_bootstrap(hv, rv, clusters, n_replicates=replicates, seed=seed, ci=ci, name=name)
        out[name] = {"diff": float(shift["diff"]), "ci_low": float(shift["ci_low"]),
                     "ci_high": float(shift["ci_high"]), "mean_hard": float(shift["mean_hard"]),
                     "mean_rand": float(shift["mean_rand"])}
    return out


# ---------------------------------------------------------------------------
# diagnostics (secondary; never enters a gate) - A11.29-31
# ---------------------------------------------------------------------------
def _subset_stats(cell: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    if int(mask.sum()) == 0:
        return {"n": 0, "accuracy": None, "stats_auroc": None, "e1b_auroc": None, "delta_auroc": None}
    auroc_fn = reval._metric_fn("auroc_correct")
    e1b = auroc_fn(SampleStats.from_conf_correct(cell["conf_e1b"][mask], cell["correct"][mask]))
    stats = auroc_fn(SampleStats.from_conf_correct(cell["conf_stats"][mask], cell["correct"][mask]))
    return {
        "n": int(mask.sum()),
        "accuracy": float(cell["correct"][mask].mean()),
        "stats_auroc": float(stats),
        "e1b_auroc": float(e1b),
        "delta_auroc": float(e1b - stats),
    }


def _diagnostics(cells: Dict[Tuple[str, str], Dict[str, Any]], meta: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Seed-1 SameCategory-K5 subgroup breakdowns of the semantic increment."""
    hard = cells[("b3_seed1", "same_category")]
    sid = hard["sentence_id"]
    tokens = np.asarray([meta.get(int(s), {}).get("n_tokens", 0) for s in sid], dtype=np.float64)
    absolute = np.asarray([bool(meta.get(int(s), {}).get("absolute", False)) for s in sid])
    person = np.asarray([str(meta.get(int(s), {}).get("category", "")).lower() == "person" for s in sid])

    length: Dict[str, Any] = {}
    labels = heval.quartile_labels(tokens)
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        length[quartile] = _subset_stats(hard, labels == quartile)

    position = {
        "contains_absolute": _subset_stats(hard, absolute),
        "no_absolute": _subset_stats(hard, ~absolute),
    }
    cat = {
        "person": _subset_stats(hard, person),
        "non_person": _subset_stats(hard, ~person),
        "reportable": bool(int(person.sum()) >= A11_THRESHOLDS["category_min_n"]
                           and int((~person).sum()) >= A11_THRESHOLDS["category_min_n"]),
    }
    return {"expression_length": length, "absolute_position": position, "category": cat,
            "note": "seed-1 SameCategory-K5 point estimates; secondary, never used for a gate"}


# ---------------------------------------------------------------------------
# 3-seed aggregation + A11.13 gates
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return float(np.mean(list(values)))


def _count_positive(values: Sequence[float]) -> int:
    return int(sum(1 for value in values if value is not None and float(value) > 0.0))


def _gate(
    per_seed: Dict[str, Dict[str, Any]], *, transfer: bool
) -> Dict[str, Any]:
    th = A11_THRESHOLDS
    hard = {s: per_seed[s]["same_category"] for s in SEEDS}
    if transfer:
        delta = [hard[s]["delta_auroc"] for s in SEEDS]
        ci_low = [hard[s]["delta_auroc_ci_low"] for s in SEEDS]
        reduction = [hard[s]["e_aurc_reduction"] for s in SEEDS]
        reduction_ci = [hard[s]["e_aurc_reduction_ci_low"] for s in SEEDS]
        rer = [hard[s]["rer50_gain_pp"] for s in SEEDS]
        rer_ci = [hard[s]["rer50_gain_ci_low_pp"] for s in SEEDS]
        mean_delta = _mean(delta)
        branch_e = _mean(reduction) >= th["q1_e_aurc_reduction"] and _count_positive(reduction_ci) >= th["seed_sig_min"]
        branch_r = _mean(rer) >= th["q1_rer50_gain_pp"] and _count_positive(rer_ci) >= th["seed_sig_min"]
        selective = branch_e or branch_r
        core = (
            mean_delta >= th["q1_delta_auroc"]
            and _count_positive(ci_low) >= th["seed_sig_min"]
            and _count_positive(delta) >= th["seed_sig_min"]
            and selective
        )
        strong = core and mean_delta >= th["q1_delta_auroc_strong"] and (
            _mean(reduction) >= th["q1_e_aurc_reduction_strong"] or _mean(rer) >= th["q1_rer50_gain_pp_strong"]
        )
        if core:
            decision = "YES"
        elif mean_delta <= 0.0:
            decision = "NO"
        else:
            decision = "GRAY"
        return {
            "gate": "Q1 semantic transfer",
            "decision": decision,
            "confirmed": bool(core),
            "strong": bool(strong),
            "mean_delta_auroc_hard": mean_delta,
            "mean_e_aurc_reduction_hard": _mean(reduction),
            "mean_rer50_gain_pp_hard": _mean(rer),
            "n_seeds_delta_positive": _count_positive(delta),
            "n_seeds_ci_positive": _count_positive(ci_low),
            "per_seed_delta_auroc": {s: hard[s]["delta_auroc"] for s in SEEDS},
            "thresholds": {k: th[k] for k in (
                "q1_delta_auroc", "q1_delta_auroc_strong", "q1_e_aurc_reduction",
                "q1_rer50_gain_pp", "seed_sig_min")},
        }

    amp = {s: per_seed[s]["amplification"] for s in SEEDS}
    ampl = [amp[s]["ampl_auroc"] for s in SEEDS]
    ampl_ci = [amp[s]["ampl_auroc_ci_low"] for s in SEEDS]
    sel_e = [amp[s]["ampl_eaurc_reduction"] for s in SEEDS]
    sel_e_ci = [amp[s]["ampl_eaurc_ci_low"] for s in SEEDS]
    sel_r = [amp[s]["ampl_rer50_gain_pp"] for s in SEEDS]
    sel_r_ci = [amp[s]["ampl_rer50_ci_low_pp"] for s in SEEDS]
    mean_ampl = _mean(ampl)
    selective = (
        _mean(sel_e) > 0.0 and _count_positive(sel_e_ci) >= th["seed_sig_min"]
    ) or (
        _mean(sel_r) > 0.0 and _count_positive(sel_r_ci) >= th["seed_sig_min"]
    )
    replicated = (
        mean_ampl >= th["q2_amplification_auroc"]
        and _count_positive(ampl_ci) >= th["seed_sig_min"]
        and selective
    )
    if replicated:
        decision = "YES"
    elif mean_ampl <= 0.0:
        decision = "NO"
    else:
        decision = "GRAY"
    return {
        "gate": "Q2 hard amplification",
        "decision": decision,
        "replicated": bool(replicated),
        "mean_amplification_auroc": mean_ampl,
        "mean_amplification_eaurc": _mean(sel_e),
        "mean_amplification_rer50_pp": _mean(sel_r),
        "n_seeds_ci_positive": _count_positive(ampl_ci),
        "per_seed_amplification_auroc": {s: amp[s]["ampl_auroc"] for s in SEEDS},
        "thresholds": {k: th[k] for k in ("q2_amplification_auroc", "seed_sig_min")},
    }


def _full_verdict(q1: Dict[str, Any], q2: Dict[str, Any], severe: bool, per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    th = A11_THRESHOLDS
    if severe:
        return {"case": "SEVERE", "full_verdict": "EXTERNAL INCONCLUSIVE",
                "reason": "severe external model failure line (A11.12) - report only, no replication claim"}
    t_yes, a_yes = q1["decision"] == "YES", q2["decision"] == "YES"
    t_no, a_no = q1["decision"] == "NO", q2["decision"] == "NO"
    strong_full = (
        t_yes and q1.get("strong") and a_yes
        and q2["mean_amplification_auroc"] >= th["q2_amplification_strong_full"]
    )
    if t_yes and a_yes:
        case, verdict = "A", "FULL EXTERNAL CONFIRMATION"
    elif t_yes and a_no:
        case, verdict = "B", "PARTIAL EXTERNAL CONFIRMATION"
    elif t_no and a_no:
        case, verdict = "C", "EXTERNAL NOT CONFIRMED"
    else:
        case, verdict = "D", "EXTERNAL INCONCLUSIVE"
    out = {"case": case, "full_verdict": verdict, "strong_full_replication": bool(strong_full)}
    if strong_full:
        out["descriptive_label"] = "STRONG FULL EXTERNAL REPLICATION"
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _seed_mean(per_seed: Dict[str, Dict[str, Any]], regime: str, field: str) -> float:
    return _mean([per_seed[s][regime][field] for s in SEEDS])


def _std(per_seed: Dict[str, Dict[str, Any]], regime: str, field: str) -> float:
    return float(np.std([per_seed[s][regime][field] for s in SEEDS]))


def _reload_cohort_for_meta(log) -> Any:
    """Rebuild the frozen cohort once, to source per-row text/category for diagnostics."""
    log("[a11-analysis] rebuilding the frozen cohort for diagnostic metadata ...")
    cohort, _sizes, report = rx.load_refcocog_cohort(persist_dir=None)
    if not (report["a10_pin"]["ok"] and report["matched_pair"]["ok"]):
        raise AssertionError("A11 cohort pin failed before diagnostics")
    return cohort


def run(args: argparse.Namespace, log) -> Dict[str, Any]:
    pred_dir = Path(args.pred_dir)
    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for scorer in SEEDS:
        for regime in REGIMES:
            cells[(scorer, regime)] = _load_cell(pred_dir, scorer, regime)
    _check_matched(cells)
    n_rows = int(len(cells[("b3_seed1", "random")]["correct"]))
    log(f"[a11-analysis] loaded {len(SEEDS)}x{len(REGIMES)} cells, n={n_rows} rows, matched-ok")

    meta = _load_cohort_meta(_reload_cohort_for_meta(log))
    per_seed: Dict[str, Dict[str, Any]] = {}
    for scorer in SEEDS:
        hard, rand = cells[(scorer, "same_category")], cells[(scorer, "random")]
        per_seed[scorer] = {
            "random": _cell_metrics(rand, replicates=args.replicates, seed=args.bootstrap_seed, ci=args.ci),
            "same_category": _cell_metrics(hard, replicates=args.replicates, seed=args.bootstrap_seed, ci=args.ci),
            "amplification": _amplification(hard, rand, replicates=args.replicates, seed=args.bootstrap_seed, ci=args.ci),
            "manipulation": _manipulation(hard, rand, replicates=args.replicates, seed=args.bootstrap_seed, ci=args.ci),
            "difficulty": _difficulty(hard, rand, replicates=args.replicates, seed=args.bootstrap_seed, ci=args.ci),
        }
        log(
            f"[a11-analysis] {scorer}: dAUROC rand={per_seed[scorer]['random']['delta_auroc']:+.4f} "
            f"hard={per_seed[scorer]['same_category']['delta_auroc']:+.4f} "
            f"ampl={per_seed[scorer]['amplification']['ampl_auroc']:+.4f} "
            f"manip_pass={per_seed[scorer]['manipulation']['n_criterion_significant']}/3"
        )

    mean_rand_acc = _seed_mean(per_seed, "random", "accuracy")
    mean_hard_acc = _seed_mean(per_seed, "same_category", "accuracy")
    th = A11_THRESHOLDS
    severe = bool(mean_rand_acc < th["severe_random_acc"] or mean_hard_acc < th["severe_hard_acc"])
    manipulation_valid = sum(1 for s in SEEDS if per_seed[s]["manipulation"]["valid"]) >= th["seed_sig_min"]

    q1 = _gate(per_seed, transfer=True)
    q2 = _gate(per_seed, transfer=False)
    if not manipulation_valid:
        q2 = {"gate": "Q2 hard amplification", "decision": "INVALID_MANIPULATION",
              "replicated": False, "note": "INVALID HARD MANIPULATION - external hard-replication invalid",
              "thresholds": {}}
    verdict = _full_verdict(q1, q2, severe, per_seed)

    table = {regime: {
        "accuracy_mean": _seed_mean(per_seed, regime, "accuracy"),
        "accuracy_std": _std(per_seed, regime, "accuracy"),
        "stats_auroc": _seed_mean(per_seed, regime, "auroc_stats"),
        "e1b_auroc": _seed_mean(per_seed, regime, "auroc_e1b"),
        "delta_auroc": _seed_mean(per_seed, regime, "delta_auroc"),
        "delta_auroc_std": _std(per_seed, regime, "delta_auroc"),
        "e_aurc_reduction": _seed_mean(per_seed, regime, "e_aurc_reduction"),
        "rer50_gain_pp": _seed_mean(per_seed, regime, "rer50_gain_pp"),
        "rer80_gain_pp": _seed_mean(per_seed, regime, "rer80_gain_pp"),
    } for regime in REGIMES}

    report = {
        "protocol": "A11",
        "schema": "a11-external-analysis-v1",
        "n_rows": n_rows,
        "n_clusters": per_seed["b3_seed1"]["same_category"]["n_clusters"],
        "replicates": int(args.replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "ci_level": float(args.ci),
        "severe_external_failure": severe,
        "mean_random_acc": mean_rand_acc,
        "mean_hard_acc": mean_hard_acc,
        "manipulation": {"valid": bool(manipulation_valid),
                         "per_seed": {s: per_seed[s]["manipulation"] for s in SEEDS}},
        "difficulty": {s: per_seed[s]["difficulty"] for s in SEEDS},
        "main_table": table,
        "per_seed": per_seed,
        "q1_semantic_transfer": q1,
        "q2_hard_amplification": q2,
        "full_verdict": verdict,
        "diagnostics": _diagnostics(cells, meta),
        "thresholds": th,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "a11_results.json")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = lambda message: print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: E731
    started = time.perf_counter()
    report = run(args, log)
    report["analysis_seconds"] = round(time.perf_counter() - started, 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(
        f"[a11-analysis] Q1={report['q1_semantic_transfer']['decision']} "
        f"Q2={report['q2_hard_amplification']['decision']} -> {report['full_verdict']['full_verdict']} "
        f"({report['analysis_seconds']}s) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
