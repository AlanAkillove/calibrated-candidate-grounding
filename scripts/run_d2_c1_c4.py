"""V2-D2 - C1 cardinality + C4 hard-semantic cross-dataset robustness analysis.

D2 changes exactly one thing relative to RefCOCO+: the referring-expression *language*
(RefCOCO keeps the spatial words RefCOCO+ removed).  The images, regions, frozen COCO
RPN bank and every model are reused verbatim, so the two frozen V1 primary analyses can
be replayed on the re-encoded RefCOCO cohort with **0 new training parameters**.  Nothing
here fits a model; the only computation is point metrics + image-cluster paired bootstrap
(5,000 reps, seed 0, CI 0.95 - identical to the frozen phase0b / phase1f / V2-G config)
over the raw predictions written by ``scripts/d2_frozen_inference.py``.

C1 - cardinality robustness (protocol "D2-C1")
    The frozen base reliability (``global_T_corrected`` = temperature-scaled B3 MSP,
    ``conf_msp``) is tracked across ``K = 5, 10, 20, 50`` on the *common* (K50-eligible)
    cohort.  The V2-G G3 gate is re-applied verbatim (``scripts/analyze_v2g_cardinality``
    thresholds): Route A ``ΔAUROC(K5->K50) >= 0.03`` with every seed CI excluding 0, or
    Route B ``E-AURC relative worsening >= 20%`` and ``RER@50 drop >= 10pp`` with CIs
    excluding 0.  Satisfying either -> ``C1 CROSS-DATASET REPLICATED``.

C4 - hard semantic amplification (protocol "D2-C4")
    On the matched Random-K5 / SameCategory-K5 cohort the R1 -> E1b semantic gain is
    measured (``Δrand`` / ``Δhard`` / amplification ``A = Δhard - Δrand``).  A frozen
    manipulation check runs first (``winner_competitor_max_cos`` up, ``winner_top2_cos``
    up, ``q_margin12`` down; per-seed >= 2/3 direction and >= 1 CI-excludes-0, backbone ok
    = >= 2/3 seeds).  Invalid manipulation -> ``C4 NOT ASSESSABLE`` (never FAIL).  Valid,
    the V2-G G4 gate (``run_v2g_hard._hard_gate``, DELTA_HARD_MIN 0.015, AMPLIFICATION_MIN
    0.01, CI lower bounds > 0) decides ``C4 CROSS-DATASET REPLICATED``.

Every quantity is reported side by side with the frozen RefCOCO+ (V1) reference so the
honest reading is *cross-dataset transfer under a shared COCO visual domain and a shared
frozen proposal system* - never cross-visual-domain generalisation.

    conda activate deepminer
    python scripts/run_d2_c1_c4.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _p in (str(_SRC), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg.experiment import phase0a, phase0b  # noqa: E402
from ccg.experiment.phase0a import SampleStats  # noqa: E402
from ccg.external import refcoco_lang as L  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.semantic import hard_eval as heval  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

# ---------------------------------------------------------------------------
# frozen inputs / constants
# ---------------------------------------------------------------------------
PRED_DIR = _REPO_ROOT / "results" / "v2_d2_refcoco_lang" / "predictions"
COHORT_CSV = _REPO_ROOT / "results" / "v2_d2_refcoco_lang" / "phase1" / "cohort.csv"
INFER_REPORT = PRED_DIR.parent / "inference_report.json"
OUT_DIR = _REPO_ROOT / "results" / "v2_d2_refcoco_lang"
PHASE1_AUDIT = OUT_DIR / "phase1" / "cohort_audit.json"
PHASE2_FEAS = OUT_DIR / "phase2" / "feasibility.json"
# RefCOCO+ (V1) references, read-only, for the side-by-side (never re-derived).
V1_PHASE0B_BOOTSTRAP = "results/phase0b_independent/seed_{seed}/bootstrap.csv"
V1_PHASE1F_GATE_JSON = _REPO_ROOT / "results" / "phase1f_hard_semantic" / "gate.json"

SEEDS: Tuple[int, ...] = (1, 2, 3)
SCORERS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
KS: Tuple[int, ...] = (5, 10, 20, 50)
K_BASELINE = 5
PRIMARY_KB = 50
STATS_NAME = "stats_logistic"          # R1
E1B_NAME = "e1b_stats_semantic"        # R2 / E1b
POOLED = "__pooled__"

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI = 0.95

# V2-G C1 cardinality gate (analyze_v2g_cardinality, frozen).
AUC_THRESHOLD = 0.03
EAURC_WORSEN_THRESHOLD = 0.20
RER50_DROP_THRESHOLD = 0.10
# V2-G C4 hard gate (run_v2g_hard, freeze item 9).
DELTA_HARD_MIN = 0.015
AMPLIFICATION_MIN = 0.01
MANIPULATION: Tuple[Tuple[str, str], ...] = tuple(v2feat.MANIPULATION_METRICS)
V2_SEM_INDEX = {name: pos for pos, name in enumerate(v2feat.V2_PRIMARY_SEMANTIC_NAMES)}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [D2-analysis] {message}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=float) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _mean(values: Sequence[float]) -> float:
    return float(np.mean([float(v) for v in values]))


def _std(values: Sequence[float]) -> float:
    return float(np.std([float(v) for v in values]))


def environment_info() -> Dict[str, Any]:
    """The canonical deepminer versions the full suite + every D2 stage ran under."""
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "device_count": int(torch.cuda.device_count()),
    }


def _extra_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for mod in ("sklearn", "transformers", "open_clip"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception as exc:  # pragma: no cover - best-effort provenance
            out[mod] = f"unavailable ({type(exc).__name__})"
    return out


# ---------------------------------------------------------------------------
# prediction loading (filter each job npz to its frozen analysis cohort)
# ---------------------------------------------------------------------------
def cohort_expr_ids(cohort: L.LangCohort, rows: Sequence[int]) -> set:
    return {int(cohort.expr_id[int(r)]) for r in rows}


def _pred_path(job: str, scorer: str) -> Path:
    return PRED_DIR / f"d2__{job}__{scorer}.npz"


def load_pred(job: str, scorer: str, keep_ids: set) -> Dict[str, np.ndarray]:
    """Load one prediction npz, keep only ``keep_ids`` rows, sort by ``sentence_id``.

    Sorting by ``sentence_id`` makes every job a canonical, order-independent view of the
    cohort, so two jobs restricted to the same id set align row-for-row (checked by the
    callers via ``sentence_id`` equality).
    """
    raw = np.load(_pred_path(job, scorer), allow_pickle=False)
    sid = np.asarray(raw["sentence_id"], dtype=np.int64)
    mask = np.isin(sid, np.fromiter(sorted(keep_ids), dtype=np.int64))
    order = np.argsort(sid[mask], kind="stable")
    idx = np.nonzero(mask)[0][order]
    out: Dict[str, np.ndarray] = {}
    for key in ("sentence_id", "ref_id", "image_id", "correct",
                "conf_msp", "conf_stats", "conf_e1b", "sem14"):
        out[key] = np.asarray(raw[key])[idx]
    out["correct"] = out["correct"].astype(np.float64)
    for key in ("conf_msp", "conf_stats", "conf_e1b", "sem14", "image_id"):
        out[key] = np.asarray(out[key], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# V1 (RefCOCO+) references for the side-by-side (read-only)
# ---------------------------------------------------------------------------
def load_v1_c1_reference() -> Dict[str, Any]:
    """The frozen phase0b K5->K50 ``global_T_corrected`` deltas (3-seed means)."""
    keys = {"auroc_correct|absolute", "e_aurc|relative", "rer_at_50|absolute"}
    per_key: Dict[str, List[float]] = {k: [] for k in keys}
    per_key_ci: Dict[str, List[float]] = {k: [] for k in keys}
    for seed in SEEDS:
        path = _REPO_ROOT / V1_PHASE0B_BOOTSTRAP.format(seed=seed)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row["eval_split"] == POOLED and row["variant"] == "global_T_corrected"
                        and int(row["K_a"]) == K_BASELINE and int(row["K_b"]) == PRIMARY_KB):
                    key = f"{row['metric']}|{row['diff_kind']}"
                    if key in per_key:
                        per_key[key].append(float(row["diff"]))
                        per_key_ci[key].append(float(row["ci_low"]))
    return {
        key: {"diff_mean": _mean(vals), "ci_low_mean": _mean(per_key_ci[key])}
        for key, vals in per_key.items() if vals
    }


def load_v1_c4_reference() -> Dict[str, Any]:
    """The frozen phase1f V1 gate: diff-of-diffs point estimate + delta."""
    if not V1_PHASE1F_GATE_JSON.exists():
        return {}
    gate = json.loads(V1_PHASE1F_GATE_JSON.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {"verdict": gate.get("verdict", {}).get("verdict")}
    dod = gate.get("diff_of_diffs_auroc", {})
    out["diff_of_diffs_auroc"] = {"diff": dod.get("diff"), "ci_low": dod.get("ci_low")}
    cell = gate.get("cell_samecat_k5", {})
    out["delta_hard"] = cell.get("delta_auroc")
    return out


# ---------------------------------------------------------------------------
# C1 - cardinality robustness
# ---------------------------------------------------------------------------
C1_METRIC_SPECS: Tuple[Tuple[str, str], ...] = (
    ("auroc_correct", "absolute"),
    ("e_aurc", "absolute"),
    ("e_aurc", "relative"),
    ("rer_at_50", "absolute"),
)
C1_POINT_FIELDS = ["model", "seed", "K", "n", "n_images",
                   "accuracy", "auroc_correct", "e_aurc", "rer_at_50"]
C1_BOOT_FIELDS = [
    "model", "seed", "K_a", "K_b", "variant", "metric", "diff_kind",
    "n", "n_clusters", "resample_unit", "mean_a", "mean_b", "diff",
    "ci_low", "ci_high", "ci_level", "n_replicates", "std_diff",
]


def c1_point_row(conf: np.ndarray, correct: np.ndarray, image_id: np.ndarray, k: int, seed: str) -> Dict[str, Any]:
    point = reval.point_metric_row(conf, correct, probability=conf)
    return {
        "model": "global_T_corrected", "seed": seed, "K": int(k),
        "n": int(conf.size), "n_images": int(np.unique(image_id).size),
        "accuracy": float(np.mean(correct)),
        "auroc_correct": float(point["auroc_correct"]),
        "e_aurc": float(point["e_aurc"]),
        "rer_at_50": float(point["rer_at_50"]),
    }


def _c1_bootstrap(fn, stats_a, stats_b, clusters, kind, metric):
    if kind == "relative":
        r = phase0b.paired_cluster_bootstrap_ratio(
            fn, stats_a, stats_b, clusters,
            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            ci=BOOTSTRAP_CI, metric_name=metric)
    else:
        r = phase0a.paired_cluster_bootstrap(
            fn, stats_a, stats_b, clusters,
            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            ci=BOOTSTRAP_CI, metric_name=metric)
    return r


def run_c1(common_ids: set) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    stage: Dict[Tuple[str, int], Dict[str, float]] = {}

    for seed in SEEDS:
        seed_key = f"seed{seed}"
        scorer = f"b3_seed{seed}"
        per_k: Dict[int, Dict[str, np.ndarray]] = {}
        for k in KS:
            d = load_pred(f"random_k{k}", scorer, common_ids)
            per_k[k] = d
            point_rows.append(c1_point_row(d["conf_msp"], d["correct"], d["image_id"], k, seed_key))
        base = per_k[K_BASELINE]
        clusters = base["image_id"]
        for k_b in (10, 20, PRIMARY_KB):
            other = per_k[k_b]
            if not np.array_equal(base["sentence_id"], other["sentence_id"]):
                raise AssertionError(f"{seed_key}: K5/K{k_b} rows not matched on the common cohort")
            stats_a = SampleStats.from_conf_correct(base["conf_msp"], base["correct"])
            stats_b = SampleStats.from_conf_correct(other["conf_msp"], other["correct"])
            for metric, kind in C1_METRIC_SPECS:
                fn = reval._metric_fn(metric)
                result = _c1_bootstrap(fn, stats_a, stats_b, clusters, kind, metric)
                boot_rows.append({
                    "model": "b3", "seed": seed_key, "K_a": K_BASELINE, "K_b": int(k_b),
                    "variant": "global_T_corrected", "metric": metric, "diff_kind": kind,
                    "n": int(result["n"]), "n_clusters": int(result["n_clusters"]),
                    "resample_unit": str(result["resample_unit"]),
                    "mean_a": float(result["mean_a"]), "mean_b": float(result["mean_b"]),
                    "diff": float(result["diff"]), "ci_low": float(result["ci_low"]),
                    "ci_high": float(result["ci_high"]), "ci_level": float(result["ci_level"]),
                    "n_replicates": int(result["n_replicates"]),
                    "std_diff": float(result.get("std_diff", float("nan"))),
                })
                stage.setdefault((f"{metric}|{kind}", k_b), {})
                stage[(f"{metric}|{kind}", k_b)][seed_key] = {
                    "diff": float(result["diff"]), "ci_low": float(result["ci_low"]),
                    "ci_high": float(result["ci_high"]),
                    "mean_a": float(result["mean_a"]), "mean_b": float(result["mean_b"]),
                }

    # 3-seed mean bootstrap rows.
    for (metric_key, k_b), per_seed in sorted(stage.items()):
        metric, kind = metric_key.split("|")
        vals = [per_seed[f"seed{s}"] for s in SEEDS]
        boot_rows.append({
            "model": "b3", "seed": "mean", "K_a": K_BASELINE, "K_b": int(k_b),
            "variant": "global_T_corrected", "metric": metric, "diff_kind": kind,
            "n": "", "n_clusters": "", "resample_unit": "image",
            "mean_a": "", "mean_b": "",
            "diff": _mean([v["diff"] for v in vals]),
            "ci_low": _mean([v["ci_low"] for v in vals]),
            "ci_high": _mean([v["ci_high"] for v in vals]),
            "ci_level": BOOTSTRAP_CI, "n_replicates": BOOTSTRAP_REPLICATES,
            "std_diff": _std([v["diff"] for v in vals]),
        })

    # b3-mean point rows per K.
    for k in KS:
        per_seed = [r for r in point_rows if r["K"] == k]
        point_rows.append({
            "model": "global_T_corrected", "seed": "mean", "K": int(k),
            "n": int(np.mean([r["n"] for r in per_seed])),
            "n_images": int(np.mean([r["n_images"] for r in per_seed])),
            "accuracy": _mean([r["accuracy"] for r in per_seed]),
            "auroc_correct": _mean([r["auroc_correct"] for r in per_seed]),
            "e_aurc": _mean([r["e_aurc"] for r in per_seed]),
            "rer_at_50": _mean([r["rer_at_50"] for r in per_seed]),
        })

    return point_rows, boot_rows, c1_gate(stage)


def _sign_excludes_zero(ci_low: float, ci_high: float) -> Tuple[bool, str]:
    if ci_low > 0:
        return True, "positive"
    if ci_high < 0:
        return True, "negative"
    return False, ("positive" if ci_low + ci_high > 0 else "negative")


def _worsening_from_rel(mean_a: float, mean_b: float, ci_low: float, ci_high: float) -> Tuple[float, float, float]:
    """Convert the (a-b)/b ratio (baseline K_b) to K5-baselined worsening + CI."""
    r = (mean_a - mean_b) / mean_b
    t = lambda x: -x / (1.0 + x)  # noqa: E731
    w = t(r)
    lo, hi = t(ci_low), t(ci_high)
    return w, min(lo, hi), max(lo, hi)


def c1_gate(stage: Dict[Tuple[str, int], Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
    """Re-apply the V2-G G3 cardinality gate to the per-seed K5->K50 bootstrap rows."""
    per_seed_auc = [stage[("auroc_correct|absolute", PRIMARY_KB)][f"seed{s}"] for s in SEEDS]
    per_seed_eaurc = [stage[("e_aurc|relative", PRIMARY_KB)][f"seed{s}"] for s in SEEDS]
    per_seed_rer = [stage[("rer_at_50|absolute", PRIMARY_KB)][f"seed{s}"] for s in SEEDS]

    auc_effects = [v["diff"] for v in per_seed_auc]
    auc_ci_lows = [v["ci_low"] for v in per_seed_auc]
    auc_all_excl = all(_sign_excludes_zero(v["ci_low"], v["ci_high"])[0] and v["ci_low"] > 0
                       for v in per_seed_auc)
    route_a = bool(_mean(auc_effects) >= AUC_THRESHOLD and auc_all_excl)

    eaurc_wors = [
        _worsening_from_rel(v["mean_a"], v["mean_b"], v["ci_low"], v["ci_high"])[0]
        for v in per_seed_eaurc
    ]
    eaurc_all_excl = all(_sign_excludes_zero(v["ci_low"], v["ci_high"])[0] for v in per_seed_eaurc)
    rer_effects = [v["diff"] for v in per_seed_rer]
    rer_all_excl = all(_sign_excludes_zero(v["ci_low"], v["ci_high"])[0] and v["ci_low"] > 0
                       for v in per_seed_rer)
    route_b = bool(_mean(eaurc_wors) >= EAURC_WORSEN_THRESHOLD
                   and _mean(rer_effects) >= RER50_DROP_THRESHOLD
                   and eaurc_all_excl and rer_all_excl)

    return {
        "primary_comparison": f"K{K_BASELINE}_vs_K{PRIMARY_KB}",
        "variant": "global_T_corrected",
        "auc_drop_mean": _mean(auc_effects),
        "auc_drop_ci_low_mean": _mean(auc_ci_lows),
        "auc_all_seeds_ci_exclude_0": bool(auc_all_excl),
        "eaurc_worsening_mean": _mean(eaurc_wors),
        "eaurc_all_seeds_ci_exclude_0": bool(eaurc_all_excl),
        "rer50_drop_mean": _mean(rer_effects),
        "rer50_all_seeds_ci_exclude_0": bool(rer_all_excl),
        "route_a_passed": route_a,
        "route_b_passed": route_b,
        "replicated": bool(route_a or route_b),
        "thresholds": {"route_a_auc_drop_min": AUC_THRESHOLD,
                       "route_b_eaurc_worsen_min": EAURC_WORSEN_THRESHOLD,
                       "route_b_rer50_drop_min": RER50_DROP_THRESHOLD},
    }


# ---------------------------------------------------------------------------
# C4 - hard semantic amplification
# ---------------------------------------------------------------------------
C4_MANIP_FIELDS = ["scorer", "feature", "direction", "mean_hard", "mean_rand",
                   "diff", "ci_low", "ci_high", "correct_direction", "ci_excludes_0",
                   "n", "n_clusters"]
C4_AMP_FIELDS = ["scorer", "n", "delta_hard", "delta_hard_ci_low", "delta_hard_ci_high",
                 "delta_rand", "delta_rand_ci_low", "delta_rand_ci_high",
                 "amplification", "amplification_ci_low", "amplification_ci_high",
                 "auroc_r1_hard", "auroc_e1b_hard", "auroc_r1_rand", "auroc_e1b_rand"]


def _auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    return reval._metric_fn("auroc_correct")(SampleStats.from_conf_correct(conf, correct))


def run_c4(matched_ids: set) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    manip_rows: List[Dict[str, Any]] = []
    amp_rows: List[Dict[str, Any]] = []
    per_seed: Dict[str, Dict[str, float]] = {}
    seed_valid_map: Dict[str, bool] = {}

    for scorer in SCORERS:
        hard = load_pred("hard_k5", scorer, matched_ids)
        rand = load_pred("random_k5", scorer, matched_ids)
        if not np.array_equal(hard["sentence_id"], rand["sentence_id"]):
            raise AssertionError(f"{scorer}: hard5/rand5 rows not matched on the cohort")
        clusters = hard["image_id"].astype(np.int64)

        dir_ok = ci_excl = 0
        for feature, direction in MANIPULATION:
            col = V2_SEM_INDEX[feature]
            res = heval.paired_shift_bootstrap(
                hard["sem14"][:, col], rand["sem14"][:, col], clusters,
                n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                ci=BOOTSTRAP_CI, name=feature)
            diff, lo, hi = float(res["diff"]), float(res["ci_low"]), float(res["ci_high"])
            correct_dir = diff > 0.0 if direction == "up" else diff < 0.0
            excludes = lo > 0.0 if direction == "up" else hi < 0.0
            dir_ok += int(correct_dir)
            ci_excl += int(excludes)
            manip_rows.append({
                "scorer": scorer, "feature": feature, "direction": direction,
                "mean_hard": float(res["mean_hard"]), "mean_rand": float(res["mean_rand"]),
                "diff": diff, "ci_low": lo, "ci_high": hi,
                "correct_direction": bool(correct_dir), "ci_excludes_0": bool(excludes),
                "n": int(res["n"]), "n_clusters": int(res["n_clusters"]),
            })
        seed_valid_map[scorer] = bool(dir_ok >= 2 and ci_excl >= 1)

        # R1 -> E1b AUROC gain, hard vs matched random, plus the shared-draw diff-of-diffs.
        auroc_fn = reval._metric_fn("auroc_correct")
        dh = reval.model_vs_model_bootstrap_row(
            hard["conf_e1b"], hard["correct"], hard["conf_stats"], hard["correct"], clusters,
            eval_split=POOLED, K=5, model_a=E1B_NAME, model_b=STATS_NAME,
            metrics=("auroc_correct",), replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)[0]
        dr = reval.model_vs_model_bootstrap_row(
            rand["conf_e1b"], rand["correct"], rand["conf_stats"], rand["correct"], clusters,
            eval_split=POOLED, K=5, model_a=E1B_NAME, model_b=STATS_NAME,
            metrics=("auroc_correct",), replicates=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)[0]
        dod = heval.paired_diff_of_diffs_bootstrap(
            auroc_fn, (hard["conf_e1b"], hard["correct"]), (hard["conf_stats"], hard["correct"]),
            (rand["conf_e1b"], rand["correct"]), (rand["conf_stats"], rand["correct"]), clusters,
            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI,
            metric_name="auroc_correct")
        amp_rows.append({
            "scorer": scorer, "n": int(dh["n"]),
            "delta_hard": float(dh["diff"]), "delta_hard_ci_low": float(dh["ci_low"]),
            "delta_hard_ci_high": float(dh["ci_high"]),
            "delta_rand": float(dr["diff"]), "delta_rand_ci_low": float(dr["ci_low"]),
            "delta_rand_ci_high": float(dr["ci_high"]),
            "amplification": float(dod["diff"]), "amplification_ci_low": float(dod["ci_low"]),
            "amplification_ci_high": float(dod["ci_high"]),
            "auroc_r1_hard": _auroc(hard["conf_stats"], hard["correct"]),
            "auroc_e1b_hard": _auroc(hard["conf_e1b"], hard["correct"]),
            "auroc_r1_rand": _auroc(rand["conf_stats"], rand["correct"]),
            "auroc_e1b_rand": _auroc(rand["conf_e1b"], rand["correct"]),
        })
        per_seed[scorer] = {
            "delta_hard": float(dh["diff"]), "delta_hard_ci_low": float(dh["ci_low"]),
            "delta_rand": float(dr["diff"]),
            "amplification": float(dod["diff"]), "amplification_ci_low": float(dod["ci_low"]),
        }

    n_valid = int(sum(seed_valid_map.values()))
    import math
    manipulation_ok = bool(n_valid >= math.ceil(2 * len(SCORERS) / 3))
    gate = d2_hard_gate(per_seed, manipulation_ok)
    manipulation_summary = {
        "metrics": [[f, d] for f, d in MANIPULATION],
        "per_seed_valid": seed_valid_map,
        "n_seed_valid": n_valid, "n_seeds": len(SCORERS),
        "rule": "per-seed: >=2/3 metrics correct dir AND >=1 CI excludes 0; ok = >=2/3 seeds",
        "manipulation_ok": manipulation_ok,
    }
    return manip_rows, amp_rows, gate, manipulation_summary


def d2_hard_gate(per_seed: Mapping[str, Mapping[str, float]], manipulation_ok: bool) -> Dict[str, Any]:
    """V2-G G4 (run_v2g_hard._hard_gate), replayed on the D2 cohort."""
    dh = [float(v["delta_hard"]) for v in per_seed.values()]
    dh_lo = [float(v["delta_hard_ci_low"]) for v in per_seed.values()]
    amp = [float(v["amplification"]) for v in per_seed.values()]
    amp_lo = [float(v["amplification_ci_low"]) for v in per_seed.values()]
    dr = [float(v["delta_rand"]) for v in per_seed.values()]
    delta_hard_mean = _mean(dh)
    delta_hard_ci_low_mean = _mean(dh_lo)
    amplification_mean = _mean(amp)
    amplification_ci_low_mean = _mean(amp_lo)
    n_same_dir = int(sum(1 for d in dh if d > 0.0))

    pass_delta = bool(delta_hard_mean >= DELTA_HARD_MIN and delta_hard_ci_low_mean > 0.0)
    pass_amp = bool(amplification_mean >= AMPLIFICATION_MIN and amplification_ci_low_mean > 0.0)
    instability = bool(pass_delta and pass_amp and n_same_dir <= 1)

    if not manipulation_ok:
        verdict = "NOT_ASSESSABLE"
    elif pass_delta and pass_amp:
        verdict = "HARD_SEMANTIC_REPLICATED"
    else:
        verdict = "NOT_REPLICATED"
    return {
        "protocol": "V2-G G4 item 9 (frozen), replayed on D2",
        "comparison": f"{STATS_NAME} -> {E1B_NAME} on SameCat-K5 (matched random control)",
        "delta_hard_mean": delta_hard_mean,
        "delta_hard_ci_low_mean": delta_hard_ci_low_mean,
        "delta_rand_mean": _mean(dr),
        "amplification_mean": amplification_mean,
        "amplification_ci_low_mean": amplification_ci_low_mean,
        "pass_delta_hard": pass_delta,
        "pass_amplification": pass_amp,
        "n_seed_same_direction": n_same_dir,
        "n_seeds": len(per_seed),
        "instability": instability,
        "manipulation_ok": bool(manipulation_ok),
        "verdict": verdict,
        "thresholds": {"delta_hard_min": DELTA_HARD_MIN,
                       "amplification_min": AMPLIFICATION_MIN,
                       "requires_manipulation_valid": True},
    }


# ---------------------------------------------------------------------------
# overall verdict
# ---------------------------------------------------------------------------
def overall_verdict(c1: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, Any]:
    c1_rep = bool(c1["replicated"])
    c4_label = {
        "HARD_SEMANTIC_REPLICATED": "C4 CROSS-DATASET REPLICATED",
        "NOT_ASSESSABLE": "C4 NOT ASSESSABLE (manipulation invalid)",
        "NOT_REPLICATED": "C4 NOT REPLICATED",
    }[gate["verdict"]]
    if c1_rep and gate["verdict"] == "HARD_SEMANTIC_REPLICATED":
        overall = "CORE FINDINGS CROSS-DATASET ROBUST"
    elif c1_rep and gate["verdict"] == "NOT_ASSESSABLE":
        overall = "CARDINALITY ROBUST; HARD AMPLIFICATION NOT ASSESSABLE"
    elif c1_rep:
        overall = "CARDINALITY ROBUST; HARD AMPLIFICATION NOT REPLICATED"
    else:
        overall = "C1 NOT REPLICATED (see checks)"
    return {
        "c1_cross_dataset": "C1 CROSS-DATASET REPLICATED" if c1_rep else "C1 NOT REPLICATED",
        "c4_cross_dataset": c4_label,
        "overall": overall,
        "boundary": "cross-dataset transfer under a shared COCO visual domain and a "
                    "shared frozen proposal system; NOT cross-visual-domain generalisation",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort-csv", type=Path, default=COHORT_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)

    cohort = L.load_cohort_csv(Path(args.cohort_csv))
    common_ids = cohort_expr_ids(cohort, L.common_cohort_rows(cohort))
    matched_ids = cohort_expr_ids(cohort, L.matched_hard_rows(cohort))
    _log(f"cohort rows={len(cohort)} common(C1)={len(common_ids)} matched_hard(C4)={len(matched_ids)}")

    c1_point, c1_boot, c1_gate_result = run_c1(common_ids)
    _log(f"C1 gate: route_a={c1_gate_result['route_a_passed']} route_b={c1_gate_result['route_b_passed']} "
         f"-> replicated={c1_gate_result['replicated']}")
    manip_rows, amp_rows, c4_gate, manip_summary = run_c4(matched_ids)
    _log(f"C4 manipulation_ok={manip_summary['manipulation_ok']} verdict={c4_gate['verdict']} "
         f"dod={c4_gate['amplification_mean']:+.5f}")

    verdict = overall_verdict(c1_gate_result, c4_gate)
    v1_c1 = load_v1_c1_reference()
    v1_c4 = load_v1_c4_reference()

    # --- write artifacts ---
    _write_csv(out / "c1_point.csv", c1_point, C1_POINT_FIELDS)
    _write_csv(out / "c1_bootstrap.csv", c1_boot, C1_BOOT_FIELDS)
    _write_csv(out / "c4_manipulation.csv", manip_rows, C4_MANIP_FIELDS)
    _write_csv(out / "c4_amplification.csv", amp_rows, C4_AMP_FIELDS)

    payload = {
        "artifact": "v2_d2_refcoco_lang_c1_c4",
        "protocol": "V2-D2 RefCOCO strict cross-dataset robustness (language-distribution transfer)",
        "cohort": {
            "rows": len(cohort), "images": cohort.n_images, "refs": cohort.n_refs,
            "common_c1": len(common_ids), "matched_hard_c4": len(matched_ids),
        },
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "ci": BOOTSTRAP_CI, "resample_unit": "image",
                      "shared_cluster_draws_for_hard_vs_rand": True},
        "models": {"scorer": "frozen phase0b B3 seeds (global_T_corrected)",
                   "reliability": f"frozen R1 {STATS_NAME} / E1b {E1B_NAME}",
                   "new_parameters": 0},
        "c1": {"gate": c1_gate_result,
               "v1_refcococo_plus_reference": v1_c1,
               "auc_by_k_mean": {str(r["K"]): r["auroc_correct"] for r in c1_point if r["seed"] == "mean"},
               "accuracy_by_k_mean": {str(r["K"]): r["accuracy"] for r in c1_point if r["seed"] == "mean"}},
        "c4": {"manipulation": manip_summary, "gate": c4_gate,
               "v1_refcococo_plus_reference": v1_c4},
        "verdict": verdict,
        "environment": {**environment_info(), **_extra_versions()},
        "provenance": {
            "cohort_csv_sha256": _sha256_file(args.cohort_csv),
            "inference_report_sha256": _sha256_file(INFER_REPORT) if INFER_REPORT.exists() else None,
            "phase1_audit_sha256": _sha256_file(PHASE1_AUDIT) if PHASE1_AUDIT.exists() else None,
            "phase2_feasibility_sha256": _sha256_file(PHASE2_FEAS) if PHASE2_FEAS.exists() else None,
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out / "c1_c4_verdict.json", payload)
    _log(f"wrote verdict -> {out / 'c1_c4_verdict.json'}")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
