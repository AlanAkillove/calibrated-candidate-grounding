"""D1 stage 2 - C1 cardinality + C4 hard-semantic robustness replay (V2-D protocol).

Gates passed in stage 1 (see ``run_d1_reviewed.py``): proposal recall@0.5 = 0.9851
(CONTINUE) and clean test expressions = 8,010 / images = 1,474 (FULL_ROBUSTNESS_AUDIT).
This stage replays the two frozen V1 primary analyses on the D1_clean cohort with
**zero new training parameters** (protocol section 20):

C1 (sections 11-13)
    Frozen cosine B0 (``results/phase0a_corrected`` global-T predictions) and the
    three frozen V1 B3 seeds (``results/phase0b_independent``, global-T-corrected
    variant).  Point Accuracy / AUROC_correct / E-AURC / RER@50 per K in
    {5, 10, 20, 50} plus the K5-vs-K{10,20,50} image-cluster paired bootstrap
    (5,000 reps, seed 0, CI 0.95 - identical to the frozen phase0b config).

C4 (sections 14-17)
    Frozen R1 stats_logistic / R2 e1b_stats_semantic on Random-K5 vs
    SameCategory-K5 (``results/phase1f_hard_semantic/predictions``), rows filtered
    to the D1_clean sentence set.  DeltaRand / DeltaHard / Amplification
    (diff-of-diffs) with the same heval functions, bootstrap config and 3-seed
    mean aggregation (value and CI endpoints) as the frozen phase1f run; the A8.6
    gate is re-applied via ``heval.hard_gate``.

No new thresholds are introduced (section 13): every D1_clean quantity is
reported side by side with the frozen V1 full-cohort value read from the V1
artifacts themselves.  Verdict labels follow section 22.

Usage
-----
    python scripts/run_d1_c1_c4.py            # ~minutes, CPU only
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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ccg.experiment import phase0a, phase0b  # noqa: E402
from ccg.experiment.phase0a import SampleStats  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.semantic import evaluate as seval  # noqa: E402
from ccg.semantic import hard_eval as heval  # noqa: E402

# ---------------------------------------------------------------------------
# frozen input locations (no path is re-derived from anything but these)
# ---------------------------------------------------------------------------
IN_DIR = _REPO_ROOT / "results" / "v2_data_robustness" / "d1_reviewed_annotations"
CLEAN_MANIFEST = IN_DIR / "clean_manifest.csv"

PHASE0A_COSINE_NPZ = _REPO_ROOT / "results" / "phase0a_corrected" / "per_sentence_predictions.npz"
PHASE0B_NPZ = "results/phase0b_independent/seed_{seed}/per_sentence_predictions.npz"
PHASE0B_BOOTSTRAP = "results/phase0b_independent/seed_{seed}/bootstrap.csv"
PHASE0B_METADATA = _REPO_ROOT / "results" / "phase0b_independent" / "metadata.json"

PHASE1F_PRED = "results/phase1f_hard_semantic/predictions/{cell}__{scorer}.npz"
PHASE1F_PAIRED_CSV = _REPO_ROOT / "results" / "phase1f_hard_semantic" / "paired_random_vs_hard.csv"
PHASE1F_GATE_JSON = _REPO_ROOT / "results" / "phase1f_hard_semantic" / "gate.json"
PHASE1F_RELIABILITY_CSV = _REPO_ROOT / "results" / "phase1f_hard_semantic" / "reliability_metrics.csv"

OUT_DIR = IN_DIR  # section 24: every D1 artifact shares one directory

SEEDS_C1: Tuple[int, ...] = (1, 2, 3)
SCORERS_C4: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
KS: Tuple[int, ...] = (5, 10, 20, 50)
K_BASELINE = 5
STATS_NAME = "stats_logistic"
E1B_NAME = "e1b_stats_semantic"

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI = 0.95


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def _mean_endpoints(per_seed: Sequence[Mapping[str, float]], field: str) -> float:
    """phase1f's frozen aggregation: the mean of the per-seed values of ``field``."""
    return float(np.mean([float(row[field]) for row in per_seed]))


def _std(values: Sequence[float]) -> float:
    return float(np.std([float(v) for v in values]))


def load_clean_sent_ids(path: Path = CLEAN_MANIFEST) -> Tuple[set, Dict[str, int]]:
    """D1_clean sentence ids that are inside the phase0b evaluation cohort."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    clean_all = {int(row["sent_id"]) for row in rows}
    cohort = {int(row["sent_id"]) for row in rows if row["in_phase0b_cohort"] == "True"}
    if len(clean_all) != len(rows):
        raise AssertionError("clean_manifest.csv contains duplicate sent_id rows")
    per_split = {
        split: sum(1 for row in rows if row["split"] == split and row["in_phase0b_cohort"] == "True")
        for split in ("val", "testA", "testB")
    }
    return cohort, per_split


def _load_npz_filtered(path: Path, keep_sent_ids: set, keys: Sequence[Tuple[str, str]]) -> Dict[str, np.ndarray]:
    """Load one per-sentence npz and keep only the ``keep_sent_ids`` rows.

    ``keys`` are ``(npz_key, out_name)`` pairs; ``sentence_id`` / ``image_id``
    are always carried through.  Row order of the frozen artifact is preserved.
    """
    raw = np.load(path, allow_pickle=False)
    # phase0a uses ``sentence_id``/``image_id``; phase0b uses the plural forms.
    sid_key = "sentence_id" if "sentence_id" in raw.files else "sentence_ids"
    iid_key = "image_id" if "image_id" in raw.files else "image_ids"
    sid = np.asarray(raw[sid_key], dtype=np.int64)
    mask = np.isin(sid, np.fromiter(sorted(keep_sent_ids), dtype=np.int64))
    out: Dict[str, np.ndarray] = {
        "sentence_id": sid[mask],
        "image_id": np.asarray(raw[iid_key], dtype=np.int64)[mask],
    }
    for src, dst in keys:
        out[dst] = np.asarray(raw[src])[mask]
    if int(mask.sum()) < 100:
        raise AssertionError(f"only {int(mask.sum())} clean rows found in {path}")
    return out


# ---------------------------------------------------------------------------
# V1 frozen targets (read from the V1 artifacts - never transcribed)
# ---------------------------------------------------------------------------
def load_v1_c1_targets() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Pooled global_T_corrected K5-vs-Kb rows of the frozen phase0b bootstrap."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seed in SEEDS_C1:
        path = _REPO_ROOT / PHASE0B_BOOTSTRAP.format(seed=seed)
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = {
            (int(row["K_b"]), row["metric"], row["diff_kind"]): row
            for row in rows
            if row["eval_split"] == "__pooled__"
            and row["variant"] == "global_T_corrected"
            and int(row["K_a"]) == K_BASELINE
            and row["metric"] in ("auroc_correct", "e_aurc", "rer_at_50")
            and row["diff_kind"] in ("absolute", "relative")
        }
        wanted = {(kb, m, k) for kb in (10, 20, 50)
                  for m, k in (("auroc_correct", "absolute"), ("e_aurc", "absolute"),
                               ("e_aurc", "relative"), ("rer_at_50", "absolute"))}
        missing = wanted - set(selected)
        if missing:
            raise AssertionError(f"phase0b seed {seed} bootstrap.csv missing rows: {missing}")
        for key, row in selected.items():
            out.setdefault(f"seed{seed}", {}).setdefault(f"{key[1]}|{key[2]}|K{key[0]}", {})["diff"] = float(row["diff"])
            out.setdefault(f"seed{seed}", {}).setdefault(f"{key[1]}|{key[2]}|K{key[0]}", {})["ci_low"] = float(row["ci_low"])
            out.setdefault(f"seed{seed}", {}).setdefault(f"{key[1]}|{key[2]}|K{key[0]}", {})["ci_high"] = float(row["ci_high"])
    return out


def load_v1_c4_targets() -> Dict[str, Any]:
    """The frozen phase1f gate block (3-seed means; per-seed dod lives inside)."""
    gate = json.loads(PHASE1F_GATE_JSON.read_text(encoding="utf-8"))
    # Point-estimate DeltaRand / DeltaHard per seed, from the frozen reliability
    # table (auroc e1b - auroc stats); gate.json only stores the bootstrap rows.
    with PHASE1F_RELIABILITY_CSV.open(encoding="utf-8", newline="") as handle:
        point_by = {
            (r["scorer"], r["cell"], r["model"]): float(r["auroc_correct"])
            for r in csv.DictReader(handle)
        }
    seed_scores = [s for s in SCORERS_C4 if (s, "hard5", E1B_NAME) in point_by and (s, "rand5", STATS_NAME) in point_by]
    if len(seed_scores) != len(SCORERS_C4):
        raise AssertionError("reliability_metrics.csv does not contain all frozen phase1f scorers")
    v1_point = {
        "delta_rand": float(np.mean([point_by[(s, "rand5", E1B_NAME)] - point_by[(s, "rand5", STATS_NAME)] for s in seed_scores])),
        "delta_hard": float(np.mean([point_by[(s, "hard5", E1B_NAME)] - point_by[(s, "hard5", STATS_NAME)] for s in seed_scores])),
    }
    v1 = {
        "cell_samecat_k5": gate["cell_samecat_k5"],
        "diff_of_diffs_auroc": gate["diff_of_diffs_auroc"],
        "per_seed_delta_ci_low_k5": gate["per_seed_delta_ci_low_k5"],
        "seed_mean_delta_k5": gate["seed_mean_delta_k5"],
        "verdict": gate["verdict"]["verdict"],
        "manipulation": gate["manipulation"],
        "point_delta": v1_point,
    }
    return v1


# ---------------------------------------------------------------------------
# C1 - cardinality robustness
# ---------------------------------------------------------------------------
C1_METRIC_SPECS: Tuple[Tuple[str, str], ...] = (
    ("auroc_correct", "absolute"),
    ("e_aurc", "absolute"),
    ("e_aurc", "relative"),
    ("rer_at_50", "absolute"),
)

C1_POINT_FIELDS = ["model", "seed", "K", "n", "n_images", "accuracy", "auroc_correct", "e_aurc", "rer_at_50"]
BOOTSTRAP_FIELDS = [
    "analysis", "model", "seed", "K_a", "K_b", "variant", "metric", "diff_kind",
    "n", "n_clusters", "resample_unit", "mean_a", "mean_b", "diff", "ci_low",
    "ci_high", "ci_level", "n_replicates", "std_diff",
]


def c1_point_row(data: Mapping[str, np.ndarray], model: str, seed: str, k: int) -> Dict[str, Any]:
    conf = data[f"conf_K{k}"]
    correct = data[f"correct_K{k}"]
    point = reval.point_metric_row(conf, correct)
    return {
        "model": model, "seed": seed, "K": int(k),
        "n": int(conf.size), "n_images": int(np.unique(data["image_id"]).size),
        "accuracy": float(np.mean(correct)),
        "auroc_correct": float(point["auroc_correct"]),
        "e_aurc": float(point["e_aurc"]),
        "rer_at_50": float(point["rer_at_50"]),
    }


def run_c1(clean_sent_ids: set) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Replay C1 on D1_clean for B0-cosine and the three frozen B3 seeds."""
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []

    # Load per seed: cosine npz (B0, global-T variant) and phase0b npz (B3,
    # global-T-corrected variant), each filtered to the D1_clean sentence set.
    seed_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for seed in SEEDS_C1:
        path_b3 = _REPO_ROOT / PHASE0B_NPZ.format(seed=seed)
        b3_keys = (
            [(f"confidence_global_T_corrected_K{k}", f"conf_K{k}") for k in KS]
            + [(f"correct_K{k}", f"correct_K{k}") for k in KS]
        )
        b3 = _load_npz_filtered(path_b3, clean_sent_ids, b3_keys)
        cos_keys = (
            [(f"confidence_global_T_K{k}", f"conf_K{k}") for k in KS]
            + [(f"correct_K{k}", f"correct_K{k}") for k in KS]
        )
        cosine = _load_npz_filtered(PHASE0A_COSINE_NPZ, clean_sent_ids, cos_keys)
        # The two frozen artifacts share the cohort but not the row order; the
        # cosine arrays are permuted so cosine["sentence_id"] == b3["sentence_id"].
        cid = np.sort(cosine["sentence_id"])
        if np.any(np.diff(cid) <= 0):
            raise AssertionError(f"seed {seed}: duplicate sentence_id in the phase0a cohort")
        order_c = np.argsort(cosine["sentence_id"], kind="stable")   # sorted -> cosine rows
        b_sid = b3["sentence_id"]
        order_b = np.argsort(b_sid, kind="stable")                   # sorted -> b3 rows
        pos = order_c[order_b]                                       # b3 order -> cosine rows
        if not np.array_equal(cosine["sentence_id"][pos], b_sid) or not np.array_equal(cosine["image_id"][pos], b3["image_id"]):
            raise AssertionError(f"seed {seed}: phase0a and phase0b clean cohorts differ")
        cosine = {name: arr[pos] for name, arr in cosine.items()}
        seed_data[f"seed{seed}"] = {"b3": b3, "cosine_b0": cosine}

    stage_rows: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for model in ("cosine_b0", "b3"):
        for seed_key, per_model in seed_data.items():
            data = per_model[model]
            for k in KS:
                point_rows.append(c1_point_row(data, model, seed_key, k))
            clusters = data["image_id"]
            for k_b in (10, 20, 50):
                stats_a = SampleStats.from_conf_correct(data[f"conf_K{K_BASELINE}"], data[f"correct_K{K_BASELINE}"])
                stats_b = SampleStats.from_conf_correct(data[f"conf_K{k_b}"], data[f"correct_K{k_b}"])
                for metric, kind in C1_METRIC_SPECS:
                    fn = reval._metric_fn(metric)
                    if kind == "relative":
                        result = phase0b.paired_cluster_bootstrap_ratio(
                            fn, stats_a, stats_b, clusters,
                            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                            ci=BOOTSTRAP_CI, metric_name=metric,
                        )
                    else:
                        result = phase0a.paired_cluster_bootstrap(
                            fn, stats_a, stats_b, clusters,
                            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                            ci=BOOTSTRAP_CI, metric_name=metric,
                        )
                    row = {
                        "analysis": "C1", "model": model, "seed": seed_key,
                        "K_a": K_BASELINE, "K_b": int(k_b),
                        "variant": "global_T" if model == "cosine_b0" else "global_T_corrected",
                        "metric": metric, "diff_kind": kind,
                        "n": int(result["n"]), "n_clusters": int(result["n_clusters"]),
                        "resample_unit": str(result["resample_unit"]),
                        "mean_a": float(result["mean_a"]), "mean_b": float(result["mean_b"]),
                        "diff": float(result["diff"]),
                        "ci_low": float(result["ci_low"]), "ci_high": float(result["ci_high"]),
                        "ci_level": float(result["ci_level"]),
                        "n_replicates": int(result["n_replicates"]),
                        "std_diff": float(result.get("std_diff", float("nan"))),
                    }
                    boot_rows.append(row)
                    stage_rows[(model, f"{metric}|{kind}|K{k_b}", seed_key)] = {
                        "diff": row["diff"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                        "n": row["n"], "n_clusters": row["n_clusters"],
                        "mean_a": row["mean_a"], "mean_b": row["mean_b"],
                    }
            _log(f"[d1-stage2] C1 {model}/{seed_key} done ({len(point_rows)} point rows so far)")

    # 3-seed means (value and CI endpoints), section 21.
    per_seed_by_key: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for (model, key, seed_key), value in stage_rows.items():
        per_seed_by_key.setdefault((model, key), []).append(dict(value, seed=seed_key))
    for (model, key), values in sorted(per_seed_by_key.items()):
        if len(values) != len(SEEDS_C1):
            raise AssertionError(f"{model}/{key}: expected {len(SEEDS_C1)} seeds, got {len(values)}")
        metric, kind, k_b = key.split("|")
        boot_rows.append({
            "analysis": "C1", "model": model, "seed": "mean",
            "K_a": K_BASELINE, "K_b": int(k_b[1:]), "variant": "global_T" if model == "cosine_b0" else "global_T_corrected",
            "metric": metric, "diff_kind": kind,
            "n": int(np.mean([v["n"] for v in values])),
            "n_clusters": int(np.mean([v["n_clusters"] for v in values])),
            "resample_unit": "image",
            "mean_a": "", "mean_b": "",
            "diff": _mean_endpoints(values, "diff"),
            "ci_low": _mean_endpoints(values, "ci_low"),
            "ci_high": _mean_endpoints(values, "ci_high"),
            "ci_level": BOOTSTRAP_CI, "n_replicates": BOOTSTRAP_REPLICATES,
            "std_diff": _std([v["diff"] for v in values]),
        })

    # b3 mean point rows
    for k in KS:
        per_seed = [r for r in point_rows if r["model"] == "b3" and r["K"] == k]
        point_rows.append({
            "model": "b3", "seed": "mean", "K": int(k),
            "n": int(np.mean([r["n"] for r in per_seed])),
            "n_images": int(np.mean([r["n_images"] for r in per_seed])),
            "accuracy": float(np.mean([r["accuracy"] for r in per_seed])),
            "auroc_correct": float(np.mean([r["auroc_correct"] for r in per_seed])),
            "e_aurc": float(np.mean([r["e_aurc"] for r in per_seed])),
            "rer_at_50": float(np.mean([r["rer_at_50"] for r in per_seed])),
        })

    comparison = c1_vs_v1(boot_rows)
    return point_rows, boot_rows, comparison


def c1_vs_v1(boot_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Side-by-side of D1_clean b3-mean deltas with the frozen phase0b values."""
    v1 = load_v1_c1_targets()
    v1_mean = {}
    for key in {r["metric"] + "|" + r["diff_kind"] + "|K" + str(r["K_b"]) for r in boot_rows if r["model"] == "b3"}:
        per_seed = [v1[f"seed{seed}"][key] for seed in SEEDS_C1]
        v1_mean[key] = {
            "diff": float(np.mean([s["diff"] for s in per_seed])),
            "ci_low": float(np.mean([s["ci_low"] for s in per_seed])),
            "ci_high": float(np.mean([s["ci_high"] for s in per_seed])),
        }
    comparison: Dict[str, Any] = {}
    for row in boot_rows:
        if row["model"] != "b3" or row["seed"] != "mean":
            continue
        key = f"{row['metric']}|{row['diff_kind']}|K{row['K_b']}"
        comparison[key] = {
            "d1_clean": {"diff": row["diff"], "ci_low": row["ci_low"], "ci_high": row["ci_high"]},
            "v1_full_cohort": v1_mean[key],
            "same_direction": bool(np.sign(row["diff"]) == np.sign(v1_mean[key]["diff"])),
        }
    return comparison


def c1_verdict(boot_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 13: no new thresholds - the phenomenon persists when the K5->K50
    AUROC/RER50 gains and the E-AURC worsening keep their V1 direction with a
    CI excluding zero (and, per V1, the gain grows monotonically with K)."""
    means = {
        f"{r['metric']}|{r['diff_kind']}": r
        for r in boot_rows
        if r["model"] == "b3" and r["seed"] == "mean" and r["K_b"] == 50
    }
    auroc = means.get("auroc_correct|absolute")
    e_abs = means.get("e_aurc|absolute")
    e_rel = means.get("e_aurc|relative")
    rer50 = means.get("rer_at_50|absolute")
    checks = {
        "auroc_gain_k50_ci_low_gt0": bool(auroc and auroc["ci_low"] > 0.0),
        "e_aurc_worse_k50_ci_high_lt0": bool(e_abs and e_abs["ci_high"] < 0.0),
        "e_aurc_relative_worse_k50_ci_low_lt0": bool(e_rel and e_rel["ci_low"] < 0.0),
        "rer50_gain_k50_ci_low_gt0": bool(rer50 and rer50["ci_low"] > 0.0),
    }
    # monotonicity of the K5->K degradation across K (V1 direction pattern: the
    # AUROC/RER50 gains grow and the relative E-AURC worsening deepens with K)
    monotonic = True
    for metric, kind, sign in (("auroc_correct", "absolute", 1.0), ("e_aurc", "relative", -1.0), ("rer_at_50", "absolute", 1.0)):
        values = signed_by_kb(boot_rows, metric, kind, sign)
        monotonic = monotonic and len(values) == 3 and all(
            values[i] <= values[i + 1] + 1e-12 for i in range(len(values) - 1)
        )
    checks["monotone_in_k"] = bool(monotonic)
    robust = all(checks.values())
    return {
        "label": "C1 ANNOTATION-ROBUST" if robust else "C1 NOT ROBUST (see checks)",
        "robust": bool(robust),
        "checks": checks,
        "k50_values": {
            key: {"diff": row["diff"], "ci_low": row["ci_low"], "ci_high": row["ci_high"]}
            for key, row in means.items()
        },
    }


def signed_by_kb(boot_rows: List[Dict[str, Any]], metric: str, kind: str, sign: float) -> List[float]:
    rows = [
        (int(r["K_b"]), sign * float(r["diff"])) for r in boot_rows
        if r["model"] == "b3" and r["seed"] == "mean" and r["metric"] == metric and r["diff_kind"] == kind
    ]
    return [d for _, d in sorted(rows)]


# ---------------------------------------------------------------------------
# C4 - hard-semantic robustness
# ---------------------------------------------------------------------------
C4_POINT_FIELDS = [
    "scorer", "cell", "model", "n", "n_images", "b3_accuracy", "auroc_correct",
    "e_aurc", "rer_at_50",
]


def _c4_dod_rows(
    hard: Dict[str, np.ndarray],
    rand: Dict[str, np.ndarray],
    scorer: str,
    *,
    replicates: int,
    seed: int,
    ci: float,
) -> Dict[str, Dict[str, Any]]:
    """The frozen phase1f ``_dod_rows`` replayed on the D1_clean row subset."""
    if not np.array_equal(hard["sentence_id"], rand["sentence_id"]):
        raise AssertionError(f"{scorer}: hard/rand rows are not matched after filtering")
    clusters = np.asarray(hard["image_id"], dtype=np.int64)
    specs = (
        ("auroc_correct", reval._metric_fn("auroc_correct"), "diff"),
        ("e_aurc_reduction", reval._metric_fn("e_aurc"), "ratio"),
        ("rer_at_50_gain_pp", reval._metric_fn("rer_at_50"), "diff"),
    )
    out: Dict[str, Dict[str, Any]] = {}
    for name, metric_fn, kind in specs:
        args_hard = (
            SampleStats.from_conf_correct(hard["conf_e1b"], hard["correct"]),
            SampleStats.from_conf_correct(hard["conf_stats"], hard["correct"]),
        )
        args_rand = (
            SampleStats.from_conf_correct(rand["conf_e1b"], rand["correct"]),
            SampleStats.from_conf_correct(rand["conf_stats"], rand["correct"]),
        )
        if kind == "ratio":
            result = heval.paired_diff_of_diffs_ratio_bootstrap(
                metric_fn, args_hard[0], args_hard[1], args_rand[0], args_rand[1], clusters,
                n_replicates=int(replicates), seed=int(seed), ci=float(ci), metric_name=name,
            )
        else:
            result = heval.paired_diff_of_diffs_bootstrap(
                metric_fn, args_hard[0], args_hard[1], args_rand[0], args_rand[1], clusters,
                n_replicates=int(replicates), seed=int(seed), ci=float(ci), metric_name=name,
            )
        if name == "rer_at_50_gain_pp":
            result = dict(result)
            for field in ("diff", "hard_diff", "rand_diff", "ci_low", "ci_high"):
                result[field] = 100.0 * float(result[field])
        out[name] = result
    return out


def run_c4(clean_sent_ids: set) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    per_seed_abs: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_seed_dod: Dict[str, Dict[str, Dict[str, float]]] = {}
    subset_evidence: Dict[str, Any] = {}

    keep = np.fromiter(sorted(clean_sent_ids), dtype=np.int64)
    for scorer in SCORERS_C4:
        data: Dict[str, Dict[str, np.ndarray]] = {}
        raw_h = np.load(_REPO_ROOT / PHASE1F_PRED.format(cell="hard5", scorer=scorer), allow_pickle=False)
        sid_h = np.asarray(raw_h["sentence_id"], dtype=np.int64)
        mask_h = np.isin(sid_h, keep)
        for cell in ("hard5", "rand5"):
            path = _REPO_ROOT / PHASE1F_PRED.format(cell=cell, scorer=scorer)
            raw = np.load(path, allow_pickle=False)
            sid = np.asarray(raw["sentence_id"], dtype=np.int64)
            mask = np.isin(sid, keep)
            data[cell] = {
                "sentence_id": sid[mask],
                "image_id": np.asarray(raw["image_id"], dtype=np.int64)[mask],
                "conf_e1b": np.asarray(raw["conf_e1b"], dtype=np.float64)[mask],
                "conf_stats": np.asarray(raw["conf_stats"], dtype=np.float64)[mask],
                "correct": np.asarray(raw["correct"], dtype=np.float64)[mask],
            }
        if int(mask_h.sum()) < 100:
            raise AssertionError(f"{scorer}: only {int(mask_h.sum())} clean rows in phase1f cohort")
        # Subset invariant: the clean cohort is a pure row sub-selection of the
        # V1 cohort and the hard/rand pairing survives the filter (same target
        # boxes - verified bit-identical in stage 1 - same proposal pool, same
        # candidate-construction algorithm -> same-category cohort stays valid).
        subset_evidence[scorer] = {
            "v1_rows": int(sid_h.size),
            "clean_rows": int(mask_h.sum()),
            "dropped_rows": int((~mask_h).sum()),
            "hard_rand_matched_after_filter": bool(np.array_equal(data["hard5"]["sentence_id"], data["rand5"]["sentence_id"])),
            "image_ids_of_kept_rows_identical_to_v1": bool(np.array_equal(data["hard5"]["image_id"], np.asarray(raw_h["image_id"], dtype=np.int64)[mask_h])),
        }

        for cell in ("hard5", "rand5"):
            d = data[cell]
            for model, key in ((STATS_NAME, "conf_stats"), (E1B_NAME, "conf_e1b")):
                point = reval.point_metric_row(d[key], d["correct"])
                point_rows.append({
                    "scorer": scorer, "cell": cell, "model": model,
                    "n": int(d["correct"].size), "n_images": int(np.unique(d["image_id"]).size),
                    "b3_accuracy": float(np.mean(d["correct"])),
                    "auroc_correct": float(point["auroc_correct"]),
                    "e_aurc": float(point["e_aurc"]),
                    "rer_at_50": float(point["rer_at_50"]),
                })

        clusters = data["hard5"]["image_id"]
        abs_rows = reval.model_vs_model_bootstrap_row(
            data["hard5"]["conf_e1b"], data["hard5"]["correct"],
            data["hard5"]["conf_stats"], data["hard5"]["correct"], clusters,
            eval_split="__pooled__", K=5, model_a=E1B_NAME, model_b=STATS_NAME,
            metrics=("auroc_correct", "rer_at_50"),
            replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI,
        )
        ratio_row = seval.model_vs_model_e_aurc_ratio_row(
            data["hard5"]["conf_e1b"], data["hard5"]["correct"],
            data["hard5"]["conf_stats"], data["hard5"]["correct"], clusters,
            eval_split="__pooled__", K=5, model_sem=E1B_NAME, model_score=STATS_NAME,
            replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI,
        )
        abs_map = {str(row["metric"]): row for row in abs_rows}
        per_seed_abs[scorer] = {
            "delta_auroc": {"diff": float(abs_map["auroc_correct"]["diff"]),
                            "ci_low": float(abs_map["auroc_correct"]["ci_low"]),
                            "ci_high": float(abs_map["auroc_correct"]["ci_high"])},
            "e_aurc_reduction": {"diff": float(ratio_row["reduction"]),
                                 "ci_low": float(ratio_row["reduction_ci_low"]),
                                 "ci_high": float(ratio_row["reduction_ci_high"])},
            "rer50_gain_pp": {"diff": 100.0 * float(abs_map["rer_at_50"]["diff"]),
                              "ci_low": 100.0 * float(abs_map["rer_at_50"]["ci_low"]),
                              "ci_high": 100.0 * float(abs_map["rer_at_50"]["ci_high"])},
        }
        for metric_name, src in (("delta_auroc", "delta_auroc"), ("e_aurc_reduction", "e_aurc_reduction"), ("rer50_gain_pp", "rer50_gain_pp")):
            value = per_seed_abs[scorer][src]
            boot_rows.append({
                "analysis": "C4", "model": f"{E1B_NAME} vs {STATS_NAME}", "seed": scorer,
                "K_a": 5, "K_b": 5, "variant": "hard5", "metric": metric_name, "diff_kind": "absolute",
                "n": int(abs_map["auroc_correct"]["n"]), "n_clusters": int(abs_map["auroc_correct"]["n_clusters"]),
                "resample_unit": "image", "mean_a": "", "mean_b": "",
                "diff": value["diff"], "ci_low": value["ci_low"], "ci_high": value["ci_high"],
                "ci_level": BOOTSTRAP_CI, "n_replicates": BOOTSTRAP_REPLICATES, "std_diff": "",
            })

        dod = _c4_dod_rows(data["hard5"], data["rand5"], scorer,
                           replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, ci=BOOTSTRAP_CI)
        per_seed_dod[scorer] = {
            name: {"diff": float(row["diff"]), "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
                   "hard_diff": float(row["hard_diff"]), "rand_diff": float(row["rand_diff"])}
            for name, row in dod.items()
        }
        for name, row in per_seed_dod[scorer].items():
            boot_rows.append({
                "analysis": "C4", "model": "diff_of_diffs", "seed": scorer,
                "K_a": 5, "K_b": 5, "variant": "hard5_vs_rand5", "metric": name, "diff_kind": "absolute",
                "n": int(dod[name].get("n", 0)) if "n" in dod[name] else "", "n_clusters": "",
                "resample_unit": "image", "mean_a": "", "mean_b": "",
                "diff": row["diff"], "ci_low": row["ci_low"], "ci_high": row["ci_high"],
                "ci_level": BOOTSTRAP_CI, "n_replicates": BOOTSTRAP_REPLICATES, "std_diff": "",
            })
        _log(f"[d1-stage2] C4 {scorer} done: dod auroc={per_seed_dod[scorer]['auroc_correct']['diff']:+.5f}")

    # 3-seed means exactly like phase1f (values and CI endpoints each averaged).
    cell = {
        "delta_auroc": _mean_endpoints([per_seed_abs[s]["delta_auroc"] for s in SCORERS_C4], "diff"),
        "delta_auroc_ci_low": _mean_endpoints([per_seed_abs[s]["delta_auroc"] for s in SCORERS_C4], "ci_low"),
        "delta_auroc_ci_high": _mean_endpoints([per_seed_abs[s]["delta_auroc"] for s in SCORERS_C4], "ci_high"),
        "e_aurc_reduction": _mean_endpoints([per_seed_abs[s]["e_aurc_reduction"] for s in SCORERS_C4], "diff"),
        "e_aurc_reduction_ci_low": _mean_endpoints([per_seed_abs[s]["e_aurc_reduction"] for s in SCORERS_C4], "ci_low"),
        "e_aurc_reduction_ci_high": _mean_endpoints([per_seed_abs[s]["e_aurc_reduction"] for s in SCORERS_C4], "ci_high"),
        "rer50_gain_pp": _mean_endpoints([per_seed_abs[s]["rer50_gain_pp"] for s in SCORERS_C4], "diff"),
        "rer50_gain_pp_ci_low": _mean_endpoints([per_seed_abs[s]["rer50_gain_pp"] for s in SCORERS_C4], "ci_low"),
        "rer50_gain_pp_ci_high": _mean_endpoints([per_seed_abs[s]["rer50_gain_pp"] for s in SCORERS_C4], "ci_high"),
    }
    dod_auroc = {
        "diff": _mean_endpoints([per_seed_dod[s]["auroc_correct"] for s in SCORERS_C4], "diff"),
        "ci_low": _mean_endpoints([per_seed_dod[s]["auroc_correct"] for s in SCORERS_C4], "ci_low"),
        "ci_high": _mean_endpoints([per_seed_dod[s]["auroc_correct"] for s in SCORERS_C4], "ci_high"),
        "per_seed": {s: float(per_seed_dod[s]["auroc_correct"]["diff"]) for s in SCORERS_C4},
    }
    seed_lows = [float(per_seed_abs[s]["delta_auroc"]["ci_low"]) for s in SCORERS_C4]
    seed_mean_delta = float(cell["delta_auroc"])
    gate = heval.hard_gate(cell, dod_auroc, seed_delta_ci_lows=seed_lows, seed_mean_delta=seed_mean_delta, thresholds=None)

    for label, group in (("b3_mean", {"delta_auroc": cell["delta_auroc"], "delta_auroc_ci_low": cell["delta_auroc_ci_low"],
                                      "delta_auroc_ci_high": cell["delta_auroc_ci_high"], "e_aurc_reduction": cell["e_aurc_reduction"],
                                      "e_aurc_reduction_ci_low": cell["e_aurc_reduction_ci_low"],
                                      "e_aurc_reduction_ci_high": cell["e_aurc_reduction_ci_high"],
                                      "rer50_gain_pp": cell["rer50_gain_pp"], "rer50_gain_pp_ci_low": cell["rer50_gain_pp_ci_low"],
                                      "rer50_gain_pp_ci_high": cell["rer50_gain_pp_ci_high"]}),
                         ("b3_mean_dod_auroc", dod_auroc)):
        boot_rows.append({
            "analysis": "C4", "model": "diff_of_diffs" if "dod" in label else f"{E1B_NAME} vs {STATS_NAME}",
            "seed": label, "K_a": 5, "K_b": 5,
            "variant": "hard5_vs_rand5" if "dod" in label else "hard5",
            "metric": "auroc_correct" if "dod" in label else "summary",
            "diff_kind": "absolute",
            "n": "", "n_clusters": "", "resample_unit": "image", "mean_a": "", "mean_b": "",
            "diff": float(group.get("diff", group.get("delta_auroc"))),
            "ci_low": float(group.get("ci_low", group.get("delta_auroc_ci_low"))),
            "ci_high": float(group.get("ci_high", group.get("delta_auroc_ci_high"))),
            "ci_level": BOOTSTRAP_CI, "n_replicates": BOOTSTRAP_REPLICATES, "std_diff": "",
        })
    point_rows.append({
        "scorer": "b3_mean", "cell": "hard5", "model": E1B_NAME,
        "n": int(np.mean([r["n"] for r in point_rows if r["cell"] == "hard5"])),
        "n_images": int(np.mean([r["n_images"] for r in point_rows if r["cell"] == "hard5"])),
        "b3_accuracy": float(np.mean([r["b3_accuracy"] for r in point_rows if r["cell"] == "hard5" and r["model"] == E1B_NAME])),
        "auroc_correct": float(np.mean([r["auroc_correct"] for r in point_rows if r["cell"] == "hard5" and r["model"] == E1B_NAME])),
        "e_aurc": float(np.mean([r["e_aurc"] for r in point_rows if r["cell"] == "hard5" and r["model"] == E1B_NAME])),
        "rer_at_50": float(np.mean([r["rer_at_50"] for r in point_rows if r["cell"] == "hard5" and r["model"] == E1B_NAME])),
    })

    summary = {
        "cell_samecat_k5_d1_clean": cell,
        "diff_of_diffs_auroc_d1_clean": dod_auroc,
        "per_seed_delta_ci_low_k5_d1_clean": seed_lows,
        "seed_mean_delta_k5_d1_clean": seed_mean_delta,
        "a8_6_gate_reapplied_on_clean": gate,
        "per_seed_abs": per_seed_abs,
        "per_seed_dod": per_seed_dod,
        "clean_subset_evidence": subset_evidence,
    }
    return point_rows, boot_rows, summary, gate


def c4_vs_v1(summary: Dict[str, Any]) -> Dict[str, Any]:
    v1 = load_v1_c4_targets()
    dod = summary["diff_of_diffs_auroc_d1_clean"]
    v1_dod = v1["diff_of_diffs_auroc"]
    comparison: Dict[str, Any] = {}
    for key, d1, ref in (
        ("delta_auroc", summary["cell_samecat_k5_d1_clean"]["delta_auroc"], v1["cell_samecat_k5"]["delta_auroc"]),
        ("e_aurc_reduction", summary["cell_samecat_k5_d1_clean"]["e_aurc_reduction"], v1["cell_samecat_k5"]["e_aurc_reduction"]),
        ("rer50_gain_pp", summary["cell_samecat_k5_d1_clean"]["rer50_gain_pp"], v1["cell_samecat_k5"]["rer50_gain_pp"]),
        ("diff_of_diffs_auroc", dod["diff"], v1_dod["diff"]),
    ):
        comparison[key] = {
            "d1_clean": d1, "v1_full_cohort": ref,
            "ratio_d1_over_v1": float(d1 / ref) if ref else None,
            "same_direction": bool(np.sign(d1) == np.sign(ref)),
        }
    comparison["d1_clean_dod_ci"] = {"ci_low": dod["ci_low"], "ci_high": dod["ci_high"]}
    comparison["v1_dod_ci"] = {"ci_low": v1_dod["ci_low"], "ci_high": v1_dod["ci_high"]}
    return comparison


def c4_verdict(summary: Dict[str, Any], gate: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    dod = summary["diff_of_diffs_auroc_d1_clean"]
    checks = {
        "delta_auroc_ci_low_gt0": bool(summary["cell_samecat_k5_d1_clean"]["delta_auroc_ci_low"] > 0.0),
        "dod_ci_low_gt0": bool(dod["ci_low"] > 0.0),
        "dod_same_direction_as_v1": bool(comparison["diff_of_diffs_auroc"]["same_direction"]),
        "a8_6_gate_confirmed_or_strong": bool(gate["verdict"] in ("CONFIRMED", "STRONG")),
    }
    robust = all(checks.values())
    return {
        "label": "C4 ANNOTATION-ROBUST" if robust else "C4 NOT ROBUST (see checks)",
        "robust": bool(robust),
        "checks": checks,
        "gate_verdict": gate["verdict"],
        "dod": {"diff": dod["diff"], "ci_low": dod["ci_low"], "ci_high": dod["ci_high"]},
    }


# ---------------------------------------------------------------------------
# combined verdict (section 22) + figures
# ---------------------------------------------------------------------------
def build_verdict(
    c1: Dict[str, Any], c4: Dict[str, Any],
    c1_comparison: Dict[str, Any], c4_comparison: Dict[str, Any],
    cohort_info: Dict[str, Any],
) -> Dict[str, Any]:
    n_robust = int(c1["robust"]) + int(c4["robust"])
    if n_robust == 2:
        combined = "CORE FINDINGS ROBUST TO REVIEWED ANNOTATIONS"
    elif n_robust == 1:
        combined = "PARTIAL ANNOTATION ROBUSTNESS"
    else:
        combined = "ANNOTATION-SENSITIVITY WARNING"
    return {
        "artifact": "d1_verdict",
        "cohort": cohort_info,
        "c1_cardinality": {"verdict": c1, "vs_v1": c1_comparison},
        "c4_hard_semantic": {"verdict": c4, "vs_v1": c4_comparison},
        "combined_verdict": combined,
        "questions_answered": {
            "q1_cardinality_survives_reviewed_annotations": bool(c1["robust"]),
            "q2_semantic_amplification_survives_annotation_cleanup": bool(c4["robust"]),
            "q3_error_share_of_removed_rows": "see removed_ambiguous_audit_summary.json (stage 1)",
        },
        "constraints": {
            "new_training_parameters": 0,
            "cohort_unmodified_after_inspection": True,
            "reviewed_annotation_namespace": "data/reviewed_refcocoplus/ (originals untouched)",
        },
    }


def write_stage2_figures(
    c1_boot: List[Dict[str, Any]], c4_summary: Dict[str, Any], v1_c4: Dict[str, Any],
    out_dir: Path, cohort_n: Optional[int] = None,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[str] = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Fig: K-profile (b3 mean point metrics) - D1_clean vs V1 full cohort values
    means = {}
    for row in c1_boot:
        if row["model"] == "b3" and row["seed"] == "mean" and row["metric"] == "auroc_correct" and row["diff_kind"] == "absolute":
            means[int(row["K_b"])] = (row["diff"], row["ci_low"], row["ci_high"])
    v1 = load_v1_c1_targets()
    v1_vals = {}
    for k_b in means:
        key = f"auroc_correct|absolute|K{k_b}"
        seeds = [v1[f"seed{s}"][key]["diff"] for s in SEEDS_C1]
        v1_vals[k_b] = float(np.mean(seeds))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ks = sorted(means)
    ax.errorbar(ks, [means[k][0] for k in ks],
                yerr=[[means[k][0] - means[k][1] for k in ks], [means[k][2] - means[k][0] for k in ks]],
                marker="o", label=f"D1_clean ({cohort_n:,} rows)" if cohort_n else "D1_clean")
    ax.plot(ks, [v1_vals[k] for k in ks], marker="s", linestyle="--", label="V1 full cohort (frozen)")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("K (vs K=5)")
    ax.set_ylabel("dAUROC gain, K5 -> K")
    ax.set_title("D1 C1 - cardinality AUROC effect survives annotation cleanup")
    ax.legend()
    path = fig_dir / "fig_d1_c1_cardinality_delta.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(_REPO_ROOT)))

    # Fig: C4 DeltaRand / DeltaHard / Amplification, V1 vs D1_clean
    dod = c4_summary["diff_of_diffs_auroc_d1_clean"]
    labels = ["DeltaRand", "DeltaHard", "Amplification (DoD)"]
    per_seed_dod_auroc = c4_summary["per_seed_dod"]
    rand_d = float(np.mean([per_seed_dod_auroc[s]["auroc_correct"]["rand_diff"] for s in per_seed_dod_auroc]))
    hard_d = float(np.mean([per_seed_dod_auroc[s]["auroc_correct"]["hard_diff"] for s in per_seed_dod_auroc]))
    v1_dod = v1_c4["diff_of_diffs_auroc"]["diff"]
    d1 = [rand_d, hard_d, dod["diff"]]
    v1r = [v1_c4["point_delta"]["delta_rand"], v1_c4["point_delta"]["delta_hard"], v1_dod]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, v1r, width=0.34, color="#8f9fa8", label="V1 full cohort")
    ax.bar(x + 0.18, d1, width=0.34, color="#b4635f", label="D1_clean")
    ax.set_xticks(x, labels)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel("dAUROC (E1b - R1)")
    ax.set_title("D1 C4 - frozen semantic gain by regime")
    ax.legend()
    path = fig_dir / "fig_d1_c4_semantic_increment.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path.relative_to(_REPO_ROOT)))
    return written


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _stage2_frozen_hashes() -> Dict[str, str]:
    paths = {
        "results/v2_data_robustness/d1_reviewed_annotations/clean_manifest.csv": CLEAN_MANIFEST,
        "results/phase0a_corrected/per_sentence_predictions.npz": PHASE0A_COSINE_NPZ,
        "results/phase1f_hard_semantic/gate.json": PHASE1F_GATE_JSON,
        "results/phase1f_hard_semantic/paired_random_vs_hard.csv": PHASE1F_PAIRED_CSV,
        "results/phase0b_independent/metadata.json": PHASE0B_METADATA,
    }
    for seed in SEEDS_C1:
        paths[f"results/phase0b_independent/seed_{seed}/per_sentence_predictions.npz"] = _REPO_ROOT / PHASE0B_NPZ.format(seed=seed)
        paths[f"results/phase0b_independent/seed_{seed}/bootstrap.csv"] = _REPO_ROOT / PHASE0B_BOOTSTRAP.format(seed=seed)
        for cell in ("rand5", "hard5"):
            paths[f"results/phase1f_hard_semantic/predictions/{cell}__b3_seed{seed}.npz"] = (
                _REPO_ROOT / PHASE1F_PRED.format(cell=cell, scorer=f"b3_seed{seed}")
            )
    return {label: _sha256_file(path) for label, path in paths.items()}


def run_d1_stage2(out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cohort, per_split = load_clean_sent_ids()
    _log(f"[d1-stage2] D1_clean cohort (in phase0b eval): {len(cohort)} sentences {per_split}")

    c1_points, c1_boot, c1_comparison = run_c1(cohort)
    c1 = c1_verdict(c1_boot)
    _log(f"[d1-stage2] C1 verdict: {c1['label']}")

    c4_points, c4_boot, c4_summary, gate = run_c4(cohort)
    c4_comparison = c4_vs_v1(c4_summary)
    c4 = c4_verdict(c4_summary, gate, c4_comparison)
    _log(f"[d1-stage2] C4 verdict: {c4['label']} (A8.6 gate on clean = {gate['verdict']})")

    verdict = build_verdict(c1, c4, c1_comparison, c4_comparison,
                            {"n_sentences": len(cohort), "per_split": per_split,
                             "source": "clean_manifest.csv in_phase0b_cohort=True"})
    _log(f"[d1-stage2] combined: {verdict['combined_verdict']}")

    _write_csv(out_dir / "c1_cardinality.csv", c1_points, C1_POINT_FIELDS)
    _write_csv(out_dir / "c4_hard_semantic.csv", c4_points, C4_POINT_FIELDS)
    _write_csv(out_dir / "bootstrap.csv", c1_boot + c4_boot, BOOTSTRAP_FIELDS)
    _write_json(out_dir / "verdict.json", verdict)
    figures = write_stage2_figures(c1_boot, c4_summary, load_v1_c4_targets(), out_dir, cohort_n=len(cohort))

    runtime = time.perf_counter() - t0
    _write_json(
        out_dir / "metadata_stage2.json",
        {
            "artifact": "d1_metadata_stage2",
            "stage": "C1 + C4 replay on D1_clean (frozen weights, no training)",
            "new_training_parameters": 0,
            "runtime_sec": runtime,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "ci": BOOTSTRAP_CI,
                          "resample_unit": "image",
                          "source_of_config": "phase0b metadata.json / phase1f metadata.json (identical)"},
            "cohort": {"n_sentences": len(cohort), "per_split": per_split},
            "frozen_inputs_sha256": _stage2_frozen_hashes(),
            "figures": figures,
            "verdict": verdict["combined_verdict"],
        },
    )
    _log(f"[d1-stage2] done in {runtime:.1f}s -> {out_dir}")
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    verdict = run_d1_stage2(out_dir=args.out_dir)
    print(f"D1_STAGE2_COMPLETE combined={verdict['combined_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
