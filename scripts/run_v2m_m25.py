#!/usr/bin/env python
"""V2-M M2.5 read-only specialist audit: random-only vs curriculum-trained E1b.

Diagnostic only.  No training, no fitting, no hyper-parameter search, no
architecture change, no new candidate sampling.  Both experts are *frozen*
31-d Stats+Semantic logistic calibrators:

* ``E1b_random``     -- M1 ``m1_random_only/seed_s/model_manifest.json``
  (random K5/K10 training);
* ``E1b_curriculum`` -- M2 ``m2_curriculum/seed_s/model_manifest_seed*.json``
  (m=0/2/4 balanced curriculum).

Both are evaluated on the identical frozen K=10 severity cells
(m in {0, 2, 4, 8}, the Phase 1F ``expb`` arrays) with their own frozen
standardisation (M1 train-only vs M2 curriculum-train-only, both rebuilt
read-only) and frozen coefficients.  Reproduction checks: the random expert
must reproduce the frozen M1 ``Random-K5__E1b`` confidences, the curriculum
expert the frozen M2 ``m{m}__E1b`` confidences, and the curriculum point
metrics the committed M2 ``point_metrics.csv`` rows (machine precision).

Reported per seed x severity: AUROC / E-AURC / RER@50 / RER@80; calibration
(mean predicted correctness / accuracy / adaptive ECE / Brier); expert
disagreement ``|p_rand - p_curr|`` (mean / median / p90); the cohort-level
oracle envelope ``max(AUROC_rand, AUROC_curr)``; and the paired image-cluster
bootstrap of ``AUROC(E1b_curriculum) - AUROC(E1b_random)`` (5000 reps, seed 0,
shared draws).

Pattern verdict (diagnostic only -- no success gate):
``SPECIALIST TRADEOFF PRESENT`` / ``CURRICULUM DOMINATES`` /
``CURRICULUM NOT USEFUL`` / ``INCONCLUSIVE``.

Usage
-----
    python -u scripts/run_v2m_m25.py --log-file logs_v2m_m25.txt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402

__all__ = ["build_parser", "main"]

OUT_ROOT = Path("results/v2_local_competition")
M25_DIR = OUT_ROOT / "m25_specialist_audit"
M1_DIR = OUT_ROOT / "m1_random_only"
M2_DIR = OUT_ROOT / "m2_curriculum"

LEVELS: Tuple[int, ...] = (0, 2, 4, 8)
MODELS: Tuple[str, ...] = ("E1b_random", "E1b_curriculum")
MODEL_TAG: Mapping[str, str] = {
    "E1b_random": "E1b random-only (M1)",
    "E1b_curriculum": "E1b curriculum (M2)",
}
REPRO_TOL: float = 1e-9
DISAGREE_QUANTILE: float = 90.0
COLOURS = {"E1b_random": "#1f77b4", "E1b_curriculum": "#d62728"}

POINT_FIELDS = (
    "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80", "ece_adaptive", "brier_binary",
)


# ---------------------------------------------------------------------------
# small helpers (house style, mirrors run_v2m_m2.py)
# ---------------------------------------------------------------------------
def _make_logger(log_file: Optional[Path]) -> Callable[[str], None]:
    stream = open(log_file, "a", encoding="utf-8") if log_file else None

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    return log


def _abs(path: Any) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO / p


def _load_module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


# ---------------------------------------------------------------------------
# aggregation (V2 house convention: seed-mean delta / seed-mean CI bounds)
# ---------------------------------------------------------------------------
def _aggregate(
    boot_rows: Sequence[Mapping[str, Any]], level: str, metric: str
) -> Optional[Dict[str, Any]]:
    sel = [r for r in boot_rows if r["level"] == level and r["metric"] == metric]
    if not sel:
        return None
    sel.sort(key=lambda r: int(r["seed"]))
    diffs = [float(r["diff"]) for r in sel]
    return {
        "n_seeds": len(sel),
        "seeds": [int(r["seed"]) for r in sel],
        "delta": float(np.mean(diffs)),
        "ci_low": float(np.mean([float(r["ci_low"]) for r in sel])),
        "ci_high": float(np.mean([float(r["ci_high"]) for r in sel])),
        "per_seed": {int(r["seed"]): float(r["diff"]) for r in sel},
        "per_seed_ci": {
            int(r["seed"]): [float(r["ci_low"]), float(r["ci_high"])] for r in sel
        },
        "n_positive": int(sum(d > 0.0 for d in diffs)),
        "n_negative": int(sum(d < 0.0 for d in diffs)),
    }


def _pattern(deltas: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Frozen diagnostic classification of the specialist trade-off (no gate)."""
    low_sig = any(float(deltas[f"m{m}"]["ci_high"]) < 0.0 for m in (0, 2))
    high_sig = float(deltas["m8"]["ci_low"]) > 0.0
    all_curr = all(float(deltas[f"m{m}"]["ci_low"]) > 0.0 for m in LEVELS)
    all_rand = all(float(deltas[f"m{m}"]["ci_high"]) < 0.0 for m in LEVELS)
    if low_sig and high_sig:
        pattern = "SPECIALIST TRADEOFF PRESENT"
    elif all_curr:
        pattern = "CURRICULUM DOMINATES"
    elif all_rand:
        pattern = "CURRICULUM NOT USEFUL"
    else:
        pattern = "INCONCLUSIVE"
    return {
        "pattern": pattern,
        "conditions": {
            "low_m_random_better_significant (m0 or m2 CI entirely < 0)": bool(low_sig),
            "m8_curriculum_better_significant (CI entirely > 0)": bool(high_sig),
            "all_levels_curriculum_better (CI > 0 everywhere)": bool(all_curr),
            "all_levels_random_better (CI < 0 everywhere)": bool(all_rand),
        },
    }


# ---------------------------------------------------------------------------
# per-seed audit
# ---------------------------------------------------------------------------
def _run_seed(
    seed: int,
    *,
    runner: Any,
    m1_module: Any,
    corpus: B3Corpus,
    val_cohort: Any,
    cohort_hc: Any,
    rows_same8: np.ndarray,
    split: Any,
    scorers: Mapping[str, Any],
    stored_m2_e1b: Mapping[Tuple[str, int], Mapping[str, str]],
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    t_seed = time.perf_counter()
    temperature = float(m1_module._corrected_T(int(seed)))
    log(f"[M2.5 seed{seed}] start (T_corrected={temperature:.6f})")

    # -- 1. M1 side: rebuild the train-only standardisation (read-only) --------
    bundles = {
        k: m1_module._load_store_bundle(int(seed), k, temperature, log) for k in (5, 10)
    }
    train_masks = {
        k: split.row_mask(
            bundles[k]["eval_split"], bundles[k]["image_id"], kind="reliability_train"
        )
        for k in (5, 10)
    }
    stats_std, stats_fit_m1 = m1_module._standardise(
        {k: bundles[k]["stats17"] for k in (5, 10)}, train_masks, m1_module.STATS17_NAMES
    )
    sem_std, sem_fit_m1 = m1_module._standardise(
        {k: bundles[k]["sem14"] for k in (5, 10)}, train_masks, m1_module.SEM14_NAMES
    )
    m1_manifest = _read_json(M1_DIR / f"seed_{int(seed)}" / "model_manifest.json")
    coef_rand = np.asarray(m1_manifest["E1b"]["coef"], dtype=np.float64)
    intercept_rand = float(m1_manifest["E1b"]["intercept"])

    # random expert must reproduce the frozen M1 confidences (read-only check)
    mask_test5 = np.isin(np.asarray(bundles[5]["eval_split"]), ("testA", "testB"))
    x31_std5 = np.hstack([stats_std[5], sem_std[5]])
    p_rand_k5 = expit(x31_std5[mask_test5] @ coef_rand + intercept_rand)
    with np.load(M1_DIR / f"seed_{int(seed)}" / "confidences.npz") as z:
        ref_rand = np.asarray(z["Random-K5__E1b"], dtype=np.float64)
    repro_rand = (
        float(np.max(np.abs(p_rand_k5 - ref_rand)))
        if p_rand_k5.shape == ref_rand.shape
        else float("inf")
    )
    if not (repro_rand <= REPRO_TOL):
        raise AssertionError(
            f"seed{seed}: random E1b does not reproduce the frozen M1 confidences ({repro_rand:.3e})"
        )
    log(f"[M2.5 seed{seed}] random E1b reproduces M1 confidences (max|delta|={repro_rand:.2e})")

    # -- 2. M2 side: curriculum standardisation + curriculum coefficients -----
    r1 = m1_manifest["R1"]
    b_of = runner._b_function(
        stats_fit_m1, np.asarray(r1["coef"], dtype=np.float64), float(r1["intercept"])
    )
    text_cache: Dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    val_blocks = runner._val_level_blocks(
        corpus, val_cohort, scorers, int(seed), temperature, text_cache, b_of, log
    )
    train_stats = np.vstack(
        [np.asarray(val_blocks["train"][m]["stats17"]) for m in runner.LEVELS_TRAIN]
    )
    train_sem = np.vstack(
        [np.asarray(val_blocks["train"][m]["sem14"]) for m in runner.LEVELS_TRAIN]
    )
    stats_fit_m2 = rfeat.normalize_fit(
        train_stats, fit_rows=np.arange(train_stats.shape[0]), keys=runner.STATS17_NAMES
    )
    sem_fit_m2 = rfeat.normalize_fit(
        train_sem, fit_rows=np.arange(train_sem.shape[0]), keys=runner.SEM14_NAMES
    )
    log(f"[M2.5 seed{seed}] curriculum train cells rebuilt ({time.perf_counter() - t0:.1f}s)")

    m2_manifest = _read_json(
        M2_DIR / f"seed_{int(seed)}" / f"model_manifest_seed{int(seed)}.json"
    )
    e1b_curr = m2_manifest["models"]["E1b"]
    coef_curr = np.asarray(e1b_curr["coef"], dtype=np.float64)
    intercept_curr = float(e1b_curr["intercept"])

    # -- 3. frozen test cells (m in {0,2,4,8}; STOP-verified inside) -----------
    t0 = time.perf_counter()
    test_blocks = runner._test_level_blocks(
        corpus, cohort_hc, rows_same8, scorers, int(seed), temperature, text_cache, b_of, log
    )
    log(f"[M2.5 seed{seed}] frozen test cells loaded ({time.perf_counter() - t0:.1f}s)")

    with np.load(M2_DIR / f"seed_{int(seed)}" / "confidences.npz") as z:
        ref_curr = {
            f"m{m}": np.asarray(z[f"m{m}__E1b"], dtype=np.float64) for m in LEVELS
        }

    # -- 4. evaluation per severity level --------------------------------------
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    disagree_rows: List[Dict[str, Any]] = []
    calib_rows: List[Dict[str, Any]] = []
    envelope_rows: List[Dict[str, Any]] = []
    repro_curr = 0.0
    repro_m2_points = 0.0
    for m in LEVELS:
        level = f"m{m}"
        block = test_blocks[int(m)]
        correct = np.asarray(block["correct"], dtype=bool)
        clusters = np.asarray(block["image_id"], dtype=np.int64)
        p_rand = expit(
            np.hstack(
                [
                    rfeat.normalize_apply(np.asarray(block["stats17"]), stats_fit_m1),
                    rfeat.normalize_apply(np.asarray(block["sem14"]), sem_fit_m1),
                ]
            )
            @ coef_rand
            + intercept_rand
        )
        p_curr = expit(
            np.hstack(
                [
                    rfeat.normalize_apply(np.asarray(block["stats17"]), stats_fit_m2),
                    rfeat.normalize_apply(np.asarray(block["sem14"]), sem_fit_m2),
                ]
            )
            @ coef_curr
            + intercept_curr
        )
        delta_curr = float(np.max(np.abs(p_curr - ref_curr[level])))
        repro_curr = max(repro_curr, delta_curr)
        if not (delta_curr <= REPRO_TOL):
            raise AssertionError(
                f"seed{seed}/{level}: curriculum E1b does not reproduce the frozen M2 "
                f"confidences ({delta_curr:.3e})"
            )

        per_model: Dict[str, Dict[str, Any]] = {}
        for model, p in (("E1b_random", p_rand), ("E1b_curriculum", p_curr)):
            row = reval.point_metric_row(p, correct, probability=p)
            row.update(
                {
                    "level": level, "K": int(runner.K_LEVEL), "seed": int(seed), "model": model,
                    "n": int(correct.size), "n_images": int(np.unique(clusters).size),
                    "b3_accuracy": float(correct.mean()),
                }
            )
            point_rows.append(row)
            per_model[model] = row
            calib_rows.append(
                {
                    "level": level, "seed": int(seed), "model": model,
                    "mean_pred": float(np.mean(p)), "accuracy": float(correct.mean()),
                    "ece_adaptive": float(row["ece_adaptive"]),
                    "brier_binary": float(row["brier_binary"]),
                }
            )

        # closed-loop vs the committed M2 point table (curriculum expert)
        stored = stored_m2_e1b[(level, int(seed))]
        for key in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
            repro_m2_points = max(
                repro_m2_points,
                abs(float(per_model["E1b_curriculum"][key]) - float(stored[key])),
            )

        diff = np.abs(p_rand - p_curr)
        disagree_rows.append(
            {
                "level": level, "seed": int(seed), "n": int(diff.size),
                "mean_abs_diff": float(np.mean(diff)),
                "median_abs_diff": float(np.median(diff)),
                "p90_abs_diff": float(np.percentile(diff, DISAGREE_QUANTILE)),
            }
        )

        rows = reval.model_vs_model_bootstrap_row(
            p_curr, correct, p_rand, correct, clusters,
            eval_split=runner.POOLED, K=int(runner.K_LEVEL),
            model_a="E1b_curriculum", model_b="E1b_random",
            metrics=runner.BOOT_METRICS, replicates=int(args.bootstrap_replicates),
            seed=int(args.bootstrap_seed), ci=float(args.ci),
        )
        for row in rows:
            row.update({"level": level, "seed": int(seed)})
            boot_rows.append(row)

        auroc_rand = float(per_model["E1b_random"]["auroc_correct"])
        auroc_curr = float(per_model["E1b_curriculum"]["auroc_correct"])
        envelope = max(auroc_rand, auroc_curr)
        envelope_rows.append(
            {
                "level": level, "seed": int(seed),
                "auroc_random": auroc_rand, "auroc_curriculum": auroc_curr,
                "envelope": envelope,
                "gap_envelope_vs_random": envelope - auroc_rand,
                "gap_envelope_vs_curriculum": envelope - auroc_curr,
            }
        )
        log(
            f"  [M2.5 seed{seed}] {level}: dAUROC(curr-rand)="
            f"{float(np.mean([r['diff'] for r in rows if r['metric'] == 'auroc_correct'])):+.4f}  "
            f"(repro curr {delta_curr:.2e})"
        )

    if not (repro_m2_points <= REPRO_TOL):
        raise AssertionError(
            f"seed{seed}: curriculum E1b point metrics disagree with the committed M2 table "
            f"({repro_m2_points:.3e})"
        )
    log(
        f"[M2.5 seed{seed}] done in {time.perf_counter() - t_seed:.1f}s "
        f"(repro rand {repro_rand:.2e}, curr {repro_curr:.2e}, m2-points {repro_m2_points:.2e})"
    )
    return {
        "seed": int(seed),
        "temperature": temperature,
        "repro_random_conf": repro_rand,
        "repro_curriculum_conf": repro_curr,
        "repro_m2_points": repro_m2_points,
        "point_rows": point_rows,
        "boot_rows": boot_rows,
        "disagree_rows": disagree_rows,
        "calib_rows": calib_rows,
        "envelope_rows": envelope_rows,
    }


# ---------------------------------------------------------------------------
# figures (section 10: the two key diagnostic plots)
# ---------------------------------------------------------------------------
def _write_figures(
    point_rows: Sequence[Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    log: Callable[[str], None],
) -> List[Path]:
    x = np.arange(len(LEVELS), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    ax1, ax2 = axes

    for model in MODELS:
        ys: List[float] = []
        es: List[float] = []
        for m in LEVELS:
            vals = [
                float(r["auroc_correct"]) for r in point_rows
                if r["model"] == model and r["level"] == f"m{m}"
            ]
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
        ax1.errorbar(
            x, ys, yerr=es, marker="o", capsize=3.0, linewidth=1.7,
            color=COLOURS[model], label=MODEL_TAG[model],
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"m={m}" for m in LEVELS])
    ax1.set_xlabel("competition severity m (same-category distractors), K=10")
    ax1.set_ylabel("AUROC(correct), frozen test split")
    ax1.set_title("E1b experts vs severity (3-seed mean +/- std)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    delta = np.array([float(deltas[f"m{m}"]["delta"]) for m in LEVELS])
    lo = np.array([float(deltas[f"m{m}"]["ci_low"]) for m in LEVELS])
    hi = np.array([float(deltas[f"m{m}"]["ci_high"]) for m in LEVELS])
    yerr = np.vstack([delta - lo, hi - delta])
    ax2.errorbar(
        x, delta, yerr=yerr, marker="o", capsize=3.5, linewidth=2.0,
        color=COLOURS["E1b_curriculum"], label="curriculum - random (95% paired CI)",
    )
    ax2.axhline(0.0, color="black", linewidth=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"m={m}" for m in LEVELS])
    ax2.set_xlabel("competition severity m (same-category distractors), K=10")
    ax2.set_ylabel("AUROC(E1b_curriculum) - AUROC(E1b_random)")
    ax2.set_title("Specialist shift: curriculum minus random expert")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "m25_specialist_audit.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    log(f"[M2.5] figure written: {path}")
    return [path]


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--bootstrap-replicates", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument(
        "--smoke", action="store_true",
        help="1 seed / 100 reps / results/v2_local_competition/m25_specialist_audit/_smoke",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="override the output directory (used by the consistency audit recompute)",
    )
    p.add_argument("--log-file", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds: Tuple[int, ...] = (
        (1,) if args.smoke else tuple(int(s) for s in str(args.seeds).split(","))
    )
    reps = 100 if args.smoke else int(args.bootstrap_replicates)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ((M25_DIR / "_smoke") if args.smoke else M25_DIR)
    )
    run_args = argparse.Namespace(
        bootstrap_replicates=int(reps), bootstrap_seed=int(args.bootstrap_seed),
        ci=float(args.ci),
    )
    log = _make_logger(Path(args.log_file) if args.log_file else None)
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    runner = _load_module("run_v2m_m2", "scripts/run_v2m_m2.py")
    m1_module = _load_module("run_v2m_m1", "scripts/run_v2m_m1.py")
    build_module = _load_module("build_v2m_m2_manifests", "scripts/build_v2m_m2_manifests.py")
    log(
        "[M2.5] read-only specialist audit start "
        "(no training / no fitting / no hyper-parameter search / no new sampling)"
    )

    # frozen split / cohort / val tables / scorers / corpus (mirror of M2 main)
    split = sdata.load_split_from_manifest(_abs(runner.SPLIT_MANIFEST))
    cohort_hc = shard.load_hard_cohort(
        features_dir=sdata.PHASE05_FEATURES_DIR, manifests_root=_abs(runner.MANIFESTS), log=log
    )
    rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    val_cohort = build_module.load_val_cohort()
    scorers = hscores.load_frozen_scorers(seeds=seeds)
    corpus = B3Corpus(
        _abs(runner.FEATURES_ROOT), _abs(runner.MANIFESTS), _abs(runner.REFS), _abs(runner.BANK),
        image_sizes_path=_abs(runner.IMAGE_SIZES), ks=(5,), regime="random", preload=True,
    )

    stored_m2_e1b = {
        (r["cohort"], int(r["seed"])): r
        for r in _read_csv(M2_DIR / "point_metrics.csv")
        if r["model"] == "E1b"
    }

    per_seed: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            per_seed.append(
                _run_seed(
                    int(seed), runner=runner, m1_module=m1_module, corpus=corpus,
                    val_cohort=val_cohort, cohort_hc=cohort_hc, rows_same8=rows_same8,
                    split=split, scorers=scorers, stored_m2_e1b=stored_m2_e1b,
                    args=run_args, log=log,
                )
            )
    finally:
        corpus.close()

    point_rows = [r for entry in per_seed for r in entry["point_rows"]]
    boot_rows = [r for entry in per_seed for r in entry["boot_rows"]]
    disagree_rows = [r for entry in per_seed for r in entry["disagree_rows"]]
    calib_rows = [r for entry in per_seed for r in entry["calib_rows"]]
    envelope_rows = [r for entry in per_seed for r in entry["envelope_rows"]]

    deltas = {
        f"m{m}": {
            metric: _aggregate(boot_rows, f"m{m}", metric) for metric in runner.BOOT_METRICS
        }
        for m in LEVELS
    }
    auroc_deltas = {f"m{m}": deltas[f"m{m}"]["auroc_correct"] for m in LEVELS}
    verdict = _pattern(auroc_deltas)

    # disagreement trend across severity (seed-mean mean|p_rand - p_curr|)
    disagree_mean = {
        f"m{m}": float(
            np.mean([r["mean_abs_diff"] for r in disagree_rows if r["level"] == f"m{m}"])
        )
        for m in LEVELS
    }
    disagree_med = {
        f"m{m}": float(
            np.mean([r["median_abs_diff"] for r in disagree_rows if r["level"] == f"m{m}"])
        )
        for m in LEVELS
    }
    trend_values = [disagree_mean[f"m{m}"] for m in LEVELS]
    trend_non_decreasing = all(
        trend_values[i + 1] >= trend_values[i] for i in range(len(trend_values) - 1)
    )

    # oracle envelope (cohort-level only; never a sample-label selector)
    envelope = {
        f"m{m}": {
            "auroc_random_mean": float(
                np.mean([r["auroc_random"] for r in envelope_rows if r["level"] == f"m{m}"])
            ),
            "auroc_curriculum_mean": float(
                np.mean([r["auroc_curriculum"] for r in envelope_rows if r["level"] == f"m{m}"])
            ),
            "envelope_mean": float(
                np.mean([r["envelope"] for r in envelope_rows if r["level"] == f"m{m}"])
            ),
            "gap_envelope_vs_random_mean": float(
                np.mean(
                    [r["gap_envelope_vs_random"] for r in envelope_rows if r["level"] == f"m{m}"]
                )
            ),
            "gap_envelope_vs_curriculum_mean": float(
                np.mean(
                    [r["gap_envelope_vs_curriculum"] for r in envelope_rows if r["level"] == f"m{m}"]
                )
            ),
            "per_seed": [
                {key: r[key] for key in ("seed", "auroc_random", "auroc_curriculum", "envelope")}
                for r in envelope_rows if r["level"] == f"m{m}"
            ],
        }
        for m in LEVELS
    }

    max_repro = {
        "random_vs_m1_confidences": max(float(e["repro_random_conf"]) for e in per_seed),
        "curriculum_vs_m2_confidences": max(float(e["repro_curriculum_conf"]) for e in per_seed),
        "curriculum_vs_m2_point_table": max(float(e["repro_m2_points"]) for e in per_seed),
    }

    _write_csv(
        out_dir / "point_metrics.csv", point_rows,
        ("level", "K", "seed", "model", "n", "n_images", "b3_accuracy") + POINT_FIELDS,
    )
    _write_csv(
        out_dir / "paired_bootstrap.csv", boot_rows,
        ("level", "K", "seed", "model_a", "model_b", "metric", "diff", "ci_low", "ci_high",
         "mean_a", "mean_b", "n", "n_clusters", "n_replicates", "ci_level"),
    )
    _write_csv(
        out_dir / "expert_disagreement.csv", disagree_rows,
        ("level", "seed", "n", "mean_abs_diff", "median_abs_diff", "p90_abs_diff"),
    )
    _write_csv(
        out_dir / "calibration.csv", calib_rows,
        ("level", "seed", "model", "mean_pred", "accuracy", "ece_adaptive", "brier_binary"),
    )

    figures = _write_figures(point_rows, auroc_deltas, out_dir, log)

    verdict_payload = {
        "artifact": "v2m_m25_specialist_audit",
        "question": (
            "Determine whether random-only E1b and competition-curriculum E1b specialize to "
            "different competition severities."
        ),
        "constraints": [
            "read-only: no training, no fitting, no hyper-parameter search, no architecture change, no new candidate sampling",
            "both experts are frozen M1/M2 manifests; only inference on the frozen K=10 severity cells",
        ],
        "delta_definition": (
            "delta(m) = AUROC(E1b_curriculum) - AUROC(E1b_random); image-cluster paired "
            "bootstrap (5000 reps, seed 0, shared draws); seed-mean delta, seed-mean CI bounds; "
            "per-seed directions recorded"
        ),
        "severity": {
            f"m{m}": {
                metric: deltas[f"m{m}"][metric] for metric in runner.BOOT_METRICS
            }
            for m in LEVELS
        },
        **verdict,
        "disagreement": {
            "mean_abs_diff_seed_mean": disagree_mean,
            "median_abs_diff_seed_mean": disagree_med,
            "non_decreasing_in_m": bool(trend_non_decreasing),
        },
        "envelope": envelope,
        "repro_checks": max_repro,
        "interpretation": {
            "if_tradeoff": (
                "SPECIALIST TRADEOFF PRESENT -> the next round may discuss a "
                "competition-adaptive mixture of simple reliability experts as a NEW method "
                "amendment (never an LCR v1 patch)."
            ),
            "otherwise": (
                "no clear trade-off -> do not enter a mixture design; keep the single-expert "
                "semantic calibrator."
            ),
            "boundary": (
                "diagnostic only; no success gate, no mixture training here, no B1/B2."
            ),
        },
    }
    _write_json(out_dir / "verdict.json", verdict_payload)

    metadata = {
        "artifact": "v2m_m25_metadata",
        "branch": runner._git(["branch", "--show-current"]),
        "head": runner._git(["rev-parse", "HEAD"]),
        "dirty": bool(runner._git(["status", "--porcelain"])),
        "started_utc": started_utc,
        "runtime_sec": round(time.perf_counter() - started, 1),
        "seeds": list(seeds),
        "levels": list(LEVELS),
        "k_level": int(runner.K_LEVEL),
        "bootstrap": {
            "replicates": int(reps), "seed": int(args.bootstrap_seed),
            "ci": float(args.ci), "metrics": list(runner.BOOT_METRICS),
            "shared_draws": "same cohort -> same draws (paired design)",
        },
        "sources": {
            "m1_manifest_sha": {
                str(s): runner._sha256_file(M1_DIR / f"seed_{int(s)}" / "model_manifest.json")
                for s in seeds
            },
            "m2_manifest_sha": {
                str(s): runner._sha256_file(
                    M2_DIR / f"seed_{int(s)}" / f"model_manifest_seed{int(s)}.json"
                )
                for s in seeds
            },
            "m2_manifest_freeze_sha": runner._sha256_file(
                M2_DIR / "manifests" / "manifest_freeze.json"
            ),
        },
        "repro_checks": max_repro,
        "repro_tolerance": REPRO_TOL,
        "figures": [str(p) for p in figures],
        "read_only": True,
        "no_training": True,
    }
    _write_json(out_dir / "metadata.json", metadata)

    d_summary = " ".join(
        f"m{m}:{float(auroc_deltas[f'm{m}']['delta']):+.4f}" for m in LEVELS
    )
    log("=" * 78)
    log("V2-M M2.5 specialist audit report (frozen K=10 severity cells)")
    log(f"  delta(m) = AUROC(curriculum) - AUROC(random): {d_summary}")
    log(f"  pattern: {verdict['pattern']}")
    for name, value in verdict["conditions"].items():
        log(f"    {name}: {value}")
    log(
        "  disagreement mean|p_rand-p_curr| (seed mean): "
        + " ".join(f"m{m}:{disagree_mean[f'm{m}']:.4f}" for m in LEVELS)
    )
    log(
        "  envelope gaps (seed mean): "
        + " ".join(
            f"m{m}:rand+{envelope[f'm{m}']['gap_envelope_vs_random_mean']:.4f}"
            for m in LEVELS
        )
    )
    log(f"  repro checks: {max_repro}")
    log("=" * 78)
    log(
        "V2M_M25_COMPLETE "
        f"pattern={verdict['pattern'].replace(' ', '_')} "
        f"d0={float(auroc_deltas['m0']['delta']):+.4f} "
        f"d2={float(auroc_deltas['m2']['delta']):+.4f} "
        f"d4={float(auroc_deltas['m4']['delta']):+.4f} "
        f"d8={float(auroc_deltas['m8']['delta']):+.4f} "
        f"runtime={time.perf_counter() - started:.1f}s out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
