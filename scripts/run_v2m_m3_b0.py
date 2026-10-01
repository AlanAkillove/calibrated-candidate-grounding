#!/usr/bin/env python
"""V2-M3 B0 developmental audit: competition-adaptive reliability mixture.

POST-HOC DEVELOPMENTAL (amendment V2-M3 sections 1 / 11): B0 validates the
implementation and the mixer-fitting protocol on the frozen M1/M2 experts of
the B0 backbone (OpenCLIP B/32) and reports a *descriptive* result.  B0 has no
GO/NO-GO; its mixture numbers must never be presented as confirmatory method
evidence (confirmatory cohorts: B1/B2, frozen as Amendment V2-M3.1).

Pipeline per seed (read-only on experts; no expert training, no fitting of
experts, no new candidate sampling):

    frozen M1/M2 artifacts -> p_R / p_C on the frozen K=10 cells
    -> per-backbone CompetitionIndex fitted on reliability_train m={0,2,4}
    -> StaticMix / AdaptiveMix fitted on reliability_tune m={0,2,4} only
    -> frozen test cells m in {0,2,4,8}: mixture inference + metrics
    -> paired image-cluster bootstrap (5000 reps, seed 0, shared draws across
       severity for the macro statistic)

Usage
-----
    python -u scripts/run_v2m_m3_b0.py --log-file logs_v2m_m3_b0.txt
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

from ccg.mixture import (  # noqa: E402
    AdaptiveMix,
    CompetitionIndex,
    EqualMix,
    StaticMix,
    competition_features_from_sem14,
    macro_paired_cluster_bootstrap,
)
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402

__all__ = ["build_parser", "main"]

OUT_ROOT = Path("results/v2_local_competition")
M3_DIR = OUT_ROOT / "m3_mixture"
B0_DIR = M3_DIR / "b0"
M1_DIR = OUT_ROOT / "m1_random_only"
M2_DIR = OUT_ROOT / "m2_curriculum"

LEVELS: Tuple[int, ...] = (0, 2, 4, 8)
MIXER_FIT_LEVELS: Tuple[int, ...] = (0, 2, 4)
MODELS: Tuple[str, ...] = ("E_random", "E_curriculum", "EqualMix", "StaticMix", "AdaptiveMix")
#: Per-severity paired comparisons of the B0 family (descriptive CIs).
PER_LEVEL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("AdaptiveMix", "StaticMix"),
    ("AdaptiveMix", "E_random"),
    ("AdaptiveMix", "E_curriculum"),
)
#: Macro comparisons (AUROC averaged over m in {0,2,4,8}, shared draws).
MACRO_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("AdaptiveMix", "StaticMix"),
    ("AdaptiveMix", "E_random"),
    ("AdaptiveMix", "E_curriculum"),
)
BOOT_REPLICATES: int = 5000
BOOT_SEED: int = 0
BOOT_CI: float = 0.95
REPRO_TOL: float = 1e-9
COLOURS = {
    "E_random": "#1f77b4", "E_curriculum": "#d62728", "EqualMix": "#7f7f7f",
    "StaticMix": "#ff7f0e", "AdaptiveMix": "#2ca02c",
}


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


def _p_logistic(
    stats17: np.ndarray,
    sem14: np.ndarray,
    stats_fit: Any,
    sem_fit: Any,
    coef: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Frozen 31-d Stats+Semantic logistic inference (exact linear logit)."""
    x31 = np.hstack(
        [
            rfeat.normalize_apply(np.asarray(stats17), stats_fit),
            rfeat.normalize_apply(np.asarray(sem14), sem_fit),
        ]
    )
    return expit(x31 @ np.asarray(coef, dtype=np.float64) + float(intercept))


def _aggregate(
    boot_rows: Sequence[Mapping[str, Any]],
    level: str,
    model_a: str,
    model_b: str,
    metric: str,
) -> Optional[Dict[str, Any]]:
    sel = [
        r for r in boot_rows
        if r["level"] == level and r["model_a"] == model_a and r["model_b"] == model_b
        and r["metric"] == metric
    ]
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


# ---------------------------------------------------------------------------
# per-seed B0 run
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
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    t_seed = time.perf_counter()
    temperature = float(m1_module._corrected_T(int(seed)))
    log(f"[M3-B0 seed{seed}] start (T_corrected={temperature:.6f})")

    # -- 1. M1 side: rebuild standardisation + random expert (read-only) ------
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
            f"seed{seed}: E_random does not reproduce the frozen M1 confidences ({repro_rand:.3e})"
        )
    log(f"[M3-B0 seed{seed}] E_random reproduces M1 confidences (max|delta|={repro_rand:.2e})")

    # -- 2. val level cells + curriculum standardisation ----------------------
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
        [np.asarray(val_blocks["train"][m]["stats17"]) for m in MIXER_FIT_LEVELS]
    )
    train_sem = np.vstack(
        [np.asarray(val_blocks["train"][m]["sem14"]) for m in MIXER_FIT_LEVELS]
    )
    stats_fit_m2 = rfeat.normalize_fit(
        train_stats, fit_rows=np.arange(train_stats.shape[0]), keys=runner.STATS17_NAMES
    )
    sem_fit_m2 = rfeat.normalize_fit(
        train_sem, fit_rows=np.arange(train_sem.shape[0]), keys=runner.SEM14_NAMES
    )
    m2_manifest = _read_json(
        M2_DIR / f"seed_{int(seed)}" / f"model_manifest_seed{int(seed)}.json"
    )
    coef_curr = np.asarray(m2_manifest["models"]["E1b"]["coef"], dtype=np.float64)
    intercept_curr = float(m2_manifest["models"]["E1b"]["intercept"])
    log(
        f"[M3-B0 seed{seed}] val cells + curriculum standardisation rebuilt "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    # -- 3. competition index: train-only CDF (amendment section 5) ------------
    train_features = {
        name: np.concatenate(
            [
                competition_features_from_sem14(
                    np.asarray(val_blocks["train"][m]["sem14"]), runner.SEM14_NAMES
                )[name]
                for m in MIXER_FIT_LEVELS
            ]
        )
        for name in ("winner_competitor_max_cos", "winner_top2_cos", "q_margin12")
    }
    index = CompetitionIndex().fit(train_features)
    log(f"[M3-B0 seed{seed}] competition index fitted on train-only CDF: {index.summary()}")

    # -- 4. mixer fitting on reliability_tune m={0,2,4} only -------------------
    groups: Dict[int, Dict[str, np.ndarray]] = {}
    for m in MIXER_FIT_LEVELS:
        block = val_blocks["tune"][m]
        sem14 = np.asarray(block["sem14"])
        a_tune = index.transform(competition_features_from_sem14(sem14, runner.SEM14_NAMES))
        groups[int(m)] = {
            "p_r": _p_logistic(
                block["stats17"], sem14, stats_fit_m1, sem_fit_m1, coef_rand, intercept_rand
            ),
            "p_c": _p_logistic(
                block["stats17"], sem14, stats_fit_m2, sem_fit_m2, coef_curr, intercept_curr
            ),
            "a": a_tune,
            "y": np.asarray(block["correct"], dtype=np.float64),
        }
    static_mix = StaticMix().fit(groups)
    adaptive_mix = AdaptiveMix().fit(groups)
    fit_record = {
        "static_c": float(static_mix.c),
        "static_nll": float(static_mix.nll),
        "static_success": bool(static_mix.success),
        "adaptive_theta": float(adaptive_mix.theta),
        "adaptive_tau": float(adaptive_mix.tau),
        "adaptive_beta": float(adaptive_mix.beta()),
        "adaptive_nll": float(adaptive_mix.nll),
        "adaptive_success": bool(adaptive_mix.success),
        "tune_rows": {f"m{m}": int(groups[m]["y"].size) for m in MIXER_FIT_LEVELS},
    }
    log(
        f"[M3-B0 seed{seed}] mixers fitted on tune m={{0,2,4}}: "
        f"c={static_mix.c:.4f} theta={adaptive_mix.theta:.4f} tau={adaptive_mix.tau:.4f} "
        f"beta={adaptive_mix.beta():.4f}"
    )

    # -- 5. frozen test cells + mixture inference ------------------------------
    t0 = time.perf_counter()
    test_blocks = runner._test_level_blocks(
        corpus, cohort_hc, rows_same8, scorers, int(seed), temperature, text_cache, b_of, log
    )
    log(f"[M3-B0 seed{seed}] frozen test cells loaded ({time.perf_counter() - t0:.1f}s)")

    with np.load(M2_DIR / f"seed_{int(seed)}" / "confidences.npz") as z:
        ref_curr = {f"m{m}": np.asarray(z[f"m{m}__E1b"], dtype=np.float64) for m in LEVELS}

    point_rows: List[Dict[str, Any]] = []
    calib_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []
    regret_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    per_level: Dict[int, Dict[str, np.ndarray]] = {}
    reference_clusters: Optional[np.ndarray] = None
    repro_curr = 0.0
    for m in LEVELS:
        level = f"m{m}"
        block = test_blocks[int(m)]
        correct = np.asarray(block["correct"], dtype=bool)
        clusters = np.asarray(block["image_id"], dtype=np.int64)
        if reference_clusters is None:
            reference_clusters = clusters.copy()
        else:
            if not (
                clusters.shape == reference_clusters.shape
                and np.array_equal(clusters, reference_clusters)
            ):
                raise AssertionError(f"seed{seed}/{level}: severity cells disagree on the shared rows")
        sem14 = np.asarray(block["sem14"])
        p_r = _p_logistic(
            block["stats17"], sem14, stats_fit_m1, sem_fit_m1, coef_rand, intercept_rand
        )
        p_c = _p_logistic(
            block["stats17"], sem14, stats_fit_m2, sem_fit_m2, coef_curr, intercept_curr
        )
        delta_curr = float(np.max(np.abs(p_c - ref_curr[level])))
        repro_curr = max(repro_curr, delta_curr)
        if not (delta_curr <= REPRO_TOL):
            raise AssertionError(
                f"seed{seed}/{level}: E_curriculum does not reproduce the frozen M2 "
                f"confidences ({delta_curr:.3e})"
            )
        a_test = index.transform(competition_features_from_sem14(sem14, runner.SEM14_NAMES))
        predictions = {
            "E_random": p_r,
            "E_curriculum": p_c,
            "EqualMix": EqualMix().predict_proba(p_r, p_c, a_test),
            "StaticMix": static_mix.predict_proba(p_r, p_c, a_test),
            "AdaptiveMix": adaptive_mix.predict_proba(p_r, p_c, a_test),
        }
        for model in MODELS:
            if predictions[model].shape != correct.shape:
                raise AssertionError(f"seed{seed}/{level}/{model}: row universe mismatch")
        per_level[int(m)] = {**predictions, "correct": correct, "clusters": clusters, "a": a_test}

        for model in MODELS:
            row = reval.point_metric_row(predictions[model], correct, probability=predictions[model])
            row.update(
                {
                    "level": level, "K": int(runner.K_LEVEL), "seed": int(seed), "model": model,
                    "n": int(correct.size), "n_images": int(np.unique(clusters).size),
                    "b3_accuracy": float(correct.mean()),
                }
            )
            point_rows.append(row)
            calib_rows.append(
                {
                    "level": level, "seed": int(seed), "model": model,
                    "mean_pred": float(np.mean(predictions[model])),
                    "accuracy": float(correct.mean()),
                    "ece_adaptive": float(row["ece_adaptive"]),
                    "brier_binary": float(row["brier_binary"]),
                    "nll_binary": float(row["nll_binary"]),
                }
            )

        alpha_values = adaptive_mix.alpha(a_test)
        alpha_rows.append(
            {
                "level": level, "seed": int(seed), "n": int(a_test.size),
                "a_mean": float(np.mean(a_test)), "a_median": float(np.median(a_test)),
                "a_p10": float(np.percentile(a_test, 10.0)),
                "a_p90": float(np.percentile(a_test, 90.0)),
                "alpha_mean": float(np.mean(alpha_values)),
                "alpha_median": float(np.median(alpha_values)),
            }
        )

        auroc_r = float(
            reval.point_metric_row(p_r, correct, probability=p_r)["auroc_correct"]
        )
        auroc_c = float(
            reval.point_metric_row(p_c, correct, probability=p_c)["auroc_correct"]
        )
        oracle = max(auroc_r, auroc_c)
        for model in MODELS:
            model_auroc = float(
                reval.point_metric_row(
                    predictions[model], correct, probability=predictions[model]
                )["auroc_correct"]
            )
            regret_rows.append(
                {
                    "level": level, "seed": int(seed), "model": model,
                    "oracle_auroc": oracle, "regret": oracle - model_auroc,
                }
            )

        for model_a, model_b in PER_LEVEL_PAIRS:
            rows = reval.model_vs_model_bootstrap_row(
                predictions[model_a], correct, predictions[model_b], correct, clusters,
                eval_split=runner.POOLED, K=int(runner.K_LEVEL),
                model_a=model_a, model_b=model_b, metrics=runner.BOOT_METRICS,
                replicates=int(args.bootstrap_replicates), seed=int(args.bootstrap_seed),
                ci=float(args.ci),
            )
            for row in rows:
                row.update({"level": level, "seed": int(seed)})
                boot_rows.append(row)
        log(
            f"  [M3-B0 seed{seed}] {level}: adaptive-static dAUROC="
            f"{float(np.mean([r['diff'] for r in boot_rows if r['level'] == level and r['model_a'] == 'AdaptiveMix' and r['model_b'] == 'StaticMix' and r['metric'] == 'auroc_correct'])):+.4f}"
        )

    # -- 6. macro bootstrap (shared draws across severity) ---------------------
    metric = reval._metric_fn("auroc_correct")
    for model_a, model_b in MACRO_PAIRS:
        severity_pairs = [
            (
                per_level[int(m)][model_a],
                per_level[int(m)]["correct"],
                per_level[int(m)][model_b],
                per_level[int(m)]["correct"],
            )
            for m in LEVELS
        ]
        pairs = [((p_a, c_a), (p_b, c_b)) for p_a, c_a, p_b, c_b in severity_pairs]
        result = macro_paired_cluster_bootstrap(
            metric, pairs, reference_clusters,
            n_replicates=int(args.bootstrap_replicates), seed=int(args.bootstrap_seed),
            ci=float(args.ci), metric_name="auroc_correct",
        )
        boot_rows.append(
            {
                "level": "macro", "K": int(runner.K_LEVEL), "seed": int(seed),
                "model_a": model_a, "model_b": model_b, "metric": "auroc_correct",
                "diff": float(result["diff"]), "ci_low": float(result["ci_low"]),
                "ci_high": float(result["ci_high"]), "mean_a": float(result["mean_a"]),
                "mean_b": float(result["mean_b"]), "n": int(result["n"]),
                "n_clusters": int(result["n_clusters"]),
                "n_replicates": int(result["n_replicates"]),
                "ci_level": float(result["ci_level"]),
            }
        )

    # -- 7. macro metrics -------------------------------------------------------
    macro_rows: List[Dict[str, Any]] = []
    for model in MODELS:
        macro_auroc = float(
            np.mean([
                float(r["auroc_correct"]) for r in point_rows
                if r["seed"] == int(seed) and r["model"] == model
            ])
        )
        macro_e_aurc = float(
            np.mean([
                float(r["e_aurc"]) for r in point_rows
                if r["seed"] == int(seed) and r["model"] == model
            ])
        )
        macro_rer50 = float(
            np.mean([
                float(r["rer_at_50"]) for r in point_rows
                if r["seed"] == int(seed) and r["model"] == model
            ])
        )
        macro_rer80 = float(
            np.mean([
                float(r["rer_at_80"]) for r in point_rows
                if r["seed"] == int(seed) and r["model"] == model
            ])
        )
        macro_rows.append(
            {
                "seed": int(seed), "model": model, "macro_auroc": macro_auroc,
                "macro_e_aurc": macro_e_aurc, "macro_rer50": macro_rer50,
                "macro_rer80": macro_rer80,
            }
        )

    log(
        f"[M3-B0 seed{seed}] done in {time.perf_counter() - t_seed:.1f}s "
        f"(repro rand {repro_rand:.2e}, curr {repro_curr:.2e})"
    )
    return {
        "seed": int(seed),
        "temperature": temperature,
        "fit": fit_record,
        "index_summary": index.summary(),
        "repro_random_conf": repro_rand,
        "repro_curriculum_conf": repro_curr,
        "point_rows": point_rows,
        "macro_rows": macro_rows,
        "calib_rows": calib_rows,
        "alpha_rows": alpha_rows,
        "regret_rows": regret_rows,
        "boot_rows": boot_rows,
    }


# ---------------------------------------------------------------------------
# summary aggregation (descriptive only -- B0 has no gate, amendment s.12/13)
# ---------------------------------------------------------------------------
def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def _add_support_counts(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Seed-level CI directions of one aggregate entry (descriptive only)."""
    if entry is None:
        return None
    lows = [float(v[0]) for v in entry["per_seed_ci"].values()]
    highs = [float(v[1]) for v in entry["per_seed_ci"].values()]
    entry["n_seeds_ci_low_gt_0"] = int(sum(lo > 0.0 for lo in lows))
    entry["n_seeds_ci_high_lt_0"] = int(sum(hi < 0.0 for hi in highs))
    return entry


def _summary(
    per_seed: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    boot_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    regret_rows: Sequence[Mapping[str, Any]],
    runner: Any,
) -> Dict[str, Any]:
    metrics = tuple(runner.BOOT_METRICS)

    fit_keys = (
        "static_c", "static_nll", "adaptive_theta", "adaptive_tau",
        "adaptive_beta", "adaptive_nll",
    )
    fits = {key: _stats([float(e["fit"][key]) for e in per_seed]) for key in fit_keys}
    fits["success"] = {
        "static": [bool(e["fit"]["static_success"]) for e in per_seed],
        "adaptive": [bool(e["fit"]["adaptive_success"]) for e in per_seed],
    }
    fits["tune_rows"] = {int(e["seed"]): e["fit"]["tune_rows"] for e in per_seed}
    fits["n_parameters"] = {
        "AdaptiveMix": 2, "StaticMix": 1, "EqualMix": 0, "hidden_layers": 0,
    }

    delta_macro = {
        "definition": (
            "MacroAUROC = mean of AUROC over m in {0,2,4,8}; delta = MacroAUROC(model_a) - "
            "MacroAUROC(model_b); image-cluster paired bootstrap (shared draws across severity)"
        ),
        "vs_static": _add_support_counts(
            _aggregate(boot_rows, "macro", "AdaptiveMix", "StaticMix", "auroc_correct")
        ),
        "vs_random": _add_support_counts(
            _aggregate(boot_rows, "macro", "AdaptiveMix", "E_random", "auroc_correct")
        ),
        "vs_curriculum": _add_support_counts(
            _aggregate(boot_rows, "macro", "AdaptiveMix", "E_curriculum", "auroc_correct")
        ),
        "draft_gate_preview": {
            "rule": "delta_macro >= 0.003 AND CI_low > 0 (per backbone)",
            "threshold": 0.003,
            "status": "DRAFT (frozen as Amendment V2-M3.1 after B0); B0 itself has no gate",
        },
    }

    extreme_safety = {
        "m0_adaptive_minus_random": _add_support_counts(
            _aggregate(boot_rows, "m0", "AdaptiveMix", "E_random", "auroc_correct")
        ),
        "m8_adaptive_minus_curriculum": _add_support_counts(
            _aggregate(boot_rows, "m8", "AdaptiveMix", "E_curriculum", "auroc_correct")
        ),
        "draft_gate_preview": {
            "rule": (
                "AUROC(Adaptive,m0) - AUROC(Random,m0) >= -0.003 AND "
                "AUROC(Adaptive,m8) - AUROC(Curriculum,m8) >= -0.003"
            ),
            "threshold": -0.003,
            "status": "DRAFT (frozen as Amendment V2-M3.1 after B0); B0 itself has no gate",
        },
    }

    macro_metrics: Dict[str, Any] = {}
    for model in MODELS:
        seed_rows = sorted(
            [r for r in macro_rows if r["model"] == model], key=lambda r: int(r["seed"])
        )
        macro_metrics[model] = {
            "macro_auroc": _stats([r["macro_auroc"] for r in seed_rows]),
            "macro_e_aurc": _stats([r["macro_e_aurc"] for r in seed_rows]),
            "macro_rer50": _stats([r["macro_rer50"] for r in seed_rows]),
            "macro_rer80": _stats([r["macro_rer80"] for r in seed_rows]),
            "per_seed": [
                {
                    key: r[key]
                    for key in (
                        "seed", "macro_auroc", "macro_e_aurc", "macro_rer50", "macro_rer80"
                    )
                }
                for r in seed_rows
            ],
        }

    static_e = float(macro_metrics["StaticMix"]["macro_e_aurc"]["mean"])
    adaptive_e = float(macro_metrics["AdaptiveMix"]["macro_e_aurc"]["mean"])
    e_aurc_rel = (
        float((static_e - adaptive_e) / static_e)
        if np.isfinite(static_e) and abs(static_e) > 0.0
        else None
    )
    static_r50 = float(macro_metrics["StaticMix"]["macro_rer50"]["mean"])
    adaptive_r50 = float(macro_metrics["AdaptiveMix"]["macro_rer50"]["mean"])
    rer50_gain = adaptive_r50 - static_r50
    selective_support = {
        "definition": "macro over m in {0,2,4,8}, relative to StaticMix",
        "macro_e_aurc": {
            "adaptive": adaptive_e, "static": static_e,
            "relative_improvement": e_aurc_rel,
            "improves_ge_3pct": bool(e_aurc_rel is not None and e_aurc_rel >= 0.03),
        },
        "macro_rer50": {
            "adaptive": adaptive_r50, "static": static_r50,
            "gain": float(rer50_gain), "gain_pp": float(100.0 * rer50_gain),
            "gain_ge_1pp": bool(rer50_gain >= 0.01),
        },
        "draft_rule_satisfied": bool(
            (e_aurc_rel is not None and e_aurc_rel >= 0.03) or rer50_gain >= 0.01
        ),
        "draft_rule": (
            ">= 3% relative macro E-AURC improvement OR >= 1pp macro RER@50 gain versus "
            "StaticMix; otherwise the result is labelled AUROC-ONLY METHOD SIGNAL"
        ),
        "status": "DRAFT (frozen as Amendment V2-M3.1 after B0); B0 itself has no gate",
    }

    alpha_by_level: Dict[str, Any] = {}
    for m in LEVELS:
        sel = [r for r in alpha_rows if r["level"] == f"m{m}"]
        alpha_by_level[f"m{m}"] = {
            "n_rows_total": int(sum(int(r["n"]) for r in sel)),
            "a_mean": _stats([r["a_mean"] for r in sel]),
            "a_median": _stats([r["a_median"] for r in sel]),
            "a_p10": _stats([r["a_p10"] for r in sel]),
            "a_p90": _stats([r["a_p90"] for r in sel]),
            "alpha_mean": _stats([r["alpha_mean"] for r in sel]),
            "alpha_median": _stats([r["alpha_median"] for r in sel]),
        }
    alpha_seq = [float(alpha_by_level[f"m{m}"]["alpha_mean"]["mean"]) for m in LEVELS]
    a_seq = [float(alpha_by_level[f"m{m}"]["a_mean"]["mean"]) for m in LEVELS]

    def _non_decreasing(seq: Sequence[float]) -> bool:
        return bool(all(seq[i + 1] >= seq[i] - 1e-12 for i in range(len(seq) - 1)))

    alpha_trend = {
        "alpha_mean_by_m": {f"m{m}": float(alpha_seq[i]) for i, m in enumerate(LEVELS)},
        "a_mean_by_m": {f"m{m}": float(a_seq[i]) for i, m in enumerate(LEVELS)},
        "alpha_non_decreasing_in_m": _non_decreasing(alpha_seq),
        "a_non_decreasing_in_m": _non_decreasing(a_seq),
        "note": "expected pattern; NOT gated (protocol section 23)",
    }

    regret: Dict[str, Any] = {}
    for model in MODELS:
        sel = [r for r in regret_rows if r["model"] == model]
        per_level: Dict[str, Any] = {}
        for m in LEVELS:
            vals = [float(r["regret"]) for r in sel if r["level"] == f"m{m}"]
            per_level[f"m{m}"] = {
                "mean": float(np.mean(vals)), "max": float(np.max(vals)), "n_seeds": len(vals),
            }
        regret[model] = {
            "per_level": per_level,
            "mean_regret": float(np.mean([float(r["regret"]) for r in sel])),
            "max_regret": float(np.max([float(r["regret"]) for r in sel])),
        }

    per_level_deltas = {
        "AdaptiveMix_minus_StaticMix": {
            f"m{m}": {
                metric: _aggregate(boot_rows, f"m{m}", "AdaptiveMix", "StaticMix", metric)
                for metric in metrics
            }
            for m in LEVELS
        },
        "AdaptiveMix_minus_E_random": {
            f"m{m}": {
                metric: _aggregate(boot_rows, f"m{m}", "AdaptiveMix", "E_random", metric)
                for metric in metrics
            }
            for m in LEVELS
        },
        "AdaptiveMix_minus_E_curriculum": {
            f"m{m}": {
                metric: _aggregate(boot_rows, f"m{m}", "AdaptiveMix", "E_curriculum", metric)
                for metric in metrics
            }
            for m in LEVELS
        },
    }

    repro = {
        "random_vs_m1_confidences_max": max(float(e["repro_random_conf"]) for e in per_seed),
        "curriculum_vs_m2_confidences_max": max(
            float(e["repro_curriculum_conf"]) for e in per_seed
        ),
        "tolerance": REPRO_TOL,
    }

    vs_static = delta_macro["vs_static"] or {}
    b0_questions = {
        "alpha_mean_by_m": alpha_trend["alpha_mean_by_m"],
        "alpha_distribution_by_m": {
            f"m{m}": {
                "alpha_mean": alpha_by_level[f"m{m}"]["alpha_mean"]["mean"],
                "alpha_median": alpha_by_level[f"m{m}"]["alpha_median"]["mean"],
                "a_p10": alpha_by_level[f"m{m}"]["a_p10"]["mean"],
                "a_p90": alpha_by_level[f"m{m}"]["a_p90"]["mean"],
            }
            for m in LEVELS
        },
        "does_alpha_increase_with_severity": bool(alpha_trend["alpha_non_decreasing_in_m"]),
        "adaptive_vs_static": {
            "delta_macro": float(vs_static.get("delta", float("nan"))),
            "ci_low": float(vs_static.get("ci_low", float("nan"))),
            "ci_high": float(vs_static.get("ci_high", float("nan"))),
        },
        "extreme_regime_regret": {
            "m0": regret["AdaptiveMix"]["per_level"]["m0"],
            "m8": regret["AdaptiveMix"]["per_level"]["m8"],
        },
    }

    return {
        "artifact": "v2m_m3_b0_summary",
        "amendment": "V2-M3",
        "label": "POST-HOC DEVELOPMENTAL",
        "no_gate": True,
        "boundary": (
            "B0 mixture numbers validate the implementation and the fitting protocol only and "
            "must never be presented as confirmatory method evidence; the confirmatory cohorts "
            "are B1 OpenCLIP B/16 and B2 SigLIP B/16 (frozen as Amendment V2-M3.1)"
        ),
        "models": list(MODELS),
        "levels": list(LEVELS),
        "mixer_fit_levels": list(MIXER_FIT_LEVELS),
        "n_point_rows": len(point_rows),
        "fits": fits,
        "delta_macro": delta_macro,
        "extreme_regime_safety": extreme_safety,
        "selective_support": selective_support,
        "macro_metrics": macro_metrics,
        "per_level_deltas": per_level_deltas,
        "alpha_trend": alpha_trend,
        "alpha_by_level": alpha_by_level,
        "regret": regret,
        "repro_checks": repro,
        "b0_questions": b0_questions,
        "interpretation": {
            "adaptive_better_than_static": (
                "delta_macro(Adaptive-Static) > 0 with CI_low > 0 -> competition-dependent "
                "weighting adds something beyond a constant ensemble (to be tested on B1/B2)"
            ),
            "adaptive_equals_static": (
                "AdaptiveMix ~ StaticMix -> competition adaptation itself has no demonstrated "
                "value (protocol section 9)"
            ),
            "extreme_regimes": (
                "the adaptive mixture should not clearly damage the specialist extremes "
                "(m0 for the random expert, m8 for the curriculum expert)"
            ),
            "next_step": (
                "freeze Amendment V2-M3.1, then prepare B1/B2 experts and mixers; B0 is never a gate"
            ),
        },
    }


# ---------------------------------------------------------------------------
# figures (descriptive: severity curves, alpha behaviour, macro signal)
# ---------------------------------------------------------------------------
def _write_figures(
    point_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    dmacro: Optional[Mapping[str, Any]],
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
            color=COLOURS[model], label=model,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"m={m}" for m in LEVELS])
    ax1.set_xlabel("competition severity m (same-category distractors), K=10")
    ax1.set_ylabel("AUROC(correct), frozen test cells")
    ax1.set_title("V2-M3 B0 family vs severity (seed mean +/- std)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    a_means: List[float] = []
    a_stds: List[float] = []
    alpha_means: List[float] = []
    alpha_stds: List[float] = []
    for m in LEVELS:
        sel = [r for r in alpha_rows if r["level"] == f"m{m}"]
        a_means.append(float(np.mean([r["a_mean"] for r in sel])))
        a_stds.append(float(np.std([r["a_mean"] for r in sel])))
        alpha_means.append(float(np.mean([r["alpha_mean"] for r in sel])))
        alpha_stds.append(float(np.std([r["alpha_mean"] for r in sel])))
    ax2.errorbar(
        x, alpha_means, yerr=alpha_stds, marker="s", capsize=3.0, linewidth=1.7,
        color=COLOURS["AdaptiveMix"], label="alpha(A) mean (AdaptiveMix)",
    )
    ax2.errorbar(
        x, a_means, yerr=a_stds, marker="^", capsize=3.0, linewidth=1.5,
        linestyle="--", color="#7f7f7f", label="A mean (competition index)",
    )
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"m={m}" for m in LEVELS])
    ax2.set_xlabel("competition severity m (same-category distractors), K=10")
    ax2.set_ylabel("mean in [0, 1]")
    ax2.set_title("Competition index A and adaptive weight alpha(A) by severity")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path1 = fig_dir / "m3_b0_adaptive_curves.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)
    log(f"[M3-B0] figure written: {path1}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    ax1, ax2 = axes
    positions = np.arange(len(MODELS), dtype=float)
    means: List[float] = []
    stds: List[float] = []
    for model in MODELS:
        vals = [float(r["macro_auroc"]) for r in macro_rows if r["model"] == model]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    ax1.bar(positions, means, yerr=stds, capsize=4.0, color=[COLOURS[m] for m in MODELS])
    for index, model in enumerate(MODELS):
        vals = [float(r["macro_auroc"]) for r in macro_rows if r["model"] == model]
        ax1.scatter(np.full(len(vals), positions[index]), vals, color="black", s=14, zorder=3)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(list(MODELS), rotation=12)
    ax1.set_ylabel("MacroAUROC over m in {0,2,4,8}")
    ax1.set_title("V2-M3 B0 family: MacroAUROC (seed mean +/- std, dots = seeds)")
    ax1.grid(alpha=0.3, axis="y")

    if dmacro is not None and dmacro.get("per_seed"):
        seeds_sorted = sorted(int(s) for s in dmacro["per_seed"])
        deltas = np.array([float(dmacro["per_seed"][s]) for s in seeds_sorted])
        lows = np.array([float(dmacro["per_seed_ci"][s][0]) for s in seeds_sorted])
        highs = np.array([float(dmacro["per_seed_ci"][s][1]) for s in seeds_sorted])
        yerr = np.vstack([deltas - lows, highs - deltas])
        ax2.errorbar(
            np.arange(len(seeds_sorted), dtype=float), deltas, yerr=yerr, marker="o",
            linestyle="none", capsize=4.0, color=COLOURS["AdaptiveMix"],
            label="per-seed delta_macro (95% paired CI)",
        )
        ax2.axhline(
            float(dmacro["delta"]), color=COLOURS["AdaptiveMix"], linewidth=1.2,
            linestyle="--", label=f"seed mean = {float(dmacro['delta']):+.4f}",
        )
        ax2.axhline(0.0, color="black", linewidth=0.9)
        ax2.axhline(
            0.003, color="#9467bd", linewidth=1.0, linestyle=":",
            label="draft gate 0.003 (not applied at B0)",
        )
        ax2.set_xticks(np.arange(len(seeds_sorted), dtype=float))
        ax2.set_xticklabels([f"seed {s}" for s in seeds_sorted])
        ax2.legend(fontsize=8, loc="best")
    ax2.set_ylabel("MacroAUROC(AdaptiveMix) - MacroAUROC(StaticMix)")
    ax2.set_title("Primary signal preview: AdaptiveMix vs StaticMix (descriptive)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path2 = fig_dir / "m3_b0_macro_delta.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    log(f"[M3-B0] figure written: {path2}")
    return [path1, path2]


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
POINT_FIELDS: Tuple[str, ...] = (
    "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80",
    "ece_adaptive", "brier_binary", "nll_binary",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--bootstrap-replicates", type=int, default=BOOT_REPLICATES)
    p.add_argument("--bootstrap-seed", type=int, default=BOOT_SEED)
    p.add_argument("--ci", type=float, default=BOOT_CI)
    p.add_argument(
        "--smoke", action="store_true",
        help="1 seed / 100 reps / results/v2_local_competition/m3_mixture/b0/_smoke",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="override the output directory (used by later consistency audits)",
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
        else ((B0_DIR / "_smoke") if args.smoke else B0_DIR)
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
        "[M3-B0] competition-adaptive reliability mixture B0 developmental audit start "
        "(POST-HOC DEVELOPMENTAL; experts frozen; no training; no gate)"
    )
    # frozen level contracts: the script constants must equal the M2 severity sets
    if tuple(LEVELS) != tuple(runner.LEVELS_TEST):
        raise AssertionError(
            f"severity levels {LEVELS} disagree with the frozen runner levels {runner.LEVELS_TEST}"
        )
    if tuple(MIXER_FIT_LEVELS) != tuple(runner.LEVELS_TRAIN):
        raise AssertionError(
            f"mixer fit levels {MIXER_FIT_LEVELS} disagree with the curriculum train levels "
            f"{runner.LEVELS_TRAIN}"
        )

    # frozen split / cohort / val tables / scorers / corpus (mirror of M2.5 main)
    split = sdata.load_split_from_manifest(_abs(runner.SPLIT_MANIFEST))
    cohort_hc = shard.load_hard_cohort(
        features_dir=sdata.PHASE05_FEATURES_DIR, manifests_root=_abs(runner.MANIFESTS), log=log
    )
    rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    val_cohort = build_module.load_val_cohort()
    scorers = hscores.load_frozen_scorers(seeds=seeds)
    corpus = B3Corpus(
        _abs(runner.FEATURES_ROOT), _abs(runner.MANIFESTS), _abs(runner.REFS),
        _abs(runner.BANK), image_sizes_path=_abs(runner.IMAGE_SIZES), ks=(5,),
        regime="random", preload=True,
    )

    per_seed: List[Dict[str, Any]] = []
    try:
        for seed in seeds:
            per_seed.append(
                _run_seed(
                    int(seed), runner=runner, m1_module=m1_module, corpus=corpus,
                    val_cohort=val_cohort, cohort_hc=cohort_hc, rows_same8=rows_same8,
                    split=split, scorers=scorers, args=run_args, log=log,
                )
            )
    finally:
        corpus.close()

    point_rows = [r for entry in per_seed for r in entry["point_rows"]]
    macro_rows = [r for entry in per_seed for r in entry["macro_rows"]]
    boot_rows = [r for entry in per_seed for r in entry["boot_rows"]]
    alpha_rows = [r for entry in per_seed for r in entry["alpha_rows"]]
    regret_rows = [r for entry in per_seed for r in entry["regret_rows"]]
    calib_rows = [r for entry in per_seed for r in entry["calib_rows"]]

    summary = _summary(
        per_seed, point_rows, macro_rows, boot_rows, alpha_rows, regret_rows, runner
    )

    _write_csv(
        out_dir / "point_metrics.csv", point_rows,
        ("level", "K", "seed", "model", "n", "n_images", "b3_accuracy") + POINT_FIELDS,
    )
    _write_csv(
        out_dir / "macro_metrics.csv", macro_rows,
        ("seed", "model", "macro_auroc", "macro_e_aurc", "macro_rer50", "macro_rer80"),
    )
    _write_csv(
        out_dir / "bootstrap_pairs.csv", boot_rows,
        ("level", "K", "seed", "model_a", "model_b", "metric", "diff", "ci_low", "ci_high",
         "mean_a", "mean_b", "n", "n_clusters", "n_replicates", "ci_level"),
    )
    _write_csv(
        out_dir / "alpha_diagnostics.csv", alpha_rows,
        ("level", "seed", "n", "a_mean", "a_median", "a_p10", "a_p90",
         "alpha_mean", "alpha_median"),
    )
    _write_csv(
        out_dir / "regret.csv", regret_rows,
        ("level", "seed", "model", "oracle_auroc", "regret"),
    )
    _write_csv(
        out_dir / "calibration.csv", calib_rows,
        ("level", "seed", "model", "mean_pred", "accuracy", "ece_adaptive",
         "brier_binary", "nll_binary"),
    )

    figures = _write_figures(
        point_rows, macro_rows, alpha_rows, summary["delta_macro"]["vs_static"], out_dir, log
    )
    _write_json(out_dir / "summary.json", summary)

    metadata = {
        "artifact": "v2m_m3_b0_metadata",
        "amendment": "V2-M3",
        "label": "POST-HOC DEVELOPMENTAL",
        "branch": runner._git(["branch", "--show-current"]),
        "head": runner._git(["rev-parse", "HEAD"]),
        "dirty": bool(runner._git(["status", "--porcelain"])),
        "started_utc": started_utc,
        "runtime_sec": round(time.perf_counter() - started, 1),
        "seeds": list(seeds),
        "levels": list(LEVELS),
        "mixer_fit_levels": list(MIXER_FIT_LEVELS),
        "k_level": int(runner.K_LEVEL),
        "bootstrap": {
            "replicates": int(reps), "seed": int(args.bootstrap_seed),
            "ci": float(args.ci), "metrics": list(runner.BOOT_METRICS),
            "shared_draws_across_severity": True,
        },
        "sources": {
            "protocol_m3_sha": runner._sha256_file(M3_DIR / "protocol_m3.json"),
            "mixer_module_sha": runner._sha256_file(_REPO / "src/ccg/mixture/mixer.py"),
            "bootstrap_module_sha": runner._sha256_file(_REPO / "src/ccg/mixture/bootstrap.py"),
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
        "parameter_budget": {
            "AdaptiveMix": 2, "StaticMix": 1, "EqualMix": 0, "hidden_layers": 0,
        },
        "repro_checks": summary["repro_checks"],
        "figures": [str(p) for p in figures],
        "read_only": True,
        "no_training": True,
        "experts_frozen": True,
    }
    _write_json(out_dir / "metadata.json", metadata)

    d_static = summary["delta_macro"]["vs_static"] or {}
    d_random = summary["delta_macro"]["vs_random"] or {}
    d_curr = summary["delta_macro"]["vs_curriculum"] or {}
    safety0 = summary["extreme_regime_safety"]["m0_adaptive_minus_random"] or {}
    safety8 = summary["extreme_regime_safety"]["m8_adaptive_minus_curriculum"] or {}
    log("=" * 78)
    log("V2-M3 B0 developmental audit report (POST-HOC DEVELOPMENTAL; no GO/NO-GO)")
    log(
        "  mixers fitted on tune m={0,2,4} (seed mean): "
        f"c={summary['fits']['static_c']['mean']:.4f} "
        f"theta={summary['fits']['adaptive_theta']['mean']:.4f} "
        f"tau={summary['fits']['adaptive_tau']['mean']:.4f} "
        f"beta={summary['fits']['adaptive_beta']['mean']:.4f}"
    )
    log(
        "  delta_macro(A-S)={:+.4f} CI=[{:+.4f},{:+.4f}]  delta_macro(A-R)={:+.4f}  "
        "delta_macro(A-C)={:+.4f}".format(
            float(d_static.get("delta", float("nan"))),
            float(d_static.get("ci_low", float("nan"))),
            float(d_static.get("ci_high", float("nan"))),
            float(d_random.get("delta", float("nan"))),
            float(d_curr.get("delta", float("nan"))),
        )
    )
    log(
        "  extreme safety: m0(A-R)={:+.4f} CI=[{:+.4f},{:+.4f}]  "
        "m8(A-C)={:+.4f} CI=[{:+.4f},{:+.4f}]".format(
            float(safety0.get("delta", float("nan"))),
            float(safety0.get("ci_low", float("nan"))),
            float(safety0.get("ci_high", float("nan"))),
            float(safety8.get("delta", float("nan"))),
            float(safety8.get("ci_low", float("nan"))),
            float(safety8.get("ci_high", float("nan"))),
        )
    )
    log(
        "  alpha mean by m: "
        + " ".join(
            f"m{m}:{summary['alpha_trend']['alpha_mean_by_m'][f'm{m}']:.4f}" for m in LEVELS
        )
        + f"  non-decreasing={summary['alpha_trend']['alpha_non_decreasing_in_m']}"
    )
    log(
        "  A mean by m: "
        + " ".join(
            f"m{m}:{summary['alpha_trend']['a_mean_by_m'][f'm{m}']:.4f}" for m in LEVELS
        )
        + f"  non-decreasing={summary['alpha_trend']['a_non_decreasing_in_m']}"
    )
    log(
        "  regret (AdaptiveMix): mean={:.4f} max={:.4f}  m0 mean={:.4f}  m8 mean={:.4f}".format(
            float(summary["regret"]["AdaptiveMix"]["mean_regret"]),
            float(summary["regret"]["AdaptiveMix"]["max_regret"]),
            float(summary["regret"]["AdaptiveMix"]["per_level"]["m0"]["mean"]),
            float(summary["regret"]["AdaptiveMix"]["per_level"]["m8"]["mean"]),
        )
    )
    log(
        "  selective preview (draft, vs Static): "
        f"E-AURC rel={summary['selective_support']['macro_e_aurc']['relative_improvement']} "
        f"RER50 gain_pp={summary['selective_support']['macro_rer50']['gain_pp']:+.3f} "
        f"satisfied={summary['selective_support']['draft_rule_satisfied']}"
    )
    log(f"  repro checks: {summary['repro_checks']}")
    log("=" * 78)
    log(
        "V2M_M3_B0_COMPLETE "
        f"dmacro={float(d_static.get('delta', float('nan'))):+.4f} "
        f"dmacro_ci_low={float(d_static.get('ci_low', float('nan'))):+.4f} "
        f"alpha_m0={summary['alpha_trend']['alpha_mean_by_m']['m0']:.4f} "
        f"alpha_m8={summary['alpha_trend']['alpha_mean_by_m']['m8']:.4f} "
        f"runtime={time.perf_counter() - started:.1f}s out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
