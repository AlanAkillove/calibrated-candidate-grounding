"""Run the corrected Phase 0A.1 audit (metric / temperature validity).

Background
----------
Phase 0A (``results/phase0a_cosine``) reported a global temperature pinned at
its search lower bound (``T* = 0.05``) and cross-``K`` AURC numbers that move
with the base error rate.  This driver redoes both:

* the global temperature is re-fitted on ``val_calib`` common-cohort rows of
  ``K in {5, 10}`` with the log-space bounded minimiser
  (:func:`ccg.calibration.temperature_opt.fit_temperature_log_space`), which
  detects a bound collision and widens the interval once;
* every selective number is paired with an accuracy-normalised companion
  (``oracle_aurc`` / ``E-AURC`` / ``RER``), and the correctness-event
  discrimination (``AUROC_correct`` / ``AUPRC_correct``) is reported next to it.

The frozen feature cache is read (never re-extracted) and the frozen
``ccg.experiment.phase0a`` helpers are reused read-only; no Phase 0A artifact is
touched.  Everything is NumPy/CPU.

Variants
--------
``native``             ``T = 1 / native_logit_scale`` (deployment probability)
``global_T_corrected`` ``T =`` the corrected global temperature
``oracle_T_K``         per-``K`` temperature fitted on that ``K``'s val_calib
                       common rows - **ORACLE / DIAGNOSTIC - NOT A VALID OOD
                       METHOD** (fitted on the split it is scored on)

Usage
-----
    python scripts/run_phase0a_corrected.py \
        --features cache/features \
        --manifests cache/manifests \
        --refs "data/raw/refcoco+/refcoco+/refs(unc).p" \
        --out results/phase0a_corrected \
        --bootstrap-replicates 5000

Progress reporting: a stage bar and a filling bootstrap bar (stderr,
``disable=None``).  Exit code 0 on success; a missing input file is an argparse
error (exit 2).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.calibration.temperature_opt import (  # noqa: E402
    DEFAULT_EXPANSION,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    fit_temperature_log_space,
)
from ccg.experiment.phase0a import (  # noqa: E402
    DEFAULT_KS,
    PRIMARY_SPLITS,
    REGIME_RANDOM,
    RELIABILITY_BINS,
    SampleStats,
    VARIANT_GLOBAL,
    VARIANT_NATIVE,
    _correctness,
    _entropy_mean,
    _variant_temperature,
    common_cohort_rows,
    load_cosine_inputs,
    paired_cluster_bootstrap,
    probabilities_for_variant,
    score_sets,
)
from ccg.metrics.calibration import (  # noqa: E402
    brier_binary,
    confidence_accuracy_gap,
    multiclass_brier,
    multiclass_nll,
    nll_binary,
    reliability_table,
    top_label_ece,
    top_label_ece_adaptive,
)
from ccg.metrics.discrimination import auprc_correct, auroc_correct  # noqa: E402
from ccg.metrics.ranking import target_rank  # noqa: E402
from ccg.metrics.selective import (  # noqa: E402
    aurc,
    oracle_aurc,
    rer_at_coverage,
    risk_coverage_curve,
)

__all__ = [
    "DEFAULT_FEATURES",
    "DEFAULT_MANIFESTS",
    "DEFAULT_OUT",
    "DEFAULT_REFS",
    "build_parser",
    "main",
    "run_corrected",
]

DEFAULT_FEATURES = Path("cache/features")
DEFAULT_MANIFESTS = Path("cache/manifests")
DEFAULT_REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
DEFAULT_OUT = Path("results/phase0a_corrected")

#: Confidence variants reported by this audit (report column order).
VARIANT_NATIVE_NAME = "native"
VARIANT_GLOBAL_CORRECTED = "global_T_corrected"
VARIANT_ORACLE = "oracle_T_K"
REPORT_VARIANTS: Tuple[str, ...] = (
    VARIANT_NATIVE_NAME,
    VARIANT_GLOBAL_CORRECTED,
    VARIANT_ORACLE,
)
#: Disclaimer carried verbatim by every oracle artifact.
ORACLE_KIND = "ORACLE / DIAGNOSTIC - NOT A VALID OOD METHOD"
#: The Ks the legal corrected temperature is fitted on.
CORRECTED_T_KINDS: Tuple[int, ...] = (5, 10)
#: The Ks of the per-K oracle diagnostic.
ORACLE_KS: Tuple[int, ...] = (5, 10, 20, 50)
#: Cross-K comparisons (baseline K5 vs larger).
K_PAIRS: Tuple[Tuple[int, int], ...] = ((5, 10), (5, 20), (5, 50))
#: Coverage levels of the RER tables.
RER_COVERAGES: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
#: Reliability diagrams: two binnings, 10 bins each.
N_BINS_RELIABILITY = 10
BINNER_FIXED = "fixed_width_10"
BINNER_EQUAL_MASS = "equal_mass_10"
#: A reliability bin with fewer samples gets ``low_support = True``.
LOW_SUPPORT_N = 100
POOLED = "__pooled__"

#: Route A / Route B thresholds (protocol Phase 0A.1).
ROUTE_A_REL_WORSENING = 0.20
ROUTE_A_RER_DROP_PP = 10.0
ROUTE_B_AUROC_DROP = 0.03
ROUTE_B_RELIABILITY_SHIFT = 0.02

_STAGES = (
    "load_inputs",
    "score_sets",
    "fit_temperatures",
    "metric_tables",
    "reliability",
    "bootstrap",
    "predictions_npz",
    "write_artifacts",
)


# ---------------------------------------------------------------------------
# small local utilities (kept independent of phase0a privates)
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value {value!r} is not JSON serialisable")


def _write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))
    return path


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="raise", restval=""
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(str(tmp), str(path))
    return path


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _python_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("scipy", "sklearn"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - environment dependent
            versions[name] = "missing"
    return versions


# ---------------------------------------------------------------------------
# variant plumbing
# ---------------------------------------------------------------------------
def _variant_temperature_value(
    variant: str,
    k: int,
    *,
    logit_scale: float,
    corrected_temperature: float,
    oracle_temperature: Dict[int, float],
) -> float:
    if variant == VARIANT_NATIVE_NAME:
        return float(_variant_temperature(VARIANT_NATIVE, logit_scale=logit_scale))
    if variant == VARIANT_GLOBAL_CORRECTED:
        return float(corrected_temperature)
    if variant == VARIANT_ORACLE:
        return float(oracle_temperature[int(k)])
    raise ValueError(f"unknown variant {variant!r}; expected one of {REPORT_VARIANTS}")


def _variant_probabilities(
    raw_scores: np.ndarray,
    variant: str,
    k: int,
    *,
    logit_scale: float,
    corrected_temperature: float,
    oracle_temperature: Dict[int, float],
) -> np.ndarray:
    if variant == VARIANT_NATIVE_NAME:
        return probabilities_for_variant(raw_scores, VARIANT_NATIVE, logit_scale=logit_scale)
    if variant == VARIANT_GLOBAL_CORRECTED:
        return probabilities_for_variant(
            raw_scores, VARIANT_GLOBAL, temperature=corrected_temperature
        )
    if variant == VARIANT_ORACLE:
        return probabilities_for_variant(
            raw_scores, VARIANT_GLOBAL, temperature=oracle_temperature[int(k)]
        )
    raise ValueError(f"unknown variant {variant!r}; expected one of {REPORT_VARIANTS}")


# ---------------------------------------------------------------------------
# bootstrap metric registry (SampleStats -> float)
# ---------------------------------------------------------------------------
def _eaurc_metric(stats: SampleStats) -> float:
    coverage, risk = risk_coverage_curve(stats.confidence, stats.correct)
    value = aurc(coverage, risk)
    accuracy = float(np.mean(stats.correct))
    return float(value - oracle_aurc(1.0 - accuracy))


def _auroc_metric(stats: SampleStats) -> float:
    return auroc_correct(stats.confidence, stats.correct)


def _rer_metric(coverage_level: float) -> Callable[[SampleStats], float]:
    def metric(stats: SampleStats) -> float:
        return rer_at_coverage(stats.confidence, stats.correct, coverage_level)

    metric.__name__ = f"rer_at_{int(round(coverage_level * 100))}"
    return metric


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features", type=Path, default=DEFAULT_FEATURES,
        help=f"frozen feature cache root (default: {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--manifests", type=Path, default=DEFAULT_MANIFESTS,
        help=f"candidate-manifest root (default: {DEFAULT_MANIFESTS})",
    )
    parser.add_argument(
        "--refs", type=Path, default=DEFAULT_REFS,
        help=f"path to refs(unc).p (default: {DEFAULT_REFS})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"artifact directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=5000,
        help="paired image-clustered bootstrap replicates (default: 5000)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="bootstrap RNG seed")
    parser.add_argument("--bootstrap-ci", type=float, default=0.95, help="bootstrap CI level")
    parser.add_argument(
        "--ks", type=int, nargs="+", default=list(DEFAULT_KS),
        help=f"candidate-set sizes (default: {list(DEFAULT_KS)})",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(PRIMARY_SPLITS), choices=list(PRIMARY_SPLITS),
        help=f"eval splits to report (default: {list(PRIMARY_SPLITS)})",
    )
    parser.add_argument(
        "--t-min", type=float, default=DEFAULT_T_MIN,
        help=f"initial temperature lower bound (default: {DEFAULT_T_MIN})",
    )
    parser.add_argument(
        "--t-max", type=float, default=DEFAULT_T_MAX,
        help=f"initial temperature upper bound (default: {DEFAULT_T_MAX})",
    )
    parser.add_argument(
        "--expansion", type=float, default=DEFAULT_EXPANSION,
        help=f"one-shot bound widening factor (default: {DEFAULT_EXPANSION})",
    )
    parser.add_argument("--device", default="cpu", help="recorded for provenance (NumPy/CPU)")
    return parser


def _make_logger(stage_bar: Optional["tqdm"]) -> Callable[[str], None]:
    def log(message: str) -> None:
        if stage_bar is not None:
            stage_bar.clear()
        print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)
        if stage_bar is not None:
            stage_bar.refresh()

    return log


def _require_file(path: Path, what: str, parser: argparse.ArgumentParser) -> Path:
    if not path.exists():
        parser.error(f"{what} not found: {path}")
    return path


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def run_corrected(
    *,
    features_root: Any,
    manifests_root: Any,
    refs: Any,
    out_dir: Any,
    ks: Sequence[int] = DEFAULT_KS,
    splits: Sequence[str] = PRIMARY_SPLITS,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 0,
    bootstrap_ci: float = 0.95,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    expansion: float = DEFAULT_EXPANSION,
    on_bootstrap: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the corrected audit and write every artifact under ``out_dir``."""
    started = time.perf_counter()
    durations: Dict[str, float] = {}

    def stage_done(name: str, stage_started: float) -> None:
        elapsed = time.perf_counter() - stage_started
        durations[name] = float(elapsed)
        if log is not None:
            log(f"[p0a1] {name}: {elapsed:.2f}s")

    out_dir = Path(out_dir)
    ks_t = tuple(sorted({int(k) for k in ks}))
    requested = [str(name) for name in splits]
    splits_t = tuple(name for name in PRIMARY_SPLITS if name in requested)

    # -- stage 1: inputs ----------------------------------------------------
    stage_started = time.perf_counter()
    records = load_cosine_inputs(
        features_root, manifests_root, refs, splits_t, regime=REGIME_RANDOM, ks=ks_t
    )
    if log is not None:
        log(f"[p0a1] loaded {len(records)} sentences")
    stage_done("load_inputs", stage_started)

    # -- stage 2: scoring (frozen cache, no re-extraction) ------------------
    stage_started = time.perf_counter()
    from ccg.features.cache import FeatureCache  # lazy: keeps light imports light

    with FeatureCache.open(features_root) as cache:
        cache_metadata = dict(getattr(cache, "metadata", {}) or {})
        logit_scale = float(cache_metadata["native_logit_scale"])
        bundle = score_sets(cache, records, ks_t)
    _, common_rows, common_ids = common_cohort_rows(bundle, ks_t)
    stage_done("score_sets", stage_started)

    def cell_rows(name: str, k: int) -> np.ndarray:
        base = common_rows[int(k)]
        if name == POOLED:
            return base
        return base[bundle.splits(int(k))[base] == name]

    def stats_for(name: str, k: int, variant: str) -> SampleStats:
        rows = cell_rows(name, int(k))
        prob = _variant_probabilities(
            bundle.raw_scores[int(k)][rows],
            variant,
            int(k),
            logit_scale=logit_scale,
            corrected_temperature=corrected_temperature,
            oracle_temperature=oracle_temperature,
        )
        return SampleStats.from_conf_correct(
            prob.max(axis=1), _correctness(bundle, int(k), rows)
        )

    # -- stage 3: temperature fitting --------------------------------------
    stage_started = time.perf_counter()
    corrected_ks = tuple(k for k in CORRECTED_T_KINDS if k in ks_t)
    corrected_sets: List[np.ndarray] = []
    corrected_targets: List[int] = []
    corrected_rows_by_k: Dict[int, int] = {}
    for k in corrected_ks:
        rows = cell_rows("val_calib", k)
        corrected_rows_by_k[int(k)] = int(rows.size)
        corrected_sets.extend(list(bundle.raw_scores[k][rows]))
        corrected_targets.extend(int(value) for value in bundle.target_local[k][rows].tolist())
    if not corrected_sets:
        raise ValueError("no val_calib common-cohort rows for K in {5, 10}; cannot fit T")
    corrected_fit = fit_temperature_log_space(
        corrected_sets, corrected_targets, t_min=float(t_min), t_max=float(t_max),
        expand_once=True, expansion=float(expansion),
    )
    corrected_temperature = float(corrected_fit.temperature)

    oracle_temperature: Dict[int, float] = {}
    oracle_fits: Dict[int, Any] = {}
    for k in ORACLE_KS:
        if k not in ks_t:
            continue
        rows = cell_rows("val_calib", k)
        if rows.size == 0:
            continue
        fit = fit_temperature_log_space(
            list(bundle.raw_scores[k][rows]),
            bundle.target_local[k][rows].tolist(),
            t_min=float(t_min), t_max=float(t_max), expand_once=True, expansion=float(expansion),
        )
        oracle_temperature[int(k)] = float(fit.temperature)
        oracle_fits[int(k)] = fit
    if log is not None:
        log(
            f"[p0a1] corrected T*={corrected_temperature:.6g} "
            f"interior={corrected_fit.interior} warning={corrected_fit.warning}; "
            f"oracle T={ {k: round(v, 6) for k, v in oracle_temperature.items()} }"
        )
    stage_done("fit_temperatures", stage_started)

    # -- stage 4: metric tables --------------------------------------------
    stage_started = time.perf_counter()
    names = list(splits_t) + [POOLED]
    calibration_rows: List[Dict[str, Any]] = []
    selective_rows: List[Dict[str, Any]] = []
    auroc_rows: List[Dict[str, Any]] = []
    for name in names:
        for k in ks_t:
            rows = cell_rows(name, int(k))
            if rows.size == 0:
                continue
            correct = _correctness(bundle, int(k), rows)
            accuracy = float(np.mean(correct))
            for variant in REPORT_VARIANTS:
                if variant == VARIANT_ORACLE and int(k) not in oracle_temperature:
                    continue
                raw = bundle.raw_scores[int(k)][rows]
                targets = bundle.target_local[int(k)][rows]
                prob = _variant_probabilities(
                    raw, variant, int(k), logit_scale=logit_scale,
                    corrected_temperature=corrected_temperature,
                    oracle_temperature=oracle_temperature,
                )
                confidence = prob.max(axis=1)
                temp = _variant_temperature_value(
                    variant, int(k), logit_scale=logit_scale,
                    corrected_temperature=corrected_temperature,
                    oracle_temperature=oracle_temperature,
                )
                calibration_rows.append({
                    "eval_split": name, "K": int(k), "variant": variant, "n": int(rows.size),
                    "accuracy": accuracy,
                    "ece_adaptive": float(
                        top_label_ece_adaptive(confidence, correct, n_bins=RELIABILITY_BINS)
                    ),
                    "ece_equal_width": float(
                        top_label_ece(confidence, correct, n_bins=RELIABILITY_BINS)
                    ),
                    "brier_binary": float(brier_binary(confidence, correct)),
                    "nll_binary": float(nll_binary(confidence, correct)),
                    "conf_acc_gap": float(confidence_accuracy_gap(confidence, correct)),
                    "multiclass_nll": float(multiclass_nll(raw, targets, temperature=temp)),
                    "multiclass_brier": float(multiclass_brier(raw, targets, temperature=temp)),
                    "mean_entropy": float(_entropy_mean(prob)),
                })
                coverage, risk = risk_coverage_curve(confidence, correct)
                aurc_value = float(aurc(coverage, risk))
                risk_full = 1.0 - accuracy
                aurc_oracle = float(oracle_aurc(risk_full))
                selective_rows.append({
                    "eval_split": name, "K": int(k), "variant": variant, "n": int(rows.size),
                    "accuracy": accuracy, "risk_full": risk_full,
                    "aurc": aurc_value, "aurc_oracle": aurc_oracle,
                    "e_aurc": float(aurc_value - aurc_oracle),
                    "auroc_correct": float(auroc_correct(confidence, correct)),
                    "auprc_correct": float(auprc_correct(confidence, correct)),
                    "rer_at_50": float(rer_at_coverage(confidence, correct, 0.5)),
                    "rer_at_80": float(rer_at_coverage(confidence, correct, 0.8)),
                    "rer_at_90": float(rer_at_coverage(confidence, correct, 0.9)),
                    "rer_at_95": float(rer_at_coverage(confidence, correct, 0.95)),
                })
                auroc_rows.append({
                    "eval_split": name, "K": int(k), "variant": variant, "n": int(rows.size),
                    "auroc_correct": float(auroc_correct(confidence, correct)),
                    "auprc_correct": float(auprc_correct(confidence, correct)),
                    "prevalence": accuracy,
                })
    stage_done("metric_tables", stage_started)

    # -- stage 5: reliability diagrams -------------------------------------
    stage_started = time.perf_counter()
    reliability_global_rows: List[Dict[str, Any]] = []
    reliability_all_rows: List[Dict[str, Any]] = []

    def _bin_rows(name: str, k: int, variant: str) -> List[Dict[str, Any]]:
        rows = cell_rows(name, int(k))
        if rows.size == 0:
            return []
        prob = _variant_probabilities(
            bundle.raw_scores[int(k)][rows], variant, int(k), logit_scale=logit_scale,
            corrected_temperature=corrected_temperature, oracle_temperature=oracle_temperature,
        )
        confidence = prob.max(axis=1)
        correct = _correctness(bundle, int(k), rows)
        out: List[Dict[str, Any]] = []
        for binning, adaptive in ((BINNER_FIXED, False), (BINNER_EQUAL_MASS, True)):
            table = reliability_table(
                confidence, correct, n_bins=N_BINS_RELIABILITY, adaptive=adaptive
            )
            edges = np.asarray(table["bin_edges"], dtype=np.float64)
            counts = np.asarray(table["counts"], dtype=np.int64)
            for index in range(int(table["n_bins"])):
                out.append({
                    "eval_split": name, "K": int(k), "variant": variant, "binning": binning,
                    "bin_index": int(index),
                    "bin_low": float(edges[index]), "bin_high": float(edges[index + 1]),
                    "n": int(counts[index]),
                    "bin_confidence": float(table["avg_confidence"][index]),
                    "bin_accuracy": float(table["accuracy"][index]),
                    "low_support": bool(counts[index] < LOW_SUPPORT_N),
                })
        return out

    for name in names:
        for k in ks_t:
            reliability_global_rows.extend(
                _bin_rows(name, int(k), VARIANT_GLOBAL_CORRECTED)
            )
            if VARIANT_NATIVE_NAME in REPORT_VARIANTS:
                reliability_all_rows.extend(_bin_rows(name, int(k), VARIANT_NATIVE_NAME))
            reliability_all_rows.extend(_bin_rows(name, int(k), VARIANT_GLOBAL_CORRECTED))
            if int(k) in oracle_temperature:
                reliability_all_rows.extend(_bin_rows(name, int(k), VARIANT_ORACLE))
    stage_done("reliability", stage_started)

    # -- stage 6: paired image-clustered bootstrap -------------------------
    stage_started = time.perf_counter()
    planned = 0
    for name in names:
        for _k_lo, k_hi in K_PAIRS:
            if cell_rows(name, 5).size >= 2 and cell_rows(name, k_hi).size >= 2:
                planned += len(REPORT_VARIANTS)  # e-auc
    pooled_pairs = [
        (k_lo, k_hi)
        for k_lo, k_hi in K_PAIRS
        if cell_rows(POOLED, 5).size >= 2 and cell_rows(POOLED, k_hi).size >= 2
    ]
    planned += len(pooled_pairs) * len(REPORT_VARIANTS) * len(RER_COVERAGES)
    planned += len(pooled_pairs) * len(REPORT_VARIANTS)
    done = 0

    def tick(label: str) -> None:
        nonlocal done
        done += 1
        if on_bootstrap is not None:
            on_bootstrap(done, planned, label)

    eaurc_boot_rows: List[Dict[str, Any]] = []
    rer_boot_rows: List[Dict[str, Any]] = []
    auroc_boot_rows: List[Dict[str, Any]] = []

    for name in names:
        for _k_lo, k_hi in K_PAIRS:
            rows5 = cell_rows(name, 5)
            rows_hi = cell_rows(name, k_hi)
            if rows5.size < 2 or rows_hi.size < 2:
                continue
            clusters = bundle.image_ids(5)[rows5]
            for variant in REPORT_VARIANTS:
                if variant == VARIANT_ORACLE and k_hi not in oracle_temperature:
                    tick(f"{name} e-auc {variant} K5vK{k_hi}")
                    continue
                stats_a = stats_for(name, 5, variant)
                stats_b = stats_for(name, k_hi, variant)
                boot = paired_cluster_bootstrap(
                    _eaurc_metric, stats_b, stats_a, clusters,
                    n_replicates=int(bootstrap_replicates), seed=int(bootstrap_seed),
                    ci=float(bootstrap_ci), metric_name="e_aurc",
                )
                e_a = _eaurc_metric(stats_a)
                e_b = _eaurc_metric(stats_b)
                denom = e_a
                diff_abs = float(boot["diff"])
                eaurc_boot_rows.append({
                    "eval_split": name, "variant": variant, "K_a": 5, "K_b": int(k_hi),
                    "n": int(rows5.size), "n_clusters": int(boot["n_clusters"]),
                    "e_aurc_a": float(e_a), "e_aurc_b": float(e_b),
                    "diff_abs": diff_abs,
                    "ci_low_abs": float(boot["ci_low"]), "ci_high_abs": float(boot["ci_high"]),
                    "diff_relative": float(diff_abs / denom) if denom != 0.0 else float("nan"),
                    "ci_low_relative": (
                        float(boot["ci_low"] / denom) if denom != 0.0 else float("nan")
                    ),
                    "ci_high_relative": (
                        float(boot["ci_high"] / denom) if denom != 0.0 else float("nan")
                    ),
                })
                tick(f"{name} e-auc {variant} K5vK{k_hi}")

    for k_lo, k_hi in pooled_pairs:
        clusters = bundle.image_ids(5)[cell_rows(POOLED, 5)]
        for variant in REPORT_VARIANTS:
            if variant == VARIANT_ORACLE and k_hi not in oracle_temperature:
                for coverage in RER_COVERAGES:
                    tick(f"pooled rer@{int(coverage * 100)} {variant} K5vK{k_hi}")
                tick(f"pooled auroc {variant} K5vK{k_hi}")
                continue
            stats_a = stats_for(POOLED, 5, variant)
            stats_b = stats_for(POOLED, k_hi, variant)
            for coverage in RER_COVERAGES:
                metric = _rer_metric(coverage)
                boot = paired_cluster_bootstrap(
                    metric, stats_b, stats_a, clusters,
                    n_replicates=int(bootstrap_replicates), seed=int(bootstrap_seed),
                    ci=float(bootstrap_ci), metric_name=f"rer_at_{int(coverage * 100)}",
                )
                rer_a = metric(stats_a)
                rer_b = metric(stats_b)
                diff = float(boot["diff"])
                rer_boot_rows.append({
                    "eval_split": POOLED, "coverage": float(coverage), "variant": variant,
                    "K_a": 5, "K_b": int(k_hi), "n": int(len(stats_a)),
                    "n_clusters": int(boot["n_clusters"]),
                    "rer_a": float(rer_a), "rer_b": float(rer_b),
                    "diff": diff, "diff_pp": diff * 100.0,
                    "ci_low": float(boot["ci_low"]), "ci_high": float(boot["ci_high"]),
                })
                tick(f"pooled rer@{int(coverage * 100)} {variant} K5vK{k_hi}")

            boot = paired_cluster_bootstrap(
                _auroc_metric, stats_b, stats_a, clusters,
                n_replicates=int(bootstrap_replicates), seed=int(bootstrap_seed),
                ci=float(bootstrap_ci), metric_name="auroc_correct",
            )
            auroc_boot_rows.append({
                "eval_split": POOLED, "variant": variant, "K_a": 5, "K_b": int(k_hi),
                "n": int(len(stats_a)), "n_clusters": int(boot["n_clusters"]),
                "auroc_a": float(_auroc_metric(stats_a)), "auroc_b": float(_auroc_metric(stats_b)),
                "diff": float(boot["diff"]),
                "ci_low": float(boot["ci_low"]), "ci_high": float(boot["ci_high"]),
            })
            tick(f"pooled auroc {variant} K5vK{k_hi}")
    stage_done("bootstrap", stage_started)

    # -- stage 7: frozen per-sentence predictions npz ----------------------
    stage_started = time.perf_counter()
    split_rank = {name: pos for pos, name in enumerate(splits_t)}
    r5 = common_rows[5]
    sid5 = bundle.sentence_ids(5)[r5]
    split5 = bundle.splits(5)[r5]
    rank5 = np.array([split_rank[str(s)] for s in split5], dtype=np.int64)
    perm = np.lexsort((sid5, rank5))
    ordered_sid = sid5[perm]
    ordered_split = split5[perm]
    ordered_ref = bundle.ref_ids(5)[r5][perm]
    ordered_image = bundle.image_ids(5)[r5][perm]

    payload: Dict[str, np.ndarray] = {
        "sentence_id": ordered_sid.astype(np.int64),
        "ref_id": ordered_ref.astype(np.int64),
        "image_id": ordered_image.astype(np.int64),
        "eval_split": ordered_split.astype("U10"),
    }
    for k in ks_t:
        rk = common_rows[int(k)]
        sidk = bundle.sentence_ids(int(k))[rk]
        splitk = bundle.splits(int(k))[rk]
        rankk = np.array([split_rank[str(s)] for s in splitk], dtype=np.int64)
        permk = np.lexsort((sidk, rankk))
        if not np.array_equal(sidk[permk], ordered_sid):
            raise RuntimeError(f"K={k}: per-sentence order disagrees with the K=5 order")
        rows_k = rk[permk]
        raw_k = bundle.raw_scores[int(k)][rows_k]
        payload[f"correct_K{k}"] = _correctness(bundle, int(k), rows_k).astype(np.int8)
        payload[f"target_rank_K{k}"] = target_rank(
            raw_k, bundle.target_local[int(k)][rows_k]
        ).astype(np.int16)
        payload[f"predicted_index_K{k}"] = np.argmax(raw_k, axis=1).astype(np.int16)
        for label, variant in (
            ("native", VARIANT_NATIVE_NAME),
            ("global_T", VARIANT_GLOBAL_CORRECTED),
            ("oracle_T", VARIANT_ORACLE),
        ):
            if variant == VARIANT_ORACLE and int(k) not in oracle_temperature:
                continue
            prob = _variant_probabilities(
                raw_k, variant, int(k), logit_scale=logit_scale,
                corrected_temperature=corrected_temperature,
                oracle_temperature=oracle_temperature,
            )
            payload[f"confidence_{label}_K{k}"] = prob.max(axis=1).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_dir / "per_sentence_predictions.npz"), **payload)
    n_rows_written = int(payload["sentence_id"].shape[0])
    stage_done("predictions_npz", stage_started)

    # -- stage 8: temperature manifest, metadata ---------------------------
    stage_started = time.perf_counter()
    oracle_values = [oracle_temperature[k] for k in sorted(oracle_temperature)]
    if len(oracle_values) >= 2 and max(oracle_values) > 0:
        spread_ratio = float(max(oracle_values) / min(oracle_values))
    else:
        spread_ratio = 1.0
    nearly_identical = bool(abs(spread_ratio - 1.0) < 0.05)
    temperature_payload = {
        "corrected_global_T": corrected_fit.to_dict(),
        "corrected_global_T_fit_split": "val_calib",
        "corrected_global_T_ks": [int(k) for k in corrected_ks],
        "corrected_global_T_n_sets_by_k": {str(k): int(v) for k, v in corrected_rows_by_k.items()},
        "oracle": {
            "kind": ORACLE_KIND,
            "note": (
                "each T_K is fitted on the val_calib common-cohort rows of its own K and "
                "is scored on the same split: diagnostic only, never a model result"
            ),
            "temperatures": {str(k): float(v) for k, v in sorted(oracle_temperature.items())},
            "fits": {str(k): fit.to_dict() for k, fit in sorted(oracle_fits.items())},
        },
        "oracle_per_k_nearly_identical": nearly_identical,
        "oracle_per_k_spread_ratio": spread_ratio,
        "oracle_per_k_conclusion": (
            "per-K oracle temperatures are (nearly) identical -> a single global "
            "temperature is sufficient"
            if nearly_identical
            else "per-K oracle temperatures differ -> the K-drift is partly a per-K "
            "sharpness effect (still diagnostic only)"
        ),
    }
    _write_json(out_dir / "temperature_fit.json", temperature_payload)

    _write_csv(
        out_dir / "calibration_metrics_corrected.csv",
        ["eval_split", "K", "variant", "n", "accuracy", "ece_adaptive", "ece_equal_width",
         "brier_binary", "nll_binary", "conf_acc_gap", "multiclass_nll", "multiclass_brier",
         "mean_entropy"],
        calibration_rows,
    )
    _write_csv(
        out_dir / "normalized_selective_metrics.csv",
        ["eval_split", "K", "variant", "n", "accuracy", "risk_full", "aurc", "aurc_oracle",
         "e_aurc", "auroc_correct", "auprc_correct", "rer_at_50", "rer_at_80", "rer_at_90",
         "rer_at_95"],
        selective_rows,
    )
    _write_csv(
        out_dir / "correctness_auroc.csv",
        ["eval_split", "K", "variant", "n", "auroc_correct", "auprc_correct", "prevalence"],
        auroc_rows,
    )
    reliability_fields = ["eval_split", "K", "variant", "binning", "bin_index", "bin_low",
                          "bin_high", "n", "bin_confidence", "bin_accuracy", "low_support"]
    _write_csv(out_dir / "reliability_bins_globalT.csv", reliability_fields, reliability_global_rows)
    _write_csv(
        out_dir / "reliability_bins_all_variants.csv", reliability_fields, reliability_all_rows
    )
    _write_csv(
        out_dir / "eaurc_bootstrap.csv",
        ["eval_split", "variant", "K_a", "K_b", "n", "n_clusters", "e_aurc_a", "e_aurc_b",
         "diff_abs", "ci_low_abs", "ci_high_abs", "diff_relative", "ci_low_relative",
         "ci_high_relative"],
        eaurc_boot_rows,
    )
    _write_csv(
        out_dir / "rer_bootstrap.csv",
        ["eval_split", "coverage", "variant", "K_a", "K_b", "n", "n_clusters", "rer_a", "rer_b",
         "diff", "diff_pp", "ci_low", "ci_high"],
        rer_boot_rows,
    )
    _write_csv(
        out_dir / "auroc_bootstrap.csv",
        ["eval_split", "variant", "K_a", "K_b", "n", "n_clusters", "auroc_a", "auroc_b", "diff",
         "ci_low", "ci_high"],
        auroc_boot_rows,
    )
    stage_done("write_artifacts", stage_started)

    cohort_counts = {
        name: int(cell_rows(name, 5).size if name != POOLED else common_ids.size)
        for name in names
    }
    metadata = {
        "artifact": "phase0a_corrected",
        "created_utc": _utc_now(),
        "git_commit": _git_commit(),
        "versions": _python_versions(),
        "config": {
            "features_root": str(features_root),
            "manifests_root": str(manifests_root),
            "refs": str(refs) if isinstance(refs, (str, Path)) else "<records>",
            "out_dir": str(out_dir),
            "ks": [int(k) for k in ks_t],
            "splits": [str(name) for name in splits_t],
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_ci": float(bootstrap_ci),
            "temperature_fit_bounds": [float(t_min), float(t_max)],
            "temperature_expansion": float(expansion),
            "device": "cpu",
        },
        "native_logit_scale": float(logit_scale),
        "n_records": int(len(records)),
        "n_common_pooled": int(common_ids.size),
        "cohort_common_counts": cohort_counts,
        "n_val_calib_sets_corrected_fit": int(len(corrected_sets)),
        "variants": list(REPORT_VARIANTS),
        "oracle_kind": ORACLE_KIND,
        "per_sentence_rows": n_rows_written,
        "counts": {
            "calibration_rows": len(calibration_rows),
            "selective_rows": len(selective_rows),
            "auroc_rows": len(auroc_rows),
            "reliability_global_rows": len(reliability_global_rows),
            "reliability_all_rows": len(reliability_all_rows),
            "eaurc_bootstrap_rows": len(eaurc_boot_rows),
            "rer_bootstrap_rows": len(rer_boot_rows),
            "auroc_bootstrap_rows": len(auroc_boot_rows),
            "bootstrap_comparisons_planned": int(planned),
        },
        "artifacts": [
            "temperature_fit.json",
            "calibration_metrics_corrected.csv",
            "normalized_selective_metrics.csv",
            "correctness_auroc.csv",
            "reliability_bins_globalT.csv",
            "reliability_bins_all_variants.csv",
            "eaurc_bootstrap.csv",
            "rer_bootstrap.csv",
            "auroc_bootstrap.csv",
            "per_sentence_predictions.npz",
            "metadata.json",
        ],
        "durations_seconds": durations,
        "total_seconds": float(time.perf_counter() - started),
    }
    _write_json(out_dir / "metadata.json", metadata)

    return {
        "out_dir": str(out_dir),
        "corrected_fit": corrected_fit,
        "corrected_temperature": corrected_temperature,
        "oracle_temperature": dict(oracle_temperature),
        "n_records": int(len(records)),
        "n_common_pooled": int(common_ids.size),
        "n_per_sentence_rows": n_rows_written,
        "calibration_rows": calibration_rows,
        "selective_rows": selective_rows,
        "eaurc_bootstrap_rows": eaurc_boot_rows,
        "rer_bootstrap_rows": rer_boot_rows,
        "auroc_bootstrap_rows": auroc_boot_rows,
        "oracle_per_k_nearly_identical": nearly_identical,
        "oracle_per_k_spread_ratio": spread_ratio,
        "durations_seconds": durations,
    }


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def _selective_lookup(
    rows: Sequence[Dict[str, Any]], split: str, k: int, variant: str
) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row["eval_split"] == split and int(row["K"]) == int(k) and row["variant"] == variant:
            return row
    return None


def _latest(
    rows: Sequence[Dict[str, Any]], key: str, value: Any
) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def _reliability_signed_gap(
    rows: Sequence[Dict[str, Any]], split: str, k: int, variant: str
) -> float:
    counts = 0.0
    total = 0.0
    for row in rows:
        if (
            row["eval_split"] == split
            and int(row["K"]) == int(k)
            and row["variant"] == variant
            and row["binning"] == BINNER_FIXED
        ):
            n = float(row["n"])
            counts += n
            total += n * (float(row["bin_confidence"]) - float(row["bin_accuracy"]))
    return float(total / counts) if counts > 0 else float("nan")


def _route_verdicts(
    selective_rows: Sequence[Dict[str, Any]],
    eaurc_rows: Sequence[Dict[str, Any]],
    rer_rows: Sequence[Dict[str, Any]],
    auroc_rows: Sequence[Dict[str, Any]],
    reliability_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    variant = VARIANT_GLOBAL_CORRECTED
    routes: Dict[str, Any] = {"variant": variant, "A": [], "B": []}
    for _k_lo, k_hi in ((5, 20), (5, 50)):
        ea = _latest(
            [r for r in eaurc_rows if r["eval_split"] == POOLED and r["variant"] == variant
             and int(r["K_b"]) == k_hi],
            "K_b", k_hi,
        )
        auroc = _latest(
            [r for r in auroc_rows if r["eval_split"] == POOLED and r["variant"] == variant
             and int(r["K_b"]) == k_hi],
            "K_b", k_hi,
        )
        rer50 = _latest(
            [r for r in rer_rows if r["eval_split"] == POOLED and r["variant"] == variant
             and int(r["K_b"]) == k_hi and float(r["coverage"]) == 0.5],
            "K_b", k_hi,
        )
        rer80 = _latest(
            [r for r in rer_rows if r["eval_split"] == POOLED and r["variant"] == variant
             and int(r["K_b"]) == k_hi and float(r["coverage"]) == 0.8],
            "K_b", k_hi,
        )
        rel_shift = _reliability_signed_gap(reliability_rows, POOLED, k_hi, variant) - (
            _reliability_signed_gap(reliability_rows, POOLED, 5, variant)
        )
        if ea is not None:
            ci_positive = float(ea["ci_low_abs"]) > 0.0
            rel_ok = float(ea["diff_relative"]) >= ROUTE_A_REL_WORSENING
            rer_drop = min(float(rer50["diff_pp"]), float(rer80["diff_pp"])) if (
                rer50 and rer80
            ) else 0.0
            rer_ok = rer_drop <= -ROUTE_A_RER_DROP_PP
            routes["A"].append({
                "K_hi": int(k_hi), "diff_relative": float(ea["diff_relative"]),
                "ci_low_abs": float(ea["ci_low_abs"]), "ci_high_abs": float(ea["ci_high_abs"]),
                "relative_ok": bool(rel_ok), "ci_ok": bool(ci_positive),
                "rer@50_pp": float(rer50["diff_pp"]) if rer50 else None,
                "rer@80_pp": float(rer80["diff_pp"]) if rer80 else None,
                "rer_ok": bool(rer_ok), "triggered": bool(rel_ok and ci_positive and rer_ok),
            })
        if auroc is not None:
            drop = float(auroc["diff"])
            drop_ok = drop <= -ROUTE_B_AUROC_DROP
            ci_ok = float(auroc["ci_high"]) < 0.0
            shift_ok = rel_shift >= ROUTE_B_RELIABILITY_SHIFT
            routes["B"].append({
                "K_hi": int(k_hi), "auroc_diff": drop,
                "ci_low": float(auroc["ci_low"]), "ci_high": float(auroc["ci_high"]),
                "auroc_ok": bool(drop_ok), "ci_ok": bool(ci_ok),
                "reliability_shift": float(rel_shift), "reliability_ok": bool(shift_ok),
                "triggered": bool(drop_ok and ci_ok and shift_ok),
            })
    routes["A_triggered"] = any(item["triggered"] for item in routes["A"])
    routes["B_triggered"] = any(item["triggered"] for item in routes["B"])
    return routes


def _print_summary(result: Dict[str, Any]) -> None:
    fit = result["corrected_fit"]
    print("[p0a1] ===== corrected audit =====")
    print(
        f"[p0a1] corrected global T* = {result['corrected_temperature']:.6g} "
        f"interior={fit.interior} warning={fit.warning} "
        f"bounds={ (round(fit.bounds[0], 8), round(fit.bounds[1], 8)) } "
        f"expanded={fit.bounds_expanded}"
    )
    oracle = result["oracle_temperature"]
    print(
        "[p0a1] oracle T: "
        + ", ".join(f"K{k}={oracle[k]:.6g}" for k in sorted(oracle))
        + f"  nearly_identical={result['oracle_per_k_nearly_identical']} "
        f"(spread x{result['oracle_per_k_spread_ratio']:.3f})"
    )
    variant = VARIANT_GLOBAL_CORRECTED
    print(f"[p0a1] per-K pooled numbers ({variant}):")
    for k in (5, 10, 20, 50):
        row = _selective_lookup(result["selective_rows"], POOLED, k, variant)
        cal = _selective_lookup(result["calibration_rows"], POOLED, k, variant)
        if row is None or cal is None:
            continue
        print(
            f"[p0a1]   K={k:>2}  n={int(row['n']):>6}  ECE={cal['ece_adaptive']:.4f}  "
            f"E-AURC={row['e_aurc']:.4f}  AUROC={row['auroc_correct']:.4f}  "
            f"RER@50={row['rer_at_50']:.3f} RER@80={row['rer_at_80']:.3f} "
            f"RER@90={row['rer_at_90']:.3f} RER@95={row['rer_at_95']:.3f}"
        )


def _print_routes(routes: Dict[str, Any]) -> None:
    print("[p0a1] ===== Route A / Route B (variant=global_T_corrected, pooled) =====")
    for item in routes["A"]:
        r50 = "n/a" if item["rer@50_pp"] is None else f"{item['rer@50_pp']:+.2f}"
        r80 = "n/a" if item["rer@80_pp"] is None else f"{item['rer@80_pp']:+.2f}"
        print(
            f"[p0a1] Route A K5->K{item['K_hi']}: E-AURC rel worsen={item['diff_relative']:+.3f} "
            f"(>=0.20? {item['relative_ok']}), CI=[{item['ci_low_abs']:+.4f},"
            f"{item['ci_high_abs']:+.4f}] not-cross-0? {item['ci_ok']}, "
            f"RER@50={r50}pp RER@80={r80}pp "
            f"(<=-10pp? {item['rer_ok']}) -> triggered={item['triggered']}"
        )
    for item in routes["B"]:
        print(
            f"[p0a1] Route B K5->K{item['K_hi']}: dAUROC={item['auroc_diff']:+.4f} "
            f"(<=-0.03? {item['auroc_ok']}), CI=[{item['ci_low']:+.4f},{item['ci_high']:+.4f}] "
            f"not-cross-0? {item['ci_ok']}, reliability shift={item['reliability_shift']:+.4f} "
            f"(>=0.02? {item['reliability_ok']}) -> triggered={item['triggered']}"
        )
    print(
        f"[p0a1] VERDICT: Route A triggered={routes['A_triggered']}, "
        f"Route B triggered={routes['B_triggered']}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    features = _require_file(args.features, "feature cache root", parser)
    manifests = _require_file(args.manifests, "manifest root", parser)
    refs = _require_file(args.refs, "refs(unc).p", parser)

    with tqdm(total=len(_STAGES), desc="p0a1", unit="stage", disable=None) as stage_bar:
        log = _make_logger(stage_bar)
        bootstrap_bar: Optional["tqdm"] = None

        def on_bootstrap(done: int, total: int, label: str) -> None:
            nonlocal bootstrap_bar
            if bootstrap_bar is None:
                bootstrap_bar = tqdm(
                    total=int(total), desc="bootstrap", unit="pair", disable=None,
                    position=1, leave=True,
                )
            bootstrap_bar.update(1)
            bootstrap_bar.set_postfix_str(label)

        log(f"[p0a1] features={features} manifests={manifests}")
        log(f"[p0a1] refs={refs} out={args.out}")
        log(
            f"[p0a1] ks={list(args.ks)} splits={list(args.splits)} "
            f"bootstrap={args.bootstrap_replicates} seed={args.bootstrap_seed} ci={args.bootstrap_ci}"
        )

        def log_and_tick(message: str) -> None:
            log(message)
            marker = message.split("[p0a1] ", 1)[-1].split(":", 1)[0]
            if marker in _STAGES:
                stage_bar.update(1)

        try:
            result = run_corrected(
                features_root=features,
                manifests_root=manifests,
                refs=refs,
                out_dir=args.out,
                ks=tuple(int(k) for k in args.ks),
                splits=tuple(str(name) for name in args.splits),
                bootstrap_replicates=int(args.bootstrap_replicates),
                bootstrap_seed=int(args.bootstrap_seed),
                bootstrap_ci=float(args.bootstrap_ci),
                t_min=float(args.t_min),
                t_max=float(args.t_max),
                expansion=float(args.expansion),
                on_bootstrap=on_bootstrap,
                log=log_and_tick,
            )
        finally:
            if bootstrap_bar is not None:
                bootstrap_bar.close()

    _print_summary(result)
    # reliability rows are not returned; re-read the artifact for the verdict
    reliability_rows: List[Dict[str, Any]] = []
    rel_path = Path(result["out_dir"]) / "reliability_bins_globalT.csv"
    if rel_path.exists():
        with rel_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                reliability_rows.append({
                    "eval_split": row["eval_split"], "K": int(row["K"]),
                    "variant": row["variant"], "binning": row["binning"],
                    "n": int(row["n"]), "bin_confidence": float(row["bin_confidence"]),
                    "bin_accuracy": float(row["bin_accuracy"]),
                })
    routes = _route_verdicts(
        result["selective_rows"],
        result["eaurc_bootstrap_rows"],
        result["rer_bootstrap_rows"],
        result["auroc_bootstrap_rows"],
        reliability_rows,
    )
    _print_routes(routes)
    print(f"[p0a1] wrote artifacts to {result['out_dir']} in {result['durations_seconds']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
