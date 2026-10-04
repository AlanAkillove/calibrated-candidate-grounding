"""Run the Phase 0.5 **score-information sufficiency audit** (Amendment A6).

Question: without candidate embeddings, is the *score set* of a frozen grounding
scorer already enough to predict whether its top-1 decision is correct and to
flatten the candidate-cardinality reliability degradation?

Design (frozen in ``docs/research_protocol.md`` Amendment A6 **before any
result was seen**):

* scorers: B3 independent MLP seeds 1-3 (primary) + frozen cosine (secondary);
  raw score matrices are reused from ``results/phase0b_independent`` /
  ``results/phase0a_cosine`` - no grounding scorer is retrained;
* reliability_train / reliability_tune = image-level 70/30 split of ``val_calib``
  (seed 20260928, manifest written before any fitting); training only at
  ``K in {5, 10}``; ``testA``/``testB``/``K20``/``K50`` never enter training,
  early stopping or model selection;
* model zoo: L0 scalars (msp / top1 / margin / -entropy / normalized entropy),
  L1 stats (<=17-d) x {logistic (C grid), tiny MLP (lr grid)} x {without-K,
  logK} (+ without-entropy and top-scores logistic variants), L2 ScoreDeepSets
  (phi: 1->16->16, mean/max/h_top1 pooling + z_top1, head ->32->1, +-logK);
* selection metric = mean AUROC_correct over reliability_tune K5/K10 only;
* evaluation = testA / testB / ``__pooled_test__`` at K5/10/20/50 with the
  primary metrics AUROC_correct / E-AURC / RER@50 / RER@80 and image-clustered
  paired bootstrap (5000 reps) for K5-vs-K20/K50 and the three pre-fixed
  model-vs-model pairs; every B3 seed runs independently (mean +- std across
  seeds, never pooled as ``n x 3``);
* Amendment A6.6 sufficiency gate -> ``sufficiency_gate.json``.

Usage
-----
    python -u scripts/run_phase05.py \
        --out results/phase05_score_sufficiency \
        --bootstrap-replicates 5000 --device cpu

Progress reporting: five live tqdm bars (load / features / train / metrics /
bootstrap, ``disable=None``); a timestamped log is appended to
``logs_phase05.txt``.  Exit code 0 on success; any exception propagates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TextIO

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.experiment import phase0a  # noqa: E402
from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.reliability import data as rdata  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.reliability import models as rmodels  # noqa: E402

__all__ = ["build_parser", "main", "run_phase05"]

DEFAULT_OUT = Path("results/phase05_score_sufficiency")
DEFAULT_LOG = Path("logs_phase05.txt")
DEFAULT_SCORERS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3", "cosine")
B3_SEEDS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
KS: Tuple[int, ...] = (5, 10, 20, 50)
TRAIN_KS: Tuple[int, ...] = (5, 10)
EVAL_SPLITS: Tuple[str, ...] = ("val_select", "testA", "testB")
POOLED_TEST = "__pooled_test__"
SPLIT_SEED = 20260928
TRAIN_FRAC = 0.70
MODEL_SEED = 20260928
LRS: Tuple[float, ...] = (1e-4, 3e-4, 1e-3)
CS: Tuple[float, ...] = (0.1, 1.0, 10.0)

L0_MODELS: Tuple[str, ...] = ("msp", "top1_score", "margin", "neg_entropy", "norm_entropy")
#: (name, kind, feature variant, hyper-parameter grid)
FAMILIES: Tuple[Tuple[str, str, str, Tuple[float, ...]], ...] = (
    ("stats_logistic", "logistic", "stats_logK", CS),
    ("stats_logistic_noK", "logistic", "stats_noK", CS),
    ("stats_logistic_noent", "logistic", "stats_logK_noent", CS),
    ("top_scores_logistic", "logistic", "top_scores", CS),
    ("stats_mlp", "mlp", "stats_logK", LRS),
    ("stats_mlp_noK", "mlp", "stats_noK", LRS),
    ("score_deepsets", "sds", "sds_logK", LRS),
    ("score_deepsets_noK", "sds", "sds_noK", LRS),
)
SUMMARY_FAMILIES = (
    "stats_logistic",
    "stats_logistic_noK",
    "stats_logistic_noent",
    "top_scores_logistic",
    "stats_mlp",
    "stats_mlp_noK",
)
HEADLINE_MODELS: Tuple[str, ...] = ("msp", "margin", "stats_logistic", "stats_mlp", "score_deepsets")
DUMP_MODELS: Tuple[str, ...] = ("msp", "margin", "norm_entropy", "stats_logistic", "stats_mlp", "score_deepsets")
INFO_OF: Dict[str, str] = {
    **{name: "scalar" for name in L0_MODELS},
    "stats_logistic": "summary_statistics",
    "stats_logistic_noK": "summary_statistics",
    "stats_logistic_noent": "summary_statistics",
    "top_scores_logistic": "summary_statistics",
    "stats_mlp": "summary_statistics",
    "stats_mlp_noK": "summary_statistics",
    "score_deepsets": "full_score_set",
    "score_deepsets_noK": "full_score_set",
}
PROBABILITY_MODELS = {"msp", "stats_logistic", "stats_logistic_noK", "stats_logistic_noent",
                      "top_scores_logistic", "stats_mlp", "stats_mlp_noK",
                      "score_deepsets", "score_deepsets_noK"}

METRIC_FIELDS = [
    "scorer", "model", "information", "params", "eval_split", "K", "n",
    "tune_mean_auroc", "selected_hp",
    "auroc_correct", "aurc", "aurc_oracle", "e_aurc",
    "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95",
    "ece_adaptive", "brier_binary", "nll_binary",
]
AGGREGATE_FIELDS = ["scorer_group", "model", "eval_split", "K", "metric", "mean", "std", "n_scorers"]
DEGRADATION_FIELDS = [
    "scorer", "model", "eval_split", "K_hi", "scorer_kind",
    "auroc_drop", "auroc_drop_ci_low", "auroc_drop_ci_high",
    "e_aurc_worsening", "e_aurc_ci_low", "e_aurc_ci_high",
    "rer_at_50_drop", "rer50_ci_low", "rer50_ci_high",
    "rer_at_80_drop", "e_aurc_abs_hi",
]
BOOTSTRAP_FIELDS = [
    "bootstrap_kind", "scorer", "model", "model_a", "model_b",
    "eval_split", "K_lo", "K_hi", "metric", "diff_kind",
    "diff", "ci_low", "ci_high", "mean_lo", "mean_hi", "mean_a", "mean_b",
    "worsening", "worsening_ci_low", "worsening_ci_high",
    "n", "n_clusters", "resample_unit", "n_replicates", "ci_level",
]
AGG_METRICS = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80",
               "rer_at_90", "rer_at_95", "ece_adaptive")


# ---------------------------------------------------------------------------
# logging / IO helpers
# ---------------------------------------------------------------------------
def _make_logger(log_file: Optional[Path]):
    handle: Optional[TextIO] = None
    if log_file is not None and str(log_file) != "":
        path = Path(log_file)
        try:
            handle = path.open("a", encoding="utf-8")
        except OSError:
            handle = None

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001 - console encoding is best effort
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return log, handle


def _bar(desc: str, position: int, total: int):
    return tqdm(total=int(total), desc=desc, unit="step", disable=None,
                position=position, leave=True)


# ---------------------------------------------------------------------------
# corpus loading (canonicalised so every K shares one row order per scorer)
# ---------------------------------------------------------------------------
def _load_scorer(corpus_id: str, *, b3_root: Path, cosine_root: Path,
                 log: Callable[[str], None]) -> Dict[int, Dict[str, Any]]:
    common = rdata.common_sentence_ids(b3_root=b3_root)
    per_k: Dict[int, Dict[str, Any]] = {}
    for k in KS:
        sample = rdata.load_scorer_scores(
            corpus_id, k, b3_root=b3_root, cosine_root=cosine_root, common_ids=common
        )
        order = np.argsort(sample.sentence_id, kind="stable")
        sid = sample.sentence_id[order]
        if not np.all(np.diff(sid) > 0):
            raise AssertionError(f"{corpus_id} K={k}: duplicate sentence ids")
        per_k[k] = {
            "scores": np.asarray(sample.scores, dtype=np.float64)[order],
            "sentence_id": sid,
            "ref_id": sample.ref_id[order],
            "image_id": sample.image_id[order],
            "eval_split": sample.eval_split[order],
            "target_local": sample.target_local[order],
            "correct": sample.correct[order].astype(np.float64),
        }
    ref = per_k[KS[0]]["sentence_id"]
    for k in KS[1:]:
        if not np.array_equal(per_k[k]["sentence_id"], ref):
            raise AssertionError(f"{corpus_id}: K={k} row universe differs from K={KS[0]}")
        for field in ("image_id", "eval_split", "target_local"):
            if not np.array_equal(per_k[k][field], per_k[KS[0]][field]):
                raise AssertionError(f"{corpus_id}: K={k} field {field} differs across K")
    log(f"[phase05] {corpus_id}: loaded {ref.size} common rows x K={list(KS)}")
    return per_k


def _scorer_temperature(corpus_id: str, *, b3_root: Path, corrected_root: Path,
                        log: Callable[[str], None]) -> float:
    if corpus_id.startswith("b3_seed"):
        seed = corpus_id.split("b3_seed")[1]
        payload = json.loads((b3_root / f"seed_{seed}" / "eval_metadata.json").read_text(encoding="utf-8"))
        value = float(payload["temperature_corrected"])
    else:
        payload = json.loads((corrected_root / "temperature_fit.json").read_text(encoding="utf-8"))
        node = payload.get("corrected_global_T", payload)
        value = float(node.get("temperature"))
    log(f"[phase05] {corpus_id}: corrected global T = {value:.6f}")
    return value


# ---------------------------------------------------------------------------
# feature construction (train-only normalisation, frozen protocol A6.3)
# ---------------------------------------------------------------------------
def _variant_columns(variant: str, temperature: float, k: int) -> Tuple[str, ...]:
    if variant == "stats_logK":
        return rfeat.stat_feature_names()
    if variant == "stats_noK":
        return rfeat.stat_feature_names(include_logk=False)
    if variant == "stats_logK_noent":
        return rfeat.stat_feature_names(include_logk=True, include_entropy=False)
    if variant == "top_scores":
        return ("ts_z1", "ts_z2", "ts_z3", "ts_z4", "ts_z5",
                "ts_g12", "ts_g23", "ts_g34", "ts_g45", "ts_g15",
                "ts_logk", "ts_mean", "ts_std")
    raise ValueError(variant)


def _raw_variant_matrix(variant: str, corpus: Dict[int, Dict[str, Any]], k: int,
                        temperature: float) -> np.ndarray:
    if variant.startswith("stats_"):
        names_full = rfeat.stat_feature_names()
        full = rfeat.stat_features(corpus[k]["scores"], temperature=temperature)
        wanted = _variant_columns(variant, temperature, k)
        idx = [names_full.index(name) for name in wanted]
        return full[:, idx]
    if variant == "top_scores":
        return rfeat.top_scores_features(corpus[k]["scores"])
    raise ValueError(variant)


def _rank_unit(values: np.ndarray) -> np.ndarray:
    """Monotone average-rank transform to (0, 1) for the [0, 1] metric contract.

    The raw L0 scalars (top-1 score, margin, -entropy) live on the unbounded
    score scale, while ``ccg.metrics.selective.risk_coverage_curve`` requires
    confidences in [0, 1].  Average ranks are tie-preserving and strictly
    monotone, and every reported metric (AUROC_correct / E-AURC / RER@coverage)
    depends on the confidence only through its ordering, so the transformed
    values reproduce the raw scores' metrics exactly.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_vals = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    starts = np.flatnonzero(np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1])))
    bounds = np.append(starts, values.size)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop - start > 1:  # average the ranks inside each tie group
            ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
    return (ranks - 0.5) / values.size


def _build_features(corpus_id: str, corpus: Dict[int, Dict[str, Any]], temperature: float,
                    split: rdata.ReliabilitySplit, *, max_epochs: int,
                    log: Callable[[str], None]) -> Dict[str, Any]:
    """Return scalar confidences, standardised L1 variants and packed SDS inputs."""
    n_ref = corpus[KS[0]]["sentence_id"].size
    train_mask = {k: split.row_mask(corpus[k]["eval_split"], corpus[k]["image_id"],
                                    kind="reliability_train") for k in KS}
    bundle: Dict[str, Any] = {
        "scalars": {}, "scalars_raw": {}, "variants": {}, "sds": {}, "normalisation": {},
        "scalar_rank_transform": {
            "formula": "(average_rank - 0.5) / n",
            "scope": "per (K, column) over all rows of the scorer corpus",
            "note": ("monotone + tie-preserving; all reported metrics are rank-based "
                     "(AUROC/E-AURC/RER), so transformed and raw scores give identical "
                     "metrics; raw values are preserved as scalars_raw_K* in the features npz"),
            "columns": {},
        },
    }
    for k in KS:
        scales = rfeat.scalar_confidence(corpus[k]["scores"], temperature=temperature)
        bundle["scalars_raw"][k] = {name: np.asarray(vals, dtype=np.float64)
                                    for name, vals in scales.items()}
        bundle["scalars"][k] = {}
        for name, vals in scales.items():
            vals = np.asarray(vals, dtype=np.float64)
            lo, hi = float(vals.min()), float(vals.max())
            if lo >= 0.0 and hi <= 1.0:
                bundle["scalars"][k][name] = vals
                continue
            bundle["scalars"][k][name] = _rank_unit(vals)
            entry = bundle["scalar_rank_transform"]["columns"].setdefault(
                name, {"ks": [], "raw_min": lo, "raw_max": hi})
            entry["ks"].append(int(k))
            entry["raw_min"] = min(entry["raw_min"], lo)
            entry["raw_max"] = max(entry["raw_max"], hi)

    # -- L1 variants: fit mu/sigma on the reliability_train rows of K5+K10 only
    for variant in ("stats_logK", "stats_noK", "stats_logK_noent", "top_scores"):
        raw = {k: _raw_variant_matrix(variant, corpus, k, temperature) for k in KS}
        block5 = raw[5][train_mask[5]]
        block10 = raw[10][train_mask[10]]
        combined = np.vstack([block5, block10])
        fit = rfeat.normalize_fit(combined, fit_rows=np.arange(combined.shape[0]),
                                  keys=_variant_columns(variant, temperature, 5))
        bundle["variants"][variant] = {k: rfeat.normalize_apply(raw[k], fit) for k in KS}
        bundle["normalisation"][variant] = fit.to_dict()

    # -- SDS packing: one global score standardiser + per-row logK / top1 columns
    mu_s, sigma_s = rfeat.entry_moments([corpus[5]["scores"][train_mask[5]],
                                         corpus[10]["scores"][train_mask[10]]])
    eps = 1e-8
    sds_z = {k: ((corpus[k]["scores"] - mu_s) / (sigma_s + eps)).astype(np.float32) for k in KS}
    lnk_vals = {k: np.full(n_ref, np.log(k), dtype=np.float64) for k in KS}
    lnk_train = np.concatenate(
        [lnk_vals[k][train_mask[k]] for k in TRAIN_KS], axis=0
    )[:, None]
    lnk_fit = rfeat.normalize_fit(
        lnk_train,
        fit_rows=np.arange(lnk_train.shape[0]),
        keys=("log_k_std",))
    lnk_std = {k: rfeat.normalize_apply(lnk_vals[k][:, None], lnk_fit)[:, 0].astype(np.float32)
               for k in KS}
    top1 = {k: rfeat.top1_column(corpus[k]["scores"]) for k in KS}
    top1_fit = rfeat.normalize_fit(
        np.concatenate([top1[5][train_mask[5]][:, None], top1[10][train_mask[10]][:, None]], axis=0),
        fit_rows=np.arange(int(train_mask[5].sum()) + int(train_mask[10].sum())),
        keys=("z_top1_std",))
    ztop1_std = {k: rfeat.normalize_apply(top1[k][:, None], top1_fit)[:, 0].astype(np.float32)
                 for k in KS}

    def pack(rows_by_k: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        mats = [sds_z[k][rows_by_k[k]] for k in TRAIN_KS]
        padded, mask = rfeat.sds_pack(mats)
        log_k = np.concatenate([lnk_std[k][rows_by_k[k]] for k in TRAIN_KS])
        zt = np.concatenate([ztop1_std[k][rows_by_k[k]] for k in TRAIN_KS])
        y = np.concatenate([corpus[k]["correct"][rows_by_k[k]] for k in TRAIN_KS])
        return {"scores": padded, "mask": mask, "log_k": log_k, "z_top1": zt, "y": y}

    tune_mask = {k: split.row_mask(corpus[k]["eval_split"], corpus[k]["image_id"],
                                   kind="reliability_tune") for k in KS}
    sds: Dict[str, Any] = {
        "train": pack({k: np.flatnonzero(train_mask[k]) for k in TRAIN_KS}),
        "tune": pack({k: np.flatnonzero(tune_mask[k]) for k in TRAIN_KS}),
        "n_tune5": int(tune_mask[5].sum()),
        "eval": {},
    }
    for k in KS:
        for split_name in EVAL_SPLITS:
            m = corpus[k]["eval_split"] == split_name
            padded, mask = rfeat.sds_pack([sds_z[k][m]])
            sds["eval"][(k, split_name)] = {
                "scores": padded, "mask": mask,
                "log_k": lnk_std[k][m], "z_top1": ztop1_std[k][m],
            }
    sds["normalisation"] = {"score_mu": float(mu_s), "score_sigma": float(sigma_s),
                            "log_k_fit": lnk_fit.to_dict(), "top1_fit": top1_fit.to_dict()}
    bundle["sds"] = sds
    bundle["train_mask"] = train_mask
    bundle["tune_mask"] = tune_mask
    log(f"[phase05] {corpus_id}: features built "
        f"(train rows K5={int(train_mask[5].sum())}/K10={int(train_mask[10].sum())}, "
        f"tune rows K5={int(tune_mask[5].sum())}/K10={int(tune_mask[10].sum())})")
    return bundle


def _combined_l1(bundle: Dict[str, Any], corpus: Dict[int, Dict[str, Any]],
                 variant: str, mask: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mats = [bundle["variants"][variant][k][mask[k]] for k in TRAIN_KS]
    ys = [corpus[k]["correct"][mask[k]] for k in TRAIN_KS]
    return np.vstack(mats), np.concatenate(ys)


# ---------------------------------------------------------------------------
# model training / selection (reliability_tune K5/K10 mean AUROC only)
# ---------------------------------------------------------------------------
def _mean_auroc_two_k(pred: np.ndarray, y: np.ndarray, n5: int) -> float:
    return 0.5 * (auroc_correct(pred[:n5], y[:n5]) + auroc_correct(pred[n5:], y[n5:]))


def _train_families(corpus_id: str, corpus: Dict[int, Dict[str, Any]], bundle: Dict[str, Any],
                    *, max_epochs: int, log: Callable[[str], None]) -> Dict[str, Any]:
    Xtr, ytr = {}, {}
    for variant in ("stats_logK", "stats_noK", "stats_logK_noent", "top_scores"):
        Xtr[variant], ytr[variant] = _combined_l1(bundle, corpus, variant, bundle["train_mask"])
    Xtu, ytu = {}, {}
    for variant in ("stats_logK", "stats_noK", "stats_logK_noent", "top_scores"):
        Xtu[variant], ytu[variant] = _combined_l1(bundle, corpus, variant, bundle["tune_mask"])
    n_tune5 = bundle["sds"]["n_tune5"]
    sds_train, sds_tune = bundle["sds"]["train"], bundle["sds"]["tune"]

    results: Dict[str, Any] = {}
    for family, kind, variant, grid in FAMILIES:
        best: Optional[Dict[str, Any]] = None
        for hp in grid:
            if kind == "logistic":
                model = rmodels.LogisticModel(C=float(hp))
                model.fit(Xtr[variant], ytr[variant])
                pred = model.predict_proba(Xtu[variant])
            elif kind == "mlp":
                model = rmodels.TinyMLP(Xtr[variant].shape[1], seed=MODEL_SEED)
                info = model.fit(Xtr[variant], ytr[variant], lr=float(hp),
                                 X_val=Xtu[variant], y_val=ytu[variant],
                                 epochs=max_epochs)
                pred = model.predict_proba(Xtu[variant])
            else:
                model = rmodels.ScoreDeepSets(seed=MODEL_SEED, include_logk=(variant == "sds_logK"))
                info = model.fit(sds_train["scores"], sds_train["mask"], sds_train["log_k"],
                                 sds_train["z_top1"], sds_train["y"], lr=float(hp),
                                 scores_val=sds_tune["scores"], mask_val=sds_tune["mask"],
                                 log_k_val=sds_tune["log_k"], z_top1_val=sds_tune["z_top1"],
                                 y_val=sds_tune["y"], epochs=max_epochs)
                pred = model.predict_proba(sds_tune["scores"], sds_tune["mask"],
                                           sds_tune["log_k"], sds_tune["z_top1"])
            y_tune = sds_tune["y"] if kind == "sds" else ytu[variant]
            score = _mean_auroc_two_k(pred, y_tune, n_tune5)
            if best is None or score > best["tune_mean_auroc"]:
                best = {"family": family, "kind": kind, "variant": variant,
                        "chosen_hp": float(hp), "tune_mean_auroc": float(score),
                        "model": model,
                        "params": int(model.n_parameters()),
                        "fit_info": info if kind != "logistic" else {}}
        results[family] = best
    log("[phase05] " + corpus_id + ": selected " + ", ".join(
        f"{family}={results[family]['chosen_hp']}(tune {results[family]['tune_mean_auroc']:.4f})"
        for family, _, _, _ in FAMILIES))
    return results


def _family_predictions(corpus: Dict[int, Dict[str, Any]], bundle: Dict[str, Any],
                        selection: Dict[str, Any], family: str) -> Dict[Tuple[int, str], np.ndarray]:
    """P(y_correct) per (K, eval split) for a selected family instance."""
    entry = selection[family]
    kind, variant, model = entry["kind"], entry["variant"], entry["model"]
    out: Dict[Tuple[int, str], np.ndarray] = {}
    for k in KS:
        for split_name in EVAL_SPLITS:
            mask = corpus[k]["eval_split"] == split_name
            if kind == "logistic" or kind == "mlp":
                out[(k, split_name)] = model.predict_proba(bundle["variants"][variant][k][mask])
            else:
                payload = bundle["sds"]["eval"][(k, split_name)]
                out[(k, split_name)] = model.predict_proba(
                    payload["scores"], payload["mask"], payload["log_k"], payload["z_top1"])
    return out


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def run_phase05(
    *,
    out_dir: Path,
    scorers: Sequence[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ci: float,
    split_seed: int,
    train_frac: float,
    max_epochs: int,
    b3_root: Path,
    cosine_root: Path,
    corrected_root: Path,
    log_file: Optional[Path],
    figure: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    log, handle = _make_logger(log_file)
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    durations: Dict[str, float] = {}

    def stage(name: str, t0: float) -> None:
        durations[name] = float(time.perf_counter() - t0)
        log(f"[phase05] stage {name}: {durations[name]:.2f}s")

    try:
        log(f"[phase05] out       : {out_dir}")
        log(f"[phase05] scorers   : {list(scorers)}")
        log(f"[phase05] bootstrap : {bootstrap_replicates} reps, seed={bootstrap_seed}, ci={ci}")

        # -- stage 1: load frozen score matrices ---------------------------
        t0 = time.perf_counter()
        corpora: Dict[str, Dict[int, Dict[str, Any]]] = {}
        temperatures: Dict[str, float] = {}
        bar = _bar("load", 0, len(scorers))
        for corpus_id in scorers:
            corpora[corpus_id] = _load_scorer(corpus_id, b3_root=b3_root,
                                              cosine_root=cosine_root, log=log)
            temperatures[corpus_id] = _scorer_temperature(corpus_id, b3_root=b3_root,
                                                          corrected_root=corrected_root, log=log)
            bar.update(1)
        bar.close()
        stage("load", t0)

        # -- stage 2: freeze the reliability split -------------------------
        t0 = time.perf_counter()
        anchor = corpora[scorers[0]][KS[0]]
        split = rdata.build_reliability_split(anchor["eval_split"], anchor["image_id"],
                                              seed=split_seed, train_frac=train_frac)
        manifest = rdata.write_split_manifest(
            out_dir / "split_manifest.json", split,
            extra={"source": "val_calib images of the frozen common cohort",
                   "train_ks": list(TRAIN_KS), "scorers": list(scorers)})
        log(f"[phase05] split frozen: {len(split.train_images)} train / "
            f"{len(split.tune_images)} tune images (seed={split_seed}, frac={train_frac})")
        stage("split", t0)

        # -- stage 3: features ---------------------------------------------
        t0 = time.perf_counter()
        bundles: Dict[str, Dict[str, Any]] = {}
        bar = _bar("features", 1, len(scorers))
        for corpus_id in scorers:
            bundles[corpus_id] = _build_features(corpus_id, corpora[corpus_id],
                                                 temperatures[corpus_id], split,
                                                 max_epochs=max_epochs, log=log)
            _dump_features(out_dir, corpus_id, corpora[corpus_id], bundles[corpus_id])
            bar.update(1)
        bar.close()
        stage("features", t0)

        # -- stage 4: train + select the model zoo --------------------------
        t0 = time.perf_counter()
        selections: Dict[str, Dict[str, Any]] = {}
        predictions: Dict[str, Dict[str, Dict[Tuple[int, str], np.ndarray]]] = {}
        for corpus_id in scorers:
            log(f"[phase05] {corpus_id}: training {len(FAMILIES)} families "
                f"(<= {max_epochs} epochs, patience 30)")
            selections[corpus_id] = _train_families(corpus_id, corpora[corpus_id],
                                                    bundles[corpus_id],
                                                    max_epochs=max_epochs, log=log)
            predictions[corpus_id] = {
                family: _family_predictions(corpora[corpus_id], bundles[corpus_id],
                                            selections[corpus_id], family)
                for family, _, _, _ in FAMILIES
            }
        stage("train", t0)

        # -- stage 5: selection bookkeeping (tune scores) ------------------
        tune_scores = _tune_scores(scorers, corpora, bundles, selections, predictions)
        best = _pick_best(scorers, tune_scores, log)

        # -- stage 6: point metrics (L0 + all families) ---------------------
        t0 = time.perf_counter()
        metric_rows = _point_metric_rows(scorers, corpora, bundles, selections, predictions,
                                         tune_scores, log)
        stage("point_metrics", t0)

        # -- stage 7: bootstrap --------------------------------------------
        t0 = time.perf_counter()
        boot_fields = _bootstrap_all(scorers, corpora, bundles, selections, predictions, best,
                                     replicates=bootstrap_replicates, seed=bootstrap_seed,
                                     ci=ci, log=log)
        stage("bootstrap", t0)

        # -- stage 8: aggregate + degradation + gate ------------------------
        t0 = time.perf_counter()
        agg_rows = _aggregate_rows(scorers, metric_rows)
        degrade_rows = _degradation_rows(scorers, metric_rows, boot_fields["cross_k"])
        gate = _gate(scorers, metric_rows, boot_fields, best, log)
        stage("aggregate_gate", t0)

        # -- stage 9: figure ------------------------------------------------
        if figure:
            t0 = time.perf_counter()
            try:
                _figure(out_dir, scorers, metric_rows, best)
                stage("figure", t0)
            except Exception as exc:  # noqa: BLE001 - figure is best-effort
                log(f"[phase05] figure skipped: {type(exc).__name__}: {exc}")

        # -- stage 10: artifacts --------------------------------------------
        t0 = time.perf_counter()
        _write_artifacts(out_dir, scorers, corpora, bundles, selections, predictions,
                         boot_fields, metric_rows, agg_rows, degrade_rows, gate, best,
                         temperatures, split, manifest, durations, bootstrap_replicates,
                         bootstrap_seed, ci, max_epochs)
        stage("write_artifacts", t0)

        total = time.perf_counter() - started
        log(f"[phase05] total: {total:.2f}s -> {out_dir}")
        result = {"out_dir": str(out_dir), "n_metric_rows": len(metric_rows),
                  "n_degradation_rows": len(degrade_rows), "verdict": gate["verdict_seed_mean"]["verdict"],
                  "best_family_b3": best["b3"]["family"], "total_seconds": float(total)}
    finally:
        if handle is not None:
            handle.close()
    return result


def _dump_features(out_dir: Path, corpus_id: str, corpus: Dict[int, Dict[str, Any]],
                   bundle: Dict[str, Any]) -> None:
    payload: Dict[str, np.ndarray] = {"scalar_names": np.asarray(L0_MODELS)}
    for k in KS:
        payload[f"correct_K{k}"] = corpus[k]["correct"].astype(np.float32)
        payload[f"eval_split_K{k}"] = corpus[k]["eval_split"]
        payload[f"image_id_K{k}"] = corpus[k]["image_id"]
        payload[f"ref_id_K{k}"] = corpus[k]["ref_id"]
        payload[f"sentence_id_K{k}"] = corpus[k]["sentence_id"]
        scales = bundle["scalars"][k]
        raw = bundle["scalars_raw"][k]
        payload[f"scalars_K{k}"] = np.stack([scales[name] for name in L0_MODELS], axis=1).astype(np.float32)
        payload[f"scalars_raw_K{k}"] = np.stack([raw[name] for name in L0_MODELS], axis=1).astype(np.float32)
    np.savez_compressed(out_dir / "features" / f"{corpus_id}.npz", **payload)
    norm = {"variants": bundle["normalisation"], "sds": bundle["sds"]["normalisation"],
            "scalar_rank_transform": bundle["scalar_rank_transform"]}
    phase0a._write_json(out_dir / "features" / f"{corpus_id}_normalisation.json", norm)


def _point_metric_rows(scorers, corpora, bundles, selections, predictions, tune_scores, log) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_cells = len(scorers) * (len(L0_MODELS) + len(FAMILIES)) * (len(EVAL_SPLITS) + 1) * len(KS)
    bar = _bar("metrics", 3, n_cells)
    for corpus_id in scorers:
        corpus = corpora[corpus_id]
        for model in list(L0_MODELS) + [f[0] for f in FAMILIES]:
            for k in KS:
                for split_name in EVAL_SPLITS:
                    mask = corpus[k]["eval_split"] == split_name
                    rows.append(_metric_row(corpus_id, model, corpus, bundles, predictions,
                                            selections, tune_scores, k, split_name, mask))
                    bar.update(1)
                mask = np.isin(corpus[k]["eval_split"], ("testA", "testB"))
                rows.append(_metric_row(corpus_id, model, corpus, bundles, predictions,
                                        selections, tune_scores, k, POOLED_TEST, mask))
                bar.update(1)
    bar.close()
    log(f"[phase05] point metrics: {len(rows)} rows")
    return rows


def _conf_pooled(corpus_id, model, corpus, bundles, predictions, k):
    """Confidence (+probability) for `__pooled_test__` = testA followed by testB."""
    parts_conf, parts_prob = [], []
    for split_name in ("testA", "testB"):
        mask = corpus[k]["eval_split"] == split_name
        conf, prob = _confidence_of(corpus_id, model, corpus, bundles, predictions,
                                    k, split_name, mask)
        parts_conf.append(conf)
        if prob is not None:
            parts_prob.append(prob)
    conf_all = np.concatenate(parts_conf)
    prob_all = np.concatenate(parts_prob) if len(parts_prob) == 2 else None
    return conf_all, prob_all


def _correct_pooled(corpus, k):
    return np.concatenate([corpus[k]["correct"][corpus[k]["eval_split"] == name]
                           for name in ("testA", "testB")])


def _confidence_of(corpus_id, model, corpus, bundles, predictions, k, split_name, mask):
    if model in L0_MODELS:
        conf = bundles[corpus_id]["scalars"][k][model][mask]
        return conf, (conf if model == "msp" else None)
    pred = predictions[corpus_id][model][(k, split_name)]
    return pred, (pred if model in PROBABILITY_MODELS else None)


def _metric_row(corpus_id, model, corpus, bundles, predictions, selections, tune_scores,
                k, split_name, mask):
    if split_name == POOLED_TEST:
        conf, prob = _conf_pooled(corpus_id, model, corpus, bundles, predictions, k)
        correct = _correct_pooled(corpus, k)
    else:
        correct = corpus[k]["correct"][mask]
        conf, prob = _confidence_of(corpus_id, model, corpus, bundles, predictions,
                                    k, split_name, mask)
    metrics = reval.point_metric_row(conf, correct, probability=prob)
    entry = selections[corpus_id].get(model, {})
    row = {
        "scorer": corpus_id, "model": model, "information": INFO_OF[model],
        "params": int(entry.get("params", 0)), "eval_split": split_name, "K": int(k),
        "n": int(correct.size),
        "tune_mean_auroc": float(tune_scores[corpus_id][model]),
        "selected_hp": entry.get("chosen_hp", "") if entry else "",
    }
    row.update(metrics)
    return row


def _tune_scores(scorers, corpora, bundles, selections, predictions) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for corpus_id in scorers:
        corpus = corpora[corpus_id]
        per_model: Dict[str, float] = {}
        for name in L0_MODELS:
            vals = []
            for k in TRAIN_KS:
                mask = bundles[corpus_id]["tune_mask"][k]
                vals.append(auroc_correct(bundles[corpus_id]["scalars"][k][name][mask],
                                          corpus[k]["correct"][mask]))
            per_model[name] = float(np.mean(vals))
        for family, _, _, _ in FAMILIES:
            per_model[family] = float(selections[corpus_id][family]["tune_mean_auroc"])
        scores[corpus_id] = per_model
    return scores


def _pick_best(scorers, tune_scores, log) -> Dict[str, Any]:
    # family chosen by the MEAN tune AUROC across the B3 seeds (A6.6)
    scorers_b3_mean = {name: float(np.mean([tune_scores[s][name] for s in B3_SEEDS if s in tune_scores]))
                       for name in list(L0_MODELS) + [f[0] for f in FAMILIES]}
    all_names = list(L0_MODELS) + [f[0] for f in FAMILIES]
    best_family_b3 = max(all_names, key=lambda name: scorers_b3_mean[name])
    best_summary_b3 = max(SUMMARY_FAMILIES, key=lambda name: scorers_b3_mean[name])
    out = {
        "b3": {"family": best_family_b3, "summary": best_summary_b3,
               "tune_scores_mean": scorers_b3_mean},
        "per_scorer": {s: {"family": max(all_names, key=lambda n: tune_scores[s][n]),
                           "summary": max(SUMMARY_FAMILIES, key=lambda n: tune_scores[s][n])}
                       for s in scorers},
    }
    log(f"[phase05] best family (B3 seed-mean tune AUROC): {best_family_b3} "
        f"({scorers_b3_mean[best_family_b3]:.4f}); best summary: {best_summary_b3}")
    return out


def _bootstrap_all(scorers, corpora, bundles, selections, predictions, best, *,
                   replicates, seed, ci, log) -> Dict[str, List[Dict[str, Any]]]:
    cross_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    n_calls = 0
    for corpus_id in scorers:
        models = set(HEADLINE_MODELS)
        models.add(best["per_scorer"][corpus_id]["family"])
        models.add(best["per_scorer"][corpus_id]["summary"])
        n_calls += len(models) * 2 * 3
    n_calls += len(scorers) * 3 * 2  # three pairwise comparisons at K=5 and K=50
    bar = _bar("bootstrap", 4, n_calls)

    def conf_for(corpus_id, model, k, split_name):
        corpus = corpora[corpus_id]
        if split_name == POOLED_TEST:
            return _conf_pooled(corpus_id, model, corpus, bundles, predictions, k)[0]
        mask = corpus[k]["eval_split"] == split_name
        return _confidence_of(corpus_id, model, corpus, bundles, predictions, k,
                              split_name, mask)[0]

    def correct_for(corpus_id, k, split_name):
        corpus = corpora[corpus_id]
        if split_name == POOLED_TEST:
            return _correct_pooled(corpus, k)
        return corpus[k]["correct"][corpus[k]["eval_split"] == split_name]

    def clusters_for(corpus_id, split_name):
        corpus = corpora[corpus_id]
        if split_name == POOLED_TEST:
            return np.concatenate([corpus[5]["image_id"][corpus[5]["eval_split"] == name]
                                   for name in ("testA", "testB")])
        return corpus[5]["image_id"][corpus[5]["eval_split"] == split_name]

    for corpus_id in scorers:
        models = sorted(set(HEADLINE_MODELS)
                        | {best["per_scorer"][corpus_id]["family"],
                           best["per_scorer"][corpus_id]["summary"]})
        for model in models:
            for k_hi in (20, 50):
                for split_name in ("testA", "testB", POOLED_TEST):
                    rows = reval.cross_k_bootstrap_row(
                        conf_for(corpus_id, model, 5, split_name), correct_for(corpus_id, 5, split_name),
                        conf_for(corpus_id, model, k_hi, split_name), correct_for(corpus_id, k_hi, split_name),
                        clusters_for(corpus_id, split_name),
                        eval_split=split_name, k_lo=5, k_hi=k_hi,
                        replicates=int(replicates), seed=int(seed), ci=float(ci))
                    for row in rows:
                        row = dict(row)
                        row["scorer"] = corpus_id
                        row["model"] = model
                        cross_rows.append(row)
                    bar.update(1)
                    bar.set_postfix_str(f"{corpus_id} {model} K5vK{k_hi} {split_name}")
        pairs = (("msp", "stats_logistic"),
                 ("stats_logistic", "stats_mlp"),
                 (best["per_scorer"][corpus_id]["summary"], "score_deepsets"))
        for model_a, model_b in pairs:
            for k in (5, 50):
                rows = reval.model_vs_model_bootstrap_row(
                    conf_for(corpus_id, model_a, k, POOLED_TEST), correct_for(corpus_id, k, POOLED_TEST),
                    conf_for(corpus_id, model_b, k, POOLED_TEST), correct_for(corpus_id, k, POOLED_TEST),
                    clusters_for(corpus_id, POOLED_TEST),
                    eval_split=POOLED_TEST, K=k, model_a=model_a, model_b=model_b,
                    replicates=int(replicates), seed=int(seed), ci=float(ci))
                for row in rows:
                    row = dict(row)
                    row["scorer"] = corpus_id
                    row["model"] = ""
                    pair_rows.append(row)
                bar.update(1)
    bar.close()
    log(f"[phase05] bootstrap: {len(cross_rows)} cross-K rows, {len(pair_rows)} pairwise rows")
    return {"cross_k": cross_rows, "pairwise": pair_rows}


def _aggregate_rows(scorers, metric_rows) -> List[Dict[str, Any]]:
    from collections import defaultdict
    buckets: Dict[Tuple[str, str, str, int, str], List[float]] = defaultdict(list)
    for row in metric_rows:
        group = "b3" if row["scorer"].startswith("b3_") else "cosine"
        for metric in AGG_METRICS:
            value = row.get(metric)
            if value in (None, ""):
                continue
            buckets[(group, row["model"], row["eval_split"], int(row["K"]), metric)].append(float(value))
    rows = []
    for (group, model, split_name, k, metric), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        rows.append({
            "scorer_group": group, "model": model, "eval_split": split_name, "K": int(k),
            "metric": metric, "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n_scorers": int(arr.size),
        })
    return rows


def _degradation_rows(scorers, metric_rows, cross_rows) -> List[Dict[str, Any]]:
    point: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {
        (row["scorer"], row["model"], row["eval_split"], int(row["K"])): row for row in metric_rows
    }
    boot: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for row in cross_rows:
        key = (row["scorer"], row["model"], row["eval_split"], int(row["K_hi"]))
        entry = boot.setdefault(key, {})
        if row["metric"] == "auroc_correct" and row["diff_kind"] == "absolute":
            entry.update({"auroc_drop": row["diff"], "auroc_drop_ci_low": row["ci_low"],
                          "auroc_drop_ci_high": row["ci_high"]})
        elif row["metric"] == "e_aurc" and row["diff_kind"] == "relative":
            entry.update({"e_aurc_worsening": row.get("worsening"),
                          "e_aurc_ci_low": row.get("worsening_ci_low"),
                          "e_aurc_ci_high": row.get("worsening_ci_high")})
        elif row["metric"] == "rer_at_50" and row["diff_kind"] == "absolute":
            entry.update({"rer_at_50_drop": row["diff"], "rer50_ci_low": row["ci_low"],
                          "rer50_ci_high": row["ci_high"]})
        elif row["metric"] == "rer_at_80" and row["diff_kind"] == "absolute":
            entry.update({"rer_at_80_drop": row["diff"]})
    rows: List[Dict[str, Any]] = []
    for corpus_id in scorers:
        models = sorted({row["model"] for row in metric_rows if row["scorer"] == corpus_id})
        for model in models:
            for split_name in ("testA", "testB", POOLED_TEST):
                for k_hi in (20, 50):
                    p5 = point.get((corpus_id, model, split_name, 5), {})
                    p_hi = point.get((corpus_id, model, split_name, k_hi), {})
                    entry = boot.get((corpus_id, model, split_name, k_hi), {})
                    def drop(metric):
                        a, b = p5.get(metric), p_hi.get(metric)
                        return float(a - b) if a is not None and b is not None else None
                    row = {
                        "scorer": corpus_id, "model": model, "eval_split": split_name,
                        "K_hi": int(k_hi),
                        "scorer_kind": "bootstrap" if entry else "point",
                        "auroc_drop": entry.get("auroc_drop", drop("auroc_correct")),
                        "auroc_drop_ci_low": entry.get("auroc_drop_ci_low"),
                        "auroc_drop_ci_high": entry.get("auroc_drop_ci_high"),
                        "e_aurc_worsening": entry.get("e_aurc_worsening"),
                        "e_aurc_ci_low": entry.get("e_aurc_ci_low"),
                        "e_aurc_ci_high": entry.get("e_aurc_ci_high"),
                        "rer_at_50_drop": entry.get("rer_at_50_drop", drop("rer_at_50")),
                        "rer50_ci_low": entry.get("rer50_ci_low"),
                        "rer50_ci_high": entry.get("rer50_ci_high"),
                        "rer_at_80_drop": entry.get("rer_at_80_drop", drop("rer_at_80")),
                        "e_aurc_abs_hi": p_hi.get("e_aurc"),
                    }
                    rows.append(row)
    # seed-mean rows over the three B3 seeds for the headline models
    for model in sorted({row["model"] for row in rows}):
        for split_name in ("testA", "testB", POOLED_TEST):
            for k_hi in (20, 50):
                sel = [r for r in rows if r["scorer"] in B3_SEEDS and r["model"] == model
                       and r["eval_split"] == split_name and r["K_hi"] == k_hi]
                if len(sel) != len(B3_SEEDS):
                    continue
                mean_row = {"scorer": "b3_mean", "model": model, "eval_split": split_name,
                            "K_hi": int(k_hi), "scorer_kind": sel[0]["scorer_kind"]}
                for field in DEGRADATION_FIELDS[5:]:
                    vals = [r[field] for r in sel if r.get(field) is not None]
                    mean_row[field] = float(np.mean(vals)) if vals else None
                rows.append(mean_row)
    return rows


def _rer_pp(value) -> Optional[float]:
    """RER@50 drops are computed as fractions; the A6.6 gate compares percentage points."""
    return None if value is None else 100.0 * float(value)


def _gate(scorers, metric_rows, boot_fields, best, log) -> Dict[str, Any]:
    family = best["b3"]["family"]
    point = {(row["scorer"], row["model"], row["eval_split"], int(row["K"])): row
             for row in metric_rows}
    degrade = _degradation_rows(scorers, metric_rows, boot_fields["cross_k"])
    b3mean = {(row["model"], row["eval_split"], int(row["K_hi"])): row for row in degrade
              if row["scorer"] == "b3_mean"}

    def cell_from(key: Tuple[str, int]):
        split_name, k_hi = key
        mean = b3mean.get((family, split_name, k_hi))
        if mean is None:
            return None
        return {"split": split_name, "K": int(k_hi),
                "auroc_drop": mean["auroc_drop"], "auroc_drop_ci_low": mean["auroc_drop_ci_low"],
                "auroc_drop_ci_high": mean["auroc_drop_ci_high"],
                "e_aurc_worsening": mean["e_aurc_worsening"], "e_aurc_ci_low": mean["e_aurc_ci_low"],
                "e_aurc_ci_high": mean["e_aurc_ci_high"],
                "rer50_drop": _rer_pp(mean["rer_at_50_drop"]),
                "rer50_ci_low": _rer_pp(mean["rer50_ci_low"]),
                "rer50_ci_high": _rer_pp(mean["rer50_ci_high"]),
                "e_aurc_abs": mean["e_aurc_abs_hi"]}

    cells = [cell_from((s, k)) for s in ("testA", "testB") for k in (20, 50)]
    cells = [c for c in cells if c is not None]
    pooled_cell = cell_from((POOLED_TEST, 50))
    all_cells = cells + ([pooled_cell] if pooled_cell else [])
    k50_vals = [point[(s, family, POOLED_TEST, 50)] for s in B3_SEEDS
                if (s, family, POOLED_TEST, 50) in point]  # smoke may omit seeds
    k50_point = {"auroc": float(np.mean([v["auroc_correct"] for v in k50_vals])),
                 "e_aurc": float(np.mean([v["e_aurc"] for v in k50_vals])),
                 "rer_at_50": float(np.mean([v["rer_at_50"] for v in k50_vals]))}
    verdict_mean = reval.sufficiency_verdict(all_cells, k50_point=k50_point)
    per_seed = {}
    seed_consistency = {}
    for seed in [s for s in B3_SEEDS if (s, family, POOLED_TEST, 50) in point]:
        seed_cells = []
        for split_name, k_hi in [(s, k) for s in ("testA", "testB") for k in (20, 50)]:
            rows = {(row["eval_split"], int(row["K_hi"])): row for row in degrade
                    if row["scorer"] == seed and row["model"] == family}
            row = rows.get((split_name, k_hi))
            if row is None:
                continue
            seed_cells.append({"split": split_name, "K": int(k_hi),
                               "auroc_drop": row["auroc_drop"],
                               "auroc_drop_ci_low": row["auroc_drop_ci_low"],
                               "auroc_drop_ci_high": row["auroc_drop_ci_high"],
                               "e_aurc_worsening": row["e_aurc_worsening"],
                               "e_aurc_ci_low": row["e_aurc_ci_low"],
                               "e_aurc_ci_high": row["e_aurc_ci_high"],
                               "rer50_drop": _rer_pp(row["rer_at_50_drop"]),
                               "rer50_ci_low": _rer_pp(row["rer50_ci_low"]),
                               "rer50_ci_high": _rer_pp(row["rer50_ci_high"]),
                               "e_aurc_abs": row["e_aurc_abs_hi"]})
        seed_pooled = None
        for row in degrade:
            if (row["scorer"] == seed and row["model"] == family
                    and row["eval_split"] == POOLED_TEST and int(row["K_hi"]) == 50):
                seed_pooled = {"split": POOLED_TEST, "K": 50,
                               "auroc_drop": row["auroc_drop"],
                               "auroc_drop_ci_low": row["auroc_drop_ci_low"],
                               "auroc_drop_ci_high": row["auroc_drop_ci_high"],
                               "e_aurc_worsening": row["e_aurc_worsening"],
                               "e_aurc_ci_low": row["e_aurc_ci_low"],
                               "e_aurc_ci_high": row["e_aurc_ci_high"],
                               "rer50_drop": _rer_pp(row["rer_at_50_drop"]),
                               "rer50_ci_low": _rer_pp(row["rer50_ci_low"]),
                               "rer50_ci_high": _rer_pp(row["rer50_ci_high"]),
                               "e_aurc_abs": row["e_aurc_abs_hi"]}
        k50_seed = point[(seed, family, POOLED_TEST, 50)]
        per_seed[seed] = reval.sufficiency_verdict(
            seed_cells + ([seed_pooled] if seed_pooled else []),
            k50_point={"auroc": k50_seed["auroc_correct"], "e_aurc": k50_seed["e_aurc"],
                       "rer_at_50": k50_seed["rer_at_50"]})
        seed_consistency[seed] = {"n_failing_ood_cells": per_seed[seed]["n_failing_ood_cells"]}
    cosine_verdict = None
    if "cosine" in scorers:
        cos_family = best["per_scorer"]["cosine"]["family"]
        cos_degrade = {(row["eval_split"], int(row["K_hi"])): row for row in degrade
                       if row["scorer"] == "cosine" and row["model"] == cos_family}
        cos_cells = []
        for split_name, k_hi in [(s, k) for s in ("testA", "testB") for k in (20, 50)]:
            row = cos_degrade.get((split_name, k_hi))
            if row is None or row["e_aurc_worsening"] is None:
                continue
            cos_cells.append({"split": split_name, "K": int(k_hi),
                              "auroc_drop": row["auroc_drop"],
                              "auroc_drop_ci_low": row["auroc_drop_ci_low"],
                              "auroc_drop_ci_high": row["auroc_drop_ci_high"],
                              "e_aurc_worsening": row["e_aurc_worsening"],
                              "e_aurc_ci_low": row["e_aurc_ci_low"],
                              "e_aurc_ci_high": row["e_aurc_ci_high"],
                              "rer50_drop": _rer_pp(row["rer_at_50_drop"]),
                              "rer50_ci_low": _rer_pp(row["rer50_ci_low"]),
                              "rer50_ci_high": _rer_pp(row["rer50_ci_high"]),
                              "e_aurc_abs": row["e_aurc_abs_hi"]})
        row = cos_degrade.get((POOLED_TEST, 50))
        cos_seed = point[("cosine", cos_family, POOLED_TEST, 50)]
        if row is not None:
            cos_cells.append({"split": POOLED_TEST, "K": 50,
                              "auroc_drop": row["auroc_drop"],
                              "auroc_drop_ci_low": row["auroc_drop_ci_low"],
                              "auroc_drop_ci_high": row["auroc_drop_ci_high"],
                              "e_aurc_worsening": row["e_aurc_worsening"],
                              "e_aurc_ci_low": row["e_aurc_ci_low"],
                              "e_aurc_ci_high": row["e_aurc_ci_high"],
                              "rer50_drop": _rer_pp(row["rer_at_50_drop"]),
                              "rer50_ci_low": _rer_pp(row["rer50_ci_low"]),
                              "rer50_ci_high": _rer_pp(row["rer50_ci_high"]),
                              "e_aurc_abs": row["e_aurc_abs_hi"]})
        cosine_verdict = {"family": cos_family,
                          "verdict": reval.sufficiency_verdict(
                              cos_cells, k50_point={"auroc": cos_seed["auroc_correct"],
                                                    "e_aurc": cos_seed["e_aurc"],
                                                    "rer_at_50": cos_seed["rer_at_50"]})}
    gate = {
        "best_family_b3": family,
        "best_summary_b3": best["b3"]["summary"],
        "per_scorer_best": best["per_scorer"],
        "tune_scores_b3_mean": best["b3"]["tune_scores_mean"],
        "cells_seed_mean": all_cells,
        "verdict_seed_mean": verdict_mean,
        "verdict_per_seed": {seed: per_seed[seed] for seed in per_seed},
        "k50_point_seed_mean": k50_point,
        "cosine": cosine_verdict,
        "notes": [
            "headline verdict = Amendment A6.6 applied to seed-mean cells; per-seed verdicts "
            "reported alongside (A6 requires >=2/3 seed consistency to count a cell).",
            "selection metric = mean AUROC_correct over reliability_tune K5/K10 only.",
            "__pooled_test__ = testA + testB (val_select/val_calib never evaluated here).",
        ],
    }
    log(f"[phase05] gate (seed-mean): verdict={verdict_mean['verdict']} "
        f"failing_ood_cells={verdict_mean['n_failing_ood_cells']}")
    return gate


def _figure(out_dir: Path, scorers, metric_rows, best) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = ("msp", "margin", "stats_logistic", "stats_mlp", "score_deepsets")
    labels = {"msp": "MSP", "margin": "Margin", "stats_logistic": "Stats Logistic",
              "stats_mlp": "Stats MLP", "score_deepsets": "ScoreDeepSets"}
    panels = (("auroc_correct", "AUROC_correct"), ("e_aurc", "E-AURC"), ("rer_at_50", "RER@50"))
    index = {(row["scorer"], row["model"], row["eval_split"], int(row["K"])): row
             for row in metric_rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ks = list(KS)
    for ax, (metric, title) in zip(axes, panels):
        for model in curves:
            b3_vals = []
            for k in ks:
                vals = [index[(s, model, POOLED_TEST, k)][metric] for s in B3_SEEDS if s in scorers
                        and index.get((s, model, POOLED_TEST, k)) is not None]
                b3_vals.append(float(np.mean(vals)) if vals else np.nan)
            style = "-" if metric != "e_aurc" else "-"
            ax.plot(ks, b3_vals, style, marker="o", label=f"{labels[model]} (B3 mean)")
            if "cosine" in scorers:
                cos_vals = []
                for k in ks:
                    row = index.get(("cosine", model, POOLED_TEST, k))
                    cos_vals.append(row[metric] if row else np.nan)
                ax.plot(ks, cos_vals, ":", marker="x", alpha=0.65,
                        label=f"{labels[model]} (cosine)")
        ax.set_xticks(ks)
        ax.set_xticklabels([f"K{k}" for k in ks])
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("metric value")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Phase 0.5 score-only reliability vs candidate cardinality "
                 f"(__pooled_test__; best family = {best['b3']['family']})")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "figures" / "score_only_reliability_vs_K.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_artifacts(out_dir, scorers, corpora, bundles, selections, predictions,
                     boot_fields, metric_rows, agg_rows, degrade_rows, gate, best, temperatures,
                     split, manifest, durations, replicates, seed, ci, max_epochs) -> None:
    phase0a._write_csv(out_dir / "reliability_metrics.csv", METRIC_FIELDS, metric_rows)
    phase0a._write_csv(out_dir / "scalar_baselines.csv", METRIC_FIELDS,
                       [row for row in metric_rows if row["model"] in L0_MODELS])
    phase0a._write_csv(out_dir / "aggregate.csv", AGGREGATE_FIELDS, agg_rows)
    phase0a._write_csv(out_dir / "cross_k_degradation.csv", DEGRADATION_FIELDS, degrade_rows)
    phase0a._write_csv(out_dir / "paired_bootstrap.csv", BOOTSTRAP_FIELDS,
                       _merged_bootstrap_rows(boot_fields))
    for family, _, _, _ in FAMILIES:
        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        phase0a._write_json(family_dir / "selection.json", {
            "family": family,
            "selection_metric": "mean AUROC_correct over reliability_tune K5/K10",
            "per_scorer": {s: {key: val for key, val in selections[s][family].items()
                               if key not in ("model", "fit_info")} for s in scorers},
        })
        phase0a._write_csv(family_dir / "metrics.csv", METRIC_FIELDS,
                           [row for row in metric_rows if row["model"] == family])
    _dump_predictions(out_dir, scorers, corpora, bundles, predictions, selections)
    phase0a._write_json(out_dir / "sufficiency_gate.json", gate)
    phase0a._write_json(out_dir / "protocol.json", {
        "amendment": "A6 (docs/research_protocol.md)",
        "split_seed": SPLIT_SEED, "train_frac": TRAIN_FRAC,
        "train_ks": list(TRAIN_KS), "eval_splits": list(EVAL_SPLITS) + [POOLED_TEST],
        "model_zoo": {"l0": list(L0_MODELS),
                      "families": [{"name": f[0], "kind": f[1], "variant": f[2], "grid": list(f[3])}
                                   for f in FAMILIES]},
        "hyperparameter_grids": {"logistic_C": list(CS), "learning_rate": list(LRS)},
        "selection_metric": "mean AUROC_correct over reliability_tune K5/K10",
        "primary_metrics": list(reval.PRIMARY_METRICS),
        "bootstrap": {"replicates": int(replicates), "seed": int(seed), "ci": float(ci),
                      "resample_unit": "image"},
        "thresholds": dict(reval.DEFAULT_THRESHOLDS),
        "temperatures": temperatures,
        "model_seed": MODEL_SEED, "max_epochs": int(max_epochs),
        "split_manifest": manifest,
        "durations_seconds": durations,
        "created_utc": phase0a._utc_now(),
        "git_commit": phase0a._git_commit(),
    })
    phase0a._write_json(out_dir / "metadata.json", {
        "artifact": "phase05_score_sufficiency",
        "created_utc": phase0a._utc_now(),
        "git_commit": phase0a._git_commit(),
        "scorers": list(scorers),
        "n_metric_rows": len(metric_rows),
        "n_degradation_rows": len(degrade_rows),
        "selection": {s: {f: {"chosen_hp": selections[s][f]["chosen_hp"],
                              "tune_mean_auroc": selections[s][f]["tune_mean_auroc"],
                              "params": selections[s][f]["params"]}
                          for f in [x[0] for x in FAMILIES]} for s in scorers},
        "best": {"b3_family": best["b3"]["family"], "b3_summary": best["b3"]["summary"],
                 "per_scorer": best["per_scorer"]},
        "gate_verdict_seed_mean": gate["verdict_seed_mean"]["verdict"],
        "durations_seconds": durations,
    })


def _merged_bootstrap_rows(boot_fields) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    blank = {field: None for field in BOOTSTRAP_FIELDS}
    for row in boot_fields["cross_k"]:
        merged.append({**blank, **row, "bootstrap_kind": "cross_k",
                       "model_a": None, "model_b": None})
    for row in boot_fields["pairwise"]:
        entry = {**blank, **row, "bootstrap_kind": "model_vs_model",
                 "model": None, "K_lo": int(row["K"]), "K_hi": None,
                 "diff_kind": "absolute", "mean_lo": None, "mean_hi": None}
        entry.pop("K", None)  # single-K pair: K_lo carries it (K not in the CSV schema)
        merged.append(entry)
    return merged


def _dump_predictions(out_dir, scorers, corpora, bundles, predictions, selections) -> None:
    for corpus_id in scorers:
        target = out_dir / "predictions" / corpus_id
        target.mkdir(parents=True, exist_ok=True)
        scorer_seed = {"b3_seed1": 1, "b3_seed2": 2, "b3_seed3": 3}.get(corpus_id, 0)
        corpus = corpora[corpus_id]
        for model in DUMP_MODELS:
            path = target / f"{model}.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["ref_id", "image_id", "K", "eval_split",
                                 "grounding_correct", "reliability_score", "model", "scorer_seed"])
                for k in KS:
                    for split_name in EVAL_SPLITS:
                        mask = corpus[k]["eval_split"] == split_name
                        conf, _ = _confidence_of(corpus_id, model, corpus, bundles,
                                                 predictions, k, split_name, mask)
                        for ref, image, correct, score in zip(
                                corpus[k]["ref_id"][mask], corpus[k]["image_id"][mask],
                                corpus[k]["correct"][mask], conf):
                            writer.writerow([int(ref), int(image), int(k), split_name,
                                             int(correct), float(score), model, scorer_seed])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--scorers", nargs="+", default=list(DEFAULT_SCORERS))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--b3-root", type=Path, default=rdata.B3_ROOT)
    parser.add_argument("--cosine-root", type=Path, default=rdata.COSINE_ROOT)
    parser.add_argument("--corrected-root", type=Path, default=Path("results/phase0a_corrected"))
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="quick end-to-end check (2 scorers, 100 bootstrap reps, 40 epochs)")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.scorers = ["b3_seed1", "cosine"]
        args.bootstrap_replicates = min(args.bootstrap_replicates, 100)
        args.max_epochs = min(args.max_epochs, 40)
    result = run_phase05(
        out_dir=args.out, scorers=tuple(args.scorers),
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed), ci=float(args.ci),
        split_seed=int(args.split_seed), train_frac=float(args.train_frac),
        max_epochs=int(args.max_epochs),
        b3_root=args.b3_root, cosine_root=args.cosine_root,
        corrected_root=args.corrected_root, log_file=args.log_file,
        figure=not args.no_figure,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
