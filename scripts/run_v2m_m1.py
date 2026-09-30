#!/usr/bin/env python
"""V2-M M1 runner: B0 random-only training (Zero-Hard-Exposure) + frozen-cohort evaluation.

Scope (protocol ``results/v2_local_competition/protocol.json``, section
``current_round_scope``: M0 + M1 only; M2 / B1 / B2 are out of scope):

For every seed ``s`` in ``{1, 2, 3}`` (seed ``s`` = frozen B3 scorer seed ``s``
for scores / corrected T, and the torch RNG seed of the reliability nets):

* extract the frozen LCR relation features + stats17 + sem14 from the embedding
  stores (Random K5/K10 for training/tuning; K5/K20/K50 for evaluation);
* fit ``R1`` (Stats Logistic) and ``E1b`` (31-d Stats+Semantic Logistic) on
  ``reliability_train`` (Random K5+K10 only), select ``C`` on
  ``reliability_tune`` (A7 rule);
* train ``Aggregate-MLP`` / ``LCR-noGate`` / ``LCR`` with the frozen protocol
  (Adam, BCEWithLogits, wd 1e-4, epochs 300, batch 256, patience 30, early stop
  on tune-block NLL with best-state restore) and select the LR on
  ``reliability_tune`` (mean AUROC, tie rule); ``b = logit(R1)`` is frozen;
* STOP-verify the frozen B3 scoring of the SameCategory-K5 (hard5) cell, then
  evaluate every model on Random-K5 / Random-K5-matched / SameCategory-K5 /
  K20-random / K50-random with image-cluster paired bootstraps (shared draws).

Zero-hard-exposure is enforced by ``m1_training_guard`` on the assembled
train/tune rows before any model is fitted; every stage re-checks the grounding
fingerprint bit-for-bit (LCR never touches a candidate score).

Usage
-----
    python -u scripts/run_v2m_m1.py --log-file logs_v2m_m1.txt
    python -u scripts/run_v2m_m1.py --smoke        # 1 seed / 100 reps / _smoke dir
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.lcr import (  # noqa: E402
    M_COMPETITORS,
    AggregateMLP,
    LCR,
    LCRNoGate,
    relation_features,
)
from ccg.lcr.audit import (  # noqa: E402
    assert_grounding_unchanged,
    grounding_fingerprint,
    m1_training_guard,
)
from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import data as rdata  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.reliability import models as rmodels  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

__all__ = ["build_parser", "main"]

# ---------------------------------------------------------------------------
# frozen configuration (cross-checked against protocol.json / m0_architecture.json)
# ---------------------------------------------------------------------------
OUT_ROOT = Path("results/v2_local_competition")
PROTOCOL_PATH = OUT_ROOT / "protocol.json"
M0_PATH = OUT_ROOT / "m0_architecture.json"
M1_DIR = OUT_ROOT / "m1_random_only"
BOOT_DIR = OUT_ROOT / "bootstrap"
ABL_DIR = OUT_ROOT / "ablations"
GATE_PATH = OUT_ROOT / "gate.json"
META_PATH = OUT_ROOT / "m1_metadata.json"

B3_ROOT = rdata.B3_ROOT
EMB_ROOT = sdata.DEFAULT_OUT_ROOT
PHASE05_DIR = sdata.PHASE05_FEATURES_DIR
MANIFESTS = Path("cache/manifests")
BANK = Path("cache/proposals.h5")
REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
IMAGE_SIZES = Path("cache/image_sizes.npz")
FEATURES_ROOT = Path("cache/features")
SPLIT_MANIFEST = Path("results/phase05_score_sufficiency/split_manifest.json")
HARD5_MANIFEST = Path("results/phase1f_hard_semantic/manifests/hard5_candidates.npz")
HARD5_PREDICTIONS = Path("results/phase1f_hard_semantic/predictions")

SEEDS: Tuple[int, ...] = (1, 2, 3)
TRAIN_KS: Tuple[int, ...] = (5, 10)
EVAL_KS: Tuple[int, ...] = (5, 20, 50)
ALL_KS: Tuple[int, ...] = (5, 10, 20, 50)
C_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0)
LR_GRID: Tuple[float, ...] = (0.0001, 0.0003, 0.001)
SELECTION_TIE: float = 0.002
EPOCHS: int = 300
BATCH_SIZE: int = 256
WEIGHT_DECAY: float = 1e-4
PATIENCE: int = 30
CHUNK: int = 2048
SCORE_BATCH: int = 256

BOOT_REPLICATES: int = 5000
BOOT_SEED: int = 0
BOOT_CI: float = 0.95
BOOT_METRICS: Tuple[str, ...] = ("auroc_correct", "e_aurc", "rer_at_50")

MODELS: Tuple[str, ...] = ("R1", "E1b", "Aggregate-MLP", "LCR-noGate", "LCR")
COHORT_ORDER: Tuple[str, ...] = (
    "Random-K5",
    "Random-K5-matched",
    "SameCategory-K5",
    "K20-random",
    "K50-random",
)
COHORT_K = {"Random-K5": 5, "Random-K5-matched": 5, "SameCategory-K5": 5, "K20-random": 20, "K50-random": 50}
PAIR_PLAN: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "Random-K5": (("LCR", "E1b"), ("LCR", "Aggregate-MLP"), ("LCR", "LCR-noGate"), ("LCR", "R1")),
    "Random-K5-matched": (("LCR", "E1b"), ("LCR", "Aggregate-MLP")),
    "SameCategory-K5": (("LCR", "E1b"), ("LCR", "Aggregate-MLP"), ("LCR", "LCR-noGate"), ("LCR", "R1")),
    "K20-random": (("LCR", "E1b"), ("LCR", "Aggregate-MLP")),
    "K50-random": (("LCR", "E1b"), ("LCR", "Aggregate-MLP")),
}
POOLED = shard.POOLED
EXPECTED_PARAMS = {"LCR": 1333, "LCR-noGate": 1329, "Aggregate-MLP": 1569}

STATS17_NAMES: Tuple[str, ...] = tuple(rfeat.stat_feature_names())
SEM14_NAMES: Tuple[str, ...] = tuple(v2feat.V2_PRIMARY_SEMANTIC_NAMES)

# Gate constants (protocol.json section M1_gates; never adjusted after M1 results).
GO_AUROC_MIN = 0.010
GO_EAURC_REDUCTION_MIN = 0.05
GO_RER50_GAIN_PP_MIN = 3.0
STRONG_AUROC_MIN = 0.015
STRONG_EAURC_REDUCTION_MIN = 0.10
STRONG_RER50_GAIN_PP_MIN = 5.0
RANDOM_GUARD_AUROC_MIN = -0.005


# ---------------------------------------------------------------------------
# small helpers (house style, mirrors run_v2m_m0.py)
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


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO), capture_output=True, text=True, timeout=30
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _corrected_T(seed: int) -> float:
    """Per-seed frozen corrected global temperature (V1 B0)."""
    em = json.loads(
        (B3_ROOT / f"seed_{seed}" / "eval_metadata.json").read_text(encoding="utf-8")
    )
    t = float(em["temperature_corrected"])
    if not (np.isfinite(t) and t > 0):
        raise ValueError(f"seed_{seed}: invalid corrected T {t}")
    if not bool(em.get("corrected_fit", {}).get("interior", em.get("corrected_interior", True))):
        raise ValueError(f"seed_{seed}: corrected T fit is not interior")
    return t


def _auroc(confidence: np.ndarray, correct: np.ndarray) -> Optional[float]:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.size == 0 or np.unique(corr).size < 2:
        return None
    return float(auroc_correct(conf, corr))


# ---------------------------------------------------------------------------
# frozen feature extraction (chunked; never touches the score matrix)
# ---------------------------------------------------------------------------
def _features_from_arrays(
    scores: np.ndarray,
    z_q: np.ndarray,
    z_i: np.ndarray,
    temperature: float,
    *,
    tag: str,
) -> Dict[str, np.ndarray]:
    """Chunked LCR features of one row block: relations, gate, stats17, sem14."""
    sc_all = np.asarray(scores, dtype=np.float64)
    n = int(sc_all.shape[0])
    stats17 = np.empty((n, len(STATS17_NAMES)), dtype=np.float64)
    sem14 = np.empty((n, len(SEM14_NAMES)), dtype=np.float64)
    r = np.empty((n, M_COMPETITORS, 7), dtype=np.float64)
    gate = np.empty((n, 3), dtype=np.float64)
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        sc = sc_all[start:stop]
        zq = np.asarray(z_q[start:stop], dtype=np.float64)
        zi = np.asarray(z_i[start:stop], dtype=np.float64)
        rb = relation_features(sc, zq, zi, temperature)
        if not np.array_equal(rb.winner_idx, np.argmax(sc, axis=1)):
            raise AssertionError(f"{tag}: relation winner differs from argmax")
        r[start:stop] = rb.r
        gate[start:stop] = rb.gate
        stats17[start:stop] = rfeat.stat_features(sc, temperature=temperature)
        sem14[start:stop] = v2feat.v2_primary_semantic_stats(zq, zi, sc)
    for name, block in (("stats17", stats17), ("sem14", sem14), ("r", r), ("gate", gate)):
        if not np.all(np.isfinite(block)):
            raise AssertionError(f"{tag}: non-finite {name}")
    return {"stats17": stats17, "sem14": sem14, "r": r, "gate": gate}


def _load_store_bundle(
    seed: int, k: int, temperature: float, log: Callable[[str], None]
) -> Dict[str, Any]:
    """Store rows of one K aligned 1:1 with the frozen B3 scorer of this seed."""
    store = sdata.load_embedding_store(k, out_root=EMB_ROOT)
    ref = sdata.load_scorer_canonical(f"b3_seed{seed}", k, b3_root=B3_ROOT)
    pos = np.searchsorted(ref.sentence_id, store.sentence_id)
    if pos.size != len(store) or not np.array_equal(ref.sentence_id[pos], store.sentence_id):
        raise AssertionError(f"K={k}/seed{seed}: store ids are not a subset of scorer rows")
    target_local = np.asarray(ref.target_local[pos], dtype=np.int32)
    if not np.all(target_local == 0):
        raise AssertionError(f"K={k}/seed{seed}: frozen target_local is not column 0")
    scores = np.asarray(ref.scores[pos], dtype=np.float64)
    pristine = scores.copy()
    fingerprint = grounding_fingerprint(scores)
    feats = _features_from_arrays(scores, store.z_q, store.z_i, temperature, tag=f"K={k}/seed{seed}")
    assert_grounding_unchanged(
        fingerprint, grounding_fingerprint(scores), context=f"seed{seed}/K{k} extraction"
    )
    if not np.array_equal(scores, pristine):
        raise AssertionError(f"K={k}/seed{seed}: score matrix was modified -- IMPLEMENTATION FAILURE")
    correct = np.argmax(scores, axis=1) == 0
    if not np.array_equal(correct, np.asarray(ref.correct[pos], dtype=bool)):
        raise AssertionError(f"K={k}/seed{seed}: correctness differs from the frozen scorer event")
    bundle: Dict[str, Any] = {
        "k": int(k),
        "scores": scores,
        "correct": correct,
        "image_id": np.asarray(store.image_id, dtype=np.int64),
        "eval_split": np.asarray(store.eval_split),
        "sentence_id": np.asarray(store.sentence_id, dtype=np.int64),
        "ref_id": np.asarray(store.ref_id, dtype=np.int64),
        "stats17": feats["stats17"],
        "sem14": feats["sem14"],
        "r": feats["r"],
        "gate": feats["gate"],
        "fingerprint": dict(fingerprint),
    }
    del store, ref
    return bundle


def _subset_bundle(bundle: Mapping[str, Any], index: Any, *, k: int) -> Dict[str, Any]:
    keys = (
        "scores",
        "correct",
        "image_id",
        "eval_split",
        "sentence_id",
        "ref_id",
        "stats17",
        "sem14",
        "r",
        "gate",
    )
    out = {key: np.asarray(bundle[key])[index] for key in keys}
    out["k"] = int(k)
    return out


# ---------------------------------------------------------------------------
# training (frozen M1 protocol)
# ---------------------------------------------------------------------------
def _standardise(
    raw_by_k: Mapping[int, np.ndarray],
    train_masks: Mapping[int, np.ndarray],
    keys: Sequence[str],
) -> Tuple[Dict[int, np.ndarray], Any]:
    """Train-only standardisation fitted on the combined K5+K10 train rows."""
    combined = np.vstack([np.asarray(raw_by_k[k])[train_masks[k]] for k in TRAIN_KS])
    fit = rfeat.normalize_fit(combined, fit_rows=np.arange(combined.shape[0]), keys=tuple(keys))
    std = {
        k: np.asarray(rfeat.normalize_apply(np.asarray(raw_by_k[k]), fit), dtype=np.float64)
        for k in raw_by_k
    }
    return std, fit


def _combine_rows(by_k: Mapping[int, Mapping[str, np.ndarray]], masks: Mapping[int, np.ndarray]) -> Dict[str, np.ndarray]:
    keys = list(by_k[TRAIN_KS[0]].keys())
    return {
        key: np.concatenate(
            [np.asarray(by_k[k][key])[masks[k]] for k in TRAIN_KS], axis=0
        )
        for key in keys
    }


def _combine_labels(y_by_k: Mapping[int, np.ndarray], masks: Mapping[int, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(y_by_k[k], dtype=np.float64)[masks[k]] for k in TRAIN_KS]
    )


def _tune_mean_auroc(
    conf_by_k: Mapping[int, np.ndarray],
    correct_by_k: Mapping[int, np.ndarray],
    tune_masks: Mapping[int, np.ndarray],
) -> Optional[float]:
    values: List[float] = []
    for k in TRAIN_KS:
        value = _auroc(
            np.asarray(conf_by_k[k])[tune_masks[k]], np.asarray(correct_by_k[k])[tune_masks[k]]
        )
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def _select_by_tie(values: Sequence[float], configs: Sequence[Any]) -> int:
    best = int(np.argmax(np.asarray(values, dtype=np.float64)))
    for idx in range(best):
        if values[best] - values[idx] < SELECTION_TIE:
            return idx
    return best


def _fit_logistic(
    std_by_k: Mapping[int, np.ndarray],
    correct_by_k: Mapping[int, np.ndarray],
    train_masks: Mapping[int, np.ndarray],
    tune_masks: Mapping[int, np.ndarray],
    *,
    name: str,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """C-grid logistic fit on standardised inputs (R1 / E1b share this path)."""
    train_x = np.vstack([np.asarray(std_by_k[k])[train_masks[k]] for k in TRAIN_KS])
    train_y = _combine_labels(correct_by_k, train_masks)
    scored: List[Tuple[float, float]] = []
    fitted: Dict[float, Any] = {}
    for c_value in C_GRID:
        clf = rmodels.LogisticModel(C=float(c_value))
        clf.fit(train_x, train_y)
        conf_k = {k: clf.predict_proba(np.asarray(std_by_k[k])) for k in std_by_k}
        value = _tune_mean_auroc(conf_k, correct_by_k, tune_masks)
        if value is None:
            raise RuntimeError(f"{name}: tune AUROC undefined for C={c_value}")
        scored.append((float(c_value), float(value)))
        fitted[float(c_value)] = clf
    best_idx = _select_by_tie([v for _, v in scored], [c for c, _ in scored])
    best_c, best_value = scored[best_idx]
    model = fitted[best_c]
    coef, intercept = model.coefficients()
    log(
        f"    {name}: C={best_c} tune_mean_auroc={best_value:.4f} "
        f"(grid {[(c, round(v, 4)) for c, v in scored]})"
    )
    return {
        "model": model,
        "chosen_C": best_c,
        "tune_mean_auroc": best_value,
        "grid": [c for c, _ in scored],
        "values": [v for _, v in scored],
        "n_params": int(model.n_parameters()),
        "coef": np.asarray(coef, dtype=np.float64),
        "intercept": float(intercept),
    }


def _fit_torch(
    model_cls: Any,
    inputs_by_k: Mapping[int, Mapping[str, np.ndarray]],
    y_by_k: Mapping[int, np.ndarray],
    train_masks: Mapping[int, np.ndarray],
    tune_masks: Mapping[int, np.ndarray],
    *,
    seed: int,
    name: str,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """LR-grid torch fit with early stop on tune NLL; LR chosen by tune mean AUROC."""
    train_inputs = _combine_rows(inputs_by_k, train_masks)
    train_y = _combine_labels(y_by_k, train_masks)
    val_inputs = _combine_rows(inputs_by_k, tune_masks)
    val_y = _combine_labels(y_by_k, tune_masks)
    scored: List[Tuple[float, float]] = []
    fitted: Dict[float, Dict[str, Any]] = {}
    for lr in LR_GRID:
        model = model_cls(seed=int(seed))
        info = model.fit(
            train_inputs,
            train_y,
            lr=float(lr),
            inputs_val=val_inputs,
            y_val=val_y,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
        )
        conf_k = {
            k: model.predict_proba({key: np.asarray(inputs_by_k[k][key]) for key in inputs_by_k[k]})
            for k in TRAIN_KS
        }
        value = _tune_mean_auroc(conf_k, y_by_k, tune_masks)
        if value is None:
            raise RuntimeError(f"{name}: tune AUROC undefined for lr={lr}")
        scored.append((float(lr), float(value)))
        fitted[float(lr)] = {"model": model, "info": info}
        log(
            f"    {name}: lr={lr:g} tune_mean_auroc={value:.4f} "
            f"best_epoch={info['best_epoch']} epochs_run={info['epochs_run']} "
            f"stopped_early={info['stopped_early']}"
        )
    best_idx = _select_by_tie([v for _, v in scored], [c for c, _ in scored])
    best_lr, best_value = scored[best_idx]
    model, info = fitted[best_lr]["model"], fitted[best_lr]["info"]
    log(f"    {name}: chosen lr={best_lr:g} (tie rule)")
    return {
        "model": model,
        "chosen_lr": best_lr,
        "tune_mean_auroc": best_value,
        "grid": [c for c, _ in scored],
        "values": [v for _, v in scored],
        "n_params": int(model.n_parameters()),
        "best_epoch": int(info["best_epoch"]),
        "epochs_run": int(info["epochs_run"]),
        "stopped_early": bool(info["stopped_early"]),
        "best_val_nll": None if info["best_val_nll"] is None else float(info["best_val_nll"]),
    }


# ---------------------------------------------------------------------------
# SameCategory-K5 (hard5) evaluation cell: frozen B3 re-scoring + STOP check
# ---------------------------------------------------------------------------
def _hard5_bundle(
    seed: int,
    temperature: float,
    batch: hscores.B3ExampleBatch,
    cohort: shard.HardCohort,
    rows_same4: np.ndarray,
    scorers: Mapping[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    model = scorers[f"b3_seed{seed}"]
    scores_full = hscores.score_examples(model, batch, batch_size=SCORE_BATCH)
    pred_path = HARD5_PREDICTIONS / f"hard5__b3_seed{seed}.npz"
    with np.load(pred_path) as pred:
        pred_ids = np.asarray(pred["sentence_id"], dtype=np.int64)
        frozen_scores = np.asarray(pred["scores"], dtype=np.float32)
    ids = np.asarray(cohort.sentence_id[rows_same4], dtype=np.int64)
    if not np.array_equal(ids, pred_ids):
        raise AssertionError(f"hard5/seed{seed}: materialised rows are not aligned with the frozen manifest")
    deltas = hscores.verify_against_raw_scores(
        {f"b3_seed{seed}": scores_full}, {f"b3_seed{seed}": frozen_scores}, k=5, atol=1e-4
    )
    scores = np.asarray(scores_full, dtype=np.float64)
    pristine = scores.copy()
    fingerprint = grounding_fingerprint(scores)
    feats = _features_from_arrays(scores, batch.z_q, batch.z_i, temperature, tag=f"hard5/seed{seed}")
    assert_grounding_unchanged(
        fingerprint, grounding_fingerprint(scores), context=f"hard5/seed{seed} extraction"
    )
    if not np.array_equal(scores, pristine):
        raise AssertionError(f"hard5/seed{seed}: score matrix was modified -- IMPLEMENTATION FAILURE")
    correct = np.argmax(scores, axis=1) == 0
    log(
        f"  [M1 seed{seed}] hard5 STOP ok (max|delta|={deltas[f'b3_seed{seed}']:.2e}); "
        f"{scores.shape[0]} rows / {np.unique(cohort.image_id[rows_same4]).size} images; "
        f"b3_accuracy={correct.mean():.4f}"
    )
    return {
        "k": 5,
        "scores": scores,
        "correct": correct,
        "image_id": np.asarray(cohort.image_id[rows_same4], dtype=np.int64),
        "eval_split": np.asarray(cohort.eval_split[rows_same4]),
        "sentence_id": ids,
        "ref_id": np.asarray(cohort.ref_id[rows_same4], dtype=np.int64),
        "stats17": feats["stats17"],
        "sem14": feats["sem14"],
        "r": feats["r"],
        "gate": feats["gate"],
        "fingerprint": dict(fingerprint),
        "stop_max_delta": float(deltas[f"b3_seed{seed}"]),
    }


# ---------------------------------------------------------------------------
# per-seed run: train, evaluate, bootstrap
# ---------------------------------------------------------------------------
def _confidences(
    models: Mapping[str, Any],
    cdata: Mapping[str, Any],
    stats_fit: Any,
    sem_fit: Any,
    r1_coef: np.ndarray,
    r1_intercept: float,
) -> Dict[str, np.ndarray]:
    stats_std = np.asarray(rfeat.normalize_apply(np.asarray(cdata["stats17"]), stats_fit), dtype=np.float64)
    sem_std = np.asarray(rfeat.normalize_apply(np.asarray(cdata["sem14"]), sem_fit), dtype=np.float64)
    x31_std = np.hstack([stats_std, sem_std])
    x31_raw = np.hstack([np.asarray(cdata["stats17"]), np.asarray(cdata["sem14"])])
    b = stats_std @ r1_coef + float(r1_intercept)
    return {
        "R1": models["R1"]["model"].predict_proba(stats_std),
        "E1b": models["E1b"]["model"].predict_proba(x31_std),
        "Aggregate-MLP": models["AggMLP"]["model"].predict_proba({"x": x31_raw}),
        "LCR-noGate": models["LCRNoGate"]["model"].predict_proba(
            {"b": b, "r": np.asarray(cdata["r"])}
        ),
        "LCR": models["LCR"]["model"].predict_proba(
            {"b": b, "r": np.asarray(cdata["r"]), "gate": np.asarray(cdata["gate"])}
        ),
    }


def _cohort_data(
    name: str,
    bundles: Mapping[int, Mapping[str, Any]],
    hard: Mapping[str, Any],
    matched_rows: np.ndarray,
) -> Dict[str, Any]:
    if name == "Random-K5":
        bundle = bundles[5]
        mask = np.isin(np.asarray(bundle["eval_split"]), ("testA", "testB"))
        return _subset_bundle(bundle, mask, k=5)
    if name == "Random-K5-matched":
        return _subset_bundle(bundles[5], matched_rows, k=5)
    if name == "SameCategory-K5":
        keys = ("scores", "correct", "image_id", "eval_split", "sentence_id", "ref_id",
                "stats17", "sem14", "r", "gate")
        out = {key: np.asarray(hard[key]) for key in keys}
        out["k"] = 5
        out["fingerprint"] = dict(hard["fingerprint"])
        out["stop_max_delta"] = float(hard["stop_max_delta"])
        return out
    if name == "K20-random":
        bundle = bundles[20]
        mask = np.isin(np.asarray(bundle["eval_split"]), ("testA", "testB"))
        return _subset_bundle(bundle, mask, k=20)
    if name == "K50-random":
        bundle = bundles[50]
        mask = np.isin(np.asarray(bundle["eval_split"]), ("testA", "testB"))
        return _subset_bundle(bundle, mask, k=50)
    raise ValueError(f"unknown cohort {name!r}")


def _run_seed(
    seed: int,
    *,
    temperature: float,
    scorers: Mapping[str, Any],
    batch5: hscores.B3ExampleBatch,
    cohort_hc: shard.HardCohort,
    rows_same4: np.ndarray,
    hard_ids: np.ndarray,
    split: rdata.ReliabilitySplit,
    out_dir: Path,
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    t_seed = time.perf_counter()
    log(f"[M1 seed{seed}] start (T_corrected={temperature:.6f})")

    bundles: Dict[int, Dict[str, Any]] = {}
    for k in ALL_KS:
        t0 = time.perf_counter()
        bundles[k] = _load_store_bundle(seed, k, temperature, log)
        log(
            f"[M1 seed{seed}] features K={k}: {len(bundles[k]['scores'])} rows "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    train_masks = {
        k: split.row_mask(bundles[k]["eval_split"], bundles[k]["image_id"], kind="reliability_train")
        for k in TRAIN_KS
    }
    tune_masks = {
        k: split.row_mask(bundles[k]["eval_split"], bundles[k]["image_id"], kind="reliability_tune")
        for k in TRAIN_KS
    }
    guard: Dict[str, Any] = {}
    for kind, masks in (("reliability_train", train_masks), ("reliability_tune", tune_masks)):
        guard[kind] = dict(
            m1_training_guard(
                np.concatenate([bundles[k]["eval_split"][masks[k]] for k in TRAIN_KS]),
                np.concatenate([bundles[k]["sentence_id"][masks[k]] for k in TRAIN_KS]),
                hard_ids,
                train_ks=TRAIN_KS,
                seen_ks=TRAIN_KS,
                context=f"M1/seed{seed}/{kind}",
            )
        )
    log(
        f"[M1 seed{seed}] zero-hard-exposure guard pass "
        f"(train rows K5={int(train_masks[5].sum())} K10={int(train_masks[10].sum())})"
    )

    # -- logistic branches (R1 / E1b) ----------------------------------------
    correct_by_k = {k: bundles[k]["correct"] for k in TRAIN_KS}
    stats_std, stats_fit = _standardise(
        {k: bundles[k]["stats17"] for k in ALL_KS}, train_masks, STATS17_NAMES
    )
    sem_std, sem_fit = _standardise(
        {k: bundles[k]["sem14"] for k in ALL_KS}, train_masks, SEM14_NAMES
    )
    r1 = _fit_logistic(
        {k: stats_std[k] for k in TRAIN_KS}, correct_by_k, train_masks, tune_masks,
        name=f"seed{seed}:R1", log=log,
    )
    e1b_std = {k: np.hstack([stats_std[k], sem_std[k]]) for k in TRAIN_KS}
    e1b = _fit_logistic(
        e1b_std, correct_by_k, train_masks, tune_masks, name=f"seed{seed}:E1b", log=log,
    )
    r1_coef = np.asarray(r1["coef"], dtype=np.float64).copy()
    r1_intercept = float(r1["intercept"])

    # -- b = logit(R1), exact linear (frozen while training LCR) -------------
    b_by_k = {
        k: stats_std[k] @ r1_coef + r1_intercept for k in TRAIN_KS
    }

    # -- torch models --------------------------------------------------------
    x31_by_k = {
        k: np.hstack([bundles[k]["stats17"], bundles[k]["sem14"]]) for k in TRAIN_KS
    }
    agg = _fit_torch(
        AggregateMLP, {k: {"x": x31_by_k[k]} for k in TRAIN_KS}, correct_by_k,
        train_masks, tune_masks, seed=seed, name=f"seed{seed}:Aggregate-MLP", log=log,
    )
    nogate = _fit_torch(
        LCRNoGate, {k: {"b": b_by_k[k], "r": bundles[k]["r"]} for k in TRAIN_KS}, correct_by_k,
        train_masks, tune_masks, seed=seed, name=f"seed{seed}:LCR-noGate", log=log,
    )
    lcr = _fit_torch(
        LCR, {k: {"b": b_by_k[k], "r": bundles[k]["r"], "gate": bundles[k]["gate"]} for k in TRAIN_KS},
        correct_by_k, train_masks, tune_masks, seed=seed, name=f"seed{seed}:LCR", log=log,
    )
    models: Dict[str, Any] = {
        "R1": r1, "E1b": e1b, "AggMLP": agg, "LCRNoGate": nogate, "LCR": lcr,
    }
    params = {
        "LCR": lcr["n_params"], "LCR-noGate": nogate["n_params"], "Aggregate-MLP": agg["n_params"],
    }
    for name, expected in EXPECTED_PARAMS.items():
        if params[name] != expected:
            raise AssertionError(f"{name}: {params[name]} params != frozen {expected}")

    # R1 must stay frozen through the LCR training (protocol: no R1 refit).
    coef_after, intercept_after = r1["model"].coefficients()
    if not (np.array_equal(np.asarray(coef_after), r1_coef) and float(intercept_after) == r1_intercept):
        raise AssertionError("R1 coefficients changed during LCR training -- protocol violation")

    # -- hard5 cell (evaluation only; never touches training) ----------------
    hard = _hard5_bundle(seed, temperature, batch5, cohort_hc, rows_same4, scorers, log)

    # -- matched rows of the K5 store (same4 subset, Random regime) ----------
    store_ids = np.asarray(bundles[5]["sentence_id"], dtype=np.int64)
    pos_map = np.searchsorted(store_ids, np.asarray(cohort_hc.sentence_id, dtype=np.int64))
    if not np.array_equal(store_ids[pos_map], np.asarray(cohort_hc.sentence_id, dtype=np.int64)):
        raise AssertionError(f"seed{seed}: hard cohort rows are not a subset of the K5 store")
    matched_rows = pos_map[np.asarray(cohort_hc.masks["same4"], dtype=bool)]

    # -- evaluation ----------------------------------------------------------
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    cohort_sizes: Dict[str, Any] = {}
    confidences: Dict[str, np.ndarray] = {}
    for name in COHORT_ORDER:
        cdata = _cohort_data(name, bundles, hard, matched_rows)
        conf = _confidences(models, cdata, stats_fit, sem_fit, r1_coef, r1_intercept)
        correct = np.asarray(cdata["correct"], dtype=bool)
        cohorts_clusters = np.asarray(cdata["image_id"], dtype=np.int64)
        cohort_sizes[name] = {
            "n": int(correct.size),
            "n_images": int(np.unique(cohorts_clusters).size),
            "K": int(cdata["k"]),
            "b3_accuracy": float(correct.mean()),
            "fingerprint": dict(cdata["fingerprint"]) if "fingerprint" in cdata else None,
        }
        for model in MODELS:
            row = reval.point_metric_row(conf[model], correct, probability=conf[model])
            row.update(
                {"cohort": name, "K": int(cdata["k"]), "seed": seed, "model": model,
                 "n": int(correct.size), "n_images": int(np.unique(cohorts_clusters).size),
                 "b3_accuracy": float(correct.mean())}
            )
            point_rows.append(row)
            confidences[f"{name}__{model}"] = np.asarray(conf[model], dtype=np.float64)
        confidences[f"{name}__correct"] = correct.astype(np.uint8)
        for (model_a, model_b) in PAIR_PLAN[name]:
            t0 = time.perf_counter()
            rows = reval.model_vs_model_bootstrap_row(
                conf[model_a], correct, conf[model_b], correct, cohorts_clusters,
                eval_split=POOLED, K=int(cdata["k"]), model_a=model_a, model_b=model_b,
                metrics=BOOT_METRICS, replicates=int(args.bootstrap_replicates),
                seed=int(args.bootstrap_seed), ci=float(args.ci),
            )
            for row in rows:
                row.update({"cohort": name, "seed": seed})
                boot_rows.append(row)
            log(
                f"  [M1 seed{seed}] bootstrap {name} {model_a}-{model_b} "
                f"({time.perf_counter() - t0:.1f}s)"
            )

    # -- persist per-seed artifacts ------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "scorer": f"b3_seed{seed}",
        "T_corrected": float(temperature),
        "training_rows": {k: int(train_masks[k].sum()) for k in TRAIN_KS},
        "tune_rows": {k: int(tune_masks[k].sum()) for k in TRAIN_KS},
        "zero_hard_exposure": guard,
        "R1": {k: v for k, v in r1.items() if k not in ("model", "coef")} | {"coef": r1_coef.tolist()},
        "E1b": {k: v for k, v in e1b.items() if k not in ("model", "coef")} | {"coef": np.asarray(e1b["coef"]).tolist()},
        "Aggregate-MLP": {k: v for k, v in agg.items() if k != "model"},
        "LCR-noGate": {k: v for k, v in nogate.items() if k != "model"},
        "LCR": {k: v for k, v in lcr.items() if k != "model"},
        "b_definition": "logit(R1) = x_std @ coef + intercept (exact linear logit, frozen)",
        "hard5_stop_max_delta": float(hard["stop_max_delta"]),
        "params": params,
    }
    _write_json(out_dir / "model_manifest.json", manifest)
    np.savez_compressed(out_dir / "confidences.npz", **confidences)
    log(
        f"[M1 seed{seed}] done in {time.perf_counter() - t_seed:.1f}s "
        f"(params {params}, cohorts {[(n, cohort_sizes[n]['n']) for n in COHORT_ORDER]})"
    )
    return {
        "seed": seed,
        "T": float(temperature),
        "guard": guard,
        "params": params,
        "chosen": {
            "R1_C": r1["chosen_C"], "E1b_C": e1b["chosen_C"],
            "Aggregate-MLP_lr": agg["chosen_lr"], "LCR-noGate_lr": nogate["chosen_lr"],
            "LCR_lr": lcr["chosen_lr"],
        },
        "tune_auroc": {
            "R1": r1["tune_mean_auroc"], "E1b": e1b["tune_mean_auroc"],
            "Aggregate-MLP": agg["tune_mean_auroc"], "LCR-noGate": nogate["tune_mean_auroc"],
            "LCR": lcr["tune_mean_auroc"],
        },
        "cohort_sizes": cohort_sizes,
        "point_rows": point_rows,
        "boot_rows": boot_rows,
        "hard5_stop_max_delta": float(hard["stop_max_delta"]),
        "fingerprints": {
            name: cohort_sizes[name]["fingerprint"] for name in COHORT_ORDER
        } | {f"K{k}": dict(bundles[k]["fingerprint"]) for k in ALL_KS},
    }


# ---------------------------------------------------------------------------
# gate aggregation (V2-G house convention: seed-mean delta / CI bounds)
# ---------------------------------------------------------------------------
def _aggregate(
    boot_rows: Sequence[Mapping[str, Any]], cohort: str, pair: Tuple[str, str], metric: str
) -> Optional[Dict[str, Any]]:
    sel = [
        r for r in boot_rows
        if r["cohort"] == cohort and r["model_a"] == pair[0] and r["model_b"] == pair[1]
        and r["metric"] == metric
    ]
    if not sel:
        return None
    sel.sort(key=lambda r: int(r["seed"]))
    diffs = [float(r["diff"]) for r in sel]
    lows = [float(r["ci_low"]) for r in sel]
    highs = [float(r["ci_high"]) for r in sel]
    return {
        "n_seeds": len(sel),
        "seeds": [int(r["seed"]) for r in sel],
        "delta": float(np.mean(diffs)),
        "ci_low": float(np.mean(lows)),
        "ci_high": float(np.mean(highs)),
        "per_seed": {int(r["seed"]): float(r["diff"]) for r in sel},
        "per_seed_ci": {
            int(r["seed"]): [float(r["ci_low"]), float(r["ci_high"])] for r in sel
        },
        "n_positive": int(sum(d > 0.0 for d in diffs)),
        "n_negative": int(sum(d < 0.0 for d in diffs)),
    }


def _point_index(point_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    return {(str(r["cohort"]), int(r["seed"]), str(r["model"])): r for r in point_rows}


def _derived(point_index: Mapping[Any, Mapping[str, Any]], cohort: str, seeds: Sequence[int]) -> Dict[str, Any]:
    red: Dict[int, float] = {}
    gain: Dict[int, float] = {}
    for seed in seeds:
        e1b_e = float(point_index[(cohort, seed, "E1b")]["e_aurc"])
        lcr_e = float(point_index[(cohort, seed, "LCR")]["e_aurc"])
        red[seed] = (e1b_e - lcr_e) / e1b_e if e1b_e > 1e-12 else float("nan")
        gain[seed] = 100.0 * (
            float(point_index[(cohort, seed, "LCR")]["rer_at_50"])
            - float(point_index[(cohort, seed, "E1b")]["rer_at_50"])
        )
    return {
        "e_aurc_reduction": float(np.nanmean(list(red.values()))),
        "rer50_gain_pp": float(np.nanmean(list(gain.values()))),
        "per_seed": {
            int(s): {"e_aurc_reduction": red[s], "rer50_gain_pp": gain[s]} for s in seeds
        },
    }


def _build_gate(
    point_rows: Sequence[Mapping[str, Any]],
    boot_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    idx = _point_index(point_rows)
    hard = "SameCategory-K5"
    rand = "Random-K5"

    d_auroc = _aggregate(boot_rows, hard, ("LCR", "E1b"), "auroc_correct")
    derived_hard = _derived(idx, hard, seeds)
    rand_guard = _aggregate(boot_rows, rand, ("LCR", "E1b"), "auroc_correct")

    guard_ok = bool(rand_guard["delta"] >= RANDOM_GUARD_AUROC_MIN)
    go_cond_auroc = bool(d_auroc["delta"] >= GO_AUROC_MIN and d_auroc["ci_low"] > 0.0)
    go_cond_secondary = bool(
        derived_hard["e_aurc_reduction"] >= GO_EAURC_REDUCTION_MIN
        or derived_hard["rer50_gain_pp"] >= GO_RER50_GAIN_PP_MIN
    )
    go = bool(go_cond_auroc and go_cond_secondary and guard_ok)

    strong_cond_auroc = bool(d_auroc["delta"] >= STRONG_AUROC_MIN)
    strong_cond_secondary = bool(
        derived_hard["e_aurc_reduction"] >= STRONG_EAURC_REDUCTION_MIN
        or derived_hard["rer50_gain_pp"] >= STRONG_RER50_GAIN_PP_MIN
    )
    strong = bool(strong_cond_auroc and strong_cond_secondary and guard_ok)

    struct = _aggregate(boot_rows, hard, ("LCR", "Aggregate-MLP"), "auroc_correct")
    structure_pass = bool(struct["delta"] > 0.0 and struct["ci_low"] > 0.0)

    gate_cmp = _aggregate(boot_rows, hard, ("LCR", "LCR-noGate"), "auroc_correct")
    gate_supported = bool(gate_cmp["delta"] > 0.0 and gate_cmp["ci_low"] > 0.0)
    gate_equal = bool(gate_cmp["ci_low"] <= 0.0 <= gate_cmp["ci_high"])

    kgen: Dict[str, Any] = {}
    for cohort in ("K20-random", "K50-random"):
        entry: Dict[str, Any] = {"cohort": cohort, "metrics": {}}
        for metric in BOOT_METRICS:
            agg = _aggregate(boot_rows, cohort, ("LCR", "E1b"), metric)
            worse_significant = bool(
                (metric == "e_aurc" and agg["ci_low"] > 0.0)
                or (metric != "e_aurc" and agg["ci_high"] < 0.0)
            )
            entry["metrics"][metric] = dict(agg) | {"systematic_degradation": worse_significant}
        entry["derived"] = _derived(idx, cohort, seeds)
        entry["material_degradation"] = bool(
            entry["metrics"]["auroc_correct"]["delta"] < -0.005
            or entry["derived"]["e_aurc_reduction"] < -0.05
            or entry["derived"]["rer50_gain_pp"] < -3.0
        )
        kgen[cohort] = entry
    kgen_ok = not any(
        entry["material_degradation"]
        or any(entry["metrics"][m]["systematic_degradation"] for m in BOOT_METRICS)
        for entry in kgen.values()
    )

    comparisons: Dict[str, Any] = {}
    for cohort in COHORT_ORDER:
        comparisons[cohort] = {}
        for pair in PAIR_PLAN[cohort]:
            comparisons[cohort][f"{pair[0]}-{pair[1]}"] = {
                metric: _aggregate(boot_rows, cohort, pair, metric) for metric in BOOT_METRICS
            }
        comparisons[cohort]["derived_LCR_vs_E1b"] = _derived(idx, cohort, seeds)

    return {
        "artifact": "v2m_m1_gate",
        "primary_cell": hard,
        "primary_question": "LCR > E1b?",
        "aggregation": "seed-mean of per-seed point deltas; seed-mean of per-seed CI bounds; per-seed direction recorded",
        "bootstrap": {"replicates": BOOT_REPLICATES, "seed": BOOT_SEED, "ci": BOOT_CI,
                       "shared_draws": "all comparisons on the same cohort share the same draws"},
        "comparisons": comparisons,
        "gates": {
            "M1_GO": {
                "delta_auroc": d_auroc["delta"], "delta_auroc_ci_low": d_auroc["ci_low"],
                "per_seed_delta": d_auroc["per_seed"],
                "condition_auroc_ge_0.010_and_ci_low_gt_0": go_cond_auroc,
                "e_aurc_reduction": derived_hard["e_aurc_reduction"],
                "rer50_gain_pp": derived_hard["rer50_gain_pp"],
                "condition_secondary": go_cond_secondary,
                "random_k5_guard": {"delta": rand_guard["delta"], "ci_low": rand_guard["ci_low"],
                                     "threshold": RANDOM_GUARD_AUROC_MIN, "pass": guard_ok},
                "verdict": go,
            },
            "M1_STRONG": {
                "delta_auroc": d_auroc["delta"],
                "condition_auroc_ge_0.015": strong_cond_auroc,
                "e_aurc_reduction": derived_hard["e_aurc_reduction"],
                "rer50_gain_pp": derived_hard["rer50_gain_pp"],
                "condition_secondary": strong_cond_secondary,
                "random_k5_guard_pass": guard_ok,
                "verdict": strong,
            },
            "structure_contribution": {
                "delta_auroc_LCR_vs_AggregateMLP": struct["delta"],
                "ci_low": struct["ci_low"], "ci_high": struct["ci_high"],
                "per_seed_delta": struct["per_seed"],
                "verdict": structure_pass,
                "claim_if_pass": "local competition architecture helps",
                "claim_if_fail": "nonlinear capacity helps",
            },
            "gate_contribution": {
                "delta_auroc_LCR_vs_noGate": gate_cmp["delta"],
                "ci_low": gate_cmp["ci_low"], "ci_high": gate_cmp["ci_high"],
                "per_seed_delta": gate_cmp["per_seed"],
                "significantly_better": gate_supported,
                "effectively_equal": gate_equal,
                "note": "if effectively equal: keep results, the gate must not be written as the main contribution",
            },
            "k_generalization": {"cohorts": kgen, "no_systematic_degradation": kgen_ok},
        },
        "interpretation_boundary": {
            "allowed_if_success": "A lightweight local-competition reliability module improves correctness estimation under candidate competition.",
            "forbidden": ["improves visual grounding accuracy", "solves candidate-set shift"],
        },
    }


# ---------------------------------------------------------------------------
# artifact writers
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _print_report(gate: Mapping[str, Any], per_seed: Sequence[Mapping[str, Any]], log: Callable[[str], None]) -> None:
    hard = "SameCategory-K5"
    cmp_hard = gate["comparisons"][hard]
    a = cmp_hard["LCR-E1b"]["auroc_correct"]
    d = gate["comparisons"][hard]["derived_LCR_vs_E1b"]
    struct = gate["gates"]["structure_contribution"]
    gcmp = gate["gates"]["gate_contribution"]
    go = gate["gates"]["M1_GO"]
    strong = gate["gates"]["M1_STRONG"]
    log("=" * 78)
    log("V2-M M1 report (B0 OpenCLIP ViT-B/32, random-only training, Zero-Hard-Exposure)")
    log(f"  params: {EXPECTED_PARAMS}")
    for entry in per_seed:
        log(f"  seed {entry['seed']}: chosen {entry['chosen']} | tune AUROC { {k: round(v, 4) for k, v in entry['tune_auroc'].items()} }")
    log(f"  [SameCategory-K5] LCR-E1b dAUROC={a['delta']:+.4f} (CI {a['ci_low']:+.4f}..{a['ci_high']:+.4f}; "
        f"per-seed { {k: round(v, 4) for k, v in a['per_seed'].items()} })")
    log(f"  [SameCategory-K5] E-AURC reduction={d['e_aurc_reduction']:+.4f}  RER50 gain={d['rer50_gain_pp']:+.3f}pp")
    log(f"  [structure] LCR-AggMLP dAUROC={struct['delta_auroc_LCR_vs_AggregateMLP']:+.4f} "
        f"(CI_low {struct['ci_low']:+.4f}) -> {struct['claim_if_pass'] if struct['verdict'] else struct['claim_if_fail']}")
    log(f"  [gate] LCR-noGate dAUROC={gcmp['delta_auroc_LCR_vs_noGate']:+.4f} "
        f"(CI {gcmp['ci_low']:+.4f}..{gcmp['ci_high']:+.4f}) -> "
        f"{'gate supported' if gcmp['significantly_better'] else ('effectively equal' if gcmp['effectively_equal'] else 'gate worse')}")
    log(f"  [GO] verdict={go['verdict']} (dAUROC {go['delta_auroc']:+.4f}, ci_low {go['delta_auroc_ci_low']:+.4f}; "
        f"random guard {go['random_k5_guard']['delta']:+.4f} pass={go['random_k5_guard']['pass']})")
    log(f"  [STRONG] verdict={strong['verdict']}")
    log(f"  [K-gen] no_systematic_degradation={gate['gates']['k_generalization']['no_systematic_degradation']}")
    log("=" * 78)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--bootstrap-replicates", type=int, default=BOOT_REPLICATES)
    p.add_argument("--bootstrap-seed", type=int, default=BOOT_SEED)
    p.add_argument("--ci", type=float, default=BOOT_CI)
    p.add_argument("--smoke", action="store_true", help="1 seed / 100 reps / results/v2_local_competition/_smoke")
    p.add_argument("--log-file", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds: Tuple[int, ...] = (1,) if args.smoke else tuple(int(s) for s in str(args.seeds).split(","))
    reps = 100 if args.smoke else int(args.bootstrap_replicates)
    out_root = (OUT_ROOT / "_smoke") if args.smoke else OUT_ROOT
    log = _make_logger(Path(args.log_file) if args.log_file else None)
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if not M0_PATH.exists():
        raise SystemExit(f"M0 gate artifact missing: {M0_PATH}; run scripts/run_v2m_m0.py first")
    m0 = json.loads(M0_PATH.read_text(encoding="utf-8"))
    if not bool(m0.get("all_checks_passed")):
        raise SystemExit("M0 gate not passed (all_checks_passed != true) -- M1 must not start")
    protocol_sha = _sha256_file(PROTOCOL_PATH)
    log(
        f"[M1] start (branch={_git(['branch', '--show-current'])}, head={_git(['rev-parse', '--short', 'HEAD'])}, "
        f"seeds={list(seeds)}, reps={reps}, smoke={bool(args.smoke)})"
    )

    # frozen split (manifest + fresh-rebuild verification on the canonical rows)
    split = sdata.load_split_from_manifest(SPLIT_MANIFEST)
    store5 = sdata.load_embedding_store(5, out_root=EMB_ROOT)
    sdata.assert_split_matches_fresh(split, store5.eval_split, store5.image_id)
    del store5
    log(f"[M1] A6.2 split verified (seed={split.seed}, train={split.train_images.size}, tune={split.tune_images.size})")

    # frozen hard cohort (row universe + zero-hard-exposure reference)
    cohort_hc = shard.load_hard_cohort(features_dir=PHASE05_DIR, manifests_root=MANIFESTS, log=log)
    rows_same4 = np.flatnonzero(np.asarray(cohort_hc.masks["same4"], dtype=bool))
    hard_ids = np.asarray(cohort_hc.sentence_id[rows_same4], dtype=np.int64)

    # hard5 evaluation batch (materialised once; NEVER used in training)
    scorers = hscores.load_frozen_scorers(seeds=tuple(seeds))
    corpus = B3Corpus(
        FEATURES_ROOT, MANIFESTS, REFS, BANK,
        image_sizes_path=IMAGE_SIZES, ks=(5,), regime="random", preload=True,
    )
    samples = [cohort_hc.sample(int(row), "same_category") for row in rows_same4.tolist()]
    batch5 = hscores.materialise_examples(corpus, samples, 5, text_cache={})
    del samples
    corpus.close()
    del corpus
    log(f"[M1] hard5 batch materialised for evaluation only: {batch5.n} rows x K=5")

    run_args = argparse.Namespace(
        bootstrap_replicates=int(reps), bootstrap_seed=int(args.bootstrap_seed), ci=float(args.ci)
    )
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        temperature = _corrected_T(seed)
        per_seed.append(
            _run_seed(
                seed,
                temperature=temperature,
                scorers=scorers,
                batch5=batch5,
                cohort_hc=cohort_hc,
                rows_same4=rows_same4,
                hard_ids=hard_ids,
                split=split,
                out_dir=out_root / "m1_random_only" / f"seed_{seed}",
                args=run_args,
                log=log,
            )
        )

    point_rows = [r for entry in per_seed for r in entry["point_rows"]]
    boot_rows = [r for entry in per_seed for r in entry["boot_rows"]]
    seed_list = [int(entry["seed"]) for entry in per_seed]
    gate = _build_gate(point_rows, boot_rows, seed_list)

    _write_csv(
        out_root / "m1_random_only" / "point_metrics.csv", point_rows,
        ("cohort", "K", "seed", "model", "n", "n_images", "b3_accuracy", "auroc_correct",
         "e_aurc", "rer_at_50", "aurc", "aurc_oracle", "ece_adaptive", "brier_binary", "nll_binary"),
    )
    _write_csv(
        out_root / "bootstrap" / "pairs.csv", boot_rows,
        ("cohort", "K", "seed", "model_a", "model_b", "metric", "diff", "ci_low", "ci_high",
         "mean_a", "mean_b", "n", "n_clusters", "n_replicates", "ci_level"),
    )
    _write_csv(
        out_root / "ablations" / "lcr_vs_nogate.csv",
        [r for r in boot_rows if r["model_a"] == "LCR" and r["model_b"] == "LCR-noGate"],
        ("cohort", "K", "seed", "metric", "diff", "ci_low", "ci_high"),
    )
    _write_json(out_root / "gate.json", gate)

    meta = {
        "artifact": "v2m_m1_metadata",
        "branch": _git(["branch", "--show-current"]),
        "head": _git(["rev-parse", "HEAD"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "protocol_sha256": protocol_sha,
        "m0_sha256": _sha256_file(M0_PATH),
        "started_utc": started_utc,
        "runtime_sec": round(time.perf_counter() - started, 1),
        "smoke": bool(args.smoke),
        "seeds": seed_list,
        "seed_pairing": "seed s: frozen B3 scorer seed s (scores + corrected T) and torch RNG seed s",
        "training_protocol": {
            "train_ks": list(TRAIN_KS), "no_hard_exposure": True,
            "optimizer": "Adam + BCEWithLogits, weight_decay 1e-4",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
            "lr_grid": list(LR_GRID), "c_grid": list(C_GRID), "selection_tie": SELECTION_TIE,
            "early_stopping": "tune-block NLL (Random K5/K10), restore best state",
        },
        "bootstrap": {"replicates": int(reps), "seed": int(args.bootstrap_seed), "ci": float(args.ci),
                       "shared_draws": "same cohort -> same draws (same seed & clusters)"},
        "chosen": {str(entry["seed"]): entry["chosen"] for entry in per_seed},
        "tune_mean_auroc": {str(entry["seed"]): entry["tune_auroc"] for entry in per_seed},
        "parameters": dict(EXPECTED_PARAMS),
        "cohorts": {str(entry["seed"]): entry["cohort_sizes"] for entry in per_seed},
        "zero_hard_exposure": {str(entry["seed"]): entry["guard"] for entry in per_seed},
        "hard5_stop_max_delta": {str(entry["seed"]): entry["hard5_stop_max_delta"] for entry in per_seed},
        "T_corrected": {str(entry["seed"]): entry["T"] for entry in per_seed},
        "grounding_fingerprints": {str(entry["seed"]): entry["fingerprints"] for entry in per_seed},
    }
    _write_json(out_root / "m1_metadata.json", meta)

    _print_report(gate, per_seed, log)
    log(
        "V2M_M1_COMPLETE "
        f"go={gate['gates']['M1_GO']['verdict']} strong={gate['gates']['M1_STRONG']['verdict']} "
        f"structure={gate['gates']['structure_contribution']['verdict']} "
        f"gate_supported={gate['gates']['gate_contribution']['significantly_better']} "
        f"kgen_ok={gate['gates']['k_generalization']['no_systematic_degradation']} "
        f"runtime={time.perf_counter() - started:.1f}s out={out_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
