#!/usr/bin/env python
"""V2-M M2 runner: competition-curriculum training (m=0/2/4) + unseen m=8 evaluation.

M2 answers the preregistered question (protocol.json ``M2_preregistration``):
once the reliability model is exposed to a controlled competition curriculum
(``m in {0, 2, 4}`` same-category distractors, the frozen Phase 1F level
construction at K=10), does structured local-competition modelling (LCR)
generalise better than the semantic-statistics baseline (E1b) and the
matched-capacity nonlinear control (Aggregate-MLP) to the *completely unseen*
severity ``m = 8`` (frozen ``expb_m8`` cell, K=10 same8 cohort)?

Fair curriculum (section 6/8 of the M2 round request):

* all four models ``E1b / Aggregate-MLP / LCR-noGate / LCR`` are retrained on
  the *identical* curriculum rows and rows-weighted objective
  ``(1/3) (L_m0 + L_m2 + L_m4)`` (regime-balanced, never per-sample-count);
* hyperparameters follow the frozen M1 rule (LR grid for the torch models,
  C grid for the logistic ones), selected only on ``reliability_tune``
  ``m in {0, 2, 4}``; ``m = 8`` rows are never materialised before all four
  models are frozen, so m=8 can never influence selection;
* ``R1`` stays the frozen score-only reference: the M1 manifest coefficients
  are adopted verbatim and the reproduction is verified against the frozen M1
  confidences (machine precision) plus a refit drift diagnostic.

Grounding is untouched: every test-side score is the frozen Phase 1F
prediction array itself (re-scored once per cell only as an A8.4 STOP check at
``<= 1e-4``); the val-side level cells are scored with the frozen checkpoints
and checked against the frozen stores on the random-regime ``C_10`` cells.
Fingerprints are recorded per cell.

Usage
-----
    python -u scripts/run_v2m_m2.py --log-file logs_v2m_m2.txt
    python -u scripts/run_v2m_m2.py --smoke        # 1 seed / 100 reps / _smoke dir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit

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
from ccg.lcr.audit import grounding_fingerprint  # noqa: E402
from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.reliability import models as rmodels  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

__all__ = [
    "build_parser",
    "main",
    "balanced_train_weights",
    "balanced_val_weights",
    "balanced_mean_loss",
    "assert_selection_levels",
    "LEVELS_TRAIN",
    "LEVELS_TEST",
    "K_LEVEL",
    "LR_GRID",
    "C_GRID",
    "SELECTION_TIE",
    "EPOCHS",
    "BATCH_SIZE",
    "WEIGHT_DECAY",
    "PATIENCE",
    "BOOT_REPLICATES",
    "BOOT_SEED",
    "BOOT_CI",
    "BOOT_METRICS",
    "GO_AUROC_MIN",
    "STOP_TOL",
    "EXPECTED_PARAMS",
]

# ---------------------------------------------------------------------------
# frozen configuration (protocol.json / manifest_freeze.json)
# ---------------------------------------------------------------------------
OUT_ROOT = Path("results/v2_local_competition")
PROTOCOL_PATH = OUT_ROOT / "protocol.json"
M0_PATH = OUT_ROOT / "m0_architecture.json"
M1_DIR = OUT_ROOT / "m1_random_only"
M2_DIR = OUT_ROOT / "m2_curriculum"
MAN_DIR = M2_DIR / "manifests"
FREEZE_PATH = MAN_DIR / "manifest_freeze.json"
GATE_PATH = M2_DIR / "gate.json"
META_PATH = M2_DIR / "metadata.json"
TRAINING_MANIFEST_PATH = M2_DIR / "training_manifest.json"

M1_RUNNER_PATH = Path("scripts/run_v2m_m1.py")
BUILD_MODULE_PATH = Path("scripts/build_v2m_m2_manifests.py")

MANIFESTS = Path("cache/manifests")
SPLIT_MANIFEST = Path("results/phase05_score_sufficiency/split_manifest.json")
EXPB_PREDICTIONS = Path("results/phase1f_hard_semantic/predictions")
FEATURES_ROOT = Path("cache/features")
BANK = Path("cache/proposals.h5")
REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
IMAGE_SIZES = Path("cache/image_sizes.npz")

SEEDS: Tuple[int, ...] = (1, 2, 3)
LEVELS_TRAIN: Tuple[int, ...] = (0, 2, 4)
LEVELS_TEST: Tuple[int, ...] = (0, 2, 4, 8)
K_LEVEL: int = 10
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
BOOT_METRICS: Tuple[str, ...] = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")

MODELS: Tuple[str, ...] = ("R1", "E1b", "Aggregate-MLP", "LCR-noGate", "LCR")
CURRICULUM_MODELS: Tuple[str, ...] = ("E1b", "Aggregate-MLP", "LCR-noGate", "LCR")
PAIR_PLAN: Mapping[int, Tuple[Tuple[str, str], ...]] = {
    8: (("LCR", "E1b"), ("LCR", "Aggregate-MLP"), ("LCR", "LCR-noGate"), ("LCR", "R1")),
    0: (("LCR", "E1b"), ("LCR", "Aggregate-MLP")),
    2: (("LCR", "E1b"), ("LCR", "Aggregate-MLP")),
    4: (("LCR", "E1b"), ("LCR", "Aggregate-MLP")),
}
EXPECTED_PARAMS = {"LCR": 1333, "LCR-noGate": 1329, "Aggregate-MLP": 1569}

#: M2 GO gate (protocol.json M2_preregistration.gate; never adjusted).
GO_AUROC_MIN: float = 0.010
#: A8.4 STOP tolerance of every frozen-score reproduction check.
STOP_TOL: float = 1e-4
#: The R1 anchor must reproduce the frozen M1 confidences at this tolerance.
ANCHOR_CONF_TOL: float = 1e-9
#: Diagnostic bound on the R1 refit drift (sklearn solver version drift ~1e-7).
R1_REFIT_DRIFT_TOL: float = 1e-6

STATS17_NAMES: Tuple[str, ...] = tuple(rfeat.stat_feature_names())
SEM14_NAMES: Tuple[str, ...] = tuple(v2feat.V2_PRIMARY_SEMANTIC_NAMES)
AMBIGUITY_NAMES: Tuple[str, ...] = (
    "winner_competitor_max_cos",
    "winner_top2_cos",
    "q_margin12",
)
POOLED = shard.POOLED


# ---------------------------------------------------------------------------
# small helpers (house style, mirrors run_v2m_m1.py)
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


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


def _auroc(confidence: np.ndarray, correct: np.ndarray) -> Optional[float]:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.size == 0 or np.unique(corr).size < 2:
        return None
    return float(auroc_correct(conf, corr))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# regime-balanced weighting (frozen M2 sections 4/6; loss-aggregation layer only)
# ---------------------------------------------------------------------------
def _balanced_weights(group_sizes: Sequence[int], *, normalisation: str) -> np.ndarray:
    """Concatenated per-row weights of the M2 curriculum (rows ordered by level).

    Group ``r`` (level m in {0, 2, 4}) has ``n_r`` rows; ``N = sum_r n_r``.

    * ``normalisation="train"`` -> ``s_i = N / (3 n_r)``: the batch loss
      ``mean(s_i * l_i)`` is an unbiased estimator of the regime-balanced
      objective ``(1/3) sum_r mean_r(l)`` and its magnitude equals the
      unweighted mean (weights have mean 1), so the frozen optimiser /
      weight-decay semantics are unchanged.  The regime-balanced objective is
      exact at the loss-aggregation layer: the batch gradient is
      ``proportional to sum_r mean_r(nabla l)``.
    * ``normalisation="val"`` -> ``v_i = 1 / (3 n_r)``: the weighted *sum*
      ``sum_i v_i * l_i`` equals ``(1/3) sum_r mean_r(l)`` exactly.
    """
    sizes = np.asarray(group_sizes, dtype=np.float64)
    if sizes.size == 0 or not np.all(np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError(f"group sizes must be positive, got {group_sizes!r}")
    n_total = float(sizes.sum())
    n_groups = int(sizes.size)
    if normalisation == "train":
        per_group = n_total / (n_groups * sizes)
    elif normalisation == "val":
        per_group = 1.0 / (n_groups * sizes)
    else:
        raise ValueError(f"unknown normalisation {normalisation!r}")
    return np.repeat(per_group, sizes.astype(np.int64))


def balanced_train_weights(group_sizes: Sequence[int]) -> np.ndarray:
    """``s_i = N / (3 n_r)`` (mean-1 weights; see :func:`_balanced_weights`)."""
    return _balanced_weights(group_sizes, normalisation="train")


def balanced_val_weights(group_sizes: Sequence[int]) -> np.ndarray:
    """``v_i = 1 / (3 n_r)`` (weighted-sum validation; see :func:`_balanced_weights`)."""
    return _balanced_weights(group_sizes, normalisation="val")


def balanced_mean_loss(losses: np.ndarray, group_sizes: Sequence[int]) -> float:
    """Reference ``(1/3) sum_r mean_r(l)`` of one concatenated loss vector."""
    values = np.asarray(losses, dtype=np.float64).reshape(-1)
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if values.size != int(sizes.sum()):
        raise ValueError(f"{values.size} losses for group sizes summing to {int(sizes.sum())}")
    start = 0
    parts: List[float] = []
    for size in sizes.tolist():
        parts.append(float(np.mean(values[start : start + size])))
        start += size
    return float(np.mean(parts))


def assert_selection_levels(levels: Sequence[int]) -> None:
    """Frozen rule: model selection may only ever touch levels m in {0, 2, 4}."""
    extra = sorted({int(level) for level in levels} - set(LEVELS_TRAIN))
    if extra:
        raise AssertionError(
            f"model selection touched levels {extra}: m=8 is completely unseen "
            "(frozen M2 sections 5/8; selection must never see it)"
        )


# ---------------------------------------------------------------------------
# frozen-artifact loading / verification
# ---------------------------------------------------------------------------
def _load_freeze(log: Callable[[str], None]) -> Dict[str, Any]:
    """Load the M2 manifest freeze record and re-verify every hash."""
    if not FREEZE_PATH.exists():
        raise SystemExit(
            f"M2 manifest freeze missing: {FREEZE_PATH}; run scripts/build_v2m_m2_manifests.py first"
        )
    doc = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    for name, digest in doc["sources"].items():
        path = SPLIT_MANIFEST if name == "split_manifest.json" else (MANIFESTS / name)
        actual = _sha256_file(path)
        if actual != digest:
            raise AssertionError(f"frozen source {name} drifted: {actual} != {digest}")
    for key, entry in doc["val_tables"].items():
        actual = _sha256_file(MAN_DIR / entry["file"])
        if actual != entry["sha256"]:
            raise AssertionError(f"frozen {entry['file']} drifted: {actual} != {entry['sha256']}")
    mapping = doc["k_mapping"]
    for level in LEVELS_TRAIN:
        if int(mapping["train"][f"m{int(level)}"]) != K_LEVEL:
            raise AssertionError(f"k_mapping train m{level} != K={K_LEVEL}")
    for level in LEVELS_TEST:
        if int(mapping["test"][f"m{int(level)}"]) != K_LEVEL:
            raise AssertionError(f"k_mapping test m{level} != K={K_LEVEL}")
    if int(doc["k_level"]) != K_LEVEL:
        raise AssertionError(f"freeze k_level {doc['k_level']} != {K_LEVEL}")
    for key, value in doc["disjointness"].items():
        if int(value) != 0:
            raise AssertionError(f"freeze isolation violated: {key}={value}")
    log("[M2] K reconciliation (preregistration + frozen Phase 1F cells):")
    log(
        f"[M2]   train m0 -> K={K_LEVEL}, m2 -> K={K_LEVEL}, m4 -> K={K_LEVEL}; "
        f"test m8 -> K={K_LEVEL} (frozen expb_m8 cell; K=5 never redefines m=8)"
    )
    log(
        f"[M2] freeze verified: train={int(doc['val_tables']['m0']['train_rows'])} rows / "
        f"{int(doc['val_tables']['m0']['train_images'])} images ... (hashes + isolation ok)"
    )
    return doc


def _load_frozen_r1(seed: int) -> Dict[str, Any]:
    """The stored M1 R1 coefficients of one seed (the frozen b anchor)."""
    path = M1_DIR / f"seed_{int(seed)}" / "model_manifest.json"
    if not path.exists():
        raise SystemExit(f"M1 seed manifest missing: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    r1 = doc["R1"]
    return {
        "chosen_C": float(r1["chosen_C"]),
        "coef": np.asarray(r1["coef"], dtype=np.float64),
        "intercept": float(r1["intercept"]),
        "tune_mean_auroc": float(r1["tune_mean_auroc"]),
    }


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


def _assert_model_inputs(inputs_by_level: Mapping[int, Mapping[str, Any]], allowed: Sequence[str]) -> None:
    """Runtime guard: every model input key must be inside the frozen contract.

    The protocol forbids any GT-category / regime / raw-embedding channel; the
    M2 runner additionally requires the exact frozen key set per model.
    """
    expected = set(allowed)
    for level, inputs in inputs_by_level.items():
        keys = set(inputs)
        if keys != expected:
            raise AssertionError(
                f"model input contract violated at m={level}: keys {sorted(keys)} != {sorted(expected)} "
                "(GT category / regime labels must never reach a model)"
            )


def _merge_rows(
    by_level: Mapping[int, Mapping[str, np.ndarray]], levels: Sequence[int]
) -> Dict[str, np.ndarray]:
    keys = list(by_level[levels[0]].keys())
    return {
        key: np.concatenate([np.asarray(by_level[m][key]) for m in levels], axis=0)
        for key in keys
    }


def _merge_labels(by_level: Mapping[int, np.ndarray], levels: Sequence[int]) -> np.ndarray:
    return np.concatenate([np.asarray(by_level[m], dtype=np.float64) for m in levels])


# ---------------------------------------------------------------------------
# hyperparameter selection (frozen M1 rule: tune m in {0,2,4}, tie rule)
# ---------------------------------------------------------------------------
def _select_by_tie(values: Sequence[float], configs: Sequence[Any]) -> int:
    best = int(np.argmax(np.asarray(values, dtype=np.float64)))
    for idx in range(best):
        if values[best] - values[idx] < SELECTION_TIE:
            return idx
    return best


def _tune_balanced_auroc(
    conf_by_level: Mapping[int, np.ndarray],
    correct_by_level: Mapping[int, np.ndarray],
    levels: Sequence[int],
) -> Optional[float]:
    """Mean of the per-level tune AUROCs (each regime weighted equally)."""
    assert_selection_levels(levels)
    values: List[float] = []
    for level in levels:
        value = _auroc(conf_by_level[level], correct_by_level[level])
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def _fit_logistic_curriculum(
    train_x_by_level: Mapping[int, np.ndarray],
    train_y_by_level: Mapping[int, np.ndarray],
    tune_x_by_level: Mapping[int, np.ndarray],
    tune_y_by_level: Mapping[int, np.ndarray],
    *,
    name: str,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """C-grid logistic fit on the M2 curriculum (regime-balanced; tune-only selection)."""
    levels = sorted(train_x_by_level)
    if levels != sorted(LEVELS_TRAIN) or sorted(tune_x_by_level) != levels:
        raise AssertionError(f"{name}: curriculum levels {levels} != frozen {sorted(LEVELS_TRAIN)}")
    assert_selection_levels(levels)
    weights = balanced_train_weights([np.asarray(train_x_by_level[m]).shape[0] for m in levels])
    train_x = np.vstack([np.asarray(train_x_by_level[m]) for m in levels])
    train_y = _merge_labels(train_y_by_level, levels)
    scored: List[Tuple[float, float]] = []
    fitted: Dict[float, Any] = {}
    for c_value in C_GRID:
        clf = rmodels.LogisticModel(C=float(c_value))
        clf.fit(train_x, train_y, sample_weight=weights)
        conf = {m: clf.predict_proba(np.asarray(tune_x_by_level[m])) for m in levels}
        value = _tune_balanced_auroc(conf, tune_y_by_level, levels)
        if value is None:
            raise RuntimeError(f"{name}: tune AUROC undefined for C={c_value}")
        scored.append((float(c_value), float(value)))
        fitted[float(c_value)] = clf
        log(f"    {name}: C={c_value:g} tune_balanced_auroc={value:.4f}")
    best_idx = _select_by_tie([v for _, v in scored], [c for c, _ in scored])
    best_c, best_value = scored[best_idx]
    model = fitted[best_c]
    coef, intercept = model.coefficients()
    log(f"    {name}: chosen C={best_c:g} (tie rule)")
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


def _fit_torch_curriculum(
    model_cls: Any,
    train_in_by_level: Mapping[int, Mapping[str, np.ndarray]],
    train_y_by_level: Mapping[int, np.ndarray],
    tune_in_by_level: Mapping[int, Mapping[str, np.ndarray]],
    tune_y_by_level: Mapping[int, np.ndarray],
    *,
    seed: int,
    name: str,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """LR-grid torch fit on the M2 curriculum; early stop = regime-balanced val NLL."""
    levels = sorted(train_in_by_level)
    if levels != sorted(LEVELS_TRAIN) or sorted(tune_in_by_level) != levels:
        raise AssertionError(f"{name}: curriculum levels {levels} != frozen {sorted(LEVELS_TRAIN)}")
    assert_selection_levels(levels)
    train_sizes = [int(np.asarray(next(iter(train_in_by_level[m].values()))).shape[0]) for m in levels]
    tune_sizes = [int(np.asarray(next(iter(tune_in_by_level[m].values()))).shape[0]) for m in levels]
    weights = balanced_train_weights(train_sizes)
    val_weights = balanced_val_weights(tune_sizes)
    train_inputs = _merge_rows(train_in_by_level, levels)
    train_y = _merge_labels(train_y_by_level, levels)
    val_inputs = _merge_rows(tune_in_by_level, levels)
    val_y = _merge_labels(tune_y_by_level, levels)
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
            sample_weight=weights,
            val_sample_weight=val_weights,
        )
        conf = {m: model.predict_proba(tune_in_by_level[m]) for m in levels}
        value = _tune_balanced_auroc(conf, tune_y_by_level, levels)
        if value is None:
            raise RuntimeError(f"{name}: tune AUROC undefined for lr={lr}")
        scored.append((float(lr), float(value)))
        fitted[float(lr)] = {"model": model, "info": info}
        log(
            f"    {name}: lr={lr:g} tune_balanced_auroc={value:.4f} "
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
# frozen-cell materialisation (Phase 1F construction only; never resamples)
# ---------------------------------------------------------------------------
def _materialise_level(
    corpus: B3Corpus,
    cohort: Any,
    rows: np.ndarray,
    m: int,
    *,
    expected_candidates: Optional[np.ndarray] = None,
    text_cache: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[hscores.B3ExampleBatch, np.ndarray]:
    """Materialise one level-``m`` cell (K=10) from the frozen construction."""
    row_list = np.asarray(rows, dtype=np.int64).tolist()
    k = int(K_LEVEL)
    cand = np.empty((len(row_list), k), dtype=np.int64)
    samples: List[Any] = []
    for i, row in enumerate(row_list):
        sample = cohort.level_sample(int(row), int(m), k)
        cand[i] = B3Corpus.candidate_bank_indices(sample, k)
        samples.append(sample)
    if expected_candidates is not None and not np.array_equal(cand, np.asarray(expected_candidates)):
        raise AssertionError(f"m={m}: materialised candidates differ from the frozen construction")
    batch = hscores.materialise_examples(corpus, samples, k, text_cache=text_cache)
    return batch, cand


def _b_function(stats_fit_m1: Any, coef: np.ndarray, intercept: float) -> Callable[[np.ndarray], np.ndarray]:
    def b_of(stats17: np.ndarray) -> np.ndarray:
        std = np.asarray(rfeat.normalize_apply(np.asarray(stats17), stats_fit_m1), dtype=np.float64)
        return std @ coef + float(intercept)

    return b_of


def _block_from_scores(
    *,
    rows: np.ndarray,
    sentence_id: np.ndarray,
    ref_id: np.ndarray,
    image_id: np.ndarray,
    scores: np.ndarray,
    batch: hscores.B3ExampleBatch,
    temperature: float,
    b_of: Callable[[np.ndarray], np.ndarray],
    tag: str,
) -> Dict[str, Any]:
    """Feature block of one frozen cell (scores already frozen / verified)."""
    feats = _features_from_arrays(scores, batch.z_q, batch.z_i, temperature, tag=tag)
    return {
        "rows": np.asarray(rows, dtype=np.int64),
        "sentence_id": np.asarray(sentence_id, dtype=np.int64),
        "ref_id": np.asarray(ref_id, dtype=np.int64),
        "image_id": np.asarray(image_id, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
        "correct": np.argmax(np.asarray(scores), axis=1) == 0,
        "stats17": feats["stats17"],
        "sem14": feats["sem14"],
        "r": feats["r"],
        "gate": feats["gate"],
        "b": b_of(feats["stats17"]),
        "fingerprint": dict(grounding_fingerprint(scores)),
        "n": int(np.asarray(scores).shape[0]),
    }


def _val_level_blocks(
    corpus: B3Corpus,
    val_cohort: Any,
    scorers: Mapping[str, Any],
    seed: int,
    temperature: float,
    text_cache: Dict[int, np.ndarray],
    b_of: Callable[[np.ndarray], np.ndarray],
    log: Callable[[str], None],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """train/tune feature blocks of the frozen val level tables (m in {0,2,4})."""
    model = scorers[f"b3_seed{seed}"]
    sid_all = np.asarray(val_cohort.sentence_id, dtype=np.int64)
    img_all = np.asarray(val_cohort.image_id, dtype=np.int64)
    ref_all = np.asarray(val_cohort.ref_id, dtype=np.int64)
    blocks: Dict[str, Dict[int, Dict[str, Any]]] = {"train": {}, "tune": {}}
    for side in ("train", "tune"):
        for m in LEVELS_TRAIN:
            with np.load(MAN_DIR / f"val_level_m{int(m)}.npz") as npz:
                st_sid = np.asarray(npz[f"{side}__sentence_id"], dtype=np.int64)
                st_cand = np.asarray(npz[f"{side}__candidate_indices"], dtype=np.int64)
                st_img = np.asarray(npz[f"{side}__image_id"], dtype=np.int64)
                st_ref = np.asarray(npz[f"{side}__ref_id"], dtype=np.int64)
            pos = np.searchsorted(sid_all, st_sid)
            if not (
                np.array_equal(sid_all[pos], st_sid)
                and np.array_equal(img_all[pos], st_img)
                and np.array_equal(ref_all[pos], st_ref)
            ):
                raise AssertionError(f"val m{m}/{side}: frozen rows are not a canonical-cohort subset")
            batch, _ = _materialise_level(
                corpus, val_cohort, pos, int(m), expected_candidates=st_cand, text_cache=text_cache
            )
            rescored = np.asarray(hscores.score_examples(model, batch, batch_size=SCORE_BATCH), dtype=np.float64)
            blocks[side][int(m)] = _block_from_scores(
                rows=pos, sentence_id=st_sid, ref_id=st_ref, image_id=st_img, scores=rescored,
                batch=batch, temperature=temperature, b_of=b_of, tag=f"val m{m}/{side}/seed{seed}",
            )
            del batch
    return blocks


def _test_level_blocks(
    corpus: B3Corpus,
    cohort_hc: shard.HardCohort,
    rows_same8: np.ndarray,
    scorers: Mapping[str, Any],
    seed: int,
    temperature: float,
    text_cache: Dict[int, np.ndarray],
    b_of: Callable[[np.ndarray], np.ndarray],
    log: Callable[[str], None],
) -> Dict[int, Dict[str, Any]]:
    """Frozen expb cells of every severity level (m in {0,2,4,8}) of one seed.

    Loads the frozen Phase 1F prediction arrays (authoritative scores), re-scores
    the cell once with the frozen checkpoint as the A8.4 STOP check and records
    the grounding fingerprint.
    """
    model = scorers[f"b3_seed{seed}"]
    ids = np.asarray(cohort_hc.sentence_id[rows_same8], dtype=np.int64)
    imgs = np.asarray(cohort_hc.image_id[rows_same8], dtype=np.int64)
    refs = np.asarray(cohort_hc.ref_id[rows_same8], dtype=np.int64)
    blocks: Dict[int, Dict[str, Any]] = {}
    for m in LEVELS_TEST:
        path = EXPB_PREDICTIONS / f"expb_m{int(m)}__b3_seed{seed}.npz"
        with np.load(path) as frozen:
            f_ids = np.asarray(frozen["sentence_id"], dtype=np.int64)
            f_img = np.asarray(frozen["image_id"], dtype=np.int64)
            f_scores = np.asarray(frozen["scores"], dtype=np.float64)
            f_correct = np.asarray(frozen["correct"], dtype=bool)
        if not (np.array_equal(f_ids, ids) and np.array_equal(f_img, imgs)):
            raise AssertionError(f"expb_m{m}/seed{seed}: frozen rows differ from the frozen same8 cell")
        batch, _ = _materialise_level(corpus, cohort_hc, rows_same8, int(m), text_cache=text_cache)
        rescored = np.asarray(hscores.score_examples(model, batch, batch_size=SCORE_BATCH), dtype=np.float64)
        delta = float(np.max(np.abs(rescored - f_scores)))
        if delta > STOP_TOL:
            raise AssertionError(
                f"expb_m{m}/seed{seed}: A8.4 re-score max|delta|={delta:.3e} > {STOP_TOL:g} -- STOP"
            )
        correct = np.argmax(f_scores, axis=1) == 0
        if not np.array_equal(correct, f_correct):
            raise AssertionError(f"expb_m{m}/seed{seed}: frozen correct column disagrees with argmax")
        blocks[int(m)] = _block_from_scores(
            rows=rows_same8, sentence_id=f_ids, ref_id=refs, image_id=f_img, scores=f_scores,
            batch=batch, temperature=temperature, b_of=b_of, tag=f"test m{m}/seed{seed}",
        )
        blocks[int(m)]["stop_max_delta"] = delta
        log(
            f"  [M2 seed{seed}] expb_m{m}: STOP ok (max|delta|={delta:.2e}); "
            f"{f_scores.shape[0]} rows / {int(np.unique(imgs).size)} images; "
            f"b3_accuracy={float(np.mean(correct)):.4f}"
        )
        del batch
    return blocks


def _val_stop_check(
    corpus: B3Corpus,
    val_cohort: Any,
    scorers: Mapping[str, Any],
    seed: int,
    bundle10: Mapping[str, Any],
    val_blocks: Mapping[str, Mapping[int, Dict[str, Any]]],
    text_cache: Dict[int, np.ndarray],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """A8.4-style STOP check on the val side: re-score the random-regime C_10 of
    every curriculum row and compare with the frozen store (<= 1e-4)."""
    model = scorers[f"b3_seed{seed}"]
    union = np.unique(
        np.concatenate(
            [val_blocks[side][m]["rows"] for side in ("train", "tune") for m in LEVELS_TRAIN]
        )
    )
    samples = [val_cohort.sample(int(row), "random") for row in union.tolist()]
    batch = hscores.materialise_examples(corpus, samples, int(K_LEVEL), text_cache=text_cache)
    rescored = np.asarray(hscores.score_examples(model, batch, batch_size=SCORE_BATCH), dtype=np.float64)
    sids = np.asarray([int(sample.sentence_id) for sample in samples], dtype=np.int64)
    store_ids = np.asarray(bundle10["sentence_id"], dtype=np.int64)
    pos = np.searchsorted(store_ids, sids)
    if not (
        pos.size == sids.size
        and np.array_equal(store_ids[pos], sids)
    ):
        raise AssertionError(f"val STOP check seed{seed}: rows are not a K10-store subset")
    reference = np.asarray(bundle10["scores"], dtype=np.float64)[pos]
    delta = float(np.max(np.abs(rescored - reference)))
    if delta > STOP_TOL:
        raise AssertionError(
            f"val random-C10 re-score seed{seed}: max|delta|={delta:.3e} > {STOP_TOL:g} -- STOP"
        )
    log(f"  [M2 seed{seed}] val random-C10 STOP ok (max|delta|={delta:.2e}; {sids.size} rows)")
    del batch
    return {"rows": int(sids.size), "max_delta": delta}


def _curriculum_sha256(sentence_id: Mapping[str, Mapping[int, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for side in ("train", "tune"):
        for m in LEVELS_TRAIN:
            digest.update(f"{side}:m{int(m)}:".encode("ascii"))
            digest.update(np.asarray(sentence_id[side][m], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _model_confidences(
    models: Mapping[str, Any],
    block: Mapping[str, Any],
    *,
    stats_fit_m1: Any,
    coef: np.ndarray,
    intercept: float,
    stats_fit_m2: Any,
    sem_fit_m2: Any,
) -> Dict[str, np.ndarray]:
    stats_std_m1 = np.asarray(
        rfeat.normalize_apply(np.asarray(block["stats17"]), stats_fit_m1), dtype=np.float64
    )
    x31_std = np.hstack(
        [
            rfeat.normalize_apply(np.asarray(block["stats17"]), stats_fit_m2),
            rfeat.normalize_apply(np.asarray(block["sem14"]), sem_fit_m2),
        ]
    )
    x31_raw = np.hstack([np.asarray(block["stats17"]), np.asarray(block["sem14"])])
    b = np.asarray(block["b"], dtype=np.float64)
    r = np.asarray(block["r"], dtype=np.float64)
    gate = np.asarray(block["gate"], dtype=np.float64)
    return {
        "R1": expit(stats_std_m1 @ coef + float(intercept)),
        "E1b": models["E1b"]["model"].predict_proba(x31_std),
        "Aggregate-MLP": models["AggMLP"]["model"].predict_proba({"x": x31_raw}),
        "LCR-noGate": models["LCRNoGate"]["model"].predict_proba({"b": b, "r": r}),
        "LCR": models["LCR"]["model"].predict_proba({"b": b, "r": r, "gate": gate}),
    }


def _gate_diagnostic_row(
    lcr_model: Any, block: Mapping[str, Any], *, level: str, seed: int
) -> Dict[str, Any]:
    """Frozen section 13/14 diagnostics of the LCR correction branch on one level."""
    inputs = {
        "b": np.asarray(block["b"], dtype=np.float64),
        "r": np.asarray(block["r"], dtype=np.float64),
        "gate": np.asarray(block["gate"], dtype=np.float64),
    }
    gd = lcr_model.correction_and_gate(inputs)
    g = np.asarray(gd["gate"], dtype=np.float64)
    d = np.asarray(gd["delta"], dtype=np.float64)
    sem14 = np.asarray(block["sem14"], dtype=np.float64)
    row: Dict[str, Any] = {
        "level": level,
        "seed": int(seed),
        "n": int(g.size),
        "mean_abs_gd": float(np.mean(np.abs(g * d))),
        "mean_g": float(np.mean(g)),
        "mean_abs_d": float(np.mean(np.abs(d))),
    }
    for name in AMBIGUITY_NAMES:
        values = sem14[:, SEM14_NAMES.index(name)]
        if float(np.std(g)) > 0.0 and float(np.std(values)) > 0.0:
            row[f"corr_{name}"] = float(np.corrcoef(g, values)[0, 1])
        else:
            row[f"corr_{name}"] = float("nan")
    return row


# ---------------------------------------------------------------------------
# per-seed run: curriculum training + frozen-test evaluation
# ---------------------------------------------------------------------------
def _run_seed(
    seed: int,
    *,
    temperature: float,
    scorers: Mapping[str, Any],
    corpus: B3Corpus,
    val_cohort: Any,
    cohort_hc: shard.HardCohort,
    rows_same8: np.ndarray,
    split: Any,
    m1_module: Any,
    stored_r1: Mapping[str, Any],
    curriculum_sha: str,
    out_dir: Path,
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    t_seed = time.perf_counter()
    log(f"[M2 seed{seed}] start (T_corrected={temperature:.6f})")

    # -- 1. frozen R1 anchor: reproduce, verify, adopt the stored coefficients --
    bundles = {k: m1_module._load_store_bundle(seed, k, temperature, log) for k in (5, 10)}
    train_masks = {
        k: split.row_mask(bundles[k]["eval_split"], bundles[k]["image_id"], kind="reliability_train")
        for k in (5, 10)
    }
    tune_masks = {
        k: split.row_mask(bundles[k]["eval_split"], bundles[k]["image_id"], kind="reliability_tune")
        for k in (5, 10)
    }
    stats_std, stats_fit_m1 = m1_module._standardise(
        {k: bundles[k]["stats17"] for k in (5, 10)}, train_masks, m1_module.STATS17_NAMES
    )
    r1_refit = m1_module._fit_logistic(
        {k: stats_std[k] for k in (5, 10)},
        {k: bundles[k]["correct"] for k in (5, 10)},
        train_masks,
        tune_masks,
        name=f"seed{seed}:R1(refit check)",
        log=log,
    )
    coef = np.asarray(stored_r1["coef"], dtype=np.float64)
    intercept = float(stored_r1["intercept"])
    anchor: Dict[str, Any] = {
        "source": f"m1_random_only/seed_{int(seed)}/model_manifest.json R1 (frozen M1 coefficients)",
        "b_definition": "b = stats_std_M1 @ stored_coef + stored_intercept (exact M1 logit; never clipped)",
        "chosen_C": float(stored_r1["chosen_C"]),
        "refit_chosen_C": float(r1_refit["chosen_C"]),
        "refit_coef_max_delta": float(
            np.max(np.abs(np.asarray(r1_refit["coef"], dtype=np.float64) - coef))
        ),
        "refit_intercept_delta": abs(float(r1_refit["intercept"]) - intercept),
        "refit_tune_mean_auroc": float(r1_refit["tune_mean_auroc"]),
        "env_note": "refit drift ~1e-7 is sklearn solver version drift; the stored coefficients are authoritative",
        "m1_confidence_repro_max_delta": None,
        "m1_confidence_repro_rows": None,
    }
    if float(r1_refit["chosen_C"]) != float(stored_r1["chosen_C"]):
        raise AssertionError(
            f"seed{seed}: R1 refit C={r1_refit['chosen_C']} != frozen {stored_r1['chosen_C']}"
        )
    if anchor["refit_coef_max_delta"] > R1_REFIT_DRIFT_TOL:
        raise AssertionError(
            f"seed{seed}: R1 refit coef drift {anchor['refit_coef_max_delta']:.3e} > {R1_REFIT_DRIFT_TOL:g}"
        )
    conf_path = M1_DIR / f"seed_{int(seed)}" / "confidences.npz"
    if conf_path.exists():
        mask_test = np.isin(np.asarray(bundles[5]["eval_split"]), ("testA", "testB"))
        stats_std5 = np.asarray(
            rfeat.normalize_apply(np.asarray(bundles[5]["stats17"]), stats_fit_m1), dtype=np.float64
        )
        repro = expit(stats_std5[mask_test] @ coef + intercept)
        with np.load(conf_path) as z:
            ref_conf = np.asarray(z["Random-K5__R1"], dtype=np.float64)
        delta = float(np.max(np.abs(repro - ref_conf))) if repro.shape == ref_conf.shape else float("inf")
        anchor["m1_confidence_repro_max_delta"] = delta
        anchor["m1_confidence_repro_rows"] = int(ref_conf.size)
        if not (delta <= ANCHOR_CONF_TOL):
            raise AssertionError(
                f"seed{seed}: R1 anchor does not reproduce the frozen M1 confidences ({delta:.3e})"
            )
    log(
        f"[M2 seed{seed}] R1 anchor ok (conf repro {anchor['m1_confidence_repro_max_delta']!r}; "
        f"refit drift {anchor['refit_coef_max_delta']:.2e})"
    )
    b_of = _b_function(stats_fit_m1, coef, intercept)

    # -- 2. val level cells (scored with the frozen checkpoint of this seed) ---
    text_cache: Dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    val_blocks = _val_level_blocks(
        corpus, val_cohort, scorers, seed, temperature, text_cache, b_of, log
    )
    for m in LEVELS_TRAIN:
        log(
            f"  [M2 seed{seed}] val m{m}: train={val_blocks['train'][m]['n']} rows / "
            f"{int(np.unique(val_blocks['train'][m]['image_id']).size)} images; "
            f"tune={val_blocks['tune'][m]['n']} rows"
        )
    log(f"[M2 seed{seed}] val cells materialised ({time.perf_counter() - t0:.1f}s)")

    # -- 3. val-side STOP check (never touches training) -----------------------
    stop_val = _val_stop_check(corpus, val_cohort, scorers, seed, bundles[10], val_blocks, text_cache, log)

    # -- 3b. the shared curriculum identity (frozen before any model trains) ---
    blocks_sha = _curriculum_sha256(
        {side: {m: val_blocks[side][m]["sentence_id"] for m in LEVELS_TRAIN} for side in ("train", "tune")}
    )
    if blocks_sha != curriculum_sha:
        raise AssertionError(f"seed{seed}: curriculum rows differ from the frozen tables")

    # -- 4. curriculum standardisation (train blocks only) ---------------------
    train_stats = np.vstack([np.asarray(val_blocks["train"][m]["stats17"]) for m in LEVELS_TRAIN])
    train_sem = np.vstack([np.asarray(val_blocks["train"][m]["sem14"]) for m in LEVELS_TRAIN])
    stats_fit_m2 = rfeat.normalize_fit(
        train_stats, fit_rows=np.arange(train_stats.shape[0]), keys=STATS17_NAMES
    )
    sem_fit_m2 = rfeat.normalize_fit(train_sem, fit_rows=np.arange(train_sem.shape[0]), keys=SEM14_NAMES)

    x31_std = {
        side: {
            m: np.hstack(
                [
                    rfeat.normalize_apply(np.asarray(val_blocks[side][m]["stats17"]), stats_fit_m2),
                    rfeat.normalize_apply(np.asarray(val_blocks[side][m]["sem14"]), sem_fit_m2),
                ]
            )
            for m in LEVELS_TRAIN
        }
        for side in ("train", "tune")
    }
    x31_raw = {
        side: {
            m: np.hstack([np.asarray(val_blocks[side][m]["stats17"]), np.asarray(val_blocks[side][m]["sem14"])])
            for m in LEVELS_TRAIN
        }
        for side in ("train", "tune")
    }
    correct_train = {m: val_blocks["train"][m]["correct"] for m in LEVELS_TRAIN}
    correct_tune = {m: val_blocks["tune"][m]["correct"] for m in LEVELS_TRAIN}

    # -- 5. train the four curriculum models (identical rows, identical weights)
    log(f"[M2 seed{seed}] training the curriculum models (regime-balanced m0/m2/m4)")
    e1b_inputs_train = {m: {"x": x31_std["train"][m]} for m in LEVELS_TRAIN}
    e1b_inputs_tune = {m: {"x": x31_std["tune"][m]} for m in LEVELS_TRAIN}
    agg_inputs_train = {m: {"x": x31_raw["train"][m]} for m in LEVELS_TRAIN}
    agg_inputs_tune = {m: {"x": x31_raw["tune"][m]} for m in LEVELS_TRAIN}
    nogate_inputs_train = {
        m: {"b": val_blocks["train"][m]["b"], "r": val_blocks["train"][m]["r"]} for m in LEVELS_TRAIN
    }
    nogate_inputs_tune = {
        m: {"b": val_blocks["tune"][m]["b"], "r": val_blocks["tune"][m]["r"]} for m in LEVELS_TRAIN
    }
    lcr_inputs_train = {
        m: dict(nogate_inputs_train[m], gate=val_blocks["train"][m]["gate"]) for m in LEVELS_TRAIN
    }
    lcr_inputs_tune = {
        m: dict(nogate_inputs_tune[m], gate=val_blocks["tune"][m]["gate"]) for m in LEVELS_TRAIN
    }
    _assert_model_inputs(e1b_inputs_train, ("x",))
    _assert_model_inputs(agg_inputs_train, ("x",))
    _assert_model_inputs(nogate_inputs_train, ("b", "r"))
    _assert_model_inputs(lcr_inputs_train, ("b", "r", "gate"))

    e1b = _fit_logistic_curriculum(
        {m: x31_std["train"][m] for m in LEVELS_TRAIN}, correct_train,
        {m: x31_std["tune"][m] for m in LEVELS_TRAIN}, correct_tune,
        name=f"seed{seed}:E1b", log=log,
    )
    agg = _fit_torch_curriculum(
        AggregateMLP, agg_inputs_train, correct_train, agg_inputs_tune, correct_tune,
        seed=seed, name=f"seed{seed}:Aggregate-MLP", log=log,
    )
    nogate = _fit_torch_curriculum(
        LCRNoGate, nogate_inputs_train, correct_train, nogate_inputs_tune, correct_tune,
        seed=seed, name=f"seed{seed}:LCR-noGate", log=log,
    )
    lcr = _fit_torch_curriculum(
        LCR, lcr_inputs_train, correct_train, lcr_inputs_tune, correct_tune,
        seed=seed, name=f"seed{seed}:LCR", log=log,
    )
    models: Dict[str, Any] = {"E1b": e1b, "AggMLP": agg, "LCRNoGate": nogate, "LCR": lcr}
    params = {
        "LCR": lcr["n_params"], "LCR-noGate": nogate["n_params"], "Aggregate-MLP": agg["n_params"],
    }
    for name, expected in EXPECTED_PARAMS.items():
        if params[name] != expected:
            raise AssertionError(f"{name}: {params[name]} params != frozen {expected}")
    curriculum_hashes = {name: curriculum_sha for name in CURRICULUM_MODELS}
    if len(set(curriculum_hashes.values())) != 1:
        raise AssertionError("curriculum mismatch across models -- unfair comparison")

    # -- 6. frozen test cells (m in {0,2,4,8}) -- FIRST touched after freezing --
    t0 = time.perf_counter()
    test_blocks = _test_level_blocks(
        corpus, cohort_hc, rows_same8, scorers, seed, temperature, text_cache, b_of, log
    )
    log(f"[M2 seed{seed}] frozen test cells loaded + STOP-verified ({time.perf_counter() - t0:.1f}s)")

    # -- 7. evaluation on every severity level ---------------------------------
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    confidences: Dict[str, np.ndarray] = {}
    cell_info: Dict[str, Any] = {}
    for m in LEVELS_TEST:
        block = test_blocks[int(m)]
        cohort = f"m{int(m)}"
        conf = _model_confidences(
            models, block, stats_fit_m1=stats_fit_m1, coef=coef, intercept=intercept,
            stats_fit_m2=stats_fit_m2, sem_fit_m2=sem_fit_m2,
        )
        correct = np.asarray(block["correct"], dtype=bool)
        clusters = np.asarray(block["image_id"], dtype=np.int64)
        cell_info[cohort] = {
            "n": int(correct.size),
            "n_images": int(np.unique(clusters).size),
            "K": int(K_LEVEL),
            "b3_accuracy": float(correct.mean()),
            "stop_max_delta": float(block["stop_max_delta"]),
            "fingerprint": dict(block["fingerprint"]),
        }
        for model in MODELS:
            row = reval.point_metric_row(conf[model], correct, probability=conf[model])
            row.update(
                {"cohort": cohort, "K": int(K_LEVEL), "seed": int(seed), "model": model,
                 "n": int(correct.size), "n_images": int(np.unique(clusters).size),
                 "b3_accuracy": float(correct.mean())}
            )
            point_rows.append(row)
            confidences[f"{cohort}__{model}"] = np.asarray(conf[model], dtype=np.float64)
        confidences[f"{cohort}__correct"] = correct.astype(np.uint8)
        confidences[f"{cohort}__image_id"] = clusters
        for (model_a, model_b) in PAIR_PLAN[int(m)]:
            t0 = time.perf_counter()
            rows = reval.model_vs_model_bootstrap_row(
                conf[model_a], correct, conf[model_b], correct, clusters,
                eval_split=POOLED, K=int(K_LEVEL), model_a=model_a, model_b=model_b,
                metrics=BOOT_METRICS, replicates=int(args.bootstrap_replicates),
                seed=int(args.bootstrap_seed), ci=float(args.ci),
            )
            for row in rows:
                row.update({"cohort": cohort, "seed": int(seed)})
                boot_rows.append(row)
            log(
                f"  [M2 seed{seed}] bootstrap {cohort} {model_a}-{model_b} "
                f"({time.perf_counter() - t0:.1f}s)"
            )
        diag = _gate_diagnostic_row(models["LCR"]["model"], block, level=cohort, seed=seed)
        diag_rows.append(diag)
        gd = models["LCR"]["model"].correction_and_gate(
            {"b": np.asarray(block["b"]), "r": np.asarray(block["r"]), "gate": np.asarray(block["gate"])}
        )
        confidences[f"{cohort}__gate"] = np.asarray(gd["gate"], dtype=np.float64)
        confidences[f"{cohort}__delta"] = np.asarray(gd["delta"], dtype=np.float64)
        log(
            f"  [M2 seed{seed}] {cohort}: LCR-E1b dAUROC="
            f"{float(np.mean([r['diff'] for r in boot_rows if r['cohort'] == cohort and r['model_a'] == 'LCR' and r['model_b'] == 'E1b' and r['metric'] == 'auroc_correct' and r['seed'] == seed])):+.4f}"
        )

    # -- 8. persist per-seed artifacts -----------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": int(seed),
        "scorer": f"b3_seed{seed}",
        "T_corrected": float(temperature),
        "r1_anchor": anchor,
        "val_stop_random_k10_max_delta": float(stop_val["max_delta"]),
        "val_stop_random_k10_rows": int(stop_val["rows"]),
        "curriculum": {
            "levels": list(LEVELS_TRAIN),
            "train_rows": {f"m{m}": int(val_blocks["train"][m]["n"]) for m in LEVELS_TRAIN},
            "tune_rows": {f"m{m}": int(val_blocks["tune"][m]["n"]) for m in LEVELS_TRAIN},
            "weights": "train s_i = N/(3 n_r); val v_i = 1/(3 n_r); L = (1/3)(L_m0+L_m2+L_m4)",
            "curriculum_sha256": curriculum_sha,
            "levels_per_model": dict(curriculum_hashes),
        },
        "chosen": {
            "E1b_C": e1b["chosen_C"], "Aggregate-MLP_lr": agg["chosen_lr"],
            "LCR-noGate_lr": nogate["chosen_lr"], "LCR_lr": lcr["chosen_lr"],
        },
        "tune_balanced_mean_auroc": {
            "E1b": e1b["tune_mean_auroc"], "Aggregate-MLP": agg["tune_mean_auroc"],
            "LCR-noGate": nogate["tune_mean_auroc"], "LCR": lcr["tune_mean_auroc"],
        },
        "models": {
            "E1b": {k: v for k, v in e1b.items() if k not in ("model", "coef")} | {"coef": np.asarray(e1b["coef"]).tolist()},
            "Aggregate-MLP": {k: v for k, v in agg.items() if k != "model"},
            "LCR-noGate": {k: v for k, v in nogate.items() if k != "model"},
            "LCR": {k: v for k, v in lcr.items() if k != "model"},
        },
        "test_cells": cell_info,
        "val_cells": {
            side: {
                f"m{m}": {
                    "n": int(val_blocks[side][m]["n"]),
                    "fingerprint": dict(val_blocks[side][m]["fingerprint"]),
                }
                for m in LEVELS_TRAIN
            }
            for side in ("train", "tune")
        },
        "params": params,
    }
    _write_json(out_dir / "model_manifest_seed{}.json".format(seed), manifest)
    np.savez_compressed(out_dir / "confidences.npz", **confidences)
    log(
        f"[M2 seed{seed}] done in {time.perf_counter() - t_seed:.1f}s "
        f"(params {params}; chosen {manifest['chosen']})"
    )
    return {
        "seed": int(seed),
        "T": float(temperature),
        "anchor": anchor,
        "stop_val": stop_val,
        "params": params,
        "chosen": manifest["chosen"],
        "tune_auroc": manifest["tune_balanced_mean_auroc"],
        "cell_info": cell_info,
        "point_rows": point_rows,
        "boot_rows": boot_rows,
        "diag_rows": diag_rows,
        "fingerprints": {cohort: cell_info[cohort]["fingerprint"] for cohort in cell_info},
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
        "per_seed_ci": {int(r["seed"]): [float(r["ci_low"]), float(r["ci_high"])] for r in sel},
        "n_positive": int(sum(d > 0.0 for d in diffs)),
        "n_negative": int(sum(d < 0.0 for d in diffs)),
    }


def _point_index(point_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    return {(str(r["cohort"]), int(r["seed"]), str(r["model"])): r for r in point_rows}


def _derived(
    point_index: Mapping[Any, Mapping[str, Any]], cohort: str, seeds: Sequence[int]
) -> Dict[str, Any]:
    red: Dict[int, float] = {}
    gain50: Dict[int, float] = {}
    gain80: Dict[int, float] = {}
    for seed in seeds:
        e1b_e = float(point_index[(cohort, seed, "E1b")]["e_aurc"])
        lcr_e = float(point_index[(cohort, seed, "LCR")]["e_aurc"])
        red[seed] = (e1b_e - lcr_e) / e1b_e if e1b_e > 1e-12 else float("nan")
        gain50[seed] = 100.0 * (
            float(point_index[(cohort, seed, "LCR")]["rer_at_50"])
            - float(point_index[(cohort, seed, "E1b")]["rer_at_50"])
        )
        gain80[seed] = 100.0 * (
            float(point_index[(cohort, seed, "LCR")]["rer_at_80"])
            - float(point_index[(cohort, seed, "E1b")]["rer_at_80"])
        )
    return {
        "e_aurc_reduction": float(np.nanmean(list(red.values()))),
        "rer50_gain_pp": float(np.nanmean(list(gain50.values()))),
        "rer80_gain_pp": float(np.nanmean(list(gain80.values()))),
        "per_seed": {
            int(s): {
                "e_aurc_reduction": red[s],
                "rer50_gain_pp": gain50[s],
                "rer80_gain_pp": gain80[s],
            }
            for s in seeds
        },
    }


def _severity_rows(
    point_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> List[Dict[str, Any]]:
    metrics = (
        ("auroc_correct", "auroc"),
        ("e_aurc", "e_aurc"),
        ("rer_at_50", "rer50"),
        ("rer_at_80", "rer80"),
    )
    rows: List[Dict[str, Any]] = []
    for level in LEVELS_TEST:
        cohort = f"m{int(level)}"
        for model in MODELS:
            sel = [r for r in point_rows if r["cohort"] == cohort and r["model"] == model]
            if not sel:
                continue
            sel.sort(key=lambda r: int(r["seed"]))
            row: Dict[str, Any] = {
                "level": cohort, "K": int(K_LEVEL), "model": model,
                "n_rows": int(sel[0]["n"]), "n_seeds": len(sel),
            }
            for key, short in metrics:
                vals = [float(r[key]) for r in sel]
                row[f"{short}_mean"] = float(np.mean(vals))
                row[f"{short}_std"] = float(np.std(vals))
            rows.append(row)
    return rows


def _build_gate(
    point_rows: Sequence[Mapping[str, Any]],
    boot_rows: Sequence[Mapping[str, Any]],
    diag_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    idx = _point_index(point_rows)
    primary = "m8"
    d1 = _aggregate(boot_rows, primary, ("LCR", "E1b"), "auroc_correct")
    d2 = _aggregate(boot_rows, primary, ("LCR", "Aggregate-MLP"), "auroc_correct")
    cond_auroc = bool(d1["delta"] >= GO_AUROC_MIN and d1["ci_low"] > 0.0)
    cond_struct = bool(d2["delta"] > 0.0 and d2["ci_low"] > 0.0)
    go = bool(cond_auroc and cond_struct)

    selective = _derived(idx, primary, seeds)
    mixed = bool(
        cond_auroc
        and selective["e_aurc_reduction"] < 0.0
        and selective["rer50_gain_pp"] < 0.0
        and selective["rer80_gain_pp"] < 0.0
    )

    nogate = _aggregate(boot_rows, primary, ("LCR", "LCR-noGate"), "auroc_correct")
    gate_supported = bool(nogate["delta"] > 0.0 and nogate["ci_low"] > 0.0)
    gate_unsupported = bool(nogate["delta"] < 0.0 and nogate["ci_high"] < 0.0)
    gate_equal = bool(nogate["ci_low"] <= 0.0 <= nogate["ci_high"])

    case = "A" if go else ("B" if cond_auroc else "C")
    case_statement = {
        "A": "local competition structure provides value beyond semantic features and matched nonlinear capacity.",
        "B": "nonlinear curriculum-trained capacity helps, but no evidence for the proposed local-competition architecture.",
        "C": "simple semantic statistics remain the stronger reliability estimator even after explicit competition curriculum.",
    }[case]

    comparisons: Dict[str, Any] = {}
    for level in LEVELS_TEST:
        cohort = f"m{int(level)}"
        comparisons[cohort] = {}
        for pair in PAIR_PLAN[int(level)]:
            comparisons[cohort][f"{pair[0]}-{pair[1]}"] = {
                metric: _aggregate(boot_rows, cohort, pair, metric) for metric in BOOT_METRICS
            }
        comparisons[cohort]["derived_LCR_vs_E1b"] = _derived(idx, cohort, seeds)

    severity: Dict[str, Any] = {}
    for level in LEVELS_TEST:
        cohort = f"m{int(level)}"
        severity[cohort] = {
            "LCR-E1b": comparisons[cohort]["LCR-E1b"]["auroc_correct"],
            "LCR-Aggregate-MLP": comparisons[cohort]["LCR-Aggregate-MLP"]["auroc_correct"],
        }

    diag: Dict[str, Any] = {}
    diag_keys = (
        "mean_abs_gd", "mean_g", "mean_abs_d",
        "corr_winner_competitor_max_cos", "corr_winner_top2_cos", "corr_q_margin12",
    )
    for level in LEVELS_TEST:
        cohort = f"m{int(level)}"
        rows = [r for r in diag_rows if r["level"] == cohort]
        entry: Dict[str, Any] = {
            "per_seed": {
                int(r["seed"]): {key: float(r[key]) for key in diag_keys} for r in rows
            }
        }
        for key in diag_keys:
            entry[f"{key}_mean"] = float(np.nanmean([float(r[key]) for r in rows]))
        diag[cohort] = entry

    return {
        "artifact": "v2m_m2_gate",
        "primary_cell": primary,
        "primary_question": "LCR > E1b at the unseen competition severity m=8?",
        "aggregation": "seed-mean of per-seed point deltas; seed-mean of per-seed CI bounds; per-seed direction recorded",
        "bootstrap": {
            "replicates": BOOT_REPLICATES, "seed": BOOT_SEED, "ci": BOOT_CI,
            "metrics": list(BOOT_METRICS),
            "shared_draws": "all comparisons on the same cohort share the same draws",
        },
        "comparisons": comparisons,
        "gates": {
            "M2_GO": {
                "delta1_LCR_vs_E1b": d1,
                "delta2_LCR_vs_AggregateMLP": d2,
                "condition_delta1_ge_0.010_and_ci_low_gt_0": cond_auroc,
                "condition_delta2_gt_0_and_ci_low_gt_0": cond_struct,
                "verdict": go,
            },
            "selective_metrics": dict(selective) | {
                "method_signal_mixed": mixed,
                "note": "AUROC gate passing while every selective metric degrades -> METHOD SIGNAL MIXED (never claim a uniform reliability improvement)",
            },
            "gate_contribution": {
                "delta_auroc_LCR_vs_noGate": nogate["delta"],
                "ci_low": nogate["ci_low"], "ci_high": nogate["ci_high"],
                "per_seed_delta": nogate["per_seed"],
                "supported": gate_supported,
                "unsupported": gate_unsupported,
                "effectively_equal": gate_equal,
                "note": "if LCR-noGate > LCR the ambiguity gate stays unsupported and must not be written as a contribution",
            },
            "capacity_case": {"case": case, "statement": case_statement, "delta2": d2},
            "severity": severity,
            "gate_diagnostics": diag,
        },
        "authorization": {
            "V2-MG": bool(go),
            "B1/B2": bool(go),
            "v2mg_not_authorized": (not go),
            "note": "V2-MG / B1/B2 are authorized only by M2_GO=TRUE; otherwise V2-MG_NOT_AUTHORIZED",
        },
        "interpretation_boundary": {
            "allowed_if_success": "A lightweight local-competition reliability module improves correctness estimation under candidate competition.",
            "forbidden": ["improves visual grounding accuracy", "solves candidate-set shift"],
            "negative_result_note": (
                "if M2 is NO-GO: LCR v1 is frozen as a negative result (M1 random-only failure + M2 curriculum result); "
                "no m=8 gate tuning, no Transformer, no hidden-size / M / raw-embedding changes within this protocol -- "
                "any new structure requires a new method amendment."
            ),
        },
    }


def _print_report(
    gate: Mapping[str, Any], per_seed: Sequence[Mapping[str, Any]], log: Callable[[str], None]
) -> None:
    go = gate["gates"]["M2_GO"]
    d1 = go["delta1_LCR_vs_E1b"]
    d2 = go["delta2_LCR_vs_AggregateMLP"]
    sel = gate["gates"]["selective_metrics"]
    gc = gate["gates"]["gate_contribution"]
    case = gate["gates"]["capacity_case"]
    log("=" * 78)
    log("V2-M M2 report (B0 OpenCLIP ViT-B/32, curriculum m in {0,2,4} at K=10, unseen m=8)")
    log(f"  params: {EXPECTED_PARAMS}")
    for entry in per_seed:
        log(
            f"  seed {entry['seed']}: chosen {entry['chosen']} | tune balanced AUROC "
            f"{ {k: round(v, 4) for k, v in entry['tune_auroc'].items()} }"
        )
    log(
        f"  [m8] LCR-E1b dAUROC={d1['delta']:+.4f} (CI {d1['ci_low']:+.4f}..{d1['ci_high']:+.4f}; "
        f"per-seed { {k: round(v, 4) for k, v in d1['per_seed'].items()} })"
    )
    log(
        f"  [m8] LCR-AggMLP dAUROC={d2['delta']:+.4f} "
        f"(CI {d2['ci_low']:+.4f}..{d2['ci_high']:+.4f})"
    )
    log(
        f"  [m8] selective: E-AURC reduction={sel['e_aurc_reduction']:+.4f}  "
        f"RER50 {sel['rer50_gain_pp']:+.3f}pp  RER80 {sel['rer80_gain_pp']:+.3f}pp  "
        f"| MIXED={sel['method_signal_mixed']}"
    )
    log(
        f"  [m8] LCR-noGate dAUROC={gc['delta_auroc_LCR_vs_noGate']:+.4f} "
        f"(CI {gc['ci_low']:+.4f}..{gc['ci_high']:+.4f}) -> "
        f"{'gate supported' if gc['supported'] else ('gate UNSUPPORTED (noGate better)' if gc['unsupported'] else 'effectively equal')}"
    )
    log(f"  [M2_GO] verdict={go['verdict']} | case {case['case']}: {case['statement']}")
    log(f"  [authorization] V2-MG={gate['authorization']['V2-MG']} B1/B2={gate['authorization']['B1/B2']}")
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
    p.add_argument(
        "--smoke", action="store_true",
        help="1 seed / 100 reps / results/v2_local_competition/m2_curriculum/_smoke",
    )
    p.add_argument("--log-file", default=None)
    return p


def _curriculum_sha256_from_tables() -> str:
    """Curriculum identity computed directly from the frozen val tables."""
    sentence_id: Dict[str, Dict[int, np.ndarray]] = {}
    for side in ("train", "tune"):
        sentence_id[side] = {}
        for m in LEVELS_TRAIN:
            with np.load(MAN_DIR / f"val_level_m{int(m)}.npz") as npz:
                sentence_id[side][m] = np.asarray(npz[f"{side}__sentence_id"], dtype=np.int64)
    return _curriculum_sha256(sentence_id)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds: Tuple[int, ...] = (1,) if args.smoke else tuple(int(s) for s in str(args.seeds).split(","))
    reps = 100 if args.smoke else int(args.bootstrap_replicates)
    out_root = (M2_DIR / "_smoke") if args.smoke else M2_DIR
    log = _make_logger(Path(args.log_file) if args.log_file else None)
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if not M0_PATH.exists():
        raise SystemExit(f"M0 gate artifact missing: {M0_PATH}; run scripts/run_v2m_m0.py first")
    m0 = json.loads(M0_PATH.read_text(encoding="utf-8"))
    if not bool(m0.get("all_checks_passed")):
        raise SystemExit("M0 gate not passed (all_checks_passed != true) -- M2 must not start")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    prereg = protocol["M2_preregistration"]
    gate_rule = prereg["gate"]
    if not (
        abs(float(gate_rule["vs_curriculum_E1b"]["delta_auroc_ge"]) - GO_AUROC_MIN) < 1e-12
        and float(gate_rule["vs_curriculum_E1b"]["ci_low_gt"]) == 0.0
        and float(gate_rule["vs_Aggregate_MLP"]["ci_low_gt"]) == 0.0
    ):
        raise AssertionError("protocol M2 gate constants differ from the runner")
    if list(prereg["training"]["levels"]) != list(LEVELS_TRAIN):
        raise AssertionError("protocol M2 training levels differ from the runner")
    protocol_sha = _sha256_file(PROTOCOL_PATH)
    log(
        f"[M2] start (branch={_git(['branch', '--show-current'])}, head={_git(['rev-parse', '--short', 'HEAD'])}, "
        f"seeds={list(seeds)}, reps={reps}, smoke={bool(args.smoke)})"
    )

    freeze = _load_freeze(log)

    # frozen split (manifest + fresh-rebuild verification on the canonical rows)
    split = sdata.load_split_from_manifest(SPLIT_MANIFEST)
    store5 = sdata.load_embedding_store(5, out_root=sdata.DEFAULT_OUT_ROOT)
    sdata.assert_split_matches_fresh(split, store5.eval_split, store5.image_id)
    del store5
    log(f"[M2] split verified (seed={split.seed}, train={split.train_images.size}, tune={split.tune_images.size})")

    # frozen hard cohort (test-side row universe + isolation reference)
    cohort_hc = shard.load_hard_cohort(
        features_dir=sdata.PHASE05_FEATURES_DIR, manifests_root=MANIFESTS, log=log
    )
    rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    log(f"[M2] frozen test universe: {rows_same8.size} rows / {int(np.unique(cohort_hc.image_id[rows_same8]).size)} images (same8 cohort)")

    build_module = _load_module("build_v2m_m2_manifests", BUILD_MODULE_PATH)
    if int(build_module.K_LEVEL) != K_LEVEL or tuple(build_module.LEVELS_TRAIN) != LEVELS_TRAIN:
        raise AssertionError("build module constants drifted from the runner")
    val_cohort = build_module.load_val_cohort()

    m1_module = _load_module("run_v2m_m1", M1_RUNNER_PATH)
    if (
        tuple(m1_module.C_GRID) != C_GRID
        or tuple(m1_module.LR_GRID) != LR_GRID
        or float(m1_module.SELECTION_TIE) != SELECTION_TIE
        or int(m1_module.EPOCHS) != EPOCHS
        or int(m1_module.BATCH_SIZE) != BATCH_SIZE
        or float(m1_module.WEIGHT_DECAY) != WEIGHT_DECAY
        or int(m1_module.PATIENCE) != PATIENCE
    ):
        raise AssertionError("M1 runner frozen hyperparameters differ from M2 (grids must stay unchanged)")
    for name, expected in EXPECTED_PARAMS.items():
        cls = {"LCR": LCR, "LCR-noGate": LCRNoGate, "Aggregate-MLP": AggregateMLP}[name]
        n = int(cls(seed=0).n_parameters())
        if n != expected:
            raise AssertionError(f"{name}: {n} params != frozen {expected}")
        if int(m0["architecture"]["parameter_counts"][name]) != expected:
            raise AssertionError(f"M0 record architecture mismatch for {name}")

    curriculum_sha = _curriculum_sha256_from_tables()
    log(f"[M2] curriculum sha256 (frozen val tables): {curriculum_sha}")

    scorers = hscores.load_frozen_scorers(seeds=tuple(seeds))
    corpus = B3Corpus(
        FEATURES_ROOT, MANIFESTS, REFS, BANK,
        image_sizes_path=IMAGE_SIZES, ks=(5,), regime="random", preload=True,
    )
    per_seed: List[Dict[str, Any]] = []
    try:
        run_args = argparse.Namespace(
            bootstrap_replicates=int(reps), bootstrap_seed=int(args.bootstrap_seed), ci=float(args.ci)
        )
        for seed in seeds:
            temperature = float(m1_module._corrected_T(int(seed)))
            stored_r1 = _load_frozen_r1(int(seed))
            per_seed.append(
                _run_seed(
                    int(seed),
                    temperature=temperature,
                    scorers=scorers,
                    corpus=corpus,
                    val_cohort=val_cohort,
                    cohort_hc=cohort_hc,
                    rows_same8=rows_same8,
                    split=split,
                    m1_module=m1_module,
                    stored_r1=stored_r1,
                    curriculum_sha=curriculum_sha,
                    out_dir=out_root / f"seed_{int(seed)}",
                    args=run_args,
                    log=log,
                )
            )
    finally:
        corpus.close()

    point_rows = [r for entry in per_seed for r in entry["point_rows"]]
    boot_rows = [r for entry in per_seed for r in entry["boot_rows"]]
    diag_rows = [r for entry in per_seed for r in entry["diag_rows"]]
    seed_list = [int(entry["seed"]) for entry in per_seed]
    gate = _build_gate(point_rows, boot_rows, diag_rows, seed_list)

    _write_csv(
        out_root / "point_metrics.csv", point_rows,
        ("cohort", "K", "seed", "model", "n", "n_images", "b3_accuracy", "auroc_correct",
         "e_aurc", "rer_at_50", "rer_at_80", "aurc", "aurc_oracle", "ece_adaptive",
         "brier_binary", "nll_binary"),
    )
    _write_csv(
        out_root / "severity_curve.csv", _severity_rows(point_rows, seed_list),
        ("level", "K", "model", "n_rows", "n_seeds", "auroc_mean", "auroc_std",
         "e_aurc_mean", "e_aurc_std", "rer50_mean", "rer50_std", "rer80_mean", "rer80_std"),
    )
    _write_csv(
        out_root / "gate_diagnostics.csv", diag_rows,
        ("level", "seed", "n", "mean_abs_gd", "mean_g", "mean_abs_d",
         "corr_winner_competitor_max_cos", "corr_winner_top2_cos", "corr_q_margin12"),
    )
    _write_csv(
        out_root / "bootstrap_pairs.csv", boot_rows,
        ("cohort", "K", "seed", "model_a", "model_b", "metric", "diff", "ci_low", "ci_high",
         "mean_a", "mean_b", "n", "n_clusters", "n_replicates", "ci_level"),
    )
    _write_json(out_root / "gate.json", gate)

    training_manifest = {
        "artifact": "v2m_m2_training_manifest",
        "rule": (
            "all four curriculum models share the identical frozen rows; m=8 is never "
            "materialised before the models freeze"
        ),
        "k_level": int(K_LEVEL),
        "levels_train": list(LEVELS_TRAIN),
        "levels_test": list(LEVELS_TEST),
        "weights": (
            "train s_i = N/(3 n_r) (mean-1; loss = mean(s.*l)); val v_i = 1/(3 n_r) "
            "(val_nll = sum(v.*l)); both equal the regime-balanced (1/3)(L_m0+L_m2+L_m4)"
        ),
        "curriculum_sha256": curriculum_sha,
        "val_tables": {
            f"m{m}": {k: freeze["val_tables"][f"m{m}"][k] for k in ("file", "sha256", "train_rows", "train_images", "tune_rows", "tune_images")}
            for m in LEVELS_TRAIN
        },
        "seeds": seed_list,
        "model_selection": "tune balanced mean AUROC on m in {0,2,4} only (m=8 never seen)",
    }
    _write_json(out_root / "training_manifest.json", training_manifest)

    meta = {
        "artifact": "v2m_m2_metadata",
        "branch": _git(["branch", "--show-current"]),
        "head": _git(["rev-parse", "HEAD"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "protocol_sha256": protocol_sha,
        "m0_sha256": _sha256_file(M0_PATH),
        "manifest_freeze_sha256": _sha256_file(FREEZE_PATH),
        "started_utc": started_utc,
        "runtime_sec": round(time.perf_counter() - started, 1),
        "smoke": bool(args.smoke),
        "seeds": seed_list,
        "seed_pairing": "seed s: frozen B3 scorer seed s (scores + corrected T) and torch RNG seed s",
        "k_mapping": freeze["k_mapping"],
        "isolation": freeze["disjointness"],
        "weights": training_manifest["weights"],
        "early_stopping": "regime-balanced tune NLL on m in {0,2,4}, best-state restore",
        "training_protocol": {
            "optimizer": "Adam + BCEWithLogits, weight_decay 1e-4",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "patience": PATIENCE,
            "lr_grid": list(LR_GRID), "c_grid": list(C_GRID), "selection_tie": SELECTION_TIE,
        },
        "r1_anchor": {str(e["seed"]): e["anchor"] for e in per_seed},
        "chosen": {str(e["seed"]): e["chosen"] for e in per_seed},
        "tune_balanced_mean_auroc": {str(e["seed"]): e["tune_auroc"] for e in per_seed},
        "parameters": dict(EXPECTED_PARAMS),
        "stop_checks": {
            str(e["seed"]): {
                "val_random_k10": e["stop_val"],
                "test_cells": {c: e["cell_info"][c]["stop_max_delta"] for c in e["cell_info"]},
            }
            for e in per_seed
        },
        "test_cells": {str(e["seed"]): e["cell_info"] for e in per_seed},
        "grounding_fingerprints": {str(e["seed"]): e["fingerprints"] for e in per_seed},
        "bootstrap": {
            "replicates": int(reps), "seed": int(args.bootstrap_seed), "ci": float(args.ci),
            "metrics": list(BOOT_METRICS),
            "shared_draws": "same cohort -> same draws (same seed & clusters)",
        },
        "v2mg_authorized": bool(gate["gates"]["M2_GO"]["verdict"]),
    }
    _write_json(out_root / "metadata.json", meta)

    _print_report(gate, per_seed, log)
    log(
        "V2M_M2_COMPLETE "
        f"go={gate['gates']['M2_GO']['verdict']} "
        f"case={gate['gates']['capacity_case']['case']} "
        f"mixed={gate['gates']['selective_metrics']['method_signal_mixed']} "
        f"d1={gate['gates']['M2_GO']['delta1_LCR_vs_E1b']['delta']:+.4f} "
        f"d2={gate['gates']['M2_GO']['delta2_LCR_vs_AggregateMLP']['delta']:+.4f} "
        f"v2mg={gate['authorization']['V2-MG']} "
        f"runtime={time.perf_counter() - started:.1f}s out={out_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
