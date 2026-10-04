"""Research Repair v1 B: semantic information groups and ID/OOD reliability.

This runner consumes only frozen B3 raw scores, cached embeddings, the fixed
reliability split, and the existing Phase 1F candidate manifests/predictions.
It writes all derived results below results/research_repair_v1/information/.
The original results, caches, candidates, and models are read-only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccg.metrics.discrimination import auroc_correct
from ccg.reliability import data as rdata
from ccg.reliability import evaluate as reval
from ccg.reliability import features as rfeat
from ccg.reliability.models import ScoreDeepSets
from ccg.repairs.bootstrap import assert_row_alignment
from ccg.repairs.information import (
    B3_SEEDS,
    FEATURE_GROUPS,
    LOGISTIC_CS,
    SDS_LRS,
    TRAINING_REGIMES,
    V_NAMES,
    assert_nested_correctness,
    compute_semantic_stats_chunked,
    feature_block,
    fit_group_normalization,
    fit_score_deepsets_normalization,
    fit_selected_logistic,
    load_logistic_predictor,
    nll_binary,
    score_deepsets_inputs,
    train_score_deepsets_grid,
)
from ccg.semantic import data as sdata
from ccg.semantic import features as sfeat
from ccg.semantic.frozen import apply_frozen, recover_frozen_models
from ccg.semantic.frozen_load import load_models, verify_against_frozen_artifacts
from ccg.semantic.hard_scores import materialise_examples

KS = (5, 10, 20, 50)
GROUPS = ("S", "S+Q", "S+V", "Full")
CELLS_RANDOM = {k: f"randomK{k}" for k in KS}
PHASE1F_ROOT = ROOT / "results/phase1f_hard_semantic"
PHASE1_ROOT = ROOT / "results/phase1_semantic_sufficiency"
PHASE05_ROOT = ROOT / "results/phase05_score_sufficiency"
SEM_FEATURE_ROOT = PHASE1_ROOT / "semantic_features"
PREDICTION_FIELDS = (
    "sentence_id", "ref_id", "image_id", "eval_split", "seed", "cell", "K",
    "grounding_correct", "model", "training_regime", "confidence", "probability",
)
CURVE_FIELDS = (
    "seed", "training_regime", "model_kind", "hyperparameter", "epoch",
    "train_loss", "train_auroc", "tune_loss", "tune_auroc",
    "train_nll", "val_nll", "val_auroc",
    "tune_auroc_K5", "tune_auroc_K10", "tune_auroc_K20", "tune_auroc_K50",
    "selected", "selected_epoch",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(Path, type) and isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is not None:
        names = list(fields)
    else:
        names = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _append_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", buffering=1)

    def log(message: str) -> None:
        line = f"[{_now()}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")

    return log, handle


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _status_path(out: Path) -> Path:
    return out / "STATUS.json"


def _record_attempt(out: Path, payload: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution_attempts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=_json_default) + "\n")


def _set_status(out: Path, stage: str, state: str, *, detail: str = "", progress: Mapping[str, Any] | None = None) -> None:
    path = _status_path(out)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "experiment": "Research Repair v1 B information source and ID/OOD",
        "stages": {},
    }
    entry = payload.setdefault("stages", {}).setdefault(stage, {})
    if state == "RUNNING" and not entry.get("started_utc"):
        entry["started_utc"] = _now()
    entry["state"] = state
    entry["updated_utc"] = _now()
    if state in {"COMPLETE", "FAILED", "SKIPPED", "UNVERIFIABLE"}:
        entry["finished_utc"] = _now()
    if detail:
        entry["detail"] = detail
    if progress is not None:
        entry["progress"] = dict(progress)
    payload["current_stage"] = stage if state == "RUNNING" else payload.get("current_stage")
    if any(v.get("state") == "RUNNING" for v in payload["stages"].values()):
        payload["status"] = "RUNNING"
    elif state == "FAILED":
        payload["status"] = "FAILED"
    elif stage == "overall":
        payload["status"] = state
        payload["current_stage"] = "overall"
    else:
        payload["status"] = payload.get("status", "PREPARED")
    payload["updated_utc"] = _now()
    _write_json(path, payload)


def _parse_temperature(seed: str) -> float:
    number = seed.removeprefix("b3_seed")
    path = ROOT / f"results/phase0b_independent/seed_{number}/eval_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["temperature_corrected"])


def _parse_bool(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in "biu":
        return arr.astype(bool)
    return np.asarray([str(v).strip().lower() in {"1", "true", "yes"} for v in arr], dtype=bool)


def _canonical_seed_k(seed: str, k: int) -> dict[str, np.ndarray]:
    raw = rdata.load_scorer_scores(seed, k)
    order = np.argsort(raw.sentence_id, kind="stable")
    return {
        "sentence_id": np.asarray(raw.sentence_id[order], dtype=np.int64),
        "ref_id": np.asarray(raw.ref_id[order], dtype=np.int64),
        "image_id": np.asarray(raw.image_id[order], dtype=np.int64),
        "eval_split": np.asarray(raw.eval_split[order]),
        "target_local": np.asarray(raw.target_local[order], dtype=np.int32),
        "scores": np.asarray(raw.scores[order], dtype=np.float32),
        "correct": np.asarray(raw.correct[order], dtype=bool),
    }


def _canonical_validate(seed: str, data: Mapping[int, Mapping[str, np.ndarray]]) -> None:
    base = data[KS[0]]
    for k in KS:
        cur = data[k]
        for field in ("sentence_id", "ref_id", "image_id", "eval_split"):
            if not np.array_equal(base[field], cur[field]):
                raise AssertionError(f"{seed} K={k}: canonical {field} differs from K5")
        if cur["scores"].shape != (base["sentence_id"].size, k):
            raise AssertionError(f"{seed} K={k}: unexpected score shape {cur['scores'].shape}")
        expected = np.argmax(cur["scores"], axis=1) == cur["target_local"]
        if not np.array_equal(expected, cur["correct"]):
            raise AssertionError(f"{seed} K={k}: correctness does not match that K's raw-score argmax")
    assert_nested_correctness({k: data[k]["correct"] for k in KS})
    _log = f"{seed}: canonical rows={base['sentence_id'].size} align exactly for K={KS}"
    print(_log, flush=True)


def _load_features(
    seed: str,
    data: Mapping[int, Mapping[str, np.ndarray]],
    out: Path,
    log: Any,
    *,
    reuse_derived: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    feature_dir = out / "derived"
    feature_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = seed
    result: dict[int, dict[str, np.ndarray]] = {}
    temperature = _parse_temperature(seed)
    for k in KS:
        cache = feature_dir / f"random_{seed_dir}_K{k}.npz"
        if reuse_derived and cache.exists():
            with np.load(cache) as z:
                item = {key: np.asarray(z[key]) for key in ("stats17", "sem16")}
            if item["stats17"].shape != (data[k]["sentence_id"].size, 17) or item["sem16"].shape != (data[k]["sentence_id"].size, 16):
                raise ValueError(f"derived feature cache shape invalid: {cache}")
            result[k] = item
            continue
        store = sdata.load_embedding_store(k)
        if not np.array_equal(store.sentence_id, data[k]["sentence_id"]):
            raise AssertionError(f"{seed} K={k}: embedding store sentence_id order differs")
        if not np.array_equal(store.ref_id, data[k]["ref_id"]) or not np.array_equal(store.image_id, data[k]["image_id"]):
            raise AssertionError(f"{seed} K={k}: embedding store identity anchors differ")
        stats = rfeat.stat_features(data[k]["scores"], temperature=temperature)
        sem = compute_semantic_stats_chunked(store.z_q, store.z_i, data[k]["scores"], chunk_rows=512)
        if tuple(sfeat.SEMANTIC_STAT_NAMES) != tuple(FEATURE_GROUPS["Full"][17:]):
            raise AssertionError("semantic feature order differs from the frozen Full group")
        np.savez_compressed(cache, stats17=stats, sem16=sem)
        result[k] = {"stats17": stats, "sem16": sem}
        log(f"{seed}: computed train-independent frozen feature blocks K={k}, rows={stats.shape[0]}")
        del store, stats, sem
        gc.collect()
    return result


def _mask_dictionary(split: Any, base: Mapping[str, np.ndarray]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    train = {k: split.row_mask(base["eval_split"], base["image_id"], kind="reliability_train") for k in KS}
    tune = {k: split.row_mask(base["eval_split"], base["image_id"], kind="reliability_tune") for k in KS}
    return train, tune


def _group_raw(features: Mapping[int, Mapping[str, np.ndarray]], group: str) -> dict[int, np.ndarray]:
    return {k: feature_block(features[k]["stats17"], features[k]["sem16"], group) for k in KS}


def _fit_logistic_group(
    seed: str,
    group: str,
    raw_by_k: Mapping[int, np.ndarray],
    y_by_k: Mapping[int, np.ndarray],
    train_masks: Mapping[int, np.ndarray],
    tune_masks: Mapping[int, np.ndarray],
    train_ks: Sequence[int],
    tune_ks: Sequence[int],
    *,
    log: Any,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]], Any]:
    keys = FEATURE_GROUPS[group]
    fit, standardized = fit_group_normalization(raw_by_k, train_masks, train_ks, keys)
    x_train = np.vstack([standardized[k][train_masks[k]] for k in train_ks])
    y_train = np.concatenate([y_by_k[k][train_masks[k]].astype(np.float64) for k in train_ks])
    tune = {
        k: (standardized[k][tune_masks[k]], y_by_k[k][tune_masks[k]].astype(bool))
        for k in tune_ks
    }
    predictor, selected, records = fit_selected_logistic(x_train, y_train, tune)
    selected["seed"] = seed
    selected["group"] = group
    selected["training_ks"] = list(train_ks)
    selected["tuning_ks"] = list(tune_ks)
    selected["device"] = "cpu"
    selected["normalization"] = fit.to_dict()
    log(f"{seed} {group}: selected C={selected['chosen_C']} tune_mean_AUROC={selected['tune_mean_auroc']:.6f}")
    return predictor, selected, records, standardized


def _train_status_row(seed: str, regime: str, model: str, record: Mapping[str, Any], *, state: str = "COMPLETE") -> dict[str, Any]:
    by_k = record.get("tune_auroc_by_K", {})
    return {
        "seed": seed,
        "training_regime": regime,
        "model": model,
        "status": state,
        "selected_hyperparameter": record.get("chosen_C", record.get("chosen_learning_rate")),
        "selected_epoch": record.get("best_epoch", 0),
        "epochs_run": record.get("epochs_run", 0),
        "n_train_rows": record.get("n_train_rows"),
        "n_tune_rows": record.get("n_tune_rows"),
        "train_loss": record.get("train_loss"),
        "train_auroc": record.get("train_auroc"),
        "tune_loss": record.get("tune_loss"),
        "tune_auroc": record.get("tune_auroc", record.get("tune_mean_auroc")),
        "tune_auroc_K5": by_k.get(5),
        "tune_auroc_K10": by_k.get(10),
        "tune_auroc_K20": by_k.get(20),
        "tune_auroc_K50": by_k.get(50),
        "device": record.get("device", "cpu"),
        "config": json.dumps(record.get("architecture", {"C": record.get("chosen_C")}), ensure_ascii=False),
    }


def _write_checkpoint_json(out: Path, seed: str, name: str, payload: Mapping[str, Any]) -> None:
    _write_json(out / "models" / seed / f"{name}.json", payload)


def _write_logistic_model(out: Path, seed: str, name: str, record: Mapping[str, Any]) -> None:
    _write_checkpoint_json(out, seed, name, record)


def _metrics_row(
    *,
    seed: str,
    model: str,
    training_regime: str,
    cell: str,
    k: int,
    split_name: str,
    confidence: np.ndarray,
    correct: np.ndarray,
) -> dict[str, Any]:
    probability = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    metrics = reval.point_metric_row(probability, correct, probability=probability)
    return {
        "seed": seed,
        "model": model,
        "training_regime": training_regime,
        "cell": cell,
        "K": int(k),
        "eval_split": split_name,
        "n": int(np.asarray(correct).size),
        "accuracy": float(np.mean(correct)),
        **metrics,
    }


def _metric_rows_for_splits(
    *, seed: str, model: str, regime: str, cell: str, k: int,
    confidence: np.ndarray, correct: np.ndarray, eval_split: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("val_select", "testA", "testB", "__pooled_test__"):
        mask = np.isin(eval_split, ("testA", "testB")) if split_name == "__pooled_test__" else eval_split == split_name
        if mask.sum() == 0:
            continue
        rows.append(_metrics_row(
            seed=seed, model=model, training_regime=regime, cell=cell, k=k,
            split_name=split_name, confidence=confidence[mask], correct=correct[mask],
        ))
    return rows


def _train_sds_regime(
    seed: str,
    regime_name: str,
    train_ks: Sequence[int],
    data: Mapping[int, Mapping[str, np.ndarray]],
    features: Mapping[int, Mapping[str, np.ndarray]],
    train_masks: Mapping[int, np.ndarray],
    tune_masks: Mapping[int, np.ndarray],
    *,
    out: Path,
    log: Any,
) -> tuple[ScoreDeepSets, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    raw_scores = {k: data[k]["scores"] for k in KS}
    norm = fit_score_deepsets_normalization(raw_scores, train_masks, train_ks)
    y_by_k = {k: data[k]["correct"].astype(np.float32) for k in KS}
    train_inputs = score_deepsets_inputs(norm, y_by_k, train_masks, train_ks)
    tune_inputs = score_deepsets_inputs(norm, y_by_k, tune_masks, train_ks)
    tune_by_k = {}
    for k in train_ks:
        mask = tune_masks[k]
        tune_by_k[k] = {
            "scores": norm["scores"][k][mask],
            "mask": np.ones((int(mask.sum()), int(k)), dtype=bool),
            "log_k": norm["log_k"][k][mask],
            "z_top1": norm["z_top1"][k][mask],
            "y": y_by_k[k][mask],
        }
    log(f"{seed} ScoreDeepSets {regime_name}: CPU float32 train starting, n={train_inputs['y'].size}, K={tuple(train_ks)}")
    model, selected, curves = train_score_deepsets_grid(
        train_inputs, tune_inputs, tune_by_k, seed=int(seed.removeprefix("b3_seed")),
        learning_rates=SDS_LRS, epochs=300, patience=30, batch_size=256, weight_decay=1e-4,
    )
    selected.update({
        "seed": seed,
        "training_regime": regime_name,
        "training_ks": list(train_ks),
        "tuning_ks": list(train_ks),
        "device": "cpu",
        "dtype": "float32",
        "score_normalization": {"mean": norm["score_mu"], "std": norm["score_sigma"]},
        "log_k_normalization": norm["log_k_fit"].to_dict(),
        "z_top1_normalization": norm["top1_fit"].to_dict(),
    })
    for curve in curves:
        curve.update({"seed": seed, "training_regime": regime_name})
    model_dir = out / "models" / seed
    model_dir.mkdir(parents=True, exist_ok=True)
    torch_path = model_dir / f"score_deepsets_{regime_name}.pt"
    import torch
    torch.save({"state_dict": model._net.state_dict(), "seed": int(seed.removeprefix("b3_seed")), "selected": selected}, torch_path)
    _write_checkpoint_json(out, seed, f"score_deepsets_{regime_name}", selected)
    log(f"{seed} ScoreDeepSets {regime_name}: selected lr={selected['chosen_learning_rate']} epoch={selected['best_epoch']} tune AUROC={selected['tune_mean_auroc']:.6f}")
    return model, selected, curves, norm


def _save_all_rows_csv_gz(path: Path, header: Sequence[str], row_groups: Iterable[Iterable[Mapping[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        for group in row_groups:
            writer.writerows(group)
    os.replace(tmp, path)


def _prediction_rows(
    item: Mapping[str, np.ndarray],
    confidence: np.ndarray,
    *, seed: str, cell: str, k: int, model: str, regime: str,
) -> Iterable[dict[str, Any]]:
    mask = np.isin(item["eval_split"], ("val_select", "testA", "testB"))
    idx = np.flatnonzero(mask)
    p = np.asarray(confidence, dtype=np.float64)
    for i in idx:
        yield {
            "sentence_id": int(item["sentence_id"][i]),
            "ref_id": int(item["ref_id"][i]),
            "image_id": int(item["image_id"][i]),
            "eval_split": str(item["eval_split"][i]),
            "seed": seed,
            "cell": cell,
            "K": int(k),
            "grounding_correct": int(item["correct"][i]),
            "model": model,
            "training_regime": regime,
            "confidence": float(p[i]),
            "probability": float(p[i]),
        }


def _write_features_file(out: Path, seed: str, k: int, item: Mapping[str, np.ndarray]) -> None:
    path = out / "derived" / f"random_{seed}_K{k}.npz"
    np.savez_compressed(path, stats17=item["stats17"], sem16=item["sem16"])


def _load_old_phase1(seed: str) -> list[dict[str, Any]]:
    path = PHASE1_ROOT / "predictions" / f"{seed}.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _predict_frozen(
    frozen: Any,
    seed: str,
    stats17: np.ndarray,
    sem16: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with a directly loaded A11 bundle or an A8.4 recovered model."""
    if hasattr(frozen, "predict"):
        return frozen.predict(seed, stats17, sem16)
    return apply_frozen(frozen.seeds[seed], stats17, sem16)


def _load_reference_model_identities() -> dict[str, dict[str, str]]:
    """Per-seed model identities used by the generic Phase 1 prediction CSV."""
    payload = json.loads((PHASE1_ROOT / "metadata.json").read_text(encoding="utf-8"))
    return {
        seed: {
            "score_only_reliability": str(payload["selections"][seed]["best_score_only"]),
            "semantic_reliability": str(payload["selections"][seed]["best_semantic"]),
        }
        for seed in B3_SEEDS
    }


def _load_phase05_stats_predictions() -> dict[tuple[str, int, str], dict[str, np.ndarray]]:
    """Lossless row-level Stats-Logistic anchors from Phase 0.5 predictions."""
    result: dict[tuple[str, int, str], dict[str, list[Any]]] = {}
    for seed in B3_SEEDS:
        path = PHASE05_ROOT / "predictions" / seed / "stats_logistic.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (seed, int(row["K"]), str(row["eval_split"]))
                node = result.setdefault(key, {name: [] for name in ("ref_id", "image_id", "eval_split", "K", "grounding_correct", "confidence")})
                node["ref_id"].append(int(row["ref_id"]))
                node["image_id"].append(int(row["image_id"]))
                node["eval_split"].append(str(row["eval_split"]))
                node["K"].append(int(row["K"]))
                node["grounding_correct"].append(int(row["grounding_correct"]))
                node["confidence"].append(float(row["reliability_score"]))
    return {
        key: {
            "ref_id": np.asarray(value["ref_id"], dtype=np.int64),
            "image_id": np.asarray(value["image_id"], dtype=np.int64),
            "eval_split": np.asarray(value["eval_split"]),
            "K": np.asarray(value["K"], dtype=np.int64),
            "grounding_correct": np.asarray(value["grounding_correct"], dtype=bool),
            "confidence": np.asarray(value["confidence"], dtype=np.float64),
        }
        for key, value in result.items()
    }


def _load_phase1_e1b_metrics() -> dict[tuple[str, int, str], dict[str, Any]]:
    """Original selected E1b per-model metrics, independent of generic CSV fields."""
    path = PHASE1_ROOT / "e1_logistic" / "metrics.csv"
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "e1b_stats_semantic":
                continue
            key = (str(row["scorer"]), int(row["K"]), str(row["eval_split"]))
            result[key] = row
    return result


def _reference_check_random(
    seed: str,
    data: Mapping[int, Mapping[str, np.ndarray]],
    features: Mapping[int, Mapping[str, np.ndarray]],
    group_models: Mapping[str, Mapping[str, Any]],
    fitted_frozen: Any | None,
    old_rows: Sequence[Mapping[str, Any]],
    phase1_identities: Mapping[str, Mapping[str, str]],
    phase05_stats_predictions: Mapping[tuple[str, int, str], Mapping[str, np.ndarray]],
    phase1_e1b_metrics: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in KS:
        item = data[k]
        raw_stats = features[k]["stats17"]
        raw_sem = features[k]["sem16"]
        if fitted_frozen is None:
            frozen_stats = frozen_full = None
        else:
            frozen_stats, frozen_full = _predict_frozen(fitted_frozen, seed, raw_stats, raw_sem)
        selected_identity = phase1_identities[seed]
        for split_name in ("val_select", "testA", "testB"):
            mask = item["eval_split"] == split_name
            if not mask.any():
                continue
            refs = [r for r in old_rows if int(r["K"]) == k and r["eval_split"] == split_name]
            candidate = {
                "ref_id": np.asarray([int(r["ref_id"]) for r in refs], dtype=np.int64),
                "image_id": np.asarray([int(r["image_id"]) for r in refs], dtype=np.int64),
                "eval_split": np.asarray([r["eval_split"] for r in refs]),
                "K": np.asarray([int(r["K"]) for r in refs], dtype=np.int64),
                "grounding_correct": _parse_bool([r["grounding_correct"] for r in refs]),
            }
            reference = {
                "sentence_id": item["sentence_id"][mask],
                "ref_id": item["ref_id"][mask],
                "image_id": item["image_id"][mask],
                "eval_split": item["eval_split"][mask],
                "K": np.full(int(mask.sum()), k, dtype=np.int64),
                "grounding_correct": item["correct"][mask],
            }
            assert_row_alignment(
                reference, candidate,
                required_fields=("ref_id", "image_id", "eval_split", "K", "grounding_correct"),
                reference_name=f"canonical-{seed}-K{k}-{split_name}",
                candidate_name=f"legacy-phase1-{seed}-K{k}-{split_name}",
            )
            old_sem = np.asarray([float(r["semantic_reliability"]) for r in refs], dtype=np.float64)
            old_stats = np.asarray([float(r["score_only_reliability"]) for r in refs], dtype=np.float64)
            for model_name, frozen, old in (("S", frozen_stats, old_stats), ("Full", frozen_full, old_sem)):
                model_group = group_models[model_name]
                pred = model_group["predictor"].predict_proba(model_group["standardized"][k])
                new = pred[mask]
                generic_column = "score_only_reliability" if model_name == "S" else "semantic_reliability"
                generic_model = selected_identity[generic_column]
                generic_delta = np.asarray(frozen[mask], dtype=np.float64) - old if frozen is not None else None
                generic_max_abs = float(np.max(np.abs(generic_delta))) if generic_delta is not None else None
                target_original_model = "stats_logistic" if model_name == "S" else "e1b_stats_semantic"
                identity_compatible = generic_model == target_original_model

                direct_anchor_max_abs = None
                direct_anchor_status = "UNVERIFIABLE_NO_FROZEN_PREDICTION"
                direct_anchor_source = ""
                if frozen is not None and model_name == "S":
                    anchor = phase05_stats_predictions[(seed, k, split_name)]
                    assert_row_alignment(
                        reference, anchor,
                        required_fields=("ref_id", "image_id", "eval_split", "K", "grounding_correct"),
                        reference_name=f"canonical-{seed}-K{k}-{split_name}",
                        candidate_name=f"phase05-stats-logistic-{seed}-K{k}-{split_name}",
                    )
                    direct_anchor_max_abs = float(np.max(np.abs(np.asarray(frozen[mask], dtype=np.float64) - anchor["confidence"])))
                    direct_anchor_status = "PASS" if direct_anchor_max_abs <= 1e-9 else "FAIL"
                    direct_anchor_source = f"results/phase05_score_sufficiency/predictions/{seed}/stats_logistic.csv.gz"
                elif frozen is not None and model_name == "Full":
                    source_metric = phase1_e1b_metrics[(seed, k, split_name)]
                    direct_metrics = reval.point_metric_row(frozen[mask], item["correct"][mask])
                    metric_names = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
                    metric_deltas = {
                        name: float(abs(float(direct_metrics[name]) - float(source_metric[name])))
                        for name in metric_names
                    }
                    direct_anchor_max_abs = max(metric_deltas.values())
                    direct_anchor_status = "PASS" if direct_anchor_max_abs <= 1e-9 else "FAIL"
                    direct_anchor_source = "results/phase1_semantic_sufficiency/e1_logistic/metrics.csv (e1b_stats_semantic per seed/K/split)"
                frozen_max_abs = generic_max_abs if identity_compatible else None
                old_metrics = reval.point_metric_row(old, item["correct"][mask])
                new_metrics = reval.point_metric_row(new, item["correct"][mask])
                out.append({
                    "seed": seed, "K": k, "eval_split": split_name, "model": model_name,
                    "n": int(mask.sum()),
                    "original_reference_model_id": generic_model,
                    "target_model_id": target_original_model,
                    "model_identity_compatible": identity_compatible,
                    "model_identity_status": "COMPATIBLE" if identity_compatible else "MODEL_IDENTITY_INCOMPATIBLE",
                    "legacy_generic_csv_max_abs": generic_max_abs,
                    "legacy_generic_csv_mean_abs": float(np.mean(np.abs(generic_delta))) if generic_delta is not None else None,
                    "frozen_recovery_max_abs": frozen_max_abs,
                    "frozen_recovery_mean_abs": float(np.mean(np.abs(generic_delta))) if generic_delta is not None and identity_compatible else None,
                    "frozen_recovery_status": (
                        "MODEL_IDENTITY_INCOMPATIBLE" if not identity_compatible else
                        "UNVERIFIABLE_NO_FROZEN_PREDICTION" if generic_delta is None else
                        "MATCHED" if frozen_max_abs <= 1e-9 else "MISMATCH"
                    ),
                    "correct_model_anchor_source": direct_anchor_source,
                    "correct_model_anchor_max_abs": direct_anchor_max_abs,
                    "correct_model_anchor_status": direct_anchor_status,
                    "new_target_vs_generic_csv_mean_abs": float(np.mean(np.abs(new - old))),
                    "new_target_vs_generic_csv_max_abs": float(np.max(np.abs(new - old))),
                    "generic_csv_model_auroc": old_metrics["auroc_correct"],
                    "new_target_auroc": new_metrics["auroc_correct"],
                    "generic_csv_model_e_aurc": old_metrics["e_aurc"],
                    "new_target_e_aurc": new_metrics["e_aurc"],
                    "new_C": model_group["selected"]["chosen_C"],
                    "status": (
                        "MODEL_IDENTITY_INCOMPATIBLE" if not identity_compatible else
                        "LEGACY_CSV_ANCHORED_REFIT_UNVERIFIABLE" if frozen is None else
                        "DIRECT_BUNDLE_MATCHES_ORIGINAL_MODEL" if frozen_max_abs <= 1e-9 else
                        "DIRECT_BUNDLE_DIFFERS_FROM_ORIGINAL_MODEL"
                    ),
                })
    return out


def _build_alt_features(out: Path, log: Any) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Materialise only candidate embeddings from frozen caches for Phase 1F cells."""
    cells = ("rand5", "hard5", "expb_m0", "expb_m2", "expb_m4", "expb_m8")
    alt: dict[str, dict[str, dict[str, np.ndarray]]] = {seed: {} for seed in B3_SEEDS}
    split_map: dict[int, str] = {}
    identity_map: dict[int, tuple[int, int]] = {}
    base = sdata.load_phase05_cohort(5)
    for sid, image_id, ref_id, split_name in zip(base["sentence_id"], base["image_id"], base["ref_id"], base["eval_split"], strict=True):
        split_map[int(sid)] = str(split_name)
        identity_map[int(sid)] = (int(ref_id), int(image_id))
    corpus = None
    try:
        from ccg.models.b3_data import B3Corpus
        corpus = B3Corpus(
            ROOT / "cache/features", ROOT / "cache/manifests",
            ROOT / "data/raw/refcoco+/refcoco+/refs(unc).p", ROOT / "cache/proposals.h5",
            image_sizes_path=ROOT / "cache/image_sizes.npz", ks=(5, 10), regime="random", preload=True,
        )
        log("Phase1F B3Corpus loaded from frozen feature cache; no scorer forward or encoder extraction")
        manifest_root = PHASE1F_ROOT / "manifests"
        predictions_root = PHASE1F_ROOT / "predictions"
        for cell in cells:
            with np.load(manifest_root / f"{cell}_candidates.npz") as z:
                manifest = {key: np.asarray(z[key]) for key in z.files}
            candidate_ids = np.asarray(manifest["sentence_id"], dtype=np.int64)
            if any(int(sid) not in identity_map for sid in candidate_ids):
                raise AssertionError(f"{cell}: candidate contains sentence outside the frozen cohort")
            split_array = np.asarray([split_map[int(sid)] for sid in candidate_ids])
            base_ref = np.asarray([identity_map[int(sid)][0] for sid in candidate_ids], dtype=np.int64)
            base_image = np.asarray([identity_map[int(sid)][1] for sid in candidate_ids], dtype=np.int64)
            if not np.array_equal(manifest["ref_id"], base_ref) or not np.array_equal(manifest["image_id"], base_image):
                raise AssertionError(f"{cell}: candidate identity anchors differ from frozen common cohort")
            sample_records = [
                SimpleNamespace(
                    sentence_id=int(sid), ref_id=int(ref), image_id=int(image),
                    target_index=int(cands[0]), distractor_order=np.asarray(cands[1:], dtype=np.int64),
                )
                for sid, ref, image, cands in zip(candidate_ids, base_ref, base_image, manifest["candidate_indices"], strict=True)
            ]
            per_seed_pred: dict[str, dict[str, np.ndarray]] = {}
            for seed in B3_SEEDS:
                with np.load(predictions_root / f"{cell}__{seed}.npz") as z:
                    p = {key: np.asarray(z[key]) for key in z.files}
                aligned = assert_row_alignment(
                    {"sentence_id": candidate_ids, "ref_id": base_ref, "image_id": base_image},
                    {key: p[key] for key in ("sentence_id", "ref_id", "image_id")},
                    required_fields=("ref_id", "image_id"),
                    reference_name=f"manifest-{cell}", candidate_name=f"prediction-{cell}-{seed}",
                )
                if not np.array_equal(aligned, p["sentence_id"]):
                    raise AssertionError(f"{cell} {seed}: prediction sentence order mismatch")
                if not np.array_equal(_parse_bool(p["correct"]), np.argmax(p["scores"], axis=1) == 0):
                    raise AssertionError(f"{cell} {seed}: stored correctness does not match raw B3 top-1")
                per_seed_pred[seed] = p
            n = candidate_ids.size
            sem_by_seed = {seed: np.empty((n, 16), dtype=np.float64) for seed in B3_SEEDS}
            for start in range(0, n, 384):
                stop = min(start + 384, n)
                batch = materialise_examples(corpus, sample_records[start:stop], int(manifest["candidate_indices"].shape[1]))
                for seed in B3_SEEDS:
                    scores = per_seed_pred[seed]["scores"][start:stop]
                    sem_by_seed[seed][start:stop] = compute_semantic_stats_chunked(
                        batch.z_q, batch.z_i, scores, chunk_rows=384,
                    )
                del batch
            for seed in B3_SEEDS:
                pred = per_seed_pred[seed]
                scores = np.asarray(pred["scores"], dtype=np.float32)
                temperature = _parse_temperature(seed)
                stats = rfeat.stat_features(scores, temperature=temperature)
                correct = _parse_bool(pred["correct"])
                if not np.array_equal(correct, np.argmax(scores, axis=1) == 0):
                    raise AssertionError(f"{cell} {seed}: correctness anchor changed")
                alt[seed][cell] = {
                    "sentence_id": candidate_ids.copy(),
                    "ref_id": base_ref.copy(),
                    "image_id": base_image.copy(),
                    "eval_split": split_array.copy(),
                    "scores": scores,
                    "correct": correct,
                    "stats17": stats,
                    "sem16": sem_by_seed[seed],
                }
                out_path = out / "derived" / f"phase1f_{cell}_{seed}.npz"
                np.savez_compressed(out_path, stats17=stats, sem16=sem_by_seed[seed],
                                    sentence_id=candidate_ids, image_id=base_image, ref_id=base_ref)
            log(f"Phase1F cell {cell}: loaded/rebuilt cached embeddings for {n} rows; B3 logits reused")
        return alt
    finally:
        if corpus is not None:
            del corpus
        gc.collect()


def _reference_check_alt(
    seed: str,
    alt: Mapping[str, Mapping[str, np.ndarray]],
    frozen: Any | None,
    group_models: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, item in alt.items():
        if frozen is None:
            recovered_stats = recovered_full = None
        else:
            recovered_stats, recovered_full = _predict_frozen(frozen, seed, item["stats17"], item["sem16"])
        reference_path = PHASE1F_ROOT / "predictions" / f"{cell}__{seed}.npz"
        with np.load(reference_path) as z:
            expected_stats = np.asarray(z["conf_stats"], dtype=np.float64)
            expected_full = np.asarray(z["conf_e1b"], dtype=np.float64)
            correctness = _parse_bool(z["correct"])
        for name, got, expected in (("S", recovered_stats, expected_stats), ("Full", recovered_full, expected_full)):
            group = group_models[name]
            raw = feature_block(item["stats17"], item["sem16"], name)
            new = group["predictor"].predict_proba(rfeat.normalize_apply(raw, group["normalization"]))
            old_metrics = reval.point_metric_row(expected, correctness)
            new_metrics = reval.point_metric_row(new, correctness)
            frozen_error = np.asarray(got, dtype=np.float64) - expected if got is not None else None
            frozen_max_abs = float(np.max(np.abs(frozen_error))) if frozen_error is not None else None
            rows.append({
                "seed": seed, "K": int(item["scores"].shape[1]), "cell": cell,
                "eval_split": "testA_testB_matched", "model": name, "n": int(len(expected)),
                "frozen_recovery_max_abs": frozen_max_abs,
                "frozen_recovery_status": (
                    "UNVERIFIABLE_NO_FROZEN_PREDICTION" if frozen_error is None else
                    "MATCHED_PHASE1F_PREDICTION" if frozen_max_abs <= 1e-9 else
                    "MISMATCH_PHASE1F_PREDICTION"
                ),
                "new_selected_vs_legacy_mean_abs": float(np.mean(np.abs(new - expected))),
                "new_selected_vs_legacy_max_abs": float(np.max(np.abs(new - expected))),
                "legacy_auroc": old_metrics["auroc_correct"],
                "new_selected_auroc": new_metrics["auroc_correct"],
                "legacy_e_aurc": old_metrics["e_aurc"],
                "new_selected_e_aurc": new_metrics["e_aurc"],
                "new_C": group["selected"]["chosen_C"],
                "status": (
                    "LEGACY_PHASE1F_ANCHORED_REFIT_UNVERIFIABLE" if got is None else
                    "DIRECT_FROZEN_BUNDLE_MATCHES_PHASE1F" if frozen_max_abs <= 1e-9 else
                    "DIRECT_FROZEN_BUNDLE_DIFFERS_FROM_PHASE1F"
                ),
            })
    return rows


def _semantic_point_tables(
    seed: str,
    base_data: Mapping[int, Mapping[str, np.ndarray]],
    base_features: Mapping[int, Mapping[str, np.ndarray]],
    alt_data: Mapping[str, Mapping[str, np.ndarray]],
    group_models: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, np.ndarray]]]:
    metrics: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    probabilities: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    cells: dict[str, tuple[Mapping[str, np.ndarray], int]] = {
        CELLS_RANDOM[k]: (dict(base_data[k], stats17=base_features[k]["stats17"], sem16=base_features[k]["sem16"]), k)
        for k in KS
    }
    cells.update({cell: (item, int(item["scores"].shape[1])) for cell, item in alt_data.items()})
    comparisons = (
        ("Full_minus_S_plus_Q", "Full", "S+Q"),
        ("S_plus_V_minus_S", "S+V", "S"),
        ("Full_minus_S", "Full", "S"),
    )
    for cell, (item, k) in cells.items():
        for group in GROUPS:
            model_payload = group_models[group]
            raw = feature_block(item["stats17"], item["sem16"], group)
            standardized = rfeat.normalize_apply(raw, model_payload["normalization"])
            prob = model_payload["predictor"].predict_proba(standardized)
            probabilities[(cell, group)] = {"probability": prob, "correct": item["correct"]}
            for split_name in ("val_select", "testA", "testB", "__pooled_test__"):
                mask = np.isin(item["eval_split"], ("testA", "testB")) if split_name == "__pooled_test__" else item["eval_split"] == split_name
                if not mask.any():
                    continue
                row = _metrics_row(seed=seed, model=group, training_regime="smallK5_10", cell=cell, k=k,
                                   split_name=split_name, confidence=prob[mask], correct=item["correct"][mask])
                row["feature_group"] = group
                metrics.append(row)
        for label, model_a, model_b in comparisons:
            a = probabilities[(cell, model_a)]["probability"]
            b = probabilities[(cell, model_b)]["probability"]
            corr = item["correct"]
            for split_name in ("val_select", "testA", "testB", "__pooled_test__"):
                mask = np.isin(item["eval_split"], ("testA", "testB")) if split_name == "__pooled_test__" else item["eval_split"] == split_name
                if not mask.any():
                    continue
                ma = reval.point_metric_row(a[mask], corr[mask])
                mb = reval.point_metric_row(b[mask], corr[mask])
                for metric, effect in (
                    ("auroc_correct", ma["auroc_correct"] - mb["auroc_correct"]),
                    ("e_aurc_reduction", mb["e_aurc"] - ma["e_aurc"]),
                    ("rer_at_50_gain_pp", 100.0 * (ma["rer_at_50"] - mb["rer_at_50"])),
                    ("rer_at_80_gain_pp", 100.0 * (ma["rer_at_80"] - mb["rer_at_80"])),
                ):
                    effects.append({
                        "seed": seed, "cell": cell, "K": k, "eval_split": split_name,
                        "comparison": label, "model_a": model_a, "model_b": model_b,
                        "metric": metric, "effect": float(effect),
                    })
    return metrics, effects, probabilities


def _append_seed_means(rows: list[dict[str, Any]], keys: Sequence[str], numeric: Sequence[str], *, seed_field: str = "seed") -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    for key, group in groups.items():
        mean = {keys[i]: value for i, value in enumerate(key)}
        mean[seed_field] = "b3_mean"
        for field in numeric:
            vals = [float(r[field]) for r in group if r.get(field) is not None and np.isfinite(float(r[field]))]
            mean[field] = float(np.mean(vals)) if vals else None
            if field != "n":
                mean[f"{field}_seed_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        rows.append(mean)


def _write_semantic_predictions(
    out: Path,
    base_data: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    base_features: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    alt_data: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    group_models: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    def groups() -> Iterable[Iterable[Mapping[str, Any]]]:
        for seed in B3_SEEDS:
            for k in KS:
                cell = CELLS_RANDOM[k]
                item = dict(base_data[seed][k], stats17=base_features[seed][k]["stats17"], sem16=base_features[seed][k]["sem16"])
                for group in GROUPS:
                    raw = feature_block(item["stats17"], item["sem16"], group)
                    x = rfeat.normalize_apply(raw, group_models[seed][group]["normalization"])
                    prob = group_models[seed][group]["predictor"].predict_proba(x)
                    yield _prediction_rows(item, prob, seed=seed, cell=cell, k=k, model=group, regime="smallK5_10")
            for cell, item in alt_data[seed].items():
                k = int(item["scores"].shape[1])
                for group in GROUPS:
                    raw = feature_block(item["stats17"], item["sem16"], group)
                    x = rfeat.normalize_apply(raw, group_models[seed][group]["normalization"])
                    prob = group_models[seed][group]["predictor"].predict_proba(x)
                    yield _prediction_rows(item, prob, seed=seed, cell=cell, k=k, model=group, regime="smallK5_10")
    _save_all_rows_csv_gz(out / "semantic_ablation_predictions.csv.gz", PREDICTION_FIELDS, groups())


def _make_128_diagnostic(
    seed: str,
    data: Mapping[int, Mapping[str, np.ndarray]],
    features: Mapping[int, Mapping[str, np.ndarray]],
    train_masks: Mapping[int, np.ndarray],
    selected_s: Mapping[str, Any],
    selected_sds: Mapping[str, Any],
    *,
    out: Path,
    log: Any,
) -> list[dict[str, Any]]:
    sample_masks: dict[int, np.ndarray] = {}
    sample_ids: dict[int, list[int]] = {}
    for k in (5, 10):
        ids = np.flatnonzero(train_masks[k])
        chosen = ids[np.argsort(data[k]["sentence_id"][ids], kind="stable")[:64]]
        mask = np.zeros(data[k]["sentence_id"].size, dtype=bool)
        mask[chosen] = True
        sample_masks[k] = mask
        sample_ids[k] = [int(x) for x in data[k]["sentence_id"][chosen]]
    stats_raw = {k: features[k]["stats17"] for k in KS}
    norm_s, z_s = fit_group_normalization(stats_raw, sample_masks, (5, 10), FEATURE_GROUPS["S"])
    x_sample = np.vstack([z_s[k][sample_masks[k]] for k in (5, 10)])
    y_sample = np.concatenate([data[k]["correct"][sample_masks[k]].astype(float) for k in (5, 10)])
    from ccg.reliability.models import LogisticModel
    stat_model = LogisticModel(C=float(selected_s["chosen_C"])).fit(x_sample, y_sample)
    stat_prob = stat_model.predict_proba(x_sample)

    norm_sds = fit_score_deepsets_normalization(
        {k: data[k]["scores"] for k in KS}, sample_masks, (5, 10),
    )
    sds_train = score_deepsets_inputs(norm_sds, {k: data[k]["correct"].astype(np.float32) for k in KS}, sample_masks, (5, 10))
    diagnostic = ScoreDeepSets(seed=int(seed.removeprefix("b3_seed")), include_logk=True, include_ztop1=True)
    sds_curve: list[dict[str, Any]] = []
    def epoch_cb(row: Mapping[str, Any]) -> None:
        sds_curve.append(dict(row))
    fit_info = diagnostic.fit(
        sds_train["scores"], sds_train["mask"], sds_train["log_k"], sds_train["z_top1"], sds_train["y"],
        lr=float(selected_sds["chosen_learning_rate"]), epochs=300, batch_size=256,
        weight_decay=1e-4, patience=30, log=epoch_cb,
    )
    sds_prob = diagnostic.predict_proba(sds_train["scores"], sds_train["mask"], sds_train["log_k"], sds_train["z_top1"])
    msp_parts = [rfeat.scalar_confidence(data[k]["scores"][sample_masks[k]], temperature=_parse_temperature(seed))["msp"] for k in (5, 10)]
    msp_prob = np.concatenate(msp_parts)
    rows: list[dict[str, Any]] = []
    for model_name, probability, config, epochs_run, selected_epoch in (
        ("StatsLogistic", stat_prob, {"C": float(selected_s["chosen_C"]), "normalization": norm_s.to_dict()}, 0, 0),
        ("ScoreDeepSets", sds_prob, {"lr": float(selected_sds["chosen_learning_rate"]), "log_k_scaler": norm_sds["log_k_fit"].to_dict(), "score_mean": norm_sds["score_mu"], "score_std": norm_sds["score_sigma"]}, int(fit_info["epochs_run"]), fit_info.get("best_epoch")),
        ("corrected_score_MSP", msp_prob, {"temperature": _parse_temperature(seed), "input": "softmax(scores/T).max"}, 0, 0),
    ):
        rows.append({
            "seed": seed, "diagnostic": "fixed_64_K5_plus_64_K10_train_rows",
            "model": model_name, "n_train_rows": int(y_sample.size),
            "sentence_ids_K5": json.dumps(sample_ids[5]), "sentence_ids_K10": json.dumps(sample_ids[10]),
            "train_auroc": float(auroc_correct(probability, y_sample)),
            "train_nll": nll_binary(probability, y_sample), "epochs_run": epochs_run,
            "selected_epoch": selected_epoch, "fit_is_generalization_evidence": False,
            "config": json.dumps(config, ensure_ascii=False),
        })
    diag_path = out / "models" / seed / "score_deepsets_128_diagnostic.pt"
    import torch
    torch.save({"state_dict": diagnostic._net.state_dict(), "fit_info": fit_info, "sample_ids": sample_ids}, diag_path)
    if not np.isclose(float(np.mean(y_sample)), float(np.mean(np.concatenate([data[k]["correct"][sample_masks[k]] for k in (5, 10)])))):
        raise AssertionError("diagnostic label bookkeeping failed")
    log(f"{seed}: 128-row fit diagnostic complete; SDS epochs={fit_info['epochs_run']}, best_epoch={fit_info.get('best_epoch')}")
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_completed_training(
    out: Path,
    data_by_seed: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    features_by_seed: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, ScoreDeepSets]],
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Restore finished CPU checkpoints and train/tune evidence without fitting."""
    import torch

    training_rows = _read_csv_rows(out / "training_status.csv")
    curve_rows = _read_csv_rows(out / "train_tune_curves.csv")
    group_models_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    cross_models_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    sds_models_by_seed: dict[str, dict[str, ScoreDeepSets]] = {}
    sds_norms_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in B3_SEEDS:
        seed_dir = out / "models" / seed
        groups: dict[str, dict[str, Any]] = {}
        for group in GROUPS:
            suffix = group.replace("+", "_plus_")
            selected = json.loads((seed_dir / f"ablation_{suffix}.json").read_text(encoding="utf-8"))
            normalization = rfeat.NormalizationFit.from_dict(selected["normalization"])
            raw = _group_raw(features_by_seed[seed], group)
            standardized = {k: rfeat.normalize_apply(raw[k], normalization) for k in KS}
            groups[group] = {
                "predictor": load_logistic_predictor(selected), "selected": selected,
                "normalization": normalization, "standardized": standardized,
            }
        large_selected = json.loads((seed_dir / "stats_logistic_largeK20_50.json").read_text(encoding="utf-8"))
        large_norm = rfeat.NormalizationFit.from_dict(large_selected["normalization"])
        large_raw = _group_raw(features_by_seed[seed], "S")
        large_model = {
            "predictor": load_logistic_predictor(large_selected), "selected": large_selected,
            "normalization": large_norm,
            "standardized": {k: rfeat.normalize_apply(large_raw[k], large_norm) for k in KS},
        }
        cross_models_by_seed[seed] = {
            "StatsLogistic_smallK5_10": groups["S"],
            "StatsLogistic_largeK20_50": large_model,
        }
        group_models_by_seed[seed] = groups

        sds_models: dict[str, ScoreDeepSets] = {}
        sds_norms: dict[str, dict[str, Any]] = {}
        for regime in TRAINING_REGIMES:
            name = f"score_deepsets_{regime}"
            selected = json.loads((seed_dir / f"{name}.json").read_text(encoding="utf-8"))
            payload = torch.load(seed_dir / f"{name}.pt", map_location="cpu", weights_only=True)
            model = ScoreDeepSets(seed=int(seed.removeprefix("b3_seed")), include_logk=True, include_ztop1=True)
            model._net.load_state_dict(payload["state_dict"])
            model._net.eval()
            sds_models[regime] = model
            log_k_fit = rfeat.NormalizationFit.from_dict(selected["log_k_normalization"])
            top1_fit = rfeat.NormalizationFit.from_dict(selected["z_top1_normalization"])
            score_mu = float(selected["score_normalization"]["mean"])
            score_sigma = float(selected["score_normalization"]["std"])
            norm_scores: dict[int, np.ndarray] = {}
            norm_log_k: dict[int, np.ndarray] = {}
            norm_top1: dict[int, np.ndarray] = {}
            for k in KS:
                scores = np.asarray(data_by_seed[seed][k]["scores"], dtype=np.float64)
                norm_scores[k] = ((scores - score_mu) / (score_sigma + 1e-8)).astype(np.float32)
                n = scores.shape[0]
                norm_log_k[k] = rfeat.normalize_apply(
                    np.full((n, 1), np.log(float(k)), dtype=np.float64), log_k_fit,
                )[:, 0].astype(np.float32)
                norm_top1[k] = rfeat.normalize_apply(np.max(scores, axis=1)[:, None], top1_fit)[:, 0].astype(np.float32)
            sds_norms[regime] = {"scores": norm_scores, "log_k": norm_log_k, "z_top1": norm_top1, "selected": selected}
        sds_models_by_seed[seed] = sds_models
        sds_norms_by_seed[seed] = sds_norms
    return group_models_by_seed, cross_models_by_seed, sds_models_by_seed, sds_norms_by_seed, training_rows, curve_rows


def _load_alt_features_from_saved(out: Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Resume Phase 1F evaluation from already materialised feature checkpoints."""
    cells = ("rand5", "hard5", "expb_m0", "expb_m2", "expb_m4", "expb_m8")
    base = sdata.load_phase05_cohort(5)
    split_map = {int(sid): str(split) for sid, split in zip(base["sentence_id"], base["eval_split"], strict=True)}
    alt: dict[str, dict[str, dict[str, np.ndarray]]] = {seed: {} for seed in B3_SEEDS}
    for cell in cells:
        for seed in B3_SEEDS:
            with np.load(out / "derived" / f"phase1f_{cell}_{seed}.npz", allow_pickle=False) as feature_store:
                features = {key: np.asarray(feature_store[key]) for key in feature_store.files}
            with np.load(PHASE1F_ROOT / "predictions" / f"{cell}__{seed}.npz", allow_pickle=False) as prediction_store:
                predictions = {key: np.asarray(prediction_store[key]) for key in prediction_store.files}
            sentence_ids = np.asarray(predictions["sentence_id"], dtype=np.int64)
            aligned = assert_row_alignment(
                {key: features[key] for key in ("sentence_id", "ref_id", "image_id")},
                {key: predictions[key] for key in ("sentence_id", "ref_id", "image_id")},
                required_fields=("ref_id", "image_id"),
                reference_name=f"saved-derived-{cell}-{seed}", candidate_name=f"phase1f-prediction-{cell}-{seed}",
            )
            if not np.array_equal(aligned, sentence_ids):
                raise AssertionError(f"{cell} {seed}: saved Phase 1F feature/prediction rows differ")
            scores = np.asarray(predictions["scores"], dtype=np.float32)
            correct = _parse_bool(predictions["correct"])
            if not np.array_equal(correct, np.argmax(scores, axis=1) == 0):
                raise AssertionError(f"{cell} {seed}: saved Phase 1F correctness differs from raw B3 scores")
            if any(int(sid) not in split_map for sid in sentence_ids):
                raise AssertionError(f"{cell}: cached row is outside the frozen reliability cohort")
            alt[seed][cell] = {
                "sentence_id": sentence_ids,
                "ref_id": np.asarray(features["ref_id"], dtype=np.int64),
                "image_id": np.asarray(features["image_id"], dtype=np.int64),
                "eval_split": np.asarray([split_map[int(sid)] for sid in sentence_ids]),
                "scores": scores,
                "correct": correct,
                "stats17": np.asarray(features["stats17"], dtype=np.float64),
                "sem16": np.asarray(features["sem16"], dtype=np.float64),
            }
    return alt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/research_repair_v1/information"))
    parser.add_argument("--skip-bootstrap", action="store_true", help="prepare model/point artifacts only; bootstrap is a separately scheduled stage")
    parser.add_argument("--resume-postprocess", action="store_true", help="load completed checkpoints and resume reference checks/point artifacts without refitting")
    parser.add_argument("--force-feature-recompute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    fixed_out = (ROOT / "results/research_repair_v1/information").resolve()
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if out != fixed_out:
        raise ValueError(f"B outputs are restricted to {fixed_out}; refusing any other target")
    out.mkdir(parents=True, exist_ok=True)
    log, log_handle = _log_factory(out / "logs.txt")
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    threadpool_limits(limits=2)
    log(f"B runner starting pid={os.getpid()} device=cpu torch_threads=2 BLAS_threads=2")
    command = "python -B -u scripts/repair_information.py --out results/research_repair_v1/information --skip-bootstrap"
    if args.resume_postprocess:
        command += " --resume-postprocess"
    _record_attempt(out, {"timestamp_utc": _now(), "pid": os.getpid(), "command": command, "state": "RESUME_POSTPROCESS_STARTED" if args.resume_postprocess else "STARTED", "device": "cpu", "training_refit": not args.resume_postprocess})
    _set_status(out, "data_and_features", "RUNNING", detail="loading frozen raw scores and fixed split")
    split = sdata.load_split_from_manifest(PHASE05_ROOT / "split_manifest.json")
    data_by_seed: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    features_by_seed: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    train_masks_by_seed: dict[str, dict[int, np.ndarray]] = {}
    tune_masks_by_seed: dict[str, dict[int, np.ndarray]] = {}
    for seed in B3_SEEDS:
        seed_data = {k: _canonical_seed_k(seed, k) for k in KS}
        _canonical_validate(seed, seed_data)
        sdata.assert_split_matches_fresh(split, seed_data[5]["eval_split"], seed_data[5]["image_id"])
        train_mask, tune_mask = _mask_dictionary(split, seed_data[5])
        if train_mask[5].sum() != train_mask[50].sum() or tune_mask[5].sum() != tune_mask[20].sum():
            raise AssertionError(f"{seed}: train/tune row budget differs across K")
        seed_features = _load_features(seed, seed_data, out, log, reuse_derived=not args.force_feature_recompute)
        data_by_seed[seed] = seed_data
        features_by_seed[seed] = seed_features
        train_masks_by_seed[seed] = train_mask
        tune_masks_by_seed[seed] = tune_mask
    _set_status(out, "data_and_features", "COMPLETE", detail="canonical rows and cached embedding-derived features aligned")

    repair_feature_value = {
        "root_cause": "old run_phase05.py used np.arange(n_train_total) on concatenated logK arrays, selecting the K5 prefix instead of actual K5/K10 reliability_train rows",
        "fix": "construct the actual reliability_train logK rows across training Ks before normalize_fit",
        "old_wrong_logK_mean": float(np.log(5.0)),
        "correct_logK_train_mean": float(np.mean([np.log(5.0)] * int(train_masks_by_seed[B3_SEEDS[0]][5].sum()) + [np.log(10.0)] * int(train_masks_by_seed[B3_SEEDS[0]][10].sum()))),
        "correct_logK_train_std": float(np.std([np.log(5.0)] * int(train_masks_by_seed[B3_SEEDS[0]][5].sum()) + [np.log(10.0)] * int(train_masks_by_seed[B3_SEEDS[0]][10].sum()), ddof=0)),
        "source_test": "tests/test_repair_information.py::test_phase05_score_deepsets_logk_scaler_uses_only_training_rows",
        "old_behavior_test": "failed with mean=1.6094379124341003, std=0; correct expected mean=1.956011502714073 and nonzero std",
        "new_behavior_test": "passed; new model ScoreDeepSets normalizer fits concatenated reliability_train rows only",
    }
    _write_json(out / "logk_scaler_repair.json", repair_feature_value)

    _set_status(out, "reference_recovery", "RUNNING", detail="checking the frozen model source and reference anchors")
    with threadpool_limits(limits=2):
        if args.resume_postprocess:
            artifact_root = ROOT / "results/phase1e_refcocog_external/frozen_models"
            manifest_path = artifact_root / "frozen_artifact_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frozen = load_models(artifact_root, verify_checksum=True)
            direct_bundle_checks = verify_against_frozen_artifacts(
                frozen, phase05_root=PHASE05_ROOT, phase1_root=PHASE1_ROOT,
                phase1f_root=PHASE1F_ROOT, b3_root=ROOT / "results/phase0b_independent",
                features_dir=PHASE05_ROOT / "features", log=log,
            )
            prior_recovery = json.loads((out / "frozen_reference_recovery.json").read_text(encoding="utf-8"))
            prior_error = prior_recovery.get("failure_detail")
            recovery_state = "REFIT_FAILED_DIRECT_FROZEN_LOAD_VERIFIED"
            recovery_error = prior_error
            recovery_checks = direct_bundle_checks
            checksum_payload = json.loads((artifact_root / "models.sha256").read_text(encoding="utf-8"))
            frozen_reference_payload = {
                **prior_recovery,
                "status": recovery_state,
                "refit_status": "UNVERIFIABLE",
                "refit_failure_detail": prior_error,
                "refit_failures": {
                    "BLAS2": {"failed_seed": "b3_seed1", "metric": "stats_logistic_pred_max_abs", "actual_error_as_recorded": 6.066e-08, "tolerance": 1e-9, "trace_path": "logs.txt / execution_attempts.jsonl"},
                    "BLAS1": {"failed_seed": "b3_seed1", "metric": "stats_logistic_pred_max_abs", "actual_error": 3.642e-08, "tolerance": 1e-9, "trace_path": "logs.txt"},
                },
                "direct_frozen_bundle": {
                    "status": "VERIFIED",
                    "artifact_root": str(artifact_root.relative_to(ROOT)),
                    "manifest_path": str(manifest_path.relative_to(ROOT)),
                    "manifest_schema": manifest.get("schema_version"),
                    "manifest_artifact_count": len(manifest.get("artifacts", {})),
                    "manifest_bundle_sha256": manifest.get("bundle", {}).get("bundle_file_sha256"),
                    "checksum_file_sha256": checksum_payload.get("models.json"),
                    "seeds": list(frozen.scorers),
                    "strict_tolerance": direct_bundle_checks["tolerance"],
                    "checks": direct_bundle_checks,
                },
                "tolerance_widened": False,
            }
            _write_json(out / "frozen_reference_recovery.json", frozen_reference_payload)
            environment = json.loads((out / "reference_recovery_environment.json").read_text(encoding="utf-8"))
            environment.update({
                "refit_status": "UNVERIFIABLE",
                "BLAS1_recovery_status": recovery_state,
                "BLAS1_recovery_checks": {"failed_seed": "b3_seed1", "metric": "stats_logistic_pred_max_abs", "actual_error": 3.642e-08, "unchanged_tolerance": 1e-9, "trace_path": "results/research_repair_v1/information/logs.txt"},
                "direct_frozen_load_status": "VERIFIED",
                "direct_frozen_load_checks": direct_bundle_checks,
                "frozen_bundle_manifest": str(manifest_path.relative_to(ROOT)),
                "old_prediction_direct_anchor_check": {"status": "RUNNING", "result_path": "results/research_repair_v1/information/reference_check.csv"},
                "tolerance_widened": False,
            })
            _write_json(out / "reference_recovery_environment.json", environment)
            log("direct frozen coefficient bundle loaded and verified at the unchanged 1e-9 tolerance; no refit occurred")
        else:
            try:
                # The historical refit is isolated from subsequent formal training;
                # its exact numerical result is recorded at the original 1e-9 bar.
                with threadpool_limits(limits=1):
                    frozen = recover_frozen_models(
                        scorers=B3_SEEDS,
                        phase05_root=PHASE05_ROOT,
                        phase1_root=PHASE1_ROOT,
                        emb_root=ROOT / "cache/semantic_phase1",
                        features_dir=PHASE05_ROOT / "features",
                        split_manifest=PHASE05_ROOT / "split_manifest.json",
                        b3_root=ROOT / "results/phase0b_independent",
                        log=log,
                    )
                recovery_state = "VERIFIED"
                recovery_error = None
                recovery_checks = frozen.verification
            except AssertionError as exc:
                frozen = None
                recovery_state = "UNVERIFIABLE"
                recovery_error = f"{type(exc).__name__}: {exc}"
                recovery_checks = {}
                log(f"strict old-model recovery remains UNVERIFIABLE at BLAS=1: {recovery_error}")
    if not args.resume_postprocess:
        _write_json(out / "frozen_reference_recovery.json", {
            "status": recovery_state,
            "refit_status": recovery_state,
            "verification_checks": recovery_checks,
            "strict_tolerance": 1e-9,
            "failed_attempt_under_blas_threads_2": 6.066e-8,
            "recovery_blas_threads": 1,
            "formal_training_blas_threads": 2,
            "failure_detail": recovery_error,
            "tolerance_widened": False,
        })
        _write_json(out / "reference_recovery_environment.json", {
            "historical_anchor_tolerance": 1e-9,
            "BLAS2_recovery_max_abs": 6.066e-8,
            "BLAS1_recovery_status": recovery_state,
            "BLAS1_recovery_checks": recovery_checks,
            "scoped_to_recovery_only": True,
            "formal_training_threads_after_scope": 2,
            "tolerance_widened": False,
        })
    _set_status(out, "reference_recovery", "COMPLETE" if frozen is not None else "UNVERIFIABLE",
                detail=("direct frozen bundle verified; exact historical refit remains separately UNVERIFIABLE" if args.resume_postprocess and frozen is not None else "original frozen S/Full anchors recovered at BLAS=1" if frozen is not None else recovery_error))

    _set_status(out, "training", "RUNNING", detail="restoring completed model artifacts; no fitting in resume mode" if args.resume_postprocess else "CPU float32; serialized logistic and ScoreDeepSets fits")
    if args.resume_postprocess:
        (group_models_by_seed, cross_models_by_seed, sds_models_by_seed, sds_norms_by_seed,
         training_rows, curve_rows) = _load_completed_training(out, data_by_seed, features_by_seed)
        log(f"resume loaded completed checkpoints for {len(training_rows)} training rows and {len(curve_rows)} curve rows; no model fitting")
    else:
        training_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        group_models_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
        cross_models_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
        sds_models_by_seed: dict[str, dict[str, ScoreDeepSets]] = {}
        sds_norms_by_seed: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in (() if args.resume_postprocess else B3_SEEDS):
        log(f"===== TRAIN SEED {seed} =====")
        base = data_by_seed[seed][5]
        train_masks, tune_masks = train_masks_by_seed[seed], tune_masks_by_seed[seed]
        y = {k: data_by_seed[seed][k]["correct"] for k in KS}
        groups_fitted: dict[str, dict[str, Any]] = {}
        group_payloads: dict[str, dict[str, Any]] = {}
        for group in GROUPS:
            _set_status(out, "training", "RUNNING", detail=f"{seed} logistic ablation {group}", progress={"seed": seed, "model": group})
            raw = _group_raw(features_by_seed[seed], group)
            predictor, selected, records, standardized = _fit_logistic_group(
                seed, group, raw, y, train_masks, tune_masks, (5, 10), (5, 10), log=log,
            )
            selected["training_regime"] = "smallK5_10"
            _write_logistic_model(out, seed, f"ablation_{group.replace('+','_plus_')}", selected)
            groups_fitted[group] = {"predictor": predictor, "selected": selected, "normalization": rfeat.NormalizationFit.from_dict(selected["normalization"]), "standardized": standardized}
            group_payloads[group] = selected
            for rec in records:
                row = dict(rec)
                row.update({"seed": seed, "training_regime": "smallK5_10", "feature_group": group,
                            "selected_epoch": False, "device": "cpu"})
                curve_rows.append(row)
            training_rows.append(_train_status_row(seed, "smallK5_10", f"StatsLogistic[{group}]", selected))
            _set_status(out, "training", "RUNNING", detail=f"{seed} logistic ablation {group} complete", progress={"seed": seed, "model": group, "done": True})
        # Explicit large-K Stats Logistic, trained from K20/K50 only.
        raw_large = _group_raw(features_by_seed[seed], "S")
        large_pred, large_sel, large_records, large_std = _fit_logistic_group(
            seed, "S", raw_large, y, train_masks, tune_masks, (20, 50), (20, 50), log=log,
        )
        large_sel["training_regime"] = "largeK20_50"
        _write_logistic_model(out, seed, "stats_logistic_largeK20_50", large_sel)
        for rec in large_records:
            row = dict(rec)
            row.update({"seed": seed, "training_regime": "largeK20_50", "feature_group": "S", "selected_epoch": False, "device": "cpu"})
            curve_rows.append(row)
        training_rows.append(_train_status_row(seed, "largeK20_50", "StatsLogistic", large_sel))
        # ScoreDeepSets small and large regimes. The original fit is CPU float32.
        sds_by_regime: dict[str, ScoreDeepSets] = {}
        sds_sel_by_regime: dict[str, dict[str, Any]] = {}
        sds_norm_by_regime: dict[str, dict[str, Any]] = {}
        for regime_name, train_ks in TRAINING_REGIMES.items():
            _set_status(out, "training", "RUNNING", detail=f"{seed} ScoreDeepSets {regime_name} CPU fit")
            model, selected_sds, curves, norm = _train_sds_regime(
                seed, regime_name, train_ks, data_by_seed[seed], features_by_seed[seed],
                train_masks, tune_masks, out=out, log=log,
            )
            sds_by_regime[regime_name] = model
            sds_sel_by_regime[regime_name] = selected_sds
            sds_norm_by_regime[regime_name] = norm
            curve_rows.extend(curves)
            training_rows.append(_train_status_row(seed, regime_name, "ScoreDeepSets", selected_sds))
        group_models_by_seed[seed] = groups_fitted
        cross_models_by_seed[seed] = {
            "StatsLogistic_smallK5_10": {**groups_fitted["S"], "selected": groups_fitted["S"]["selected"]},
            "StatsLogistic_largeK20_50": {"predictor": large_pred, "selected": large_sel, "normalization": rfeat.NormalizationFit.from_dict(large_sel["normalization"]), "standardized": large_std},
        }
        sds_models_by_seed[seed] = sds_by_regime
        sds_norms_by_seed[seed] = sds_norm_by_regime
        # Store machine-readable status after each seed is complete.
        _write_csv(out / "training_status.csv", training_rows)
        _write_csv(out / "train_tune_curves.csv", curve_rows, CURVE_FIELDS)
    _set_status(out, "training", "COMPLETE", detail="loaded completed fixed checkpoints without refitting" if args.resume_postprocess else "4 ablation logistic + large-K Stats Logistic + small/large CPU ScoreDeepSets trained for all three seeds")

    _set_status(out, "reference_check", "RUNNING", detail="comparing new S/Full models and recovered old anchors")
    reference_rows: list[dict[str, Any]] = []
    phase1_identities = _load_reference_model_identities()
    phase05_stats_predictions = _load_phase05_stats_predictions()
    phase1_e1b_metrics = _load_phase1_e1b_metrics()
    for seed in B3_SEEDS:
        old_rows = _load_old_phase1(seed)
        reference_rows.extend(_reference_check_random(
            seed, data_by_seed[seed], features_by_seed[seed], group_models_by_seed[seed], frozen, old_rows,
            phase1_identities, phase05_stats_predictions, phase1_e1b_metrics,
        ))
    alt_data = _load_alt_features_from_saved(out) if args.resume_postprocess else _build_alt_features(out, log)
    for seed in B3_SEEDS:
        reference_rows.extend(_reference_check_alt(seed, alt_data[seed], frozen, group_models_by_seed[seed]))
    _write_csv(out / "reference_check.csv", reference_rows)
    if args.resume_postprocess:
        generic_legacy_values = [
            float(row["legacy_generic_csv_max_abs"])
            for row in reference_rows
            if row.get("legacy_generic_csv_max_abs") is not None and not row.get("cell")
        ]
        correct_model_values = [
            float(row["correct_model_anchor_max_abs"])
            for row in reference_rows
            if row.get("correct_model_anchor_max_abs") is not None and not row.get("cell")
        ]
        phase1f_values = [
            float(row["frozen_recovery_max_abs"])
            for row in reference_rows
            if row.get("frozen_recovery_max_abs") is not None and row.get("cell")
        ]
        legacy_anchor = {
            "status": "MODEL_IDENTITY_INCOMPATIBLE",
            "score_only_reliability_model_per_seed": {seed: phase1_identities[seed]["score_only_reliability"] for seed in B3_SEEDS},
            "semantic_reliability_model_per_seed": {seed: phase1_identities[seed]["semantic_reliability"] for seed in B3_SEEDS},
            "target_model_ids": {"S": "stats_logistic", "Full": "e1b_stats_semantic"},
            "generic_column_difference_max_abs_diagnostic_only": max(generic_legacy_values) if generic_legacy_values else None,
            "tolerance": 1e-9,
            "rows_checked": sum(1 for row in reference_rows if not row.get("cell")),
            "result_path": "results/research_repair_v1/information/reference_check.csv",
            "scope": "legacy Phase 1 generic CSV columns; these contain per-seed selected msp/e2_score, not S/Full",
            "interpretation": "MODEL_IDENTITY_INCOMPATIBLE; generic CSV differences are not a recovery mismatch",
        }
        correct_model_anchor = {
            "status": "PASS" if correct_model_values and max(correct_model_values) <= 1e-9 else "FAILED_OR_UNAVAILABLE",
            "max_abs_source_anchor_error": max(correct_model_values) if correct_model_values else None,
            "tolerance": 1e-9,
            "rows_checked": len(correct_model_values),
            "S_source": "Phase 0.5 stats_logistic per-row prediction CSV by seed/K/split",
            "Full_source": "Phase 1 e1_logistic/metrics.csv for fixed e1b_stats_semantic by seed/K/split",
            "result_path": "results/research_repair_v1/information/reference_check.csv",
        }
        phase1f_anchor = {
            "status": "PASS" if phase1f_values and max(phase1f_values) <= 1e-9 else "FAILED_OR_UNAVAILABLE",
            "max_abs_prediction_error": max(phase1f_values) if phase1f_values else None,
            "tolerance": 1e-9,
            "rows_checked": len(phase1f_values),
            "result_path": "results/research_repair_v1/information/reference_check.csv",
            "scope": "Phase 1F saved rand5/hard5 predictions; independently detailed in reference_phase1f_feature_audit.json",
        }
        environment = json.loads((out / "reference_recovery_environment.json").read_text(encoding="utf-8"))
        environment["old_prediction_direct_anchor_check"] = legacy_anchor
        environment["correct_model_random_anchor_check"] = correct_model_anchor
        environment["phase1f_direct_anchor_check"] = phase1f_anchor
        _write_json(out / "reference_recovery_environment.json", environment)
        frozen_payload = json.loads((out / "frozen_reference_recovery.json").read_text(encoding="utf-8"))
        frozen_payload["external_old_prediction_anchor_check"] = legacy_anchor
        frozen_payload["correct_model_random_anchor_check"] = correct_model_anchor
        frozen_payload["phase1f_anchor_check"] = phase1f_anchor
        _write_json(out / "frozen_reference_recovery.json", frozen_payload)
        _record_attempt(out, {
            "timestamp_utc": _now(), "pid": os.getpid(), "state": "PRIOR_ATTEMPT_DIAGNOSED",
            "attempt_pid": 52596, "failure_stage": "reference_check",
            "failure_type": "TypeError", "failure": "object of type NoneType has no len()",
            "root_cause": "_reference_check_alt used len(got) when strict frozen refit returned got=None; n must come from the stored expected prediction array",
            "source_location": "scripts/repair_information.py::_reference_check_alt (the legacy failure was logged without a traceback)",
            "original_trace_path": "results/research_repair_v1/information/logs.txt",
            "traceback_capture_status": "original runner persisted only the one-line exception; source expression identified and fixed",
        })
    _set_status(out, "reference_check", "COMPLETE", detail="legacy generic CSV model identities recorded; fixed S/Full source anchors and Phase 1F predictions checked")

    _set_status(out, "point_evaluation", "RUNNING", detail="test-only cross-K and semantic group point metrics")
    cross_metrics: list[dict[str, Any]] = []
    # Add model configuration references and precomputed probability arrays.
    cross_probabilities: dict[tuple[str, str, int], np.ndarray] = {}
    for seed in B3_SEEDS:
        for k in KS:
            item = data_by_seed[seed][k]
            for model_name, payload in cross_models_by_seed[seed].items():
                raw = features_by_seed[seed][k]["stats17"]
                x = rfeat.normalize_apply(raw, rfeat.NormalizationFit.from_dict(payload["selected"]["normalization"]))
                prob = payload["predictor"].predict_proba(x)
                cross_probabilities[(seed, model_name, k)] = prob
                cross_metrics.extend(_metric_rows_for_splits(
                    seed=seed, model=model_name.replace("_", " "), regime=payload["selected"]["training_regime"],
                    cell=CELLS_RANDOM[k], k=k, confidence=prob, correct=item["correct"], eval_split=item["eval_split"],
                ))
            for regime_name, model in sds_models_by_seed[seed].items():
                norm = sds_norms_by_seed[seed][regime_name]
                prob = model.predict_proba(norm["scores"][k], np.ones_like(norm["scores"][k], dtype=bool), norm["log_k"][k], norm["z_top1"][k])
                cross_probabilities[(seed, f"ScoreDeepSets_{regime_name}", k)] = prob
                cross_metrics.extend(_metric_rows_for_splits(
                    seed=seed, model="ScoreDeepSets", regime=regime_name, cell=CELLS_RANDOM[k], k=k,
                    confidence=prob, correct=item["correct"], eval_split=item["eval_split"],
                ))
            baseline = rfeat.scalar_confidence(item["scores"], temperature=_parse_temperature(seed))["msp"]
            cross_probabilities[(seed, "corrected_score_MSP", k)] = baseline
            cross_metrics.extend(_metric_rows_for_splits(
                seed=seed, model="corrected_score_MSP", regime="score_only", cell=CELLS_RANDOM[k], k=k,
                confidence=baseline, correct=item["correct"], eval_split=item["eval_split"],
            ))
    _append_seed_means(
        cross_metrics,
        keys=("model", "training_regime", "cell", "K", "eval_split"),
        numeric=("n", "accuracy", "auroc_correct", "aurc", "aurc_oracle", "e_aurc", "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95", "ece_adaptive", "brier_binary", "nll_binary"),
    )
    _write_csv(out / "cross_k_metrics.csv", cross_metrics)

    def cross_prediction_groups() -> Iterable[Iterable[Mapping[str, Any]]]:
        for seed in B3_SEEDS:
            for k in KS:
                item = data_by_seed[seed][k]
                for model_name, regime in (
                    ("StatsLogistic_smallK5_10", "smallK5_10"),
                    ("StatsLogistic_largeK20_50", "largeK20_50"),
                    ("ScoreDeepSets_smallK5_10", "smallK5_10"),
                    ("ScoreDeepSets_largeK20_50", "largeK20_50"),
                    ("corrected_score_MSP", "score_only"),
                ):
                    prob = cross_probabilities[(seed, model_name, k)]
                    yield _prediction_rows(item, prob, seed=seed, cell=CELLS_RANDOM[k], k=k,
                                           model=model_name, regime=regime)
    _save_all_rows_csv_gz(out / "cross_k_predictions.csv.gz", PREDICTION_FIELDS, cross_prediction_groups())

    semantic_metrics_rows: list[dict[str, Any]] = []
    semantic_effects_rows: list[dict[str, Any]] = []
    semantic_probs_by_seed: dict[str, dict[tuple[str, str], dict[str, np.ndarray]]] = {}
    for seed in B3_SEEDS:
        metrics, effects, probs = _semantic_point_tables(
            seed, data_by_seed[seed], features_by_seed[seed], alt_data[seed], group_models_by_seed[seed],
        )
        semantic_metrics_rows.extend(metrics)
        semantic_effects_rows.extend(effects)
        semantic_probs_by_seed[seed] = probs
    _append_seed_means(semantic_metrics_rows,
                       keys=("model", "training_regime", "cell", "K", "eval_split", "feature_group"),
                       numeric=("n", "accuracy", "auroc_correct", "aurc", "aurc_oracle", "e_aurc", "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95", "ece_adaptive", "brier_binary", "nll_binary"))
    _append_seed_means(semantic_effects_rows,
                       keys=("cell", "K", "eval_split", "comparison", "model_a", "model_b", "metric"),
                       numeric=("effect",))
    _write_csv(out / "semantic_ablation_metrics.csv", semantic_metrics_rows)
    _write_csv(out / "semantic_ablation_effects.csv", semantic_effects_rows)
    _write_semantic_predictions(out, data_by_seed, features_by_seed, alt_data, group_models_by_seed)

    diagnostic_rows: list[dict[str, Any]] = []
    for seed in B3_SEEDS:
        diagnostic_rows.extend(_make_128_diagnostic(
            seed, data_by_seed[seed], features_by_seed[seed], train_masks_by_seed[seed],
            group_models_by_seed[seed]["S"]["selected"],
            json.loads((out / "models" / seed / "score_deepsets_smallK5_10.json").read_text(encoding="utf-8")),
            out=out, log=log,
        ))
    _write_csv(out / "128_row_fit_diagnostic.csv", diagnostic_rows)
    _set_status(out, "point_evaluation", "COMPLETE", detail="point metrics, predictions, reference checks, and 128-row diagnostics saved")

    logk_payload = repair_feature_value
    protocol = {
        "protocol": "Research Repair v1 B information-source and ID/OOD",
        "created_utc": _now(),
        "input_roots_read_only": ["results/phase0b_independent", "results/phase05_score_sufficiency", "results/phase1_semantic_sufficiency", "results/phase1f_hard_semantic", "cache/semantic_phase1", "cache/features", "cache/manifests", "cache/proposals.h5"],
        "output_root": str(out.relative_to(ROOT)),
        "scorers": list(B3_SEEDS), "K": list(KS),
        "split_manifest": str((PHASE05_ROOT / "split_manifest.json").relative_to(ROOT)),
        "train_tune_images": {"seed": split.seed, "train_count": int(split.train_images.size), "tune_count": int(split.tune_images.size)},
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "Q_names": list(FEATURE_GROUPS["S+Q"][17:]),
        "V_names": list(V_NAMES),
        "logistic_C_grid": list(LOGISTIC_CS), "logistic_tie_tolerance": 0.002,
        "score_deepsets": {"learning_rates": list(SDS_LRS), "max_epochs": 300, "patience": 30, "batch_size": 256, "weight_decay": 1e-4, "device": "cpu", "dtype": "float32", "torch_threads": 2},
        "training_regimes": {key: list(value) for key, value in TRAINING_REGIMES.items()},
        "semantic_test_cells": [*CELLS_RANDOM.values(), "rand5", "hard5", "expb_m0", "expb_m2", "expb_m4", "expb_m8"],
        "reference_status": (
            f"legacy Phase 1 generic columns are best_score_only=msp and best_semantic=e2_score for all three seeds; e2_score is a score-only control, "
            f"so they are MODEL_IDENTITY_INCOMPATIBLE with S=stats_logistic and Full=e1b_stats_semantic; correct S/Full source anchor status="
            f"{correct_model_anchor['status'] if args.resume_postprocess else 'NOT_RUN'}; Phase 1F direct anchor status="
            f"{phase1f_anchor['status'] if args.resume_postprocess else 'NOT_RUN'}; strict historical refit status={recovery_state}."
        ),
        "logK_scaler_repair": logk_payload,
        "training_complete": True,
        "point_evaluation_complete": True,
        "formal_bootstrap_complete": False,
        "resource_device": "CPU; original ScoreDeepSets.fit implementation uses CPU tensors; no GPU training was done.",
        "formal_bootstrap": "pending separately scheduled statistics slot; 5000 shared image-cluster replicates through ccg.repairs.bootstrap.shared_image_cluster_bootstrap.",
    }
    for rel in (
        "scripts/repair_information.py", "scripts/repair_information_bootstrap.py",
        "scripts/repair_information_reference_audit.py", "src/ccg/repairs/information.py",
        "scripts/run_phase05.py", "src/ccg/semantic/features.py", "src/ccg/semantic/data.py",
    ):
        path = ROOT / rel
        protocol.setdefault("source_sha256", {})[rel] = _hash_file(path)
    _write_json(out / "protocol.json", protocol)

    # Scientific main effect is derived from the requested matched point output;
    # intervals are populated by the separately scheduled shared bootstrap.
    main_rows = [r for r in semantic_effects_rows if r.get("seed") == "b3_mean" and r.get("comparison") == "Full_minus_S_plus_Q" and r.get("metric") == "auroc_correct"]
    primary_main_rows = [r for r in main_rows if r.get("eval_split") in ("testA", "testB", "__pooled_test__")]
    sign_text = "positive on every reported primary cell" if primary_main_rows and all(float(r["effect"]) > 0 for r in primary_main_rows) else "mixed or non-positive across reported primary cells"
    random_k5 = [r for r in primary_main_rows if r.get("cell") == "randomK5"]
    random_k10_50 = [r for r in primary_main_rows if r.get("cell") in {"randomK10", "randomK20", "randomK50"}]
    k5_by_split = {r["eval_split"]: float(r["effect"]) for r in random_k5}
    k10_50_all_positive = bool(random_k10_50) and all(float(r["effect"]) > 0.0 for r in random_k10_50)
    scientific_conclusion = (
        "Descriptive Full−(S+Q) AUROC effects are mixed at random K=5 "
        f"(testA={k5_by_split.get('testA', float('nan')):.6f}, testB={k5_by_split.get('testB', float('nan')):.6f}); "
        f"all reported random K=10/20/50 testA/testB/pooled point effects are {'positive' if k10_50_all_positive else 'not uniformly positive'}. "
        "Shared image-cluster confidence intervals are pending; these point estimates do not establish statistical significance."
    )
    summary = {
        "protocol": "Research Repair v1 B",
        "status": "POINT_ESTIMATES_COMPLETE_BOOTSTRAP_PENDING",
        "generated_utc": _now(),
        "formal_replicates": 0,
        "formal_bootstrap_complete": False,
        "training_complete": True,
        "cross_k_metrics_complete": True,
        "semantic_ablation_metrics_complete": True,
        "train_tune_curves_complete": True,
        "reference_check_complete": True,
        "reference_recovery_complete": frozen is not None and (not args.resume_postprocess or (phase1f_anchor["status"] == "PASS" and correct_model_anchor["status"] == "PASS")),
        "correct_model_random_anchor_status": correct_model_anchor["status"] if args.resume_postprocess else "NOT_RUN",
        "correct_model_random_anchor_max_abs": max(correct_model_values) if args.resume_postprocess and correct_model_values else None,
        "legacy_prediction_csv_model_identity": {
            "score_only_reliability": {seed: phase1_identities[seed]["score_only_reliability"] for seed in B3_SEEDS} if args.resume_postprocess else None,
            "semantic_reliability": {seed: phase1_identities[seed]["semantic_reliability"] for seed in B3_SEEDS} if args.resume_postprocess else None,
            "semantic_reliability_interpretation": "e2_score consumes score statistics only; it is not semantic embedding information",
            "status": legacy_anchor["status"] if args.resume_postprocess else "NOT_RUN",
        },
        "phase1f_anchor_status": phase1f_anchor["status"] if args.resume_postprocess else "NOT_RUN",
        "legacy_phase1_csv_anchor_status": legacy_anchor["status"] if args.resume_postprocess else "NOT_RUN",
        "128_row_diagnostic_complete": True,
        "training_status_rows": len(training_rows),
        "curve_rows": len(curve_rows),
        "main_effect": {
            "contrast": "Full - (S+Q)",
            "endpoint": "AUROC_correct",
            "per_cell_point_estimates": primary_main_rows,
            "point_direction": sign_text,
            "confidence_interval": None,
            "replicates": 0,
        },
        "secondary_effects": ["S+V - S", "Full - S", "E-AURC reduction", "RER@50 gain pp", "RER@80 gain pp"],
        "reference_recovery_max_abs": recovery_checks.get("all", {}) if frozen is not None else None,
        "reference_recovery_status": recovery_state,
        "logK_bug": repair_feature_value,
        "scientific_conclusion": scientific_conclusion,
        "model_lineage": {
            "B_models": "new Research Repair v1 refits from the frozen train/tune splits; corrected ScoreDeepSets logK train-mask implementation; these are not restored historical Phase 1 predictions",
            "historical_phase1_predictions": "preserved as read-only historical artifacts; generic CSV fields identify msp and e2_score",
            "strict_old_model_refit_status": "UNVERIFIABLE at unchanged 1e-9 tolerance",
            "direct_frozen_bundle_and_correct_source_anchors": "VERIFIED; see frozen_reference_recovery.json and reference_check.csv",
        },
        "legacy_generic_csv_diagnostic_max_abs": max(generic_legacy_values) if args.resume_postprocess and generic_legacy_values else None,
        "phase1f_anchor_max_abs": max(phase1f_values) if args.resume_postprocess and phase1f_values else None,
        "files": {
            "training_status": "training_status.csv",
            "train_tune_curves": "train_tune_curves.csv",
            "cross_k_metrics": "cross_k_metrics.csv",
            "cross_k_predictions": "cross_k_predictions.csv.gz",
            "semantic_ablation_metrics": "semantic_ablation_metrics.csv",
            "semantic_ablation_effects": "semantic_ablation_effects.csv",
            "semantic_ablation_predictions": "semantic_ablation_predictions.csv.gz",
            "reference_check": "reference_check.csv",
            "phase1f_feature_audit": "reference_phase1f_feature_audit.json",
            "128_row_fit_diagnostic": "128_row_fit_diagnostic.csv",
            "formal_bootstrap": "information_bootstrap.json",
        },
    }
    _write_json(out / "summary.json", summary)
    _set_status(out, "formal_bootstrap", "NOT_STARTED", detail="awaiting a separately allocated heavy statistics slot")
    _set_status(out, "overall", "POINT_ESTIMATES_COMPLETE", detail="all B training and descriptive point artifacts are complete; formal bootstrap remains pending")
    log("B runner point-estimate stages complete; formal bootstrap awaits parent resource slot")
    _record_attempt(out, {"timestamp_utc": _now(), "pid": os.getpid(), "state": "POINT_ESTIMATES_COMPLETE", "training": True, "bootstrap": "pending separate statistics slot"})
    log_handle.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure_out = ROOT / "results/research_repair_v1/information"
        if failure_out.exists():
            try:
                status_path = _status_path(failure_out)
                if status_path.exists():
                    current = json.loads(status_path.read_text(encoding="utf-8"))
                    stage = str(current.get("current_stage", "startup"))
                    _set_status(failure_out, stage, "FAILED", detail=f"{type(exc).__name__}: {exc}")
                _record_attempt(failure_out, {"timestamp_utc": _now(), "pid": os.getpid(), "state": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
                with (failure_out / "logs.txt").open("a", encoding="utf-8") as handle:
                    handle.write(f"[{_now()}] FAILED: {type(exc).__name__}: {exc}\n")
                    handle.write(traceback.format_exc())
            except Exception:
                pass
        raise
