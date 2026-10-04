"""V3 OpenCLIP ViT-B/16 replication of the frozen S/Q/V reliability groups.

This module consumes the existing B16 region/text feature cache, cached random
K5/K10 embeddings, and the three V2-G B16 scorer checkpoints. It never runs an
image or text encoder and never fits or changes a grounding scorer. Logistic
models are fitted only on the original reliability-train rows and selected on
the original reliability-tune rows.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.experiment import phase0a, phase0b  # noqa: E402
from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import data as rdata  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.repairs.bootstrap import assert_row_alignment, shared_image_cluster_bootstrap  # noqa: E402
from ccg.repairs.information import (  # noqa: E402
    B3_SEEDS,
    FEATURE_GROUPS,
    LOGISTIC_CS,
    assert_nested_correctness,
    compute_semantic_stats_chunked,
    feature_block,
    fit_group_normalization,
    fit_selected_logistic,
    load_logistic_predictor,
    nll_binary,
)
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic.hard_scores import materialise_examples, score_examples, verify_against_raw_scores  # noqa: E402

__all__ = [
    "B16_EMBEDDING_ROOT", "B16_FEATURES_ROOT", "B16_SCORER_ROOT", "BOOTSTRAP_SEED",
    "BOOTSTRAP_REPLICATES", "GROUPS", "K_TRAIN", "OUT_ROOT", "SEEDS",
    "assert_candidate_identities", "assert_score_anchor", "build_parser", "load_model_json",
    "main", "portable_model_payload", "run_bootstrap", "run_replication",
]

OUT_ROOT = ROOT / "results/v3_final_validation/replication"
B16_FEATURES_ROOT = ROOT / "cache/v2_backbones/openclip_b16"
B16_EMBEDDING_ROOT = ROOT / "cache/v2_semantic/openclip_b16"
B16_SCORER_ROOT = ROOT / "results/v2_backbone_generalization/b1_phase0b"
SPLIT_PATH = ROOT / "results/phase05_score_sufficiency/split_manifest.json"
MANIFEST_ROOT = ROOT / "cache/manifests"
PHASE1F_ROOT = ROOT / "results/phase1f_hard_semantic"
BANK_PATH = ROOT / "cache/proposals.h5"
REFS_PATH = ROOT / "data/raw/refcoco+/refcoco+/refs(unc).p"
IMAGE_SIZES_PATH = ROOT / "cache/image_sizes.npz"
K_TRAIN = (5, 10)
K_RANDOM_EVAL = (5, 10, 50)
SEEDS = (1, 2, 3)
SEED_NAMES = tuple(f"b3_seed{s}" for s in SEEDS)
GROUPS = ("S", "S+Q", "S+V", "Full")
CELLS = ("randomK5", "randomK10", "randomK50", "rand5", "hard5", "expb_m0", "expb_m8")
ALT_CELLS = ("rand5", "hard5", "expb_m0", "expb_m8")
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 0
ANCHOR_ATOL = 1e-4
CHUNK_ROWS = 256
LEGACY_INPUT_MANIFEST = ROOT / "results/research_repair_v1/input_manifest.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", buffering=1)

    def log(message: str) -> None:
        line = f"[{_now()}] {message}"
        print(line, flush=True)
        handle.write(line + "\n")

    return log, handle


def _status(out: Path, stage: str, state: str, *, detail: str = "") -> None:
    path = out / "STATUS.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "experiment": "V3 OpenCLIP B16 S/Q/V replication", "stages": {},
    }
    row = payload.setdefault("stages", {}).setdefault(stage, {})
    if state == "RUNNING" and not row.get("started_utc"):
        row["started_utc"] = _now()
    row.update({"state": state, "updated_utc": _now()})
    if detail:
        row["detail"] = detail
    if state in {"COMPLETE", "FAILED", "SKIPPED", "WAITING"}:
        row["finished_utc"] = _now()
    if state == "RUNNING":
        payload["current_stage"] = stage
        payload["status"] = "RUNNING"
    elif state == "FAILED":
        payload["status"] = "FAILED"
    elif stage == "overall":
        payload["status"] = state
        payload["current_stage"] = "overall"
    payload["updated_utc"] = _now()
    _write_json(path, payload)


def _input_manifest() -> dict[str, Any]:
    """Fingerprint every frozen file directly used by the run."""
    paths = [
        B16_FEATURES_ROOT / "metadata.json",
        B16_FEATURES_ROOT / "region_features.h5",
        B16_FEATURES_ROOT / "text_features.h5",
        B16_EMBEDDING_ROOT / "embedding_index.json",
        B16_EMBEDDING_ROOT / "embeddings_K5.npz",
        B16_EMBEDDING_ROOT / "embeddings_K10.npz",
        SPLIT_PATH,
        PHASE1F_ROOT / "manifests/index.json",
        ROOT / "src/ccg/repairs/information.py",
        ROOT / "src/ccg/semantic/features.py",
        ROOT / "src/ccg/reliability/features.py",
        ROOT / "src/ccg/v3/evaluation.py",
        ROOT / "src/ccg/v3/protocol.py",
        ROOT / "src/ccg/v3/replication.py",
        ROOT / "scripts/v3_replicate_features.py",
        ROOT / "tests/test_v3_replication.py",
    ]
    for seed in SEEDS:
        seed_root = B16_SCORER_ROOT / f"seed_{seed}"
        paths.extend([
            seed_root / "model.npz", seed_root / "training.json", seed_root / "eval_metadata.json",
            *(seed_root / "raw_scores" / f"K{k}.npz" for k in (5, 10, 50)),
        ])
    for cell in ALT_CELLS:
        paths.append(PHASE1F_ROOT / "manifests" / f"{cell}_candidates.npz")
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen replication inputs: {missing}")
    upstream: dict[str, Mapping[str, Any]] = {}
    upstream_payload: dict[str, Any] = {}
    if LEGACY_INPUT_MANIFEST.exists():
        upstream_payload = json.loads(LEGACY_INPUT_MANIFEST.read_text(encoding="utf-8"))
        upstream = {str(row["path"]): row for row in upstream_payload.get("inputs", [])}
    inherited = 0
    files: dict[str, Any] = {}
    for path in paths:
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        old = upstream.get(relative)
        size = int(path.stat().st_size)
        if old is not None:
            if size != int(old["size_bytes"]):
                raise AssertionError(f"{relative}: size differs from previously audited source input")
            digest = str(old["sha256"])
            fingerprint_source = str(LEGACY_INPUT_MANIFEST.relative_to(ROOT))
            inherited += 1
        else:
            digest = _sha256(path)
            fingerprint_source = "sha256_computed_for_this_replication"
        files[relative] = {
            "sha256": digest, "size_bytes": size, "fingerprint_source": fingerprint_source,
        }
    return {
        "created_utc": _now(),
        "inherited_from_verified_legacy_manifest": inherited,
        "upstream_manifest": str(LEGACY_INPUT_MANIFEST.relative_to(ROOT)) if upstream_payload else None,
        "upstream_manifest_sha256": _sha256(LEGACY_INPUT_MANIFEST) if upstream_payload else None,
        "files": files,
    }


def _read_temperature(seed: int) -> float:
    path = B16_SCORER_ROOT / f"seed_{int(seed)}" / "eval_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = float(payload["temperature_corrected"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid corrected temperature in {path}: {value}")
    fit = payload.get("corrected_fit", {})
    if not bool(fit.get("interior", payload.get("corrected_interior", True))):
        raise ValueError(f"corrected temperature fit is not interior in {path}")
    return value


def _canonical_random(seed: int, k: int) -> dict[str, np.ndarray]:
    scorer = sdata.load_scorer_canonical(f"b3_seed{seed}", k, b3_root=B16_SCORER_ROOT)
    if not np.all(scorer.target_local == 0):
        raise AssertionError(f"B16 seed{seed} K{k}: target_local is not the frozen local column 0")
    scores = np.asarray(scorer.scores, dtype=np.float32)
    correct = np.argmax(scores, axis=1) == np.asarray(scorer.target_local)
    return {
        "sentence_id": np.asarray(scorer.sentence_id, dtype=np.int64),
        "ref_id": np.asarray(scorer.ref_id, dtype=np.int64),
        "image_id": np.asarray(scorer.image_id, dtype=np.int64),
        "eval_split": np.asarray(scorer.eval_split),
        "target_local": np.asarray(scorer.target_local, dtype=np.int32),
        "scores": scores,
        "correct": np.asarray(correct, dtype=bool),
    }


def _validate_random_rows(seed: int, rows: Mapping[int, Mapping[str, np.ndarray]]) -> None:
    reference = rows[K_TRAIN[0]]
    for k in K_RANDOM_EVAL:
        current = rows[k]
        for field in ("sentence_id", "ref_id", "image_id", "eval_split"):
            if not np.array_equal(reference[field], current[field]):
                raise AssertionError(f"B16 seed{seed} K{k}: canonical {field} differs from K5")
        if not np.array_equal(current["correct"], np.argmax(current["scores"], axis=1) == 0):
            raise AssertionError(f"B16 seed{seed} K{k}: correctness differs from stored score argmax")
    assert_nested_correctness({k: rows[k]["correct"] for k in K_RANDOM_EVAL})


def _validate_store(k: int, data: Mapping[str, np.ndarray]) -> sdata.EmbeddingStore:
    store = sdata.load_embedding_store(k, out_root=B16_EMBEDDING_ROOT)
    for field in ("sentence_id", "ref_id", "image_id", "eval_split"):
        expected = np.asarray(data[field])
        observed = np.asarray(getattr(store, field))
        if not np.array_equal(expected, observed):
            raise AssertionError(f"B16 embedding K{k}: {field} is not aligned to the scorer")
    if store.feature_dim != 512 or not np.all(store.target_local == 0):
        raise AssertionError(f"B16 embedding K{k}: unexpected feature dim or target column")
    return store


def assert_score_anchor(recomputed: np.ndarray, frozen: np.ndarray, *, atol: float = ANCHOR_ATOL) -> float:
    """Strict frozen-logit comparison used by the B16 scorer forward guard."""
    actual = np.asarray(recomputed, dtype=np.float32)
    expected = np.asarray(frozen, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"score anchor shape mismatch: recomputed={actual.shape}, frozen={expected.shape}")
    delta = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0.0))
    if delta > float(atol):
        raise AssertionError(f"frozen score anchor max|delta|={delta:.3e} exceeds {atol:.1e}")
    return delta


def assert_candidate_identities(
    sentence_id: np.ndarray,
    ref_id: np.ndarray,
    image_id: np.ndarray,
    candidate_indices: np.ndarray,
    target_index: np.ndarray,
    *,
    expected_k: int,
) -> None:
    """Reject malformed saved candidates before materializing cached features."""
    sid = np.asarray(sentence_id, dtype=np.int64)
    refs = np.asarray(ref_id, dtype=np.int64)
    images = np.asarray(image_id, dtype=np.int64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    targets = np.asarray(target_index, dtype=np.int64)
    n = sid.size
    if any(v.shape != (n,) for v in (refs, images, targets)):
        raise ValueError("candidate identity anchors must be aligned 1-D arrays")
    if candidates.shape != (n, int(expected_k)):
        raise ValueError(f"candidate indices must have shape {(n, expected_k)}, got {candidates.shape}")
    if not np.array_equal(candidates[:, 0], targets):
        raise AssertionError("candidate local column 0 differs from the saved target_index")
    if any(np.unique(row).size != int(expected_k) for row in candidates):
        raise AssertionError("a frozen candidate row contains duplicate bank indices")
    if np.unique(sid).size != sid.size:
        raise AssertionError("saved candidate rows contain duplicate sentence_id values")


def _make_corpus(*, preload: bool, ks: Sequence[int]) -> B3Corpus:
    corpus = B3Corpus(
        B16_FEATURES_ROOT, MANIFEST_ROOT, REFS_PATH, BANK_PATH,
        image_sizes_path=IMAGE_SIZES_PATH, ks=tuple(ks), regime="random", preload=False,
    )
    if preload:
        corpus.preload(region=True, text=True, bank=False)
    return corpus


def _record_map(corpus: B3Corpus, ks: Sequence[int]) -> dict[int, Any]:
    records = corpus.eval_records(phase0a.PRIMARY_SPLITS, ks)
    by_sid = {int(record.sentence_id): record for record in records}
    if len(by_sid) != len(records):
        raise AssertionError("Phase 0A random records contain duplicate sentence_id values")
    return by_sid


def _feature_path(out: Path, cell: str, seed: int) -> Path:
    return out / "derived" / f"{cell}_b3_seed{seed}.npz"


def _save_feature_item(path: Path, item: Mapping[str, np.ndarray]) -> None:
    _save_npz(path, **{key: np.asarray(value) for key, value in item.items()})


def _load_feature_item(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _semantic_cache_valid(path: Path, data: Mapping[str, np.ndarray], k: int) -> bool:
    if not path.exists():
        return False
    try:
        item = _load_feature_item(path)
        return (
            item["stats17"].shape == (len(data["sentence_id"]), 17)
            and item["sem16"].shape == (len(data["sentence_id"]), 16)
            and all(np.array_equal(item[field], data[field]) for field in ("sentence_id", "ref_id", "image_id", "eval_split"))
            and item["scores"].shape == (len(data["sentence_id"]), int(k))
            and np.array_equal(item["correct"], data["correct"])
        )
    except (OSError, KeyError, ValueError):
        return False


def _random_feature_item(
    seed: int,
    k: int,
    data: Mapping[str, np.ndarray],
    corpus: B3Corpus,
    records: Mapping[int, Any],
    out: Path,
    log: Any,
) -> dict[str, np.ndarray]:
    path = _feature_path(out, f"randomK{k}", seed)
    if _semantic_cache_valid(path, data, k):
        log(f"seed{seed} randomK{k}: loaded identity-checked semantic feature cache")
        return _load_feature_item(path)

    if k in (5, 10):
        store = _validate_store(k, data)
        sem = compute_semantic_stats_chunked(store.z_q, store.z_i, data["scores"], chunk_rows=512)
        candidates = np.vstack([
            B3Corpus.candidate_bank_indices(records[int(sid)], k)
            for sid in data["sentence_id"]
        ]).astype(np.int64, copy=False)
    elif k == 50:
        sem = np.empty((len(data["sentence_id"]), len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
        candidates = np.empty((len(data["sentence_id"]), k), dtype=np.int64)
        for start in range(0, len(data["sentence_id"]), CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, len(data["sentence_id"]))
            samples = [records[int(sid)] for sid in data["sentence_id"][start:stop]]
            batch = materialise_examples(corpus, samples, k)
            candidates[start:stop] = np.stack([B3Corpus.candidate_bank_indices(row, k) for row in samples])
            sem[start:stop] = compute_semantic_stats_chunked(
                batch.z_q, batch.z_i, data["scores"][start:stop], chunk_rows=CHUNK_ROWS,
            )
            if stop == len(data["sentence_id"]) or stop % 4096 == 0:
                log(f"randomK50 B16 cached-feature semantic summaries: {stop}/{len(data['sentence_id'])} rows")
            del batch, samples
    else:
        raise ValueError(f"unsupported random feature K={k}")
    stats = rfeat.stat_features(data["scores"], temperature=_read_temperature(seed))
    item = {
        **{key: np.asarray(data[key]) for key in ("sentence_id", "ref_id", "image_id", "eval_split")},
        "scores": np.asarray(data["scores"], dtype=np.float32),
        "correct": np.asarray(data["correct"], dtype=bool),
        "candidate_indices": candidates,
        "stats17": np.asarray(stats, dtype=np.float64),
        "sem16": np.asarray(sem, dtype=np.float64),
    }
    if tuple(sfeat.SEMANTIC_STAT_NAMES) != tuple(FEATURE_GROUPS["Full"][17:]):
        raise AssertionError("B16 semantic feature order differs from the frozen Full group")
    _save_feature_item(path, item)
    log(f"seed{seed} randomK{k}: computed {len(data['sentence_id'])} feature rows using cached B16 embeddings")
    return item


def _split_masks(split: rdata.ReliabilitySplit, rows: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = split.row_mask(rows["eval_split"], rows["image_id"], kind="reliability_train")
    tune = split.row_mask(rows["eval_split"], rows["image_id"], kind="reliability_tune")
    test = np.isin(rows["eval_split"], ("testA", "testB"))
    if np.any(train & tune) or np.any(train & test) or np.any(tune & test):
        raise AssertionError("train/tune/test image partitions overlap")
    return train, tune, test


def portable_model_payload(
    *, seed: int, group: str, temperature: float, feature_names: Sequence[str],
    normalization: rfeat.NormalizationFit, predictor: Mapping[str, Any],
    grid_records: Sequence[Mapping[str, Any]], train_images: np.ndarray, tune_images: np.ndarray,
) -> dict[str, Any]:
    """Build the stable JSON model contract shared with the V3 inference stage."""
    if tuple(feature_names) != tuple(FEATURE_GROUPS[group]):
        raise ValueError(f"{group}: feature keys differ from frozen FEATURE_GROUPS order")
    return {
        "schema_version": 1,
        "artifact": "v3_b16_s_q_v_replication_logistic",
        "backbone": "openclip_b16",
        "seed": f"b3_seed{int(seed)}",
        "temperature_corrected": float(temperature),
        "group": group,
        "feature_names": list(feature_names),
        "normalization": normalization.to_dict(),
        "normalization_method": "training-only population mean/std; (x-mean)/(std+1e-8)",
        "predictor": dict(predictor),
        "selection_grid": [dict(row) for row in grid_records],
        "training_ks": list(K_TRAIN),
        "selection_ks": list(K_TRAIN),
        "selection_split": "reliability_tune",
        "fit_split": "reliability_train",
        "n_train_images": int(np.unique(train_images).size),
        "n_tune_images": int(np.unique(tune_images).size),
        "source": {
            "scorer_root": str(B16_SCORER_ROOT.relative_to(ROOT)),
            "embedding_root": str(B16_EMBEDDING_ROOT.relative_to(ROOT)),
            "split_manifest": str(SPLIT_PATH.relative_to(ROOT)),
        },
    }


def load_model_json(path: Path) -> tuple[rfeat.NormalizationFit, Any, dict[str, Any]]:
    """Load a V3 portable JSON checkpoint for standardization and inference."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("artifact") != "v3_b16_s_q_v_replication_logistic":
        raise ValueError(f"unsupported portable replication checkpoint: {path}")
    group = str(payload["group"])
    if tuple(payload["feature_names"]) != tuple(FEATURE_GROUPS[group]):
        raise ValueError(f"{path}: feature_names differ from frozen group order")
    fit = rfeat.NormalizationFit.from_dict(payload["normalization"])
    if fit.keys != tuple(payload["feature_names"]):
        raise ValueError(f"{path}: normalizer keys differ from feature_names")
    predictor = load_logistic_predictor(payload["predictor"])
    if predictor.coefficients.size != len(payload["feature_names"]):
        raise ValueError(f"{path}: predictor width differs from feature_names")
    return fit, predictor, payload


def _fit_groups(
    seed: int,
    data: Mapping[int, Mapping[str, np.ndarray]],
    features: Mapping[int, Mapping[str, np.ndarray]],
    split: rdata.ReliabilitySplit,
    out: Path,
    log: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    base = data[5]
    train_masks: dict[int, np.ndarray] = {}
    tune_masks: dict[int, np.ndarray] = {}
    for k in K_TRAIN:
        train, tune, _ = _split_masks(split, data[k])
        train_masks[k] = train
        tune_masks[k] = tune
    if any(int(train_masks[k].sum()) != int(train_masks[5].sum()) for k in K_TRAIN):
        raise AssertionError("train row budgets differ across K5/K10")
    if any(int(tune_masks[k].sum()) != int(tune_masks[5].sum()) for k in K_TRAIN):
        raise AssertionError("tune row budgets differ across K5/K10")

    fitted: dict[str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    for group in GROUPS:
        raw_by_k = {
            k: feature_block(features[k]["stats17"], features[k]["sem16"], group)
            for k in K_TRAIN
        }
        fit, standardized = fit_group_normalization(raw_by_k, train_masks, K_TRAIN, FEATURE_GROUPS[group])
        x_train = np.vstack([standardized[k][train_masks[k]] for k in K_TRAIN])
        y_train = np.concatenate([data[k]["correct"][train_masks[k]].astype(np.float64) for k in K_TRAIN])
        tune_by_k = {
            k: (standardized[k][tune_masks[k]], data[k]["correct"][tune_masks[k]])
            for k in K_TRAIN
        }
        predictor, selected, grid_records = fit_selected_logistic(x_train, y_train, tune_by_k)
        selected.update({"seed": f"b3_seed{seed}", "group": group, "training_ks": list(K_TRAIN)})
        model_path = out / "models" / f"b3_seed{seed}" / f"{group}.json"
        model_payload = portable_model_payload(
            seed=seed, group=group, temperature=_read_temperature(seed),
            feature_names=FEATURE_GROUPS[group], normalization=fit, predictor=selected,
            grid_records=grid_records, train_images=split.train_images, tune_images=split.tune_images,
        )
        _write_json(model_path, model_payload)
        loaded_fit, loaded_predictor, loaded_payload = load_model_json(model_path)
        if not np.allclose(loaded_fit.mean, fit.mean, rtol=0, atol=0):
            raise AssertionError(f"{group}: portable normalizer round-trip changed its mean")
        if not np.allclose(loaded_predictor.coefficients, predictor.coefficients, rtol=0, atol=0):
            raise AssertionError(f"{group}: portable predictor round-trip changed its coefficients")
        fitted[group] = {
            "normalization": loaded_fit, "predictor": loaded_predictor,
            "payload": loaded_payload, "selected": selected,
        }
        for rec in grid_records:
            selection_rows.append({"seed": f"b3_seed{seed}", "group": group, **dict(rec)})
        log(
            f"seed{seed} {group}: selected C={selected['chosen_C']:g}, "
            f"tune AUROC={selected['tune_mean_auroc']:.6f}, "
            f"rows={selected['n_train_rows']}"
        )
    return fitted, {"selection_rows": selection_rows, "train_masks": train_masks, "tune_masks": tune_masks}


def _load_fitted_models(out: Path) -> dict[int, dict[str, dict[str, Any]]]:
    fitted: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for group in GROUPS:
            path = out / "models" / f"b3_seed{seed}" / f"{group}.json"
            fit, predictor, payload = load_model_json(path)
            fitted[seed][group] = {
                "normalization": fit, "predictor": predictor,
                "payload": payload, "selected": dict(payload["predictor"]),
            }
    return fitted


def run_fit_stage(*, out: Path = OUT_ROOT) -> dict[str, Any]:
    """Fast model stage using only frozen raw scores and the cached K5/K10 NPZs."""
    out = Path(out).resolve()
    if out != OUT_ROOT.resolve():
        raise ValueError(f"V3 replication outputs are restricted to {OUT_ROOT}; refusing {out}")
    out.mkdir(parents=True, exist_ok=True)
    log, handle = _logger(out / "replication.log")
    try:
        import torch
        from threadpoolctl import threadpool_limits

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True)
        threadpool_limits(limits=2)
        _status(out, "fit_stage_inputs", "RUNNING", detail="loading only fixed split, B16 K5/K10 scores and cached embeddings")
        split = sdata.load_split_from_manifest(SPLIT_PATH)
        if split.train_images.size != 523 or split.tune_images.size != 224:
            raise AssertionError("frozen B16 split must reuse the original 523/224 train/tune image lists")
        data_by_seed = {
            seed: {k: _canonical_random(seed, k) for k in K_TRAIN}
            for seed in SEEDS
        }
        for seed in SEEDS:
            for k in K_TRAIN:
                current, base = data_by_seed[seed][k], data_by_seed[seed][5]
                for field in ("sentence_id", "ref_id", "image_id", "eval_split"):
                    if not np.array_equal(current[field], base[field]):
                        raise AssertionError(f"seed{seed} K{k}: canonical {field} mismatch")
            assert_nested_correctness({k: data_by_seed[seed][k]["correct"] for k in K_TRAIN})
            sdata.assert_split_matches_fresh(
                split, data_by_seed[seed][5]["eval_split"], data_by_seed[seed][5]["image_id"],
            )
        fingerprints: dict[str, Any] = {}
        small_inputs = [SPLIT_PATH, B16_FEATURES_ROOT / "metadata.json", B16_EMBEDDING_ROOT / "embedding_index.json"]
        small_inputs += [B16_EMBEDDING_ROOT / f"embeddings_K{k}.npz" for k in K_TRAIN]
        for seed in SEEDS:
            seed_root = B16_SCORER_ROOT / f"seed_{seed}"
            small_inputs += [seed_root / "model.npz", seed_root / "training.json", seed_root / "eval_metadata.json"]
            small_inputs += [seed_root / "raw_scores" / f"K{k}.npz" for k in K_TRAIN]
        small_inputs += [
            ROOT / "src/ccg/repairs/information.py", ROOT / "src/ccg/semantic/features.py",
            ROOT / "src/ccg/reliability/features.py",
        ]
        fingerprints = {
            str(path.relative_to(ROOT)): {"sha256": _sha256(path), "size_bytes": int(path.stat().st_size)}
            for path in small_inputs
        }
        _write_json(out / "fit_stage_fingerprints.json", {"files": fingerprints, "created_utc": _now()})
        _status(out, "fit_stage_inputs", "COMPLETE", detail="K5/K10 scores, embeddings and split align; only small fit inputs fingerprinted")

        corpus = _make_corpus(preload=False, ks=K_TRAIN)
        records_by_k = {k: _record_map(corpus, (k,)) for k in K_TRAIN}
        random_items: dict[int, dict[int, dict[str, np.ndarray]]] = {seed: {} for seed in SEEDS}
        for k in K_TRAIN:
            for seed in SEEDS:
                random_items[seed][k] = _random_feature_item(
                    seed, k, data_by_seed[seed][k], corpus, records_by_k[k], out, log,
                )

        _status(out, "training", "RUNNING", detail="CPU logistic fits with frozen S/Q/V groups, C grid and selected tie rule")
        fitted_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
        selection_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            fitted, training_meta = _fit_groups(seed, data_by_seed[seed], random_items[seed], split, out, log)
            fitted_by_seed[seed] = fitted
            selection_rows.extend(training_meta["selection_rows"])
        _write_csv(out / "selection_grid.csv", selection_rows)
        _status(out, "training", "COMPLETE", detail="twelve portable JSON checkpoints selected on reliability_tune")

        tune_metrics: list[dict[str, Any]] = []
        predictions_dir = out / "predictions"
        for seed in SEEDS:
            for k in K_TRAIN:
                item = random_items[seed][k]
                probs = _predict_all(seed, item, fitted_by_seed[seed])
                _save_npz(
                    predictions_dir / f"randomK{k}_b3_seed{seed}.npz",
                    sentence_id=item["sentence_id"], ref_id=item["ref_id"], image_id=item["image_id"],
                    eval_split=item["eval_split"], candidate_indices=item["candidate_indices"],
                    scores=item["scores"], grounding_correct=item["correct"],
                    probabilities=probs, group_names=np.asarray(GROUPS),
                )
                _, tune_mask, test_mask = _split_masks(split, item)
                for group_index, group in enumerate(GROUPS):
                    for split_name, mask in (("reliability_tune", tune_mask), ("testA_testB", test_mask)):
                        packed = _metric_pack(probs[mask, group_index], item["correct"][mask])
                        tune_metrics.append({
                            "seed": f"b3_seed{seed}", "K": k, "group": group,
                            "split": split_name, "n_images": int(np.unique(item["image_id"][mask]).size), **packed,
                        })
        _write_csv(out / "train_stage_point_metrics.csv", tune_metrics)
        result = {
            "status": "PASS",
            "completed_utc": _now(),
            "backbone": "openclip_b16",
            "train_images": int(split.train_images.size), "tune_images": int(split.tune_images.size),
            "n_models": len(SEEDS) * len(GROUPS),
            "model_paths": [f"models/b3_seed{s}/{group}.json" for s in SEEDS for group in GROUPS],
            "selection_grid": "selection_grid.csv",
            "random_k5_k10_point_metrics": "train_stage_point_metrics.csv",
            "fit_input_fingerprints": "fit_stage_fingerprints.json",
            "selection_rule": "C grid 0.1/1/10, equal mean reliability_tune AUROC K5/K10, lower C wins if within 0.002",
        }
        _write_json(out / "fit_stage.json", result)
        _status(out, "fit_stage", "COMPLETE", detail="all train/tune model artifacts and random K5/K10 point predictions are ready")
        log("B16 fit stage complete; portable model paths are available for C3 inference")
        return result
    except Exception as exc:
        _status(out, "fit_stage", "FAILED", detail=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        handle.close()


def _score_anchor_forward(
    data_by_seed: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    records_by_k: Mapping[int, Mapping[int, Any]],
    corpus: B3Corpus,
    out: Path,
    log: Any,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Replay the frozen B16 scorers and verify raw K5/K10 score anchors."""
    from ccg.semantic.hard_scores import load_frozen_scorers

    scorers = load_frozen_scorers(b3_root=B16_SCORER_ROOT, seeds=SEEDS, device="cpu")
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        model = scorers[f"b3_seed{seed}"]
        for k in K_TRAIN:
            reference = data_by_seed[seed][k]
            record_map = records_by_k[k]
            missing = [int(sid) for sid in reference["sentence_id"] if int(sid) not in record_map]
            if missing:
                raise AssertionError(f"seed{seed} K{k}: {len(missing)} canonical rows lack Phase 0A records")
            actual = np.empty_like(reference["scores"], dtype=np.float32)
            for start in range(0, len(actual), CHUNK_ROWS):
                stop = min(start + CHUNK_ROWS, len(actual))
                samples = [record_map[int(sid)] for sid in reference["sentence_id"][start:stop]]
                batch = materialise_examples(corpus, samples, k)
                actual[start:stop] = score_examples(model, batch, batch_size=64)
                del batch, samples
            delta = verify_against_raw_scores(
                {f"b3_seed{seed}": actual}, {f"b3_seed{seed}": reference["scores"]},
                k=k, atol=ANCHOR_ATOL,
            )[f"b3_seed{seed}"]
            rows.append({
                "seed": f"b3_seed{seed}", "K": k, "n": int(len(actual)),
                "max_abs_error": delta, "tolerance": ANCHOR_ATOL,
                "status": "PASS",
                "checkpoint": str((B16_SCORER_ROOT / f"seed_{seed}" / "model.npz").relative_to(ROOT)),
                "raw_score_anchor": str((B16_SCORER_ROOT / f"seed_{seed}" / "raw_scores" / f"K{k}.npz").relative_to(ROOT)),
            })
            log(f"seed{seed} K{k}: frozen B16 forward anchor PASS, max|delta|={delta:.3e} (atol {ANCHOR_ATOL:.1e})")
    _write_json(out / "anchor_forward.json", {
        "status": "PASS", "tolerance": ANCHOR_ATOL,
        "method": "ccg.semantic.hard_scores.verify_against_raw_scores",
        "rows": rows,
    })
    return scorers, rows


def _manifest_cell(cell: str) -> dict[str, np.ndarray]:
    path = PHASE1F_ROOT / "manifests" / f"{cell}_candidates.npz"
    with np.load(path, allow_pickle=False) as data:
        item = {key: np.asarray(data[key]) for key in data.files}
    required = ("sentence_id", "ref_id", "image_id", "candidate_indices", "target_index")
    if any(key not in item for key in required):
        raise ValueError(f"{path}: missing one of frozen candidate identity fields {required}")
    k = int(item["candidate_indices"].shape[1])
    assert_candidate_identities(
        item["sentence_id"], item["ref_id"], item["image_id"],
        item["candidate_indices"], item["target_index"], expected_k=k,
    )
    return item


def _build_alternative_items(
    data_by_seed: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    random_items: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    records_by_k: Mapping[int, Mapping[int, Any]],
    split_by_sid: Mapping[int, str],
    scorers: Mapping[str, Any],
    corpus: B3Corpus,
    out: Path,
    log: Any,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    """Score saved rand/hard/dose candidate identities with frozen B16 checkpoints."""
    result: dict[int, dict[str, dict[str, np.ndarray]]] = {seed: {} for seed in SEEDS}
    candidate_rows: list[dict[str, Any]] = []
    loaded = {cell: _manifest_cell(cell) for cell in ALT_CELLS}

    # Check exact matched cohort identities before doing any candidate forward.
    for left, right in (("rand5", "hard5"), ("expb_m0", "expb_m8")):
        assert_row_alignment(
            loaded[left], loaded[right],
            required_fields=("ref_id", "image_id"),
            reference_name=f"frozen-candidates-{left}", candidate_name=f"frozen-candidates-{right}",
        )
        if not np.array_equal(loaded[left]["sentence_id"], loaded[right]["sentence_id"]):
            raise AssertionError(f"{left}/{right}: frozen paired candidate rows differ by sentence_id")
        if not np.array_equal(loaded[left]["target_index"], loaded[right]["target_index"]):
            raise AssertionError(f"{left}/{right}: paired candidate targets differ")

    for cell in ALT_CELLS:
        manifest = loaded[cell]
        k = int(manifest["candidate_indices"].shape[1])
        rows = len(manifest["sentence_id"])
        if any(int(sid) not in split_by_sid for sid in manifest["sentence_id"]):
            raise AssertionError(f"{cell}: candidate manifest includes rows outside the frozen score cohort")
        eval_split = np.asarray([split_by_sid[int(sid)] for sid in manifest["sentence_id"]])
        if not np.all(np.isin(eval_split, ("testA", "testB"))):
            raise AssertionError(f"{cell}: frozen Phase 1F cell contains non-test rows")
        for field, source in (("ref_id", "ref_id"), ("image_id", "image_id")):
            identity = {
                int(sid): int(value)
                for sid, value in zip(data_by_seed[1][5]["sentence_id"], data_by_seed[1][5][source], strict=True)
            }
            observed = np.asarray([identity[int(sid)] for sid in manifest["sentence_id"]], dtype=np.int64)
            if not np.array_equal(observed, np.asarray(manifest[field], dtype=np.int64)):
                raise AssertionError(f"{cell}: saved {field} differs from the frozen B16 scorer cohort")
        candidate_rows.append({
            "cell": cell, "K": k, "n_rows": rows, "n_images": int(np.unique(manifest["image_id"]).size),
            "manifest": str((PHASE1F_ROOT / "manifests" / f"{cell}_candidates.npz").relative_to(ROOT)),
            "manifest_sha256": _sha256(PHASE1F_ROOT / "manifests" / f"{cell}_candidates.npz"),
            "sentence_id_sha256": _sha256_array(manifest["sentence_id"]),
            "candidate_indices_sha256": _sha256_array(manifest["candidate_indices"]),
        })

        if cell == "rand5":
            # Reuse the already-anchored random K5 model outputs; prove that the
            # Phase 1F manifest is exactly the same candidate construction.
            canonical = data_by_seed[1][5]
            sid_to_pos = {int(sid): pos for pos, sid in enumerate(canonical["sentence_id"])}
            positions = np.asarray([sid_to_pos[int(sid)] for sid in manifest["sentence_id"]], dtype=np.int64)
            expected_candidates = random_items[1][5]["candidate_indices"][positions]
            if not np.array_equal(expected_candidates, manifest["candidate_indices"]):
                raise AssertionError("rand5 Phase 1F identities differ from the B16 frozen random K5 identities")
            for seed in SEEDS:
                base_item = random_items[seed][5]
                scores = np.asarray(base_item["scores"][positions], dtype=np.float32)
                result[seed][cell] = {
                    key: np.asarray(base_item[key][positions])
                    for key in ("sentence_id", "ref_id", "image_id", "candidate_indices", "scores", "correct", "stats17", "sem16")
                }
                result[seed][cell]["eval_split"] = eval_split.copy()
            log(f"rand5: exact candidate identities verified against B16 random K5; rows={rows}")
            continue

        samples = [
            SimpleNamespace(
                sentence_id=int(sid), ref_id=int(ref), image_id=int(image),
                target_index=int(target), distractor_order=np.asarray(candidates[1:], dtype=np.int64),
            )
            for sid, ref, image, target, candidates in zip(
                manifest["sentence_id"], manifest["ref_id"], manifest["image_id"],
                manifest["target_index"], manifest["candidate_indices"], strict=True,
            )
        ]
        score_by_seed = {seed: np.empty((rows, k), dtype=np.float32) for seed in SEEDS}
        semantic_by_seed = {seed: np.empty((rows, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64) for seed in SEEDS}
        for start in range(0, rows, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, rows)
            batch_samples = samples[start:stop]
            batch = materialise_examples(corpus, batch_samples, k)
            for seed in SEEDS:
                seed_name = f"b3_seed{seed}"
                scores = score_examples(scorers[seed_name], batch, batch_size=64)
                score_by_seed[seed][start:stop] = scores
                semantic_by_seed[seed][start:stop] = compute_semantic_stats_chunked(
                    batch.z_q, batch.z_i, scores, chunk_rows=CHUNK_ROWS,
                )
            del batch, batch_samples
            if stop == rows or stop % 2048 == 0:
                log(f"{cell}: frozen B16 scorer forwards from cached embeddings {stop}/{rows} rows")

        for seed in SEEDS:
            scores = score_by_seed[seed]
            correct = np.argmax(scores, axis=1) == 0
            if not np.array_equal(manifest["candidate_indices"][:, 0], manifest["target_index"]):
                raise AssertionError(f"{cell}: target column moved before frozen B16 inference")
            stats = rfeat.stat_features(scores, temperature=_read_temperature(seed))
            item = {
                "sentence_id": np.asarray(manifest["sentence_id"], dtype=np.int64),
                "ref_id": np.asarray(manifest["ref_id"], dtype=np.int64),
                "image_id": np.asarray(manifest["image_id"], dtype=np.int64),
                "eval_split": eval_split.copy(),
                "candidate_indices": np.asarray(manifest["candidate_indices"], dtype=np.int64),
                "scores": scores,
                "correct": np.asarray(correct, dtype=bool),
                "stats17": np.asarray(stats, dtype=np.float64),
                "sem16": semantic_by_seed[seed],
            }
            _save_feature_item(_feature_path(out, cell, seed), item)
            result[seed][cell] = item
        log(f"{cell}: saved exact-identity B16 derived inputs and frozen scores ({rows} rows, K={k})")

    # Candidate-blind scorers must give the identical target score when only
    # distractor identities change on a paired row.
    for seed in SEEDS:
        for left, right in (("rand5", "hard5"), ("expb_m0", "expb_m8")):
            a = result[seed][left]["scores"][:, 0]
            b = result[seed][right]["scores"][:, 0]
            delta = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))
            if delta > ANCHOR_ATOL:
                raise AssertionError(f"seed{seed} {left}/{right}: target score changed by {delta:.3e}")
            log(f"seed{seed} {left}/{right}: target score invariant PASS, max|delta|={delta:.3e}")
    _write_json(out / "candidate_identity_checks.json", {
        "status": "PASS", "matched_pairs": ["rand5/hard5", "expb_m0/expb_m8"],
        "candidate_index_anchor": "saved Phase 1F manifests, loaded verbatim",
        "target_score_atol": ANCHOR_ATOL,
        "cells": [{key: value for key, value in row.items() if key != "candidate_indices"} for row in candidate_rows],
    })
    return result, candidate_rows


def _predict_all(
    seed: int,
    item: Mapping[str, np.ndarray],
    fitted: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    matrix = np.empty((len(item["sentence_id"]), len(GROUPS)), dtype=np.float64)
    for group_index, group in enumerate(GROUPS):
        raw = feature_block(item["stats17"], item["sem16"], group)
        standardized = rfeat.normalize_apply(raw, fitted[group]["normalization"])
        matrix[:, group_index] = fitted[group]["predictor"].predict_proba(standardized)
    if matrix.shape != (len(item["correct"]), len(GROUPS)):
        raise AssertionError(f"seed{seed}: group predictions are not aligned to correctness")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise AssertionError(f"seed{seed}: a group produced an invalid probability")
    return matrix


def _risk(confidence: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """The shared V3 risk convention: partial uniform acceptance at tied cutoff."""
    from ccg.v3.evaluation import fractional_boundary_risk

    return float(fractional_boundary_risk(confidence, correct, coverage))


def _metric_pack(probability: np.ndarray, correct: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(correct, dtype=bool).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if p.size != y.size or p.size == 0:
        raise ValueError("probability and correctness must be non-empty aligned vectors")
    auc: float | None = None
    if np.unique(y).size > 1:
        auc = float(auroc_correct(p, y))
    return {
        "auroc_correct": auc,
        "risk_at_50": _risk(p, y, 0.50),
        "risk_at_80": _risk(p, y, 0.80),
        "nll_binary": float(nll_binary(p, y)),
        "accuracy": float(np.mean(y)),
        "n": int(y.size),
    }


def _metric_and_prediction_rows(
    data_by_seed: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    random_items: Mapping[int, Mapping[int, Mapping[str, np.ndarray]]],
    alternative_items: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    fitted_by_seed: Mapping[int, Mapping[str, Mapping[str, Any]]],
    out: Path,
    log: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    metrics: list[dict[str, Any]] = []
    point_effects: list[dict[str, Any]] = []
    predictions: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in SEEDS}
    items_by_seed: dict[int, dict[str, Mapping[str, np.ndarray]]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for k in K_RANDOM_EVAL:
            items_by_seed[seed][f"randomK{k}"] = random_items[seed][k]
        items_by_seed[seed].update(alternative_items[seed])
        for cell in CELLS:
            item = items_by_seed[seed][cell]
            probs = _predict_all(seed, item, fitted_by_seed[seed])
            predictions[seed][cell] = probs
            _save_npz(
                out / "predictions" / f"{cell}_b3_seed{seed}.npz",
                sentence_id=np.asarray(item["sentence_id"], dtype=np.int64),
                ref_id=np.asarray(item["ref_id"], dtype=np.int64),
                image_id=np.asarray(item["image_id"], dtype=np.int64),
                eval_split=np.asarray(item["eval_split"]),
                candidate_indices=np.asarray(item["candidate_indices"], dtype=np.int64),
                scores=np.asarray(item["scores"], dtype=np.float32),
                grounding_correct=np.asarray(item["correct"], dtype=bool),
                probabilities=probs,
                group_names=np.asarray(GROUPS),
            )
            test_mask = np.isin(item["eval_split"], ("testA", "testB"))
            if not test_mask.any():
                raise AssertionError(f"seed{seed} {cell}: no original test rows")
            test = np.flatnonzero(test_mask)
            shared_correct = np.asarray(item["correct"], dtype=bool)[test]
            for group_index, group in enumerate(GROUPS):
                packed = _metric_pack(probs[test, group_index], shared_correct)
                metrics.append({
                    "seed": f"b3_seed{seed}", "cell": cell,
                    "K": int(np.asarray(item["scores"]).shape[1]), "group": group,
                    "split": "testA_testB", "n_images": int(np.unique(np.asarray(item["image_id"])[test]).size),
                    **packed,
                })
        # Every group for a cell uses exactly the frozen scorer's same labels.
        for cell in CELLS:
            item = items_by_seed[seed][cell]
            probs = predictions[seed][cell]
            if probs.shape[0] != len(item["correct"]):
                raise AssertionError(f"seed{seed} {cell}: probability row count changed")

        # Save chosen configuration and all tune-grid scores in portable reports.
        for group in GROUPS:
            for metric in ("auroc_correct", "risk_at_50", "risk_at_80"):
                for cell in CELLS:
                    item = items_by_seed[seed][cell]
                    mask = np.isin(item["eval_split"], ("testA", "testB"))
                    group_pos = GROUPS.index(group)
                    value = _metric_pack(predictions[seed][cell][mask, group_pos], np.asarray(item["correct"])[mask])[metric]
                    point_effects.append({
                        "seed": f"b3_seed{seed}", "cell": cell, "group": group,
                        "metric": metric, "effect": value, "contrast": "absolute",
                        "interpretation": "higher AUROC is better; lower risk is better",
                    })

    # Paired Full - (S + Q) effects share the exact same rows by construction.
    contrasts = (("Full_minus_S_plus_Q", "Full", "S+Q"),)
    for seed in SEEDS:
        for cell in CELLS:
            item = items_by_seed[seed][cell]
            mask = np.isin(item["eval_split"], ("testA", "testB"))
            y = np.asarray(item["correct"], dtype=bool)[mask]
            p = predictions[seed][cell][mask]
            a = _metric_pack(p[:, GROUPS.index("Full")], y)
            b = _metric_pack(p[:, GROUPS.index("S+Q")], y)
            for label, _group_a, _group_b in contrasts:
                for metric in ("auroc_correct", "risk_at_50", "risk_at_80"):
                    av, bv = a[metric], b[metric]
                    effect = None if av is None or bv is None else float(av) - float(bv)
                    point_effects.append({
                        "seed": f"b3_seed{seed}", "cell": cell,
                        "group": label, "metric": metric,
                        "effect": effect, "contrast": "Full_minus_S_plus_Q",
                        "interpretation": (
                            "positive favors Full" if metric == "auroc_correct" else
                            "negative favors Full"
                        ),
                    })
        for contrast, left, right in (
            ("Hard_minus_random__Full_minus_SQ", "hard5", "rand5"),
            ("m8_minus_m0__Full_minus_SQ", "expb_m8", "expb_m0"),
        ):
            left_item, right_item = items_by_seed[seed][left], items_by_seed[seed][right]
            assert_row_alignment(
                left_item, right_item,
                required_fields=("ref_id", "image_id", "eval_split"),
                reference_name=left, candidate_name=right,
            )
            for metric in ("auroc_correct", "risk_at_50", "risk_at_80"):
                left_p = predictions[seed][left]
                right_p = predictions[seed][right]
                y_left = np.asarray(left_item["correct"], dtype=bool)
                y_right = np.asarray(right_item["correct"], dtype=bool)
                left_full = _metric_pack(left_p[:, GROUPS.index("Full")], y_left)[metric]
                left_sq = _metric_pack(left_p[:, GROUPS.index("S+Q")], y_left)[metric]
                right_full = _metric_pack(right_p[:, GROUPS.index("Full")], y_right)[metric]
                right_sq = _metric_pack(right_p[:, GROUPS.index("S+Q")], y_right)[metric]
                left_effect = None if left_full is None or left_sq is None else float(left_full) - float(left_sq)
                right_effect = None if right_full is None or right_sq is None else float(right_full) - float(right_sq)
                point_effects.append({
                    "seed": f"b3_seed{seed}", "cell": f"{left}_vs_{right}",
                    "group": contrast, "metric": metric,
                    "effect": None if left_effect is None or right_effect is None else float(left_effect) - float(right_effect),
                    "contrast": "paired_difference_of_differences",
                    "interpretation": (
                        "positive means the high-hardness cell increases Full's AUROC gain" if metric == "auroc_correct" else
                        "negative means the high-hardness cell increases Full's risk reduction"
                    ),
                })

    _write_csv(out / "metrics.csv", metrics)
    _write_csv(out / "point_effects.csv", point_effects)
    _write_sentence_predictions(out, items_by_seed, predictions)
    log(f"wrote test metrics for {len(metrics)} seed/cell/group rows and per-sentence predictions")
    return metrics, point_effects, predictions


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_sentence_predictions(
    out: Path,
    items_by_seed: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    predictions: Mapping[int, Mapping[str, np.ndarray]],
) -> None:
    path = out / "semantic_ablation_predictions.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "seed", "cell", "K", "sentence_id", "ref_id", "image_id", "eval_split",
        "grounding_correct", "candidate_indices", "prob_S", "prob_S_plus_Q", "prob_S_plus_V", "prob_Full",
    )
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for seed in SEEDS:
            for cell in CELLS:
                item = items_by_seed[seed][cell]
                probs = predictions[seed][cell]
                k = int(np.asarray(item["scores"]).shape[1])
                for row in range(len(item["sentence_id"])):
                    writer.writerow({
                        "seed": f"b3_seed{seed}", "cell": cell, "K": k,
                        "sentence_id": int(item["sentence_id"][row]),
                        "ref_id": int(item["ref_id"][row]),
                        "image_id": int(item["image_id"][row]),
                        "eval_split": str(item["eval_split"][row]),
                        "grounding_correct": int(item["correct"][row]),
                        "candidate_indices": json.dumps(np.asarray(item["candidate_indices"][row], dtype=np.int64).tolist()),
                        "prob_S": float(probs[row, 0]), "prob_S_plus_Q": float(probs[row, 1]),
                        "prob_S_plus_V": float(probs[row, 2]), "prob_Full": float(probs[row, 3]),
                    })


def _write_csv_or_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_csv(path, rows)


def run_replication(*, out: Path = OUT_ROOT) -> dict[str, Any]:
    """Fit all twelve portable models, replay anchors, and evaluate frozen COCO cells."""
    out = Path(out).resolve()
    if out != OUT_ROOT.resolve():
        raise ValueError(f"V3 replication outputs are restricted to {OUT_ROOT}; refusing {out}")
    out.mkdir(parents=True, exist_ok=True)
    log, handle = _logger(out / "replication.log")
    try:
        import torch
        from threadpoolctl import threadpool_limits

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True)
        threadpool_limits(limits=2)
        _write_json(out / "execution.json", {
            "started_utc": _now(), "pid": os.getpid(), "device": "cpu",
            "torch_threads": 2, "blas_threads": 2,
            "command": "python -u scripts/v3_replicate_features.py --phase run",
            "bootstrap": "not launched; parent allocation required",
            "frozen_group_order": list(GROUPS), "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
        })
        _status(out, "input_fingerprints", "RUNNING", detail="hashing frozen caches, scorer anchors, candidate manifests and feature definitions")
        fingerprints = _input_manifest()
        _write_json(out / "input_fingerprints.json", fingerprints)
        log(f"fingerprinted {len(fingerprints['files'])} frozen inputs")

        split = sdata.load_split_from_manifest(SPLIT_PATH)
        if split.train_images.size != 523 or split.tune_images.size != 224:
            raise AssertionError(f"frozen reliability split expected 523/224 images, got {split.train_images.size}/{split.tune_images.size}")
        _write_json(out / "split_manifest_reuse.json", split.to_dict() | {
            "source": str(SPLIT_PATH.relative_to(ROOT)),
            "source_sha256": fingerprints["files"][str(SPLIT_PATH.relative_to(ROOT)).replace("\\", "/")]["sha256"],
            "frozen_reused": True,
        })
        _status(out, "input_fingerprints", "COMPLETE", detail="exact B16 caches, scorer checkpoints, split, code and candidate manifest hashes recorded")

        _status(out, "data_alignment", "RUNNING", detail="loading B16 raw-score anchors and original 523/224 image split")
        data_by_seed: dict[int, dict[int, dict[str, np.ndarray]]] = {}
        for seed in SEEDS:
            data_by_seed[seed] = {k: _canonical_random(seed, k) for k in K_RANDOM_EVAL}
            _validate_random_rows(seed, data_by_seed[seed])
            sdata.assert_split_matches_fresh(split, data_by_seed[seed][5]["eval_split"], data_by_seed[seed][5]["image_id"])
        _status(out, "data_alignment", "COMPLETE", detail="all seed/K identities and nested correctness align")

        _status(out, "cached_features", "RUNNING", detail="computing S/Q/V from existing B16 embeddings and H5 region/text features")
        corpus = _make_corpus(preload=True, ks=(5, 10, 50))
        records_by_k = {k: _record_map(corpus, (k,)) for k in K_RANDOM_EVAL}
        canonical_ids = data_by_seed[1][5]["sentence_id"]
        for k, records in records_by_k.items():
            canonical = data_by_seed[1][k]
            expected_ids = set(int(value) for value in canonical["sentence_id"])
            missing_ids = expected_ids.difference(records)
            if missing_ids:
                raise AssertionError(
                    f"Phase 0A B16 cached-feature K{k} rows omit {len(missing_ids)} canonical scorer rows"
                )
            # Phase 0A lists all candidate-eligible rows; the frozen Phase 0.5
            # scorer cohort is a strict subset. Verify every used identity and
            # ignore rows outside that already-fixed scorer cohort.
            row_pos = {int(sid): pos for pos, sid in enumerate(canonical["sentence_id"])}
            for sid in expected_ids:
                record = records[sid]
                pos = row_pos[sid]
                observed = (int(record.ref_id), int(record.image_id), str(record.eval_split))
                expected = (
                    int(canonical["ref_id"][pos]), int(canonical["image_id"][pos]),
                    str(canonical["eval_split"][pos]),
                )
                if observed != expected:
                    raise AssertionError(f"Phase 0A B16 cached-feature K{k} identity mismatch at sentence {sid}: {observed} != {expected}")
            log(
                f"Phase 0A K{k}: {len(expected_ids)} canonical scorer rows align; "
                f"{len(records) - len(expected_ids)} non-cohort Phase 0A rows excluded"
            )
        random_items: dict[int, dict[int, dict[str, np.ndarray]]] = {seed: {} for seed in SEEDS}
        for k in K_RANDOM_EVAL:
            if k in (5, 10):
                _validate_store(k, data_by_seed[1][k])
            for seed in SEEDS:
                random_items[seed][k] = _random_feature_item(
                    seed, k, data_by_seed[seed][k], corpus, records_by_k[k], out, log,
                )
        _status(out, "cached_features", "COMPLETE", detail="random K5/K10 used frozen B16 embedding NPZs; K50 and COCO inputs use cached feature H5 with chunked semantic summaries")

        _status(out, "anchor_forward", "RUNNING", detail=f"replaying frozen B16 scorer checkpoints on random K5/K10; stop tolerance {ANCHOR_ATOL:g}")
        scorers, anchor_rows = _score_anchor_forward(data_by_seed, records_by_k, corpus, out, log)
        _status(out, "anchor_forward", "COMPLETE", detail="all six seed/K forward anchors passed unchanged tolerance")

        _status(out, "training", "RUNNING", detail="CPU logistic fits, two BLAS threads, train-only normalization, frozen C grid and tune tie rule")
        fit_stage_path = out / "fit_stage.json"
        if fit_stage_path.exists() and json.loads(fit_stage_path.read_text(encoding="utf-8")).get("status") == "PASS":
            fitted_by_seed = _load_fitted_models(out)
            selection_rows = [
                {"seed": f"b3_seed{seed}", "group": group, **dict(row)}
                for seed in SEEDS for group in GROUPS
                for row in fitted_by_seed[seed][group]["payload"].get("selection_grid", [])
            ]
            log("restored all 12 fit-stage JSON checkpoints; no duplicate fitting")
        else:
            fitted_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
            selection_rows: list[dict[str, Any]] = []
            for seed in SEEDS:
                fit, training_meta = _fit_groups(seed, data_by_seed[seed], random_items[seed], split, out, log)
                fitted_by_seed[seed] = fit
                selection_rows.extend(training_meta["selection_rows"])
        _write_csv(out / "selection_grid.csv", selection_rows)
        _status(out, "training", "COMPLETE", detail="twelve portable per-seed/per-group JSON models fit and round-tripped")

        _status(out, "alternative_cells", "RUNNING", detail="loading saved Phase 1F candidate IDs and scoring them with frozen B16 checkpoints")
        split_by_sid = {
            int(sid): str(eval_split)
            for sid, eval_split in zip(data_by_seed[1][5]["sentence_id"], data_by_seed[1][5]["eval_split"], strict=True)
        }
        alternative_items, candidate_rows = _build_alternative_items(
            data_by_seed, random_items, records_by_k, split_by_sid, scorers, corpus, out, log,
        )
        _write_csv(out / "candidate_manifest_fingerprints.csv", candidate_rows)
        _status(out, "alternative_cells", "COMPLETE", detail="exact saved candidate identities evaluated; paired target-score checks passed")

        _status(out, "point_evaluation", "RUNNING", detail="group AUROC, V3 fractional-boundary risk and losses on pooled test rows")
        metrics, point_effects, _predictions = _metric_and_prediction_rows(
            data_by_seed, random_items, alternative_items, fitted_by_seed, out, log,
        )
        _status(out, "point_evaluation", "COMPLETE", detail=f"wrote {len(metrics)} group/cell test metric rows and per-sentence probabilities")

        summary = {
            "artifact": "v3_b16_s_q_v_replication",
            "status": "POINT_ESTIMATES_COMPLETE_BOOTSTRAP_PENDING_PARENT_ALLOCATION",
            "backbone": "openclip_b16",
            "temperature_corrected": {f"b3_seed{seed}": _read_temperature(seed) for seed in SEEDS},
            "train_images": int(split.train_images.size), "tune_images": int(split.tune_images.size),
            "feature_groups": {group: list(FEATURE_GROUPS[group]) for group in GROUPS},
            "anchor_forward": anchor_rows,
            "n_model_checkpoints": len(GROUPS) * len(SEEDS),
            "metrics_file": "metrics.csv", "point_effects_file": "point_effects.csv",
            "predictions_file": "semantic_ablation_predictions.csv.gz",
            "bootstrap": {"status": "WAITING_FOR_PARENT_ALLOCATION", "planned_replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "provenance_file": "input_fingerprints.json",
            "classification": "developmental/configuration replication on the already-observed COCO test cohort; not independent confirmation",
        }
        _write_json(out / "replication_summary.json", summary)
        _status(out, "bootstrap", "WAITING", detail="parent must allocate the formal 5,000-replicate audit-stat slot")
        _status(out, "overall", "PREPARED", detail="point estimates and portable models complete; formal bootstrap awaits parent allocation")
        return summary
    except Exception as exc:
        _status(out, "overall", "FAILED", detail=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        handle.close()


def run_bootstrap(*, out: Path = OUT_ROOT, n_replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    """Run shared image-cluster draws after the parent has allocated resources."""
    out = Path(out).resolve()
    if out != OUT_ROOT.resolve():
        raise ValueError(f"V3 replication outputs are restricted to {OUT_ROOT}; refusing {out}")
    if int(n_replicates) != BOOTSTRAP_REPLICATES:
        raise ValueError(f"formal V3 bootstrap is frozen at {BOOTSTRAP_REPLICATES} replicates")
    log, handle = _logger(out / "replication.log")
    try:
        summary_path = out / "replication_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError("point-evaluation artifacts are missing; run --phase run first")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") not in {
            "POINT_ESTIMATES_COMPLETE_BOOTSTRAP_PENDING_PARENT_ALLOCATION", "BOOTSTRAP_RUNNING",
        }:
            raise ValueError(f"unexpected replication status before bootstrap: {summary.get('status')}")
        _status(out, "bootstrap", "RUNNING", detail=f"formal shared image-cluster bootstrap n={BOOTSTRAP_REPLICATES}, seed={BOOTSTRAP_SEED}")
        log(f"starting formal shared image-cluster bootstrap reps={BOOTSTRAP_REPLICATES}, seed={BOOTSTRAP_SEED}")

        def _load_cell(cell: str, seed: int) -> dict[str, np.ndarray]:
            path = out / "predictions" / f"{cell}_b3_seed{seed}.npz"
            with np.load(path, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}

        def _cell_pack(cell: str) -> dict[int, dict[str, np.ndarray]]:
            return {seed: _load_cell(cell, seed) for seed in SEEDS}

        rows_by_cohort = {
            "random_all": {cell: _cell_pack(cell) for cell in ("randomK5", "randomK10", "randomK50")},
            "same4": {cell: _cell_pack(cell) for cell in ("rand5", "hard5")},
            "same8": {cell: _cell_pack(cell) for cell in ("expb_m0", "expb_m8")},
        }
        bootstrap_results: dict[str, Any] = {}
        for cohort_name, cells in rows_by_cohort.items():
            first_cell = next(iter(cells))
            anchor = cells[first_cell][SEEDS[0]]
            test_rows = np.flatnonzero(np.isin(anchor["eval_split"], ("testA", "testB")))
            canonical = {
                seed: {
                    cell: {
                        key: np.asarray(item[key])[test_rows]
                        for key in ("sentence_id", "ref_id", "image_id", "eval_split", "grounding_correct", "probabilities")
                    }
                    for cell, cell_rows in cells.items()
                    for item in (cell_rows[seed],)
                }
                for seed in SEEDS
            }
            # All cells in an evaluation cohort must have exactly the same row order.
            for cell in cells:
                for seed in SEEDS:
                    assert_row_alignment(
                        canonical[SEEDS[0]][first_cell], canonical[seed][cell],
                        required_fields=("ref_id", "image_id", "eval_split"),
                        canonical_id_field="sentence_id",
                        reference_name=f"{first_cell}-seed{SEEDS[0]}", candidate_name=f"{cell}-seed{seed}",
                    )
            image_ids = canonical[SEEDS[0]][first_cell]["image_id"]

            def statistic(indices: np.ndarray) -> dict[str, dict[str, float]]:
                estimates: dict[str, dict[str, float]] = {}
                metrics_by_cell_seed_group: dict[tuple[str, int, str], dict[str, float | None]] = {}
                for cell, cell_rows in cells.items():
                    for group_index, group in enumerate(GROUPS):
                        estimates[f"absolute__{cell}__{group}__auroc_correct"] = {}
                        estimates[f"absolute__{cell}__{group}__risk_at_50"] = {}
                        estimates[f"absolute__{cell}__{group}__risk_at_80"] = {}
                        for seed in SEEDS:
                            item = canonical[seed][cell]
                            y = np.asarray(item["grounding_correct"], dtype=bool)[indices]
                            p = np.asarray(item["probabilities"], dtype=np.float64)[indices, group_index]
                            packed = _metric_pack(p, y)
                            metrics_by_cell_seed_group[(cell, seed, group)] = packed
                            for metric in ("auroc_correct", "risk_at_50", "risk_at_80"):
                                name = f"absolute__{cell}__{group}__{metric}"
                                estimates[name][f"b3_seed{seed}"] = float("nan") if packed[metric] is None else float(packed[metric])
                    for metric in ("auroc_correct", "risk_at_50", "risk_at_80"):
                        name = f"Full_minus_SQ__{cell}__{metric}"
                        estimates[name] = {}
                        for seed in SEEDS:
                            a = metrics_by_cell_seed_group[(cell, seed, "Full")][metric]
                            b = metrics_by_cell_seed_group[(cell, seed, "S+Q")][metric]
                            estimates[name][f"b3_seed{seed}"] = float("nan") if a is None or b is None else float(a) - float(b)
                paired = None
                if cohort_name == "same4":
                    paired = ("hard5", "rand5", "Hard_minus_random__Full_minus_SQ")
                elif cohort_name == "same8":
                    paired = ("expb_m8", "expb_m0", "m8_minus_m0__Full_minus_SQ")
                if paired:
                    high, low, prefix = paired
                    for metric in ("auroc_correct", "risk_at_50", "risk_at_80"):
                        name = f"{prefix}__{metric}"
                        estimates[name] = {}
                        for seed in SEEDS:
                            a1 = metrics_by_cell_seed_group[(high, seed, "Full")][metric]
                            b1 = metrics_by_cell_seed_group[(high, seed, "S+Q")][metric]
                            a0 = metrics_by_cell_seed_group[(low, seed, "Full")][metric]
                            b0 = metrics_by_cell_seed_group[(low, seed, "S+Q")][metric]
                            values = (a1, b1, a0, b0)
                            estimates[name][f"b3_seed{seed}"] = (
                                float(a1) - float(b1) - float(a0) + float(b0)
                                if all(value is not None for value in values) else float("nan")
                            )
                return estimates

            bootstrap_results[cohort_name] = shared_image_cluster_bootstrap(
                image_ids, statistic, n_replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED, ci=0.95,
            )
            log(f"bootstrap cohort {cohort_name}: completed {BOOTSTRAP_REPLICATES} shared image draws")

        serializable = {}
        raw_draws: dict[str, np.ndarray] = {}
        draw_manifest: dict[str, Any] = {}
        for cohort_name, result in bootstrap_results.items():
            cohort_first_cell = next(iter(rows_by_cohort[cohort_name]))
            cohort_anchor = rows_by_cohort[cohort_name][cohort_first_cell][SEEDS[0]]
            cohort_test_mask = np.isin(cohort_anchor["eval_split"], ("testA", "testB"))
            estimates_json = {}
            for estimate_index, (name, record) in enumerate(result["estimates"].items()):
                prefix = f"{cohort_name}_e{estimate_index:03d}"
                mean_key = f"{prefix}_mean"
                raw_draws[mean_key] = np.asarray(record["replicates"], dtype=np.float64)
                per_seed = {}
                seed_keys = {}
                for seed_key, seed_record in record["per_seed"].items():
                    seed_short = str(seed_key).replace("b3_seed", "seed")
                    array_key = f"{prefix}_{seed_short}"
                    raw_draws[array_key] = np.asarray(seed_record["replicates"], dtype=np.float64)
                    seed_keys[str(seed_key)] = array_key
                    per_seed[str(seed_key)] = {
                        key: (float(value) if isinstance(value, np.floating) else value)
                        for key, value in seed_record.items() if key != "replicates"
                    }
                record_json = {
                    key: (float(value) if isinstance(value, np.floating) else value)
                    for key, value in record.items()
                    if key not in {"replicates", "per_seed", "per_seed_replicates"}
                }
                record_json["per_seed"] = per_seed
                record_json["raw_draw_arrays"] = {"mean": mean_key, **seed_keys}
                record_json["interpretation"] = (
                    "higher AUROC is better; lower risk is better" if name.startswith("absolute__") else
                    "positive AUROC effect favors Full; negative risk effect favors Full" if name.startswith("Full_minus_SQ__") else
                    "positive AUROC difference means greater Full gain in the higher-hardness cell; negative risk difference means greater Full risk reduction there"
                )
                estimates_json[name] = record_json
                draw_manifest[f"{cohort_name}:{name}"] = {"mean": mean_key, "per_seed": seed_keys}
            serializable[cohort_name] = {
                "n_images": int(np.unique(cohort_anchor["image_id"][cohort_test_mask]).size),
                "n_rows": int(cohort_test_mask.sum()),
                "estimates": estimates_json,
            }
        _save_npz(out / "bootstrap_draws.npz", **raw_draws)
        _write_json(out / "bootstrap_draws_manifest.json", {
            "file": "bootstrap_draws.npz", "sha256": _sha256(out / "bootstrap_draws.npz"),
            "size_bytes": int((out / "bootstrap_draws.npz").stat().st_size),
            "keys": draw_manifest,
        })
        _write_json(out / "bootstrap_summary.json", {
            "status": "COMPLETE", "method": "ccg.repairs.bootstrap.shared_image_cluster_bootstrap",
            "classification": "developmental/configuration replication on an already-observed test set; not independent confirmation",
            "cluster": "image_id", "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED, "marginal_ci": 0.95,
            "risk_convention": "fractional_boundary_tie",
            "effect_direction": "higher AUROC and lower risk favor better reliability; negative Full-minus-S+Q risk means Full improves risk",
            "raw_draws": "bootstrap_draws.npz",
            "raw_draws_manifest": "bootstrap_draws_manifest.json",
            "contrasts_use_shared_draws": True, "cohorts": serializable,
        })
        summary["status"] = "COMPLETE"
        summary["bootstrap"] = {"status": "COMPLETE", "planned_replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "file": "bootstrap_summary.json"}
        _write_json(summary_path, summary)
        _status(out, "bootstrap", "COMPLETE", detail="three shared-draw cohorts completed")
        _status(out, "overall", "COMPLETE", detail="B16 S/Q/V replication, point metrics and formal bootstrap complete")
        log("formal bootstrap complete")
        return json.loads((out / "bootstrap_summary.json").read_text(encoding="utf-8"))
    except Exception as exc:
        _status(out, "bootstrap", "FAILED", detail=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        handle.close()


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "run", "bootstrap"), required=True)
    parser.add_argument("--out", type=Path, default=Path("results/v3_final_validation/replication"))
    parser.add_argument("--n-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "fit":
        run_fit_stage(out=args.out)
    elif args.phase == "run":
        run_replication(out=args.out)
    else:
        run_bootstrap(out=args.out, n_replicates=args.n_replicates)
    return 0

