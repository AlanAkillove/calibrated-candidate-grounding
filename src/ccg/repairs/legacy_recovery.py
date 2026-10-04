"""Schema and atomic writers for deterministic legacy-prediction recovery.

The recovery artifacts live only below ``results/research_repair_v1/statistics/recovery``.
Each archive is one (method, cohort, backbone, seed) unit and stores row identity
beside every cell's correctness labels and frozen-model probabilities.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RECOVERY_ROOT = ROOT / "results" / "research_repair_v1" / "statistics" / "recovery"
ARCHIVE_SCHEMA = "ccg.research_repair_v1.legacy_prediction_archive.v1"
MANIFEST_SCHEMA = "ccg.research_repair_v1.legacy_recovery_manifest.v1"
IDENTITY_FIELDS = ("sentence_id", "image_id", "ref_id", "eval_split", "correct")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path, *, frozen_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Hash one source and check it against the baseline when it was frozen there."""
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"source is outside the repository: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    relative = resolved.relative_to(ROOT.resolve()).as_posix()
    size = int(resolved.stat().st_size)
    digest = sha256(resolved)
    baseline = None if frozen_manifest is None else frozen_manifest.get(relative)
    if baseline is not None:
        if size != int(baseline["size_bytes"]) or digest != str(baseline["sha256"]):
            raise RuntimeError(f"frozen baseline source changed: {relative}")
        status = "BASELINE_HASH_MATCH"
    else:
        status = "SUPPLEMENTAL_SOURCE_SNAPSHOT"
    return {"path": relative, "size_bytes": size, "sha256": digest, "status": status}


def load_baseline_manifest(path: Path | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = path or ROOT / "results" / "research_repair_v1" / "input_manifest.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    return doc, {str(item["path"]): item for item in doc.get("inputs", [])}


def _replace_atomic(tmp: Path, target: Path) -> None:
    # Shared helper retries transient Windows file locks and preserves the target.
    try:
        from ccg.repairs.atomic import replace_with_retry
    except ImportError:
        os.replace(tmp, target)
    else:
        replace_with_retry(tmp, target)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _replace_atomic(tmp, path)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not token:
        raise ValueError(f"empty archive key derived from {value!r}")
    return token


def write_prediction_archive(
    archive_path: Path,
    sidecar_path: Path,
    *,
    job_id: str,
    scope: str,
    cohort: str,
    seed: int,
    cells: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and atomically write cell-prefixed row arrays plus a JSON sidecar.

    A cell contains all ``IDENTITY_FIELDS`` and a ``predictions`` mapping from
    the original model identity to one probability per canonical row.
    """
    if not cells:
        raise ValueError("a recovery archive must contain at least one cell")
    arrays: dict[str, np.ndarray] = {}
    cell_docs: list[dict[str, Any]] = []
    for cell_name, cell in cells.items():
        cell_token = _safe_token(cell_name)
        missing = [key for key in IDENTITY_FIELDS if key not in cell]
        if missing:
            raise ValueError(f"{job_id}/{cell_name}: missing identity fields {missing}")
        identity = {key: np.asarray(cell[key]) for key in IDENTITY_FIELDS}
        row_count = int(identity["sentence_id"].reshape(-1).size)
        if row_count == 0:
            raise ValueError(f"{job_id}/{cell_name}: empty cell")
        for key, values in identity.items():
            flat = values.reshape(-1)
            if flat.size != row_count:
                raise ValueError(f"{job_id}/{cell_name}: {key} has {flat.size} rows, expected {row_count}")
            identity[key] = flat
        identity["sentence_id"] = identity["sentence_id"].astype(np.int64, copy=False)
        identity["image_id"] = identity["image_id"].astype(np.int64, copy=False)
        identity["ref_id"] = identity["ref_id"].astype(np.int64, copy=False)
        identity["eval_split"] = identity["eval_split"].astype(str, copy=False)
        identity["correct"] = identity["correct"].astype(bool, copy=False)
        if np.unique(identity["sentence_id"]).size != row_count:
            raise ValueError(f"{job_id}/{cell_name}: sentence_id is not unique")
        for field, values in identity.items():
            arrays[f"{cell_token}__{field}"] = values

        prediction_doc: dict[str, str] = {}
        predictions = cell.get("predictions", {})
        if not predictions:
            raise ValueError(f"{job_id}/{cell_name}: no model predictions")
        for condition, raw in predictions.items():
            condition_token = _safe_token(condition)
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
            if values.size != row_count:
                raise ValueError(f"{job_id}/{cell_name}/{condition}: {values.size} predictions != {row_count} rows")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"{job_id}/{cell_name}/{condition}: probabilities must be finite and in [0,1]")
            array_key = f"{cell_token}__confidence__{condition_token}"
            if array_key in arrays:
                raise ValueError(f"archive key collision: {array_key}")
            arrays[array_key] = values
            prediction_doc[str(condition)] = array_key
        cell_docs.append({
            "cell": str(cell_name),
            "array_prefix": cell_token,
            "n_rows": row_count,
            "n_images": int(np.unique(identity["image_id"]).size),
            "eval_splits": sorted(np.unique(identity["eval_split"]).tolist()),
            "conditions": prediction_doc,
            "sentence_id_sha256": hashlib.sha256(np.ascontiguousarray(identity["sentence_id"]).tobytes()).hexdigest(),
        })

    for parent in (archive_path.parent, sidecar_path.parent):
        parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"recovery output already exists and is immutable: {archive_path} or {sidecar_path}")
    tmp = archive_path.with_name(f".{archive_path.stem}.{os.getpid()}.tmp.npz")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    _replace_atomic(tmp, archive_path)
    archive_hash = sha256(archive_path)
    sidecar = {
        "schema": ARCHIVE_SCHEMA,
        "job_id": str(job_id),
        "scope": str(scope),
        "cohort": str(cohort),
        "seed": int(seed),
        "archive_path": archive_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "archive_sha256": archive_hash,
        "archive_size_bytes": int(archive_path.stat().st_size),
        "row_identity": list(IDENTITY_FIELDS),
        "canonical_id_field": "sentence_id",
        "resample_cluster_field": "image_id",
        "probability_dtype": "float64",
        "cells": cell_docs,
        **dict(metadata),
    }
    atomic_json(sidecar_path, sidecar)
    sidecar["sidecar_path"] = sidecar_path.resolve().relative_to(ROOT.resolve()).as_posix()
    sidecar["sidecar_sha256"] = sha256(sidecar_path)
    return sidecar


def load_prediction_archive(archive_path: Path, sidecar_path: Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one archive and verify its sidecar's content hash and row shapes."""
    if sidecar_path is None:
        sidecar_path = archive_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema") != ARCHIVE_SCHEMA:
        raise ValueError(f"unsupported recovery archive schema: {sidecar.get('schema')}")
    if sha256(archive_path) != sidecar.get("archive_sha256"):
        raise RuntimeError(f"prediction archive hash mismatch: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    for cell in sidecar["cells"]:
        prefix = cell["array_prefix"]
        ids = arrays[f"{prefix}__sentence_id"].reshape(-1)
        if ids.size != int(cell["n_rows"]):
            raise ValueError(f"{sidecar['job_id']}/{cell['cell']}: archived row count mismatch")
        for condition, key in cell["conditions"].items():
            values = arrays[key].reshape(-1)
            if values.size != ids.size or not np.all(np.isfinite(values)):
                raise ValueError(f"{sidecar['job_id']}/{cell['cell']}/{condition}: invalid archived predictions")
    return arrays, sidecar


def get_cell(arrays: Mapping[str, np.ndarray], sidecar: Mapping[str, Any], cell_name: str) -> dict[str, Any]:
    """Return one cell in a convenient representation for statistics adapters."""
    cell = next((item for item in sidecar["cells"] if item["cell"] == cell_name), None)
    if cell is None:
        raise KeyError(f"{sidecar.get('job_id')}: no cell {cell_name!r}")
    prefix = cell["array_prefix"]
    out: dict[str, Any] = {field: arrays[f"{prefix}__{field}"] for field in IDENTITY_FIELDS}
    out["predictions"] = {name: arrays[key] for name, key in cell["conditions"].items()}
    return out


def source_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map a source record to the compact fields consumed by repair statistics."""
    return {key: record[key] for key in ("path", "size_bytes", "sha256")}

