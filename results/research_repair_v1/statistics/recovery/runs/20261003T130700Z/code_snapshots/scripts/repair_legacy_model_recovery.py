#!/usr/bin/env python
"""Recover legacy gate predictions into repair-owned, row-level archives.

No original result is written. Frozen model artifacts are reused where
available; the confirmatory V2-M3.1 curriculum expert follows its original
train/tune fit path because the fitted per-row predictions were not archived.
Formal bootstrap work belongs to ``scripts/repair_statistics.py`` and is never
run here.

Examples
--------
    python -B -u scripts/repair_legacy_model_recovery.py --scope g4 --g4-device cuda
    python -B -u scripts/repair_legacy_model_recovery.py --scope m25-m3b0
    python -B -u scripts/repair_legacy_model_recovery.py --scope m3-conf
"""
from __future__ import annotations

import os

# Constrain numerical libraries before NumPy/Torch/scikit-learn import.
for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccg.repairs.legacy_recovery import (
    ARCHIVE_SCHEMA,
    RECOVERY_ROOT,
    atomic_json,
    load_baseline_manifest,
    load_prediction_archive,
    sha256,
    source_record,
    write_prediction_archive,
)

OUT_ROOT = Path("results/v2_local_competition")
V2G_ROOT = Path("results/v2_backbone_generalization")
M1_DIR = OUT_ROOT / "m1_random_only"
M2_DIR = OUT_ROOT / "m2_curriculum"
PHASEA_DIR = V2G_ROOT / "g4_phaseA"
PHASEB_DIR = V2G_ROOT / "g4_phaseB"
SPLIT_MANIFEST = Path("results/phase05_score_sufficiency/split_manifest.json")
BASELINE_MANIFEST = ROOT / "results/research_repair_v1/input_manifest.json"

_BASELINE_DOC, _BASELINE_INPUTS = load_baseline_manifest(BASELINE_MANIFEST)

RECOVERY_CODE_SOURCES = (
    "scripts/run_v2g_hard.py",
    "scripts/run_v2g_reliability.py",
    "scripts/run_v2m_m1.py",
    "scripts/run_v2m_m2.py",
    "scripts/build_v2m_m2_manifests.py",
    "scripts/run_v2m_m25.py",
    "scripts/run_v2m_m3_b0.py",
    "scripts/run_v2m_m3_conf.py",
    "src/ccg/repairs/legacy_recovery.py",
    "src/ccg/reliability/evaluate.py",
    "src/ccg/reliability/features.py",
    "src/ccg/reliability/models.py",
    "src/ccg/semantic/data.py",
    "src/ccg/semantic/hard.py",
    "src/ccg/semantic/hard_scores.py",
    "src/ccg/v2/semantic_features.py",
    "src/ccg/mixture/mixer.py",
    "results/v2_backbone_generalization/g4_protocol_freeze.json",
    "results/v2_local_competition/m3_mixture/protocol_m3.json",
    "results/v2_local_competition/m3_mixture/amendment_v2m31.json",
)
MUTABLE_RECOVERY_CODE = {
    "scripts/repair_legacy_model_recovery.py",
    "src/ccg/repairs/legacy_recovery.py",
}


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def _load_module(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return source_record(p, frozen_manifest=_BASELINE_INPUTS)


def _source_map(extra: Sequence[Path | str] = ()) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for relative in RECOVERY_CODE_SOURCES:
        record = _sha(relative)
        by_path[record["path"]] = record
    for path in extra:
        record = _sha(path)
        by_path[record["path"]] = record
    return [by_path[key] for key in sorted(by_path)]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_scope_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_json(run_dir / "manifest.json", manifest)


def _snapshot_attempt_code(run_dir: Path, run_id: str) -> dict[str, Any]:
    """Freeze this attempt's repair code as provenance, never as model input."""
    code_paths = [
        ROOT / "scripts" / "repair_legacy_model_recovery.py",
        ROOT / "src" / "ccg" / "repairs" / "legacy_recovery.py",
    ]
    snapshots: list[dict[str, Any]] = []
    for source in code_paths:
        relative = source.resolve().relative_to(ROOT.resolve()).as_posix()
        snapshot = run_dir / "code_snapshots" / Path(relative)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        source_sha, snapshot_sha = sha256(source), sha256(snapshot)
        if source_sha != snapshot_sha:
            raise RuntimeError(f"attempt code snapshot differs from source: {relative}")
        snapshots.append({
            "source_path": relative,
            "source_size_bytes": int(source.stat().st_size),
            "source_sha256": source_sha,
            "snapshot_path": _relative(snapshot),
            "snapshot_size_bytes": int(snapshot.stat().st_size),
            "snapshot_sha256": snapshot_sha,
            "bytes_captured": True,
        })
    provenance = {
        "schema": "ccg.research_repair_v1.attempt_source_provenance.v1",
        "run_id": run_id,
        "classification": "ATTEMPT_CODE_PROVENANCE_NOT_IMMUTABLE_RESEARCH_INPUT",
        "note": "Exact mutable recovery-code bytes are preserved under code_snapshots; this manifest is separate from supplemental_input_manifest.json.",
        "code_snapshots": snapshots,
    }
    path = run_dir / "attempt_source_provenance_manifest.json"
    atomic_json(path, provenance)
    return {"path": _relative(path), "size_bytes": int(path.stat().st_size), "sha256": sha256(path)}


def _new_run(scope: str, run_id: str | None, g4_device: str) -> tuple[str, Path, dict[str, Any]]:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{6,80}", run_id):
        raise ValueError("--run-id must contain 6-80 letters, digits, dot, underscore, or hyphen")
    run_dir = RECOVERY_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    code_provenance = _snapshot_attempt_code(run_dir, run_id)
    log = run_dir / "recovery.log"
    sources = _source_map([Path(__file__)])
    manifest = {
        "schema": "ccg.research_repair_v1.legacy_recovery_run.v1",
        "archive_schema": ARCHIVE_SCHEMA,
        "run_id": run_id,
        "scope": scope,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "python": sys.executable,
        "pid": os.getpid(),
        "blas_thread_limit": 2,
        "g4_device": g4_device,
        "formal_bootstrap_replicates": 0,
        "formal_bootstrap_ran": False,
        "source_files": sources,
        "attempt_source_provenance": code_provenance,
        "baseline_manifest": _sha(BASELINE_MANIFEST),
        "log_path": _relative(log),
        "jobs": [],
        "failures": [],
    }
    _write_scope_manifest(run_dir, manifest)
    atomic_json(RECOVERY_ROOT / "latest.json", {
        "schema": "ccg.research_repair_v1.legacy_recovery_latest.v1",
        "run_id": run_id,
        "manifest_path": _relative(run_dir / "manifest.json"),
    })
    return run_id, run_dir, manifest


def _archive_one(
    run_dir: Path,
    *,
    job_id: str,
    scope: str,
    cohort: str,
    seed: int,
    cells: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
    manifest: dict[str, Any],
) -> None:
    archive_dir = run_dir / "archives" / job_id / f"seed_{int(seed)}"
    npz_path = archive_dir / "predictions.npz"
    json_path = archive_dir / "predictions.json"
    extra_sources = list(metadata.get("source_paths", []))
    records = _source_map(extra_sources + [Path(__file__)])
    sidecar = write_prediction_archive(
        npz_path, json_path, job_id=job_id, scope=scope, cohort=cohort, seed=seed,
        cells=cells,
        metadata={
            **{key: value for key, value in metadata.items() if key != "source_paths"},
            "source_files": records,
            "recovery_run_id": manifest["run_id"],
            "formal_bootstrap_ran": False,
        },
    )
    # Snapshot recovered arrays and the provenance JSON into the shared supplemental
    # manifest consumed by the statistics runner. Original files stay untouched.
    archive_record = {
        "path": _relative(npz_path), "size_bytes": int(npz_path.stat().st_size),
        "sha256": sha256(npz_path),
    }
    sidecar_record = {
        "path": _relative(json_path), "size_bytes": int(json_path.stat().st_size),
        "sha256": sha256(json_path),
    }
    sidecar["sidecar_sha256"] = sidecar_record["sha256"]
    manifest["jobs"].append({
        "job_id": job_id, "scope": scope, "cohort": cohort, "seed": int(seed),
        "status": "COMPLETE", "archive": archive_record, "sidecar": sidecar_record,
        "anchors": metadata.get("anchor_checks", []),
        "model_fit_provenance": metadata.get("model_fit_provenance", {}),
    })
    _write_scope_manifest(run_dir, manifest)
    supplemental_path = run_dir / "supplemental_input_manifest.json"
    supplemental: dict[str, Any] = {
        "schema": "ccg.research_repair_v1.supplemental_input_manifest.v1",
        "run_id": manifest["run_id"],
        "purpose": "immutable research data, model, manifest, and recovered prediction inputs",
        "attempt_code_provenance_manifest": manifest.get("attempt_source_provenance", {}).get("path"),
        "inputs": [],
    }
    old = RECOVERY_ROOT.rglob("supplemental_input_manifest.json")
    existing: dict[str, dict[str, Any]] = {}
    for candidate in old:
        try:
            doc = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in doc.get("inputs", []):
            existing[str(item["path"])] = {key: item[key] for key in ("path", "size_bytes", "sha256")}
    # The two mutable B-owned recovery modules are captured byte-for-byte in
    # code_snapshots and listed in attempt_source_provenance_manifest.json.
    # They are execution provenance, not immutable model/data inputs consumed
    # by the statistics runner, so do not add them to the cross-attempt input
    # namespace where a later legitimate source edit would look like conflict.
    new_items = {
        item["path"]: {key: item[key] for key in ("path", "size_bytes", "sha256")}
        for item in records if item["path"] not in MUTABLE_RECOVERY_CODE
    }
    new_items[archive_record["path"]] = archive_record
    new_items[sidecar_record["path"]] = sidecar_record
    for path, item in new_items.items():
        if path in existing and existing[path] != item:
            raise RuntimeError(f"supplemental source snapshot conflict for {path}")
        existing[path] = item
    supplemental["inputs"] = [existing[key] for key in sorted(existing)]
    atomic_json(supplemental_path, supplemental)


def _write_failure(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    scope_id: str,
    attempt_id: str,
    attempted_route: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path | str],
    original_tolerance: Mapping[str, Any],
    error: BaseException,
    log: RunLog,
) -> None:
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    log(f"FAILURE {scope_id}: {type(error).__name__}: {error}")
    for line in trace.rstrip().splitlines():
        log(line)
    source_records = _source_map(list(source_paths) + [Path(__file__)])
    scope_token = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_id)
    log_path = run_dir / "failures" / f"{scope_token}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(trace + "\n", encoding="utf-8")
    observed = None
    match = re.search(r"(?:delta|max\|delta\||error|diff)\s*[=:]?\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)", str(error))
    if match:
        observed = float(match.group(1))
    record = {
        "schema": "research-repair-v1-recovery-failure-v1",
        "scope_id": scope_id,
        "attempt_id": attempt_id,
        "status": "UNVERIFIABLE",
        "attempted_route": [dict(item) for item in attempted_route],
        "source_sha256": {item["path"]: item["sha256"] for item in source_records},
        "original_tolerance": dict(original_tolerance),
        "observed_error": observed,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "failure_reason": str(error),
        "log_path": _relative(log_path),
        "log_sha256": sha256(log_path),
    }
    path = run_dir / "failures" / f"{scope_token}.json"
    atomic_json(path, record)
    failure_record = {
        "scope_id": scope_id,
        "attempt_id": attempt_id,
        "status": "UNVERIFIABLE",
        "record_path": _relative(path),
        "record_sha256": sha256(path),
        "log_path": record["log_path"],
        "log_sha256": record["log_sha256"],
    }
    manifest["failures"].append(failure_record)
    _write_scope_manifest(run_dir, manifest)


def _neutral_bootstrap_rows(reval_module: Any, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Return exact point rows with explicit missing-CI markers; never resample.

    The historical runners inspect the regular model-pair fields even when the
    returned CIs are not used by this prediction-recovery task. Keep that row
    contract, but mark these as internal, nonformal rows so they cannot be
    mistaken for scientific bootstrap evidence.
    """
    if len(args) < 5:
        raise TypeError("expected confidence_a, correct_a, confidence_b, correct_b, clusters")
    conf_a, correct_a = np.asarray(args[0]).reshape(-1), np.asarray(args[1]).reshape(-1)
    conf_b, correct_b = np.asarray(args[2]).reshape(-1), np.asarray(args[3]).reshape(-1)
    clusters = np.asarray(args[4]).reshape(-1)
    if not (conf_a.size == correct_a.size == conf_b.size == correct_b.size == clusters.size):
        raise ValueError("paired point-row inputs have different lengths")
    if conf_a.size == 0:
        raise ValueError("paired point-row inputs cannot be empty")
    split = str(kwargs["eval_split"])
    k_value = int(kwargs["K"])
    model_a, model_b = str(kwargs["model_a"]), str(kwargs["model_b"])
    metrics = tuple(kwargs.get("metrics", reval_module.PRIMARY_METRICS))
    a = reval_module.SampleStats.from_conf_correct(conf_a, correct_a)
    b = reval_module.SampleStats.from_conf_correct(conf_b, correct_b)
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        metric_fn = reval_module._metric_fn(metric)
        point_a, point_b = float(metric_fn(a)), float(metric_fn(b))
        rows.append({
            "eval_split": split,
            "K": k_value,
            "model_a": model_a,
            "model_b": model_b,
            "metric": str(metric),
            "diff": point_a - point_b,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "mean_a": point_a,
            "mean_b": point_b,
            "n": int(conf_a.size),
            "n_clusters": int(np.unique(clusters).size),
            "resample_unit": "image",
            "n_replicates": 0,
            "ci_level": float(kwargs.get("ci", 0.95)),
            "bootstrap_status": "NOT_COMPUTED",
            "formal_bootstrap": False,
            "recovery_internal_stub": True,
        })
    return rows


def _neutral_macro_bootstrap(reval_module: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Macro point estimate only; no bootstrap draw or pseudo-interval."""
    if len(args) < 3:
        raise TypeError("expected metric_fn, severity_pairs, cluster_ids")
    metric_fn, severity_pairs = args[0], args[1]
    clusters = np.asarray(args[2]).reshape(-1)
    if not severity_pairs:
        raise ValueError("severity_pairs cannot be empty")
    pairs = []
    for entry in severity_pairs:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise TypeError("severity pair must be (model_a, model_b) prediction tuples")
        a_entry, b_entry = entry
        a = a_entry if isinstance(a_entry, reval_module.SampleStats) else reval_module.SampleStats.from_conf_correct(*a_entry)
        b = b_entry if isinstance(b_entry, reval_module.SampleStats) else reval_module.SampleStats.from_conf_correct(*b_entry)
        if len(a) != len(b):
            raise ValueError("macro paired point rows have different lengths")
        pairs.append((a, b))
    n = len(pairs[0][0])
    if clusters.size != n or any(len(a) != n for a, _ in pairs):
        raise ValueError("macro severities and cluster IDs must share one row universe")
    means_a = [float(metric_fn(a)) for a, _ in pairs]
    means_b = [float(metric_fn(b)) for _, b in pairs]
    return {
        "diff": float(np.mean(np.asarray(means_a) - np.asarray(means_b))),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "mean_a": float(np.mean(means_a)),
        "mean_b": float(np.mean(means_b)),
        "n": int(n),
        "n_clusters": int(np.unique(clusters).size),
        "n_replicates": 0,
        "ci_level": float(kwargs.get("ci", 0.95)),
        "bootstrap_status": "NOT_COMPUTED",
        "formal_bootstrap": False,
        "recovery_internal_stub": True,
    }


class PointCapture:
    """Capture prediction rows from an original runner without running its CI code."""
    def __init__(self, reval_module: Any, method: str, levels: Sequence[str], models: Sequence[str]):
        self.reval = reval_module
        self.method = method
        self.levels = tuple(levels)
        self.models = tuple(models)
        self.original = reval_module.point_metric_row
        self.calls = 0
        self.captured: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def __enter__(self) -> "PointCapture":
        self.reval.point_metric_row = self._wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.reval.point_metric_row = self.original

    def _wrapped(self, confidence: np.ndarray, correct: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        result = self.original(confidence, correct, *args, **kwargs)
        call = self.calls
        self.calls += 1
        if self.method == "m3-b0":
            # The K5 random anchor is a direct max-delta check, not a
            # point_metric_row call. Each test severity then contributes 5
            # model rows, 2 expert AUROCs, and 5 regret AUROCs.
            relative = call
            stride = 12
            if relative >= 0:
                level_index, position = divmod(relative, stride)
                if level_index < len(self.levels) and 0 <= position < len(self.models):
                    self._store(self.levels[level_index], self.models[position], confidence, correct)
        elif self.method == "m3-conf":
            # Original order: one Phase-A K5 anchor; per level, one dose-anchor
            # AUROC, five model rows, one additional curriculum AUROC, then five
            # regret AUROCs. The dose-anchor call is already the random-expert AUROC.
            relative = call - 1
            stride = 12
            if relative >= 0:
                level_index, position = divmod(relative, stride)
                model_index = position - 1
                if level_index < len(self.levels) and 0 <= model_index < len(self.models):
                    self._store(self.levels[level_index], self.models[model_index], confidence, correct)
        else:
            raise ValueError(self.method)
        return result

    def _store(self, level: str, model: str, confidence: np.ndarray, correct: np.ndarray) -> None:
        key = (level, model)
        if key in self.captured:
            raise AssertionError(f"duplicate point-row capture {key}")
        self.captured[key] = (
            np.asarray(confidence, dtype=np.float64).reshape(-1).copy(),
            np.asarray(correct, dtype=bool).reshape(-1).copy(),
        )

    def assert_complete(self, *, expected_calls: int) -> None:
        expected = {(level, model) for level in self.levels for model in self.models}
        if set(self.captured) != expected:
            missing = sorted(expected - set(self.captured))
            extra = sorted(set(self.captured) - expected)
            raise AssertionError(f"captured point rows differ: missing={missing}, extra={extra}")
        if self.calls != expected_calls:
            raise AssertionError(f"original point-metric call order changed: {self.calls} != {expected_calls}")


@contextmanager
def suppress_bootstrap(reval_module: Any, runner_module: Any):
    """Keep original point/inference code, explicitly suppress every CI draw."""
    old_pair = reval_module.model_vs_model_bootstrap_row
    reval_module.model_vs_model_bootstrap_row = (
        lambda *args, **kwargs: _neutral_bootstrap_rows(reval_module, *args, **kwargs)
    )
    old_macro = getattr(runner_module, "macro_paired_cluster_bootstrap", None)
    if old_macro is not None:
        runner_module.macro_paired_cluster_bootstrap = (
            lambda *args, **kwargs: _neutral_macro_bootstrap(reval_module, *args, **kwargs)
        )
    try:
        yield
    finally:
        reval_module.model_vs_model_bootstrap_row = old_pair
        if old_macro is not None:
            runner_module.macro_paired_cluster_bootstrap = old_macro


def _cell_from_capture(
    captured: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    cell_name: str,
    models: Sequence[str],
    sentence_id: np.ndarray,
    image_id: np.ndarray,
    ref_id: np.ndarray,
    eval_split: np.ndarray,
) -> dict[str, Any]:
    labels: np.ndarray | None = None
    preds: dict[str, np.ndarray] = {}
    for model in models:
        conf, correct = captured[(cell_name, model)]
        if labels is None:
            labels = correct
        elif not np.array_equal(labels, correct):
            raise AssertionError(f"{cell_name}: model conditions do not share correctness rows")
        preds[model] = conf
    if labels is None or labels.size != sentence_id.size:
        raise AssertionError(f"{cell_name}: prediction/identity row counts differ")
    return {
        "sentence_id": sentence_id,
        "image_id": image_id,
        "ref_id": ref_id,
        "eval_split": eval_split,
        "correct": labels,
        "predictions": preds,
    }


def _validate_row_alignment(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    shared: bool,
) -> None:
    for cell_name, cell in cells.items():
        n = np.asarray(cell["sentence_id"]).size
        for field in ("image_id", "ref_id", "eval_split", "correct"):
            if np.asarray(cell[field]).size != n:
                raise ValueError(f"{cell_name}/{field}: row count mismatch")
        for model, confidence in cell["predictions"].items():
            if np.asarray(confidence).size != n:
                raise ValueError(f"{cell_name}/{model}: prediction row count mismatch")
    if shared and cells:
        first_name = next(iter(cells))
        reference = cells[first_name]
        for cell_name, cell in list(cells.items())[1:]:
            # Grounding correctness can change as candidate composition changes
            # across severity; only row identity must stay fixed across cells.
            for field in ("sentence_id", "image_id", "ref_id", "eval_split"):
                if not np.array_equal(np.asarray(reference[field]), np.asarray(cell[field])):
                    raise AssertionError(f"shared-row cells disagree on {field}: {first_name} vs {cell_name}")


def _assert_point_rows(
    captured: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    point_rows: Sequence[Mapping[str, Any]],
    levels: Sequence[str],
    models: Sequence[str],
    original_point_metric: Callable[..., Any],
    *,
    atol: float = 1e-12,
) -> None:
    table = {(str(row["level"]), str(row["model"])): row for row in point_rows}
    for level in levels:
        for model in models:
            confidence, correct = captured[(level, model)]
            recomputed = original_point_metric(confidence, correct, probability=confidence)
            stored = table[(level, model)]
            for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
                a, b = float(recomputed[metric]), float(stored[metric])
                if not np.isfinite(a) or not np.isfinite(b) or abs(a - b) > atol:
                    raise AssertionError(
                        f"capture path mismatch {level}/{model}/{metric}: {a!r} vs {b!r} (atol={atol})"
                    )


def _record_failure(
    run_dir: Path, manifest: dict[str, Any], *, scope_id: str, route: Sequence[Mapping[str, Any]],
    source_paths: Sequence[Path | str], tolerance: Mapping[str, Any], error: BaseException,
    log: RunLog,
) -> None:
    from ccg.repairs.legacy_recovery import atomic_json

    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    log(f"{scope_id} FAILED: {type(error).__name__}: {error}")
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", scope_id)
    log_path = run_dir / "failures" / f"{token}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(trace + "\n", encoding="utf-8")
    records = _source_map(list(source_paths) + [Path(__file__)])
    observed = None
    match = re.search(r"(?:delta|max\|delta\||error|diff)\s*[=:]?\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?\d+)?)", str(error))
    if match:
        observed = float(match.group(1))
    runner_defect = (
        isinstance(error, AttributeError) and "PHASE_A_ROOT" in str(error)
    ) or (
        isinstance(error, KeyError) and str(error).strip("'\"") in {"model_a", "model_b"}
    ) or (
        isinstance(error, AssertionError) and "original point-metric call order changed" in str(error)
    ) or (
        isinstance(error, RuntimeError) and "supplemental source snapshot conflict" in str(error)
    )
    failure_status = "RUNNER_DEFECT_RETRY_REQUIRED" if runner_defect else "UNVERIFIABLE"
    rec = {
        "schema": "research-repair-v1-recovery-failure-v1",
        "scope_id": scope_id,
        "attempt_id": manifest["run_id"],
        "status": failure_status,
        "attempted_route": [dict(x) for x in route],
        "source_sha256": {r["path"]: r["sha256"] for r in records},
        "original_tolerance": dict(tolerance),
        "observed_error": observed,
        "error_type": type(error).__name__,
        "failure_reason": str(error),
        "classification_note": (
            "Recovery wrapper defect; this is not a legacy model recovery failure."
            if runner_defect else "A deterministic recovery route failed its original anchor or source validation."
        ),
        "log_path": _relative(log_path),
        "log_sha256": sha256(log_path),
    }
    failure_path = run_dir / "failures" / f"{token}.json"
    atomic_json(failure_path, rec)
    manifest["failures"].append({
        "scope_id": scope_id, "status": failure_status,
        "record_path": _relative(failure_path), "record_sha256": sha256(failure_path),
        "log_path": rec["log_path"], "log_sha256": rec["log_sha256"],
    })
    _write_scope_manifest(run_dir, manifest)


def _finish_run(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    statuses = {item.get("status") for item in manifest["failures"]}
    if not statuses:
        manifest["status"] = "COMPLETE"
    elif statuses == {"RUNNER_DEFECT_RETRY_REQUIRED"}:
        manifest["status"] = "RUNNER_DEFECT_RETRY_REQUIRED"
    else:
        manifest["status"] = "PARTIAL_UNVERIFIABLE" if manifest["jobs"] else "FAILED_UNVERIFIABLE"
    _write_scope_manifest(run_dir, manifest)


def _recover_g4(run_dir: Path, manifest: dict[str, Any], log: RunLog, device: str) -> None:
    import torch
    from ccg.models.b3_data import B3Corpus
    from ccg.semantic import data as sdata, hard as shard, hard_scores as hscores

    g4 = _load_module("repair_v2g_hard", "scripts/run_v2g_hard.py")
    freeze = g4._load_freeze()
    cohort = shard.load_hard_cohort(
        features_dir=g4.PHASE05_DIR, manifests_root=g4.MANIFESTS, log=log
    )
    selected = {"base_r5", "hard5", "expb_m0", "expb_m2", "expb_m4", "expb_m8"}
    args = argparse.Namespace(stop_atol=1e-4, batch_size=64)
    for tag, bb_dir in g4.BACKBONES:
        paths = g4._backbone_paths(freeze, bb_dir)
        corpus = B3Corpus(
            paths["features_root"], g4.MANIFESTS, g4.REFS, g4.BANK,
            image_sizes_path=g4.IMAGE_SIZES, ks=(5, 10), regime="random", preload=True,
        )
        try:
            for seed in g4.SEEDS:
                sid = f"v2g_g4_{tag}_seed_{seed}"
                source_paths = [
                    g4.FREEZE_PATH,
                    g4.A_OUT / tag / f"seed_{seed}" / "models.pkl",
                    g4.A_OUT / tag / f"seed_{seed}" / "model_manifest.json",
                    PHASEB_DIR / tag / "dose_response.csv",
                ]
                try:
                    phase_a = {f"b3_seed{seed}": g4._load_phase_a(tag, seed)}
                    scorers = hscores.load_frozen_scorers(
                        b3_root=paths["b3_root"], seeds=(seed,), device=device
                    )
                    if device.startswith("cuda") and not torch.cuda.is_available():
                        raise RuntimeError("requested G4 CUDA recovery but CUDA is unavailable")
                    stop = g4._stop_check(
                        cohort, corpus, scorers, paths, [f"b3_seed{seed}"], args, log
                    )
                    cell_data: dict[str, Any] = {}
                    text_cache: dict[int, np.ndarray] = {}
                    for variant, regime, k in g4.VARIANTS:
                        if variant not in selected:
                            continue
                        cohort_mask = next(s.cohort for s in g4.CELL_SPECS if s.source == variant)
                        rows = g4._variant_rows(cohort, cohort_mask)
                        level = g4.LEVEL_OF_VARIANT.get(variant)
                        samples = g4._variant_samples(cohort, regime, level, k, rows)
                        batch = hscores.materialise_examples(corpus, samples, k, text_cache=text_cache)
                        scores = hscores.score_examples(
                            scorers[f"b3_seed{seed}"], batch, batch_size=args.batch_size
                        )
                        for cell_spec in g4.CELLS_BY_SOURCE[variant]:
                            if cell_spec.cell not in {"rand5", "hard5", "expb_m0", "expb_m2", "expb_m4", "expb_m8"}:
                                continue
                            local = g4._cell_local(cohort, rows, cell_spec)
                            cell_data[cell_spec.cell] = g4._build_cell(
                                cell_spec, batch, scores, local, cohort, rows, phase_a[f"b3_seed{seed}"]
                            )
                        del batch, scores
                    cells: dict[str, dict[str, Any]] = {}
                    for cell_name, bundle in cell_data.items():
                        sentence_id = np.asarray(bundle.sentence_id, dtype=np.int64)
                        pos = np.searchsorted(cohort.sentence_id, sentence_id)
                        if pos.size != sentence_id.size or not np.array_equal(cohort.sentence_id[pos], sentence_id):
                            raise AssertionError(f"{tag}/{seed}/{cell_name}: row IDs do not map to canonical hard cohort")
                        cells[cell_name] = {
                            "sentence_id": sentence_id,
                            "image_id": np.asarray(bundle.image_id, dtype=np.int64),
                            "ref_id": np.asarray(cohort.ref_id[pos], dtype=np.int64),
                            "eval_split": np.asarray(cohort.eval_split[pos]),
                            "correct": np.asarray(bundle.correct, dtype=bool),
                            "predictions": {name: np.asarray(values, dtype=np.float64)
                                            for name, values in bundle.conf.items()},
                        }
                    _validate_row_alignment(cells, shared=False)
                    anchor_checks = [
                        {"anchor": "G4_random_raw_score_STOP", "cell": key,
                         "max_abs_delta": value, "tolerance": 1e-4, "status": "PASS"}
                        for key, value in stop.items() if key.endswith(f"b3_seed{seed}")
                    ]
                    if len(anchor_checks) != 2:
                        raise AssertionError(f"{tag}/{seed}: expected K5 and K10 STOP anchors, found {len(anchor_checks)}")
                    _archive_one(
                        run_dir, job_id=f"v2g_g4_{tag}", scope="V2G_G4", cohort="pooled_test",
                        seed=seed, cells=cells,
                        metadata={
                            "backbone": tag, "backbone_dir": bb_dir,
                            "cell_contract": "rand5/hard5 use same4 rows; m0/m2/m4/m8 use same8 rows; K10 cells retain original Phase-B composition",
                            "anchor_checks": anchor_checks,
                            "model_fit_provenance": {"route": "FROZEN_PHASE_A_CHECKPOINT", "fit_performed": False,
                                "model_artifact": _relative(source_paths[1]),
                                "phase_a_manifest": _relative(source_paths[2])},
                            "source_paths": source_paths + [g4.A_OUT / "phaseA_summary.csv"],
                        }, manifest=manifest,
                    )
                    log(f"{sid} COMPLETE (device={device}, cells={len(cells)}, n={sum(len(v['correct']) for v in cells.values())})")
                except BaseException as error:
                    _record_failure(
                        run_dir, manifest, scope_id=sid,
                        route=[{"step": "load Phase-A models.pkl", "applicable": True},
                               {"step": "CPU/GPU frozen-scoring forward", "applicable": True},
                               {"step": "original Phase-A fit route", "applicable": False,
                                "reason": "the frozen model checkpoint is present and must be applied verbatim; refitting would change the historical model"}],
                        source_paths=source_paths,
                        tolerance={"raw_score_stop_atol": 1e-4}, error=error, log=log,
                    )
                finally:
                    del scorers
        finally:
            corpus.close()
            del corpus


def _recover_m25_and_b0(
    run_dir: Path, manifest: dict[str, Any], log: RunLog, *, archive_m25: bool = True
) -> None:
    from ccg.models.b3_data import B3Corpus
    from ccg.semantic import data as sdata, hard as shard, hard_scores as hscores

    b0 = _load_module("repair_v2m_m3_b0", "scripts/run_v2m_m3_b0.py")
    runner = b0._load_module("run_v2m_m2", "scripts/run_v2m_m2.py")
    m1 = b0._load_module("run_v2m_m1", "scripts/run_v2m_m1.py")
    build = b0._load_module("build_v2m_m2_manifests", "scripts/build_v2m_m2_manifests.py")
    split = sdata.load_split_from_manifest(runner.SPLIT_MANIFEST)
    cohort_hc = shard.load_hard_cohort(
        features_dir=sdata.PHASE05_FEATURES_DIR, manifests_root=runner.MANIFESTS, log=log
    )
    rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    val_cohort = build.load_val_cohort()
    m3_sources = [
        Path("scripts/run_v2m_m3_b0.py"), Path("scripts/run_v2m_m2.py"), Path("scripts/run_v2m_m1.py"),
        Path("scripts/build_v2m_m2_manifests.py"), Path("results/v2_local_competition/m3_mixture/b0/point_metrics.csv"),
    ]
    for seed in (1, 2, 3):
        scope_id = f"v2m_m3_b0_seed_{seed}"
        seed_sources = m3_sources + [
            M1_DIR / f"seed_{seed}" / "model_manifest.json",
            M1_DIR / f"seed_{seed}" / "confidences.npz",
            M2_DIR / f"seed_{seed}" / f"model_manifest_seed{seed}.json",
            M2_DIR / f"seed_{seed}" / "confidences.npz",
            M2_DIR / "point_metrics.csv",
        ]
        try:
            scorers = hscores.load_frozen_scorers(seeds=(seed,), device="cpu")
            corpus = B3Corpus(
                runner.FEATURES_ROOT, runner.MANIFESTS, runner.REFS, runner.BANK,
                image_sizes_path=runner.IMAGE_SIZES, ks=(5,), regime="random", preload=True,
            )
            try:
                levels = tuple(f"m{m}" for m in b0.LEVELS)
                models = tuple(b0.MODELS)
                capture = PointCapture(b0.reval, "m3-b0", levels, models)
                run_args = argparse.Namespace(bootstrap_replicates=0, bootstrap_seed=0, ci=0.95)
                with suppress_bootstrap(b0.reval, b0), capture:
                    result = b0._run_seed(
                        seed, runner=runner, m1_module=m1, corpus=corpus,
                        val_cohort=val_cohort, cohort_hc=cohort_hc, rows_same8=rows_same8,
                        split=split, scorers=scorers, args=run_args, log=log,
                    )
                capture.assert_complete(expected_calls=len(levels) * 12)
                _assert_point_rows(
                    capture.captured, result["point_rows"], levels, models,
                    capture.original, atol=1e-12,
                )
                common = {
                    "sentence_id": np.asarray(cohort_hc.sentence_id[rows_same8], dtype=np.int64),
                    "image_id": np.asarray(cohort_hc.image_id[rows_same8], dtype=np.int64),
                    "ref_id": np.asarray(cohort_hc.ref_id[rows_same8], dtype=np.int64),
                    "eval_split": np.asarray(cohort_hc.eval_split[rows_same8]),
                }
                cells = {
                    level: _cell_from_capture(capture.captured, level, models, **common)
                    for level in levels
                }
                _validate_row_alignment(cells, shared=True)
                # The original runner performed the K5 vector comparison and
                # returned its observed max error; preserve that evidence instead
                # of substituting the stored vector for a recomputed prediction.
                k5_err = float(result["repro_random_conf"])
                if not np.isfinite(k5_err) or k5_err > float(b0.REPRO_TOL):
                    raise AssertionError(f"seed{seed}: M1 Random-K5 anchor error={k5_err:.3e} > {b0.REPRO_TOL:g}")
                m2_checks = _verify_m2_confidence_archive(seed, cells, b0.REPRO_TOL)
                anchors = [
                    {"anchor": "M1_Random-K5_E1b_confidences", "max_abs_delta": k5_err,
                     "tolerance": b0.REPRO_TOL, "status": "PASS"},
                    *m2_checks,
                ]
                fit = result["fit"]
                if archive_m25:
                    # M2.5 used the same B0 M1/M2 experts, row construction, normalizers,
                    # and deterministic confidence formula. Export this independent
                    # adapter-ready view before checking the developmental B0 mixer
                    # table, so an M3-B0-only mismatch cannot hide valid M25 evidence.
                    m25_cells = {
                        level: {
                            **{field: cells[level][field] for field in ("sentence_id", "image_id", "ref_id", "eval_split", "correct")},
                            "predictions": {
                                "E1b_random": cells[level]["predictions"]["E_random"],
                                "E1b_curriculum": cells[level]["predictions"]["E_curriculum"],
                            },
                        }
                        for level in levels
                    }
                    m2_point_rows = list(csv.DictReader(open(M2_DIR / "point_metrics.csv", encoding="utf-8")))
                    m25_anchor_checks = _check_existing_point_table(
                        m25_cells, m2_point_rows, seed=seed, level_prefix="m", model_field="model",
                        runner_models={"E1b_random": "E1b", "E1b_curriculum": "E1b"},
                        point_metric=capture.original, tol=1e-9,
                        split_name="M25_curriculum_vs_M2_point_metrics",
                        only_condition="E1b_curriculum",
                    )
                    _archive_one(
                        run_dir, job_id="v2m_m25", scope="V2M_M25_SPECIALIST_AUDIT",
                        cohort="same8_m0_m2_m4_m8", seed=seed, cells=m25_cells,
                        metadata={
                            "backbone": "b0", "k": 10, "label": "DIAGNOSTIC; no success gate",
                            "anchor_checks": [*anchors[:len(m2_checks)+1], *m25_anchor_checks],
                            "model_fit_provenance": {"E1b_random": "M1 frozen coefficient/normalization",
                                "E1b_curriculum": "M2 frozen coefficient/normalization",
                                "classifier_refit": False,
                                "recovery_source": "same deterministic B0 M1/M2 predictions archived for M3-B0"},
                            "source_paths": seed_sources + [Path("results/v2_local_competition/m25_specialist_audit/metadata.json")],
                        }, manifest=manifest,
                    )
                    log(f"v2m_m25_seed_{seed} COMPLETE (point-anchor checks={len(m25_anchor_checks)})")

                # B0 is a post-hoc developmental result. Keep its committed
                # point-table check strict, but do not let it gate the separate
                # M25 archive, whose frozen M1/M2 anchors are verified above.
                old_b0 = list(csv.DictReader(open(ROOT / "results/v2_local_competition/m3_mixture/b0/point_metrics.csv", encoding="utf-8")))
                b0_anchor_checks = _check_existing_point_table(
                    cells, old_b0, seed=seed, level_prefix="m", model_field="model",
                    runner_models=models, point_metric=capture.original, tol=1e-9,
                    split_name="M3_B0_committed_point_metrics",
                )
                b0_anchors = [*anchors, *b0_anchor_checks]
                _archive_one(
                    run_dir, job_id="v2m_m3_b0", scope="V2M_M3_B0_DEVELOPMENTAL",
                    cohort="same8_m0_m2_m4_m8", seed=seed, cells=cells,
                    metadata={
                        "backbone": "b0", "k": 10, "label": "POST-HOC DEVELOPMENTAL; no confirmatory gate",
                        "anchor_checks": b0_anchors,
                        "model_fit_provenance": {"E_random": "M1 frozen coefficient/normalization",
                            "E_curriculum": "M2 frozen coefficient/normalization",
                            "mixer_fit": "original train-only CompetitionIndex + tune-only StaticMix/AdaptiveMix",
                            "fit_record": fit, "classifier_refit": False},
                        "source_paths": seed_sources + [
                            Path("results/v2_local_competition/m3_mixture/b0/metadata.json"),
                            Path("results/v2_local_competition/m3_mixture/protocol_m3.json"),
                        ],
                    }, manifest=manifest,
                )
                log(f"{scope_id} COMPLETE (anchors={len(b0_anchors)}, cells={len(cells)})")
            finally:
                corpus.close()
        except BaseException as error:
            _record_failure(
                run_dir, manifest, scope_id=scope_id,
                route=[{"step": "load M1/M2 frozen manifests and scalers", "applicable": True},
                       {"step": "run original M3-B0 frozen scorer/feature/inference path", "applicable": True},
                       {"step": "M3-B0 mixer fit", "applicable": True, "fit_data": "reliability_train/tune m={0,2,4} only"},
                       {"step": "M25 original expert fit", "applicable": False,
                        "reason": "M25 specifies frozen M1/M2 experts; re-fitting would violate its read-only design"}],
                source_paths=seed_sources,
                tolerance={
                    "M1_M2_confidence": 1e-9,
                    "M2_point_metrics": 1e-9,
                    "M3_B0_committed_point_metrics": 1e-9,
                },
                error=error, log=log,
            )
        finally:
            del scorers


def _max_abs_or_inf(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _verify_m2_confidence_archive(seed: int, cells: Mapping[str, Mapping[str, Any]], tol: float) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    with np.load(M2_DIR / f"seed_{seed}" / "confidences.npz", allow_pickle=False) as frozen:
        for level, cell in cells.items():
            observed = np.asarray(cell["predictions"]["E_curriculum"], dtype=np.float64)
            expected = np.asarray(frozen[f"{level}__E1b"], dtype=np.float64)
            error = _max_abs_or_inf(observed, expected)
            if error > tol:
                raise AssertionError(f"seed{seed}/{level}: M2 confidence error={error:.3e} > {tol:g}")
            checks.append({"anchor": "M2_curriculum_confidences", "cell": level,
                           "max_abs_delta": error, "tolerance": tol, "status": "PASS"})
    return checks


def _check_existing_point_table(
    cells: Mapping[str, Mapping[str, Any]],
    table_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    level_prefix: str,
    model_field: str,
    runner_models: Mapping[str, str] | Sequence[str],
    point_metric: Callable[..., Any],
    tol: float,
    split_name: str,
    only_condition: str | None = None,
) -> list[dict[str, Any]]:
    expected_by_cell: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in table_rows:
        try:
            cohort_field = row.get("cohort", row.get("level"))
            key = (str(cohort_field), str(row[model_field]), int(row["seed"]))
        except (KeyError, ValueError):
            continue
        expected_by_cell[key] = row
    checks: list[dict[str, Any]] = []
    for level, cell in cells.items():
        if isinstance(runner_models, Mapping):
            conditions = [only_condition] if only_condition is not None else list(runner_models)
            model_map = runner_models
        else:
            conditions = list(runner_models)
            model_map = {name: name for name in conditions}
        for condition in conditions:
            if condition is None or condition not in cell["predictions"]:
                continue
            old_model = model_map[condition]
            # M3 point tables call this column ``level``; M2 calls it ``cohort``.
            key = (level, old_model, int(seed))
            if key not in expected_by_cell:
                raise KeyError(f"{split_name} missing anchor row {key}")
            confidence = np.asarray(cell["predictions"][condition], dtype=np.float64)
            correct = np.asarray(cell["correct"], dtype=bool)
            observed_metrics = point_metric(confidence, correct, probability=confidence)
            old = expected_by_cell[key]
            for metric in ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"):
                if metric not in old:
                    continue
                expected = float(old[metric])
                observed = float(observed_metrics[metric])
                error = abs(observed - expected)
                if not np.isfinite(error) or error > tol:
                    raise AssertionError(
                        f"{split_name} {key}/{metric}: observed={observed:.17g} expected={expected:.17g} "
                        f"abs_error={error:.3e} > {tol:g}"
                    )
                checks.append({"anchor": split_name, "cell": level, "model": condition,
                               "metric": metric, "observed": observed, "expected": expected,
                               "absolute_error": error, "tolerance": tol, "status": "PASS"})
    return checks


def _recover_m3_conf(run_dir: Path, manifest: dict[str, Any], log: RunLog) -> None:
    from ccg.models.b3_data import B3Corpus
    from ccg.semantic import data as sdata, hard as shard, hard_scores as hscores

    conf = _load_module("repair_v2m_m3_conf", "scripts/run_v2m_m3_conf.py")
    m2 = conf._load_module("run_v2m_m2", "scripts/run_v2m_m2.py")
    freeze = _read_json(conf.FREEZE_PATH)
    cohort_hc = shard.load_hard_cohort(
        features_dir=sdata.PHASE05_FEATURES_DIR, manifests_root=conf._abs(conf.MANIFESTS), log=log
    )
    rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    build = conf._load_module("build_v2m_m2_manifests", "scripts/build_v2m_m2_manifests.py")
    val_cohort = build.load_val_cohort()
    levels = tuple(f"m{m}" for m in conf.LEVELS)
    models = tuple(conf.MODELS)
    amendment = _read_json(conf.M3_DIR / "amendment_v2m31.json")
    script_expected = amendment.get("freeze_hashes", {}).get("scripts/run_v2m_m3_conf.py")
    script_record = _sha("scripts/run_v2m_m3_conf.py")
    if script_expected and script_record["sha256"] != script_expected:
        raise RuntimeError(
            f"frozen M3.1 runner hash changed: {script_record['sha256']} != {script_expected}"
        )
    for tag, bb_dir in conf.BACKBONES:
        paths = conf._backbone_paths(freeze, bb_dir)
        for seed in (1, 2, 3):
            scope_id = f"v2m_m3_conf_{tag}_seed_{seed}"
            source_paths = [
                Path("scripts/run_v2m_m3_conf.py"), Path("scripts/run_v2m_m2.py"),
                Path("scripts/build_v2m_m2_manifests.py"), conf.FREEZE_PATH,
                conf.M3_DIR / "amendment_v2m31.json",
                PHASEA_DIR / tag / f"seed_{seed}" / "models.pkl",
                PHASEA_DIR / "phaseA_summary.csv",
                PHASEB_DIR / tag / "dose_response.csv",
                M1_DIR / f"seed_{seed}" / "model_manifest.json",
                M2_DIR / "manifests" / "manifest_freeze.json",
            ]
            try:
                phase_a = {f"b3_seed{seed}": conf._load_phase_a(tag, seed)}
                scorers = hscores.load_frozen_scorers(
                    b3_root=paths["b3_root"], seeds=(seed,), device="cpu"
                )
                corpus = B3Corpus(
                    paths["features_root"], conf._abs(conf.MANIFESTS), conf._abs(conf.REFS), conf._abs(conf.BANK),
                    image_sizes_path=conf._abs(conf.IMAGE_SIZES), ks=(5,), regime="random", preload=True,
                )
                try:
                    capture = PointCapture(conf.reval, "m3-conf", levels, models)
                    run_args = argparse.Namespace(bootstrap_replicates=0, bootstrap_seed=0, ci=0.95)
                    with suppress_bootstrap(conf.reval, conf), capture:
                        result = conf._run_seed(
                            seed, tag=tag, bb_dir=bb_dir, paths=paths, phase_a=phase_a,
                            runner=m2, corpus=corpus, cohort_hc=cohort_hc,
                            rows_same8=rows_same8, val_cohort=val_cohort,
                            scorers=scorers, args=run_args, log=log,
                        )
                    capture.assert_complete(expected_calls=1 + len(levels) * 12)
                    _assert_point_rows(
                        capture.captured, result["point_rows"], levels, models,
                        capture.original, atol=1e-12,
                    )
                    common = {
                        "sentence_id": np.asarray(cohort_hc.sentence_id[rows_same8], dtype=np.int64),
                        "image_id": np.asarray(cohort_hc.image_id[rows_same8], dtype=np.int64),
                        "ref_id": np.asarray(cohort_hc.ref_id[rows_same8], dtype=np.int64),
                        "eval_split": np.asarray(cohort_hc.eval_split[rows_same8]),
                    }
                    cells = {
                        level: _cell_from_capture(capture.captured, level, models, **common)
                        for level in levels
                    }
                    _validate_row_alignment(cells, shared=True)
                    old_rows = list(csv.DictReader(open(conf.CONF_DIR / tag / "point_metrics.csv", encoding="utf-8")))
                    point_checks = _check_existing_point_table(
                        cells, old_rows, seed=seed, level_prefix="m", model_field="model",
                        runner_models=models, point_metric=capture.original, tol=1e-9,
                        split_name="M3_1_committed_point_metrics",
                    )
                    anchors = [
                        {"anchor": "V2G_PhaseA_K5_R2_AUROC", "absolute_error": float(result["anchor_phase_a_max_delta"]),
                         "tolerance": conf.ANCHOR_STORE_TOL, "status": "PASS"},
                        {"anchor": "V2G_PhaseB_dose_R2_AUROC_max", "absolute_error": float(result["anchor_dose_max_delta"]),
                         "tolerance": conf.ANCHOR_RESCORE_TOL, "status": "PASS"},
                        {"anchor": "M2_random_K10_validation_STOP", "absolute_error": float(result["val_stop_max_delta"]),
                         "tolerance": float(m2.STOP_TOL), "status": "PASS"},
                        *point_checks,
                    ]
                    fit = dict(result["fit"])
                    fit["curriculum_protocol"] = {
                        "levels": list(m2.LEVELS_TRAIN), "K": int(m2.K_LEVEL),
                        "C_grid": list(m2.C_GRID), "selection_tie": float(m2.SELECTION_TIE),
                        "train_weighting": "balanced_train_weights, equal total weight per m=0/2/4",
                        "selection": "mean tune AUROC across m=0/2/4; tie favors simpler/earlier C",
                    }
                    _archive_one(
                        run_dir, job_id=f"v2m_m3_conf_{tag}", scope="V2M_M3_1_CONFIRMATORY",
                        cohort="same8_m0_m2_m4_m8", seed=seed, cells=cells,
                        metadata={
                            "backbone": tag, "backbone_dir": bb_dir, "k": 10,
                            "label": "CONFIRMATORY V2-M3.1; original fit/inference path",
                            "anchor_checks": anchors,
                            "model_fit_provenance": {
                                "E_random": "frozen V2-G Phase-A R2 checkpoint",
                                "E_curriculum": "fresh original M2 curriculum Logistic fit",
                                "curriculum_fit_record": fit,
                                "CompetitionIndex": "fitted from reliability_train m0/m2/m4 rows only",
                                "StaticMix/AdaptiveMix": "fitted from reliability_tune m0/m2/m4 only",
                                "m8_used_for_fit_or_selection": False,
                                "bootstrap_ran": False,
                            },
                            "source_paths": source_paths + [
                                conf.PHASE_B_ROOT / tag / "dose_response.csv",
                                conf.CONF_DIR / tag / "point_metrics.csv",
                            ],
                        }, manifest=manifest,
                    )
                    log(f"{scope_id} COMPLETE (C={fit['curriculum_C']:g}, anchors={len(anchors)})")
                finally:
                    corpus.close()
            except BaseException as error:
                _record_failure(
                    run_dir, manifest, scope_id=scope_id,
                    route=[{"step": "load frozen V2-G Phase-A R2 checkpoint", "applicable": True},
                           {"step": "original M3.1 E_curriculum M2 Logistic C-grid fit", "applicable": True,
                            "parameters": {"levels": [0, 2, 4], "K": 10, "C_grid": [0.1, 1, 10], "tie": 0.002}},
                           {"step": "original CDF and mixer tune-fit", "applicable": True},
                           {"step": "image-cluster CI bootstrap", "applicable": False,
                            "reason": "prediction recovery must not run bootstrap; statistics runner owns formal CI"}],
                    source_paths=source_paths,
                    tolerance={"phase_a_k5_store": 1e-9, "dose_rescore": 1e-4, "val_stop": float(m2.STOP_TOL)},
                    error=error, log=log,
                )
            finally:
                if "scorers" in locals():
                    del scorers


def _preflight() -> dict[str, Any]:
    """Cheap, data-free contract checks; never starts model work."""
    if not BASELINE_MANIFEST.is_file():
        raise FileNotFoundError(BASELINE_MANIFEST)
    missing = [path for path in RECOVERY_CODE_SOURCES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing original source files: {missing}")
    _source_map([Path(__file__)])
    g4 = _load_module("repair_preflight_v2g_hard", "scripts/run_v2g_hard.py")
    if not hasattr(g4, "A_OUT") or not hasattr(g4, "_load_phase_a"):
        raise AssertionError("G4 Phase-A checkpoint entry point is missing")
    missing_checkpoints = [
        _relative(g4.A_OUT / tag / f"seed_{seed}" / "models.pkl")
        for tag, _ in g4.BACKBONES for seed in g4.SEEDS
        if not (g4.A_OUT / tag / f"seed_{seed}" / "models.pkl").is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(f"G4 frozen Phase-A checkpoints missing: {missing_checkpoints}")
    conf = _load_module("repair_preflight_v2m_m3_conf", "scripts/run_v2m_m3_conf.py")
    amendment = _read_json(conf.M3_DIR / "amendment_v2m31.json")
    expected_conf_sha = amendment.get("freeze_hashes", {}).get("scripts/run_v2m_m3_conf.py")
    if expected_conf_sha and sha256(ROOT / "scripts/run_v2m_m3_conf.py") != expected_conf_sha:
        raise RuntimeError("frozen M3.1 source hash differs from amendment")
    conf_freeze = _read_json(conf.FREEZE_PATH)
    conf_path_fixture = {}
    for recovery_tag, backbone_dir in conf.BACKBONES:
        resolved = conf._backbone_paths(conf_freeze, backbone_dir)
        if resolved["key"].lower() != recovery_tag or not resolved["features_root"].is_dir() or not resolved["b3_root"].is_dir():
            raise AssertionError(f"M3.1 backbone path mapping failed for {recovery_tag}/{backbone_dir}")
        conf_path_fixture[recovery_tag] = {"freeze_key": resolved["key"], "backbone_dir": backbone_dir}
    if len(set(("m0", "m2", "m4", "m8"))) != 4:
        raise AssertionError("M3 severity-cell fixture failed")
    # Capture-index unit fixtures: exercise exact original call layouts with
    # sentinel predictions; anchors and diagnostics must not be archived.
    for method, levels, models, stride in (
        ("m3-b0", ["m0", "m2", "m4", "m8"], ["E_random", "E_curriculum", "EqualMix", "StaticMix", "AdaptiveMix"], 12),
        ("m3-conf", ["m0", "m2", "m4", "m8"], ["E_random", "E_curriculum", "EqualMix", "StaticMix", "AdaptiveMix"], 12),
    ):
        expected = {(level, model) for level in levels for model in models}
        observed: set[tuple[str, str]] = set()
        has_anchor_metric_call = method == "m3-conf"
        anchor_offset = 1 if has_anchor_metric_call else 0
        n_calls = anchor_offset + len(levels) * stride
        for call in range(n_calls):
            relative = call - anchor_offset
            if relative < 0:
                continue
            level_index, position = divmod(relative, stride)
            model_index = position if method == "m3-b0" else position - 1
            if level_index < len(levels) and 0 <= model_index < len(models):
                observed.add((levels[level_index], models[model_index]))
        if observed != expected:
            raise AssertionError(f"{method} point-capture fixture failed")
        class FakeEval:
            @staticmethod
            def point_metric_row(confidence: np.ndarray, correct: np.ndarray, *args: Any, **kwargs: Any) -> dict[str, float]:
                return {"sentinel": float(np.asarray(confidence).reshape(-1)[0])}

        capture = PointCapture(FakeEval, method, levels, models)
        with capture:
            for call in range(n_calls):
                marker = np.full(3, (call + 1) / 100.0, dtype=np.float64)
                capture._wrapped(marker, np.asarray([True, False, True]))
        capture.assert_complete(expected_calls=n_calls)
        for level_index, level in enumerate(levels):
            for model_index, model in enumerate(models):
                if method == "m3-b0":
                    source_call = level_index * stride + model_index
                else:
                    source_call = 1 + level_index * stride + 1 + model_index
                got = capture.captured[(level, model)][0]
                # B0 has no wrapped anchor metric call; M3-CONF does.
                expected_marker = (source_call + 1) / 100.0
                if not np.array_equal(got, np.full(3, expected_marker)):
                    raise AssertionError(f"{method}/{level}/{model}: capture-index sentinel mismatch")

    # Caller-level stub fixture: exercise the exact per-level row selection and
    # formatted diagnostic used by the original M3-B0 runner. The one synthetic
    # point row has a real paired estimate and NaN CI endpoints with an explicit
    # NOT_COMPUTED marker; no bootstrap draw is performed.
    from ccg.reliability import evaluate as reval

    class FakeRunner:
        macro_paired_cluster_bootstrap = staticmethod(lambda *args, **kwargs: None)

    fixture_conf_a = np.asarray([0.9, 0.8, 0.3, 0.2], dtype=np.float64)
    fixture_conf_b = np.asarray([0.1, 0.9, 0.8, 0.7], dtype=np.float64)
    fixture_correct = np.asarray([True, False, True, False])
    fixture_clusters = np.asarray([11, 11, 22, 22], dtype=np.int64)
    with suppress_bootstrap(reval, FakeRunner):
        stub_rows = reval.model_vs_model_bootstrap_row(
            fixture_conf_a, fixture_correct, fixture_conf_b, fixture_correct, fixture_clusters,
            eval_split="testA", K=10, model_a="AdaptiveMix", model_b="StaticMix",
            metrics=("auroc_correct",), replicates=1, seed=0, ci=0.95,
        )
        macro_stub = FakeRunner.macro_paired_cluster_bootstrap(
            reval._metric_fn("auroc_correct"),
            [((fixture_conf_a, fixture_correct), (fixture_conf_b, fixture_correct))],
            fixture_clusters,
            n_replicates=1, seed=0, ci=0.95, metric_name="auroc_correct",
        )
    required_stub_fields = {
        "eval_split", "K", "model_a", "model_b", "metric", "diff", "ci_low", "ci_high",
        "mean_a", "mean_b", "n", "n_clusters", "resample_unit", "n_replicates", "ci_level",
    }
    if len(stub_rows) != 1 or not required_stub_fields.issubset(stub_rows[0]):
        raise AssertionError("M3-B0 caller-level stub did not return the original model-pair row schema")
    expected_a = float(reval._metric_fn("auroc_correct")(
        reval.SampleStats.from_conf_correct(fixture_conf_a, fixture_correct)
    ))
    expected_b = float(reval._metric_fn("auroc_correct")(
        reval.SampleStats.from_conf_correct(fixture_conf_b, fixture_correct)
    ))
    stub = dict(stub_rows[0], level="m0", seed=1)
    caller_delta = float(np.mean([
        row["diff"] for row in [stub]
        if row["level"] == "m0" and row["model_a"] == "AdaptiveMix"
        and row["model_b"] == "StaticMix" and row["metric"] == "auroc_correct"
    ]))
    caller_message = (
        f"  [M3-B0 seed1] m0: adaptive-static dAUROC={caller_delta:+.4f}"
    )
    if not np.isclose(stub["mean_a"], expected_a) or not np.isclose(stub["mean_b"], expected_b):
        raise AssertionError("M3-B0 internal stub point values do not match the original metric function")
    if not np.isclose(caller_delta, expected_a - expected_b):
        raise AssertionError("M3-B0 caller-level diagnostic did not use the real paired point estimate")
    if not (np.isnan(stub["ci_low"]) and np.isnan(stub["ci_high"])):
        raise AssertionError("M3-B0 internal stub must not manufacture CI endpoints")
    if (
        stub["bootstrap_status"] != "NOT_COMPUTED" or stub["formal_bootstrap"]
        or not stub["recovery_internal_stub"] or stub["n_replicates"] != 0
    ):
        raise AssertionError("M3-B0 internal stub was not explicitly marked nonformal")
    if (
        not np.isclose(macro_stub["diff"], expected_a - expected_b)
        or not np.isnan(macro_stub["ci_low"])
        or macro_stub["bootstrap_status"] != "NOT_COMPUTED"
        or macro_stub["formal_bootstrap"]
        or macro_stub["n_replicates"] != 0
    ):
        raise AssertionError("M3-B0 macro internal stub point/status contract failed")
    if "dAUROC=" not in caller_message:
        raise AssertionError("M3-B0 caller-level log fixture was not formatted")

    # Recovery archive fixture: same canonical IDs across levels, while target
    # correctness is permitted to change with candidate composition.
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="schema-fixture-", dir=RECOVERY_ROOT) as tmpdir:
        base = Path(tmpdir)
        ids = np.asarray([10, 20, 30], dtype=np.int64)
        identity = {
            "sentence_id": ids,
            "image_id": np.asarray([1, 1, 2], dtype=np.int64),
            "ref_id": np.asarray([100, 200, 300], dtype=np.int64),
            "eval_split": np.asarray(["testA", "testA", "testB"]),
        }
        fixture = {
            "m5": {**identity, "correct": np.asarray([True, True, False]),
                   "predictions": {"R1": np.asarray([0.8, 0.7, 0.2])}},
            "m50": {**identity, "correct": np.asarray([False, True, False]),
                    "predictions": {"R1": np.asarray([0.6, 0.9, 0.1])}},
        }
        _validate_row_alignment(fixture, shared=True)
        archive_path = base / "predictions.npz"
        sidecar_path = base / "predictions.json"
        write_prediction_archive(
            archive_path, sidecar_path, job_id="fixture", scope="fixture", cohort="fixture",
            seed=1, cells=fixture, metadata={"formal_bootstrap_ran": False},
        )
        arrays, sidecar = load_prediction_archive(archive_path, sidecar_path)
        if len(sidecar["cells"]) != 2 or arrays["m5__sentence_id"].tolist() != ids.tolist():
            raise AssertionError("recovery archive fixture failed to round-trip")
        code_provenance = _snapshot_attempt_code(base, "fixture-code-snapshot")
        code_doc = _read_json(base / "attempt_source_provenance_manifest.json")
        if len(code_doc["code_snapshots"]) != 2 or not all(
            item["bytes_captured"] and item["source_sha256"] == item["snapshot_sha256"]
            for item in code_doc["code_snapshots"]
        ):
            raise AssertionError("attempt code snapshot fixture failed")
    return {
        "status": "PREFLIGHT_PASS",
        "source_files": len(RECOVERY_CODE_SOURCES),
        "baseline_inputs": len(_BASELINE_INPUTS),
        "formal_bootstrap_ran": False,
        "output_root": _relative(RECOVERY_ROOT),
        "point_capture_fixtures": 2,
        "m3_conf_backbone_path_fixture": {"status": "PASS", "resolved": conf_path_fixture},
        "archive_roundtrip_fixture": "PASS; cross-cell correctness may differ",
        "attempt_code_snapshot_fixture": {
            "status": "PASS", "sources": len(code_doc["code_snapshots"]),
            "provenance_path": code_provenance["path"],
        },
        "bootstrap_stub_fixture": {
            "status": "PASS",
            "caller_level_rows": 1,
            "original_schema_fields": sorted(required_stub_fields),
            "point_diff": caller_delta,
            "ci": "NaN / NOT_COMPUTED",
            "n_replicates": 0,
            "formal": False,
            "macro_point_diff": float(macro_stub["diff"]),
            "macro_ci": "NaN / NOT_COMPUTED",
            "caller_log": caller_message,
        },
        "g4_frozen_checkpoints": 6,
        "m3_conf_source_hash": "PASS",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("g4", "m25-m3b0", "m3-b0-check", "m3-conf", "all"), default="all")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--g4-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        print(json.dumps(_preflight(), indent=2), flush=True)
        return 0
    run_id, run_dir, manifest = _new_run(args.scope, args.run_id, args.g4_device)
    log = RunLog(run_dir / "recovery.log")
    log(f"START run_id={run_id} scope={args.scope} pid={os.getpid()} g4_device={args.g4_device} BLAS=2")
    try:
        with __import__("threadpoolctl").threadpool_limits(limits=2):
            if args.scope in ("g4", "all"):
                _recover_g4(run_dir, manifest, log, args.g4_device)
            if args.scope in ("m25-m3b0", "m3-b0-check", "all"):
                _recover_m25_and_b0(run_dir, manifest, log, archive_m25=args.scope != "m3-b0-check")
            if args.scope in ("m3-conf", "all"):
                _recover_m3_conf(run_dir, manifest, log)
    except BaseException as error:
        _record_failure(
            run_dir, manifest, scope_id=f"scope_{args.scope}_fatal",
            route=[{"step": "source-hash validation + deterministic original inference/recovery", "applicable": True}],
            source_paths=[Path(__file__)], tolerance={}, error=error, log=log,
        )
    finally:
        _finish_run(run_dir, manifest)
        log(f"FINISH run_id={run_id} status={manifest['status']} jobs={len(manifest['jobs'])} failures={len(manifest['failures'])}")
        log.close()
    return 0 if manifest["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
