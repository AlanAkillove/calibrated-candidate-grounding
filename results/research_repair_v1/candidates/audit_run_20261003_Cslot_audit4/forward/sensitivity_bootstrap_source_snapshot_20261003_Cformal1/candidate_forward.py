"""Frozen old/new candidate sensitivity inference for Research Repair v1 C.

The only candidate input comes from the hash-pinned ``current_audit.json`` and
its completed, versioned audit run. The old candidate arm is re-scored first and
must reproduce the archived Phase 1F or V2-P predictions at their original
tolerances before any corrected-candidate result is accepted. This module only
loads frozen B3 scorers and the already-exported reliability bundle; it never
fits, trains, extracts, or downloads anything.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np

from ccg.models.b3_data import B3Corpus
from ccg.reliability import features as rfeat
from ccg.semantic import features as sfeat
from ccg.semantic import frozen as sfrozen
from ccg.semantic import frozen_load as fl
from ccg.semantic import hard_scores as hscores
from ccg.repairs.atomic import replace_with_retry
from ccg.repairs.bootstrap import assert_row_alignment
from ccg.repairs.candidates import (
    OUT_DIR as CANDIDATE_OUT_DIR, ROOT, SEEDS, VARIANT,
    _sha256,
    load_test_manifests,
    verify_supplemental_input_manifest,
)

CURRENT_AUDIT_PATH = CANDIDATE_OUT_DIR / "current_audit.json"
INPUT_MANIFEST_PATH = ROOT / "results" / "research_repair_v1" / "input_manifest.json"
FROZEN_MODELS_ROOT = ROOT / "results" / "phase1e_refcocog_external" / "frozen_models"
B3_ROOT = ROOT / "results" / "phase0b_independent"
PHASE1F_PRED_DIR = ROOT / "results" / "phase1f_hard_semantic" / "predictions"
V2_PRED_DIR = ROOT / "results" / "v2_proposal_robustness" / "predictions"
IMAGE_SIZES_PATH = ROOT / "cache" / "image_sizes.npz"
FAMILY_CONFIG: Mapping[str, Mapping[str, Path]] = {
    "RPN": {
        "features": ROOT / "cache" / "features",
        "manifests": ROOT / "cache" / "manifests",
        "bank": ROOT / "cache" / "proposals.h5",
    },
    "DETR": {
        "features": ROOT / "cache" / "features_detr_r50",
        "manifests": ROOT / "cache" / "manifests_detr",
        "bank": ROOT / "cache" / "proposals_detr_r50.h5",
    },
}
ANCHOR_RAW_SCORE_ATOL = 1e-4
ANCHOR_CONFIDENCE_ATOL = float(sfrozen._TOL)
SEED_NAMES = tuple(f"b3_seed{seed}" for seed in SEEDS)


class CandidateForwardError(RuntimeError):
    """A frozen input, identity, or old-anchor invariant failed."""


@dataclass(frozen=True)
class FrozenSample:
    """Minimal B3 sample protocol, carrying an already-frozen candidate row."""

    sentence_id: int
    ref_id: int
    image_id: int
    target_index: int
    distractor_order: np.ndarray


def _rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename short: on Windows a long destination name is
    # otherwise duplicated into the temp path and can exceed MAX_PATH before
    # the atomic replace is reached.
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    replace_with_retry(temp, path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    replace_with_retry(temp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CandidateForwardError(f"expected a JSON object at {path}")
    return value


def _safe_repo_path(value: str, *, base: Path | None = None) -> Path:
    root = Path(ROOT if base is None else base).resolve()
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CandidateForwardError(f"input path escapes repository root: {path}") from exc
    return path


def resolve_current_audit(pointer_path: Path = CURRENT_AUDIT_PATH) -> Dict[str, Any]:
    """Verify the root pointer, completed summary, and candidate-array artifact."""
    pointer_path = Path(pointer_path).resolve()
    if pointer_path != CURRENT_AUDIT_PATH.resolve():
        # Test callers may use an alternate contained output root; production CLI
        # always uses the designated shared pointer path.
        try:
            pointer_path.relative_to(CANDIDATE_OUT_DIR.resolve())
        except ValueError as exc:
            raise CandidateForwardError("current-audit pointer must stay under candidates/") from exc
    pointer = _read_json(pointer_path)
    if pointer.get("schema") != "research-repair-v1-current-audit-pointer-v1":
        raise CandidateForwardError("current_audit.json schema mismatch")
    if pointer.get("status") not in ("AUDIT_COMPLETE_MISMATCH_FOUND", "AUDIT_COMPLETE_NO_MISMATCH"):
        raise CandidateForwardError("current audit pointer does not reference a completed audit")
    summary_path = _safe_repo_path(str(pointer.get("summary_path", "")), base=ROOT)
    audit_root = CANDIDATE_OUT_DIR.resolve()
    try:
        summary_path.relative_to(audit_root)
    except ValueError as exc:
        raise CandidateForwardError("pointer summary is outside the candidate output root") from exc
    expected_run = f"audit_run_{pointer.get('audit_run_id', '')}"
    if summary_path.parent.name != expected_run or summary_path.name != "summary.json":
        raise CandidateForwardError("pointer run ID and summary path disagree")
    summary_hash = _sha256(summary_path)
    if summary_hash != pointer.get("summary_sha256"):
        raise CandidateForwardError("current audit summary SHA256 does not match its pointer")
    summary = _read_json(summary_path)
    if summary.get("schema") != "research-repair-v1-candidate-summary-v1":
        raise CandidateForwardError("candidate summary schema mismatch")
    if summary.get("status") != pointer.get("status"):
        raise CandidateForwardError("pointer status differs from the candidate summary")
    if summary.get("audit_run_id") != pointer.get("audit_run_id"):
        raise CandidateForwardError("pointer run ID differs from the candidate summary")
    if int(summary.get("mismatch", {}).get("total_mismatches", 0)) <= 0:
        raise CandidateForwardError("mismatch count is zero; corrected-candidate forward is not applicable")
    if summary.get("candidate_variant") != VARIANT or not summary.get("candidate_variant_constructed"):
        raise CandidateForwardError("completed audit does not contain the required corrected candidate variant")
    artifact = summary.get("output_artifacts", {}).get("versioned_candidate_indices")
    if not isinstance(artifact, dict):
        raise CandidateForwardError("candidate summary has no hash-pinned versioned candidate archive")
    candidate_path = _safe_repo_path(str(artifact.get("path", "")))
    try:
        candidate_path.relative_to(summary_path.parent.resolve())
    except ValueError as exc:
        raise CandidateForwardError("candidate index archive is outside the pointed audit run") from exc
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if int(candidate_path.stat().st_size) != int(artifact.get("size_bytes", -1)):
        raise CandidateForwardError("candidate index archive size differs from audit summary")
    if _sha256(candidate_path) != artifact.get("sha256"):
        raise CandidateForwardError("candidate index archive SHA256 differs from audit summary")
    declared_path = summary.get("output_files", {}).get("versioned_candidate_indices")
    if declared_path != artifact["path"]:
        raise CandidateForwardError("candidate archive paths disagree within the audit summary")
    return {
        "pointer_path": pointer_path,
        "pointer": pointer,
        "summary_path": summary_path,
        "summary_sha256": summary_hash,
        "summary": summary,
        "candidate_indices_path": candidate_path,
        "candidate_indices_sha256": str(artifact["sha256"]),
        "forward_dir": summary_path.parent / "forward",
    }


def _baseline_records() -> tuple[Dict[str, Dict[str, Any]], str]:
    payload = _read_json(INPUT_MANIFEST_PATH)
    if payload.get("schema") != "research-repair-input-manifest-v1":
        raise CandidateForwardError("baseline input manifest schema mismatch")
    records = {str(row["path"]): dict(row) for row in payload.get("inputs", [])}
    if len(records) != len(payload.get("inputs", [])):
        raise CandidateForwardError("baseline input manifest contains duplicate paths")
    return records, _sha256(INPUT_MANIFEST_PATH)


def _require_baseline_coverage(paths: Iterable[Path], records: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    covered: Dict[str, Dict[str, Any]] = {}
    missing = []
    for path in paths:
        key = _rel(path)
        row = records.get(key)
        if row is None:
            missing.append(key)
        else:
            covered[key] = dict(row)
    if missing:
        raise CandidateForwardError(f"required frozen inputs are absent from the 794-file baseline: {missing}")
    return covered


def _feature_tree_manifest(
    features_root: Path, out_dir: Path, baseline_sha256: str, *, manifest_path: Path | None = None,
) -> Dict[str, Any]:
    """Snapshot/verify every DETR feature-cache file before FeatureCache opens it."""
    manifest_path = out_dir / "supplemental_input_manifest.json" if manifest_path is None else Path(manifest_path)
    manifest_path = manifest_path.resolve()
    try:
        manifest_path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise CandidateForwardError("DETR supplemental manifest must stay inside the repository") from exc
    root = features_root.resolve()
    try:
        root.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise CandidateForwardError("DETR feature root escapes repository") from exc
    if not root.is_dir():
        raise FileNotFoundError(root)
    required = {
        "metadata.json", "region_features.h5", "text_features.h5", "text_index.csv",
    }
    existing = {path.name for path in root.iterdir() if path.is_file()}
    absent = sorted(required - existing)
    if absent:
        raise CandidateForwardError(f"DETR feature cache is missing required files: {absent}")

    def current_inputs() -> list[dict[str, Any]]:
        rows = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError as exc:
                raise CandidateForwardError(f"feature-cache symlink escapes repository: {path}") from exc
            rows.append({"path": _rel(resolved), "size_bytes": int(resolved.stat().st_size), "sha256": _sha256(resolved)})
        if not rows:
            raise CandidateForwardError(f"no feature files found under {root}")
        return rows

    if manifest_path.exists():
        payload = _read_json(manifest_path)
        if payload.get("schema") != "research-repair-input-manifest-v1":
            raise CandidateForwardError("DETR feature supplemental manifest schema mismatch")
        if payload.get("baseline_manifest_sha256") != baseline_sha256:
            raise CandidateForwardError("baseline changed after the DETR feature snapshot")
        expected = {item["path"]: item for item in payload.get("inputs", [])}
        actual = current_inputs()
        actual_map = {row["path"]: row for row in actual}
        if expected != actual_map:
            raise CandidateForwardError("DETR feature files changed after the supplemental snapshot")
        return payload

    payload = {
        "schema": "research-repair-input-manifest-v1",
        "purpose": "DETR frozen feature-cache input omitted from the prepared 794-file baseline",
        "baseline_manifest": _rel(INPUT_MANIFEST_PATH),
        "baseline_manifest_sha256": baseline_sha256,
        "inputs": current_inputs(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, payload)
    return payload


def _load_source_availability(path: Path) -> Dict[tuple[str, str, str], Dict[int, Dict[str, Any]]]:
    tables: Dict[tuple[str, str, str], Dict[int, Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["family"]), str(row["candidate_source"]), str(row["cell"]))
            sid = int(row["sentence_id"])
            table = tables.setdefault(key, {})
            if sid in table:
                raise CandidateForwardError(f"availability table duplicates {key}/sentence{sid}")
            table[sid] = row
    return tables


def _load_split_by_sentence(path: Path) -> Dict[tuple[str, int], str]:
    out: Dict[tuple[str, int], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["family"]), int(row["sentence_id"]))
            split = str(row["eval_split"])
            old = out.setdefault(key, split)
            if old != split:
                raise CandidateForwardError(f"conflicting evaluation splits for {key}")
    return out


def _source_prediction_path(family: str, source: str, cell: str, seed: int) -> Path:
    scorer = f"b3_seed{seed}"
    if source == "phase1f_frozen_candidate_archive":
        return PHASE1F_PRED_DIR / f"{cell}__{scorer}.npz"
    if source == "v2_p1_frozen_hard_k5_predictions" and cell == "hard5":
        return V2_PRED_DIR / f"p1__{family}__hard_k5__{scorer}.npz"
    raise CandidateForwardError(f"unsupported historical source {source!r}/{family}/{cell}")


def _random_control_cell(source: str, cell: str) -> str:
    if source == "phase1f_frozen_candidate_archive":
        if cell == "hard5":
            return "rand5"
        if cell == "hard10":
            return "rand10"
        if cell in ("expb_m0", "expb_m2", "expb_m4", "expb_m8"):
            return "expb_m0"
        raise CandidateForwardError(f"no Phase 1F matched control for {cell!r}")
    if source == "v2_p1_frozen_hard_k5_predictions" and cell == "hard5":
        return "random_k5"
    raise CandidateForwardError(f"no matched random control defined for {source!r}/{cell}")


def _random_prediction_path(family: str, source: str, cell: str, seed: int) -> Path:
    scorer = f"b3_seed{seed}"
    random_cell = _random_control_cell(source, cell)
    if source == "phase1f_frozen_candidate_archive":
        return PHASE1F_PRED_DIR / f"{random_cell}__{scorer}.npz"
    return V2_PRED_DIR / f"p1__{family}__{random_cell}__{scorer}.npz"


def _random_candidates(
    *, family: str, source_rows: Mapping[str, np.ndarray], old_indices: np.ndarray,
    k: int, random_entries: Mapping[int, Any],
) -> np.ndarray:
    result = np.empty((len(source_rows["sentence_id"]), int(k)), dtype=np.int64)
    for row, (sid, ref_id) in enumerate(zip(source_rows["sentence_id"], source_rows["ref_id"], strict=True)):
        entry = random_entries.get(int(ref_id))
        if entry is None:
            raise CandidateForwardError(f"{family}: random manifest lacks ref_id={int(ref_id)}")
        if int(entry.image_id) != int(source_rows["image_id"][row]):
            raise CandidateForwardError(
                f"{family}/sentence{int(sid)}: matched random manifest image_id differs for ref_id={int(ref_id)}"
            )
        if entry.target_index is None or np.asarray(entry.distractor_order).size < int(k) - 1:
            raise CandidateForwardError(f"{family}/sentence{sid}: random manifest cannot form K={k}")
        result[row, 0] = int(entry.target_index)
        result[row, 1:] = np.asarray(entry.distractor_order, dtype=np.int64)[:int(k) - 1]
    if not np.array_equal(result[:, 0], old_indices[:, 0]):
        raise CandidateForwardError(f"{family}: matched random control target proposals differ from hard source")
    return result


def _load_source_reference(path: Path, canonical_ids: np.ndarray) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if "sentence_id" not in arrays:
        raise CandidateForwardError(f"{path}: missing sentence_id")
    source_ids = np.asarray(arrays["sentence_id"], dtype=np.int64).reshape(-1)
    if np.unique(source_ids).size != source_ids.size:
        raise CandidateForwardError(f"{path}: sentence_id is not unique")
    order = np.argsort(source_ids, kind="stable")
    sorted_ids = source_ids[order]
    positions = np.searchsorted(sorted_ids, canonical_ids)
    if np.any(positions >= sorted_ids.size) or not np.array_equal(sorted_ids[positions], canonical_ids):
        raise CandidateForwardError(f"{path}: historical predictions do not contain the candidate common cohort")
    return {name: value[order][positions] for name, value in arrays.items() if value.ndim >= 1 and value.shape[0] == source_ids.size}


def _align_candidate_rows(
    *, source_reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray],
    reference_name: str, candidate_name: str,
) -> None:
    if not all(field in source_reference and field in candidate for field in ("sentence_id", "ref_id", "image_id")):
        raise CandidateForwardError("candidate/source identity table is missing a required field")
    assert_row_alignment(
        source_reference, candidate, required_fields=("ref_id", "image_id"),
        reference_name=reference_name, candidate_name=candidate_name,
    )


def _samples(ids: Mapping[str, np.ndarray], candidates: np.ndarray) -> list[FrozenSample]:
    if candidates.ndim != 2 or candidates.shape[0] != np.asarray(ids["sentence_id"]).size:
        raise CandidateForwardError("candidate index matrix is misaligned with canonical IDs")
    if candidates.shape[1] < 2 or np.any(candidates[:, 0] < 0):
        raise CandidateForwardError("candidate set is malformed or has no target proposal")
    if np.any(candidates[:, 1:] < 0):
        raise CandidateForwardError("candidate set contains a negative distractor index")
    if any(np.unique(row).size != row.size for row in candidates):
        raise CandidateForwardError("candidate set contains duplicate bank indices")
    return [
        FrozenSample(
            int(ids["sentence_id"][i]), int(ids["ref_id"][i]), int(ids["image_id"][i]),
            int(candidates[i, 0]), np.asarray(candidates[i, 1:], dtype=np.int64),
        )
        for i in range(candidates.shape[0])
    ]


def _score_batch(
    *, models: Mapping[str, Any], bundle: fl.FrozenExternalModels,
    samples: Sequence[FrozenSample], corpus: B3Corpus, batch_size: int, row_chunk: int,
    reliability_score_overrides: Mapping[str, np.ndarray] | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    n = len(samples)
    if n == 0:
        raise CandidateForwardError("cannot forward an empty old/new common cohort")
    outputs: Dict[str, Dict[str, np.ndarray]] = {
        scorer: {
            "scores": np.empty((n, samples[0].distractor_order.size + 1), dtype=np.float32),
            "correct": np.empty(n, dtype=bool), "conf_msp": np.empty(n, dtype=np.float64),
            "conf_stats": np.empty(n, dtype=np.float64), "conf_e1b": np.empty(n, dtype=np.float64),
            "fresh_conf_msp": np.empty(n, dtype=np.float64),
            "fresh_conf_stats": np.empty(n, dtype=np.float64),
            "fresh_conf_e1b": np.empty(n, dtype=np.float64),
        }
        for scorer in SEED_NAMES
    }
    text_cache: Dict[int, np.ndarray] = {}
    for start in range(0, n, max(1, int(row_chunk))):
        stop = min(start + max(1, int(row_chunk)), n)
        block = samples[start:stop]
        batch = hscores.materialise_examples(corpus, block, samples[0].distractor_order.size + 1,
                                             text_cache=text_cache)
        for scorer in SEED_NAMES:
            temperature = float(bundle.seed(scorer).temperature)
            scores = hscores.score_examples(models[scorer], batch, batch_size=batch_size)
            scores64 = np.asarray(scores, dtype=np.float64)

            def confidence(logits64: np.ndarray) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
                scalars = rfeat.scalar_confidence(logits64, temperature=temperature)
                stats17 = np.asarray(rfeat.stat_features(logits64, temperature=temperature), dtype=np.float64)
                sem16 = np.empty((len(block), len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
                for sem_start in range(0, len(block), 4096):
                    sem_stop = min(sem_start + 4096, len(block))
                    sem16[sem_start:sem_stop] = sfeat.semantic_stats(
                        batch.z_q[sem_start:sem_stop], batch.z_i[sem_start:sem_stop],
                        logits64[sem_start:sem_stop],
                    )
                stats_conf, e1b_conf = bundle.predict(scorer, stats17, sem16)
                return scalars, stats_conf, e1b_conf

            fresh_scalars, fresh_stats_conf, fresh_e1b_conf = confidence(scores64)
            if reliability_score_overrides is not None and scorer in reliability_score_overrides:
                reference_logits = np.asarray(reliability_score_overrides[scorer], dtype=np.float32)
                if reference_logits.shape != (n, scores.shape[1]):
                    raise CandidateForwardError(
                        f"{scorer}: exact saved reliability logits shape {reference_logits.shape} "
                        f"does not match ({n}, {scores.shape[1]})"
                    )
                used_logits64 = reference_logits[start:stop].astype(np.float64)
                used_scalars, stats_conf, e1b_conf = confidence(used_logits64)
            else:
                used_scalars, stats_conf, e1b_conf = fresh_scalars, fresh_stats_conf, fresh_e1b_conf
            out = outputs[scorer]
            out["scores"][start:stop] = scores
            out["correct"][start:stop] = np.argmax(scores, axis=1) == 0
            out["conf_msp"][start:stop] = used_scalars["msp"]
            out["conf_stats"][start:stop] = stats_conf
            out["conf_e1b"][start:stop] = e1b_conf
            out["fresh_conf_msp"][start:stop] = fresh_scalars["msp"]
            out["fresh_conf_stats"][start:stop] = fresh_stats_conf
            out["fresh_conf_e1b"][start:stop] = fresh_e1b_conf
        del batch
    return outputs


def _anchor_old_predictions(
    *, source: str, family: str, cell: str,
    observed: Mapping[str, Mapping[str, np.ndarray]],
    references: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Dict[str, Any]]:
    checks: Dict[str, Dict[str, Any]] = {}
    for seed in SEEDS:
        scorer = f"b3_seed{seed}"
        ref = references[scorer]
        actual = observed[scorer]
        identity_reference = {
            "sentence_id": ref["sentence_id"], "ref_id": ref["ref_id"],
            "image_id": ref["image_id"], "correct": np.asarray(ref["correct"], dtype=bool),
        }
        identity_actual = {
            "sentence_id": actual["sentence_id"], "ref_id": actual["ref_id"],
            "image_id": actual["image_id"], "correct": actual["correct"],
        }
        assert_row_alignment(
            identity_reference, identity_actual, required_fields=("ref_id", "image_id", "correct"),
            reference_name=f"{source}/{family}/{cell}/{scorer} frozen old prediction",
            candidate_name=f"{source}/{family}/{cell}/{scorer} recomputed old prediction",
        )
        maxima: Dict[str, float] = {}
        if "scores" in ref:
            expected_scores = np.asarray(ref["scores"], dtype=np.float32)
            if expected_scores.shape != actual["scores"].shape:
                raise CandidateForwardError(f"{family}/{source}/{cell}/{scorer}: raw-score shape changed")
            maxima["raw_scores_max_abs"] = float(np.abs(
                expected_scores.astype(np.float64) - actual["scores"].astype(np.float64)
            ).max(initial=0.0))
            expected_order = np.argsort(-expected_scores.astype(np.float64), axis=1, kind="stable")
            actual_order = np.argsort(-np.asarray(actual["scores"], dtype=np.float64), axis=1, kind="stable")
            full_ranking_identical: bool | None = bool(np.array_equal(expected_order, actual_order))
            if not full_ranking_identical:
                raise CandidateForwardError(
                    f"{family}/{source}/{cell}/{scorer}: full stable candidate ranking differs from the frozen source"
                )
            confidence_route = "EXACT_SAVED_SOURCE_LOGITS_REPLAYED_THROUGH_FROZEN_BUNDLE"
        else:
            expected_top1 = np.asarray(ref["raw_top1"], dtype=np.float64)
            expected_margin = np.asarray(ref["raw_margin12"], dtype=np.float64)
            actual_ordered = np.sort(actual["scores"].astype(np.float64), axis=1)[:, ::-1]
            maxima["raw_top1_max_abs"] = float(np.abs(actual_ordered[:, 0] - expected_top1).max(initial=0.0))
            maxima["raw_margin12_max_abs"] = float(np.abs(
                actual_ordered[:, 0] - actual_ordered[:, 1] - expected_margin
            ).max(initial=0.0))
            full_ranking_identical = None
            confidence_route = "EXACT_ARCHIVED_SOURCE_CONFIDENCE_PRESERVED_NO_FULL_LOGITS"
        for field in ("conf_msp", "conf_stats", "conf_e1b"):
            if field not in ref:
                raise CandidateForwardError(f"{family}/{source}/{cell}/{scorer}: reference lacks {field}")
            maxima[f"fresh_{field}_max_abs"] = float(np.abs(
                np.asarray(ref[field], dtype=np.float64) - np.asarray(actual.get(f"fresh_{field}", actual[field]), dtype=np.float64)
            ).max(initial=0.0))
            if "scores" in ref:
                maxima[f"{field}_exact_saved_logits_max_abs"] = float(np.abs(
                    np.asarray(ref[field], dtype=np.float64) - np.asarray(actual[field], dtype=np.float64)
                ).max(initial=0.0))
        raw_names = [name for name in maxima if name.startswith("raw_")]
        conf_names = [name for name in maxima if name.endswith("_exact_saved_logits_max_abs")]
        if any(maxima[name] > ANCHOR_RAW_SCORE_ATOL for name in raw_names):
            raise CandidateForwardError(
                f"{family}/{source}/{cell}/{scorer}: raw B3 anchor failed tolerance "
                f"{ANCHOR_RAW_SCORE_ATOL:g}: {maxima}"
            )
        if any(maxima[name] > ANCHOR_CONFIDENCE_ATOL for name in conf_names):
            raise CandidateForwardError(
                f"{family}/{source}/{cell}/{scorer}: frozen reliability anchor failed tolerance "
                f"{ANCHOR_CONFIDENCE_ATOL:g}: {maxima}"
            )
        checks[scorer] = {
            "passed": True, "rows": int(identity_actual["sentence_id"].size),
            "raw_score_tolerance": ANCHOR_RAW_SCORE_ATOL,
            "confidence_route": confidence_route,
            "confidence_tolerance": ANCHOR_CONFIDENCE_ATOL if "scores" in ref else None,
            "confidence_anchor_status": "PASS_1E-9" if "scores" in ref else "PRESERVED_SOURCE_CONFIDENCE_NO_FULL_LOGITS",
            "full_stable_ranking_identical": full_ranking_identical,
            "max_abs_deltas": maxima,
        }
    return checks


def _preserve_source_predictions(
    scored: Dict[str, Dict[str, np.ndarray]],
    references: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Keep exact historical predictions after independent frozen scorer replay.

    Phase 1F archives full float32 logits, so their reliability route can be
    recomputed and checked before preserving the source rows. V2-P archives do
    not contain full logits; its already-frozen correctness/confidence are kept
    verbatim while the independent raw-top1/margin scorer replay is checked.
    """
    for scorer in SEED_NAMES:
        ref = references[scorer]
        actual = scored[scorer]
        actual["correct"] = np.asarray(ref["correct"], dtype=bool).copy()
        if "scores" in ref:
            actual["scores"] = np.asarray(ref["scores"], dtype=np.float32).copy()
        for field in ("conf_msp", "conf_stats", "conf_e1b"):
            actual[field] = np.asarray(ref[field], dtype=np.float64).copy()
    return scored


def _read_candidate_cell(
    archive: Mapping[str, np.ndarray], availability: Mapping[tuple[str, str, str], Mapping[int, Dict[str, Any]]],
    *, family: str, source: str, cell: str,
) -> Dict[str, np.ndarray]:
    prefix = f"{family}__{source}__{cell}"
    required = {
        "sentence_id": f"{prefix}__sentence_id", "ref_id": f"{prefix}__ref_id",
        "image_id": f"{prefix}__image_id", "ann_id": f"{prefix}__ann_id",
        "old_indices": f"{prefix}__old_indices",
        "new_indices": f"{prefix}__true_target_category_v1_indices",
    }
    missing = [name for name, key in required.items() if key not in archive]
    if missing:
        return {}
    out = {name: np.asarray(archive[key]) for name, key in required.items()}
    for name in ("sentence_id", "ref_id", "image_id", "ann_id"):
        out[name] = np.asarray(out[name], dtype=np.int64).reshape(-1)
    if np.unique(out["sentence_id"]).size != out["sentence_id"].size:
        raise CandidateForwardError(f"{family}/{source}/{cell}: duplicate candidate sentence_id")
    order = np.argsort(out["sentence_id"], kind="stable")
    out = {name: values[order] for name, values in out.items()}
    if not np.array_equal(out["sentence_id"], np.sort(out["sentence_id"])):
        raise CandidateForwardError(f"{family}/{source}/{cell}: candidates are not canonical sentence order")
    if out["old_indices"].shape != out["new_indices"].shape or out["old_indices"].shape[0] != out["sentence_id"].size:
        raise CandidateForwardError(f"{family}/{source}/{cell}: old/new candidate sets are misaligned")
    if not np.array_equal(out["old_indices"][:, 0], out["new_indices"][:, 0]):
        raise CandidateForwardError(f"{family}/{source}/{cell}: correction changed the target proposal")
    key = (family, source, cell)
    available = availability.get(key)
    if available is None:
        raise CandidateForwardError(f"source availability table lacks {key}")
    common_ids = np.asarray(sorted(
        sid for sid, row in available.items()
        if str(row.get("in_old_new_common_intersection")) == "1"
    ), dtype=np.int64)
    for sid in common_ids.tolist():
        row = available[int(sid)]
        if str(row.get("historical_candidate_available")) != "1" or str(row.get("corrected_candidate_available")) != "1":
            raise CandidateForwardError(f"{family}/{source}/{cell}/sentence{sid}: common flag conflicts with availability")
    if not np.array_equal(out["sentence_id"], common_ids):
        raise CandidateForwardError(f"{family}/{source}/{cell}: candidate rows differ from old/new availability intersection")
    return out


def _dose_fixed_m0_indices(
    candidate_archive: Mapping[str, np.ndarray],
    availability: Mapping[tuple[str, str, str], Mapping[int, Dict[str, Any]]],
    *, family: str, source: str, ids: Mapping[str, np.ndarray],
    current_old_indices: np.ndarray,
) -> np.ndarray:
    """Return original Phase 1F expb_m0 candidates aligned to a dose cell."""
    m0_candidate = _read_candidate_cell(
        candidate_archive, availability, family=family, source=source, cell="expb_m0",
    )
    if not m0_candidate:
        raise CandidateForwardError(f"{family}/{source}/expb_m0: fixed dose control is absent")
    m0_ids = np.asarray(m0_candidate["sentence_id"], dtype=np.int64)
    target_ids = np.asarray(ids["sentence_id"], dtype=np.int64)
    m0_positions = np.searchsorted(m0_ids, target_ids)
    present = m0_positions < m0_ids.size
    if np.any(present):
        present[present] &= m0_ids[m0_positions[present]] == target_ids[present]
    if not bool(np.all(present)):
        absent = target_ids[~present]
        raise CandidateForwardError(
            f"{family}/{source}: {absent.size} old/new rows are outside the fixed m0 common cohort; "
            f"example sentence_ids={absent[:8].tolist()}"
        )
    for field in ("ref_id", "image_id", "ann_id"):
        if not np.array_equal(m0_candidate[field][m0_positions], np.asarray(ids[field], dtype=np.int64)):
            raise CandidateForwardError(f"{family}/{source}: fixed m0 control {field} identities do not align")
    indices = np.asarray(m0_candidate["old_indices"][m0_positions], dtype=np.int64)
    current = np.asarray(current_old_indices, dtype=np.int64)
    if indices.shape != current.shape:
        raise CandidateForwardError(
            f"{family}/{source}: fixed m0 control candidate matrix shape {indices.shape} != dose source {current.shape}"
        )
    if not np.array_equal(indices[:, 0], current[:, 0]):
        raise CandidateForwardError(f"{family}/{source}: fixed m0 control target proposal differs from dose source")
    return indices


def build_prediction_archive(
    *, family: str, source: str, cell: str, candidate: Mapping[str, np.ndarray],
    eval_split: np.ndarray, old_result: Mapping[str, Mapping[str, np.ndarray]],
    new_result: Mapping[str, Mapping[str, np.ndarray]],
    random_result: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    random_indices: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Build the seed-aligned, row-canonical archive consumed by A's bootstrap."""
    ids = np.asarray(candidate["sentence_id"], dtype=np.int64)
    n = ids.size
    split = np.asarray(eval_split, dtype="U8").reshape(-1)
    if split.size != n:
        raise CandidateForwardError("eval_split length differs from candidate common cohort")
    old_indices = np.asarray(candidate["old_indices"], dtype=np.int64)
    new_indices = np.asarray(candidate["new_indices"], dtype=np.int64)
    if old_indices.shape != new_indices.shape or old_indices.shape[0] != n:
        raise CandidateForwardError("old/new candidate index matrices are misaligned")
    archive: Dict[str, np.ndarray] = {
        "sentence_id": ids,
        "ref_id": np.asarray(candidate["ref_id"], dtype=np.int64),
        "image_id": np.asarray(candidate["image_id"], dtype=np.int64),
        "ann_id": np.asarray(candidate["ann_id"], dtype=np.int64),
        "eval_split": split,
        "cell": np.asarray(cell), "k": np.asarray(int(old_indices.shape[1])),
        "family": np.asarray(family), "candidate_source": np.asarray(source),
        "condition_names": np.asarray(
            ("old", "random", VARIANT) if random_result is not None else ("old", VARIANT), dtype="U64"
        ),
        "seed_names": np.asarray(SEED_NAMES, dtype="U16"),
        "old_candidate_indices": old_indices,
        "true_target_category_v1_candidate_indices": new_indices,
    }
    if random_result is not None:
        if random_indices is None or np.asarray(random_indices).shape != old_indices.shape:
            raise CandidateForwardError("matched random candidate indices must align with old/new sets")
        archive["random_candidate_indices"] = np.asarray(random_indices, dtype=np.int64)
    for scorer in SEED_NAMES:
        conditions = [("old", old_result[scorer])]
        if random_result is not None:
            conditions.append(("random", random_result[scorer]))
        conditions.append((VARIANT, new_result[scorer]))
        for condition, values in conditions:
            for field in ("scores", "correct", "conf_msp", "conf_stats", "conf_e1b"):
                array = np.asarray(values[field])
                if array.shape[0] != n:
                    raise CandidateForwardError(f"{condition}/{scorer}/{field} row count differs from canonical cohort")
                archive[f"{condition}__{scorer}__{field}"] = array
    return archive


def _collect_expected_sources(summary: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    rows = summary.get("historical_candidate_source_scope", {}).get("sources", [])
    sources = []
    for row in rows:
        family, cell, source = str(row["family"]), str(row["cell"]), str(row["candidate_source"])
        if int(row.get("source_rows_covered_by_category_audit", 0)) != int(row.get("source_rows", -1)):
            raise CandidateForwardError(f"{family}/{source}/{cell}: historical cohort is not fully category-audited")
        sources.append((family, source, cell))
    if not sources:
        raise CandidateForwardError("candidate summary has no historical candidate sources")
    return sources


def run_category_sensitivity_forward(
    *, pointer_path: Path = CURRENT_AUDIT_PATH, device: str = "cuda",
    batch_size: int = 64, row_chunk: int = 128,
    preflight_rows: int | None = None,
    detr_feature_manifest_path: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> Dict[str, Any]:
    """Forward historical old/new common cohorts, or preflight their anchors."""
    emit = log or (lambda message: print(f"[{time.strftime('%H:%M:%S')}] [repair-candidate-forward] {message}", flush=True))
    audit = resolve_current_audit(pointer_path)
    if preflight_rows is not None and int(preflight_rows) <= 0:
        raise ValueError("preflight_rows must be positive")
    if preflight_rows is None:
        out_dir = audit["forward_dir"]
    else:
        out_dir = audit["forward_dir"].parent / (
            f"forward_preflight_rows{int(preflight_rows)}_"
            f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        )
    if out_dir.exists():
        raise FileExistsError(f"candidate forward output already exists; refusing overwrite: {out_dir}")
    summary = audit["summary"]
    candidate_path = audit["candidate_indices_path"]
    availability_path = audit["summary_path"].parent / summary["output_files"]["candidate_source_availability"]
    mismatch_path = audit["summary_path"].parent / summary["output_files"]["expression_audit"]
    if not availability_path.is_file() or not mismatch_path.is_file():
        raise CandidateForwardError("pointed audit run is missing availability or mismatch detail tables")

    baseline_records, baseline_sha = _baseline_records()
    # Verify only required paths are present in the immutable 794-file baseline;
    # the root task already performs whole-manifest content verification.
    required = [
        IMAGE_SIZES_PATH, FAMILY_CONFIG["RPN"]["bank"], FAMILY_CONFIG["DETR"]["bank"],
        FROZEN_MODELS_ROOT / "models.json", FROZEN_MODELS_ROOT / "models.sha256",
        FROZEN_MODELS_ROOT / "frozen_artifact_manifest.json",
        FROZEN_MODELS_ROOT / "bundle_verification.json",
    ]
    required.extend(FAMILY_CONFIG["RPN"]["features"] / name for name in (
        "metadata.json", "region_features.h5", "text_features.h5", "text_index.csv",
    ))
    for family, config in FAMILY_CONFIG.items():
        manifest_root = config["manifests"] / "manifests" if family == "DETR" else config["manifests"]
        for regime in ("random", "same_category"):
            for split in ("testA", "testB"):
                required.append(manifest_root / f"{regime}_{split}.jsonl")
    for seed in SEEDS:
        required.extend((B3_ROOT / f"seed_{seed}" / "model.npz", B3_ROOT / f"seed_{seed}" / "training.json"))
    sources = _collect_expected_sources(summary)
    for family, source, cell in sources:
        for seed in SEEDS:
            required.append(_source_prediction_path(family, source, cell, seed))
            required.append(_random_prediction_path(family, source, cell, seed))
    baseline_coverage = _require_baseline_coverage(required, baseline_records)

    supplemental_candidate = verify_supplemental_input_manifest()
    if supplemental_candidate["inputs"][0]["sha256"] != summary["inputs"]["coco_gt_sha256"]:
        raise CandidateForwardError("COCO supplemental manifest identity differs from candidate summary")

    # A new, run-local manifest pins all DETR feature-cache bytes before the first
    # B3Corpus/FeatureCache access to that tree.
    out_dir.mkdir(parents=True, exist_ok=False)
    detr_feature_manifest_path = (
        out_dir / "supplemental_input_manifest.json"
        if detr_feature_manifest_path is None else Path(detr_feature_manifest_path)
    ).resolve()
    detr_feature_manifest = _feature_tree_manifest(
        FAMILY_CONFIG["DETR"]["features"], out_dir, baseline_sha,
        manifest_path=detr_feature_manifest_path,
    )
    if not {item["path"] for item in detr_feature_manifest["inputs"]}.issuperset({
        _rel(FAMILY_CONFIG["DETR"]["features"] / name) for name in (
            "metadata.json", "region_features.h5", "text_features.h5", "text_index.csv",
        )
    }):
        raise CandidateForwardError("DETR supplemental snapshot does not cover the required cache files")

    # Static + runtime no-fit guards precede even loading the prediction models.
    fl.assert_no_fit_path([Path(__file__)])
    import torch

    torch.set_num_threads(2)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise CandidateForwardError("CUDA was requested for the frozen forward but is unavailable")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _log_lines = []

    def log(message: str) -> None:
        _log_lines.append(message)
        emit(message)

    started = time.perf_counter()
    with fl.fit_is_forbidden() as tripwire:
        bundle = fl.load_models(FROZEN_MODELS_ROOT)
        if tuple(bundle.scorers) != tuple(SEED_NAMES):
            raise CandidateForwardError(f"frozen reliability bundle seeds differ: {bundle.scorers}")
        models = hscores.load_frozen_scorers(b3_root=B3_ROOT, seeds=SEEDS, device=str(device))
        bundle_verification = fl.verify_against_frozen_artifacts(
            bundle, scorers=SEED_NAMES,
            phase05_root=ROOT / "results" / "phase05_score_sufficiency",
            phase1_root=ROOT / "results" / "phase1_semantic_sufficiency",
            phase1f_root=ROOT / "results" / "phase1f_hard_semantic",
            b3_root=B3_ROOT,
            features_dir=ROOT / "results" / "phase05_score_sufficiency" / "features",
            log=log,
        )
        with np.load(candidate_path, allow_pickle=False) as store:
            candidate_arrays = {name: np.asarray(store[name]) for name in store.files}
        source_availability = _load_source_availability(availability_path)
        split_by_sentence = _load_split_by_sentence(mismatch_path)
        random_manifest_entries = {
            family: load_test_manifests(family)[0]
            for family in sorted({row[0] for row in sources})
        }
        source_results: list[Dict[str, Any]] = []
        progress = {
            "schema": "research-repair-v1-candidate-forward-progress-v1",
            "status": "RUNNING", "audit_run_id": audit["pointer"]["audit_run_id"],
            "completed_sources": [], "n_sources": len(sources), "device": str(device),
        }
        _write_json(out_dir / "progress.json", progress)
        for family, source, cell in sources:
            availability_key = (family, source, cell)
            availability_rows = source_availability.get(availability_key)
            if availability_rows is None:
                raise CandidateForwardError(f"source availability table lacks {availability_key}")
            common_ids = [
                sid for sid, row in availability_rows.items()
                if str(row.get("in_old_new_common_intersection")) == "1"
            ]
            if not common_ids:
                source_results.append({
                    "family": family, "candidate_source": source, "cell": cell,
                    "status": "NOT_FORWARDABLE_EMPTY_OLD_NEW_INTERSECTION",
                    "n_rows": 0, "candidate_source_rows": len(availability_rows),
                    "old_new_intersection_rows": 0,
                    "rows_lost_under_true_category": len(availability_rows),
                    "reason": "no source expression has both a historical and corrected candidate set",
                    "anchor_checks": None, "output": None,
                })
                progress["completed_sources"] = [
                    f"{row['family']}/{row['candidate_source']}/{row['cell']}" for row in source_results
                ]
                progress["status"] = "PARTIAL"
                _write_json(out_dir / "progress.json", progress)
                log(f"{family}/{source}/{cell}: no old/new common rows; recorded availability attrition")
                continue
            candidate = _read_candidate_cell(candidate_arrays, source_availability,
                                             family=family, source=source, cell=cell)
            if not candidate:
                raise CandidateForwardError(f"{family}/{source}/{cell}: no old/new common candidate rows")
            if preflight_rows is not None:
                limit = min(int(preflight_rows), int(candidate["sentence_id"].size))
                candidate = {name: values[:limit] for name, values in candidate.items()}
                log(f"{family}/{source}/{cell}: preflight checks first {limit} canonical rows")
            ids = {name: candidate[name] for name in ("sentence_id", "ref_id", "image_id", "ann_id")}
            refs_by_seed: Dict[str, Dict[str, np.ndarray]] = {}
            ref_paths: Dict[str, Path] = {}
            random_refs_by_seed: Dict[str, Dict[str, np.ndarray]] = {}
            random_ref_paths: Dict[str, Path] = {}
            for seed in SEEDS:
                scorer = f"b3_seed{seed}"
                ref_path = _source_prediction_path(family, source, cell, seed)
                ref_paths[scorer] = ref_path
                refs_by_seed[scorer] = _load_source_reference(ref_path, ids["sentence_id"])
                ref_identity = refs_by_seed[scorer]
                _align_candidate_rows(
                    source_reference=ref_identity,
                    candidate=ids,
                    reference_name=f"{source}/{family}/{cell}/{scorer} source identities",
                    candidate_name=f"{source}/{family}/{cell} candidate archive",
                )
                random_path = _random_prediction_path(family, source, cell, seed)
                random_ref_paths[scorer] = random_path
                random_refs_by_seed[scorer] = _load_source_reference(random_path, ids["sentence_id"])
                _align_candidate_rows(
                    source_reference=random_refs_by_seed[scorer], candidate=ids,
                    reference_name=f"{source}/{family}/{cell}/{scorer} matched-random identities",
                    candidate_name=f"{family}/{source}/{cell} candidate archive",
                )
            old_samples = _samples(ids, np.asarray(candidate["old_indices"], dtype=np.int64))
            new_samples = _samples(ids, np.asarray(candidate["new_indices"], dtype=np.int64))
            k = int(candidate["old_indices"].shape[1])
            if k != int(candidate["new_indices"].shape[1]):
                raise CandidateForwardError(f"{family}/{source}/{cell}: old/new K differs")
            if source == "phase1f_frozen_candidate_archive" and cell in (
                "expb_m0", "expb_m2", "expb_m4", "expb_m8",
            ):
                # Dose-level controls are the historical expb_m0 construction
                # ([target] + first nine of the level-rest shuffle), not the
                # canonical random-manifest rand10 order. Reuse the versioned
                # old m0 rows and align only by canonical sentence_id.
                random_indices = _dose_fixed_m0_indices(
                    candidate_arrays, source_availability, family=family, source=source,
                    ids=ids, current_old_indices=np.asarray(candidate["old_indices"], dtype=np.int64),
                )
            else:
                random_indices = _random_candidates(
                    family=family, source_rows=ids,
                    old_indices=np.asarray(candidate["old_indices"], dtype=np.int64),
                    k=k, random_entries=random_manifest_entries[family],
                )
            random_samples = _samples(ids, random_indices)

            config = FAMILY_CONFIG[family]
            corpus = B3Corpus(
                config["features"], config["manifests"], [], config["bank"],
                image_sizes_path=IMAGE_SIZES_PATH, region_cache_size=64,
                bank_cache_size=128, ks=(len(candidate["old_indices"][0]),),
                regime="same_category", preload=False,
            )
            try:
                old_score_overrides = {
                    scorer: np.asarray(reference["scores"], dtype=np.float32)
                    for scorer, reference in refs_by_seed.items() if "scores" in reference
                }
                old_result = _score_batch(
                    models=models,
                    bundle=bundle, samples=old_samples, corpus=corpus,
                    batch_size=batch_size, row_chunk=row_chunk,
                    reliability_score_overrides=old_score_overrides,
                )
                old_refs = refs_by_seed
                anchor_checks = _anchor_old_predictions(
                    source=source, family=family, cell=cell,
                    observed={scorer: {"sentence_id": ids["sentence_id"], "ref_id": ids["ref_id"],
                                       "image_id": ids["image_id"], **values}
                              for scorer, values in old_result.items()},
                    references=old_refs,
                )
                old_result = _preserve_source_predictions(old_result, old_refs)
                random_score_overrides = {
                    scorer: np.asarray(reference["scores"], dtype=np.float32)
                    for scorer, reference in random_refs_by_seed.items() if "scores" in reference
                }
                random_result = _score_batch(
                    models=models, bundle=bundle, samples=random_samples, corpus=corpus,
                    batch_size=batch_size, row_chunk=row_chunk,
                    reliability_score_overrides=random_score_overrides,
                )
                random_anchor_checks = _anchor_old_predictions(
                    source=f"matched_random_{_random_control_cell(source, cell)}",
                    family=family, cell=cell,
                    observed={scorer: {"sentence_id": ids["sentence_id"], "ref_id": ids["ref_id"],
                                       "image_id": ids["image_id"], **values}
                              for scorer, values in random_result.items()},
                    references=random_refs_by_seed,
                )
                random_result = _preserve_source_predictions(random_result, random_refs_by_seed)
                new_result = _score_batch(
                    models=models,
                    bundle=bundle, samples=new_samples, corpus=corpus,
                    batch_size=batch_size, row_chunk=row_chunk,
                )
            finally:
                corpus.close()

            eval_split = np.asarray([
                split_by_sentence[(family, int(sid))] for sid in ids["sentence_id"]
            ], dtype="U8")
            archive = build_prediction_archive(
                family=family, source=source, cell=cell, candidate=candidate,
                eval_split=eval_split, old_result=old_result, new_result=new_result,
                random_result=random_result, random_indices=random_indices,
            )
            filename = f"{family}__{source}__{cell}__all_seeds.npz"
            output_path = out_dir / filename
            _write_npz(output_path, archive)
            source_row = {
                "family": family, "candidate_source": source, "cell": cell,
                "status": "FORWARDED_WITH_OLD_ANCHOR_PASS",
                "k": int(candidate["old_indices"].shape[1]),
                "n_rows": int(ids["sentence_id"].size),
                "n_images": int(np.unique(ids["image_id"]).size),
                "candidate_source_rows": int(len(source_availability[(family, source, cell)])),
                "old_new_intersection_rows": int(ids["sentence_id"].size),
                "rows_lost_under_true_category": int(
                    len(source_availability[(family, source, cell)]) - ids["sentence_id"].size
                ),
                "canonical_order": "ascending sentence_id; rows are not joined or deduplicated by ref_id",
                "anchor_checks": anchor_checks,
                "historical_prediction_routes": {
                    "raw_scorer_replay": "fresh frozen B3 FP32 replay; separately checked against original source STOP tolerance and stable ranking when full logits are archived",
                    "old_confidence": {scorer: anchor_checks[scorer]["confidence_route"] for scorer in SEED_NAMES},
                    "matched_random_confidence": {scorer: random_anchor_checks[scorer]["confidence_route"] for scorer in SEED_NAMES},
                    "correctness": "source archive preserved after exact zero-difference replay check",
                    "fresh_replay_confidence_residual_is_diagnostic_only": True,
                },
                "matched_random_control": {
                    "cell": _random_control_cell(source, cell),
                    "control_type": "fixed_original_m0_candidate_control" if cell.startswith("expb_") else "matched_random_manifest_candidate_control",
                    "anchor_checks": random_anchor_checks,
                    "source_prediction_hashes": {
                        scorer: {"path": _rel(path), "size_bytes": int(path.stat().st_size), "sha256": _sha256(path)}
                        for scorer, path in random_ref_paths.items()
                    },
                },
                "source_prediction_hashes": {
                    scorer: {"path": _rel(path), "size_bytes": int(path.stat().st_size), "sha256": _sha256(path)}
                    for scorer, path in ref_paths.items()
                },
                "output": {
                    "path": _rel(output_path), "size_bytes": int(output_path.stat().st_size),
                    "sha256": _sha256(output_path),
                },
            }
            source_results.append(source_row)
            progress["completed_sources"] = [
                f"{row['family']}/{row['candidate_source']}/{row['cell']}" for row in source_results
            ]
            progress["status"] = "PARTIAL"
            _write_json(out_dir / "progress.json", progress)
            log(f"{family}/{source}/{cell}: forwarded old/new n={source_row['n_rows']} rows, K={source_row['k']}")

        if tripwire.hits:
            raise CandidateForwardError(f"fit tripwire detected forbidden calls: {tripwire.hits}")

    preflight = preflight_rows is not None
    forward_manifest = {
        "schema": (
            "research-repair-v1-candidate-forward-preflight-manifest-v1"
            if preflight else "research-repair-v1-candidate-forward-manifest-v1"
        ),
        "status": "PREFLIGHT_ONLY" if preflight else "COMPLETE",
        "preflight_rows_per_source": int(preflight_rows) if preflight else None,
        "audit_run_id": audit["pointer"]["audit_run_id"],
        "candidate_audit_summary_sha256": audit["summary_sha256"],
        "archive_schema": "canonical-row-three-seed-old-random-corrected-v1",
        "row_fields": ["sentence_id", "ref_id", "image_id", "ann_id", "eval_split", "cell", "k"],
        "condition_names": ["old", "random", VARIANT],
        "prediction_key_template": "{condition}__b3_seed{1|2|3}__{scores|correct|conf_msp|conf_stats|conf_e1b}",
        "canonical_row_order": "ascending sentence_id; each archive holds one source/family/cell old-new availability intersection",
        "entries": source_results,
    }
    manifest_path = out_dir / "predictions_manifest.json"
    _write_json(manifest_path, forward_manifest)

    report: Dict[str, Any] = {
        "schema": (
            "research-repair-v1-candidate-forward-preflight-v1"
            if preflight else "research-repair-v1-candidate-forward-summary-v1"
        ),
        "status": "PREFLIGHT_PASS" if preflight else "COMPLETE",
        "preflight_only": preflight,
        "preflight_rows_per_source": int(preflight_rows) if preflight else None,
        "audit_run_id": audit["pointer"]["audit_run_id"],
        "candidate_audit_pointer": {
            "path": _rel(audit["pointer_path"]), "summary_path": _rel(audit["summary_path"]),
            "summary_sha256": audit["summary_sha256"],
            "candidate_indices_path": _rel(candidate_path),
            "candidate_indices_sha256": audit["candidate_indices_sha256"],
        },
        "method": "frozen B3 scorer plus frozen load-and-predict Stats/E1b bundle; no refit, training, extraction, or download",
        "anchor_contract": {
            "old_candidate_first": True,
            "candidate_id_source": "hash-pinned versioned candidate NPZ; paired source cohorts are checked by canonical sentence_id/ref_id/image_id",
            "raw_score_tolerance": ANCHOR_RAW_SCORE_ATOL,
            "reliability_tolerance": ANCHOR_CONFIDENCE_ATOL,
            "reliability_bundle_verification": bundle_verification,
        },
        "cohort_contract": {
            "source_cohorts_remain_separate": True,
            "intersection": "historical old candidate availability ∩ corrected true-target-category availability, per family/source/cell",
            "row_order": "ascending sentence_id; no ref_id joins or deduplication",
            "families_and_cells": source_results,
        },
        "predictions_manifest": {
            "path": _rel(manifest_path), "size_bytes": int(manifest_path.stat().st_size),
            "sha256": _sha256(manifest_path),
            "schema": forward_manifest["schema"],
        },
        "bootstrap": {
            "status": "NOT_RUN_BY_C_FORWARD",
            "consumer": "C candidate sensitivity runner using ccg.repairs.statistics.joint_prediction_bootstrap",
            "planned_output": _rel(out_dir / "sensitivity_bootstrap"),
        },
        "inputs": {
            "baseline_manifest_path": _rel(INPUT_MANIFEST_PATH),
            "baseline_manifest_sha256": baseline_sha,
            "baseline_covered_input_count": len(baseline_coverage),
            "baseline_covered_inputs": baseline_coverage,
            "candidate_coco_supplemental_manifest": _rel(CANDIDATE_OUT_DIR / "supplemental_input_manifest.json"),
            "candidate_cooco_sha256": supplemental_candidate["inputs"][0]["sha256"],
            "detr_features_supplemental_manifest": _rel(detr_feature_manifest_path),
            "detr_features_manifest_sha256": _sha256(detr_feature_manifest_path),
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "CPU",
        "forward_seconds": round(time.perf_counter() - started, 3),
        "new_training": 0, "new_feature_extraction": 0, "model_refits": 0,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if torch.cuda.is_available():
        report["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    _write_json(out_dir / "summary.json", report)
    _write_json(out_dir / "progress.json", {
        "schema": "research-repair-v1-candidate-forward-progress-v1",
        "status": "PREFLIGHT_PASS" if preflight else "COMPLETE",
        "preflight_only": preflight, "audit_run_id": audit["pointer"]["audit_run_id"],
        "completed_sources": [f"{row['family']}/{row['candidate_source']}/{row['cell']}" for row in source_results],
        "n_sources": len(sources), "summary": "summary.json",
    })
    return report


__all__ = [
    "ANCHOR_CONFIDENCE_ATOL", "ANCHOR_RAW_SCORE_ATOL", "CandidateForwardError",
    "FrozenSample", "_anchor_old_predictions", "_feature_tree_manifest", "_samples",
    "build_prediction_archive",
    "resolve_current_audit", "run_category_sensitivity_forward",
]
