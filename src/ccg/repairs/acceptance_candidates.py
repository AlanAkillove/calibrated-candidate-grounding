"""Read-only validation of the frozen candidate-forward/sensitivity evidence chain.

This module intentionally validates metadata, identities, hashes, and NPZ member
names only. It never resamples or loads large bootstrap arrays. The caller's raw
replicate verifier must still inspect the stored arrays and confirm their lengths.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FORMAL_REPLICATES = 5000
FORMAL_SEED = 0
FORMAL_CI = 0.95
RESAMPLE_UNIT = "image_cluster"
SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
CONDITIONS = ("old", "random", "true_target_category_v1")
PREDICTION_FIELDS = ("scores", "correct", "conf_msp", "conf_stats", "conf_e1b")
ROW_FIELDS = ("sentence_id", "ref_id", "image_id", "ann_id", "eval_split", "cell", "k")
ROW_ID_FIELDS = ("sentence_id", "ref_id", "image_id", "ann_id", "eval_split")
EVAL_SPLITS = ("pooled_testA_testB", "testA", "testB")
DOSE_CELLS = ("expb_m0", "expb_m2", "expb_m4", "expb_m8")
EXPECTED_FORWARD_SOURCES = 8
EXPECTED_SOURCE_KEYS = {
    ("RPN", "phase1f_frozen_candidate_archive", "hard5"),
    ("RPN", "phase1f_frozen_candidate_archive", "hard10"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m0"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m2"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m4"),
    ("RPN", "phase1f_frozen_candidate_archive", "expb_m8"),
    ("RPN", "v2_p1_frozen_hard_k5_predictions", "hard5"),
    ("DETR", "v2_p1_frozen_hard_k5_predictions", "hard5"),
}
EXPECTED_SENSITIVITY_GROUPS = 15
EXPECTED_SOURCE_SPLIT_ROWS = 24
EXPECTED_ESTIMATE_ROWS = 1404


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_source_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    values = tuple(row.get(field) for field in ("family", "candidate_source", "cell"))
    if any(not isinstance(value, str) or not value for value in values):
        return None
    return values  # type: ignore[return-value]


class _Validator:
    def __init__(self, root: Path, out: Path) -> None:
        self.root = root.resolve()
        self.out = out.resolve()
        self.failures: list[str] = []
        self.source_sha256: dict[str, str] = {}
        self._hash_cache: dict[Path, str] = {}

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.fail(message)

    def resolve(
        self,
        value: Any,
        label: str,
        *,
        base: Path | None = None,
        boundary: Path | None = None,
    ) -> Path | None:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            self.fail(f"{label}: missing path")
            return None
        try:
            candidate = Path(value)
            resolved = candidate.resolve() if candidate.is_absolute() else ((base or self.root) / candidate).resolve()
            if boundary is not None:
                resolved.relative_to(boundary.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            self.fail(f"{label}: invalid or out-of-scope path {value!s} ({type(exc).__name__})")
            return None
        return resolved

    def display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except (OSError, ValueError):
            return str(path)

    def hash_file(self, path: Path | None, label: str, expected: Any = None) -> str | None:
        if path is None:
            return None
        resolved = path.resolve()
        if not resolved.is_file():
            self.fail(f"{label}: missing file {self.display_path(resolved)}")
            return None
        try:
            digest = self._hash_cache.get(resolved)
            if digest is None:
                digest = _sha256(resolved)
                self._hash_cache[resolved] = digest
            self.source_sha256[self.display_path(resolved)] = digest
        except OSError as exc:
            self.fail(f"{label}: cannot hash {self.display_path(resolved)} ({type(exc).__name__})")
            return None
        if expected is not None:
            if not isinstance(expected, str) or len(expected) != 64 or digest.lower() != expected.lower():
                self.fail(f"{label}: SHA256 mismatch for {self.display_path(resolved)}")
        return digest

    def read_json(self, path: Path | None, label: str) -> dict[str, Any]:
        if path is None:
            return {}
        if not path.is_file():
            self.fail(f"{label}: missing JSON {self.display_path(path)}")
            return {}
        try:
            raw = path.read_bytes()
            self.source_sha256[self.display_path(path)] = hashlib.sha256(raw).hexdigest()
            payload = json.loads(raw.decode("utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.fail(f"{label}: unreadable JSON {self.display_path(path)} ({type(exc).__name__})")
            return {}
        if not isinstance(payload, dict):
            self.fail(f"{label}: JSON root must be an object")
            return {}
        return payload

    def check_npz_members(self, path: Path | None, label: str, expected_keys: Iterable[str]) -> set[str]:
        if path is None or not path.is_file():
            if path is not None:
                self.fail(f"{label}: missing NPZ {self.display_path(path)}")
            return set()
        try:
            with zipfile.ZipFile(path, "r") as archive:
                members = set(archive.namelist())
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            self.fail(f"{label}: invalid NPZ/ZIP {self.display_path(path)} ({type(exc).__name__})")
            return set()
        missing = sorted(f"{key}.npy" for key in set(expected_keys) if f"{key}.npy" not in members)
        if missing:
            self.fail(f"{label}: missing NPZ members {missing[:8]}" + (" …" if len(missing) > 8 else ""))
        return members

    def formal_contract(self, method: Any, label: str) -> bool:
        if not isinstance(method, Mapping):
            self.fail(f"{label}: missing bootstrap contract")
            return False
        ok = (
            type(method.get("n_replicates")) is int
            and method.get("n_replicates") == FORMAL_REPLICATES
            and type(method.get("seed")) is int
            and method.get("seed") == FORMAL_SEED
            and method.get("ci_level") == FORMAL_CI
            and method.get("resample_unit") == RESAMPLE_UNIT
            and method.get("method") == "percentile"
        )
        if not ok:
            self.fail(
                f"{label}: expected n=5000, seed=0, ci=.95, image_cluster, percentile; "
                f"got n={method.get('n_replicates')!r}, seed={method.get('seed')!r}, "
                f"ci={method.get('ci_level')!r}, unit={method.get('resample_unit')!r}, "
                f"method={method.get('method')!r}"
            )
        return ok


def _expected_group_contract(source_keys: set[tuple[str, str, str]]) -> tuple[set[str], dict[tuple[str, str, str, str], str]]:
    group_names: set[str] = set()
    mapping: dict[tuple[str, str, str, str], str] = {}
    dose_sources: dict[tuple[str, str], set[str]] = {}
    for family, source, cell in source_keys:
        if cell in DOSE_CELLS:
            dose_sources.setdefault((family, source), set()).add(cell)
        else:
            for split in EVAL_SPLITS:
                group = f"{family}__{source}__{cell}__{split}"
                group_names.add(group)
                mapping[(family, source, cell, split)] = group
    for (family, source), cells in dose_sources.items():
        if cells != set(DOSE_CELLS):
            continue
        for split in EVAL_SPLITS:
            group = f"{family}__{source}__dose_joint__{split}"
            group_names.add(group)
            for cell in DOSE_CELLS:
                mapping[(family, source, cell, split)] = group
    return group_names, mapping


def _validate_candidate_chain_impl(root: Path, out: Path) -> dict[str, Any]:
    """Validate audit→forward→manifest→formal sensitivity metadata without mutation.

    Raw NPZ file hashes and seed-key names are verified here. The caller's raw
    replicate verifier remains responsible for reading each array and confirming
    that its first dimension is exactly 5000; this function returns that deferral
    explicitly and never loads bootstrap arrays into memory.
    """
    try:
        root_path = Path(root).resolve()
        out_path = Path(out)
        if not out_path.is_absolute():
            out_path = root_path / out_path
        validator = _Validator(root_path, out_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "status": "INCOMPLETE", "failures": [f"Invalid root/output path ({type(exc).__name__})"],
            "source_sha256": {}, "raw_array_length_validation": "DEFERRED_TO_ROOT_RAW_REPLICATE_VERIFIER",
        }

    v = validator
    try:
        v.out.relative_to(v.root)
    except ValueError:
        v.fail("Output directory must be inside repository root")

    pointer_path = (v.out / "candidates" / "current_audit.json").resolve()
    pointer = v.read_json(pointer_path, "current candidate audit pointer")
    v.require(pointer.get("schema") == "research-repair-v1-current-audit-pointer-v1",
              "Candidate pointer schema mismatch")
    pointer_status = pointer.get("status")
    if pointer_status not in ("AUDIT_COMPLETE_MISMATCH_FOUND", "AUDIT_COMPLETE_NO_MISMATCH"):
        v.fail(f"Candidate pointer has unsupported status {pointer_status!r}")
    audit_path = v.resolve(
        pointer.get("summary_path"), "candidate audit summary path", base=v.root,
        boundary=v.out / "candidates",
    )
    audit_hash = v.hash_file(audit_path, "candidate audit summary", pointer.get("summary_sha256"))
    audit = v.read_json(audit_path, "candidate audit summary")
    v.require(audit.get("schema") == "research-repair-v1-candidate-summary-v1",
              "Candidate audit summary schema mismatch")
    audit_run_id = pointer.get("audit_run_id")
    v.require(bool(audit_run_id) and audit.get("audit_run_id") == audit_run_id,
              "Candidate audit pointer/run identity mismatch")
    v.require(audit.get("status") == pointer_status,
              "Candidate audit status differs from current pointer")

    mismatch = audit.get("mismatch")
    mismatch_count: int | None = None
    if not isinstance(mismatch, Mapping) or "total_mismatches" not in mismatch:
        v.fail("Candidate audit mismatch count is missing; it cannot default to zero")
    else:
        raw_count = mismatch.get("total_mismatches")
        if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0:
            mismatch_count = raw_count
        else:
            v.fail(f"Candidate audit mismatch count is not a non-negative integer: {raw_count!r}")
    if mismatch_count is not None:
        if pointer_status == "AUDIT_COMPLETE_MISMATCH_FOUND":
            v.require(mismatch_count > 0, "Audit says mismatch-found but total_mismatches is not positive")
        else:
            v.require(mismatch_count == 0, "Audit says no-mismatch but total_mismatches is not zero")

    historical_scope = audit.get("historical_candidate_source_scope", {})
    if not isinstance(historical_scope, Mapping):
        historical_scope = {}
        v.fail("Candidate audit historical candidate source scope is malformed")
    audit_sources_payload = historical_scope.get("sources", [])
    audit_source_keys: list[tuple[str, str, str]] = []
    if isinstance(audit_sources_payload, list):
        for row in audit_sources_payload:
            key = _as_source_key(row) if isinstance(row, Mapping) else None
            if key is None:
                v.fail("Candidate audit has a malformed historical source identity")
            else:
                audit_source_keys.append(key)
    else:
        v.fail("Candidate audit historical source list is missing")
    audit_source_set = set(audit_source_keys)
    v.require(len(audit_source_keys) == len(audit_source_set), "Candidate audit source identities are duplicated")
    v.require(audit_source_set == EXPECTED_SOURCE_KEYS,
              "Candidate audit source inventory differs from the frozen 8-source acceptance scope")

    needs_sensitivity = pointer_status == "AUDIT_COMPLETE_MISMATCH_FOUND" or (mismatch_count or 0) > 0
    result_extra: dict[str, Any] = {
        "audit_run_id": audit_run_id,
        "mismatch_count": mismatch_count,
        "sensitivity_required": needs_sensitivity,
        "raw_array_length_validation": "DEFERRED_TO_ROOT_RAW_REPLICATE_VERIFIER",
        "raw_array_length_contract": {
            "expected_replicates_per_aggregate_and_seed_key": FORMAL_REPLICATES,
            "seed_labels": list(SEEDS),
            "validated_here": False,
        },
    }
    if not needs_sensitivity:
        result_extra["candidate_sensitivity"] = "NOT_REQUIRED_ZERO_MISMATCH"
        return _report(v, result_extra)

    if mismatch_count is None:
        v.fail("Candidate sensitivity cannot be conditionally skipped with an unknown mismatch count")
    v.require(audit.get("candidate_variant") == "true_target_category_v1"
              and audit.get("candidate_variant_constructed") is True,
              "Positive mismatch requires the versioned true-target-category candidate construction")

    if audit_path is None:
        return _report(v, result_extra)
    forward_dir = audit_path.parent / "forward"
    forward_path = forward_dir / "summary.json"
    forward = v.read_json(forward_path, "candidate forward summary")
    forward_hash = v.source_sha256.get(v.display_path(forward_path))
    v.require(forward.get("schema") == "research-repair-v1-candidate-forward-summary-v1",
              "Candidate forward summary schema mismatch")
    v.require(forward.get("status") == "COMPLETE" and forward.get("preflight_only") is False,
              "Candidate forward is incomplete or preflight-only")
    v.require(forward.get("audit_run_id") == audit_run_id,
              "Candidate forward audit_run_id differs from current audit")

    forward_pointer = forward.get("candidate_audit_pointer")
    if not isinstance(forward_pointer, Mapping):
        v.fail("Candidate forward is missing its source audit pointer")
        forward_pointer = {}
    else:
        pointer_ref = v.resolve(forward_pointer.get("path"), "forward audit pointer path", base=v.root,
                                boundary=v.out / "candidates")
        summary_ref = v.resolve(forward_pointer.get("summary_path"), "forward audit summary path", base=v.root,
                                boundary=v.out / "candidates")
        v.require(pointer_ref == pointer_path, "Candidate forward points to another audit pointer")
        v.require(summary_ref == audit_path, "Candidate forward points to another audit summary")
        v.require(forward_pointer.get("summary_sha256") == audit_hash,
                  "Candidate forward audit summary SHA differs from current audit")
        candidate_path = v.resolve(forward_pointer.get("candidate_indices_path"),
                                   "forward candidate index path", base=v.root,
                                   boundary=audit_path.parent)
        v.hash_file(candidate_path, "versioned corrected candidate indices",
                    forward_pointer.get("candidate_indices_sha256"))

    manifest_spec = forward.get("predictions_manifest")
    if not isinstance(manifest_spec, Mapping):
        v.fail("Candidate forward summary lacks predictions_manifest identity")
        manifest_spec = {}
    manifest_path = v.resolve(manifest_spec.get("path"), "candidate predictions manifest path",
                              base=v.root, boundary=forward_dir)
    manifest_hash = v.hash_file(manifest_path, "candidate predictions manifest", manifest_spec.get("sha256"))
    manifest = v.read_json(manifest_path, "candidate predictions manifest")
    v.require(manifest.get("status") == "COMPLETE" and manifest.get("preflight_rows_per_source") is None,
              "Candidate predictions manifest is incomplete or preflight-only")
    v.require(manifest.get("audit_run_id") == audit_run_id,
              "Candidate predictions manifest audit_run_id differs from current audit")
    v.require(manifest.get("candidate_audit_summary_sha256") == audit_hash,
              "Candidate predictions manifest audit summary SHA mismatch")
    v.require(manifest_spec.get("sha256") == manifest_hash,
              "Candidate forward summary manifest SHA does not match the actual manifest")
    v.require(manifest_spec.get("schema") == "research-repair-v1-candidate-forward-manifest-v1",
              "Candidate forward manifest schema mismatch")
    v.require(manifest.get("archive_schema") == "canonical-row-three-seed-old-random-corrected-v1",
              "Candidate forward archive schema mismatch")
    v.require(set(manifest.get("row_fields", [])) >= set(ROW_FIELDS),
              "Candidate forward manifest omits canonical row identity fields")
    v.require(manifest.get("condition_names") == list(CONDITIONS),
              "Candidate forward conditions do not identify old/random/true-category predictions")
    v.require(manifest.get("canonical_row_order") ==
              "ascending sentence_id; each archive holds one source/family/cell old-new availability intersection",
              "Candidate forward manifest row order/cohort contract mismatch")

    if manifest_path is None:
        entries: list[dict[str, Any]] = []
    else:
        entries_payload = manifest.get("entries", [])
        if not isinstance(entries_payload, list):
            v.fail("Candidate predictions manifest entries must be a list")
            entries = []
        else:
            entries = [entry for entry in entries_payload if isinstance(entry, dict)]
            if len(entries) != len(entries_payload):
                v.fail("Candidate predictions manifest contains malformed entries")

    manifest_keys = [_as_source_key(entry) for entry in entries]
    if any(key is None for key in manifest_keys):
        v.fail("Candidate predictions manifest contains an invalid source identity")
    manifest_source_set = {key for key in manifest_keys if key is not None}
    v.require(len(manifest_keys) == len(manifest_source_set), "Candidate predictions manifest source identities are duplicated")
    v.require(len(manifest_source_set) == EXPECTED_FORWARD_SOURCES,
              f"Candidate forward must contain {EXPECTED_FORWARD_SOURCES} source cohorts; found {len(manifest_source_set)}")
    v.require(manifest_source_set == EXPECTED_SOURCE_KEYS,
              "Candidate forward source identities differ from the frozen 8-source acceptance scope")
    v.require(manifest_source_set == audit_source_set,
              "Candidate forward sources differ from the historical candidate-audit source inventory")

    cohort = forward.get("cohort_contract", {})
    if not isinstance(cohort, Mapping):
        v.fail("Candidate forward cohort contract is malformed")
        cohort = {}
    cohort_rows = cohort.get("families_and_cells", [])
    if not isinstance(cohort_rows, list):
        v.fail("Candidate forward families_and_cells must be a list")
        cohort_rows = []
    cohort_keys = [_as_source_key(row) for row in cohort_rows if isinstance(row, Mapping)]
    v.require(cohort.get("source_cohorts_remain_separate") is True
              and "sentence_id" in str(cohort.get("row_order", ""))
              and "no ref_id joins" in str(cohort.get("row_order", "")),
              "Candidate forward cohort contract does not preserve canonical source rows")
    v.require(len(cohort_keys) == EXPECTED_FORWARD_SOURCES and set(cohort_keys) == manifest_source_set,
              "Candidate forward summary cohort identities differ from predictions manifest")

    identity_arrays: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    manifest_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    cohort_by_key = {_as_source_key(row): row for row in cohort_rows if isinstance(row, Mapping)}
    for entry, source_key in zip(entries, manifest_keys):
        if source_key is None:
            continue
        manifest_by_key[source_key] = entry
        contract = cohort_by_key.get(source_key, {})
        v.require(entry.get("status") == "FORWARDED_WITH_OLD_ANCHOR_PASS",
                  f"{source_key}: forward source lacks old-anchor pass")
        for field in ("k", "n_rows", "n_images", "status"):
            if field in contract:
                v.require(entry.get(field) == contract.get(field),
                          f"{source_key}: manifest/forward cohort {field} mismatch")
        anchor_checks = entry.get("anchor_checks", {})
        for seed_name in SEEDS:
            check = anchor_checks.get(seed_name, {}) if isinstance(anchor_checks, Mapping) else {}
            v.require(check.get("passed") is True and check.get("rows") == entry.get("n_rows"),
                      f"{source_key}/{seed_name}: old prediction anchor missing or row-misaligned")
            anchor_status = check.get("confidence_anchor_status")
            expected_anchor = ("PASS_1E-9" if source_key[1] == "phase1f_frozen_candidate_archive"
                               else "PRESERVED_SOURCE_CONFIDENCE_NO_FULL_LOGITS")
            v.require(anchor_status == expected_anchor,
                      f"{source_key}/{seed_name}: unexpected confidence anchor route {anchor_status!r}")
        output = entry.get("output")
        if not isinstance(output, Mapping):
            v.fail(f"{source_key}: forward source has no output artifact identity")
            continue
        output_path = v.resolve(output.get("path"), f"{source_key} prediction archive path",
                                base=v.root, boundary=forward_dir)
        v.hash_file(output_path, f"{source_key} prediction archive", output.get("sha256"))
        if output_path is not None and output_path.is_file():
            try:
                with np.load(output_path, allow_pickle=False) as archive:
                    required_arrays = set(ROW_FIELDS) | {"family", "candidate_source", "condition_names", "seed_names"}
                    missing_arrays = sorted(required_arrays - set(archive.files))
                    if missing_arrays:
                        v.fail(f"{source_key}: prediction archive misses arrays {missing_arrays}")
                    else:
                        ids = np.asarray(archive["sentence_id"]).reshape(-1)
                        n_rows = int(entry.get("n_rows", -1))
                        v.require(ids.size == n_rows, f"{source_key}: archive row count differs from manifest")
                        v.require(ids.size > 0 and np.all(np.diff(ids.astype(np.int64)) > 0),
                                  f"{source_key}: sentence_id rows are not unique ascending canonical IDs")
                        image_ids = np.asarray(archive["image_id"]).reshape(-1)
                        v.require(image_ids.size == n_rows
                                  and int(np.unique(image_ids).size) == int(entry.get("n_images", -1)),
                                  f"{source_key}: image_id count differs from manifest")
                        for field in ROW_ID_FIELDS:
                            values = np.asarray(archive[field]).reshape(-1)
                            v.require(values.size == n_rows,
                                      f"{source_key}: {field} row count differs from manifest")
                        for field, expected_value in (("family", source_key[0]), ("candidate_source", source_key[1]),
                                                      ("cell", source_key[2])):
                            values = np.asarray(archive[field]).astype(str).reshape(-1)
                            v.require(values.size == 1 and values[0] == expected_value,
                                      f"{source_key}: archive scalar {field} label differs from source identity")
                        k_value = np.asarray(archive["k"]).reshape(-1)
                        v.require(k_value.size == 1 and k_value[0] == entry.get("k"),
                                  f"{source_key}: archive scalar K label differs from manifest")
                        split_values = np.asarray(archive["eval_split"]).astype(str).reshape(-1)
                        v.require(split_values.size == n_rows and set(split_values) <= {"testA", "testB"},
                                  f"{source_key}: archive has invalid evaluation split labels")
                        v.require(list(np.asarray(archive["condition_names"]).astype(str).reshape(-1)) == list(CONDITIONS),
                                  f"{source_key}: archive condition identities differ")
                        v.require(list(np.asarray(archive["seed_names"]).astype(str).reshape(-1)) == list(SEEDS),
                                  f"{source_key}: archive seed identities differ")
                        required_prediction_keys = {
                            f"{condition}__{seed_name}__{field}"
                            for condition in CONDITIONS for seed_name in SEEDS for field in PREDICTION_FIELDS
                        }
                        v.require(required_prediction_keys <= set(archive.files),
                                  f"{source_key}: prediction archive lacks old/new seed condition arrays")
                        identity_arrays[source_key] = {
                            field: np.asarray(archive[field]).copy()
                            for field in ("sentence_id", "ref_id", "image_id", "ann_id", "eval_split")
                        }
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                v.fail(f"{source_key}: unreadable forward prediction archive ({type(exc).__name__})")

    # Sensitivity is required only when the real candidate audit found mismatches.
    sensitivity_dir = forward_dir / "sensitivity_bootstrap"
    sensitivity_path = sensitivity_dir / "summary.json"
    sensitivity = v.read_json(sensitivity_path, "formal candidate sensitivity summary")
    sensitivity_hash = v.source_sha256.get(v.display_path(sensitivity_path))
    v.require(sensitivity.get("schema") == "research-repair-v1-candidate-sensitivity-summary-v1",
              "Candidate sensitivity summary schema mismatch")
    v.require(sensitivity.get("status") == "COMPLETE" and sensitivity.get("formal") is True,
              "Candidate sensitivity is not a completed formal run")
    v.require(sensitivity.get("audit_run_id") == audit_run_id,
              "Candidate sensitivity audit_run_id differs from current audit")
    v.require(sensitivity.get("candidate_audit_summary_sha256") == audit_hash,
              "Candidate sensitivity audit summary SHA differs from current audit")
    v.require(sensitivity.get("candidate_forward_summary_sha256") == forward_hash,
              "Candidate sensitivity forward summary SHA differs from actual forward summary")
    v.formal_contract(sensitivity.get("bootstrap"), "Candidate sensitivity bootstrap")
    v.formal_contract(sensitivity.get("bootstrap_contract"), "Candidate sensitivity bootstrap_contract")
    sensitivity_manifest_spec = sensitivity.get("predictions_manifest", {})
    if not isinstance(sensitivity_manifest_spec, Mapping):
        v.fail("Candidate sensitivity predictions_manifest identity is malformed")
        sensitivity_manifest_spec = {}
    v.require(sensitivity_manifest_spec.get("sha256") == manifest_hash,
              "Candidate sensitivity manifest SHA differs from actual forward manifest")
    sensitivity_manifest_path = v.resolve(
        sensitivity_manifest_spec.get("path"), "sensitivity predictions manifest path",
        base=v.root, boundary=forward_dir,
    )
    v.require(sensitivity_manifest_path == manifest_path,
              "Candidate sensitivity consumes a different predictions manifest path")

    forwarded = sensitivity.get("forwarded_source_cohorts", [])
    if not isinstance(forwarded, list):
        v.fail("Candidate sensitivity forwarded_source_cohorts must be a list")
        forwarded = []
    forwarded_keys = {
        tuple(row) for row in forwarded
        if isinstance(row, (list, tuple)) and len(row) == 3 and all(isinstance(x, str) for x in row)
    } if isinstance(forwarded, list) else set()
    v.require(forwarded_keys == manifest_source_set and len(forwarded) == len(forwarded_keys),
              "Candidate sensitivity forwarded source identities differ from the forward manifest")
    v.require(sensitivity.get("nonforwardable_empty_source_cohorts") == [],
              "Candidate sensitivity reports missing/nonforwardable source cohorts")

    expected_groups, expected_mapping = _expected_group_contract(manifest_source_set)
    v.require(len(expected_groups) == EXPECTED_SENSITIVITY_GROUPS,
              f"Derived candidate sensitivity group contract is not {EXPECTED_SENSITIVITY_GROUPS} groups")
    expected_rows = sensitivity.get("expected_source_cohort_split_groups", [])
    seen_source_splits: set[tuple[str, str, str, str]] = set()
    if not isinstance(expected_rows, list):
        v.fail("Candidate sensitivity expected source/split mapping must be a list")
        expected_rows = []
    for row in expected_rows:
        if not isinstance(row, Mapping):
            v.fail("Candidate sensitivity source/split mapping contains malformed row")
            continue
        key = _as_source_key(row)
        split = row.get("eval_split")
        if key is None or split not in EVAL_SPLITS:
            v.fail("Candidate sensitivity source/split mapping contains invalid identity")
            continue
        source_split = (*key, split)
        if source_split in seen_source_splits:
            v.fail(f"Candidate sensitivity source/split identity is duplicated: {source_split}")
        seen_source_splits.add(source_split)
        group_expected = expected_mapping.get(source_split)
        v.require(row.get("group") == group_expected,
                  f"Candidate sensitivity source/split {source_split} maps to wrong group")
        expected_status = ("COMPONENT_INCLUDED_IN_DOSE_JOINT_GROUP_COMPLETE" if key[2] in DOSE_CELLS
                           else "COMPLETE")
        v.require(row.get("status") == expected_status,
                  f"Candidate sensitivity source/split {source_split} is not complete")
    v.require(len(expected_rows) == EXPECTED_SOURCE_SPLIT_ROWS
              and seen_source_splits == set(expected_mapping),
              f"Candidate sensitivity must cover exactly {EXPECTED_SOURCE_SPLIT_ROWS} source/split mappings")

    group_rows = sensitivity.get("groups", [])
    if not isinstance(group_rows, list):
        v.fail("Candidate sensitivity groups must be a list")
        group_rows = []
    group_by_name: dict[str, dict[str, Any]] = {}
    for row in group_rows:
        if not isinstance(row, dict) or not isinstance(row.get("group"), str):
            v.fail("Candidate sensitivity contains a malformed group summary row")
            continue
        if row["group"] in group_by_name:
            v.fail(f"Candidate sensitivity group is duplicated: {row['group']}")
        group_by_name[row["group"]] = row
    v.require(set(group_by_name) == expected_groups and len(group_rows) == EXPECTED_SENSITIVITY_GROUPS,
              f"Candidate sensitivity must contain exactly {EXPECTED_SENSITIVITY_GROUPS} expected groups")
    v.require(sensitivity.get("n_estimates") == EXPECTED_ESTIMATE_ROWS,
              f"Candidate sensitivity summary must declare {EXPECTED_ESTIMATE_ROWS} estimates")

    estimates_path = v.resolve(sensitivity.get("estimates_csv"), "candidate sensitivity estimates CSV",
                               base=sensitivity_dir, boundary=sensitivity_dir)
    estimate_rows: list[dict[str, str]] = []
    if estimates_path is not None and estimates_path.is_file():
        v.hash_file(estimates_path, "candidate sensitivity estimates CSV")
        try:
            with estimates_path.open("r", encoding="utf-8-sig", newline="") as handle:
                estimate_rows = list(csv.DictReader(handle))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            v.fail(f"Candidate sensitivity estimates CSV is unreadable ({type(exc).__name__})")
    else:
        v.fail("Candidate sensitivity estimates CSV is missing")
    v.require(len(estimate_rows) == EXPECTED_ESTIMATE_ROWS,
              f"Candidate sensitivity estimates CSV must have {EXPECTED_ESTIMATE_ROWS} data rows; found {len(estimate_rows)}")

    rows_by_group: dict[str, list[dict[str, str]]] = {}
    for row in estimate_rows:
        rows_by_group.setdefault(row.get("group", ""), []).append(row)

    for group_name in sorted(expected_groups):
        group_row = group_by_name.get(group_name)
        if group_row is None:
            continue
        dose_joint = "__dose_joint__" in group_name
        expected_group_estimates = 260 if dose_joint else 52
        v.require(group_row.get("status") == "COMPLETE",
                  f"{group_name}: group is not COMPLETE")
        v.require(group_row.get("n_estimates") == expected_group_estimates,
                  f"{group_name}: expected {expected_group_estimates} estimates per group")
        actual_rows = rows_by_group.get(group_name, [])
        v.require(len(actual_rows) == expected_group_estimates,
                  f"{group_name}: estimate CSV row count differs from group contract")
        family, source, _, split = group_name.split("__", 3)
        v.require(group_row.get("family") == family and group_row.get("candidate_source") == source
                  and group_row.get("eval_split") == split,
                  f"{group_name}: group summary identity mismatch")

        summary_path = v.resolve(group_row.get("summary_path"), f"{group_name} summary path",
                                 base=sensitivity_dir, boundary=sensitivity_dir)
        summary_file_hash = v.hash_file(summary_path, f"{group_name} group summary",
                                        group_row.get("summary_sha256"))
        group_summary = v.read_json(summary_path, f"{group_name} group summary")
        v.require(group_summary.get("schema") == "research-repair-v1-candidate-sensitivity-group-v1"
                  and group_summary.get("status") == "COMPLETE" and group_summary.get("formal") is True
                  and group_summary.get("group") == group_name,
                  f"{group_name}: group summary is not a matching formal completion")
        v.formal_contract(group_summary.get("bootstrap_contract"), f"{group_name} group bootstrap")
        cohort = group_summary.get("cohort", {})
        if not isinstance(cohort, Mapping):
            v.fail(f"{group_name}: group cohort metadata is malformed")
            cohort = {}
        v.require(cohort.get("family") == family and cohort.get("candidate_source") == source
                  and cohort.get("eval_split") == split
                  and cohort.get("n_rows") == group_row.get("n_rows")
                  and cohort.get("n_images") == group_row.get("n_images"),
                  f"{group_name}: group cohort metadata differs from group index")
        if dose_joint:
            expected_dose_keys = {key for key in manifest_source_set if key[0] == family and key[1] == source and key[2] in DOSE_CELLS}
            cohort_dose_keys = {
                tuple(key) for key in cohort.get("source_cohort_keys", [])
                if isinstance(key, (list, tuple)) and len(key) == 3 and all(isinstance(x, str) for x in key)
            }
            v.require(cohort_dose_keys == expected_dose_keys and len(expected_dose_keys) == 4,
                      f"{group_name}: dose group does not contain exactly the four common source cells")
        else:
            cell = group_name.split("__")[-2]
            source_keys = {
                tuple(key) for key in cohort.get("source_cohort_keys", [])
                if isinstance(key, (list, tuple)) and len(key) == 3 and all(isinstance(x, str) for x in key)
            }
            v.require(source_keys == {(family, source, cell)},
                      f"{group_name}: regular group source identity mismatch")

        raw_path = v.resolve(group_row.get("raw_replicates_path"), f"{group_name} raw NPZ path",
                             base=sensitivity_dir, boundary=sensitivity_dir)
        raw_hash = v.hash_file(raw_path, f"{group_name} raw replicates", group_row.get("raw_replicates_sha256"))
        raw_members = set()
        if raw_path is not None:
            raw_members = v.check_npz_members(raw_path, f"{group_name} raw replicates", ())
        estimate_names: set[str] = set()
        expected_raw_keys: set[str] = set()
        for row in actual_rows:
            estimate_name = row.get("estimate", "")
            if estimate_name in estimate_names:
                v.fail(f"{group_name}: duplicate estimate name {estimate_name}")
            estimate_names.add(estimate_name)
            v.require(row.get("family") == family and row.get("candidate_source") == source
                      and row.get("eval_split") == split,
                      f"{group_name}/{estimate_name}: estimate source identity mismatch")
            if not dose_joint:
                cell = group_name.split("__")[-2]
                v.require(row.get("cell") == cell,
                          f"{group_name}/{estimate_name}: estimate cell mismatch")
            else:
                dose_cell = row.get("cell")
                is_dose_macro = (
                    dose_cell == "dose_macro"
                    and row.get("contrast") == "dose_macro_old_minus_new"
                    and estimate_name.startswith("dose_macro_old_minus_new__dose_macro__")
                )
                v.require(dose_cell in DOSE_CELLS or is_dose_macro,
                          f"{group_name}/{estimate_name}: dose estimate has invalid cell")
            v.require(row.get("n_replicates") == str(FORMAL_REPLICATES),
                      f"{group_name}/{estimate_name}: estimate is pilot or has wrong replicate count")
            v.require(row.get("n_rows") == str(group_row.get("n_rows"))
                      and row.get("n_images") == str(group_row.get("n_images")),
                      f"{group_name}/{estimate_name}: estimate cohort size differs from its group")
            v.require(row.get("raw_replicates_path") == group_row.get("raw_replicates_path")
                      and row.get("raw_replicates_sha256") == raw_hash,
                      f"{group_name}/{estimate_name}: estimate points to a different raw source")
            expected_mean_key = f"bootstrap__mean__{estimate_name}"
            v.require(row.get("raw_replicates_key") == expected_mean_key,
                      f"{group_name}/{estimate_name}: aggregate raw key identity mismatch")
            expected_raw_keys.add(expected_mean_key)
            try:
                seed_keys = json.loads(row.get("per_seed_raw_replicates_keys_json", ""))
            except json.JSONDecodeError:
                seed_keys = None
            expected_seed_keys = [f"bootstrap__{seed_name}__{estimate_name}" for seed_name in SEEDS]
            v.require(seed_keys == expected_seed_keys,
                      f"{group_name}/{estimate_name}: per-seed raw keys are missing, reordered, or mismatched")
            if isinstance(seed_keys, list):
                expected_raw_keys.update(key for key in seed_keys if isinstance(key, str))
        missing_raw_keys = sorted(f"{key}.npy" for key in expected_raw_keys if f"{key}.npy" not in raw_members)
        if missing_raw_keys:
            v.fail(f"{group_name}: raw NPZ is missing expected estimate/seed members {missing_raw_keys[:8]}"
                   + (" …" if len(missing_raw_keys) > 8 else ""))

        summary_estimates = group_summary.get("estimates", {})
        v.require(isinstance(summary_estimates, Mapping) and set(summary_estimates) == estimate_names,
                  f"{group_name}: group summary estimates differ from estimates CSV")
        v.require(cohort.get("n_rows", 0) > 0 and cohort.get("n_images", 0) > 1,
                  f"{group_name}: empty/underspecified image-cluster cohort")

    # Compare the row-identity cohort used by sensitivity against the actual
    # forward archives. This reads only canonical ID/split arrays, never scores.
    dose_by_family_source: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for source_key, arrays in identity_arrays.items():
        if source_key[2] in DOSE_CELLS:
            dose_by_family_source.setdefault((source_key[0], source_key[1]), {})[source_key[2]] = arrays
    for group_name, group_row in group_by_name.items():
        if group_name not in expected_groups:
            continue
        family, source, _, split = group_name.split("__", 3)
        group_summary_path = v.resolve(group_row.get("summary_path"), f"{group_name} identity summary",
                                       base=sensitivity_dir, boundary=sensitivity_dir)
        group_summary = v.read_json(group_summary_path, f"{group_name} identity summary")
        group_cohort = group_summary.get("cohort", {})
        if not isinstance(group_cohort, Mapping):
            v.fail(f"{group_name}: identity cohort metadata is malformed")
            group_cohort = {}
        if "__dose_joint__" in group_name:
            dose_arrays = dose_by_family_source.get((family, source), {})
            if set(dose_arrays) != set(DOSE_CELLS):
                v.fail(f"{group_name}: actual forward archives lack a complete four-level dose source")
                continue
            ids_by_level = {level: np.asarray(dose_arrays[level]["sentence_id"], dtype=np.int64)
                            for level in DOSE_CELLS}
            common = set(ids_by_level[DOSE_CELLS[0]].tolist())
            for level in DOSE_CELLS[1:]:
                common.intersection_update(ids_by_level[level].tolist())
            common_ids = np.asarray(sorted(common), dtype=np.int64)
            base = dose_arrays[DOSE_CELLS[0]]
            base_ids = np.asarray(base["sentence_id"], dtype=np.int64)
            base_positions = np.searchsorted(base_ids, common_ids)
            split_values = np.asarray(base["eval_split"]).astype(str)[base_positions]
            if split == "pooled_testA_testB":
                keep = np.isin(split_values, ("testA", "testB"))
            else:
                keep = split_values == split
            selected_ids = common_ids[keep]
            canonical_hash = hashlib.sha256(selected_ids.astype("<i8", copy=False).tobytes()).hexdigest()
            v.require(group_cohort.get("common_sentence_ids_sha256") == canonical_hash,
                      f"{group_name}: sensitivity dose sentence IDs differ from actual forward intersection")
            v.require(group_row.get("n_rows") == int(selected_ids.size),
                      f"{group_name}: sensitivity dose row count differs from actual forward intersection")
            selected_images = np.asarray(base["image_id"], dtype=np.int64)[base_positions][keep]
            v.require(group_row.get("n_images") == int(np.unique(selected_images).size),
                      f"{group_name}: sensitivity dose image count differs from actual forward intersection")
            for level in DOSE_CELLS[1:]:
                current = dose_arrays[level]
                current_ids = np.asarray(current["sentence_id"], dtype=np.int64)
                positions = np.searchsorted(current_ids, selected_ids)
                v.require(np.all(positions < current_ids.size)
                          and np.array_equal(current_ids[positions], selected_ids),
                          f"{group_name}: dose level {level} is missing common sentence IDs")
                for field in ("ref_id", "image_id", "ann_id", "eval_split"):
                    expected_values = np.asarray(base[field])[base_positions][keep]
                    actual_values = np.asarray(current[field])[positions]
                    v.require(np.array_equal(actual_values, expected_values),
                              f"{group_name}: dose level {level} misaligns {field} on common IDs")
        else:
            cell = group_name.split("__")[-2]
            source_key = (family, source, cell)
            arrays = identity_arrays.get(source_key)
            if arrays is None:
                v.fail(f"{group_name}: sensitivity source has no actual forward archive")
                continue
            ids = np.asarray(arrays["sentence_id"], dtype=np.int64)
            splits = np.asarray(arrays["eval_split"]).astype(str)
            keep = np.isin(splits, ("testA", "testB")) if split == "pooled_testA_testB" else splits == split
            selected_ids = ids[keep]
            canonical_hash = hashlib.sha256(selected_ids.astype("<i8", copy=False).tobytes()).hexdigest()
            v.require(group_cohort.get("canonical_sentence_id_sha256") == canonical_hash,
                      f"{group_name}: sensitivity sentence IDs differ from actual forward source")
            v.require(group_row.get("n_rows") == int(selected_ids.size),
                      f"{group_name}: sensitivity row count differs from actual forward source/split")
            image_ids = np.asarray(arrays["image_id"], dtype=np.int64)[keep]
            v.require(group_row.get("n_images") == int(np.unique(image_ids).size),
                      f"{group_name}: sensitivity image count differs from actual forward source/split")

    return _report(v, result_extra | {
        "forward_source_count": len(manifest_source_set),
        "sensitivity_group_count": len(group_by_name),
        "source_split_mapping_count": len(seen_source_splits),
        "estimate_row_count": len(estimate_rows),
        "forward_manifest_sha256": manifest_hash,
        "candidate_forward_summary_sha256": forward_hash,
        "candidate_sensitivity_summary_sha256": sensitivity_hash,
        "raw_group_sha256_checked": True,
        "raw_seed_keys_checked": True,
        "raw_array_lengths_checked": False,
    })


def validate_candidate_chain(root: Path, out: Path) -> dict[str, Any]:
    """Return failures for absent or malformed evidence instead of raising."""
    try:
        return _validate_candidate_chain_impl(root, out)
    except Exception as exc:
        return {
            "status": "INCOMPLETE",
            "failures": [f"Candidate evidence chain is malformed or unreadable ({type(exc).__name__}: {exc})"],
            "source_sha256": {},
            "raw_array_length_validation": "DEFERRED_TO_ROOT_RAW_REPLICATE_VERIFIER",
        }


def _report(validator: _Validator, extra: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS_METADATA_RAW_LENGTHS_DEFERRED" if not validator.failures else "INCOMPLETE",
        "failures": list(validator.failures),
        "source_sha256": dict(sorted(validator.source_sha256.items())),
        **dict(extra),
    }


__all__ = ["validate_candidate_chain"]
