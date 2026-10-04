from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccg.repairs import candidates as repair
from ccg.repairs import atomic as atomic_io


def test_true_category_candidates_are_sorted_and_keep_frozen_fillers() -> None:
    categories = np.asarray([-1, 3, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2], dtype=np.int64)
    valid = np.arange(1, categories.size, dtype=np.int64)
    old_tail = np.asarray([3, 4, 5, 7, 9, 10, 11, 12], dtype=np.int64)
    hard = repair.build_versioned_candidate_indices(
        target_index=0, valid_indices=valid, proposal_categories=categories,
        true_target_category=1, old_hard_order=old_tail, spec=repair.CELL_SPECS["hard5"],
    )
    dose = repair.build_versioned_candidate_indices(
        target_index=0, valid_indices=valid, proposal_categories=categories,
        true_target_category=1, old_hard_order=old_tail,
        spec=repair.CELL_SPECS["expb_m4"],
    )
    assert hard is not None and hard.tolist() == [0, 2, 4, 6, 8]
    assert dose is not None and dose.tolist() == [0, 2, 4, 6, 8, 3, 5, 7, 9, 10]
    assert 0 not in valid


def test_candidate_builder_rejects_target_inside_distractor_pool() -> None:
    with pytest.raises(ValueError, match="must not occur"):
        repair.build_versioned_candidate_indices(
            target_index=1, valid_indices=[1, 2, 3, 4, 5],
            proposal_categories=[0, 1, 1, 1, 1, 2], true_target_category=1,
            old_hard_order=[2, 3, 4, 5], spec=repair.CELL_SPECS["hard5"],
        )


def test_candidate_builder_marks_insufficient_true_category_availability() -> None:
    candidates = repair.build_versioned_candidate_indices(
        target_index=0, valid_indices=[1, 2, 3],
        proposal_categories=[0, 1, 1, 2], true_target_category=1,
        old_hard_order=[1, 2, 3], spec=repair.CELL_SPECS["hard5"],
    )
    assert candidates is None


def test_max_iou_object_assignment_can_disagree_with_referring_target_category() -> None:
    assignment = repair.assign_proposals_to_objects(
        boxes=[[0, 0, 10, 10], [0, 0, 10, 10]],
        gt_boxes=[[0, 0, 8, 10], [1, 0, 10, 10]],
        gt_categories=[1, 2], gt_object_ids=[101, 202],
    )
    # The target proposal overlaps both objects, but the category comes from the
    # higher-IoU object (category 2) even though the referring ann is category 1.
    assert assignment["category_id"].tolist() == [2, 2]
    assert assignment["object_id"].tolist() == [202, 202]
    assert assignment["best_iou"][0] == pytest.approx(0.9)


def test_source_intersection_reports_attrition_and_four_dose_common_cohort() -> None:
    attrition = repair.availability_intersection_counts([10, 11, 12, 13], [11, 12, 14])
    assert attrition["historical_old_available_rows"] == 4
    assert attrition["corrected_true_category_available_rows"] == 3
    assert attrition["old_new_common_rows"] == 2
    assert attrition["old_only_rows_lost_under_corrected_category"] == 2
    assert attrition["corrected_only_rows_not_in_historical_cohort"] == 1
    assert attrition["old_common_retention"] == pytest.approx(0.5)
    assert attrition["new_common_retention"] == pytest.approx(2 / 3)
    common = repair.dose_common_cohort({
        "expb_m0": [1, 2, 3, 4], "expb_m2": [2, 3, 4],
        "expb_m4": [2, 4, 5], "expb_m8": [2, 4, 6],
    })
    assert common == [2, 4]


def test_geometry_distinguishes_true_category_and_counts_proposals_vs_objects() -> None:
    result = repair.candidate_geometry(
        [0, 1, 2, 3],
        boxes=np.asarray([[0, 0, 4, 4], [0, 0, 4, 4], [5, 5, 7, 7], [8, 8, 9, 9]], dtype=np.float32),
        objectness=[0.9, 0.8, 0.7, 0.6],
        proposal_object_ids=[10, 10, 12, -1],
        proposal_best_iou=[0.8, 0.8, 0.6, 0.1],
        proposal_categories=[3, 1, 1, 2], target_index=0,
        true_target_category_id=1, true_target_ann_id=10,
        target_proposal_object_id=10, target_gt_box=[0, 0, 4, 4], image_area=100,
    )
    assert result["n_proposals"] == 4
    assert result["n_distinct_matched_gt_objects"] == 2
    assert result["duplicate_excess_proposals"] == 1
    assert result["n_unmatched_to_coco_object_iou_ge_0_5"] == 1
    assert result["n_proposals_assigned_true_target_category"] == 2
    assert result["n_proposals_assigned_target_proposal_category"] == 1
    assert result["n_proposals_assigned_true_target_object"] == 2


def test_supplemental_manifest_is_schema_compatible_and_immutable(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    base = root / "results/research_repair_v1/input_manifest.json"
    coco = root / "data/raw/annotations/instances_train2014.json"
    base.parent.mkdir(parents=True)
    coco.parent.mkdir(parents=True)
    base.write_text(json.dumps({"schema": "research-repair-input-manifest-v1", "n_files": 794, "inputs": []}))
    coco.write_bytes(b"supplemental-test-input")
    monkeypatch.setattr(repair, "ROOT", root)
    monkeypatch.setattr(repair, "BASE_MANIFEST_PATH", base)
    monkeypatch.setattr(repair, "COCO_PATH", coco)
    manifest_path = root / "results/research_repair_v1/candidates/supplemental_input_manifest.json"
    created = repair.prepare_supplemental_input_manifest(manifest_path=manifest_path)
    original_bytes = manifest_path.read_bytes()
    assert created["inputs"] == [{
        "path": "data/raw/annotations/instances_train2014.json",
        "size_bytes": len(b"supplemental-test-input"),
        "sha256": repair._sha256(coco),
    }]
    assert created["source_path_gap"]["prepared_expected_path_exists"] is False
    assert repair.prepare_supplemental_input_manifest(manifest_path=manifest_path) == created
    assert manifest_path.read_bytes() == original_bytes
    coco.write_bytes(b"changed")
    with pytest.raises(repair.CandidateAuditError, match="verification failed"):
        repair.verify_supplemental_input_manifest(manifest_path=manifest_path)
    assert manifest_path.read_bytes() == original_bytes


def test_csv_writer_atomically_replaces_existing_file_and_preserves_it_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "candidate_geometry_sensitivity_per_expression.csv"
    path.write_text("old,artifact\nkeep,this\n", encoding="utf-8")
    repair._write_csv(path, [{"id": 7, "value": "new"}], ["id", "value"])
    assert path.read_text(encoding="utf-8") == "id,value\n7,new\n"
    assert list(tmp_path.glob(".*.tmp")) == []

    original = path.read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError(22, "synthetic replace failure")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)
    with pytest.raises(atomic_io.AtomicReplaceError, match="synthetic replace failure"):
        repair._write_csv(path, [{"id": 8, "value": "not committed"}], ["id", "value"])
    assert path.read_bytes() == original
    retained = list(tmp_path.glob(".*.tmp"))
    assert len(retained) == 1
    assert retained[0].read_text(encoding="utf-8") == "id,value\n8,not committed\n"


def _sharing_violation(code: int = 32) -> PermissionError:
    error = PermissionError(13, "synthetic Windows sharing violation")
    error.winerror = code
    return error


def test_atomic_csv_writer_retries_transient_winerror_32_and_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "retry.csv"
    path.write_text("old,value\n1,old\n", encoding="utf-8")
    real_replace = atomic_io.os.replace
    calls = []
    delays = []

    def transient_twice_then_replace(source: Path, target: Path) -> None:
        calls.append((source, target))
        if len(calls) <= 2:
            raise _sharing_violation(32)
        real_replace(source, target)

    monkeypatch.setattr(atomic_io.os, "replace", transient_twice_then_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)
    repair._write_csv(path, [{"id": 7, "value": "new"}], ["id", "value"])
    assert path.read_text(encoding="utf-8") == "id,value\n7,new\n"
    assert len(calls) == 3
    assert delays == [0.1, 0.2]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_csv_writer_exhaustion_retains_temp_and_original(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "locked.csv"
    original = b"old,value\n1,old\n"
    path.write_bytes(original)
    delays = []

    def always_locked(_source: Path, _target: Path) -> None:
        raise _sharing_violation(33)

    monkeypatch.setattr(atomic_io.os, "replace", always_locked)
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)
    with pytest.raises(atomic_io.AtomicReplaceError, match="temporary artifact retained") as caught:
        repair._write_csv(path, [{"id": 8, "value": "recovery"}], ["id", "value"])
    error = caught.value
    assert error.winerror == 33
    assert error.attempts == 6
    assert len(delays) == 5
    assert delays == [0.1, 0.2, 0.4, 0.8, 1.6]
    assert path.read_bytes() == original
    assert error.source.is_file()
    assert error.source.read_text(encoding="utf-8") == "id,value\n8,recovery\n"


def test_candidate_audit_run_directory_is_unique_and_contained(tmp_path: Path) -> None:
    root = tmp_path / "candidates"
    output = repair.candidate_audit_run_dir("try_4", base_dir=root)
    assert output == (root / "audit_run_try_4").resolve()
    assert output.parent == root.resolve()
    with pytest.raises(ValueError, match="1-64 ASCII"):
        repair.candidate_audit_run_dir("../outside", base_dir=root)
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        repair.candidate_audit_run_dir("try_4", base_dir=root)


def test_current_audit_pointer_is_hash_pinned_and_ignores_incomplete_runs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = repo / "results/research_repair_v1/candidates"
    complete_dir = root / "audit_run_complete"
    complete_dir.mkdir(parents=True)
    complete_summary = {
        "schema": "research-repair-v1-candidate-summary-v1",
        "status": "AUDIT_COMPLETE_MISMATCH_FOUND",
    }
    repair._write_json(complete_dir / "summary.json", complete_summary)
    pointer = repair.write_current_audit_pointer(
        complete_dir / "summary.json", base_dir=root, repo_root=repo
    )
    assert pointer["audit_run_id"] == "complete"
    assert pointer["summary_path"] == "results/research_repair_v1/candidates/audit_run_complete/summary.json"
    assert pointer["summary_sha256"] == repair._sha256(complete_dir / "summary.json")
    pointer_path = root / "current_audit.json"
    original_pointer = pointer_path.read_bytes()

    failed_dir = root / "audit_run_failed"
    failed_dir.mkdir()
    repair._write_json(failed_dir / "summary.json", {
        "schema": "research-repair-v1-candidate-summary-v1", "status": "RUNNING",
    })
    with pytest.raises(repair.CandidateAuditError, match="not complete"):
        repair.write_current_audit_pointer(
            failed_dir / "summary.json", base_dir=root, repo_root=repo
        )
    assert pointer_path.read_bytes() == original_pointer
    assert json.loads(pointer_path.read_text(encoding="utf-8")) == pointer
