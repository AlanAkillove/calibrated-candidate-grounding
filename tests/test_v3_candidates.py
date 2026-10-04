from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccg.v3.candidates import (
    CandidateCell,
    FineCopsRow,
    controlled_candidate_cells,
    load_cohort,
    natural_candidate_cells,
)


def _row(expr_id: str = "71") -> FineCopsRow:
    return FineCopsRow(
        record_id=f"finecops:train:{expr_id}",
        expr_id=expr_id,
        sentence_id=int(expr_id) if expr_id.isdigit() else 99,
        image_id=123,
        gqa_image_id="123",
        source_split="train",
        expression="the red cup",
        gt_boxxyxy=np.asarray([0, 0, 10, 10], dtype=np.float32),
        actual_image_path=Path("image.jpg"),
    )


def test_load_canonical_cohort_preserves_source_identity_and_expression(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.jsonl"
    cohort.write_text(
        json.dumps({
            "expr_id": "71", "sentence_id": "71", "image_id": "123", "gqa_image_id": "123",
            "gt_boxxyxy": [0, 0, 10, 12], "expression": "  the cup  ", "level": "2",
            "source_split": "train", "actual_image_path": "images/123.jpg",
        }) + "\n",
        encoding="utf-8",
    )
    rows = load_cohort(cohort, expected_split="train", root=tmp_path)
    assert len(rows) == 1
    assert rows[0].record_id == "finecops:train:71"
    assert rows[0].expression == "  the cup  "
    assert rows[0].actual_image_path == (tmp_path / "images/123.jpg").resolve()
    np.testing.assert_array_equal(rows[0].gt_boxxyxy, [0, 0, 10, 12])


def test_duplicate_source_qualified_expression_is_rejected(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.jsonl"
    row = {
        "expr_id": "71", "sentence_id": 71, "image_id": "123", "gt_boxxyxy": [0, 0, 10, 10],
        "expression": "cup", "source_split": "train", "actual_image_path": "123.jpg",
    }
    cohort.write_text("\n".join(json.dumps(row) for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source-qualified expression"):
        load_cohort(cohort, root=tmp_path)


def test_natural_prefix_uses_stable_identity_keeps_all_targets_and_reports_counts() -> None:
    boxes = np.asarray([
        [20, 20, 30, 30],  # 0 non-target, tied highest objectness
        [0, 0, 10, 10],    # 1 correct
        [0, 0, 10, 10],    # 2 another correct answer, tied with 0
        [40, 40, 50, 50],  # 3 non-target
        [0, 0, 10, 10],    # 4 invalid crop even though geometrically correct
    ], dtype=np.float32)
    objectness = np.asarray([0.9, 0.8, 0.9, 0.7, 1.0], dtype=np.float32)
    valid = np.asarray([True, True, True, True, False])
    cells = natural_candidate_cells(boxes, objectness, valid, np.asarray([0, 0, 10, 10]), ks=(1, 2, 4))
    np.testing.assert_array_equal(cells[1].candidate_indices, [0])
    np.testing.assert_array_equal(cells[2].candidate_indices, [0, 2])
    np.testing.assert_array_equal(cells[4].candidate_indices, [0, 2, 1, 3])
    assert cells[1].target_present is False
    assert cells[1].bank_valid_target_count == 2
    assert cells[1].presented_valid_target_count == 0
    assert cells[2].presented_valid_target_count == 1
    assert cells[4].presented_valid_target_count == 2
    assert cells[1].missing_reason == "no_valid_target_in_top_k"
    assert cells[4].target_proposal_index == 1  # stable original identity breaks equal-IoU ties


def test_natural_small_and_empty_candidate_sets_have_no_padding() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [20, 20, 25, 25]], dtype=np.float32)
    partial = natural_candidate_cells(boxes, np.asarray([0.7, 0.3]), np.asarray([True, False]), boxes[0], ks=(5,))
    assert partial[5].k_eff == 1
    np.testing.assert_array_equal(partial[5].candidate_indices, [0])
    empty = natural_candidate_cells(boxes, np.asarray([0.7, 0.3]), np.asarray([False, False]), boxes[0], ks=(5,))
    assert empty[5].k_eff == 0
    assert empty[5].target_present is False
    assert empty[5].missing_reason == "no_valid_proposals"
    assert empty[5].candidate_indices.size == 0


def test_controlled_candidates_choose_one_target_remove_equivalents_and_nest() -> None:
    boxes = np.asarray([
        [0, 0, 10, 10],   # target; IoU=1.0
        [0, 0, 8, 10],    # equivalent target; IoU=.8, removed
        [20, 20, 30, 30],
        [30, 30, 40, 40],
        [40, 40, 50, 50],
        [50, 50, 60, 60],
        [60, 60, 70, 70],
    ], dtype=np.float32)
    valid = np.ones(7, dtype=bool)
    scores = np.linspace(0.9, 0.3, 7, dtype=np.float32)
    cells = controlled_candidate_cells(_row(), boxes, scores, valid, boxes[0], ks=(2, 3, 5, 7))
    for k, cell in cells.items():
        assert cell.target_proposal_index == 0
        assert cell.candidate_indices[0] == 0
        assert cell.presented_valid_target_count == 1
        assert cell.bank_valid_target_count == 2
        assert not np.any(np.isin(cell.candidate_indices, [1]))
        assert cell.k_eff == min(k, 6)  # target plus five legal distractors; no fake padding
    np.testing.assert_array_equal(cells[2].candidate_indices, cells[3].candidate_indices[:2])
    np.testing.assert_array_equal(cells[3].candidate_indices, cells[5].candidate_indices[:3])
    np.testing.assert_array_equal(cells[5].candidate_indices, cells[7].candidate_indices[:5])
    assert cells[7].missing_reason == "insufficient_control_distractors"


def test_controlled_noninteger_expression_uses_deterministic_hash_seed() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [20, 20, 30, 30], [30, 30, 40, 40]], dtype=np.float32)
    scores = np.asarray([0.8, 0.6, 0.4], dtype=np.float32)
    row = _row("expression-key")
    left = controlled_candidate_cells(row, boxes, scores, np.ones(3, bool), boxes[0], ks=(2, 3))
    right = controlled_candidate_cells(row, boxes, scores, np.ones(3, bool), boxes[0], ks=(2, 3))
    np.testing.assert_array_equal(left[3].candidate_indices, right[3].candidate_indices)
    np.testing.assert_array_equal(left[2].candidate_indices, left[3].candidate_indices[:2])
