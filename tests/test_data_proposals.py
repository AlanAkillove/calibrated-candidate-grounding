"""Proposal geometry, unique-target assignment, audits and the cache roundtrip."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.data.proposals import (
    TargetAssignment,
    assign_target,
    audit_examples,
    box_iou,
    dump_audit_json,
    feature_ref,
    hdf5_available,
    iou_distribution,
    iou_matrix,
    iter_banks,
    list_image_ids,
    natural_miss_rate,
    proposal_recall,
    read_all_banks,
    read_bank,
    read_features,
    remove_proposals,
    summarize_proposal_quality,
    target_max_ious,
    write_bank,
    write_banks,
    xywh_to_xyxy,
    xyxy_to_xywh,
)
from ccg.data.types import ProposalBank, ReferringExample

GT = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)

BANK_BOXES = np.array(
    [
        [0.0, 0.0, 10.0, 10.0],   # row 0: IoU 1.0      -> target
        [1.0, 1.0, 11.0, 11.0],   # row 1: IoU 81/119   -> equivalent, must be removed
        [5.0, 5.0, 15.0, 15.0],   # row 2: IoU 25/175   -> distractor
        [0.0, 0.0, 20.0, 20.0],   # row 3: IoU 100/400  -> distractor
    ],
    dtype=np.float32,
)


def make_bank(n: int = 4, image_id: int = 7, with_gt: bool = True):
    objectness = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)[:n]
    gt_ious = np.array([1.0, 0.68, 0.14, 0.25], dtype=np.float32)[:n] if with_gt else None
    gt_assignment = np.array([11, 11, 12, 13], dtype=np.int64)[:n] if with_gt else None
    return ProposalBank(
        image_id=image_id,
        boxes=BANK_BOXES[:n],
        objectness=objectness,
        gt_ious=gt_ious,
        gt_assignment=gt_assignment,
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_iou_known_values():
    assert box_iou(GT, GT) == pytest.approx(1.0)
    assert box_iou(GT, BANK_BOXES[1]) == pytest.approx(81.0 / 119.0, abs=1e-5)
    assert box_iou(GT, BANK_BOXES[2]) == pytest.approx(25.0 / 175.0, abs=1e-5)
    disjoint = np.array([100.0, 100.0, 110.0, 110.0], dtype=np.float32)
    assert box_iou(GT, disjoint) == pytest.approx(0.0)
    # matrix form: [N, M]
    matrix = iou_matrix(np.stack([GT, GT]), BANK_BOXES)
    assert matrix.shape == (2, 4)
    np.testing.assert_allclose(matrix[0], matrix[1], atol=1e-7)


def test_box_conversions_roundtrip():
    xywh = np.array([[5.0, 6.0, 10.0, 20.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    xyxy = xywh_to_xyxy(xywh)
    np.testing.assert_allclose(xyxy, [[5, 6, 15, 26], [0, 0, 1, 1]])
    np.testing.assert_allclose(xyxy_to_xywh(xyxy), xywh)


# ---------------------------------------------------------------------------
# unique target assignment (section 4)
# ---------------------------------------------------------------------------
def test_assign_target_picks_max_iou_and_lists_equivalents():
    assignment = assign_target(make_bank(), GT, iou_thresh=0.5)
    assert isinstance(assignment, TargetAssignment)
    assert assignment.target_proposal_idx == 0
    assert assignment.best_iou == pytest.approx(1.0)
    assert assignment.equivalent_indices.tolist() == [0, 1]
    assert assignment.to_remove.tolist() == [1], "the second IoU>=0.5 proposal must go"
    assert assignment.is_miss is False
    assert assignment.num_equivalent == 2
    payload = assignment.to_dict()
    assert payload["target_proposal_idx"] == 0 and payload["to_remove"] == [1]


def test_assign_target_natural_miss_has_no_target():
    bank = ProposalBank(
        image_id=8,
        boxes=BANK_BOXES[2:],  # best IoU 0.25
        objectness=np.array([0.7, 0.6], dtype=np.float32),
    )
    assignment = assign_target(bank, GT, iou_thresh=0.5)
    assert assignment.target_proposal_idx is None
    assert assignment.is_miss is True
    assert assignment.best_iou == pytest.approx(0.25, abs=1e-5)
    assert assignment.equivalent_indices.size == 0 and assignment.to_remove.size == 0
    assert assignment.to_dict()["target_proposal_idx"] is None


def test_assign_target_threshold_is_respected():
    bank = make_bank()
    strict = assign_target(bank, GT, iou_thresh=0.9)
    assert strict.equivalent_indices.tolist() == [0] and strict.to_remove.tolist() == []
    loose = assign_target(bank, GT, iou_thresh=0.1)
    assert set(loose.equivalent_indices.tolist()) >= {0, 1, 3}


def test_assign_target_tie_break_uses_objectness_then_row():
    tied_boxes = np.stack([BANK_BOXES[0], BANK_BOXES[2], BANK_BOXES[0].copy()])
    low_objectness = ProposalBank(
        image_id=9, boxes=tied_boxes, objectness=np.array([0.1, 0.5, 0.9], dtype=np.float32)
    )
    # rows 0 and 2 both have IoU 1.0; the detector's favourite (row 2) wins
    assert assign_target(low_objectness, GT).target_proposal_idx == 2

    flat = ProposalBank(
        image_id=10, boxes=tied_boxes, objectness=np.array([0.5, 0.5, 0.5], dtype=np.float32)
    )
    # identical objectness -> smallest row index, so the choice stays deterministic
    assert assign_target(flat, GT).target_proposal_idx == 0


def test_assign_target_uses_the_configured_default_threshold():
    bank = make_bank()
    # row 1 sits at IoU 0.68, so it is an equivalent under 0.5 but not under 0.9
    assert assign_target(bank, GT).iou_thresh == pytest.approx(0.5)
    assert assign_target(bank, GT).equivalent_indices.tolist() == [0, 1]
    assert assign_target(bank, GT, iou_thresh=0.75).equivalent_indices.tolist() == [0]


def test_remove_proposals_returns_remap():
    bank = make_bank()
    assignment = assign_target(bank, GT)
    cleaned, remap = remove_proposals(bank, assignment.to_remove)
    assert cleaned.N == 3
    assert remap.tolist() == [0, -1, 1, 2], "dropped row maps to -1, others shift down"
    np.testing.assert_allclose(cleaned.boxes[1], bank.boxes[2])
    assert cleaned.gt_ious is not None and cleaned.gt_ious.size == 3
    with pytest.raises(ValueError, match="out of range"):
        remove_proposals(bank, [99])
    with pytest.raises(ValueError, match="empty the proposal bank"):
        remove_proposals(bank, range(bank.N))


# ---------------------------------------------------------------------------
# audits (pure functions called by scripts/audit_proposals.py)
# ---------------------------------------------------------------------------
def make_arrays():
    """Two expressions: one covered by the bank, one missed."""
    good_bank = BANK_BOXES.copy()
    bad_bank = np.array(
        [[50.0, 50.0, 60.0, 60.0], [51.0, 51.0, 61.0, 61.0], [52.0, 52.0, 62.0, 62.0]],
        dtype=np.float32,
    )
    return [GT, GT], [good_bank, bad_bank]


def test_recall_and_natural_miss_are_complementary():
    gts, banks = make_arrays()
    ious = target_max_ious(gts, banks)
    assert ious.shape == (2,)
    assert ious[0] == pytest.approx(1.0)
    assert ious[1] == pytest.approx(0.0)

    recall = proposal_recall(gts, banks, iou_thresh=0.5)
    miss = natural_miss_rate(gts, banks, iou_thresh=0.5)
    assert recall == pytest.approx(0.5)
    assert recall + miss == pytest.approx(1.0)


def test_top_k_truncation_lowers_recall():
    # the only useful proposal sits in row 3, i.e. outside a top-2 bank
    boxes = np.array(
        [
            [50.0, 50.0, 60.0, 60.0],
            [51.0, 51.0, 61.0, 61.0],
            [90.0, 90.0, 99.0, 99.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    assert proposal_recall([GT], [boxes], iou_thresh=0.5, top_k=2) == pytest.approx(0.0)
    assert proposal_recall([GT], [boxes], iou_thresh=0.5, top_k=4) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="aligned"):
        target_max_ious([GT], [boxes, boxes])
    with pytest.raises(ValueError, match="no examples"):
        proposal_recall([], [])


def test_iou_distribution_counts_and_quantiles():
    report = iou_distribution([1.0, 0.68, 0.14, 0.25])
    assert report["num_examples"] == 4
    assert report["bin_edges"] == [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    assert sum(report["counts"]) == 4
    assert sum(report["fractions"]) == pytest.approx(1.0)
    assert 0.0 <= report["mean"] <= 1.0
    assert set(report["quantiles"]) == {"q10", "q25", "q50", "q75", "q90"}
    assert report["quantiles"]["q50"] == pytest.approx(0.465, abs=1e-3)
    assert report["frac_ge_0.5"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no values"):
        iou_distribution([])


def test_summarize_proposal_quality_reports_every_k():
    gts, banks = make_arrays()
    summary = summarize_proposal_quality(gts, banks, iou_thresh=0.5, top_ks=(2, 4))
    assert summary["num_examples"] == 2
    assert summary["recall_by_k"]["4"] == pytest.approx(0.5)
    # top-2 still covers the first expression (row 0 is the target), so the miss
    # rate is 1/2 there and 1/2 for the full banks as well
    assert summary["natural_miss_rate_by_k"]["2"] == pytest.approx(0.5)
    assert summary["recall_by_k"]["2"] == pytest.approx(0.5)
    assert summary["proposal_counts"] == [4, 3]
    assert summary["max_iou_distribution"]["num_examples"] == 2
    equivalents = summary["num_equivalent_target_proposals"]
    # banks: [2 proposals >= 0.5, 0 proposals >= 0.5] -> mean 1.0, max 2
    assert equivalents["max"] == 2 and equivalents["mean"] == pytest.approx(1.0)
    assert equivalents["frac_examples_with_multiple"] == pytest.approx(0.5)


def test_audit_examples_counts_missing_banks_instead_of_dropping_them():
    examples = [
        ReferringExample(ref_id=1, image_id=7, text="a", gt_box=GT, split="val"),
        ReferringExample(ref_id=2, image_id=7, text="b", gt_box=GT, split="val"),
        ReferringExample(ref_id=3, image_id=999, text="c", gt_box=GT, split="testA"),
    ]
    banks = {7: make_bank()}
    report = audit_examples(examples, banks, iou_thresh=0.5, top_ks=(4,))
    assert report["missing_banks"] == 1
    assert report["num_requested_examples"] == 3
    assert report["num_examples"] == 2
    # sequence form (aligned by image_id) gives the same answer
    report_seq = audit_examples(examples[:2], [make_bank()], top_ks=(4,))
    assert report_seq["num_examples"] == 2
    with pytest.raises(ValueError, match="no examples"):
        audit_examples([], {})


def test_dump_audit_json(tmp_path):
    gts, banks = make_arrays()
    path = dump_audit_json(summarize_proposal_quality(gts, banks), tmp_path / "audit.json")
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["num_examples"] == 2


# ---------------------------------------------------------------------------
# cache: HDF5 preferred, NPZ fallback
# ---------------------------------------------------------------------------
def test_hdf5_write_read_roundtrip(tmp_path):
    assert hdf5_available(), "h5py is a declared dependency of this project"
    path = tmp_path / "proposals.h5"
    features = {
        7: np.linspace(-1.0, 1.0, 4 * 8, dtype=np.float32).reshape(4, 8),
        8: np.linspace(1.0, -1.0, 4 * 8, dtype=np.float32).reshape(4, 8),
    }
    write_banks(path, [make_bank(7), make_bank(4, image_id=8)], features=features, mode="w")

    assert list_image_ids(path) == [7, 8]
    bank = read_bank(path, 7)
    assert bank.image_id == 7 and bank.N == 4
    np.testing.assert_allclose(bank.boxes, BANK_BOXES, atol=1e-6)
    np.testing.assert_allclose(bank.gt_ious, [1.0, 0.68, 0.14, 0.25], atol=1e-6)
    assert bank.gt_assignment.tolist() == [11, 11, 12, 13]

    # features stay on disk: the dataclass only carries a handle
    assert bank.feature_ref == feature_ref(path, 7)
    assert "features" in bank.feature_ref
    assert not hasattr(bank, "features") and not hasattr(bank, "proposal_features")
    stored = read_features(path, 7)
    assert stored.shape == (4, 8) and stored.dtype == np.float32
    np.testing.assert_allclose(stored, features[7], atol=2e-3)  # float16 on disk

    banks = read_all_banks(path)
    assert sorted(banks) == [7, 8]
    order = [b.image_id for b in iter_banks(path)]
    assert order == [7, 8]
    assert read_bank(path).image_id == 7, "no image_id means the first stored image"

    # appending a third image keeps the earlier ones readable
    write_bank(path, make_bank(4, image_id=12), mode="a")
    assert list_image_ids(path) == [7, 8, 12]

    with pytest.raises(KeyError):
        read_bank(path, 424242)
    with pytest.raises(FileNotFoundError):
        list_image_ids(tmp_path / "missing.h5")


def test_hdf5_bank_without_features_has_no_ref(tmp_path):
    path = tmp_path / "plain.h5"
    write_bank(path, make_bank(), mode="w")
    bank = read_bank(path)
    assert bank.feature_ref is None, "no features dataset -> the handle must stay None"
    with pytest.raises(KeyError):
        read_features(path, bank.image_id)
    with pytest.raises(ValueError, match="features must have shape"):
        write_bank(path, make_bank(), np.zeros((2, 4), dtype=np.float32), mode="a")


def test_npz_fallback_roundtrip(tmp_path):
    path = tmp_path / "proposals.npz"
    bank = make_bank(4, image_id=21)
    features = np.eye(4, dtype=np.float32)
    write_bank(path, bank, features, mode="w")
    write_bank(path, make_bank(4, image_id=22), mode="a")

    assert list_image_ids(path) == [21, 22]
    restored = read_bank(path, 21)
    np.testing.assert_allclose(restored.boxes, bank.boxes, atol=1e-6)
    assert restored.gt_ious is not None and restored.gt_assignment is not None
    assert restored.feature_ref == feature_ref(path, 21)
    np.testing.assert_allclose(read_features(path, 21), features, atol=2e-3)
    assert read_bank(path, 22).feature_ref is None
    with pytest.raises(KeyError):
        read_features(path, 22)


def test_backend_detection_and_errors(tmp_path):
    path = tmp_path / "bank.bin"
    with pytest.raises(ValueError, match="cannot infer"):
        write_bank(path, make_bank())
    with pytest.raises(ValueError, match="unknown backend"):
        write_bank(tmp_path / "x.h5", make_bank(), backend="zarr")
    # an explicit backend wins over the extension
    np_path = tmp_path / "forced.npz"
    write_bank(np_path, make_bank(image_id=5), mode="w", backend="npz")
    assert list_image_ids(np_path, backend="npz") == [5]
