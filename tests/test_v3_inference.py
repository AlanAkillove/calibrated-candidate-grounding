from __future__ import annotations

import json
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ccg.v3.candidates import FineCopsRow
from ccg.v3.inference import (
    CONFIG_PATH,
    _cell_metadata,
    _batched_inference_rows,
    _load_bank_data,
    _predict_sds_batch,
    build_candidate_manifest,
    load_candidate_manifest,
    save_candidate_manifest,
    validate_candidate_manifest_against_sources,
    validate_cohort_geometry,
    cohort_bbox_boundary_report,
    validate_config,
    validate_config_file,
    validate_exposure_manifest,
)
from ccg.reliability.features import NormalizationFit
from ccg.v3.exposure import fingerprint_image


def _row(path: Path | None = None) -> FineCopsRow:
    return FineCopsRow(
        record_id="finecops:train:71",
        expr_id="71",
        sentence_id=71,
        image_id=123,
        gqa_image_id="123",
        source_split="train",
        expression="the red cup",
        gt_boxxyxy=np.asarray([0, 0, 10, 10], dtype=np.float32),
        actual_image_path=path or Path("image.jpg"),
    )


def test_frozen_config_matches_protocol_and_rejects_any_natural_force() -> None:
    config = validate_config_file(CONFIG_PATH)
    assert config["candidate_seed"] == 20260927
    assert config["K"] == [5, 10, 20, 50]
    changed = dict(config)
    changed["natural_force_target"] = True
    with pytest.raises(ValueError, match="natural_force_target"):
        validate_config(changed)
    changed = dict(config)
    changed["natural_primary_filter_N_ge_50"] = True
    with pytest.raises(ValueError, match="natural_primary_filter_N_ge_50"):
        validate_config(changed)


def test_candidate_manifest_round_trip_and_source_recomputation(tmp_path: Path) -> None:
    row = _row()
    boxes = np.asarray([
        [0, 0, 10, 10], [0, 0, 8, 10], [20, 20, 30, 30], [30, 30, 40, 40],
        [40, 40, 50, 50], [50, 50, 60, 60], [60, 60, 70, 70],
    ], dtype=np.float32)
    objectness = np.asarray([0.99, 0.95, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    valid = np.ones(7, dtype=bool)
    bank = {123: (boxes, objectness)}
    manifest = build_candidate_manifest([row], bank, {123: valid})
    path = save_candidate_manifest(tmp_path / "candidates.jsonl", manifest)
    loaded = load_candidate_manifest(path, [row])
    validate_candidate_manifest_against_sources([row], loaded, bank, {123: valid})
    assert loaded[row.record_id]["natural"][5].size == 5
    assert loaded[row.record_id]["controlled"][50].size == 6

    modified = json.loads(path.read_text(encoding="utf-8").strip())
    changed_order = [6, 1, 2, 3, 4, 0, 5]
    for key in ("5", "10", "20", "50"):
        k = int(key)
        modified["candidate_indices"]["natural"][key] = changed_order[:k]
    path.write_text(json.dumps(modified) + "\n", encoding="utf-8")
    altered = load_candidate_manifest(path, [row])
    with pytest.raises(ValueError, match="differs from frozen source construction"):
        validate_candidate_manifest_against_sources([row], altered, bank, {123: valid})


def test_candidate_manifest_requires_nested_rows(tmp_path: Path) -> None:
    row = _row()
    one = {
        "record_id": row.record_id,
        "sentence_id": row.sentence_id,
        "image_id": row.image_id,
        "candidate_indices": {
            mode: {"5": [0, 1], "10": [0, 2, 1], "20": [0, 1], "50": [0, 1]}
            for mode in ("natural", "controlled")
        },
    }
    path = tmp_path / "candidates.jsonl"
    path.write_text(json.dumps(one) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not nested"):
        load_candidate_manifest(path, [row])


def test_released_gt_box_overhang_is_reported_and_preserved_without_clamping(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (20, 12), color="white").save(image_path)
    row = _row(image_path)
    row = FineCopsRow(**{**row.__dict__, "gt_boxxyxy": np.asarray([1, 2, 20, 12], dtype=np.float32)})
    image_sizes = validate_cohort_geometry([row])
    assert image_sizes == {123: (20, 12)}
    overhang = FineCopsRow(**{**row.__dict__, "gt_boxxyxy": np.asarray([1, 2, 20.01, 12], dtype=np.float32)})
    overhang_sizes = validate_cohort_geometry([overhang])
    report = cohort_bbox_boundary_report([overhang], overhang_sizes)
    assert report["n_rows_with_boundary_overhang"] == 1
    assert report["maximum_overhang_pixels_by_side"]["right"] > 0
    assert report["raw_released_boxes_preserved"] is True
    assert report["ground_truth_clipping_or_rescaling"] is False
    np.testing.assert_array_equal(overhang.gt_boxxyxy, np.asarray([1, 2, 20.01, 12], dtype=np.float32))
    disjoint = FineCopsRow(**{**row.__dict__, "gt_boxxyxy": np.asarray([21, 2, 22, 3], dtype=np.float32)})
    with pytest.raises(ValueError, match="degenerate or disjoint"):
        validate_cohort_geometry([disjoint])


def test_exposure_manifest_binds_actual_image_bytes_and_eligibility(tmp_path: Path) -> None:
    image_path = tmp_path / "123.jpg"
    Image.new("RGB", (20, 12), color=(25, 30, 35)).save(image_path)
    fingerprint = fingerprint_image(image_path)
    manifest = tmp_path / "exposure.csv"
    fields = [
        "source_split", "gqa_image_id", "image_path", "file_sha256", "pixel_sha256", "dhash64",
        "width", "height", "final_eligible", "image_audited", "model_trained", "proposal_scored",
    ]
    row_values = {
        "source_split": "train", "gqa_image_id": "123", "image_path": str(image_path),
        "file_sha256": fingerprint.file_sha256, "pixel_sha256": fingerprint.pixel_sha256,
        "dhash64": fingerprint.dhash64, "width": fingerprint.width, "height": fingerprint.height,
        "final_eligible": "True", "image_audited": "True", "model_trained": "False",
        "proposal_scored": "False",
    }
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row_values)
    row = _row(image_path)
    result = validate_exposure_manifest([row], [manifest])
    assert result["status"] == "PASS"
    assert result["n_cohort_images"] == 1

    row_values["model_trained"] = "True"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row_values)
    with pytest.raises(ValueError, match="prohibited pre-inference exposure"):
        validate_exposure_manifest([row], [manifest])


def test_cell_metadata_distinguishes_bank_targets_from_presented_targets() -> None:
    row = _row()
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 9, 10], [20, 20, 30, 30]], dtype=np.float32)
    metadata = _cell_metadata(row, "natural", 1, np.asarray([2]), boxes, np.ones(3, dtype=bool))
    assert metadata["bank_valid_target_count"] == 2
    assert metadata["valid_target_count"] == 2
    assert metadata["presented_valid_target_count"] == 0
    assert metadata["bank_target_coverage"] is True
    assert metadata["target_coverage"] is False
    assert metadata["candidate_indices"] == [2]


class _ToySDS:
    def predict_proba(self, scores, mask, log_k, z_top1):
        z = (scores * mask).sum(axis=1) / mask.sum(axis=1)
        return 1.0 / (1.0 + np.exp(-(z + log_k * 0.1 + z_top1 * 0.01)))


def test_score_deepsets_batch_and_single_row_helpers_match() -> None:
    fit = NormalizationFit(np.asarray([0.0]), np.asarray([1.0]), ("value",))
    state = (_ToySDS(), 0.0, 1.0, fit, fit)
    scores = np.asarray([[0.9, 0.6, 0.4, 0.2, 0.1], [1.2, 0.3, 0.2, 0.1, 0.0]], dtype=np.float32)
    batch = _predict_sds_batch(state, scores)
    singles = np.asarray([_predict_sds_batch(state, row[None, :])[0] for row in scores])
    np.testing.assert_allclose(batch, singles, rtol=0, atol=0)


def test_confirmation_cli_hard_refuses_without_parent_gates() -> None:
    from scripts.v3_run_pipeline import _infer, _prepare

    prepare_args = SimpleNamespace(
        stage="confirmation", cohort=Path("missing_cohort.jsonl"), config=CONFIG_PATH,
        exposure=[], authorization=None, device="cuda",
    )
    with pytest.raises(ValueError, match="parent-issued authorization"):
        _prepare(prepare_args)
    infer_args = SimpleNamespace(stage="confirmation", freeze=None, ledger=None)
    with pytest.raises(ValueError, match="freeze and run ledger"):
        _infer(infer_args)


def test_native_zero_proposal_sidecar_flows_as_failed_grounding_without_padding(tmp_path: Path) -> None:
    from ccg.data.bank import finalize_bank, write_bank_entry
    from ccg.v3.candidates import REQUESTED_KS

    bank_path = tmp_path / "proposals.h5"
    import h5py

    with h5py.File(bank_path, "a") as handle:
        write_bank_entry(
            handle, 123, np.asarray([[0, 0, 10, 10]], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32), attrs_extra={"n_raw_post_nms": 1},
        )
    finalize_bank(bank_path, {
        "model_name": "frozen-test-rpn", "weights": "COCO_V1", "proposal_type": "native_nms",
        "top_n": 64, "torchvision_version": "test", "images_root": "test",
    })
    sidecar = tmp_path / "empty_proposal_images.jsonl"
    sidecar.write_text(json.dumps({
        "image_id": 124, "failure_type": "native_zero_proposals", "failure_reason": "test empty RPN output",
    }) + "\n", encoding="utf-8")
    bank = _load_bank_data(bank_path)
    assert bank[124][0].shape == (0, 4)
    assert bank[124][1].shape == (0,)

    row = FineCopsRow(
        record_id="finecops:train:72", expr_id="72", sentence_id=72, image_id=124,
        gqa_image_id="124", source_split="train", expression="the empty image",
        gt_boxxyxy=np.asarray([0, 0, 10, 10], dtype=np.float32), actual_image_path=Path("124.jpg"),
    )
    empty = {k: np.empty(0, dtype=np.int64) for k in REQUESTED_KS}
    assignments = {row.record_id: {"natural": empty, "controlled": empty}}

    class _EmptyImageCache:
        def region_features(self, image_id):
            raise AssertionError("zero-proposal images must not request region-cache rows")

        def text_features(self, sentence_id):
            return np.ones(512, dtype=np.float32)

    predictions, raw_rows = _batched_inference_rows(
        [row], assignments, {124: bank[124]},
        {"b0": _EmptyImageCache(), "b16": _EmptyImageCache()},
        {124: (20, 20)}, {}, {},
    )
    assert len(predictions) == len(raw_rows) == 2 * len(REQUESTED_KS) * 2 * 3
    assert all(item["k_eff"] == 0 and item["candidate_indices"] == [] for item in predictions)
    assert all(item["correct"] is None and item["winner_proposal_index"] is None for item in predictions)
    assert all(item["confidence"] == {} and set(item["confidence_missing_reason"]) == {
        "MSP", "S", "S+Q", "S+V", "Full", "SDS_small", "SDS_large"
    } for item in predictions if item["backbone"] == "b0")
    assert all(row.size == 0 for row in raw_rows)


def test_development_acceptance_grid_checks_lossless_rows_and_basic_flow() -> None:
    from scripts.v3_run_pipeline import _validate_development_prediction_grid
    from ccg.v3.candidates import REQUESTED_KS

    record_ids = ["finecops:train:71", "finecops:train:72", "finecops:train:73"]
    valid_counts = {record_ids[0]: 7, record_ids[1]: 3, record_ids[2]: 0}
    predictions = []
    raw = []
    offsets = [0]
    for mode in ("natural", "controlled"):
        for k in REQUESTED_KS:
            for record_id in record_ids:
                ids = list(range(min(k, valid_counts[record_id])))
                bank_target_count = int(valid_counts[record_id] > 0)
                presented_target_count = int(bool(ids) and bank_target_count)
                for backbone in ("b0", "b16"):
                    for seed in (1, 2, 3):
                        models = {"MSP", "S", "S+Q", "S+V", "Full"}
                        if backbone == "b0":
                            models |= {"SDS_small", "SDS_large"}
                        missing_reason = (
                            "empty_candidate_set" if not ids else "k_eff_below_5"
                        )
                        prediction = {
                            "record_id": record_id, "mode": mode, "requested_k": k,
                            "backbone": backbone, "seed": f"b3_seed{seed}", "source_split": "train",
                            "raw_logits_row": len(predictions), "candidate_indices": ids, "k_eff": len(ids),
                            "valid_target_count": bank_target_count,
                            "bank_valid_target_count": bank_target_count,
                            "presented_valid_target_count": presented_target_count,
                            "target_present": bool(presented_target_count),
                            "target_coverage": bool(presented_target_count),
                            "bank_target_coverage": bool(bank_target_count),
                            "correct": None if not ids else True,
                            "winner_proposal_index": None if not ids else ids[0],
                            "confidence": {name: 0.5 for name in models} if len(ids) >= 5 else {},
                            "confidence_missing_reason": {} if len(ids) >= 5 else {
                                name: missing_reason for name in models
                            },
                        }
                        predictions.append(prediction)
                        raw.extend([0.1] * len(ids))
                        offsets.append(len(raw))
    result = _validate_development_prediction_grid(record_ids, predictions, np.asarray(raw), np.asarray(offsets))
    assert result == {
        "prediction_rows": 3 * 2 * 4 * 2 * 3,
        "candidate_cells": 3 * 2 * 4,
        "raw_logit_values": len(raw),
        "empty_candidate_rows": 2 * 4 * 2 * 3,
        "basic_grounding_only_rows": 2 * 4 * 2 * 3,
    }
