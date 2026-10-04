"""Frozen candidate preparation and inference for the V3 FineCops study.

Proposal/embedding preparation is separate from frozen scorer inference. The
confirmation command must verify the parent-owned freeze and consume the
already pinned candidate/feature artifacts before it can run any B3 forward.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .candidates import (
    CONTROL_SEED,
    IOU_THRESHOLD,
    REQUESTED_KS,
    CandidateCell,
    FineCopsRow,
    controlled_candidate_cells,
    natural_candidate_cells,
    prediction_correctness,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "results/v3_final_validation/config.json"
DEFAULT_OUTPUT_ROOT = ROOT / "results/v3_final_validation/pipeline"
LEGACY_GQA_BANK = ROOT / "cache/phase1e_finecops/proposals_gqa.h5"
B0_FEATURE_SOURCE = ROOT / "cache/features"
B16_FEATURE_SOURCE = ROOT / "cache/v2_backbones/openclip_b16"
B0_ENCODER_CHECKPOINT = ROOT / "cache/features/metadata.json"
B16_ENCODER_CHECKPOINT = ROOT / "cache/v2_backbones/openclip_b16/metadata.json"
B0_GROUNDING_ROOT = ROOT / "results/phase0b_independent"
B16_GROUNDING_ROOT = ROOT / "results/v2_backbone_generalization/b1_phase0b"
B0_RELIABILITY_ROOT = ROOT / "results/research_repair_v1/information/models"
B16_RELIABILITY_ROOT = ROOT / "results/v3_final_validation/replication/models"
B0_SOURCE_PREDICTIONS = ROOT / "results/research_repair_v1/information/cross_k_predictions.csv.gz"
B0_SOURCE_DERIVED = ROOT / "results/research_repair_v1/information/derived"
B0_RAW_SCORES = ROOT / "results/phase0b_independent"
B0_TEMPERATURE = "results/phase0b_independent/seed_{seed}/eval_metadata.json"
B16_TEMPERATURE = "results/v2_backbone_generalization/b1_phase0b/seed_{seed}/eval_metadata.json"
RPN_CHECKPOINT_NAME = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
RPN_CHECKPOINT_PATH = ROOT / "cache/v3/source_weights/torch_hub/checkpoints" / RPN_CHECKPOINT_NAME
EMPTY_PROPOSAL_FILENAME = "empty_proposal_images.jsonl"
FEATURE_ROOTS = {"b0": B0_FEATURE_SOURCE, "b16": B16_FEATURE_SOURCE}
GROUPS = {"S": "S", "SQ": "S+Q", "SV": "S+V", "Full": "Full"}
GROUP_FILES = {"S": "ablation_S.json", "SQ": "ablation_S_plus_Q.json", "SV": "ablation_S_plus_V.json", "Full": "ablation_Full.json"}
SOURCE_LOGIT_ATOL = 1e-4
SOURCE_PROBABILITY_ATOL = 1e-9


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed if the run config differs from V3's frozen design."""
    if config.get("schema") != "v3-config-v1":
        raise ValueError("unsupported V3 config schema")
    expected = {
        "candidate_seed": CONTROL_SEED,
        "K": list(REQUESTED_KS),
        "target_iou": IOU_THRESHOLD,
        "feature_groups": ["S", "S+Q", "S+V", "Full"],
        "natural_force_target": False,
        "natural_remove_other_targets": False,
        "natural_primary_filter_N_ge_50": False,
        "risk_ties": "fractional_boundary_tie",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"config {name}={config.get(name)!r}, expected {value!r}")
    proposal = config.get("proposal", {})
    if (
        proposal.get("weights") != "COCO_V1"
        or int(proposal.get("top_n", -1)) != 64
        or proposal.get("native_nms") is not True
        or proposal.get("extra_nms_or_dedup") is not False
    ):
        raise ValueError("proposal config differs from frozen COCO_V1/native-NMS/N64 settings")
    stats = config.get("statistics", {})
    if stats.get("risk_ties") != "fractional_boundary_tie":
        raise ValueError("risk tie convention differs from the parent-owned V3 protocol")


def validate_cohort_geometry(rows: Sequence[FineCopsRow]) -> dict[int, tuple[int, int]]:
    """Validate released annotation boxes against actual JPEG dimensions.

    The official COCO-format xywh box is preserved as released. An annotation
    may extend beyond the image edge; it remains valid when its unmodified box
    has positive intersection with the actual JPEG pixel rectangle. The box is
    never clipped or silently reinterpreted.
    """
    from PIL import Image

    sizes: dict[int, tuple[int, int]] = {}
    path_by_id: dict[int, Path] = {}
    errors: list[dict[str, Any]] = []
    for row in rows:
        prior = path_by_id.setdefault(row.image_id, row.actual_image_path)
        if prior.resolve() != row.actual_image_path.resolve():
            errors.append({"record_id": row.record_id, "reason": "conflicting_image_path_for_image_id"})
            continue
        if not row.actual_image_path.is_file():
            errors.append({"record_id": row.record_id, "reason": "missing_image", "path": str(row.actual_image_path)})
            continue
        if row.image_id not in sizes:
            with Image.open(row.actual_image_path) as image:
                sizes[row.image_id] = tuple(map(int, image.size))
        width, height = sizes[row.image_id]
        x1, y1, x2, y2 = map(float, row.gt_boxxyxy)
        intersects_x = max(0.0, min(x2, float(width)) - max(x1, 0.0))
        intersects_y = max(0.0, min(y2, float(height)) - max(y1, 0.0))
        if not (np.isfinite((x1, y1, x2, y2)).all() and x1 < x2 and y1 < y2 and intersects_x > 0 and intersects_y > 0):
            errors.append({
                "record_id": row.record_id,
                "reason": "released_box_degenerate_or_disjoint_from_jpeg_pixel_space",
                "box_xyxy": [x1, y1, x2, y2],
                "image_size_wh": [width, height],
                "intersection_wh": [intersects_x, intersects_y],
            })
    if errors:
        raise ValueError(f"{len(errors)} cohort row(s) have degenerate or disjoint released boxes: {errors[:8]}")
    return sizes


def cohort_bbox_boundary_report(
    rows: Sequence[FineCopsRow], image_sizes: Mapping[int, tuple[int, int]]
) -> dict[str, Any]:
    """Report official boxes that extend beyond actual JPEG boundaries.

    Ground-truth coordinates remain the original released xyxy values. The
    report makes edge overhang visible without changing the evaluation target.
    """
    overhang_rows: list[dict[str, Any]] = []
    n_overhang_rows = 0
    maxima = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    side_counts = {key: 0 for key in maxima}
    for row in rows:
        width, height = image_sizes[row.image_id]
        x1, y1, x2, y2 = map(float, row.gt_boxxyxy)
        excess = {
            "left": max(0.0, -x1),
            "top": max(0.0, -y1),
            "right": max(0.0, x2 - float(width)),
            "bottom": max(0.0, y2 - float(height)),
        }
        for side, value in excess.items():
            if value > 0:
                side_counts[side] += 1
                maxima[side] = max(maxima[side], float(value))
        if any(value > 0 for value in excess.values()):
            n_overhang_rows += 1
            if len(overhang_rows) < 20:
                overhang_rows.append({
                    "record_id": row.record_id,
                    "image_id": row.image_id,
                    "box_xyxy_released": [x1, y1, x2, y2],
                    "image_size_wh": [width, height],
                    "overhang_pixels": excess,
                    "ground_truth_clipped": False,
                })
    return {
        "coordinate_source": "official_positive_COCO_xywh_converted_to_xyxy",
        "image_dimension_source": "actual_decoded_JPEG",
        "n_expression_rows": len(rows),
        "n_rows_with_boundary_overhang": n_overhang_rows,
        "boundary_overhang_side_row_counts": side_counts,
        "maximum_overhang_pixels_by_side": maxima,
        "reported_examples_max_20": overhang_rows,
        "raw_released_boxes_preserved": True,
        "ground_truth_clipping_or_rescaling": False,
        "interpretation": (
            "Official positive boxes with a positive intersection are evaluated unchanged; "
            "IoU uses the released box area and its geometric intersection with each proposal."
        ),
    }


def validate_exposure_manifest(
    rows: Sequence[FineCopsRow], manifests: Sequence[str | Path]
) -> dict[str, Any]:
    """Verify cohort-image eligibility and audited bytes from exposure files."""
    from .exposure import fingerprint_image

    expected: dict[tuple[str, str], Path] = {
        (row.source_split, row.gqa_image_id): row.actual_image_path for row in rows
    }
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    matched_manifest_rows = 0
    duplicate_image_rows = 0
    for path in manifests:
        source = Path(path)
        if source.suffix.lower() in {".jsonl", ".ndjson"}:
            with source.open("r", encoding="utf-8-sig") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        else:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                records = list(csv.DictReader(handle))
        for line, record in enumerate(records, start=1):
            key = (str(record.get("source_split", "")), str(record.get("gqa_image_id", record.get("image_id", ""))))
            if key not in expected:
                continue
            matched_manifest_rows += 1
            if key in observed:
                if dict(record) != observed[key]:
                    raise ValueError(f"exposure manifests disagree on repeated image-level audit record {key}")
                duplicate_image_rows += 1
                continue
            observed[key] = dict(record)
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"exposure manifests do not cover {len(missing)} cohort images: {missing[:8]}")

    def truth(value: Any, *, field: str, key: tuple[str, str]) -> bool:
        if isinstance(value, bool):
            result = value
        else:
            normalized = str(value).strip().lower()
            if normalized in {"true", "1", "yes"}:
                result = True
            elif normalized in {"false", "0", "no"}:
                result = False
            else:
                raise ValueError(f"exposure {key}: {field} is not an explicit boolean: {value!r}")
        return result

    checked: set[tuple[str, str]] = set()
    for key, row_path in expected.items():
        record = observed[key]
        for field in ("final_eligible", "image_audited", "source_identity_eligible", "pixel_audit_eligible"):
            if field in record and not truth(record[field], field=field, key=key):
                raise ValueError(f"exposure {key}: {field}=false")
        if "final_eligible" not in record or "image_audited" not in record:
            raise ValueError(f"exposure {key}: missing final eligibility or image audit flag")
        for field in ("model_trained", "model_scored", "proposal_scored", "reliability_model_scored"):
            if field in record and truth(record[field], field=field, key=key):
                raise ValueError(f"exposure {key}: prohibited pre-inference exposure flag {field}=true")
        audit_path = record.get("image_path", record.get("actual_image_path", ""))
        if audit_path:
            audit_path = Path(str(audit_path))
            if not audit_path.is_absolute():
                audit_path = (ROOT / audit_path).resolve()
            if audit_path.resolve() != row_path.resolve():
                raise ValueError(f"exposure {key}: audited image path differs from canonical cohort image")
        fingerprint = fingerprint_image(row_path)
        expected_file_hash = str(record.get("file_sha256", record.get("image_sha256", ""))).strip()
        expected_pixel_hash = str(record.get("pixel_sha256", record.get("image_pixel_sha256", ""))).strip()
        expected_dhash = str(record.get("dhash64", record.get("image_phash", ""))).strip()
        if not expected_file_hash or not expected_pixel_hash or not expected_dhash:
            raise ValueError(f"exposure {key}: audited file/pixel/perceptual hashes are required")
        if fingerprint.file_sha256 != expected_file_hash:
            raise ValueError(f"exposure {key}: image file SHA256 differs from the audit manifest")
        if fingerprint.pixel_sha256 != expected_pixel_hash or fingerprint.dhash64 != expected_dhash:
            raise ValueError(f"exposure {key}: decoded pixels or perceptual hash differ from the audit manifest")
        if "width" in record and str(record["width"]).strip() and int(record["width"]) != fingerprint.width:
            raise ValueError(f"exposure {key}: audited image width changed")
        if "height" in record and str(record["height"]).strip() and int(record["height"]) != fingerprint.height:
            raise ValueError(f"exposure {key}: audited image height changed")
        checked.add(key)
    def display_path(path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return str(resolved)

    return {
        "status": "PASS",
        "n_cohort_images": len(checked),
        "n_manifest_rows": matched_manifest_rows,
        "n_duplicate_image_rows_with_identical_audit": duplicate_image_rows,
        "image_content_verified": True,
        "manifests": [display_path(path) for path in manifests],
    }


def validate_config_file(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(payload)
    return payload


def unique_images(rows: Sequence[FineCopsRow]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for row in rows:
        prior = result.setdefault(row.image_id, row.actual_image_path)
        if prior.resolve() != row.actual_image_path.resolve():
            raise ValueError(f"GQA image_id {row.image_id} maps to two image paths")
    return dict(sorted(result.items()))


def materialize_rpn_checkpoint() -> Path:
    """Copy the already-cached COCO_V1 bytes into the hashable V3 namespace.

    Torchvision locates its detector checkpoint through ``torch.hub``. Pinning
    a byte-identical workspace copy lets both the preparation authorization
    and final provenance bind the actual weights used, while leaving the
    user's shared model cache untouched.
    """
    import torch

    original_hub = Path(torch.hub.get_dir()).resolve()
    source = original_hub / "checkpoints" / RPN_CHECKPOINT_NAME
    if not source.is_file():
        raise FileNotFoundError(f"frozen COCO_V1 RPN checkpoint is not cached locally: {source}")
    source_hash = sha256_file(source)
    if not source_hash.startswith("258fb6c6"):
        raise ValueError(f"COCO_V1 RPN checkpoint SHA does not match torchvision's published file identity: {source_hash}")
    destination = RPN_CHECKPOINT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise ValueError(f"workspace RPN checkpoint copy differs from the frozen source: {destination}")
    else:
        shutil.copy2(source, destination)
        if sha256_file(destination) != source_hash:
            raise OSError("RPN checkpoint copy failed SHA256 verification")
    torch.hub.set_dir(str(destination.parent.parent))
    return destination


def ensure_proposal_bank(
    rows: Sequence[FineCopsRow],
    output: str | Path,
    *,
    device: str = "cuda",
    log: Callable[[str], None] = print,
) -> Path:
    """Create/resume a V3 bank, copying matching legacy GQA proposals first.

    Every source input remains read-only. The old GQA bank is reused only by
    the same numeric GQA image ID; absent images are extracted with the frozen
    torchvision COCO_V1 RPN and saved in the V3 namespace.
    """
    import torch
    import torchvision

    from ..data.bank import bank_attrs, finalize_bank, image_ids as new_bank_ids, write_bank_entry
    from ..data.proposals import list_image_ids as legacy_image_ids, read_bank as read_legacy_bank
    from ..data.rpn import RPNProposalCountError, build_rpn_model, extract_proposal_bank
    from ..features.clip_encoder import load_rgb_image

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_paths = unique_images(rows)
    materialize_rpn_checkpoint()
    existing = set(int(i) for i in new_bank_ids(destination)) if destination.exists() else set()
    empty_path = destination.parent / EMPTY_PROPOSAL_FILENAME
    empty_records = _read_empty_proposal_records(empty_path) if empty_path.exists() else {}
    if existing & set(empty_records):
        raise ValueError("an image cannot have both a proposal bank entry and an empty-RPN flow record")
    expected_ids = set(image_paths)
    extra_ids = (existing | set(empty_records)) - expected_ids
    if extra_ids:
        raise ValueError(f"V3 proposal bank contains images outside the selected cohort: {sorted(extra_ids)[:10]}")
    legacy_ids: set[int] = set()
    if LEGACY_GQA_BANK.exists():
        legacy_ids = set(int(i) for i in legacy_image_ids(LEGACY_GQA_BANK))
    missing = [image_id for image_id in image_paths if image_id not in existing and image_id not in empty_records]
    to_extract = [image_id for image_id in missing if image_id not in legacy_ids]
    rpn = None
    if to_extract:
        torch.manual_seed(0)
        rpn = build_rpn_model(device=device, weights="COCO_V1")
        model_identity = str(getattr(rpn, "ccg_model_name", "")).lower()
        if model_identity != "fasterrcnn_resnet50_fpn(coco_v1)":
            raise RuntimeError(f"unexpected frozen RPN model identity: {model_identity!r}")
    with __import__("h5py").File(destination, "a") as handle:
        for image_id in missing:
            if image_id in legacy_ids:
                legacy = read_legacy_bank(LEGACY_GQA_BANK, image_id)
                boxes = np.asarray(legacy.boxes, dtype=np.float32)
                objectness = np.asarray(legacy.objectness, dtype=np.float32)
                source = "legacy_phase1e_finecops_gqa_bank"
                if boxes.shape[0] > 64:
                    boxes, objectness = boxes[:64], objectness[:64]
                n_raw = int(boxes.shape[0])
            else:
                assert rpn is not None
                try:
                    bank, meta = extract_proposal_bank(
                        load_rgb_image(image_paths[image_id]), rpn, image_id, top_n=64
                    )
                except RPNProposalCountError as exc:
                    empty_records[image_id] = {
                        "image_id": int(image_id),
                        "source_image_path": image_paths[image_id].resolve().relative_to(ROOT).as_posix(),
                        "proposal_source": "frozen_coco_v1_rpn_native_nms",
                        "failure_type": "native_zero_proposals",
                        "failure_reason": str(exc),
                    }
                    log(f"proposal bank {image_id}: native zero-proposal flow failure retained")
                    continue
                boxes = np.asarray(bank.boxes, dtype=np.float32)
                objectness = np.asarray(bank.objectness, dtype=np.float32)
                source = "frozen_coco_v1_rpn_extraction"
                n_raw = int(meta.get("n_raw_post_nms", boxes.shape[0]))
            if boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes.shape[0] or boxes.shape[0] > 64:
                raise ValueError(f"image {image_id}: malformed frozen proposal block {boxes.shape}")
            if objectness.shape != (boxes.shape[0],) or not np.all(np.isfinite(objectness)):
                raise ValueError(f"image {image_id}: malformed proposal objectness")
            if np.any(np.diff(objectness) > 1e-7):
                raise ValueError(f"image {image_id}: proposal bank is not in stable objectness order")
            write_bank_entry(
                handle,
                image_id,
                boxes,
                objectness,
                attrs_extra={"n_raw_post_nms": max(int(n_raw), int(boxes.shape[0])), "proposal_source": source},
            )
            log(f"proposal bank {image_id}: {boxes.shape[0]} rows from {source}")
    finalize_bank(destination, {
        "model_name": "torchvision Faster R-CNN ResNet50 FPN RPN",
        "weights": "COCO_V1",
        "proposal_type": "class_agnostic_rpn_native_nms_objectness",
        "top_n": 64,
        "torchvision_version": str(torchvision.__version__),
        "images_root": "data/raw/v3_finecops/images",
    })
    _write_empty_proposal_records(empty_path, empty_records)
    attrs = bank_attrs(destination)
    if attrs.get("weights") != "COCO_V1" or int(attrs.get("top_n", -1)) != 64:
        raise ValueError(f"V3 proposal bank metadata differs from frozen COCO_V1/N64: {attrs}")
    if set(int(i) for i in new_bank_ids(destination)) | set(empty_records) != expected_ids:
        raise ValueError("proposal bank plus empty-flow sidecar does not exactly cover the selected cohort images")
    return destination


def _read_empty_proposal_records(path: str | Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    source = Path(path)
    if not source.exists():
        return records
    with source.open("r", encoding="utf-8") as handle:
        for line, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            image_id = int(item["image_id"])
            if item.get("failure_type") != "native_zero_proposals" or image_id in records:
                raise ValueError(f"{source}:{line}: malformed/duplicate zero-proposal flow record")
            records[image_id] = item
    return records


def _write_empty_proposal_records(path: str | Path, records: Mapping[int, Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for image_id in sorted(records):
            handle.write(json.dumps(dict(records[image_id]), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, destination)
    return destination


def ensure_region_cache(
    rows: Sequence[FineCopsRow],
    bank_path: str | Path,
    output: str | Path,
    *,
    backbone: str,
    device: str = "cuda",
    batch_size: int = 64,
    log: Callable[[str], None] = print,
) -> Path:
    """Resume frozen region extraction using the shared exact-crop machinery."""
    from ..data.bank import image_ids as bank_image_ids
    from ..features.cache import StreamingRegionWriter, update_metadata
    from ..features.clip_encoder import build_encoder
    from ..features.extract_regions import InvalidCropLog, iter_region_entries, run_region_extraction

    if backbone == "b0":
        encoder_name, pretrained, checkpoint = (
            "ViT-B-32", "laion2b_s34b_b79k", ROOT / "cache/hf_hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/open_clip_pytorch_model.bin"
        )
        source_metadata = B0_ENCODER_CHECKPOINT
    elif backbone == "b16":
        encoder_name, pretrained, checkpoint = (
            "ViT-B-16", "laion2b_s34b_b88k", ROOT / "cache/hf_local/openclip_b16/open_clip_pytorch_model.bin"
        )
        source_metadata = B16_ENCODER_CHECKPOINT
    else:
        raise ValueError(f"unknown frozen backbone {backbone!r}")
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    expected_hash = str(source["backbone"]["checkpoint_sha256"])
    if sha256_file(checkpoint) != expected_hash:
        raise ValueError(f"{backbone} encoder checkpoint SHA256 differs from its frozen metadata")

    root = Path(output)
    target_ids = [int(i) for i in bank_image_ids(bank_path)]
    writer = StreamingRegionWriter(root / "region_features.h5", resume=True, feature_dim=512)
    cached_ids = sorted(writer.processed)
    missing = [image_id for image_id in target_ids if image_id not in writer.processed]
    if missing and writer.source == "final":
        writer.adopt_final()
    if not missing:
        writer.abandon()
        log(f"{backbone}: region cache already complete for {len(target_ids)} images")
        return root
    encoder = build_encoder(
        device=device,
        precision="fp16",
        cache_dir=ROOT / "cache/hf_hub",
        model_name=encoder_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        download_attempts=0,
    )
    paths = unique_images(rows)
    if set(missing) - set(paths):
        raise ValueError(f"region cache has bank images absent from cohort: {sorted(set(missing) - set(paths))[:10]}")
    invalid_path = root / "invalid_crops.csv"
    invalid_log = InvalidCropLog(invalid_path, resume_keep_ids=set(cached_ids))
    invalid_log.start()
    try:
        stats = run_region_extraction(
            encoder,
            iter_region_entries(bank_path, ROOT / "data/raw/v3_finecops/images", only_ids=set(missing), paths=paths),
            writer,
            batch_size=batch_size,
            on_invalid=invalid_log.add_image,
            log=log,
            total=len(missing),
        )
        writer.finish()
    except BaseException:
        writer.abandon()
        invalid_log.close()
        raise
    invalid_log.close()
    update_metadata(root, {
        "cache_version": "v3-v1",
        "backbone": encoder.metadata(),
        "source_checkpoint_sha256": expected_hash,
        "source_encoder_metadata": str(source_metadata.relative_to(ROOT)),
        "native_logit_scale": encoder.logit_scale,
        "n_images": len(target_ids),
        "extraction_stats": {"region": stats, "invalid_crop_count": invalid_log.n_rows},
    })
    return root


def ensure_text_cache(
    rows: Sequence[FineCopsRow],
    output: str | Path,
    *,
    backbone: str,
    device: str = "cuda",
    batch_size: int = 128,
    log: Callable[[str], None] = print,
) -> Path:
    """Encode official positive expressions into a new, split-local text cache."""
    from ..features.cache import FeatureCache, update_metadata
    from ..features.clip_encoder import build_encoder
    from ..features.extract_text import run_text_extraction

    if backbone == "b0":
        encoder_name, pretrained, checkpoint = (
            "ViT-B-32", "laion2b_s34b_b79k", ROOT / "cache/hf_hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/1a25a446712ba5ee05982a381eed697ef9b435cf/open_clip_pytorch_model.bin"
        )
        source_metadata = B0_ENCODER_CHECKPOINT
    elif backbone == "b16":
        encoder_name, pretrained, checkpoint = (
            "ViT-B-16", "laion2b_s34b_b88k", ROOT / "cache/hf_local/openclip_b16/open_clip_pytorch_model.bin"
        )
        source_metadata = B16_ENCODER_CHECKPOINT
    else:
        raise ValueError(f"unknown frozen backbone {backbone!r}")
    source = json.loads(source_metadata.read_text(encoding="utf-8"))
    expected_hash = str(source["backbone"]["checkpoint_sha256"])
    if sha256_file(checkpoint) != expected_hash:
        raise ValueError(f"{backbone} encoder checkpoint SHA256 differs from its frozen metadata")
    root = Path(output)
    corpus = [
        {
            "sentence_id": int(row.sentence_id),
            "ref_id": int(row.sentence_id),
            "sent_id": 0,
            "image_id": int(row.image_id),
            "split": row.source_split,
            "text": row.expression,
        }
        for row in rows
    ]
    try:
        with FeatureCache.open(root) as cache:
            if set(cache.sentence_ids().tolist()) >= {r.sentence_id for r in rows}:
                return root
    except (FileNotFoundError, KeyError, ValueError):
        pass
    encoder = build_encoder(
        device=device,
        precision="fp16",
        cache_dir=ROOT / "cache/hf_hub",
        model_name=encoder_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        download_attempts=0,
    )
    stats = run_text_extraction(encoder, corpus, root, batch_size=batch_size, resume=True, log=log)
    update_metadata(root, {
        "cache_version": "v3-v1",
        "backbone": encoder.metadata(),
        "source_checkpoint_sha256": expected_hash,
        "source_encoder_metadata": str(source_metadata.relative_to(ROOT)),
        "native_logit_scale": encoder.logit_scale,
        "n_sentences": len(rows),
        "text_stats": stats,
    })
    return root


def ensure_feature_caches(
    rows: Sequence[FineCopsRow],
    bank_path: str | Path,
    stage_root: str | Path,
    *,
    device: str = "cuda",
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Create both exact frozen CLIP caches sequentially (one GPU process)."""
    stage = Path(stage_root)
    result = {}
    for backbone in ("b0", "b16"):
        cache_root = stage / f"features_{backbone}"
        ensure_region_cache(rows, bank_path, cache_root, backbone=backbone, device=device, log=log)
        ensure_text_cache(rows, cache_root, backbone=backbone, device=device, log=log)
        result[backbone] = cache_root
    return result


def _json_row(cell: CandidateCell) -> list[int]:
    return [int(index) for index in np.asarray(cell.candidate_indices, dtype=np.int64)]


def build_candidate_manifest(
    rows: Sequence[FineCopsRow],
    bank_data: Mapping[int, tuple[np.ndarray, np.ndarray]],
    valid_by_image: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    """Construct both candidate regimes once from the same frozen bank."""
    saved: list[dict[str, Any]] = []
    for row in rows:
        boxes, objectness = bank_data[row.image_id]
        valid = np.asarray(valid_by_image[row.image_id], dtype=bool)
        natural = natural_candidate_cells(boxes, objectness, valid, row.gt_boxxyxy)
        controlled = controlled_candidate_cells(
            row, boxes, objectness, valid, row.gt_boxxyxy, seed=CONTROL_SEED
        )
        saved.append({
            "record_id": row.record_id,
            "expr_id": row.expr_id,
            "sentence_id": row.sentence_id,
            "image_id": row.image_id,
            "gqa_image_id": row.gqa_image_id,
            "source_split": row.source_split,
            "candidate_indices": {
                "natural": {str(k): _json_row(natural[k]) for k in REQUESTED_KS},
                "controlled": {str(k): _json_row(controlled[k]) for k in REQUESTED_KS},
            },
        })
    return saved


def save_candidate_manifest(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Atomic JSONL candidate identity artifact; no candidate padding is stored."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, output)
    return output


def load_candidate_manifest(
    path: str | Path, rows: Sequence[FineCopsRow]
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    """Load candidate identities after enforcing exact cohort row alignment."""
    source = Path(path)
    records: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    row_by_id = {row.record_id: row for row in rows}
    with source.open("r", encoding="utf-8") as handle:
        for line, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            record_id = str(item["record_id"])
            if record_id in records:
                raise ValueError(f"{source}:{line}: duplicate candidate identity row {record_id}")
            row = row_by_id.get(record_id)
            if row is None:
                raise ValueError(f"{source}:{line}: unknown source-qualified record_id {record_id}")
            if int(item["sentence_id"]) != row.sentence_id or int(item["image_id"]) != row.image_id:
                raise ValueError(f"{source}:{line}: candidate identity anchors differ for {record_id}")
            by_mode: dict[str, dict[int, np.ndarray]] = {}
            for mode in ("natural", "controlled"):
                raw_ks = item["candidate_indices"][mode]
                cells: dict[int, np.ndarray] = {}
                for k in REQUESTED_KS:
                    indices = np.asarray(raw_ks[str(k)], dtype=np.int64).reshape(-1)
                    if indices.size > k or np.unique(indices).size != indices.size or np.any(indices < 0):
                        raise ValueError(f"{source}:{line}: malformed {mode} K{k} identities for {record_id}")
                    cells[k] = indices
                previous: np.ndarray | None = None
                for k in REQUESTED_KS:
                    current = cells[k]
                    if previous is not None and not np.array_equal(current[: previous.size], previous):
                        raise ValueError(f"{source}:{line}: {mode} candidate identities are not nested")
                    previous = current
                by_mode[mode] = cells
            records[record_id] = by_mode
    expected = [row.record_id for row in rows]
    if list(records) != expected:
        raise ValueError(f"candidate manifest rows/order do not exactly match cohort ({len(records)} vs {len(expected)})")
    return records


def _load_temperature(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = float(payload.get("temperature_corrected", payload.get("T_corrected", np.nan)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid corrected source temperature in {path}: {value}")
    return value


def _load_b0_reliability(seed: int) -> dict[str, Any]:
    from ..reliability import features as rfeat
    from ..repairs.information import load_logistic_predictor
    import torch
    from ..reliability.models import ScoreDeepSets

    seed_root = B0_RELIABILITY_ROOT / f"b3_seed{seed}"
    models: dict[str, Any] = {"temperature": _load_temperature(ROOT / B0_TEMPERATURE.format(seed=seed))}
    for short, filename in GROUP_FILES.items():
        path = seed_root / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        group = GROUPS[short]
        if payload.get("group") != group:
            raise ValueError(f"{path}: expected feature group {group!r}")
        fit = rfeat.NormalizationFit.from_dict(payload["normalization"])
        predictor = load_logistic_predictor(payload)
        if fit.keys != tuple(payload["normalization"]["keys"]) or predictor.coefficients.size != fit.mean.size:
            raise ValueError(f"{path}: predictor/normalizer shape mismatch")
        models[short] = (fit, predictor)
    for short in ("small", "large"):
        regime = "smallK5_10" if short == "small" else "largeK20_50"
        stem = f"score_deepsets_{regime}"
        meta_path, pt_path = seed_root / f"{stem}.json", seed_root / f"{stem}.pt"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
        model = ScoreDeepSets(seed=seed, include_logk=True, include_ztop1=True)
        model._net.load_state_dict(checkpoint["state_dict"])
        model._net.eval()
        score_norm = metadata["score_normalization"]
        logk_fit = rfeat.NormalizationFit.from_dict(metadata["log_k_normalization"])
        top1_fit = rfeat.NormalizationFit.from_dict(metadata["z_top1_normalization"])
        models[f"sds_{short}"] = (model, float(score_norm["mean"]), float(score_norm["std"]), logk_fit, top1_fit)
    return models


def _load_b16_reliability(seed: int) -> dict[str, Any]:
    from ..reliability import features as rfeat
    from ..repairs.information import load_logistic_predictor

    models: dict[str, Any] = {}
    for short, filename in GROUP_FILES.items():
        path = B16_RELIABILITY_ROOT / f"b3_seed{seed}" / f"{GROUPS[short]}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("seed") not in (f"b3_seed{seed}", seed):
            raise ValueError(f"{path}: seed identity mismatch")
        if payload.get("group") != GROUPS[short]:
            raise ValueError(f"{path}: feature group identity mismatch")
        fit = rfeat.NormalizationFit.from_dict(payload["normalization"])
        predictor = load_logistic_predictor(payload["predictor"])
        if tuple(payload["feature_names"]) != fit.keys or predictor.coefficients.size != fit.mean.size:
            raise ValueError(f"{path}: predictor/normalizer shape mismatch")
        models[short] = (fit, predictor)
        models[f"temperature_{short}"] = float(payload["temperature_corrected"])
    temperatures = [models[f"temperature_{short}"] for short in ("S", "SQ", "SV", "Full")]
    if not np.allclose(temperatures, temperatures[0], rtol=0.0, atol=0.0):
        raise ValueError(f"B16 seed{seed}: frozen feature-group temperatures disagree")
    models["temperature"] = float(temperatures[0])
    return models


def _predict_sds(
    model_state: tuple[Any, float, float, Any, Any], raw_scores: np.ndarray
) -> float:
    return float(_predict_sds_batch(model_state, np.asarray(raw_scores).reshape(1, -1))[0])


def _predict_sds_batch(
    model_state: tuple[Any, float, float, Any, Any], raw_scores: np.ndarray
) -> np.ndarray:
    from ..reliability import features as rfeat

    model, mean, std, logk_fit, top1_fit = model_state
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("ScoreDeepSets raw_scores must be a [n,K] matrix")
    k = int(scores.shape[1])
    if k < 5 or not np.isfinite(scores).all():
        raise ValueError("ScoreDeepSets confidence is defined only for K_eff >= 5")
    # The source replay passes each K cell at its actual width. The small-
    # training model was trained with width 10, and the large model at 50,
    # but ScoreDeepSets uses a masked set encoder and accepts wider/narrower
    # candidate sets directly; adding artificial padded columns here would
    # change the exact source forward.
    width = k
    normalized = ((scores - mean) / (std + 1e-8)).astype(np.float32)
    packed = np.zeros((scores.shape[0], width), dtype=np.float32)
    mask = np.zeros((scores.shape[0], width), dtype=bool)
    packed[:, :k] = normalized
    mask[:, :k] = True
    logk = rfeat.normalize_apply(
        np.full((scores.shape[0], 1), np.log(float(k)), dtype=np.float64), logk_fit
    )[:, 0].astype(np.float32)
    top1 = rfeat.normalize_apply(np.max(scores, axis=1)[:, None], top1_fit)[:, 0].astype(np.float32)
    return model.predict_proba(packed, mask, logk, top1)


def _load_bank_data(bank_path: str | Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    from ..data.bank import iter_bank

    result = {int(image_id): (boxes, objectness) for image_id, boxes, objectness in iter_bank(bank_path)}
    sidecar = Path(bank_path).parent / EMPTY_PROPOSAL_FILENAME
    for image_id in _read_empty_proposal_records(sidecar):
        if image_id in result:
            raise ValueError(f"image {image_id} has both proposals and an empty-RPN flow record")
        result[image_id] = (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32))
    return result


def _load_invalid_crop_ids(path: str | Path) -> dict[int, set[int]]:
    from ..features.extract_regions import read_invalid_crop_rows

    result: dict[int, set[int]] = defaultdict(set)
    if Path(path).exists():
        for image_id, proposal_index, *_ in read_invalid_crop_rows(path):
            result[int(image_id)].add(int(proposal_index))
    return result


def _cache_invalid_to_valid(
    rows: Sequence[FineCopsRow],
    feature_roots: Mapping[str, Path],
    bank_data: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    from ..features.cache import FeatureCache

    valid_by_backbone: dict[str, dict[int, np.ndarray]] = {"b0": {}, "b16": {}}
    for backbone, root in feature_roots.items():
        with FeatureCache.open(root) as cache:
            if cache.feature_dim != 512:
                raise ValueError(f"{backbone}: expected 512-d frozen embeddings")
            for image_id in sorted({row.image_id for row in rows}):
                n_proposals = int(bank_data[image_id][0].shape[0])
                if n_proposals == 0:
                    valid_by_backbone[backbone][image_id] = np.zeros(0, dtype=bool)
                    continue
                region = np.asarray(cache.region_features(image_id), dtype=np.float32)
                if region.shape != (n_proposals, 512):
                    raise ValueError(f"{backbone}: region cache proposal count differs from bank for image {image_id}")
                valid = np.isfinite(region).all(axis=1) & (np.linalg.norm(region, axis=1) > 0)
                valid_by_backbone[backbone][image_id] = valid
    for image_id in sorted({row.image_id for row in rows}):
        if not np.array_equal(valid_by_backbone["b0"][image_id], valid_by_backbone["b16"][image_id]):
            raise ValueError(f"B0/B16 region caches disagree on crop-valid proposals for image {image_id}")
    return valid_by_backbone["b0"], valid_by_backbone


def prepare_candidates_from_caches(
    rows: Sequence[FineCopsRow],
    bank_path: str | Path,
    feature_roots: Mapping[str, Path],
    output_path: str | Path,
) -> Path:
    """Derive natural/controlled IDs from frozen proposals and crop validity."""
    bank_data = _load_bank_data(bank_path)
    missing = sorted({row.image_id for row in rows} - set(bank_data))
    if missing:
        raise ValueError(f"proposal bank is missing cohort image IDs: {missing[:10]}")
    valid_by_image, _ = _cache_invalid_to_valid(rows, feature_roots, bank_data)
    payload = build_candidate_manifest(rows, bank_data, valid_by_image)
    return save_candidate_manifest(output_path, payload)


def validate_candidate_manifest_against_sources(
    rows: Sequence[FineCopsRow],
    assignments: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    bank_data: Mapping[int, tuple[np.ndarray, np.ndarray]],
    valid_by_image: Mapping[int, np.ndarray],
) -> None:
    """Recompute and compare every candidate identity before scorer inference."""
    expected = build_candidate_manifest(rows, bank_data, valid_by_image)
    for row, item in zip(rows, expected, strict=True):
        for mode in ("natural", "controlled"):
            for k in REQUESTED_KS:
                observed = np.asarray(assignments[row.record_id][mode][k], dtype=np.int64)
                target = np.asarray(item["candidate_indices"][mode][str(k)], dtype=np.int64)
                if not np.array_equal(observed, target):
                    raise ValueError(
                        f"candidate manifest differs from frozen source construction: {row.record_id}/{mode}/K{k}"
                    )


def source_code_paths() -> list[Path]:
    """Python modules whose implementations directly affect V3 preparation/inference."""
    relative = (
        "src/ccg/v3/candidates.py", "src/ccg/v3/inference.py", "src/ccg/v3/evaluation.py",
        "src/ccg/v3/protocol.py", "src/ccg/v3/exposure.py", "scripts/v3_run_pipeline.py",
        "src/ccg/data/bank.py", "src/ccg/data/proposals.py", "src/ccg/data/rpn.py",
        "src/ccg/features/cache.py", "src/ccg/features/clip_encoder.py", "src/ccg/features/extract_regions.py",
        "src/ccg/features/extract_text.py", "src/ccg/models/b3_data.py", "src/ccg/models/independent.py",
        "src/ccg/experiment/phase0b.py", "src/ccg/repairs/information.py", "src/ccg/reliability/features.py",
        "src/ccg/reliability/models.py", "src/ccg/semantic/features.py", "src/ccg/metrics/discrimination.py",
    )
    return [ROOT / path for path in relative]


def frozen_model_paths() -> list[Path]:
    """All directly consumed frozen scorer, reliability, temperature, and encoder files."""
    paths: list[Path] = []
    for seed in (1, 2, 3):
        for scorer_root in (B0_GROUNDING_ROOT, B16_GROUNDING_ROOT):
            seed_root = scorer_root / f"seed_{seed}"
            paths.extend((seed_root / "model.npz", seed_root / "training.json", seed_root / "eval_metadata.json"))
        b0_root = B0_RELIABILITY_ROOT / f"b3_seed{seed}"
        for filename in GROUP_FILES.values():
            paths.append(b0_root / filename)
        for regime in ("smallK5_10", "largeK20_50"):
            paths.extend((b0_root / f"score_deepsets_{regime}.json", b0_root / f"score_deepsets_{regime}.pt"))
        for group in GROUPS.values():
            paths.append(B16_RELIABILITY_ROOT / f"b3_seed{seed}" / f"{group}.json")
    for metadata_path in (B0_ENCODER_CHECKPOINT, B16_ENCODER_CHECKPOINT):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths.append(ROOT / metadata["backbone"]["checkpoint_path"].replace("\\", "/"))
        paths.append(metadata_path)
    paths.append(RPN_CHECKPOINT_PATH)
    return paths


def verify_prepare_authorization(
    path: str | Path,
    *,
    config_path: str | Path,
    cohort_path: str | Path,
    exposure_paths: Sequence[str | Path],
    image_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Validate parent's content-bound authorization for confirmation preparation."""
    auth_path = Path(path).resolve()
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "v3-confirmation-prepare-v1":
        raise ValueError("confirmation preparation requires parent schema v3-confirmation-prepare-v1")
    if payload.get("allowed_operations") != ["rpn_bank", "clip_features", "candidate_manifest"]:
        raise ValueError("confirmation preparation authorization has unexpected allowed operations")
    if payload.get("disallowed_operations") != ["grounding_forward", "reliability_forward", "performance_aggregation"]:
        raise ValueError("confirmation preparation authorization has unexpected disallowed operations")
    inputs = payload.get("inputs", [])
    by_path = {str(item["path"]).replace("\\", "/"): item for item in inputs}
    if len(by_path) != len(inputs):
        raise ValueError("preparation authorization has duplicate input paths")
    for relative, item in by_path.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or sha256_file(source) != item.get("sha256"):
            raise ValueError(f"preparation authorization input changed, missing, or outside workspace: {relative}")
    required = {
        Path(config_path).resolve(), Path(cohort_path).resolve(),
        *(Path(value).resolve() for value in exposure_paths),
        *(Path(value).resolve() for value in image_paths),
        *source_code_paths(), *frozen_model_paths(),
    }
    roles = {"config", "cohort", "exposure", "code", "encoder_model"}
    if image_paths:
        roles.add("image")
    if not roles.issubset({str(item.get("role")) for item in inputs}):
        raise ValueError(f"preparation authorization missing required roles {sorted(roles - {str(x.get('role')) for x in inputs})}")
    expected_roles: dict[Path, str] = {
        Path(config_path).resolve(): "config",
        Path(cohort_path).resolve(): "cohort",
        **{Path(value).resolve(): "exposure" for value in exposure_paths},
        **{Path(value).resolve(): "image" for value in image_paths},
        **{value.resolve(): "code" for value in source_code_paths()},
    }
    encoder_metadata = {B0_ENCODER_CHECKPOINT.resolve(), B16_ENCODER_CHECKPOINT.resolve()}
    encoder_weights = {
        (ROOT / json.loads(path.read_text(encoding="utf-8"))["backbone"]["checkpoint_path"].replace("\\", "/")).resolve()
        for path in (B0_ENCODER_CHECKPOINT, B16_ENCODER_CHECKPOINT)
    }
    for source in frozen_model_paths():
        expected_roles[source.resolve()] = "encoder_model" if source.resolve() in encoder_metadata | encoder_weights else "model"
    for source in required:
        relative = source.relative_to(ROOT).as_posix()
        item = by_path.get(relative)
        if item is None:
            raise ValueError(f"preparation authorization does not bind consumed input {relative}")
        if str(item.get("role")) != expected_roles.get(source):
            raise ValueError(f"preparation authorization role mismatch for {relative}: expected {expected_roles.get(source)!r}")
    expected_cohort = Path(cohort_path).resolve().relative_to(ROOT).as_posix()
    accepted = payload.get("acceptance")
    if not isinstance(accepted, dict):
        raise ValueError("preparation authorization must contain content-bound acceptance reports")
    for name in ("exposure", "development", "replication"):
        record = accepted.get(name, {})
        relative = str(record.get("path", "")).replace("\\", "/")
        if record.get("status") != "PASS" or relative not in by_path:
            raise ValueError(f"preparation authorization acceptance is missing or not passed: {name}")
        report_path = (ROOT / relative).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise ValueError(f"preparation acceptance report does not itself pass: {name}")
    return payload


def write_preparation_manifest(
    output: str | Path,
    *,
    stage: str,
    config_path: str | Path,
    cohort_path: str | Path,
    exposure_paths: Sequence[str | Path],
    image_paths: Sequence[str | Path],
    bank_path: str | Path,
    feature_roots: Mapping[str, Path],
    candidate_path: str | Path,
    authorization_path: str | Path | None = None,
) -> Path:
    """Content inventory for candidate/feature artifacts to be sealed later."""
    files: list[Path] = [
        Path(config_path), Path(cohort_path), Path(bank_path), Path(candidate_path),
        Path(bank_path).parent / EMPTY_PROPOSAL_FILENAME, RPN_CHECKPOINT_PATH,
    ]
    files.extend(Path(path) for path in exposure_paths)
    files.extend(Path(path) for path in image_paths)
    for root in feature_roots.values():
        files.extend(root / name for name in ("metadata.json", "region_features.h5", "text_features.h5", "invalid_crops.csv"))
    if authorization_path is not None:
        files.append(Path(authorization_path))
    unique = {path.resolve() for path in files}
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preparation manifest inputs missing: {missing[:10]}")
    payload = {
        "schema": "v3-preparation-manifest-v1",
        "stage": stage,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(unique, key=lambda value: value.relative_to(ROOT).as_posix())
        ],
        "operations": ["rpn_bank", "clip_features", "candidate_manifest"],
        "scorer_forward": False,
        "reliability_forward": False,
        "performance_aggregation": False,
        "authorization_sha256": sha256_file(authorization_path) if authorization_path else None,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def _source_model(backbone: str, seed: int):
    from ..experiment import phase0b

    model_root = B0_GROUNDING_ROOT if backbone == "b0" else B16_GROUNDING_ROOT
    model, metadata = phase0b.load_b3_model(model_root / f"seed_{seed}", device="cpu")
    return model, metadata


def _build_b3_inputs(
    row: FineCopsRow,
    candidate_indices: np.ndarray,
    boxes: np.ndarray,
    objectness: np.ndarray,
    image_size: tuple[int, int],
    region: np.ndarray,
    text: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ..models.b3_data import geometry_features_centered

    indices = np.asarray(candidate_indices, dtype=np.int64)
    z_q = np.asarray(text, dtype=np.float32)
    z_i = np.asarray(region[indices], dtype=np.float32)
    geometry = geometry_features_centered(boxes[indices], image_size, objectness[indices])
    return z_q, z_i, geometry


def _feature_confidences(
    backbone: str,
    seed: int,
    raw_scores: np.ndarray,
    z_q: np.ndarray,
    z_i: np.ndarray,
    models: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, str]]:
    from ..reliability import features as rfeat
    from ..repairs.information import feature_block
    from ..semantic import features as sfeat

    scores = np.asarray(raw_scores, dtype=np.float32).reshape(1, -1)
    if scores.shape[1] < 5:
        names = [name for name in ("S", "S+Q", "S+V", "Full")]
        if backbone == "b0":
            names.extend(("MSP", "SDS_small", "SDS_large"))
        return {}, {name: "k_eff_below_5" for name in names}
    stats = rfeat.stat_features(scores, temperature=float(models["temperature"]))
    semantic = sfeat.semantic_stats(
        np.asarray(z_q, dtype=np.float32).reshape(1, -1),
        np.asarray(z_i, dtype=np.float32).reshape(1, scores.shape[1], -1),
        scores,
    )
    if not np.isfinite(stats).all() or not np.isfinite(semantic).all():
        raise ValueError("non-finite frozen reliability features")
    confidence: dict[str, float] = {}
    for short in ("S", "SQ", "SV", "Full"):
        fit, predictor = models[short]
        group = GROUPS[short]
        raw = feature_block(stats, semantic, group)
        normalized = rfeat.normalize_apply(raw, fit)
        confidence[GROUPS[short]] = float(predictor.predict_proba(normalized)[0])
    confidence["MSP"] = float(rfeat.scalar_confidence(scores, temperature=float(models["temperature"]))["msp"][0])
    if backbone == "b0":
        confidence["SDS_small"] = _predict_sds(models["sds_small"], scores[0])
        confidence["SDS_large"] = _predict_sds(models["sds_large"], scores[0])
    return confidence, {}


def _feature_confidences_batch(
    backbone: str,
    scores: np.ndarray,
    z_q: np.ndarray,
    z_i: np.ndarray,
    models: Mapping[str, Any],
) -> tuple[list[dict[str, float]], list[dict[str, str]]]:
    """Vectorized reliability inference for one fixed effective candidate width."""
    from ..reliability import features as rfeat
    from ..repairs.information import feature_block
    from ..semantic import features as sfeat

    raw = np.asarray(scores, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("batch scorer logits must be [n,K]")
    if raw.shape[1] < 5:
        expected = [name for name in ("S", "S+Q", "S+V", "Full")]
        expected.append("MSP")
        if backbone == "b0":
            expected.extend(("SDS_small", "SDS_large"))
        return ([{} for _ in range(raw.shape[0])], [{key: "k_eff_below_5" for key in expected} for _ in range(raw.shape[0])])
    stats = rfeat.stat_features(raw, temperature=float(models["temperature"]))
    semantic = sfeat.semantic_stats(np.asarray(z_q, dtype=np.float32), np.asarray(z_i, dtype=np.float32), raw)
    if not np.isfinite(stats).all() or not np.isfinite(semantic).all():
        raise ValueError("non-finite frozen reliability features")
    columns: dict[str, np.ndarray] = {}
    for short in ("S", "SQ", "SV", "Full"):
        fit, predictor = models[short]
        columns[GROUPS[short]] = predictor.predict_proba(
            rfeat.normalize_apply(feature_block(stats, semantic, GROUPS[short]), fit)
        )
    columns["MSP"] = rfeat.scalar_confidence(raw, temperature=float(models["temperature"]))["msp"]
    if backbone == "b0":
        columns["SDS_small"] = _predict_sds_batch(models["sds_small"], raw)
        columns["SDS_large"] = _predict_sds_batch(models["sds_large"], raw)
    return (
        [{key: float(values[i]) for key, values in columns.items()} for i in range(raw.shape[0])],
        [{} for _ in range(raw.shape[0])],
    )


def _cell_metadata(
    row: FineCopsRow,
    mode: str,
    k: int,
    ids: np.ndarray,
    boxes: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    from .candidates import box_iou_one_to_many

    indices = np.asarray(ids, dtype=np.int64)
    ious = box_iou_one_to_many(row.gt_boxxyxy, boxes)
    valid_indices = np.flatnonzero(valid_mask)
    bank_target_count = int(np.sum(ious[valid_indices] >= IOU_THRESHOLD))
    presented_target_count = int(np.sum(ious[indices] >= IOU_THRESHOLD))
    target_present = bool(presented_target_count > 0)
    if valid_indices.size == 0:
        missing_reason = "no_valid_proposals"
    elif bank_target_count == 0:
        missing_reason = "no_valid_target_proposal"
    elif mode == "natural" and not target_present:
        missing_reason = "no_valid_target_in_top_k"
    elif mode == "controlled" and indices.size < int(k):
        missing_reason = "insufficient_control_distractors"
    elif indices.size == 0:
        missing_reason = "no_valid_target_in_top_k"
    else:
        missing_reason = None
    return {
        "record_id": row.record_id,
        "expr_id": row.expr_id,
        "sentence_id": row.sentence_id,
        "image_id": row.image_id,
        "gqa_image_id": row.gqa_image_id,
        "source_split": row.source_split,
        "expression": row.expression,
        "level": row.level,
        "mode": mode,
        "requested_k": int(k),
        "k_eff": int(indices.size),
        "candidate_indices": indices.astype(np.int64).tolist(),
        "target_present": target_present,
        "target_coverage": target_present,
        "bank_target_coverage": bool(bank_target_count > 0),
        "presented_target_coverage": target_present,
        "valid_target_count": bank_target_count,
        "bank_valid_target_count": bank_target_count,
        "presented_valid_target_count": presented_target_count,
        "missing_reason": missing_reason,
    }


def _batched_inference_rows(
    rows: Sequence[FineCopsRow],
    assignments: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    bank_data: Mapping[int, tuple[np.ndarray, np.ndarray]],
    caches: Mapping[str, Any],
    image_sizes: Mapping[int, tuple[int, int]],
    scorer_models: Mapping[tuple[str, int], Any],
    reliability_models: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    batch_size: int = 64,
    log: Callable[[str], None] = print,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Run candidate and confidence forwards in bounded fixed-width batches."""
    from ..experiment.phase0b import _forward_superset
    from ..models.b3_data import geometry_features_centered
    from .candidates import prediction_correctness

    predictions: list[dict[str, Any]] = []
    raw_rows: list[np.ndarray] = []
    region_lru: dict[str, OrderedDict[int, np.ndarray]] = {
        "b0": OrderedDict(), "b16": OrderedDict()
    }
    text_lru: dict[str, OrderedDict[int, np.ndarray]] = {
        "b0": OrderedDict(), "b16": OrderedDict()
    }

    def cache_row(backbone: str, row: FineCopsRow) -> tuple[np.ndarray, np.ndarray]:
        if row.image_id not in region_lru[backbone]:
            boxes = bank_data[row.image_id][0]
            region_lru[backbone][row.image_id] = (
                np.empty((0, 512), dtype=np.float32)
                if boxes.shape[0] == 0
                else np.asarray(caches[backbone].region_features(row.image_id), dtype=np.float32)
            )
        region_lru[backbone].move_to_end(row.image_id)
        if row.sentence_id not in text_lru[backbone]:
            text_lru[backbone][row.sentence_id] = np.asarray(caches[backbone].text_features(row.sentence_id), dtype=np.float32)
        text_lru[backbone].move_to_end(row.sentence_id)
        while len(region_lru[backbone]) > 64:
            region_lru[backbone].popitem(last=False)
        while len(text_lru[backbone]) > 128:
            text_lru[backbone].popitem(last=False)
        return region_lru[backbone][row.image_id], text_lru[backbone][row.sentence_id]

    n_cells = len(rows) * 2 * len(REQUESTED_KS)
    cell_number = 0
    for mode in ("natural", "controlled"):
        for k in REQUESTED_KS:
            by_width: dict[int, list[tuple[FineCopsRow, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]]] = defaultdict(list)
            for row in rows:
                boxes, objectness = bank_data[row.image_id]
                valid_by = []
                for backbone in ("b0", "b16"):
                    region, _text = cache_row(backbone, row)
                    valid_by.append(np.isfinite(region).all(axis=1) & (np.linalg.norm(region, axis=1) > 0))
                if not np.array_equal(valid_by[0], valid_by[1]):
                    raise ValueError(f"B0/B16 validity mismatch for image {row.image_id}")
                ids = np.asarray(assignments[row.record_id][mode][k], dtype=np.int64)
                if np.any(ids >= boxes.shape[0]) or np.any(~valid_by[0][ids]):
                    raise ValueError(f"{row.record_id} {mode} K{k}: invalid frozen candidate identity")
                metadata = _cell_metadata(row, mode, k, ids, boxes, valid_by[0])
                by_width[int(ids.size)].append((row, ids, metadata, boxes, objectness))

            for width, group in sorted(by_width.items()):
                if width == 0:
                    for row, ids, metadata, _boxes, _obj in group:
                        for backbone in ("b0", "b16"):
                            missing_models = ["S", "S+Q", "S+V", "Full", "MSP"]
                            if backbone == "b0":
                                missing_models.extend(("SDS_small", "SDS_large"))
                            for seed in (1, 2, 3):
                                prediction = dict(metadata)
                                prediction.update({
                                    "backbone": backbone,
                                    "seed": f"b3_seed{seed}",
                                    "winner_proposal_index": None,
                                    "winner_iou": None,
                                    "correct": None,
                                    "confidence": {},
                                    "confidence_missing_reason": {
                                        name: "empty_candidate_set" for name in missing_models
                                    },
                                })
                                predictions.append(prediction)
                                raw_rows.append(np.empty(0, dtype=np.float32))
                    continue
                for start in range(0, len(group), max(1, int(batch_size))):
                    chunk = group[start : start + max(1, int(batch_size))]
                    sample_rows = [entry[0] for entry in chunk]
                    candidate_ids = [entry[1] for entry in chunk]
                    q_by_backbone: dict[str, list[np.ndarray]] = {"b0": [], "b16": []}
                    zi_by_backbone: dict[str, list[np.ndarray]] = {"b0": [], "b16": []}
                    geo_by_backbone: dict[str, list[np.ndarray]] = {"b0": [], "b16": []}
                    validity: list[np.ndarray] = []
                    for row, ids, _metadata, boxes, objectness in chunk:
                        for backbone in ("b0", "b16"):
                            region, text = cache_row(backbone, row)
                            q_by_backbone[backbone].append(text)
                            zi_by_backbone[backbone].append(region[ids])
                            geo_by_backbone[backbone].append(
                                geometry_features_centered(boxes[ids], image_sizes[row.image_id], objectness[ids])
                            )
                        region, _text = cache_row("b0", row)
                        validity.append(np.isfinite(region).all(axis=1) & (np.linalg.norm(region, axis=1) > 0))

                    for backbone in ("b0", "b16"):
                        queries = q_by_backbone[backbone]
                        candidate_embeddings = zi_by_backbone[backbone]
                        geometries = geo_by_backbone[backbone]
                        for seed in (1, 2, 3):
                            model = scorer_models[(backbone, seed)]
                            score_matrix = _forward_superset(model, queries, candidate_embeddings, geometries)
                            confidences, missing_by_row = _feature_confidences_batch(
                                backbone,
                                score_matrix,
                                np.stack(queries),
                                np.stack(candidate_embeddings),
                                reliability_models[(backbone, seed)],
                            )
                            for i, (row, ids, metadata, boxes, _objectness) in enumerate(chunk):
                                winner, winner_iou, correct = prediction_correctness(
                                    ids, score_matrix[i], boxes, row.gt_boxxyxy
                                )
                                prediction = dict(metadata)
                                prediction.update({
                                    "backbone": backbone,
                                    "seed": f"b3_seed{seed}",
                                    "winner_proposal_index": winner,
                                    "winner_iou": winner_iou,
                                    "correct": correct,
                                    "confidence": confidences[i],
                                    "confidence_missing_reason": missing_by_row[i],
                                })
                                predictions.append(prediction)
                                raw_rows.append(np.asarray(score_matrix[i], dtype=np.float32))
                    cell_number += len(chunk)
                    if cell_number % 2048 < len(chunk) or cell_number == n_cells:
                        log(f"V3 frozen candidate cells: {cell_number}/{n_cells}")
                    # Keep only a small working set; candidate embeddings are
                    # never accumulated for the whole FineCops cohort.
    return predictions, raw_rows


def _score_one_cell(
    *,
    row: FineCopsRow,
    mode: str,
    k: int,
    candidate_indices: np.ndarray,
    boxes: np.ndarray,
    objectness: np.ndarray,
    image_size: tuple[int, int],
    valid_mask: np.ndarray,
    region_by_backbone: Mapping[str, np.ndarray],
    text_by_backbone: Mapping[str, np.ndarray],
    scorer_models: Mapping[str, Any],
    reliability_models: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from ..repairs.information import score_deepsets_inputs

    ids = np.asarray(candidate_indices, dtype=np.int64)
    ious = None
    if ids.size:
        from .candidates import box_iou_one_to_many

        ious = box_iou_one_to_many(row.gt_boxxyxy, boxes)
    result = []
    for backbone in ("b0", "b16"):
        for seed in (1, 2, 3):
            if ids.size:
                z_q, z_i, geometry = _build_b3_inputs(
                    row, ids, boxes, objectness, image_size, region_by_backbone[backbone], text_by_backbone[backbone]
                )
                model = scorer_models[(backbone, seed)]
                from ..experiment.phase0b import _forward_superset

                scores = _forward_superset(model, [z_q], [z_i], [geometry])[0]
                winner, winner_iou, is_correct = prediction_correctness(ids, scores, boxes, row.gt_boxxyxy)
                all_valid = np.flatnonzero(valid_mask)
                target_count = int(np.sum(ious[all_valid] >= IOU_THRESHOLD)) if ious is not None else 0
                target_present = bool(np.any(ious[ids] >= IOU_THRESHOLD))
                coverage = bool(target_count > 0)
                confidence, missing = _feature_confidences(
                    backbone, seed, scores, z_q, z_i,
                    reliability_models[(backbone, seed)],
                )
            else:
                scores = np.empty(0, dtype=np.float32)
                winner, winner_iou, is_correct = None, None, None
                all_valid = np.flatnonzero(valid_mask)
                target_count = int(np.sum(ious[all_valid] >= IOU_THRESHOLD)) if ious is not None else 0
                target_present = False
                coverage = bool(target_count > 0)
                confidence, missing = {}, {"all": "empty_candidate_set"}
            result.append({
                "record_id": row.record_id,
                "expr_id": row.expr_id,
                "sentence_id": row.sentence_id,
                "image_id": row.image_id,
                "gqa_image_id": row.gqa_image_id,
                "source_split": row.source_split,
                "expression": row.expression,
                "level": row.level,
                "mode": mode,
                "requested_k": int(k),
                "k_eff": int(ids.size),
                "target_present": target_present,
                "target_coverage": coverage,
                "valid_target_count": target_count,
                "missing_reason": (
                    "no_valid_proposals" if ids.size == 0 and all_valid.size == 0
                    else "no_valid_target_proposal" if ids.size == 0 and target_count == 0
                    else "no_valid_target_in_top_k" if ids.size > 0 and not target_present and mode == "natural"
                    else "insufficient_control_distractors" if mode == "controlled" and target_count > 0 and ids.size < k
                    else "no_valid_target_in_top_k" if ids.size == 0
                    else None
                ),
                "winner_proposal_index": winner,
                "winner_iou": winner_iou,
                "correct": is_correct,
                "backbone": backbone,
                "seed": seed,
                "confidence": confidence,
                "confidence_missing_reason": missing,
                "raw_logits": np.asarray(scores, dtype=np.float32),
            })
    return result


def run_frozen_inference(
    rows: Sequence[FineCopsRow],
    candidate_path: str | Path,
    bank_path: str | Path,
    feature_roots: Mapping[str, Path],
    image_sizes: Mapping[int, tuple[int, int]],
    output: str | Path,
    *,
    cohort_path: str | Path,
    allowed_confirmation: bool = False,
    freeze_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    exposure_paths: Sequence[str | Path] = (),
    config_path: str | Path = CONFIG_PATH,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Score lossless natural/controlled V3 cells with frozen checkpoints.

    No scorer/model is loaded until confirmation freeze hashes, cohort split,
    required consumed files, and the write-once run claim are validated.
    """
    from ..data.bank import image_ids as proposal_ids
    from ..features.cache import FeatureCache
    from .evaluation import summarize_prediction_rows, primary_point_estimates
    from .protocol import claim_confirmation, verify_freeze

    config = validate_config_file(config_path)
    assignments = load_candidate_manifest(candidate_path, rows)
    bank_data = _load_bank_data(bank_path)
    if set(proposal_ids(bank_path)) - set(bank_data):
        raise AssertionError("proposal bank image index changed during loading")
    if set(bank_data) != {row.image_id for row in rows}:
        raise ValueError("proposal bank IDs must exactly equal this split's distinct image IDs")
    valid_by_image, _ = _cache_invalid_to_valid(rows, feature_roots, bank_data)
    validate_candidate_manifest_against_sources(rows, assignments, bank_data, valid_by_image)
    exposure_report = validate_exposure_manifest(rows, exposure_paths)
    image_paths = list(unique_images(rows).values())
    consumed = [
        Path(candidate_path), Path(bank_path), Path(bank_path).parent / EMPTY_PROPOSAL_FILENAME,
        Path(config_path), Path(cohort_path),
    ]
    consumed += [Path(exposure) for exposure in exposure_paths]
    consumed += image_paths
    for root in feature_roots.values():
        consumed.extend([root / "metadata.json", root / "region_features.h5", root / "text_features.h5", root / "invalid_crops.csv"])
    if set(feature_roots) != {"b0", "b16"}:
        raise ValueError("inference requires exactly the frozen b0 and b16 feature caches")
    model_inputs = frozen_model_paths()
    code_inputs = source_code_paths()
    consumed.extend(model_inputs)
    consumed.extend(code_inputs)
    for path in consumed:
        if not path.is_file():
            raise FileNotFoundError(f"required V3 inference input missing: {path}")
    freeze_payload = None
    if allowed_confirmation:
        if freeze_path is None or ledger_path is None:
            raise ValueError("confirmation inference requires a freeze manifest and single-run ledger")
        freeze_payload = verify_freeze(freeze_path, root=ROOT)
        stage = freeze_payload.get("cohort_stage")
        if not isinstance(stage, dict):
            raise ValueError("confirmation freeze is missing cohort_stage metadata")
        cohort_relative = Path(cohort_path).resolve().relative_to(ROOT).as_posix()
        if (
            stage.get("name") != "confirmation"
            or stage.get("split") != "val"
            or str(stage.get("manifest", "")).replace("\\", "/") != cohort_relative
        ):
            raise ValueError("confirmation cohort_stage must bind this exact official positive val manifest")
        if any(row.source_split != "val" for row in rows):
            raise ValueError("confirmation cohort rows must all be official positive val rows")
        input_by_path = {str(item["path"]).replace("\\", "/"): item for item in freeze_payload["inputs"]}
        if len(input_by_path) != len(freeze_payload["inputs"]):
            raise ValueError("confirmation freeze repeats an input path")
        for path in consumed:
            relative = path.resolve().relative_to(ROOT).as_posix()
            if relative not in input_by_path:
                raise ValueError(f"confirmation freeze does not bind consumed input {relative}")
            item = input_by_path[relative]
            if sha256_file(path) != item["sha256"]:
                raise ValueError(f"confirmation consumed input changed after freeze: {relative}")
        required_roles = {"code", "model", "cohort", "config", "exposure"}
        roles = {str(item.get("role")) for item in freeze_payload["inputs"]}
        if not required_roles.issubset(roles):
            raise ValueError(f"confirmation freeze misses roles {sorted(required_roles - roles)}")
        if input_by_path[cohort_relative].get("role") != "cohort":
            raise ValueError("confirmation cohort manifest is not bound as role=cohort")
        config_relative = Path(config_path).resolve().relative_to(ROOT).as_posix()
        if input_by_path[config_relative].get("role") != "config":
            raise ValueError("V3 config is not bound as role=config")
        for path in exposure_paths:
            relative = Path(path).resolve().relative_to(ROOT).as_posix()
            if input_by_path[relative].get("role") != "exposure":
                raise ValueError(f"exposure input has incorrect freeze role: {relative}")
        for path in image_paths:
            relative = path.resolve().relative_to(ROOT).as_posix()
            if input_by_path[relative].get("role") != "image":
                raise ValueError(f"cohort image has incorrect freeze role=image: {relative}")
        for path in code_inputs:
            relative = path.resolve().relative_to(ROOT).as_posix()
            if input_by_path[relative].get("role") != "code":
                raise ValueError(f"pipeline dependency has incorrect freeze role=code: {relative}")
        for path in model_inputs:
            relative = path.resolve().relative_to(ROOT).as_posix()
            if input_by_path[relative].get("role") not in {"model", "encoder_model"}:
                raise ValueError(f"frozen checkpoint has incorrect freeze model role: {relative}")
        claim_confirmation(freeze_path, ledger_path, root=ROOT)
    elif any(row.source_split not in {"train", "dev", "development"} for row in rows):
        raise ValueError("unfrozen inference is restricted to the V3 development cohort")
    else:
        if any(row.source_split not in {"train", "dev", "development"} for row in rows):
            raise ValueError("unfrozen inference is restricted to development rows")

    # All raw scorer forwards start after the claim for formal confirmation.
    scorer_models: dict[tuple[str, int], Any] = {}
    reliability_models: dict[tuple[str, int], dict[str, Any]] = {}
    for backbone in ("b0", "b16"):
        for seed in (1, 2, 3):
            scorer_models[(backbone, seed)] = _source_model(backbone, seed)[0]
            reliability_models[(backbone, seed)] = (
                _load_b0_reliability(seed) if backbone == "b0" else _load_b16_reliability(seed)
            )

    predictions: list[dict[str, Any]] = []
    raw_flat: list[np.ndarray] = []
    with ExitStack() as stack:
        caches = {
            backbone: stack.enter_context(FeatureCache.open(root))
            for backbone, root in feature_roots.items()
        }
        predictions, raw_flat = _batched_inference_rows(
            rows,
            assignments,
            bank_data,
            caches,
            image_sizes,
            scorer_models,
            reliability_models,
            batch_size=64,
            log=log,
        )

    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    tmp = predictions_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for index, prediction in enumerate(predictions):
            value = dict(prediction)
            value["raw_logits_row"] = index
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(tmp, predictions_path)
    offsets = np.concatenate(([0], np.cumsum([scores.size for scores in raw_flat], dtype=np.int64)))
    raw_scores = np.concatenate(raw_flat).astype(np.float32, copy=False) if raw_flat else np.empty(0, dtype=np.float32)
    np.savez_compressed(
        output_root / "scorer_logits.npz",
        logits=raw_scores,
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    grouped: dict[str, Any] = {}
    for mode in ("natural", "controlled"):
        for backbone in ("b0", "b16"):
            for seed in (1, 2, 3):
                subset = [
                    item for item in predictions
                    if item["mode"] == mode and item["backbone"] == backbone and item["seed"] == f"b3_seed{seed}"
                ]
                grouped[f"{mode}/{backbone}/seed_{seed}"] = summarize_prediction_rows(subset)
    by_cell = {
        (str(row["record_id"]), str(row["mode"]), int(row["requested_k"]), str(row["backbone"]), str(row["seed"])): row
        for row in predictions
    }
    k50_rows = [
        row for row in rows
        if (
            (candidate := assignments[row.record_id]["controlled"][50]).size == 50
            and by_cell[(row.record_id, "controlled", 50, "b0", "b3_seed1")]["target_present"]
        )
    ]
    primary: dict[str, Any] | None = None
    if k50_rows:
        common_ids = [row.record_id for row in k50_rows]
        confidence_by_seed: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        correctness_by_seed: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        for seed in (1, 2, 3):
            seed_name = f"b3_seed{seed}"
            confidence_sources = (
                ("B0_MSP", "MSP", "b0", (5, 50)),
                ("B0_Full", "Full", "b0", (50,)),
                ("B0_SQ", "S+Q", "b0", (50,)),
                ("B16_Full", "Full", "b16", (50,)),
                ("B16_SQ", "S+Q", "b16", (50,)),
                ("B0_SDS_large", "SDS_large", "b0", (50,)),
                ("B0_SDS_small", "SDS_small", "b0", (50,)),
            )
            confidence_by_seed[seed_name] = {
                output_name: {
                    k: np.asarray([
                        by_cell[(record_id, "controlled", k, backbone, f"b3_seed{seed}")]["confidence"].get(confidence_name, np.nan)
                        for record_id in common_ids
                    ], dtype=np.float64)
                    for k in ks
                }
                for output_name, confidence_name, backbone, ks in confidence_sources
            }
            correctness_by_seed[seed_name] = {
                backbone: {
                    k: np.asarray([
                        by_cell[(record_id, "controlled", k, backbone, f"b3_seed{seed}")]["correct"]
                        for record_id in common_ids
                    ], dtype=np.bool_)
                    for k in (5, 50)
                }
                for backbone in ("b0", "b16")
            }
        primary = primary_point_estimates(
            image_ids=[by_cell[(record_id, "controlled", 50, "b0", "b3_seed1")]["image_id"] for record_id in common_ids],
            common_mask=np.ones(len(common_ids), dtype=bool),
            confidence_by_seed=confidence_by_seed,
            correctness_by_seed=correctness_by_seed,
        )
    report = {
        "schema": "v3-pipeline-inference-v1",
        "stage": "confirmation" if allowed_confirmation else "development",
        "split": rows[0].source_split if rows else None,
        "n_expression_rows": len(rows),
        "n_unique_images": len({row.image_id for row in rows}),
        "cohort_path": Path(cohort_path).resolve().relative_to(ROOT).as_posix(),
        "cohort_sha256": sha256_file(cohort_path),
        "config_sha256": sha256_file(config_path),
        "cohort_boundary_annotation_report": cohort_bbox_boundary_report(rows, image_sizes),
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "proposal_bank_sha256": sha256_file(bank_path),
        "empty_proposal_flow_sha256": sha256_file(Path(bank_path).parent / EMPTY_PROPOSAL_FILENAME),
        "exposure_audit": exposure_report,
        "feature_cache_sha256": {
            backbone: {
                name: sha256_file(root / name)
                for name in ("metadata.json", "region_features.h5", "text_features.h5", "invalid_crops.csv")
            }
            for backbone, root in feature_roots.items()
        },
        "risk_tie_rule": "fractional_boundary_tie",
        "summary_by_mode_backbone_seed": grouped,
        "controlled_primary_point_estimates": primary,
        "full_confirmatory_bootstrap": "not run by V3 pipeline; requires parent resource grant",
    }
    summary_tmp = output_root / "summary.json.tmp"
    summary_tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(summary_tmp, output_root / "summary.json")
    if allowed_confirmation:
        ledger = Path(ledger_path)
        ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
        if ledger_payload.get("schema") != "v3-confirmation-run-v1" or ledger_payload.get("status") != "RUNNING":
            raise ValueError("confirmation run ledger changed while inference was running")
        ledger_payload["status"] = "INFERENCE_COMPLETE"
        ledger_payload["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ledger_payload["predictions"] = {
            "path": predictions_path.resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(predictions_path),
        }
        ledger_payload["summary"] = {
            "path": (output_root / "summary.json").resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256_file(output_root / "summary.json"),
        }
        ledger_tmp = ledger.with_suffix(ledger.suffix + ".tmp")
        ledger_tmp.write_text(json.dumps(ledger_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(ledger_tmp, ledger)
    return report


def verify_source_model_anchors(*, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Reproduce B0 random-source logits and repaired reliability probabilities.

    Uses the fixed source caches and published canonical predictions only; no
    FineCops row is read or scored by this check.
    """
    from ..experiment import phase0a, phase0b
    from ..models.b3_data import B3Corpus, B3Example
    from ..repairs.information import feature_block, load_logistic_predictor
    from ..reliability import features as rfeat
    import pandas as pd

    corpus = B3Corpus(
        B0_FEATURE_SOURCE,
        ROOT / "cache/manifests",
        ROOT / "data/raw/refcoco+/refcoco+/refs(unc).p",
        ROOT / "cache/proposals.h5",
        image_sizes_path=ROOT / "cache/image_sizes.npz",
        ks=(5, 10),
        regime="random",
        preload=False,
    )
    records = corpus.eval_records(phase0a.PRIMARY_SPLITS, (5,))
    by_id = {int(sample.sentence_id): sample for sample in records}
    if len(by_id) != len(records):
        raise ValueError("source anchor records have duplicate sentence IDs")
    selected_ids = sorted(by_id)[:24]
    rows_report: list[dict[str, Any]] = []
    max_logit_delta = 0.0
    for seed in (1, 2, 3):
        model, _ = phase0b.load_b3_model(B0_GROUNDING_ROOT / f"seed_{seed}", device="cpu")
        frozen_root = B0_GROUNDING_ROOT / f"seed_{seed}" / "raw_scores" / "K5.npz"
        with np.load(frozen_root, allow_pickle=False) as store:
            sentence_ids = np.asarray(store["sentence_id"], dtype=np.int64)
            scores_source = np.asarray(store["scores"], dtype=np.float32)
        source_map = {int(sid): index for index, sid in enumerate(sentence_ids)}
        samples = [by_id[sid] for sid in selected_ids]
        from ..semantic.hard_scores import materialise_examples, score_examples

        batch = materialise_examples(corpus, samples, 5)
        recomputed = score_examples(model, batch, batch_size=16)
        source = np.stack([scores_source[source_map[sid]] for sid in selected_ids])
        delta = float(np.max(np.abs(recomputed.astype(np.float64) - source.astype(np.float64))))
        max_logit_delta = max(max_logit_delta, delta)
        if delta > SOURCE_LOGIT_ATOL:
            raise AssertionError(f"B0 seed{seed} source anchor delta {delta:.3e} > {SOURCE_LOGIT_ATOL}")

    # Exact repaired S/Full probability anchors use the audited derived-feature
    # arrays and the frozen cross-K source prediction CSV. The derived arrays
    # were written in canonical ascending sentence_id order, while the raw
    # score NPZs retain their source order, so keep separate identity maps.
    reference = pd.read_csv(ROOT / "results/research_repair_v1/information/semantic_ablation_predictions.csv.gz")
    sds_reference = pd.read_csv(B0_SOURCE_PREDICTIONS)
    max_logistic_probability_delta = 0.0
    max_sds_probability_delta = 0.0
    logistic_rows_checked = 0
    sds_rows_checked = 0
    checkpoint_mapping: list[dict[str, Any]] = []
    for seed in (1, 2, 3):
        reliability = _load_b0_reliability(seed)
        for k in (5, 10):
            derived = np.load(B0_SOURCE_DERIVED / f"random_b3_seed{seed}_K{k}.npz", allow_pickle=False)
            raw_path = B0_RAW_SCORES / f"seed_{seed}/raw_scores/K{k}.npz"
            with np.load(raw_path, allow_pickle=False) as source_scores:
                source_ids = np.asarray(source_scores["sentence_id"], dtype=np.int64)
                logits = np.asarray(source_scores["scores"], dtype=np.float32)
            if source_ids.size != np.unique(source_ids).size or logits.ndim != 2 or logits.shape[0] != source_ids.size:
                raise ValueError(f"B0 seed{seed} K{k}: malformed/duplicate raw scorer identities")
            if not np.array_equal(source_ids, np.sort(source_ids)):
                log(f"B0 seed{seed} K{k}: raw score rows use source order; derived feature rows use canonical sentence_id order")
            source_indices = {int(sid): index for index, sid in enumerate(source_ids)}
            canonical_ids = np.sort(source_ids, kind="stable")
            canonical_indices = {int(sid): index for index, sid in enumerate(canonical_ids)}
            temperature = _load_temperature(B0_GROUNDING_ROOT / f"seed_{seed}/eval_metadata.json")
            stats, sem = derived["stats17"], derived["sem16"]
            if stats.shape != (source_ids.size, 17) or sem.shape != (source_ids.size, 16):
                raise ValueError(f"B0 seed{seed} K{k}: derived feature dimensions/row count do not align to source identities")
            if not np.array_equal(canonical_ids, np.sort(canonical_ids, kind="stable")):
                raise AssertionError("canonical source identity order is not ascending")
            identity = reference[(reference["seed"] == f"b3_seed{seed}") & (reference["cell"] == f"randomK{k}")]
            for short in ("S", "Full"):
                model_path = B0_RELIABILITY_ROOT / f"b3_seed{seed}" / GROUP_FILES[short]
                payload = json.loads(model_path.read_text(encoding="utf-8"))
                fit = rfeat.NormalizationFit.from_dict(payload["normalization"])
                predictor = load_logistic_predictor(payload)
                selected = identity[
                    (identity["model"] == short) & identity["eval_split"].isin(("testA", "testB"))
                ].sort_values("sentence_id").head(128)
                if len(selected) != 128:
                    raise AssertionError(f"B0 {short} seed{seed} K{k}: expected 128 source anchor rows, got {len(selected)}")
                for item in selected.itertuples(index=False):
                    sentence_id = int(item.sentence_id)
                    if sentence_id not in source_indices or sentence_id not in canonical_indices:
                        raise ValueError(f"B0 {short} seed{seed} K{k}: missing source identity {sentence_id}")
                    source_index = source_indices[sentence_id]
                    feature_index = canonical_indices[sentence_id]
                    raw = logits[source_index:source_index + 1]
                    if short == "S":
                        matrix = rfeat.stat_features(raw, temperature=temperature)
                    else:
                        matrix = feature_block(
                            stats[feature_index:feature_index + 1], sem[feature_index:feature_index + 1], "Full"
                        )
                    probability = float(predictor.predict_proba(rfeat.normalize_apply(matrix, fit))[0])
                    delta = abs(probability - float(item.probability))
                    max_logistic_probability_delta = max(max_logistic_probability_delta, delta)
                    logistic_rows_checked += 1
                    if delta > SOURCE_PROBABILITY_ATOL:
                        raise AssertionError(
                            f"B0 logistic {short} seed{seed} K{k} source probability delta {delta:.3e} > {SOURCE_PROBABILITY_ATOL}"
                        )
                checkpoint_mapping.append({
                    "backbone": "b0",
                    "seed": seed,
                    "source_k": k,
                    "group": short,
                    "checkpoint": model_path.relative_to(ROOT).as_posix(),
                    "checkpoint_sha256": sha256_file(model_path),
                    "reference": "semantic_ablation_predictions.csv.gz",
                    "reference_rows_checked": int(len(selected)),
                })
            # The corrected ScoreDeepSets models are separately anchored on
            # canonical source outputs for every scorer seed and source K.
            for regime, short_name in (("smallK5_10", "small"), ("largeK20_50", "large")):
                source_name = f"ScoreDeepSets_{regime}"
                selected = sds_reference[
                    (sds_reference["seed"] == f"b3_seed{seed}")
                    & (sds_reference["cell"] == f"randomK{k}")
                    & (sds_reference["model"] == source_name)
                    & (sds_reference["eval_split"].isin(("testA", "testB")))
                ].sort_values("sentence_id").head(128)
                if len(selected) != 128:
                    raise AssertionError(f"B0 SDS {short_name} seed{seed} K{k}: expected 128 source anchor rows, got {len(selected)}")
                model_state = reliability[f"sds_{short_name}"]
                # Replay the full canonical source matrix in its original
                # batching pattern. Torch float32 matmul can differ by one ULP
                # when a 128-row subset is evaluated as one batch instead of
                # the source artifact's 1024-row batches.
                canonical_logits = logits[np.asarray([source_indices[int(sid)] for sid in canonical_ids])]
                all_probabilities = _predict_sds_batch(model_state, canonical_logits)
                feature_indices = np.asarray(
                    [canonical_indices[int(sid)] for sid in selected["sentence_id"].to_numpy()], dtype=np.int64
                )
                probability = all_probabilities[feature_indices]
                delta = float(np.max(np.abs(probability - selected["probability"].to_numpy(dtype=np.float64))))
                max_sds_probability_delta = max(max_sds_probability_delta, delta)
                sds_rows_checked += int(selected.shape[0])
                if delta > SOURCE_PROBABILITY_ATOL:
                    raise AssertionError(
                        f"B0 repaired SDS {short_name} seed{seed} K{k} source probability delta {delta:.3e} > {SOURCE_PROBABILITY_ATOL}"
                    )
                stem = f"score_deepsets_{regime}"
                checkpoint = B0_RELIABILITY_ROOT / f"b3_seed{seed}" / f"{stem}.pt"
                checkpoint_mapping.append({
                    "backbone": "b0",
                    "seed": seed,
                    "source_k": k,
                    "group": f"SDS_{short_name}",
                    "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "reference": B0_SOURCE_PREDICTIONS.relative_to(ROOT).as_posix(),
                    "reference_rows_checked": int(len(selected)),
                })
            del derived
    if (
        logistic_rows_checked != 3 * 2 * 2 * 128
        or sds_rows_checked != 3 * 2 * 2 * 128
        or len(checkpoint_mapping) != 3 * 2 * 4
        or not np.isfinite(max_logit_delta + max_logistic_probability_delta + max_sds_probability_delta)
    ):
        raise AssertionError("B0 source anchor validation did not check all frozen source-model rows")
    report = {
        "status": "PASS",
        "source_only": True,
        "b0_scorer_max_abs_logit_delta": max_logit_delta,
        "b0_repaired_logistic_s_full_max_abs_probability_delta": max_logistic_probability_delta,
        "b0_repaired_sds_max_abs_probability_delta": max_sds_probability_delta,
        "b0_scorer_tolerance": SOURCE_LOGIT_ATOL,
        "b0_probability_tolerance": SOURCE_PROBABILITY_ATOL,
        "b0_scorer_anchor_rows": len(selected_ids) * 3,
        "b0_logistic_anchor_rows": logistic_rows_checked,
        "b0_sds_anchor_rows": sds_rows_checked,
        "source_reference": str(B0_SOURCE_PREDICTIONS.relative_to(ROOT)),
        "source_mapping": checkpoint_mapping,
    }
    log(json.dumps(report))
    return report
