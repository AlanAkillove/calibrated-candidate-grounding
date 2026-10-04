"""Fetch and audit V3 FineCops-Ref metadata and image exposure.

No model is trained or scored here.  Historical inputs are read-only.  V3 images
are written only under ``data/raw/v3_finecops/``; audit products are written only
under ``results/v3_final_validation/exposure/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccg.external.gqa_images import DEFAULT_WORKERS, fetch_images  # noqa: E402
from ccg.v3.exposure import (  # noqa: E402
    AUDIT_RULE_VERSION,
    DHASH_ALGORITHM,
    DHASH_DISTANCE_MAX,
    BKTree,
    fingerprint_image,
    hamming64,
    sha256_file,
)

FIGSHARE_API = "https://api.figshare.com/v2/articles/26048050"
VG_OFFICIAL_MAPPING_URL = "https://visualgenome.org/static/data/dataset/image_data.json.zip"
VG_MIRROR_URL = "https://huggingface.co/datasets/jn12/VisualGenome/resolve/main/image_data.json.zip"
VG_MAPPING_SHA256 = "b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a"
FIGSHARE_FILES = {
    47682931: "expression_pos_train_set.json",
    47682928: "expression_pos_val_set.json",
    47682937: "expression_pos_train_set_coco_format.json",
    47682934: "expression_pos_val_set_coco_format.json",
}
OLD_FINECOPS_TEST = REPO_ROOT / "data/raw/finecops/test_expression_pos.json"
GQA_TRAIN_GRAPHS = REPO_ROOT / "data/raw/gqa/train_sceneGraphs.json"
GQA_VAL_GRAPHS = REPO_ROOT / "data/raw/gqa/val_sceneGraphs.json"
RAW_V3 = REPO_ROOT / "data/raw/v3_finecops"
ANNOTATIONS_V3 = RAW_V3 / "annotations"
V3_IMAGES = RAW_V3 / "images"
EXPOSURE_OUT = REPO_ROOT / "results/v3_final_validation/exposure"
SOURCE_OUT = EXPOSURE_OUT / "source"
SEED = 20261004
DEV_IMAGE_CAP = 200


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _iter_json_object_items(path: Path, chunk_bytes: int = 1 << 20) -> Iterable[Tuple[str, Any]]:
    """Stream key/value pairs from a top-level JSON object with bounded memory."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    with Path(path).open("r", encoding="utf-8") as stream:
        def refill() -> bool:
            nonlocal buffer, position, eof
            if position >= chunk_bytes:
                buffer = buffer[position:]
                position = 0
            addition = stream.read(chunk_bytes)
            if not addition:
                eof = True
                return False
            buffer += addition
            return True

        def skip_space() -> str:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if eof or not refill():
                    return ""

        def decode_value() -> Any:
            nonlocal position
            while True:
                skip_space()
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not refill():
                        raise
                else:
                    position = end
                    return value

        if skip_space() != "{":
            raise ValueError(f"{path}: expected top-level JSON object")
        position += 1
        while True:
            character = skip_space()
            if character == "}":
                return
            if character == ",":
                position += 1
                continue
            if not character:
                raise ValueError(f"{path}: truncated top-level JSON object")
            key = decode_value()
            if not isinstance(key, str):
                raise ValueError(f"{path}: expected a string object key")
            if skip_space() != ":":
                raise ValueError(f"{path}: expected colon after key {key!r}")
            position += 1
            yield key, decode_value()


def _audit_gqa_target_boxes(
    cohort_rows: Mapping[str, Sequence[Mapping[str, Any]]], released_image_ids: Mapping[str, Set[str]]
) -> Dict[str, Dict[str, Any]]:
    """Compare released official FineCops boxes with the named GQA target object."""
    rows_by_image: Dict[str, List[Mapping[str, Any]]] = {}
    for split in ("train", "val"):
        for row in cohort_rows[split]:
            if row["gqa_image_id"] in released_image_ids[split]:
                rows_by_image.setdefault(row["gqa_image_id"], []).append(row)
    graph_records: Dict[str, Tuple[str, Mapping[str, Any]]] = {}
    for split, path in (("train", GQA_TRAIN_GRAPHS), ("val", GQA_VAL_GRAPHS)):
        for image_id, graph in _iter_json_object_items(path):
            if image_id in rows_by_image:
                graph_records[image_id] = (split, graph)
    out: Dict[str, Dict[str, Any]] = {}
    for image_id, rows in rows_by_image.items():
        graph_info = graph_records.get(image_id)
        for row in rows:
            expr_id = str(row["expr_id"])
            if graph_info is None:
                out[expr_id] = {"gqa_graph_status": "image_record_missing"}
                continue
            graph_split, graph = graph_info
            objects = graph.get("objects", {})
            object_id = str(row.get("target_instance_id", ""))
            target = objects.get(object_id)
            record: Dict[str, Any] = {
                "gqa_graph_status": "target_object_found" if target is not None else "target_object_missing",
                "gqa_graph_split": graph_split,
                "gqa_graph_width": graph.get("width"),
                "gqa_graph_height": graph.get("height"),
                "target_instance_id": object_id,
            }
            if target is not None:
                graph_bbox = [target.get(key) for key in ("x", "y", "w", "h")]
                record["gqa_target_bbox_xywh"] = graph_bbox
                record["official_bbox_equals_gqa_target_bbox"] = (
                    [float(value) for value in row["source_bbox_xywh"]]
                    == [float(value) for value in graph_bbox]
                )
            out[expr_id] = record
    return out


def _json_records(path: Path) -> List[Dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        records = list(payload.values())
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"{path}: expected a JSON object or array")
    records.sort(key=lambda row: int(row.get("id", row.get("sent_id", -1))))
    return records


def _validate_bbox_xywh(bbox: Any, image_width: Any, image_height: Any) -> Dict[str, Any]:
    """Check finite positive geometry and image intersection without clipping."""
    reasons: List[str] = []
    warnings: List[str] = []
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return {"xyxy": None, "status": "invalid", "reasons": ["bbox_not_four_values"], "warnings": []}
    try:
        x, y, box_width, box_height = (float(value) for value in bbox)
        width, height = float(image_width), float(image_height)
    except (TypeError, ValueError, OverflowError):
        return {"xyxy": None, "status": "invalid", "reasons": ["bbox_or_dimensions_not_numeric"], "warnings": []}
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        return {"xyxy": None, "status": "invalid", "reasons": ["bbox_has_nonfinite_value"], "warnings": []}
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return {"xyxy": [x, y, x + box_width, y + box_height], "status": "invalid", "reasons": ["image_dimensions_invalid"], "warnings": []}
    if box_width <= 0 or box_height <= 0:
        reasons.append("bbox_nonpositive_extent")
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(width, x + box_width), min(height, y + box_height)
    intersection_width, intersection_height = max(0.0, right - left), max(0.0, bottom - top)
    if intersection_width <= 0 or intersection_height <= 0:
        reasons.append("bbox_has_no_positive_image_intersection")
    if x < 0:
        warnings.append(f"bbox_overhang_left_px={-x:g}")
    if y < 0:
        warnings.append(f"bbox_overhang_top_px={-y:g}")
    if x + box_width > width:
        warnings.append(f"bbox_overhang_right_px={x + box_width - width:g}")
    if y + box_height > height:
        warnings.append(f"bbox_overhang_bottom_px={y + box_height - height:g}")
    return {
        "xyxy": [x, y, x + box_width, y + box_height],
        "status": "valid" if not reasons else "invalid",
        "reasons": reasons,
        "warnings": warnings,
        "image_intersection_xyxy": [left, top, right, bottom],
        "image_intersection_area": intersection_width * intersection_height,
    }


def _summarize_bbox_overhang(rows: Sequence[Mapping[str, Any]], warning_field: str) -> Dict[str, Any]:
    magnitudes: List[float] = []
    side_counts: Dict[str, int] = {}
    affected_images: Set[str] = set()
    for row in rows:
        row_warnings = row.get(warning_field) or []
        if row_warnings:
            affected_images.add(str(row.get("gqa_image_id", "")))
        row_magnitudes: List[float] = []
        for warning in row_warnings:
            try:
                side, raw_amount = str(warning).split("=", 1)
                amount = float(raw_amount)
            except (ValueError, TypeError):
                continue
            row_magnitudes.append(amount)
            side_counts[side] = side_counts.get(side, 0) + 1
        if row_magnitudes:
            magnitudes.append(max(row_magnitudes))
    return {
        "expression_count_with_overhang": len(magnitudes),
        "unique_image_count_with_overhang": len(affected_images - {""}),
        "max_overhang_px": max(magnitudes, default=0.0),
        "max_overhang_bins_expression_count": {
            "le_1px": sum(value <= 1 for value in magnitudes),
            "gt_1_le_2px": sum(1 < value <= 2 for value in magnitudes),
            "gt_2_le_5px": sum(2 < value <= 5 for value in magnitudes),
            "gt_5px": sum(value > 5 for value in magnitudes),
        },
        "overhang_edge_counts_nonexclusive": side_counts,
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _fetch_bytes(url: str, path: Path, expected_bytes: int | None = None) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=(30, 300))
    response.raise_for_status()
    payload = response.content
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise RuntimeError(f"{url}: received {len(payload)} bytes, expected {expected_bytes}")
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return payload


def fetch_metadata() -> Dict[str, Any]:
    """Fetch/verify official positive train/val files and the VG identity map."""
    ANNOTATIONS_V3.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    api_response = requests.get(FIGSHARE_API, timeout=(30, 60))
    api_response.raise_for_status()
    article = api_response.json()
    api_payload = json.dumps(article, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    api_path = ANNOTATIONS_V3 / "figshare_article_26048050.json"
    api_path.write_bytes(api_payload)
    by_id = {int(item["id"]): item for item in article["files"]}
    records: List[Dict[str, Any]] = []
    for file_id, local_name in FIGSHARE_FILES.items():
        source = by_id[file_id]
        path = ANNOTATIONS_V3 / local_name
        if path.exists() and path.stat().st_size == int(source["size"]):
            payload = path.read_bytes()
        else:
            payload = _fetch_bytes(source["download_url"], path, int(source["size"]))
        md5 = hashlib.md5(payload).hexdigest()
        if md5 != source["computed_md5"]:
            raise RuntimeError(f"Figshare MD5 mismatch for {source['name']} (file id {file_id})")
        records.append(
            {
                "file_id": file_id,
                "source_file_name": source["name"],
                "local_path": path.relative_to(REPO_ROOT).as_posix(),
                "download_url": source["download_url"],
                "bytes": len(payload),
                "figshare_computed_md5": source["computed_md5"],
                "sha256": hashlib.sha256(payload).hexdigest(),
                "mimetype": source.get("mimetype", ""),
            }
        )

    duplicate_val = [
        {
            "file_id": int(item["id"]),
            "source_file_name": item["name"],
            "bytes": int(item["size"]),
            "figshare_computed_md5": item["computed_md5"],
            "download_url": item["download_url"],
        }
        for item in article["files"]
        if int(item["id"]) in (47682898, 47682928)
    ]
    vg_path = SOURCE_OUT / "visual_genome_image_data.json.zip"
    if vg_path.exists() and sha256_file(vg_path) == VG_MAPPING_SHA256:
        vg_payload = vg_path.read_bytes()
    else:
        vg_payload = _fetch_bytes(VG_MIRROR_URL, vg_path, 1_780_854)
    vg_sha = hashlib.sha256(vg_payload).hexdigest()
    if vg_sha != VG_MAPPING_SHA256:
        raise RuntimeError(f"Visual Genome metadata SHA mismatch: {vg_sha}")
    with zipfile.ZipFile(vg_path) as archive:
        archive_entries = archive.namelist()
        if archive_entries != ["image_data.json"]:
            raise RuntimeError(f"unexpected Visual Genome metadata archive: {archive_entries}")
    manifest = {
        "article_id": 26048050,
        "article_url": "https://figshare.com/articles/dataset/FineCops-Ref_A_new_Dataset_and_Task_for_Fine-Grained_Compositional_Referring_Expression_Comprehension/26048050",
        "api_url": FIGSHARE_API,
        "doi": article.get("doi"),
        "citation": article.get("citation"),
        "license": article.get("license"),
        "metadata_sha256": hashlib.sha256(api_payload).hexdigest(),
        "positive_files": records,
        "duplicate_val_file_ids": duplicate_val,
        "val_duplicate_reconciled": len(duplicate_val) == 2
        and len({x["figshare_computed_md5"] for x in duplicate_val}) == 1,
        "finecops_article_image_source_license_caveat": "FineCops CC BY 4.0 applies to annotations, not source photographs. GQA says images are from COCO and Flickr; COCO does not own the photos and source terms apply; Flickr photo licenses vary. Individual photo licenses were not resolved, and no blanket photo license or redistribution permission is claimed.",
        "visual_genome_mapping": {
            "official_source_url": VG_OFFICIAL_MAPPING_URL,
            "official_api_reference": "https://homes.cs.washington.edu/~ranjay/visualgenome/api_readme.html",
            "download_mirror": VG_MIRROR_URL,
            "mirror_sha256": vg_sha,
            "official_source_sha256": VG_MAPPING_SHA256,
            "bytes": len(vg_payload),
            "archive_entries": archive_entries,
            "field_used": "image_id -> coco_id",
            "use": "Image identity mapping only; no object/category mapping is inferred.",
        },
        "fetched_at_utc": _utc_now(),
    }
    _json_dump(ANNOTATIONS_V3 / "source_manifest.json", manifest)
    return manifest


def _load_vg_metadata() -> Tuple[Dict[str, Dict[str, Any]], str]:
    path = SOURCE_OUT / "visual_genome_image_data.json.zip"
    with zipfile.ZipFile(path) as archive:
        records = json.load(archive.open("image_data.json"))
    result = {str(item["image_id"]): item for item in records}
    if len(result) != len(records):
        raise ValueError("Visual Genome mapping contains duplicate image_id values")
    return result, sha256_file(path)


def _load_finecops_splits() -> Dict[str, List[Dict[str, Any]]]:
    split_paths = {
        "train": ANNOTATIONS_V3 / FIGSHARE_FILES[47682931],
        "val": ANNOTATIONS_V3 / FIGSHARE_FILES[47682928],
        "test": OLD_FINECOPS_TEST,
    }
    return {split: _json_records(path) for split, path in split_paths.items()}


def _load_historical_split_ids() -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    from ccg.data.refcoco import image_ids_by_split_unc_pickle

    refcoco_plus = image_ids_by_split_unc_pickle(REPO_ROOT / "data/raw/refcoco+/refcoco+/refs(unc).p")
    refcocog_umd = image_ids_by_split_unc_pickle(REPO_ROOT / "data/raw/refcocog/refcocog/refs(umd).p")
    plus = {key: {str(value) for value in values.tolist()} for key, values in refcoco_plus.items()}
    refg = {key: {str(value) for value in values.tolist()} for key, values in refcocog_umd.items()}
    return plus, refg


def source_identity_audit() -> Dict[str, Any]:
    """Write lightweight source-ID overlap counts before any V3 inference."""
    from ccg.data.refcoco import image_ids_by_split_unc_pickle

    vg, vg_sha = _load_vg_metadata()
    splits = _load_finecops_splits()
    fine_ids = {split: {str(row["image_id"]) for row in rows} for split, rows in splits.items()}
    with (REPO_ROOT / "data/full_image_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        refplus_pool = {str(row["image_id"]) for row in csv.DictReader(stream)}
    refplus, refg = _load_historical_split_ids()
    refg_pool = set().union(*refg.values())
    gqa_graph_sets: Dict[str, Set[str]] = {}
    for name, path in (("train", GQA_TRAIN_GRAPHS), ("val", GQA_VAL_GRAPHS)):
        graph = _read_json(path)
        gqa_graph_sets[name] = set(map(str, graph.keys()))
        del graph

    mapped_by_split: Dict[str, Set[str]] = {}
    report: Dict[str, Any] = {
        "audit_date": "2026-10-04",
        "audit_rule_version": AUDIT_RULE_VERSION,
        "official_finecops_positive": {},
        "source_pools": {},
        "gqa_scene_graph_metadata": {},
        "cross_split_image_identity": {},
        "input_sha256": {},
    }
    for split, image_ids in fine_ids.items():
        gqa_mapped = {
            image_id
            for image_id in image_ids
            if image_id in vg and vg[image_id].get("coco_id") is not None
        }
        coco_ids = {str(vg[image_id]["coco_id"]) for image_id in gqa_mapped}
        mapped_by_split[split] = coco_ids
        report["official_finecops_positive"][split] = {
            "unique_gqa_image_ids": len(image_ids),
            "vg_metadata_id_covered": sum(image_id in vg for image_id in image_ids),
            "vg_metadata_id_missing": sum(image_id not in vg for image_id in image_ids),
            "mapped_to_coco_image_ids": len(gqa_mapped),
            "vg_records_with_null_coco_id": sum(
                image_id in vg and vg[image_id].get("coco_id") is None for image_id in image_ids
            ),
            "unique_coco_ids": len(coco_ids),
            "gqa_train_scenegraph_ids_with_metadata": len(image_ids & gqa_graph_sets["train"]),
            "gqa_val_scenegraph_ids_with_metadata": len(image_ids & gqa_graph_sets["val"]),
            "gqa_scenegraph_union_ids_with_metadata": len(
                image_ids & (gqa_graph_sets["train"] | gqa_graph_sets["val"])
            ),
            "coco_ids_overlapping_refcoco_plus_images": len(coco_ids & refplus_pool),
            "coco_ids_overlapping_refcocog_umd_images": len(coco_ids & refg_pool),
            "coco_ids_overlapping_finecops_other_splits": {},
        }
    for split in report["official_finecops_positive"]:
        report["official_finecops_positive"][split]["coco_ids_overlapping_finecops_other_splits"] = {
            other: len(mapped_by_split[split] & mapped_by_split[other])
            for other in mapped_by_split
            if other != split
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        report["cross_split_image_identity"][f"{left}__{right}"] = {
            "gqa_image_id_overlap": len(fine_ids[left] & fine_ids[right]),
            "coco_id_overlap": len(mapped_by_split[left] & mapped_by_split[right]),
        }

    report["source_pools"] = {
        "refcoco_plus_images_all_splits": len(refplus_pool),
        "refcoco_plus_images_by_split": {key: len(value) for key, value in refplus.items()},
        "refcocog_umd_images_by_split": {key: len(value) for key, value in refg.items()},
        "refcocog_umd_images_all_splits": len(refg_pool),
        "finecops_old_positive_test_metadata_images": len(fine_ids["test"]),
        "finecops_old_local_audit_images": len(list((REPO_ROOT / "data/raw/gqa/images").glob("*.jpg"))),
        "coco_local_images": len(list((REPO_ROOT / "data/raw/mscoco/train2014").glob("*.jpg"))),
        "refcocog_local_images": len(list((REPO_ROOT / "data/raw/refcocog/images/train2014").glob("*.jpg"))),
    }
    report["gqa_scene_graph_metadata"] = {
        "train_graph_image_ids": len(gqa_graph_sets["train"]),
        "val_graph_image_ids": len(gqa_graph_sets["val"]),
        "train_val_id_overlap": len(gqa_graph_sets["train"] & gqa_graph_sets["val"]),
        "historical_role": "GQA train_sceneGraphs.json and val_sceneGraphs.json were already downloaded; the old FineCops feasibility audit loaded the full val graph file. This records metadata availability, not model training/scoring.",
    }
    source_paths = [
        REPO_ROOT / "data/full_image_manifest.csv",
        REPO_ROOT / "data/audit_subset.csv",
        OLD_FINECOPS_TEST,
        GQA_TRAIN_GRAPHS,
        GQA_VAL_GRAPHS,
        REPO_ROOT / "data/raw/refcoco+/refcoco+/refs(unc).p",
        REPO_ROOT / "data/raw/refcocog/refcocog/refs(umd).p",
        SOURCE_OUT / "visual_genome_image_data.json.zip",
    ]
    report["input_sha256"] = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in source_paths
    }
    report["visual_genome_mapping_sha256"] = vg_sha
    report["note"] = "Identity intersections are conservative exclusions; mapped COCO image identity is not a COCO category mapping and does not imply historical model scoring."
    path = EXPOSURE_OUT / "source_identity_counts.json"
    _json_dump(path, report)
    return report


def build_cohort_rows() -> Dict[str, List[Dict[str, Any]]]:
    """Join official positive files without deduplicating expressions."""
    vg, _ = _load_vg_metadata()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "val"):
        vanilla = _json_records(ANNOTATIONS_V3 / FIGSHARE_FILES[47682931 if split == "train" else 47682928])
        coco = _read_json(
            ANNOTATIONS_V3 / FIGSHARE_FILES[47682937 if split == "train" else 47682934]
        )
        if set(coco) != {"images", "annotations", "categories"}:
            raise ValueError(f"{split}: unexpected COCO-format FineCops keys")
        image_by_id = {str(row["id"]): row for row in coco["images"]}
        annotation_by_id = {str(row["id"]): row for row in coco["annotations"]}
        if len(image_by_id) != len(coco["images"]) or len(annotation_by_id) != len(coco["annotations"]):
            raise ValueError(f"{split}: repeated expression/image record id in official file")
        if len(vanilla) != len(image_by_id) or len(vanilla) != len(annotation_by_id):
            raise ValueError(f"{split}: vanilla/COCO-format source counts disagree")
        rows: List[Dict[str, Any]] = []
        for record in vanilla:
            expr_id = str(record["id"])
            image_id = str(record["image_id"])
            image = image_by_id[expr_id]
            annotation = annotation_by_id[expr_id]
            source_file = Path(str(image["file_name"])).stem
            if source_file != image_id:
                raise ValueError(
                    f"{split} record {expr_id}: file_name stem {source_file} != GQA id {image_id}"
                )
            if str(image.get("caption", "")) != str(record.get("expression", "")):
                raise ValueError(f"{split} record {expr_id}: image caption differs from vanilla text")
            if str(annotation.get("expression", annotation.get("caption", ""))) != str(
                record.get("expression", "")
            ):
                raise ValueError(f"{split} record {expr_id}: annotation text differs from vanilla text")
            bbox_audit = _validate_bbox_xywh(
                annotation.get("bbox"), image.get("width"), image.get("height")
            )
            mapped = vg.get(image_id)
            rows.append(
                {
                    "expr_id": expr_id,
                    "sentence_id": expr_id,
                    "image_id": image_id,
                    "gqa_image_id": image_id,
                    "gt_boxxyxy": bbox_audit["xyxy"],
                    "source_bbox_xywh": annotation.get("bbox"),
                    "source_bbox_validation_status": bbox_audit["status"],
                    "source_bbox_validation_reasons": bbox_audit["reasons"],
                    "source_bbox_validation_warnings": bbox_audit["warnings"],
                    "expression": str(record["expression"]),
                    "level": int(record["level"]),
                    "source_split": split,
                    "source_namespace": "FineCops-Ref positive",
                    "source_article_id": 26048050,
                    "source_file_id_vanilla": 47682931 if split == "train" else 47682928,
                    "source_file_id_coco": 47682937 if split == "train" else 47682934,
                    "source_record_key": expr_id,
                    "tuple_type": record.get("tuple_type"),
                    "objects_id": record.get("objects_id"),
                    "tuple": record.get("tuple"),
                    "target_instance_id": annotation.get("instance_id", record.get("instance_id")),
                    "source_image_file_name": image["file_name"],
                    "source_image_width": int(image["width"]),
                    "source_image_height": int(image["height"]),
                    "vg_coco_id": mapped.get("coco_id") if mapped else None,
                    "vg_mapping_status": (
                        "mapped_to_coco_id"
                        if mapped and mapped.get("coco_id") is not None
                        else "vg_id_present_no_coco_id"
                        if mapped
                        else "unresolved_vg_image_id"
                    ),
                    "actual_image_path": (Path("data/raw/v3_finecops/images") / f"{image_id}.jpg").as_posix(),
                    "target_present": None,
                    "target_coverage": None,
                    "valid_target_proposal_count": None,
                    "metadata_downloaded": True,
                    "image_downloaded": False,
                    "audit_status": "pending_image_identity_audit",
                    "exposure_status": "pending_identity_and_pixel_audit",
                    "model_trained": False,
                    "model_scored": False,
                    "license": "FineCops annotation: CC BY 4.0; source photo license unresolved per image; GQA photos derive from COCO/Flickr and upstream terms apply.",
                    "audit_rule_version": AUDIT_RULE_VERSION,
                }
            )
        out[split] = rows
    return out


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _history_image_paths() -> List[Path]:
    """Find all locally available historical image inputs, without touching them."""
    roots = [
        REPO_ROOT / "data/raw/mscoco",
        REPO_ROOT / "data/raw/refcocog/images",
        REPO_ROOT / "data/raw/gqa/images",
        REPO_ROOT / "data/reviewed_refcocoplus",
        REPO_ROOT / "audit",
    ]
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(set(paths), key=lambda path: path.as_posix().lower())


def _load_historical_exposure() -> Dict[str, Any]:
    """Collect known historical image identities and use roles from read-only inputs."""
    plus_splits, refg_splits = _load_historical_split_ids()
    refplus_pool = set().union(*plus_splits.values())
    refg_pool = set().union(*refg_splits.values())
    old_test_records = _json_records(OLD_FINECOPS_TEST)
    old_finecops_gqa = {str(row["image_id"]) for row in old_test_records}
    old_finecops_expressions_by_image: Dict[str, int] = {}
    for row in old_test_records:
        key = str(row["image_id"])
        old_finecops_expressions_by_image[key] = old_finecops_expressions_by_image.get(key, 0) + 1

    manual_review_coco: Set[str] = set()
    manual_review_files: List[Dict[str, Any]] = []
    review_root = REPO_ROOT / "data/reviewed_refcocoplus/raw"
    for path in sorted(review_root.glob("*.json")):
        payload = _read_json(path)
        images = payload.get("images", []) if isinstance(payload, dict) else []
        ids = {str(row.get("original_id")) for row in images if row.get("original_id") is not None}
        manual_review_coco.update(ids)
        manual_review_files.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "image_records": len(images),
                "unique_original_coco_ids": len(ids),
            }
        )

    audit_ids: Set[str] = set()
    finecops_audit_csv = REPO_ROOT / "results/phase1e_finecops/audit_subset.csv"
    if finecops_audit_csv.exists():
        with finecops_audit_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = row.get("gqa_image_id") or row.get("image_id")
                if value:
                    audit_ids.add(str(value))
    audit_coco_ids: Set[str] = set()
    audit_csv = REPO_ROOT / "data/audit_subset.csv"
    if audit_csv.exists():
        with audit_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                value = row.get("image_id")
                if value:
                    audit_coco_ids.add(str(value))

    vg, _ = _load_vg_metadata()
    old_test_coco = {
        str(vg[image_id]["coco_id"])
        for image_id in old_finecops_gqa
        if image_id in vg and vg[image_id].get("coco_id") is not None
    }
    local_coco_ids: Set[str] = set()
    local_gqa_ids: Set[str] = set()
    local_unknown: List[str] = []
    for path in _history_image_paths():
        stem = path.stem
        if path.is_relative_to(REPO_ROOT / "data/raw/gqa/images") and stem.isdigit():
            local_gqa_ids.add(stem)
        elif stem.isdigit():
            local_coco_ids.add(stem)
        else:
            import re

            match = re.search(r"(?:COCO_)?(\d{1,12})$", stem, flags=re.IGNORECASE)
            if match:
                local_coco_ids.add(str(int(match.group(1))))
            else:
                local_unknown.append(path.relative_to(REPO_ROOT).as_posix())

    historical_coco = refplus_pool | refg_pool | manual_review_coco | audit_coco_ids | local_coco_ids | old_test_coco
    historical_gqa = old_finecops_gqa | local_gqa_ids | audit_ids
    roles: Dict[str, Set[str]] = {}
    for split, ids in plus_splits.items():
        for image_id in ids:
            roles.setdefault(image_id, set()).add(f"refcoco_plus_{split}")
    for split, ids in refg_splits.items():
        for image_id in ids:
            roles.setdefault(image_id, set()).add(f"refcocog_umd_{split}")
    for image_id in manual_review_coco:
        roles.setdefault(image_id, set()).add("human_reviewed_refcoco_plus")
    for image_id in old_finecops_gqa:
        roles.setdefault(image_id, set()).add("old_finecops_positive_test_metadata")
    for image_id in audit_ids:
        roles.setdefault(image_id, set()).add("old_finecops_proposal_audit_subset")
    for image_id in audit_coco_ids:
        roles.setdefault(image_id, set()).add("historical_refcoco_plus_audit_subset")
    for image_id in local_gqa_ids:
        roles.setdefault(image_id, set()).add("historical_local_gqa_image")
    for image_id in local_coco_ids:
        roles.setdefault(image_id, set()).add("historical_local_coco_image")

    return {
        "refcoco_plus_by_split": plus_splits,
        "refcocog_umd_by_split": refg_splits,
        "refcoco_plus_coco_ids": refplus_pool,
        "refcocog_umd_coco_ids": refg_pool,
        "manual_review_coco_ids": manual_review_coco,
        "old_finecops_test_gqa_ids": old_finecops_gqa,
        "old_finecops_test_coco_ids": old_test_coco,
        "historical_coco_ids": historical_coco,
        "historical_gqa_ids": historical_gqa,
        "local_historical_coco_ids": local_coco_ids,
        "local_historical_gqa_ids": local_gqa_ids,
        "audit_subset_gqa_ids": audit_ids,
        "audit_subset_coco_ids": audit_coco_ids,
        "old_finecops_expressions_by_image": old_finecops_expressions_by_image,
        "historical_roles_by_coco_id": roles,
        "manual_review_files": manual_review_files,
        "local_image_paths": _history_image_paths(),
        "unknown_local_image_paths": local_unknown,
    }


def _build_historical_identity_registry(history: Mapping[str, Any], vg: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Create an auditable row per known historical source identity/local file."""
    rows: List[Dict[str, Any]] = []

    def add(
        source: str,
        split: str,
        namespace: str,
        image_id: str,
        role: str,
        source_file: str,
        image_path: str = "",
    ) -> None:
        gqa_id = image_id if namespace == "gqa" else ""
        coco_id = image_id if namespace == "coco" else ""
        mapping_status = "not_applicable"
        mapped_coco_id = ""
        if namespace == "gqa":
            entry = vg.get(image_id)
            if entry is None:
                mapping_status = "missing_visual_genome_source_id"
            elif entry.get("coco_id") is None:
                mapping_status = "vg_record_has_null_coco_id"
            else:
                mapping_status = "mapped_to_coco_id"
                mapped_coco_id = str(entry["coco_id"])
        rows.append(
            {
                "source": source,
                "source_split": split,
                "identity_namespace": namespace,
                "source_image_id": image_id,
                "gqa_image_id": gqa_id,
                "coco_image_id": coco_id or mapped_coco_id,
                "gqa_to_coco_mapping_status": mapping_status,
                "exposure_role": role,
                "source_file": source_file,
                "local_image_path": image_path,
                "pixel_fingerprint_path": (
                    "historical_image_fingerprints.csv" if image_path else ""
                ),
                "audit_rule_version": AUDIT_RULE_VERSION,
            }
        )

    for split, image_ids in history["refcoco_plus_by_split"].items():
        for image_id in sorted(image_ids, key=lambda value: (int(value), value)):
            add(
                "RefCOCO+",
                str(split),
                "coco",
                image_id,
                "historical_refcoco_plus_cohort_identity; conservatively excluded across train/tune/test roles",
                "data/raw/refcoco+/refcoco+/refs(unc).p",
            )
    for split, image_ids in history["refcocog_umd_by_split"].items():
        for image_id in sorted(image_ids, key=lambda value: (int(value), value)):
            add(
                "RefCOCOg-UMD",
                str(split),
                "coco",
                image_id,
                "historical_refcocog_umd_cohort_identity; conservatively excluded across train/tune/test roles",
                "data/raw/refcocog/refcocog/refs(umd).p",
            )

    manual_files = REPO_ROOT / "data/reviewed_refcocoplus/raw"
    for path in sorted(manual_files.glob("*.json")):
        payload = _read_json(path)
        images = payload.get("images", []) if isinstance(payload, dict) else []
        image_ids = {
            str(image["original_id"])
            for image in images
            if image.get("original_id") is not None
        }
        for image_id in sorted(image_ids, key=lambda value: (int(value), value)):
            add(
                "RefCOCO+-Ref-L4-manual-review",
                path.stem,
                "coco",
                image_id,
                "human_annotation_review_identity; conservatively excluded",
                path.relative_to(REPO_ROOT).as_posix(),
            )

    for image_id in sorted(history["old_finecops_test_gqa_ids"], key=lambda value: (int(value), value)):
        add(
            "old FineCops-Ref positive test",
            "test",
            "gqa",
            image_id,
            "historical_test_metadata; old RPN audit subset is separately recorded",
            "data/raw/finecops/test_expression_pos.json",
        )
    for image_id in sorted(history["audit_subset_gqa_ids"], key=lambda value: (int(value), value)):
        add(
            "old FineCops proposal audit",
            "audit_subset",
            "gqa",
            image_id,
            "historical_RPN_proposal_recall_audit; not V3 scorer/reliability scoring",
            "results/phase1e_finecops/audit_subset.csv",
        )
    for image_id in sorted(history["audit_subset_coco_ids"], key=lambda value: (int(value), value)):
        add(
            "RefCOCO+ audit subset",
            "audit_subset",
            "coco",
            image_id,
            "historical_RefCOCO_plus_image_audit_identity; separate from old FineCops RPN audit",
            "data/audit_subset.csv",
        )

    for path in history["local_image_paths"]:
        relative = path.relative_to(REPO_ROOT).as_posix()
        stem = path.stem
        if path.is_relative_to(REPO_ROOT / "data/raw/gqa/images") and stem.isdigit():
            add(
                "locally cached GQA historical image",
                "local_cache",
                "gqa",
                stem,
                "historical_local_image_input; pixel fingerprint recorded separately",
                relative,
                relative,
            )
        else:
            import re

            match = re.search(r"(?:COCO_)?(\d{1,12})$", stem, flags=re.IGNORECASE)
            if match:
                image_id = str(int(match.group(1)))
                source = "locally cached RefCOCOg image" if path.is_relative_to(REPO_ROOT / "data/raw/refcocog/images") else "locally cached COCO image"
                add(
                    source,
                    "local_cache",
                    "coco",
                    image_id,
                    "historical_local_image_input; pixel fingerprint recorded separately",
                    relative,
                    relative,
                )
            else:
                add(
                    "unknown-origin local image",
                    "unknown",
                    "unknown",
                    relative,
                    "historical_local_image_input_with_unresolved_filename_identity",
                    relative,
                    relative,
                )

    return rows


def build_identity_candidate_manifest() -> Dict[str, Any]:
    """Exclude known historical and cross-split identities before acquisition."""
    vg, vg_sha = _load_vg_metadata()
    splits = _load_finecops_splits()
    history = _load_historical_exposure()
    history_registry_rows = _build_historical_identity_registry(history, vg)
    ids_by_split = {
        split: sorted({str(row["image_id"]) for row in rows}, key=lambda value: (int(value), value))
        for split, rows in splits.items()
    }
    coco_by_id: Dict[str, str | None] = {}
    for ids in ids_by_split.values():
        for image_id in ids:
            entry = vg.get(image_id)
            coco_by_id[image_id] = (
                str(entry["coco_id"])
                if entry is not None and entry.get("coco_id") is not None
                else None
            )
    map_count: Dict[str, int] = {}
    for image_id in ids_by_split["train"] + ids_by_split["val"]:
        coco_id = coco_by_id.get(image_id)
        if coco_id is not None:
            map_count[coco_id] = map_count.get(coco_id, 0) + 1
    duplicate_coco_groups: Dict[str, List[str]] = {}
    for split in ("train", "val"):
        by_coco: Dict[str, List[str]] = {}
        for image_id in ids_by_split[split]:
            coco_id = coco_by_id.get(image_id)
            if coco_id is not None:
                by_coco.setdefault(coco_id, []).append(image_id)
        for coco_id, image_ids in by_coco.items():
            if len(image_ids) > 1:
                duplicate_coco_groups[f"{split}:{coco_id}"] = image_ids

    rows: List[Dict[str, Any]] = []
    excluded: Dict[str, Dict[str, int]] = {"train": {}, "val": {}}
    eligible_by_split: Dict[str, List[str]] = {"train": [], "val": []}
    cross_split_coco = set()
    train_coco = {coco_by_id.get(i) for i in ids_by_split["train"] if coco_by_id.get(i) is not None}
    val_coco = {coco_by_id.get(i) for i in ids_by_split["val"] if coco_by_id.get(i) is not None}
    cross_split_coco = train_coco & val_coco
    for split in ("train", "val"):
        for image_id in ids_by_split[split]:
            coco_id = coco_by_id.get(image_id)
            reasons: List[str] = []
            if image_id not in vg:
                reasons.append("missing_visual_genome_source_id")
            if image_id in history["old_finecops_test_gqa_ids"]:
                reasons.append("old_finecops_test_gqa_image_id")
            if coco_id is not None:
                if coco_id in history["refcoco_plus_coco_ids"]:
                    reasons.append("historical_refcoco_plus_coco_image")
                if coco_id in history["refcocog_umd_coco_ids"]:
                    reasons.append("historical_refcocog_umd_coco_image")
                if coco_id in history["manual_review_coco_ids"]:
                    reasons.append("historical_human_reviewed_coco_image")
                if coco_id in history["old_finecops_test_coco_ids"]:
                    reasons.append("old_finecops_test_mapped_coco_image")
                if coco_id in cross_split_coco:
                    reasons.append("cross_split_mapped_coco_identity")
                if f"{split}:{coco_id}" in duplicate_coco_groups:
                    reasons.append("ambiguous_duplicate_coco_mapping_within_split")
            if reasons:
                for reason in reasons:
                    excluded[split][reason] = excluded[split].get(reason, 0) + 1
            else:
                eligible_by_split[split].append(image_id)
            row = {
                "source_split": split,
                "gqa_image_id": image_id,
                "vg_coco_id": coco_id,
                "vg_mapping_status": (
                    "mapped_to_coco_id"
                    if coco_id is not None
                    else "vg_record_has_null_coco_id"
                    if image_id in vg
                    else "missing_visual_genome_source_id"
                ),
                "metadata_downloaded": True,
                "historical_identity_excluded": bool(reasons),
                "identity_exclusion_reasons": ";".join(reasons),
                "metadata_identity_eligible": not reasons,
                "pixel_audit_status": "not_started",
                "pixel_sha256": "",
                "dhash64": "",
                "historical_exact_pixel_matches": "",
                "historical_dhash_le4_matches": "",
                "cross_split_pixel_matches": "",
                "final_eligible": False,
                "final_exclusion_reason": "pending_image_hash_audit",
                "audit_rule_version": AUDIT_RULE_VERSION,
            }
            rows.append(row)
    dev_rng = np.random.default_rng(SEED)
    train_order = [eligible_by_split["train"][int(i)] for i in dev_rng.permutation(len(eligible_by_split["train"]))]
    identity_report = {
        "schema": "v3-source-identity-audit-v1",
        "audit_date": "2026-10-04",
        "audit_rule_version": AUDIT_RULE_VERSION,
        "vg_mapping_sha256": vg_sha,
        "finecops_positive_counts": {
            split: {
                "expressions": len(splits[split]),
                "unique_gqa_images": len(ids_by_split[split]),
                "unique_mapped_coco_ids": len({coco_by_id[i] for i in ids_by_split[split] if coco_by_id[i] is not None}),
                "source_mapping_status_counts": {
                    "mapped_to_coco_id": sum(coco_by_id[i] is not None for i in ids_by_split[split]),
                    "vg_record_has_null_coco_id": sum(
                        i in vg and vg[i].get("coco_id") is None for i in ids_by_split[split]
                    ),
                    "missing_visual_genome_source_id": sum(i not in vg for i in ids_by_split[split]),
                },
                "metadata_identity_eligible_images": len(eligible_by_split[split]),
                "metadata_excluded_images": len(ids_by_split[split]) - len(eligible_by_split[split]),
                "exclusions_by_reason_nonexclusive": excluded[split],
            }
            for split in ("train", "val")
        },
        "historical_sources": {
            "refcoco_plus_images_by_split": {key: len(value) for key, value in history["refcoco_plus_by_split"].items()},
            "refcocog_umd_images_by_split": {key: len(value) for key, value in history["refcocog_umd_by_split"].items()},
            "manual_review_files": history["manual_review_files"],
            "unique_manual_review_coco_ids": len(history["manual_review_coco_ids"]),
            "old_finecops_test_images": len(history["old_finecops_test_gqa_ids"]),
            "old_finecops_test_expressions": len(_json_records(OLD_FINECOPS_TEST)),
            "old_finecops_proposal_audit_images": len(history["audit_subset_gqa_ids"]),
            "old_refcoco_plus_audit_subset_images": len(history["audit_subset_coco_ids"]),
            "local_historical_image_files": len(history["local_image_paths"]),
            "historical_identity_registry_rows": len(history_registry_rows),
            "local_historical_coco_ids": len(history["local_historical_coco_ids"]),
            "local_historical_gqa_ids": len(history["local_historical_gqa_ids"]),
            "unidentified_local_image_files": history["unknown_local_image_paths"],
            "historical_image_use_roles": {
                "refcoco_plus": "original train/val/testA/testB cohort; older model train/tune/test roles are source split-specific and conservatively excluded as a full image pool",
                "refcocog_umd": "official train/val/test image pool, conservatively excluded as a full source pool",
                "reviewed_refcoco_plus": "manual annotation review source; original COCO IDs mapped through annotations, conservatively excluded",
                "finecops_test": "old positive test metadata 4,313 GQA images; prior proposal/RPN audit used a 1,000-image subset; no old V3 scorer fit authorized",
            },
            "unknown_scope_note": "Image files under audit/ and data/reviewed_refcocoplus were enumerated read-only; image files with unparseable source identity remain a conservative audit blocker until pixel matching resolves them.",
        },
        "mapping_ambiguity": {
            "train_val_shared_coco_ids_excluded_from_both": len(cross_split_coco),
            "within_split_duplicate_coco_groups_excluded": len(duplicate_coco_groups),
            "within_split_duplicate_coco_images_excluded": sum(len(values) for values in duplicate_coco_groups.values()),
            "groups": duplicate_coco_groups,
        },
        "development_selection": {
            "seed": SEED,
            "rng": "numpy.random.default_rng(20261004) over sorted metadata-eligible train GQA image IDs",
            "maximum_images": DEV_IMAGE_CAP,
            "selection": "walk frozen permutation; hash selected candidate images; exclude exact/near historical matches and candidates duplicating confirmation/previous dev, then continue in order until 200 pass or pool exhausted",
            "candidate_order_sha256": hashlib.sha256("\n".join(train_order).encode()).hexdigest(),
            "candidate_order_count": len(train_order),
        },
        "perceptual_rule_frozen_before_confirmation_scores": {
            "version": AUDIT_RULE_VERSION,
            "pixel_identity": "SHA-256 over EXIF-oriented RGB decoded pixels preceded by version tag and little-endian width/height",
            "near_duplicate_screen": f"{DHASH_ALGORITHM}; Hamming distance <= {DHASH_DISTANCE_MAX}",
            "limits": "A dHash match is a conservative screening flag, not proof of identity; missed crop, resize, color, or small-content changes remain possible. Ambiguous matches are excluded; no conclusion that every near duplicate is found.",
        },
        "gqa_annotation_metadata_exposure": {
            "train_sceneGraphs_image_ids": 74942,
            "val_sceneGraphs_image_ids": 10696,
            "historically_loaded_val_sceneGraphs": True,
            "historical_val_graph_use": "The prior FineCops feasibility stage loaded the full val graph dictionary and queried it for old FineCops test image IDs only; graph file availability is not scored/model exposure for all IDs.",
            "train_sceneGraphs_loaded_during_this_read_only_identity_audit": True,
            "v3_train_ids_with_graph_metadata": 31795,
            "v3_val_ids_with_graph_metadata": 3567,
            "interpretation": "Bulk scene-graph metadata availability is disclosed separately; this audit does not claim the FineCops expressions were previously scored or that images/annotations were unseen by pretraining.",
        },
        "status": "IDENTITY_FILTERS_BUILT_PIXEL_AUDIT_PENDING",
    }
    _write_rows(EXPOSURE_OUT / "identity_candidate_manifest.csv", rows)
    _write_rows(EXPOSURE_OUT / "historical_identity_registry.csv", history_registry_rows)
    _write_rows(
        EXPOSURE_OUT / "development_candidate_order.csv",
        [{"rank": rank + 1, "gqa_image_id": image_id, "seed": SEED} for rank, image_id in enumerate(train_order)],
    )
    identity_report["historical_sources"]["historical_identity_registry_sha256"] = sha256_file(
        EXPOSURE_OUT / "historical_identity_registry.csv"
    )
    _json_dump(EXPOSURE_OUT / "identity_audit.json", identity_report)
    return identity_report


def _fingerprint_rows(paths: Sequence[Path], workers: int = 4) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Hash local images deterministically and record decode failures explicitly."""
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    def one(path: Path) -> Tuple[Path, Any, str | None]:
        try:
            return path, fingerprint_image(path), None
        except Exception as exc:  # noqa: BLE001 - each unreadable file must be reported
            return path, None, f"{type(exc).__name__}: {exc}"
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(one, path) for path in paths]
        results = [future.result() for future in as_completed(futures)]
    for path, fingerprint, error in sorted(results, key=lambda item: item[0].as_posix().lower()):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if fingerprint is None:
            failures.append({"path": relative, "error": str(error)})
            continue
        rows.append(
            {
                "path": relative,
                "file_sha256": fingerprint.file_sha256,
                "pixel_sha256": fingerprint.pixel_sha256,
                "dhash64": fingerprint.dhash64,
                "width": fingerprint.width,
                "height": fingerprint.height,
                "audit_rule_version": AUDIT_RULE_VERSION,
            }
        )
    return rows, failures


def fingerprint_history(workers: int = 4) -> Dict[str, Any]:
    """Perform exact-pixel and frozen dHash screening over local historical images."""
    paths = _history_image_paths()
    t0 = time.time()
    rows, failures = _fingerprint_rows(paths, workers=workers)
    _write_rows(EXPOSURE_OUT / "historical_image_fingerprints.csv", rows)
    report = {
        "schema": "v3-historical-image-fingerprint-audit-v1",
        "audit_rule_version": AUDIT_RULE_VERSION,
        "image_files_discovered": len(paths),
        "images_decoded_and_hashed": len(rows),
        "decode_failures": failures,
        "path_inventory_sha256": hashlib.sha256("\n".join(p.relative_to(REPO_ROOT).as_posix() for p in paths).encode()).hexdigest(),
        "fingerprint_csv_sha256": sha256_file(EXPOSURE_OUT / "historical_image_fingerprints.csv"),
        "seconds": round(time.time() - t0, 2),
        "status": "PASS" if not failures else "BLOCKED_CONFIRMATION",
    }
    _json_dump(EXPOSURE_OUT / "historical_fingerprint_status.json", report)
    return report


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _identity_pool(split: str) -> List[str]:
    rows = [row for row in _read_csv_rows(EXPOSURE_OUT / "identity_candidate_manifest.csv") if row["source_split"] == split]
    return [row["gqa_image_id"] for row in rows if row["metadata_identity_eligible"].lower() == "true"]


def _historical_hash_indices() -> Tuple[Dict[str, List[str]], BKTree, List[Dict[str, str]]]:
    rows = _read_csv_rows(EXPOSURE_OUT / "historical_image_fingerprints.csv")
    exact: Dict[str, List[str]] = {}
    tree = BKTree()
    for row in rows:
        exact.setdefault(row["pixel_sha256"], []).append(row["path"])
        tree.add(row["dhash64"], row["path"])
    return exact, tree, rows


def _new_image_path(image_id: str) -> Path:
    return V3_IMAGES / f"{image_id}.jpg"


def _image_collection_identity_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash a canonical ID + byte/pixel/perceptual fingerprint inventory."""
    fields = ("gqa_image_id", "file_sha256", "pixel_sha256", "dhash64")
    lines = ["\t".join(str(row.get(field, "")) for field in fields) for row in sorted(rows, key=lambda value: str(value.get("gqa_image_id", "")))]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _write_fetch_report(
    name: str, report: Any, requested_ids: Sequence[str], workers: int
) -> None:
    value = report.to_dict()
    value.update({"requested_image_ids_sha256": hashlib.sha256("\n".join(requested_ids).encode()).hexdigest(), "workers": int(workers), "out_dir": V3_IMAGES.relative_to(REPO_ROOT).as_posix(), "completed_at_utc": _utc_now()})
    _json_dump(EXPOSURE_OUT / f"fetch_{name}.json", value)


def fetch_confirmation_images(workers: int = 3) -> Dict[str, Any]:
    """Resumably fetch only metadata-clean positive validation images."""
    if not 1 <= int(workers) <= 3:
        raise ValueError("network concurrency must be between 1 and 3")
    if not (EXPOSURE_OUT / "identity_candidate_manifest.csv").exists():
        build_identity_candidate_manifest()
    clean_val = _identity_pool("val")
    report = fetch_images(clean_val, V3_IMAGES, workers=int(workers), log=lambda message: print(message, flush=True))
    _write_fetch_report("confirmation", report, clean_val, workers)
    result = report.to_dict()
    result.update({"eligible_identity_clean_val_image_ids": len(clean_val), "image_root": V3_IMAGES.relative_to(REPO_ROOT).as_posix(), "historical_namespace_mutated": False})
    _json_dump(EXPOSURE_OUT / "fetch_confirmation_status.json", result)
    return result


def acquire_confirmation_and_development(workers: int = 3, batch_size: int = 50) -> Dict[str, Any]:
    """Fetch eligible val, audit it, then fill a deterministic dev pool to 200."""
    workers = int(workers)
    if not 1 <= workers <= 3:
        raise ValueError("network concurrency must be between 1 and 3")
    if not (EXPOSURE_OUT / "identity_candidate_manifest.csv").exists():
        build_identity_candidate_manifest()
    if not (EXPOSURE_OUT / "historical_image_fingerprints.csv").exists():
        raise RuntimeError("run --stage history before image acquisition")

    clean_val = _identity_pool("val")
    identity_by_id = {
        row["gqa_image_id"]: row for row in _read_csv_rows(EXPOSURE_OUT / "identity_candidate_manifest.csv")
    }
    fetch_report = fetch_images(clean_val, V3_IMAGES, workers=workers, log=lambda message: print(message, flush=True))
    _write_fetch_report("confirmation", fetch_report, clean_val, workers)

    hist_exact, hist_tree, _ = _historical_hash_indices()
    val_records: List[Dict[str, Any]] = []
    val_failures: List[Dict[str, str]] = []
    val_paths = [_new_image_path(image_id) for image_id in clean_val if _new_image_path(image_id).exists()]
    fingerprint_rows, val_failures = _fingerprint_rows(val_paths, workers=4)
    fp_by_id = {Path(row["path"]).stem: row for row in fingerprint_rows}
    val_failure_by_id = {
        Path(row["path"]).stem: row["error"] for row in val_failures
    }
    for image_id in clean_val:
        fp = fp_by_id.get(image_id)
        source_identity = identity_by_id[image_id]
        row: Dict[str, Any] = {
            "gqa_image_id": image_id,
            "source_split": "val",
            "metadata_identity_eligible": True,
            "vg_coco_id": source_identity["vg_coco_id"],
            "vg_mapping_status": source_identity["vg_mapping_status"],
            "identity_exclusion_reasons": source_identity["identity_exclusion_reasons"],
        }
        if fp is None:
            row.update(
                {
                    "audit_status": "image_missing_or_decode_failed",
                    "audit_error": val_failure_by_id.get(image_id, "image file missing"),
                    "final_eligible": False,
                    "final_exclusion_reason": "unverified_image",
                }
            )
            val_records.append(row)
            continue
        exact_matches = hist_exact.get(fp["pixel_sha256"], [])
        near_matches = hist_tree.query(fp["dhash64"], DHASH_DISTANCE_MAX)
        matched_history = sorted(set(exact_matches) | {path for path, _ in near_matches})
        row.update(
            {
                "image_path": fp["path"],
                "file_sha256": fp["file_sha256"],
                "pixel_sha256": fp["pixel_sha256"],
                "dhash64": fp["dhash64"],
                "width": int(fp["width"]),
                "height": int(fp["height"]),
                "historical_exact_pixel_matches": exact_matches,
                "historical_dhash_le4_matches": near_matches,
                "audit_status": "hashed",
                "final_eligible": not matched_history,
                "final_exclusion_reason": "historic_exact_or_dhash_match" if matched_history else "",
            }
        )
        val_records.append(row)

    # Remove whole near-duplicate groups within validation; this is conservative
    # and deterministic and prevents same-content images from receiving split weight.
    val_tree = BKTree((row["dhash64"], row["gqa_image_id"]) for row in val_records if row.get("dhash64"))
    val_by_id = {row["gqa_image_id"]: row for row in val_records}
    val_peer_matches: Dict[str, List[Tuple[str, int]]] = {}
    for row in val_records:
        if not row.get("dhash64"):
            continue
        peers = [(key, distance) for key, distance in val_tree.query(row["dhash64"], DHASH_DISTANCE_MAX) if key != row["gqa_image_id"]]
        if peers:
            val_peer_matches[row["gqa_image_id"]] = peers
    for image_id, peers in val_peer_matches.items():
        row = val_by_id[image_id]
        row["within_confirmation_dhash_matches"] = peers
        row["final_eligible"] = False
        prior = row.get("final_exclusion_reason") or ""
        row["final_exclusion_reason"] = ";".join(filter(None, [prior, "within_confirmation_dhash_cluster"]))

    hashed_val_fingerprints = [row for row in val_records if row.get("dhash64")]
    val_exact_eligible = {row["pixel_sha256"] for row in hashed_val_fingerprints}
    val_tree_eligible = BKTree((row["dhash64"], row["gqa_image_id"]) for row in hashed_val_fingerprints)
    clean_val_fingerprints = [row for row in val_records if row.get("final_eligible") and row.get("dhash64")]

    train_order = [row["gqa_image_id"] for row in _read_csv_rows(EXPOSURE_OUT / "development_candidate_order.csv")]
    selected: List[Dict[str, Any]] = []
    rejected_dev: List[Dict[str, Any]] = []
    scanned_ids: Set[str] = set()
    next_rank = 0
    while len(selected) < DEV_IMAGE_CAP and next_rank < len(train_order):
        batch_ids = []
        while next_rank < len(train_order) and len(batch_ids) < max(1, batch_size):
            candidate = train_order[next_rank]
            next_rank += 1
            if candidate not in scanned_ids:
                scanned_ids.add(candidate)
                batch_ids.append(candidate)
        if not batch_ids:
            break
        dev_fetch = fetch_images(batch_ids, V3_IMAGES, workers=workers, log=lambda message: print(message, flush=True))
        _write_fetch_report(
            f"development_batch_{len(scanned_ids):06d}", dev_fetch, batch_ids, workers
        )
        local_paths = [_new_image_path(image_id) for image_id in batch_ids if _new_image_path(image_id).exists()]
        batch_fps, batch_failures = _fingerprint_rows(local_paths, workers=4)
        by_id = {Path(row["path"]).stem: row for row in batch_fps}
        failed_paths = {Path(row["path"]).stem: row["error"] for row in batch_failures}
        for image_id in batch_ids:
            if len(selected) >= DEV_IMAGE_CAP:
                break
            fp = by_id.get(image_id)
            if fp is None:
                rejected_dev.append({"gqa_image_id": image_id, "reason": "download_or_decode_failure", "detail": failed_paths.get(image_id, "image file missing")})
                continue
            exact_hist = hist_exact.get(fp["pixel_sha256"], [])
            near_hist = hist_tree.query(fp["dhash64"], DHASH_DISTANCE_MAX)
            exact_val = fp["pixel_sha256"] in val_exact_eligible
            near_val = val_tree_eligible.query(fp["dhash64"], DHASH_DISTANCE_MAX)
            exact_dev = any(row["pixel_sha256"] == fp["pixel_sha256"] for row in selected)
            near_dev = any(hamming64(row["dhash64"], fp["dhash64"]) <= DHASH_DISTANCE_MAX for row in selected)
            reasons = []
            if exact_hist:
                reasons.append("historic_exact_pixel_match")
            if near_hist:
                reasons.append("historic_dhash_le4_match")
            if exact_val or near_val:
                reasons.append("confirmation_exact_or_dhash_match")
            if exact_dev or near_dev:
                reasons.append("selected_development_exact_or_dhash_match")
            record = {
                "gqa_image_id": image_id,
                "source_split": "train",
                "vg_coco_id": identity_by_id[image_id]["vg_coco_id"],
                "vg_mapping_status": identity_by_id[image_id]["vg_mapping_status"],
                "identity_exclusion_reasons": identity_by_id[image_id]["identity_exclusion_reasons"],
                "image_path": fp["path"],
                "file_sha256": fp["file_sha256"],
                "pixel_sha256": fp["pixel_sha256"],
                "dhash64": fp["dhash64"],
                "width": int(fp["width"]),
                "height": int(fp["height"]),
                "audit_status": "hashed",
                "historical_exact_pixel_matches": exact_hist,
                "historical_dhash_le4_matches": near_hist,
                "confirmation_exact_pixel_match": exact_val,
                "confirmation_dhash_le4_matches": near_val,
                "development_exact_pixel_match": exact_dev,
                "development_dhash_le4_match": near_dev,
                "final_eligible": not reasons,
                "final_exclusion_reason": ";".join(reasons),
                "seed_rank": train_order.index(image_id) + 1,
                "backfill_attempt": len(scanned_ids),
            }
            if reasons:
                rejected_dev.append(record)
            else:
                selected.append(record)

    # Preserve the entire official positive validation expression cohort; image
    # eligibility is repeated onto each expression without collapsing duplicates.
    cohort_rows = build_cohort_rows()
    source_bbox_invalid_rows = [
        row
        for split_rows in cohort_rows.values()
        for row in split_rows
        if row["source_bbox_validation_status"] != "valid"
    ]
    dev_ids = {row["gqa_image_id"] for row in selected}
    val_image_screen_ids = {row["gqa_image_id"] for row in clean_val_fingerprints}
    val_audit = {row["gqa_image_id"]: row for row in val_records}
    dev_audit = {row["gqa_image_id"]: row for row in selected}
    invalid_positive_boxes_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for row in cohort_rows["val"]:
        image_id = row["gqa_image_id"]
        if image_id not in val_image_screen_ids:
            continue
        fingerprint = val_audit[image_id]
        geometry = _validate_bbox_xywh(
            row["source_bbox_xywh"], fingerprint.get("width"), fingerprint.get("height")
        )
        if geometry["status"] != "valid":
            if "bbox_has_no_positive_image_intersection" in geometry["reasons"]:
                reason = "official_gt_box_has_no_image_intersection"
            elif "bbox_nonpositive_extent" in geometry["reasons"]:
                reason = "official_gt_box_has_nonpositive_extent"
            else:
                reason = "official_gt_box_has_nonfinite_or_unreadable_geometry"
            invalid_positive_boxes_by_image.setdefault(image_id, []).append(
                {
                    "expr_id": row["expr_id"],
                    "sentence_id": row["sentence_id"],
                    "target_instance_id": row.get("target_instance_id"),
                    "source_bbox_xywh": row["source_bbox_xywh"],
                    "decoded_jpeg_bbox_validation_reasons": geometry["reasons"],
                    "reason": reason,
                }
            )
    for image_id, invalid_rows in invalid_positive_boxes_by_image.items():
        image_record = val_audit[image_id]
        image_record["final_eligible"] = False
        image_record["prescore_annotation_exclusion_reasons"] = sorted(
            {row["reason"] for row in invalid_rows}
        )
        prior = image_record.get("final_exclusion_reason") or ""
        image_record["final_exclusion_reason"] = ";".join(
            filter(None, [prior, *image_record["prescore_annotation_exclusion_reasons"]])
        )
    prescore_excluded_image_ids = set(invalid_positive_boxes_by_image)
    val_clean_ids = val_image_screen_ids - prescore_excluded_image_ids
    gqa_target_audit = _audit_gqa_target_boxes(
        cohort_rows, {"train": dev_ids, "val": val_image_screen_ids}
    )
    prescore_annotation_exclusions: List[Dict[str, Any]] = []
    for image_id, invalid_rows in sorted(invalid_positive_boxes_by_image.items()):
        graph_comparisons = [gqa_target_audit.get(row["expr_id"], {}) for row in invalid_rows]
        source_rows_on_image = [row for row in cohort_rows["val"] if row["gqa_image_id"] == image_id]
        fingerprint = val_audit[image_id]
        prescore_annotation_exclusions.append(
            {
                "source_split": "val",
                "gqa_image_id": image_id,
                "excluded_image_path": fingerprint.get("image_path"),
                "official_expression_ids_preserved_in_source": [row["expr_id"] for row in source_rows_on_image],
                "offending_annotations": invalid_rows,
                "decoded_jpeg_dimensions": [fingerprint.get("width"), fingerprint.get("height")],
                "decoded_jpeg_file_sha256": fingerprint.get("file_sha256"),
                "decoded_pixel_sha256": fingerprint.get("pixel_sha256"),
                "dhash64": fingerprint.get("dhash64"),
                "historical_exact_pixel_matches": fingerprint.get("historical_exact_pixel_matches", []),
                "historical_dhash_le4_matches": fingerprint.get("historical_dhash_le4_matches", []),
                "source_metadata_dimensions": [source_rows_on_image[0]["source_image_width"], source_rows_on_image[0]["source_image_height"]] if source_rows_on_image else None,
                "gqa_graph_dimensions": [graph_comparisons[0].get("gqa_graph_width"), graph_comparisons[0].get("gqa_graph_height")] if graph_comparisons else None,
                "source_agreement_with_gqa_target": [
                    {
                        "target_instance_id": comparison.get("target_instance_id"),
                        "gqa_target_bbox_xywh": comparison.get("gqa_target_bbox_xywh"),
                        "exact_match": comparison.get("official_bbox_equals_gqa_target_bbox"),
                    }
                    for comparison in graph_comparisons
                ],
                "reason": invalid_rows[0]["reason"],
                "model_training_observed": False,
                "reliability_scoring_observed": False,
                "performance_metrics_observed": False,
            }
        )
    prescore_exclusion_report = {
        "schema": "v3-prescore-positive-box-exclusion-v1",
        "rule_id": "official-positive-box-visible-intersection",
        "rule_frozen_at_utc": _utc_now(),
        "performance_results_observed_before_rule": False,
        "model_training_observed": False,
        "reliability_scoring_observed": False,
        "performance_metrics_observed": False,
        "rule": "For a candidate confirmation image, exclude the whole image before scoring if any official positive-expression xywh annotation is nonfinite, has nonpositive extent, or has no positive-area intersection with the decoded JPEG. Preserve the original annotation and image in the source records and exclusion ledger. Never clip, impute, union, or silently drop boxes. Intersecting edge overhangs remain eligible and are reported. Source/GQA/JPEG dimension or identity disagreement blocks rather than being handled by this annotation rule.",
        "excluded_image_count": len(prescore_annotation_exclusions),
        "excluded_offending_expression_count": sum(len(row["offending_annotations"]) for row in prescore_annotation_exclusions),
        "excluded_images": prescore_annotation_exclusions,
    }
    dev_manifest: List[Dict[str, Any]] = []
    confirmation_manifest: List[Dict[str, Any]] = []
    exposure_manifest: List[Dict[str, Any]] = []
    bbox_annotation_audit_rows: List[Dict[str, Any]] = []
    for split in ("train", "val"):
        eligible_images = dev_ids if split == "train" else val_clean_ids
        image_audit = dev_audit if split == "train" else val_audit
        for row in cohort_rows[split]:
            image_id = row["gqa_image_id"]
            if image_id not in eligible_images:
                continue
            fingerprint = image_audit[image_id]
            actual_bbox = _validate_bbox_xywh(
                row["source_bbox_xywh"], fingerprint.get("width"), fingerprint.get("height")
            )
            dimension_mismatch = (
                int(row["source_image_width"]) != int(fingerprint.get("width", -1))
                or int(row["source_image_height"]) != int(fingerprint.get("height", -1))
            )
            bbox_reasons = list(actual_bbox["reasons"])
            if row["source_bbox_validation_status"] != "valid":
                bbox_reasons.extend(
                    f"source_metadata:{reason}"
                    for reason in row["source_bbox_validation_reasons"]
                )
            if dimension_mismatch:
                bbox_reasons.append("source_metadata_dimensions_differ_from_decoded_jpeg")
            gqa_target = gqa_target_audit.get(row["expr_id"], {"gqa_graph_status": "audit_record_missing"})
            gqa_graph_dims = (gqa_target.get("gqa_graph_width"), gqa_target.get("gqa_graph_height"))
            source_gqa_dimension_mismatch = (
                gqa_graph_dims[0] is not None
                and gqa_graph_dims[1] is not None
                and (int(row["source_image_width"]) != int(gqa_graph_dims[0])
                     or int(row["source_image_height"]) != int(gqa_graph_dims[1]))
            )
            graph_jpeg_dimension_mismatch = (
                gqa_graph_dims[0] is not None
                and gqa_graph_dims[1] is not None
                and (int(gqa_graph_dims[0]) != int(fingerprint.get("width", -1))
                     or int(gqa_graph_dims[1]) != int(fingerprint.get("height", -1)))
            )
            if source_gqa_dimension_mismatch or graph_jpeg_dimension_mismatch:
                bbox_reasons.append("gqa_graph_dimensions_differ_from_source_or_decoded_jpeg")
            bbox_warnings = sorted(
                set(actual_bbox.get("warnings", [])) | set(row.get("source_bbox_validation_warnings", []))
            )
            bbox_annotation_audit_rows.append(
                {
                    "expr_id": row["expr_id"],
                    "sentence_id": row["sentence_id"],
                    "gqa_image_id": image_id,
                    "target_instance_id": row.get("target_instance_id"),
                    "source_split": split,
                    "source_bbox_xywh": row["source_bbox_xywh"],
                    "gt_boxxyxy_official_no_clip": actual_bbox["xyxy"],
                    "source_bbox_validation_status": row["source_bbox_validation_status"],
                    "source_bbox_validation_reasons": row["source_bbox_validation_reasons"],
                    "source_bbox_validation_warnings": row["source_bbox_validation_warnings"],
                    "source_image_width": row["source_image_width"],
                    "source_image_height": row["source_image_height"],
                    "decoded_jpeg_width": fingerprint.get("width"),
                    "decoded_jpeg_height": fingerprint.get("height"),
                    "decoded_jpeg_bbox_validation_status": actual_bbox["status"],
                    "decoded_jpeg_bbox_validation_warnings": actual_bbox.get("warnings", []),
                    "dimension_match": not dimension_mismatch,
                    "source_dimensions_match_gqa_graph": not source_gqa_dimension_mismatch,
                    "gqa_graph_dimensions_match_decoded_jpeg": not graph_jpeg_dimension_mismatch,
                    "gqa_graph_status": gqa_target.get("gqa_graph_status"),
                    "gqa_graph_split": gqa_target.get("gqa_graph_split"),
                    "gqa_graph_width": gqa_target.get("gqa_graph_width"),
                    "gqa_graph_height": gqa_target.get("gqa_graph_height"),
                    "gqa_target_bbox_xywh": gqa_target.get("gqa_target_bbox_xywh"),
                    "official_bbox_equals_gqa_target_bbox": gqa_target.get("official_bbox_equals_gqa_target_bbox"),
                    "released_bbox_validation_status": "valid" if not bbox_reasons else "invalid",
                    "released_bbox_validation_reasons": sorted(set(bbox_reasons)),
                    "released_bbox_validation_warnings": bbox_warnings,
                }
            )
            canonical = {
                key: row[key]
                for key in (
                    "expr_id", "sentence_id", "image_id", "gqa_image_id", "gt_boxxyxy", "expression", "level", "source_split", "actual_image_path"
                )
            }
            canonical.update(
                {
                    "target_present": None,
                    "target_coverage": None,
                    "valid_target_proposal_count": None,
                    "metadata_downloaded": True,
                    "image_downloaded": True,
                    "image_audit_pass": True,
                    "model_trained": False,
                    "model_scored": False,
                    "audit_rule_version": AUDIT_RULE_VERSION,
                }
            )
            if split == "train":
                dev_manifest.append(canonical)
            else:
                confirmation_manifest.append(canonical)
            exposure_manifest.append(
                {
                    "gqa_image_id": image_id,
                    "source_split": split,
                    "vg_coco_id": row.get("vg_coco_id"),
                    "vg_mapping_status": fingerprint.get("vg_mapping_status", row["vg_mapping_status"]),
                    "identity_exclusion_reasons": fingerprint.get("identity_exclusion_reasons", []),
                    "image_path": fingerprint.get("image_path", ""),
                    "file_sha256": fingerprint.get("file_sha256", ""),
                    "pixel_sha256": fingerprint.get("pixel_sha256", ""),
                    "dhash64": fingerprint.get("dhash64", ""),
                    "width": fingerprint.get("width", ""),
                    "height": fingerprint.get("height", ""),
                    "metadata_downloaded": True,
                    "image_downloaded": True,
                    "image_audited": True,
                    "model_trained": False,
                    "reliability_model_scored": False,
                    "proposal_scored": False,
                    "manual_historical_review": False,
                    "source_identity_eligible": True,
                    "pixel_audit_eligible": True,
                    "final_eligible": True,
                    "matched_historical_paths": fingerprint.get("historical_exact_pixel_matches", []),
                    "matched_historical_near_duplicates": fingerprint.get("historical_dhash_le4_matches", []),
                    "audit_rule_version": AUDIT_RULE_VERSION,
                }
            )
    _write_rows(EXPOSURE_OUT / "dev_image_manifest.csv", selected)
    _write_rows(EXPOSURE_OUT / "dev_rejected_candidates.csv", rejected_dev)
    _write_rows(EXPOSURE_OUT / "confirmation_image_manifest.csv", val_records)
    _write_rows(EXPOSURE_OUT / "bbox_annotation_audit.csv", bbox_annotation_audit_rows)
    _write_rows(EXPOSURE_OUT / "prescore_annotation_exclusions.csv", prescore_annotation_exclusions)
    _json_dump(EXPOSURE_OUT / "prescore_annotation_exclusions.json", prescore_exclusion_report)
    _write_rows(EXPOSURE_OUT / "exposure_cohort_manifest.csv", exposure_manifest)
    def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_jsonl(EXPOSURE_OUT / "dev_cohort_manifest.jsonl", dev_manifest)
    write_jsonl(EXPOSURE_OUT / "confirmation_cohort_manifest.jsonl", confirmation_manifest)
    history_status_path = EXPOSURE_OUT / "historical_fingerprint_status.json"
    history_status = _read_json(history_status_path) if history_status_path.exists() else {}
    confirmation_final_images = [row for row in val_records if row.get("final_eligible")]
    released_bbox_failures = [
        row for row in bbox_annotation_audit_rows if row["released_bbox_validation_status"] != "valid"
    ]
    bbox_audit_report = {
        "schema": "v3-positive-bbox-dimension-audit-v1",
        "policy": "Validate each official COCO xywh box for finite values, positive extents, and positive-area image intersection against source metadata and decoded JPEG dimensions for every released expression. An edge overhang is reported, not clipped or automatically rejected. Preserve official coordinates exactly; do not union, impute, or silently drop annotations.",
        "source_annotation_count_train_val": sum(len(rows) for rows in cohort_rows.values()),
        "source_metadata_bbox_invalid_count_train_val": len(source_bbox_invalid_rows),
        "source_metadata_bbox_invalid_expr_ids_train_val": [row["expr_id"] for row in source_bbox_invalid_rows],
        "source_metadata_bbox_overhang_by_split": {
            split: _summarize_bbox_overhang(cohort_rows[split], "source_bbox_validation_warnings")
            for split in ("train", "val")
        },
        "released_expression_count": len(bbox_annotation_audit_rows),
        "released_bbox_invalid_count": len(released_bbox_failures),
        "released_bbox_invalid_expr_ids": [row["expr_id"] for row in released_bbox_failures],
        "released_image_dimension_mismatch_expression_count": sum(
            not row["dimension_match"] for row in bbox_annotation_audit_rows
        ),
        "released_image_dimension_mismatch_gqa_ids": sorted(
            {row["gqa_image_id"] for row in bbox_annotation_audit_rows if not row["dimension_match"]}
        ),
        "released_decoded_jpeg_bbox_overhang": _summarize_bbox_overhang(
            bbox_annotation_audit_rows, "decoded_jpeg_bbox_validation_warnings"
        ),
        "released_gqa_target_object_box_comparison": {
            "graph_image_record_missing_count": sum(row["gqa_graph_status"] == "image_record_missing" for row in bbox_annotation_audit_rows),
            "target_instance_missing_count": sum(row["gqa_graph_status"] == "target_object_missing" for row in bbox_annotation_audit_rows),
            "target_box_exact_match_count": sum(row["official_bbox_equals_gqa_target_bbox"] is True for row in bbox_annotation_audit_rows),
            "target_box_mismatch_count": sum(row["official_bbox_equals_gqa_target_bbox"] is False for row in bbox_annotation_audit_rows),
            "target_box_unverifiable_count": sum(row["official_bbox_equals_gqa_target_bbox"] is None for row in bbox_annotation_audit_rows),
            "target_box_mismatch_expr_ids": [row["expr_id"] for row in bbox_annotation_audit_rows if row["official_bbox_equals_gqa_target_bbox"] is False],
            "target_box_unverifiable_expr_ids": [row["expr_id"] for row in bbox_annotation_audit_rows if row["official_bbox_equals_gqa_target_bbox"] is None],
        },
        "status": "PASS" if not released_bbox_failures else "BLOCKED_CONFIRMATION",
    }
    _json_dump(EXPOSURE_OUT / "bbox_annotation_audit.json", bbox_audit_report)
    unresolved_source_id_mapping_ids = sorted(
        image_id
        for image_id in dev_ids | val_clean_ids
        if identity_by_id[image_id]["vg_mapping_status"] == "missing_visual_genome_source_id"
    )
    identity_mapping_counts = {
        status: sum(row["vg_mapping_status"] == status for row in identity_by_id.values())
        for status in (
            "mapped_to_coco_id",
            "vg_record_has_null_coco_id",
            "missing_visual_genome_source_id",
        )
    }
    status = "PASS" if (
        0 < len(selected) <= DEV_IMAGE_CAP
        and len(val_records) == len(clean_val)
        and all(row.get("dhash64") for row in val_records)
        and not released_bbox_failures
        and not unresolved_source_id_mapping_ids
        and not history_decode_failures()
        and not history_unknown_image_files()
    ) else "BLOCKED_CONFIRMATION"
    result = {
        "schema": "v3-data-exposure-acceptance-v1",
        "status": status,
        "audit_rule_version": AUDIT_RULE_VERSION,
        "confirmation_label": "independent image-level confirmation only if status is PASS; FineCops annotations are newly acquired official positive val expressions, and no FineCops scorer training/reliability score was run by this audit",
        "pretraining_exposure_claim": "not assessed; no claim that any image or annotation was unseen during foundation-model pretraining",
        "official_positive_val_identity_eligible_image_count": len(clean_val),
        "source_mapping_status_counts_over_candidate_manifest": identity_mapping_counts,
        "source_ids_missing_from_visual_genome_mapping_in_final_cohorts": unresolved_source_id_mapping_ids,
        "val_images_downloaded_and_hashed": sum(bool(row.get("dhash64")) for row in val_records),
        "sample_flow": {
            "official_positive_val_images": len({row["gqa_image_id"] for row in cohort_rows["val"]}),
            "official_positive_val_expressions": len(cohort_rows["val"]),
            "metadata_identity_clean_val_images": len(clean_val),
            "downloaded_and_fingerprinted_val_images": sum(bool(row.get("dhash64")) for row in val_records),
            "historical_exact_pixel_excluded_val_images": sum(bool(row.get("historical_exact_pixel_matches")) for row in val_records),
            "historical_dhash_le4_excluded_val_images": sum(bool(row.get("historical_dhash_le4_matches")) for row in val_records),
            "within_confirmation_dhash_excluded_val_images": len(val_peer_matches),
            "pre_annotation_rule_image_screen_eligible_val_images": len(val_image_screen_ids),
            "pre_annotation_rule_image_screen_eligible_val_expressions": sum(
                row["gqa_image_id"] in val_image_screen_ids for row in cohort_rows["val"]
            ),
            "prescore_positive_box_excluded_val_images": len(prescore_exclusion_report["excluded_images"]),
            "prescore_positive_box_excluded_val_expressions": prescore_exclusion_report["excluded_offending_expression_count"],
            "final_confirmation_val_images": len(val_clean_ids),
            "final_confirmation_val_expressions": len(confirmation_manifest),
            "official_positive_train_images": len({row["gqa_image_id"] for row in cohort_rows["train"]}),
            "official_positive_train_expressions": len(cohort_rows["train"]),
            "metadata_identity_clean_train_images": len(_identity_pool("train")),
            "development_candidates_scanned_in_frozen_order": len(scanned_ids),
            "development_candidates_rejected": len(rejected_dev),
            "development_cross_confirmation_rejected_candidates": sum(
                "confirmation_exact_or_dhash_match" in str(row.get("final_exclusion_reason", ""))
                for row in rejected_dev
            ),
            "historical_exact_pixel_matches_in_selected_dev": sum(bool(row.get("historical_exact_pixel_matches")) for row in selected),
            "historical_dhash_le4_matches_in_selected_dev": sum(bool(row.get("historical_dhash_le4_matches")) for row in selected),
            "final_development_images": len(selected),
            "final_development_expressions": len(dev_manifest),
        },
        "prescore_annotation_exclusion_report": "prescore_annotation_exclusions.json",
        "val_images_final_eligible": len(val_clean_ids),
        "val_expressions_final_eligible": len(confirmation_manifest),
        "train_images_selected": len(selected),
        "development_image_cap": DEV_IMAGE_CAP,
        "development_image_shortfall_from_cap": max(0, DEV_IMAGE_CAP - len(selected)),
        "development_candidate_order_exhausted": next_rank >= len(train_order),
        "train_expressions_selected": len(dev_manifest),
        "dev_candidate_images_scanned": len(scanned_ids),
        "dev_candidate_images_rejected": len(rejected_dev),
        "released_expression_bbox_validation": bbox_audit_report,
        "image_collection_hashes": {
            "historical_path_inventory_sha256": history_status.get("path_inventory_sha256"),
            "historical_fingerprint_manifest_sha256": history_status.get("fingerprint_csv_sha256"),
            "confirmation_final_image_inventory_sha256": _image_collection_identity_sha256(confirmation_final_images),
            "confirmation_image_manifest_csv_sha256": sha256_file(EXPOSURE_OUT / "confirmation_image_manifest.csv"),
            "dev_selected_image_inventory_sha256": _image_collection_identity_sha256(selected),
            "dev_image_manifest_csv_sha256": sha256_file(EXPOSURE_OUT / "dev_image_manifest.csv"),
        },
        "confirmation_download_report": "fetch_confirmation.json",
        "history_fingerprint_report": "historical_fingerprint_status.json",
        "known_annotation_metadata_exposure": "GQA train/val scene graph JSON files were bulk downloaded; prior FineCops feasibility loaded the full GQA val dictionary and queried the old FineCops test IDs. This exposure is logged separately from model training/reliability scoring; see identity_audit.json.",
        "unresolved_conditions": [
            reason
            for condition, reason in (
                (not selected, "No deterministic metadata-clean train candidates passed actual image audit."),
                (len(val_records) != len(clean_val), "Not every metadata-clean official positive val image has an audit row."),
                (any(not row.get("dhash64") for row in val_records), "At least one metadata-clean official positive val image was not decoded and hashed."),
                (bool(released_bbox_failures), "At least one released official positive COCO box is nonfinite, nonpositive, has no positive decoded-JPEG intersection, or has source/image dimension ambiguity."),
                (bool(unresolved_source_id_mapping_ids), "A final dev/confirmation image ID is absent from the pinned Visual Genome source identity mapping."),
                (bool(history_decode_failures()), "At least one locally available historical image failed fingerprinting."),
                (bool(history_unknown_image_files()), "At least one historical image path has unresolved source identity."),
            )
            if condition
        ],
        "artifacts": {
            "source_identity_counts": "source_identity_counts.json",
            "identity_candidate_manifest": "identity_candidate_manifest.csv",
            "historical_identity_registry": "historical_identity_registry.csv",
            "historical_image_fingerprints": "historical_image_fingerprints.csv",
            "historical_fingerprint_status": "historical_fingerprint_status.json",
            "dev_images": "dev_image_manifest.csv",
            "confirmation_images": "confirmation_image_manifest.csv",
            "dev_expressions": "dev_cohort_manifest.jsonl",
            "confirmation_expressions": "confirmation_cohort_manifest.jsonl",
            "bbox_annotation_audit": "bbox_annotation_audit.csv",
            "bbox_annotation_audit_report": "bbox_annotation_audit.json",
            "prescore_annotation_exclusions": "prescore_annotation_exclusions.json",
            "prescore_annotation_exclusion_rows": "prescore_annotation_exclusions.csv",
            "acquisition_attempts": "acquisition_attempts.jsonl",
            "all_cohort_exposure": "exposure_cohort_manifest.csv",
            "rejected_dev_candidates": "dev_rejected_candidates.csv",
        },
        "generated_at_utc": _utc_now(),
    }
    _json_dump(EXPOSURE_OUT / "final_acceptance.json", result)
    return result


def history_decode_failures() -> List[Any]:
    path = EXPOSURE_OUT / "historical_fingerprint_status.json"
    if not path.exists():
        return ["historical fingerprint pass not run"]
    return _read_json(path).get("decode_failures", [])


def history_unknown_image_files() -> List[str]:
    return _load_historical_exposure()["unknown_local_image_paths"]


def _append_acquisition_attempt(record: Mapping[str, Any]) -> None:
    path = EXPOSURE_OUT / "acquisition_attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("metadata", "identities", "history", "fetch-validation", "acquire", "all"), default="all")
    parser.add_argument("--workers", type=int, default=3, help="GQA range-fetch worker count (1-3)")
    parser.add_argument("--hash-workers", type=int, default=4, help="local image fingerprint workers")
    parser.add_argument("--dev-batch-size", type=int, default=50)
    args = parser.parse_args(argv)
    EXPOSURE_OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("metadata", "all"):
        manifest = fetch_metadata()
        print(json.dumps({"metadata_manifest": "data/raw/v3_finecops/annotations/source_manifest.json", "positive_files": len(manifest["positive_files"])}), flush=True)
    if args.stage in ("identities", "all"):
        summary = source_identity_audit()
        identity = build_identity_candidate_manifest()
        print(json.dumps({"source_identity_counts": str(EXPOSURE_OUT / "source_identity_counts.json"), "eligible": identity["finecops_positive_counts"]}, ensure_ascii=False), flush=True)
    if args.stage in ("history", "all"):
        result = fingerprint_history(workers=args.hash_workers)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.stage == "fetch-validation":
        result = fetch_confirmation_images(workers=args.workers)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.stage in ("acquire", "all"):
        started = _utc_now()
        try:
            result = acquire_confirmation_and_development(workers=args.workers, batch_size=args.dev_batch_size)
        except Exception as exc:
            _append_acquisition_attempt(
                {
                    "started_at_utc": started,
                    "recorded_at_utc": _utc_now(),
                    "stage": args.stage,
                    "outcome": "FAILED_OUTPUT_ASSEMBLY",
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "workers": args.workers,
                    "dev_batch_size": args.dev_batch_size,
                    "finecops_model_training": False,
                    "finecops_reliability_scoring": False,
                    "finecops_performance_metrics_computed": False,
                    "note": "Failure record is audit-only; preserve completed fetch/fingerprint work and rerun resumably after the serialization fix.",
                }
            )
            raise
        _append_acquisition_attempt(
            {
                "started_at_utc": started,
                "recorded_at_utc": _utc_now(),
                "stage": args.stage,
                "outcome": "COMPLETED",
                "status": result.get("status"),
                "workers": args.workers,
                "dev_batch_size": args.dev_batch_size,
                "train_images_selected": result.get("train_images_selected"),
                "val_images_downloaded_and_hashed": result.get("val_images_downloaded_and_hashed"),
                "finecops_model_training": False,
                "finecops_reliability_scoring": False,
                "finecops_performance_metrics_computed": False,
            }
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["status"] != "PASS":
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

