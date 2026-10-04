from pathlib import Path

from PIL import Image, ImageDraw

from ccg.v3.exposure import BKTree, dhash64, fingerprint_image, hamming64, pixel_sha256
from scripts import v3_audit_exposure


def test_pixel_hash_tracks_decoded_pixels_not_file_encoding(tmp_path: Path) -> None:
    image = Image.new("RGB", (32, 24), "white")
    ImageDraw.Draw(image).rectangle((4, 5, 18, 17), fill="navy")
    png = tmp_path / "first.png"
    bmp = tmp_path / "second.bmp"
    image.save(png)
    image.save(bmp)

    first = fingerprint_image(png)
    second = fingerprint_image(bmp)

    assert first.file_sha256 != second.file_sha256
    assert first.pixel_sha256 == second.pixel_sha256
    assert first.width == second.width == 32
    assert first.height == second.height == 24


def test_frozen_dhash_rule_and_bk_tree_find_all_matches() -> None:
    base = int(dhash64(Image.new("RGB", (16, 16), "black")), 16)
    values = [f"{base ^ bit:016x}" for bit in (0, 1, 3, 7, 15, 31)]
    tree = BKTree((value, f"item-{index}") for index, value in enumerate(values))

    matches = tree.query(values[0], max_distance=4)

    assert {key for key, _ in matches} == {"item-0", "item-1", "item-2", "item-3", "item-4"}
    assert all(distance == hamming64(values[0], values[int(key.split('-')[1])]) for key, distance in matches)
    assert hamming64(values[0], values[5]) == 5


def test_pixel_sha256_includes_dimensions() -> None:
    wide = Image.new("RGB", (4, 2), "#123456")
    tall = Image.new("RGB", (2, 4), "#123456")
    wide_hash, _, _ = pixel_sha256(wide)
    tall_hash, _, _ = pixel_sha256(tall)
    assert wide.tobytes() == tall.tobytes()
    assert wide_hash != tall_hash


def test_historical_registry_distinguishes_null_mapping_from_missing_id(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(v3_audit_exposure, "REPO_ROOT", tmp_path)
    history = {
        "refcoco_plus_by_split": {},
        "refcocog_umd_by_split": {},
        "old_finecops_test_gqa_ids": {"10", "11", "12"},
        "audit_subset_gqa_ids": set(),
        "audit_subset_coco_ids": set(),
        "local_image_paths": [],
    }
    vg = {"10": {"coco_id": 123}, "11": {"coco_id": None}}

    rows = v3_audit_exposure._build_historical_identity_registry(history, vg)

    by_id = {row["gqa_image_id"]: row for row in rows}
    assert by_id["10"]["gqa_to_coco_mapping_status"] == "mapped_to_coco_id"
    assert by_id["10"]["coco_image_id"] == "123"
    assert by_id["11"]["gqa_to_coco_mapping_status"] == "vg_record_has_null_coco_id"
    assert by_id["12"]["gqa_to_coco_mapping_status"] == "missing_visual_genome_source_id"


def test_image_collection_hash_is_order_independent_and_fingerprint_sensitive() -> None:
    rows = [
        {"gqa_image_id": "20", "file_sha256": "a", "pixel_sha256": "b", "dhash64": "c"},
        {"gqa_image_id": "10", "file_sha256": "d", "pixel_sha256": "e", "dhash64": "f"},
    ]

    expected = v3_audit_exposure._image_collection_identity_sha256(rows)

    assert expected == v3_audit_exposure._image_collection_identity_sha256(list(reversed(rows)))
    changed = [dict(rows[0]), dict(rows[1])]
    changed[0]["pixel_sha256"] = "changed"
    assert expected != v3_audit_exposure._image_collection_identity_sha256(changed)


def test_bbox_audit_preserves_intersecting_overhang_and_rejects_empty_geometry() -> None:
    overhang = v3_audit_exposure._validate_bbox_xywh([9, 8, 3, 4], 10, 10)
    disjoint = v3_audit_exposure._validate_bbox_xywh([11, 8, 3, 4], 10, 10)
    nonpositive = v3_audit_exposure._validate_bbox_xywh([1, 1, 0, 4], 10, 10)

    assert overhang["status"] == "valid"
    assert overhang["xyxy"] == [9.0, 8.0, 12.0, 12.0]
    assert overhang["warnings"] == ["bbox_overhang_right_px=2", "bbox_overhang_bottom_px=2"]
    assert disjoint["status"] == "invalid"
    assert "bbox_has_no_positive_image_intersection" in disjoint["reasons"]
    assert nonpositive["status"] == "invalid"
    assert "bbox_nonpositive_extent" in nonpositive["reasons"]


def test_streaming_json_object_iterator_handles_chunk_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "graphs.json"
    path.write_text('{"2383": {"objects": {"1": {"x": 2}}}, "图像": [1, 2, 3]}', encoding="utf-8")

    rows = dict(v3_audit_exposure._iter_json_object_items(path, chunk_bytes=7))

    assert rows == {"2383": {"objects": {"1": {"x": 2}}}, "图像": [1, 2, 3]}

