"""End-to-end tests for the frozen audit-subset builder (scripts/build_audit_subset.py).

A synthetic ``refs(unc).p`` world lives in ``tmp_path``: 200 train images, 80
val images and 5+5 testA/testB images, so every case runs in milliseconds.
The frozen algorithm is re-implemented *independently* inside this file (see
:func:`_reference_draw`) and used to cross-check the production helpers - a
self-consistent refactor of the builder alone would not survive that.

The real-data check lives in ``test_real_csv_matches_frozen_algorithm``
(marked ``slow``): it rebuilds the subset from ``data/raw`` with the default
parameters and compares all five columns *and* the row order against the frozen
``data/audit_subset.csv``; it skips when the raw data is absent.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT / "src", ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import build_audit_subset as bas  # noqa: E402
from ccg.data.refcoco import (  # noqa: E402
    AnnotationFormatError,
    load_instances_json,
    load_refs_pickle,
    refs_by_image,
)

FAKE_SEED = 7
FAKE_N_TRAIN = 50
FAKE_N_VAL = 10
TRAIN_IDS = list(range(5000, 5200))  # 200 train images
VAL_IDS = list(range(1000, 1080))  # 80 val images -> 40 val_select
TEST_A_IDS = list(range(9001, 9006))
TEST_B_IDS = list(range(9101, 9106))

REAL_REF_PICKLE = ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "refs(unc).p"
REAL_ANNOTATIONS = ROOT / "data" / "raw" / "annotations"
REAL_CSV = ROOT / "data" / "audit_subset.csv"


# ---------------------------------------------------------------------------
# synthetic raw-data world
# ---------------------------------------------------------------------------
def _write_instances_json(path: Path, images: list) -> None:
    payload = {"images": images, "annotations": [], "categories": []}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_fake_world(tmp_path: Path):
    """``refs(unc).p`` + a COCO-style manifest under ``tmp_path/raw``.

    The train images are deliberately given COCO ``train2014`` file names with
    ids *larger* than the ``val2014`` ones: the frozen row order is by
    ``(coco_split, image_id)``, so a plain image-id sort would reorder them.
    """
    raw = tmp_path / "raw"
    ref_dir = raw / "refcoco+" / "refcoco+"
    ref_dir.mkdir(parents=True)
    ann_dir = raw / "annotations"
    ann_dir.mkdir(parents=True)

    records = []
    ref_counts = {}

    def add(image_id: int, split: str, count: int) -> None:
        ref_counts[image_id] = count
        for index in range(count):
            records.append(
                {
                    "ref_id": f"{image_id}_{index}",
                    "image_id": image_id,
                    "split": split,
                    "ann_id": 10_000_000 + image_id * 10 + index,
                    "category_id": 47,
                    "file_name": f"COCO_train2014_{image_id:012d}_{index}.jpg",
                    "sent_ids": [index],
                    "sentences": [
                        {"sent_id": index, "tokens": ["blue", "cup"], "raw": f"the cup {index}"}
                    ],
                }
            )

    for position, image_id in enumerate(TRAIN_IDS):
        add(image_id, "train", (position % 3) + 1)
    for position, image_id in enumerate(VAL_IDS):
        add(image_id, "val", (position % 2) + 1)
    for image_id in TEST_A_IDS:
        add(image_id, "testA", 1)
    for image_id in TEST_B_IDS:
        add(image_id, "testB", 1)

    with (ref_dir / "refs(unc).p").open("wb") as handle:
        pickle.dump(records, handle, protocol=4)

    val_images = [
        {"id": image_id, "file_name": f"COCO_val2014_{image_id:012d}.jpg"} for image_id in VAL_IDS
    ]
    train_images = [
        {"id": image_id, "file_name": f"COCO_train2014_{image_id:012d}.jpg"}
        for image_id in [*TRAIN_IDS, *TEST_A_IDS, *TEST_B_IDS]
    ]
    _write_instances_json(ann_dir / "instances_val2014.json", val_images)
    _write_instances_json(ann_dir / "instances_train2014.json", train_images)
    return raw, ref_counts


def _reference_draw(records: list, *, seed: int, n_train: int, n_val: int):
    """Independent re-implementation of the frozen two-draw selection."""
    train_pool = sorted({int(r["image_id"]) for r in records if r["split"] == "train"})
    val_pool = sorted({int(r["image_id"]) for r in records if r["split"] == "val"})
    perm = np.random.default_rng(seed + 1).permutation(len(val_pool))
    val_select = sorted(val_pool[int(i)] for i in perm[: len(val_pool) // 2])
    rng = np.random.default_rng(seed)
    train_chosen = [
        train_pool[int(i)] for i in rng.choice(len(train_pool), size=n_train, replace=False)
    ]
    val_chosen = [
        val_select[int(i)] for i in rng.choice(len(val_select), size=n_val, replace=False)
    ]
    return train_chosen + val_chosen, val_select


@pytest.fixture()
def fake_world(tmp_path):
    return _write_fake_world(tmp_path)


def _records(raw: Path) -> list:
    refs_path = bas.find_refs_pickle(raw)
    assert refs_path is not None
    return load_refs_pickle(refs_path)


def _build(raw: Path, *, seed: int = FAKE_SEED) -> list:
    refs_path = bas.find_refs_pickle(raw)
    assert refs_path is not None
    return bas.build_subset_rows(
        refs_path=refs_path,
        annotations_dir=raw / "annotations",
        seed=seed,
        n_train=FAKE_N_TRAIN,
        n_val=FAKE_N_VAL,
    )


def _extract(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


# ---------------------------------------------------------------------------
# refcoco loaders used by the builder
# ---------------------------------------------------------------------------
def test_load_refs_pickle_reads_the_fake_world(fake_world):
    raw, ref_counts = fake_world
    records = _records(raw)
    assert len(records) == sum(ref_counts.values())
    assert records[0]["split"] == "train"
    for record in records[:5]:
        assert set(record) >= {"ref_id", "image_id", "split", "ann_id", "sentences"}


def test_load_refs_pickle_validates_records(tmp_path):
    path = tmp_path / "refs.p"
    with path.open("wb") as handle:
        pickle.dump([{"ref_id": 1, "image_id": 2, "split": "train", "ann_id": 3}], handle)
    with pytest.raises(AnnotationFormatError):
        load_refs_pickle(path)  # missing 'sentences'

    with path.open("wb") as handle:
        pickle.dump(
            [{"ref_id": 1, "image_id": 2, "split": "train", "ann_id": 3, "sentences": []}],
            handle,
        )
    with pytest.raises(AnnotationFormatError):
        load_refs_pickle(path)  # empty sentences


def test_load_instances_json_builds_the_ann_id_table(tmp_path):
    path = tmp_path / "instances.json"
    payload = {
        "images": [{"id": 7, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 21, "image_id": 7, "bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 47},
            {"id": 22.0, "image_id": 7, "bbox": [0, 0, 1, 1], "category_id": 3},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    table = load_instances_json(path)
    assert set(table) == {"21", "22"}  # canonical str(int(id)) keys
    assert table["21"] == {"image_id": 7, "bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 47}


def test_load_instances_json_rejects_duplicates_and_bad_bbox(tmp_path):
    dup = tmp_path / "dup.json"
    dup.write_text(
        json.dumps(
            {
                "annotations": [
                    {"id": 5, "image_id": 1, "bbox": [0, 0, 1, 1], "category_id": 1},
                    {"id": 5, "image_id": 2, "bbox": [0, 0, 1, 1], "category_id": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AnnotationFormatError):
        load_instances_json(dup)

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {"annotations": [{"id": 6, "image_id": 1, "bbox": [0, 0, 1], "category_id": 1}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(AnnotationFormatError):
        load_instances_json(bad)


def test_refs_by_image_groups_and_preserves_records():
    refs = [
        {"image_id": 5, "tag": "a"},
        {"image_id": 5, "tag": "b"},
        {"image_id": 3, "tag": "c"},
    ]
    grouped = refs_by_image(refs)
    assert list(grouped) == [5, 3]  # first-appearance order
    assert [record["tag"] for record in grouped[5]] == ["a", "b"]
    assert grouped[3][0] is refs[2]  # the very records, not copies


def test_refs_by_image_rejects_records_without_image_id():
    with pytest.raises(AnnotationFormatError):
        refs_by_image([{"ref_id": 1}])


# ---------------------------------------------------------------------------
# frozen selection algorithm
# ---------------------------------------------------------------------------
def test_select_val_subset_is_first_half_of_seed_plus_one_permutation():
    val_pool = list(range(3000, 3040))
    chosen = bas.select_val_subset(val_pool, seed=10)
    perm = np.random.default_rng(11).permutation(len(val_pool))
    expected = sorted(val_pool[int(i)] for i in perm[: len(val_pool) // 2])
    assert chosen == expected
    assert chosen == bas.select_val_subset(val_pool, seed=10)  # deterministic


def test_draw_matches_independent_reference(fake_world):
    raw, ref_counts = fake_world
    records = _records(raw)
    pairs, counts = bas.draw_audit_images(
        records, seed=FAKE_SEED, n_train=FAKE_N_TRAIN, n_val=FAKE_N_VAL
    )
    expected_ids, val_select = _reference_draw(
        records, seed=FAKE_SEED, n_train=FAKE_N_TRAIN, n_val=FAKE_N_VAL
    )
    assert [image_id for image_id, _ in pairs] == expected_ids  # draw order, not sorted
    assert {label for _, label in pairs[:FAKE_N_TRAIN]} == {"train"}
    assert {label for _, label in pairs[FAKE_N_TRAIN:]} == {"val_select"}
    assert counts == ref_counts
    assert len(val_select) == 40 and set(val_select) <= set(VAL_IDS)


def test_same_seed_reproduces_same_rows(fake_world):
    raw, _ = fake_world
    assert _build(raw, seed=FAKE_SEED) == _build(raw, seed=FAKE_SEED)


def test_different_seed_changes_selection(fake_world):
    raw, _ = fake_world
    train_7 = {row["image_id"] for row in _build(raw, seed=7) if row["refcoco_split"] == "train"}
    train_8 = {row["image_id"] for row in _build(raw, seed=8) if row["refcoco_split"] == "train"}
    assert train_7 != train_8


def test_pool_too_small_raises_instead_of_padding(fake_world):
    raw, _ = fake_world
    records = _records(raw)
    with pytest.raises(ValueError):
        bas.draw_audit_images(records, seed=FAKE_SEED, n_train=len(TRAIN_IDS) + 1, n_val=1)
    with pytest.raises(ValueError):
        bas.draw_audit_images(records, seed=FAKE_SEED, n_train=1, n_val=41)  # val_select is 40


# ---------------------------------------------------------------------------
# row semantics: splits, ordering, columns, joins
# ---------------------------------------------------------------------------
def test_test_a_and_test_b_images_never_selected(fake_world):
    raw, _ = fake_world
    rows = _build(raw)
    selected = {row["image_id"] for row in rows}
    assert selected.isdisjoint(set(TEST_A_IDS) | set(TEST_B_IDS))
    labels = [row["refcoco_split"] for row in rows]
    assert labels.count("train") == FAKE_N_TRAIN
    assert labels.count("val_select") == FAKE_N_VAL


def test_rows_sorted_by_coco_split_then_image_id(fake_world):
    raw, _ = fake_world
    rows = _build(raw)
    keys = [(row["coco_split"], row["image_id"]) for row in rows]
    assert keys == sorted(keys)
    # train2014 ids are larger than val2014 ids in the fake world, yet the
    # train2014 block comes first - only the (coco_split, image_id) key does that
    assert [row["coco_split"] for row in rows] == ["train2014"] * FAKE_N_TRAIN + [
        "val2014"
    ] * FAKE_N_VAL


def test_row_columns_and_joins(fake_world):
    raw, ref_counts = fake_world
    rows = _build(raw)
    assert bas.CSV_FIELDS == ("image_id", "file_name", "coco_split", "refcoco_split", "ref_count")
    for row in rows:
        assert set(row) == set(bas.CSV_FIELDS)
        assert row["file_name"] == f"COCO_{row['coco_split']}_{row['image_id']:012d}.jpg"
        assert row["ref_count"] == ref_counts[row["image_id"]]


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        bas.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--raw", "--seed", "--n-train", "--n-val", "--out", "--force"):
        assert flag in out


def test_cli_refuses_to_overwrite_without_force(fake_world, tmp_path, capsys):
    raw, _ = fake_world
    out = tmp_path / "out.csv"
    argv = ["--raw", str(raw), "--seed", "7", "--n-train", "50", "--n-val", "10", "--out", str(out)]
    assert bas.main([*argv, "--force"]) == 0
    before = out.read_bytes()
    capsys.readouterr()

    assert bas.main(argv) == 0  # friendly refusal, not a crash
    captured = capsys.readouterr().out
    assert "already exists" in captured and "--force" in captured
    assert out.read_bytes() == before  # untouched


def test_cli_force_writes_the_api_rows(fake_world, tmp_path, capsys):
    raw, _ = fake_world
    out = tmp_path / "out.csv"
    argv = ["--raw", str(raw), "--seed", "7", "--n-train", "50", "--n-val", "10", "--out", str(out)]
    assert bas.main([*argv, "--force"]) == 0
    captured = capsys.readouterr().out
    assert "wrote" in captured and "train=50" in captured and "val_select=10" in captured

    expected = _build(raw, seed=7)
    fields, written = _extract(out)
    assert fields == list(bas.CSV_FIELDS)
    assert [{key: str(value) for key, value in row.items()} for row in expected] == written


def test_cli_missing_raw_data_returns_one(tmp_path, capsys):
    out = tmp_path / "out.csv"
    assert bas.main(["--raw", str(tmp_path / "absent"), "--out", str(out)]) == 1
    captured = capsys.readouterr().out
    assert "no refs(unc).p" in captured
    assert not out.exists()


# ---------------------------------------------------------------------------
# real data: the frozen csv must be reproducible (exists-only, slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_csv_matches_frozen_algorithm():
    if not REAL_REF_PICKLE.exists() or not REAL_CSV.exists():
        pytest.skip("real RefCOCO+ pickle and/or data/audit_subset.csv not present")

    rows = bas.build_subset_rows(
        refs_path=REAL_REF_PICKLE,
        annotations_dir=REAL_ANNOTATIONS,
    )
    fields, frozen_rows = _extract(REAL_CSV)
    assert fields == list(bas.CSV_FIELDS)
    assert len(rows) == len(frozen_rows) == 1500
    # every cell of every row, in order - ids, labels, file names and ref counts
    assert [{key: str(value) for key, value in row.items()} for row in rows] == frozen_rows
