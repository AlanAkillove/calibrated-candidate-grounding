"""Phase 1E FineCops-Ref external feasibility tests (instruction section 33).

Every check is offline: the released annotation *shape* is reproduced by small
synthetic fixtures, so the contract of the external data layer is pinned without
downloading 20 GB of GQA images or running any model.

The twelve pre-registered items and where they live:

1.  ``xywh`` -> ``xyxy`` target conversion .............. ``test_target_box_is_xywh_converted_to_xyxy``
2.  image-id mapping (GQA string -> int, negatives out) ``test_gqa_image_id_mapping_*``
3.  target assignment / equivalent removal ............. ``test_audit_expression_*``
4.  frozen ``N = 64`` proposal budget .................. ``test_frozen_proposal_budget_is_64_and_pinned``
5.  deterministic candidate/audit manifest ............. ``test_audit_subset_selection_is_deterministic``
6.  no FineCops train/val label reachable .............. ``test_only_the_positive_test_split_is_reachable``
7.  RefCOCO+ normalisation reused, never re-fitted ..... ``test_apply_frozen_reuses_stored_normalisation``
8.  frozen coefficient / artifact checksums ............ ``test_frozen_refcoco_artifact_checksums_are_recorded``
9.  no calibration / threshold fitting on FineCops ..... ``test_external_modules_fit_nothing``
10. difficulty metadata preserved ...................... ``test_difficulty_distribution_*``
11. bootstrap clusters by image ....................... ``test_paired_bootstrap_clusters_by_image_id``
12. positive and negative paths separated .............. ``test_negative_*``

Extras: scene-graph target resolution, exact same-name identity, the frozen
engineering gate and the candidate-regime branch rule, and the aggregate
arithmetic of :func:`ccg.external.feasibility.summarise_rows`.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _extra in (_REPO / "src", _REPO / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from ccg.data.types import ProposalBank  # noqa: E402
from ccg.external import feasibility as fe  # noqa: E402
from ccg.external import finecops as fc  # noqa: E402
from ccg.external import gqa_images as gi  # noqa: E402
from ccg.metrics.bootstrap import paired_bootstrap  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.reliability import models as rmodels  # noqa: E402
from ccg.reliability.models import LogisticModel  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic.frozen import FrozenSeedModels, apply_frozen  # noqa: E402

_DRIVER = _REPO / "scripts" / "run_phase1e_feasibility.py"

# ---------------------------------------------------------------------------
# synthetic fixtures that mirror the released FineCops-Ref files
# ---------------------------------------------------------------------------
_IMAGE_A = 230  # GQA image id -> "images/230.jpg"
_IMAGE_B = 451


def _graph_objects(*specs: tuple[str, str, int, int, int, int]) -> Dict[str, Any]:
    """``(object_id, name, x, y, w, h)`` -> a GQA scene-graph ``objects`` dict."""
    return {
        oid: {"name": name, "x": x, "y": y, "w": w, "h": h, "word": name.split()}
        for oid, name, x, y, w, h in specs
    }


def _scene_graph() -> Dict[str, Any]:
    return {
        # image A: two objects that share the name "dog" (the level-2 situation)
        str(_IMAGE_A): {
            "width": 500,
            "height": 375,
            "objects": _graph_objects(
                ("0", "dog", 100, 100, 80, 60),
                ("1", "dog", 300, 100, 80, 60),
                ("2", "Cat", 40, 40, 30, 30),
            ),
        },
        # image B: the released box matches nothing (forces the objects_id fallback)
        str(_IMAGE_B): {
            "width": 375,
            "height": 500,
            "objects": _graph_objects(("7", "person", 10, 10, 20, 20)),
        },
    }


def _annotation(
    aid: int,
    image_id: int,
    bbox_xywh: List[float],
    *,
    level: int = 2,
    tuple_type: str = "1_hop",
    objects_id: tuple[str, ...] = ("0", "1"),
    caption: str = "the brown dog",
) -> Dict[str, Any]:
    return {
        "id": aid,
        "image_id": image_id,
        "bbox": list(bbox_xywh),
        "caption": caption,
        "category_id": 1,
        "area": float(bbox_xywh[2] * bbox_xywh[3]),
        "level": level,
        "tuple_type": tuple_type,
        "objects_id": list(objects_id),
        "attribute": [["brown"]],
        "spatial": "on the left",
    }


def _coco_doc(annotations: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> Dict[str, Any]:
    # the released categories list is a single placeholder - it carries no class
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "category 1", "supercategory": ""}],
    }


def _images(gqa_ids: Dict[int, str], dims: Dict[int, tuple[int, int]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": coco_id,
            "file_name": f"images/{gqa_id}.jpg",
            "width": dims[coco_id][0],
            "height": dims[coco_id][1],
            "caption": "",
            "dataset_name": "ref",
        }
        for coco_id, gqa_id in sorted(gqa_ids.items())
    ]


@pytest.fixture()
def cohort(tmp_path: Path) -> tuple[Path, Path]:
    """A two-image / three-expression positive test cohort on disk."""
    dims = {1: (500, 375), 2: (500, 375), 3: (375, 500)}
    annotations = [
        _annotation(11, 1, [100, 100, 80, 60], level=2, objects_id=("0", "1")),
        _annotation(12, 2, [300, 100, 80, 60], level=1, tuple_type="0_hop", objects_id=("1",)),
        _annotation(13, 3, [200, 200, 40, 40], level=3, tuple_type="2_hop", objects_id=("7",)),
    ]
    ann_dir = tmp_path / "finecops"
    ann_dir.mkdir()
    # the release lists one image row per expression, so GQA image 230 appears twice
    payload = _coco_doc(annotations, _images({1: str(_IMAGE_A), 2: str(_IMAGE_A), 3: str(_IMAGE_B)}, dims))
    (ann_dir / fc.COCO_POS_FILE).write_text(json.dumps(payload), encoding="utf-8")
    graph_path = tmp_path / "val_sceneGraphs.json"
    graph_path.write_text(json.dumps(_scene_graph()), encoding="utf-8")
    return ann_dir, graph_path


def _expressions(cohort: tuple[Path, Path]) -> List[fc.FineCopsExpression]:
    ann_dir, graph_path = cohort
    return fc.load_test_expressions(ann_dir, graph_path)


# ---------------------------------------------------------------------------
# 1 / 2 - geometry and id mapping
# ---------------------------------------------------------------------------
def test_target_box_is_xywh_converted_to_xyxy(cohort) -> None:
    by_id = {e.expr_id: e for e in _expressions(cohort)}
    expr = by_id[11]
    assert np.allclose(expr.gt_box, [100.0, 100.0, 180.0, 160.0])
    assert expr.gt_box.dtype == np.float32
    assert expr.gt_box[2] > expr.gt_box[0] and expr.gt_box[3] > expr.gt_box[1]


def test_gqa_image_id_mapping_and_geometry(cohort) -> None:
    rows = {e.expr_id: e for e in _expressions(cohort)}
    assert rows[11].image_id == _IMAGE_A and rows[11].gqa_image_id == str(_IMAGE_A)
    assert isinstance(rows[11].image_id, int)  # the bootstrap cluster key must be numeric
    assert (rows[13].width, rows[13].height) == (375, 500)


@pytest.mark.parametrize("gqa_id", ["neg_1", "230abc"])
def test_non_positive_image_ids_are_rejected(cohort, tmp_path, gqa_id) -> None:
    ann_dir, graph_path = cohort
    doc = json.loads((ann_dir / fc.COCO_POS_FILE).read_text(encoding="utf-8"))
    doc["images"][0]["file_name"] = f"images/{gqa_id}.jpg"
    (ann_dir / fc.COCO_POS_FILE).write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric image id"):
        fc.load_test_expressions(ann_dir, graph_path)


def test_degenerate_target_box_is_rejected(cohort, tmp_path) -> None:
    ann_dir, graph_path = cohort
    doc = json.loads((ann_dir / fc.COCO_POS_FILE).read_text(encoding="utf-8"))
    doc["annotations"][0]["bbox"] = [100, 100, 0, 0]
    (ann_dir / fc.COCO_POS_FILE).write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="degenerate target box"):
        fc.load_test_expressions(ann_dir, graph_path)


# ---------------------------------------------------------------------------
# 6 / 12 - what the loader is allowed to touch
# ---------------------------------------------------------------------------
def test_only_the_positive_test_split_is_reachable(cohort, tmp_path, monkeypatch) -> None:
    """No train/val file and no negative file can enter the model path."""
    ann_dir, graph_path = cohort
    for name in (
        "train_expression_pos_coco_format.json",
        "val_expression_pos_coco_format.json",
        fc.COCO_ALL_FILE,
        fc.VANILLA_ALL_FILE,
    ):
        (ann_dir / name).write_text(
            json.dumps({"poison": [{"id": 1, "bbox": [0, 0, 1, 1]}]}), encoding="utf-8"
        )

    seen: List[str] = []
    real_read = fc.read_json

    def spy(path: Path) -> Any:
        seen.append(Path(path).name)
        return real_read(path)

    monkeypatch.setattr(fc, "read_json", spy)
    expressions = fc.load_test_expressions(ann_dir, graph_path)
    assert [e.expr_id for e in expressions] == [11, 12, 13]
    assert set(seen) <= {fc.COCO_POS_FILE, graph_path.name}
    assert not any(n in seen for n in ("train_expression_pos_coco_format.json",
                                      "val_expression_pos_coco_format.json",
                                      fc.COCO_ALL_FILE, fc.VANILLA_ALL_FILE))


def test_downloader_never_fetches_train_val_or_negative_images() -> None:
    """F0 only pins the test split; the negative image tarball stays deferred."""
    sys.path.insert(0, str(_REPO / "tools"))
    import download_finecops as dl  # noqa: WPS433 - tool script, not a package

    names = [f["name"] for f in dl.TEST_FILES]
    assert all(n.startswith("test_") for n in names)
    assert not any(n.startswith(("train_", "val_")) for n in names)
    blob = json.dumps(names) + dl.GQA_SCENE_GRAPHS
    assert "neg_images" not in blob and "images.zip" not in blob


# ---------------------------------------------------------------------------
# 3 - target assignment on the frozen bank
# ---------------------------------------------------------------------------
def _bank(boxes: List[List[float]], scores: List[float], image_id: int = _IMAGE_A) -> ProposalBank:
    return ProposalBank(
        image_id=image_id,
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        objectness=np.asarray(scores, dtype=np.float32),
    )


def test_audit_expression_removes_equivalent_proposals(cohort) -> None:
    expr = next(e for e in _expressions(cohort) if e.expr_id == 11)  # target box 100,100,180,160
    boxes = [
        [101, 101, 179, 159],   # IoU ~ 0.94 with the target  -> equivalent
        [100, 100, 180, 160],   # IoU = 1.0                   -> the assigned target (argmax)
        [110, 110, 190, 170],   # IoU ~ 0.57                  -> equivalent, must be removed
        [300, 100, 380, 160],   # the other "dog"             -> same-name distractor
        [40, 40, 70, 70],       # the "Cat"
        [1, 1, 2, 2],           # degenerate crop
    ]
    bank = _bank(boxes, [0.9, 0.95, 0.8, 0.7, 0.6, 0.5])
    gt_boxes, gt_codes, _ids = fe.graph_object_boxes(
        _scene_graph()[str(_IMAGE_A)]["objects"], fe.build_name_code_table([], {"x": ["dog", "Cat"]})
    )
    row = fe.audit_expression(
        expr, bank, gt_object_boxes_xyxy=gt_boxes, gt_object_name_codes=gt_codes,
        Ks=(2, 3, fe.PRIMARY_K, fe.SECONDARY_K),
    )
    assert row.target_present and row.n_equivalent == 3      # three proposals reach 0.5
    assert row.target_best_iou > 0.99
    # 6 proposals - target - 2 equivalents = 3 distractors
    assert row.n_valid_distractors == 3
    assert row.n_same_name_distractors == 1                  # the second "dog"
    assert row.available[fe.PRIMARY_K] is False              # K=5 needs 4 distractors
    assert row.available[3] is True
    assert row.same_name_available[2] is True and row.same_name_available[3] is False
    assert row.n_invalid_crop == 1
    assert row.level == expr.level and row.tuple_type == expr.tuple_type
    assert set(row.to_row()) >= {"available_K5", "same_name_available_K5", "object_recall_at_05"}


def test_audit_expression_records_natural_omission(cohort) -> None:
    expr = next(e for e in _expressions(cohort) if e.expr_id == 11)
    bank = _bank([[0, 0, 20, 20], [400, 300, 480, 370]], [0.5, 0.4])
    gt_boxes, gt_codes, _ids = fe.graph_object_boxes(
        _scene_graph()[str(_IMAGE_A)]["objects"], {"dog": 0, "Cat": 1}
    )
    row = fe.audit_expression(
        expr, bank, gt_object_boxes_xyxy=gt_boxes, gt_object_name_codes=gt_codes
    )
    assert not row.target_present and row.n_equivalent == 0
    assert row.target_best_iou < 0.5
    assert not any(row.available.values()) and not any(row.same_name_available.values())


# ---------------------------------------------------------------------------
# 4 - the frozen proposal budget
# ---------------------------------------------------------------------------
def test_frozen_proposal_budget_is_64_and_pinned() -> None:
    assert fe.N_PROPOSALS == 64
    assert fe.PRIMARY_K == 5 and fe.SECONDARY_K == 10
    src = _DRIVER.read_text(encoding="utf-8")
    assert "top_n=fe.N_PROPOSALS" in src            # never the 128 default of rpn.py
    assert "top_n=128" not in src and "DEFAULT_TOP_N" not in src


# ---------------------------------------------------------------------------
# 5 - deterministic manifest
# ---------------------------------------------------------------------------
def _wide_cohort(n: int = 40) -> List[fc.FineCopsExpression]:
    def make(i: int) -> fc.FineCopsExpression:
        return fc.FineCopsExpression(
            expr_id=i,
            image_id=1000 + (i % n),
            gqa_image_id=str(1000 + (i % n)),
            text=f"expression {i}",
            gt_box=np.asarray([0, 0, 10, 10], dtype=np.float32),
            width=100,
            height=100,
            level=1 + (i % 3),
            tuple_type="0_hop",
            objects_id=("0",),
        )

    return [make(i) for i in range(n * 2)]


def test_audit_subset_selection_is_deterministic() -> None:
    exprs = _wide_cohort()
    a = fc.select_audit_subset(exprs, 10, seed=20260929)
    b = fc.select_audit_subset(exprs, 10, seed=20260929)
    assert a == b and len(a) == 10 and a == sorted(a)
    assert len(set(a)) == 10
    assert a != fc.select_audit_subset(exprs, 10, seed=20260930)
    # requesting more images than exist cannot crash and never duplicates
    whole = fc.select_audit_subset(exprs, 10_000, seed=1)
    assert whole == sorted({e.image_id for e in exprs})


# ---------------------------------------------------------------------------
# 9 / 7 / 8 - nothing on FineCops is fitted; the RefCOCO+ models are reused
# ---------------------------------------------------------------------------
_FORBIDDEN = (
    ".fit(",
    "normalize_fit",
    "fit_temperature",
    "IsotonicRegression",
    "PLATT",
    "Platt",
    "GridSearchCV",
    "LogisticRegression",
    "train_expression",
    "val_expression",
    "groundingdino",
    "GroundingDINO",
)


@pytest.mark.parametrize(
    "module",
    [fc.__file__, gi.__file__, fe.__file__, str(_DRIVER)],
    ids=["finecops", "gqa_images", "feasibility", "driver"],
)
def test_external_modules_fit_nothing(module: str) -> None:
    src = Path(module).read_text(encoding="utf-8")
    for token in _FORBIDDEN:
        assert token not in src, f"{Path(module).name} contains {token!r}"


def _synthetic_frozen_models(seed: int = 0) -> FrozenSeedModels:
    rng = np.random.default_rng(seed)
    n_stats, n_sem = len(rfeat.stat_feature_names()), len(sfeat.SEMANTIC_STAT_NAMES)
    stats_raw = rng.normal(size=(60, n_stats))
    sem_raw = rng.normal(size=(60, n_sem))
    stats_fit = rfeat.normalize_fit(stats_raw, fit_rows=np.arange(30), keys=rfeat.stat_feature_names())
    sem_fit = rfeat.normalize_fit(sem_raw, fit_rows=np.arange(30), keys=sfeat.SEMANTIC_STAT_NAMES)
    y = (rng.normal(size=30) > 0).astype(float)
    stats_clf = LogisticModel(C=1.0)
    stats_clf.fit(np.asarray(rfeat.normalize_apply(stats_raw[:30], stats_fit)), y)
    e1b_clf = LogisticModel(C=1.0)
    e1b_clf.fit(
        np.hstack(
            [rfeat.normalize_apply(stats_raw[:30], stats_fit), rfeat.normalize_apply(sem_raw[:30], sem_fit)]
        ),
        y,
    )
    return FrozenSeedModels(
        scorer="b3_seed1",
        temperature=1.5,
        stats_fit=stats_fit,
        sem_fit=sem_fit,
        stats_clf=stats_clf,
        e1b_clf=e1b_clf,
        verification={k: 0.0 for k in ("stats_logistic_pred_max_abs", "e1b_coefficients_max_abs")},
    )


def test_apply_frozen_reuses_stored_normalisation() -> None:
    """External scores must go through the RefCOCO+ fit unchanged (section 14)."""
    models = _synthetic_frozen_models()
    before_stats = models.stats_clf.coefficients()[0].copy()
    before_mean = np.asarray(models.stats_fit.mean).copy()
    rng = np.random.default_rng(7)
    stats_raw = rng.normal(size=(12, len(rfeat.stat_feature_names())))
    sem_raw = rng.normal(size=(12, len(sfeat.SEMANTIC_STAT_NAMES)))

    stats_conf, e1b_conf = apply_frozen(models, stats_raw, sem_raw)

    manual_stats = models.stats_clf.predict_proba(rfeat.normalize_apply(stats_raw, models.stats_fit))
    manual_e1b = models.e1b_clf.predict_proba(
        np.hstack(
            [rfeat.normalize_apply(stats_raw, models.stats_fit), rfeat.normalize_apply(sem_raw, models.sem_fit)]
        )
    )
    assert np.allclose(stats_conf, manual_stats) and np.allclose(e1b_conf, manual_e1b)
    assert stats_conf.shape == e1b_conf.shape == (12,)
    assert np.all((0.0 < stats_conf) & (stats_conf < 1.0))
    # nothing re-fitted: coefficients and normalisation are byte-identical
    assert np.array_equal(models.stats_clf.coefficients()[0], before_stats)
    assert np.array_equal(np.asarray(models.stats_fit.mean), before_mean)


def test_frozen_refcoco_artifact_checksums_are_recorded() -> None:
    """The driver pins the frozen Phase 0.5 / 1 artifacts it will reuse (section 33 test 8)."""
    sys.path.insert(0, str(_REPO / "scripts"))
    import run_phase1e_feasibility as driver  # noqa: WPS433

    payload = driver._frozen_artifacts()  # noqa: SLF001 - white-box contract check
    assert payload["recovery_tolerance"] == 1e-9
    assert payload["train_ks"] == [5, 10]
    assert len(payload["files"]) == payload["n_files"] >= 8
    assert all(len(digest) == 16 and digest == digest.lower() for digest in payload["files"].values())
    joined = " ".join(payload["files"])
    assert "e1_logistic/coefficients.csv" in joined
    assert all(f"b3_seed{i}" in joined for i in (1, 2, 3))


def test_refcoco_reference_is_read_from_the_frozen_audit() -> None:
    """The comparison baseline is read from ``results/proposal_audit/``, not retyped."""
    sys.path.insert(0, str(_REPO / "scripts"))
    import run_phase1e_feasibility as driver  # noqa: WPS433

    ref = driver._refcoco_reference()  # noqa: SLF001 - white-box contract check
    assert {"source_dir", "recall", "availability", "comparison_caveat"} <= set(ref)
    assert "error" not in ref["recall"] and "error" not in ref["availability"]
    assert ref["recall"]["N"] == str(driver.fe.N_PROPOSALS)
    for key in ("gt_object_recall@0.5", "ref_target_recall@0.5", "ref_target_recall@0.7"):
        assert 0.0 < float(ref["recall"][key]) <= 1.0, key
    # same-category availability is keyed by K, so the K=5 primary row cannot be
    # mistaken for the easiest threshold in that table
    k5 = ref["availability"][f"k{driver.fe.PRIMARY_K}"]
    assert k5["N"] == str(driver.fe.N_PROPOSALS) and k5["threshold"] == str(driver.fe.PRIMARY_K - 1)
    assert int(k5["num_expressions"]) > 0 and 0.0 < float(k5["frac"]) <= 1.0
    assert "A9.4" in ref["comparison_caveat"]


# ---------------------------------------------------------------------------
# 10 / 13 - difficulty metadata
# ---------------------------------------------------------------------------
def test_difficulty_distribution_keeps_official_levels(cohort) -> None:
    rows = fc.difficulty_distribution(_expressions(cohort))
    levels = [r["level"] for r in rows]
    assert levels == [1, 2, 3, "all"]                      # never re-merged (section 13)
    assert rows[-1]["n_expressions"] == 3
    assert sum(r["n_expressions"] for r in rows[:-1]) == rows[-1]["n_expressions"]
    assert abs(sum(r["share_of_test"] for r in rows[:-1]) - 1.0) < 1e-4   # per-row rounding
    dog = next(r for r in rows if r["level"] == 2)
    assert dog["graph_coverage"] == 1.0 and dog["name_from_graph_box"] == 1.0
    assert dog["same_name_objects_mean"] == 2.0            # the two "dog" objects


def test_difficulty_distribution_does_not_silently_drop_unexpected_level() -> None:
    exprs = _wide_cohort()
    exprs[0].level = 7                                      # out-of-range level
    rows = fc.difficulty_distribution(exprs)
    assert [r["level"] for r in rows] == [1, 2, 3, 7, "all"]
    assert sum(r["n_expressions"] for r in rows[:-1]) == rows[-1]["n_expressions"]


def test_tuple_type_distribution_reportable_threshold(cohort) -> None:
    rows = fc.tuple_type_distribution(_expressions(cohort))          # min_group = 300
    by_tt = {r["tuple_type"]: r for r in rows}
    assert set(by_tt) == {"0_hop", "1_hop", "2_hop"}
    assert all(r["reportable"] is False for r in rows)               # tiny groups
    assert fc.tuple_type_distribution(_expressions(cohort), min_group=1)[0]["reportable"] is True
    assert by_tt["1_hop"]["level2"] == 1 and by_tt["2_hop"]["level3"] == 1
    assert by_tt["0_hop"]["level1"] == 1


def test_expression_row_preserves_metadata(cohort) -> None:
    row = next(e for e in _expressions(cohort) if e.expr_id == 12).to_row()
    assert row["level"] == 1 and row["tuple_type"] == "0_hop"
    assert row["target_name"] == "dog" and row["target_name_source"] == "graph_box"
    assert row["spatial"] == "on the left"
    assert row["gt_box_xyxy"] == "300.0,100.0,380.0,160.0"


# ---------------------------------------------------------------------------
# scene-graph join: target resolution and same-name identity
# ---------------------------------------------------------------------------
def test_resolve_target_object_prefers_box_match_then_objects_id(cohort) -> None:
    by_id = {e.expr_id: e for e in _expressions(cohort)}
    assert by_id[11].target_name_source == "graph_box" and by_id[11].target_object_id == "0"
    assert by_id[11].n_same_name == 2 and by_id[11].n_same_name_distractors == 1
    assert by_id[11].target_box_iou > 0.99
    # expression 13 points where the graph has nothing -> labelled fallback
    assert by_id[13].target_name_source == "objects_id_first" and by_id[13].target_name == "person"
    assert by_id[13].n_same_name == 1


def test_same_name_supply_is_exact_string_identity() -> None:
    objects = _graph_objects(
        ("0", "dog", 0, 0, 5, 5), ("1", "Dog", 0, 0, 5, 5), ("2", "dog ", 0, 0, 5, 5),
        ("3", "dogs", 0, 0, 5, 5), ("4", "dog", 0, 0, 5, 5),
    )
    assert fc.same_name_supply(objects, "dog") == 2
    assert fc.same_name_supply(objects, "Cat") == 0
    assert fc.same_name_supply(objects, None) == 0
    assert fc.same_name_supply({}, "dog") == 0


def test_graph_objects_have_no_scene_graph_exit_path(cohort, tmp_path) -> None:
    ann_dir, graph_path = cohort
    missing = tmp_path / "empty_graphs.json"
    missing.write_text("{}", encoding="utf-8")
    expressions = fc.load_test_expressions(ann_dir, missing, require_scene_graph=False)
    assert all(not e.graph_present and e.target_name is None for e in expressions)
    with pytest.raises(ValueError, match="no public scene graph"):
        fc.load_test_expressions(ann_dir, missing, require_scene_graph=True)


# ---------------------------------------------------------------------------
# 11 - image-clustered bootstrap keeps the audit's cluster key
# ---------------------------------------------------------------------------
def test_paired_bootstrap_clusters_by_image_id() -> None:
    rows = fe.audit_cohort(
        [
            fe.ProposalAuditRow(
                expr_id=i, image_id=1 if i < 4 else 2, level=1, tuple_type="0_hop",
                target_name="dog", n_proposals=64, target_best_iou=0.9, target_present=True,
                n_equivalent=1, n_valid_distractors=63, n_invalid_crop=0,
                n_same_name_distractors=0,
            )
            for i in range(8)
        ]
    )
    cluster_ids = np.asarray([r["image_id"] for r in rows], dtype=np.int64)
    assert cluster_ids.tolist() == [1, 1, 1, 1, 2, 2, 2, 2]     # image_id is the cluster key
    a = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    b = a - 0.05
    clustered = paired_bootstrap(a, b, cluster_ids=cluster_ids, n_replicates=200, seed=3)
    assert clustered["resample_unit"] == "cluster" and clustered["n_clusters"] == 2
    plain = paired_bootstrap(a, b, n_replicates=200, seed=3)
    assert plain["resample_unit"] == "sample"
    # an expression-level (non-clustered) interval would be narrower than the image one
    assert float(clustered["std"]) > 0.0
    with pytest.raises(ValueError, match="at least two clusters"):
        paired_bootstrap(a, b, cluster_ids=np.ones(8, dtype=int), n_replicates=50)


# ---------------------------------------------------------------------------
# F2/F3/F4 aggregates and the pre-registered decision rules
# ---------------------------------------------------------------------------
def test_iou_tag_is_the_only_key_naming_rule() -> None:
    assert fe._iou_tag(0.5) == "05" and fe._iou_tag(0.7) == "07"  # noqa: SLF001
    with pytest.raises(ValueError, match="multiple of 0.1"):
        fe._iou_tag(0.95)  # noqa: SLF001
    summary = fe.summarise_rows(_rows())
    assert set(summary) >= {"recall_at_05", "recall_at_07", "gt_object_recall_at_05"}
    assert "recall_at_50" not in summary        # the old 0.5 -> "50" slip must not return
    row = fe.audit_cohort(_rows())[0]
    assert {"target_present_at_05", "object_recall_at_05", "available_K5"} <= set(row)


def _rows() -> List[fe.ProposalAuditRow]:
    out: List[fe.ProposalAuditRow] = []
    for i in range(10):
        present = i < 9                       # one natural omission
        k5 = present and i != 1               # one expression without 4 distractors
        out.append(
            fe.ProposalAuditRow(
                expr_id=i,
                image_id=100 + (i // 2),       # 5 images, 2 expressions each
                level=1 + (i % 3),
                tuple_type="0_hop" if i % 2 else "1_hop",
                target_name="dog",
                n_proposals=64,
                target_best_iou=0.95 if present else 0.3,
                target_present=present,
                n_equivalent=1,
                n_valid_distractors=63 if present else 64,
                n_invalid_crop=0,
                n_same_name_distractors=5 if k5 and i < 4 else 0,
                available={fe.PRIMARY_K: k5, fe.SECONDARY_K: k5},
                same_name_available={fe.PRIMARY_K: k5 and i < 4, fe.SECONDARY_K: False},
                object_recall={0.5: 0.8, 0.7: 0.6},
                redundancy={"frac_pairs_gt_mid": 0.1, "frac_pairs_gt_high": 0.0},
            )
        )
    return out


def test_summarise_rows_arithmetic() -> None:
    summary = fe.summarise_rows(_rows())
    assert summary["n_rows"] == 10 and summary["n_images"] == 5
    assert summary["recall_at_05"]["rate"] == pytest.approx(0.9)
    assert summary["natural_omission_at_05"] == pytest.approx(0.1)
    assert summary["k5_availability"]["rate"] == pytest.approx(0.8)
    assert summary["k5_availability"]["same_name_rate"] == pytest.approx(0.3)
    assert summary["same_name_distractors"]["share_ge_4"] == pytest.approx(0.3)
    assert summary["same_name_distractors"]["share_ge_1"] == pytest.approx(0.3)
    # Wilson intervals bracket the point estimate and stay inside [0, 1]
    for key in ("recall_at_05", "k5_availability"):
        block = summary[key]
        assert 0.0 <= block["wilson_low"] <= block["rate"] <= block["wilson_high"] <= 1.0
    assert set(summary["by_level"]) == {"1", "2", "3"}
    assert summary["by_tuple_type"]["0_hop"]["n_rows"] == 5
    assert summary["by_tuple_type"]["0_hop"]["reportable"] is False   # 5 < 300


def test_engineering_verdict_boundaries() -> None:
    def fake(recall: float, k5: float) -> Dict[str, Any]:
        return {
            "recall_at_05": {"rate": recall},
            f"k{fe.PRIMARY_K}_availability": {"rate": k5},
        }

    assert fe.engineering_verdict(fake(0.95, 0.95))["verdict"] == "EXTERNAL GO"
    assert fe.engineering_verdict(fake(0.90, 0.90))["verdict"] == "EXTERNAL GO"   # >= is inclusive
    assert fe.engineering_verdict(fake(0.89, 0.95))["verdict"] == "ENGINEERING GRAY ZONE"
    assert fe.engineering_verdict(fake(0.95, 0.50))["verdict"] == "ENGINEERING GRAY ZONE"
    assert fe.engineering_verdict(fake(0.79, 0.99))["verdict"] == "EXTERNAL STOP"
    assert fe.engineering_verdict({"n_rows": 0})["verdict"] == "AUDIT INCOMPLETE"
    # the generator may never be adapted to the verdict (sections 6/7)
    assert "forbidden" in fe.engineering_verdict(fake(0.5, 0.5))["note"]


@pytest.mark.parametrize(
    "same_rate,branch",
    [
        (0.95, "same_category_primary"),
        (0.90, "same_category_primary"),
        (0.75, "level_primary_same_category_diagnostic"),
        (0.50, "level_primary_same_category_diagnostic"),
        (0.015, "level_primary_only"),
        (float("nan"), "level_primary_only"),
    ],
)
def test_candidate_regime_branch_selection(same_rate: float, branch: str) -> None:
    summary = {
        f"k{fe.PRIMARY_K}_availability": {
            "rate": 0.97,
            "same_name_rate": same_rate,
            "same_name_wilson_low": same_rate - 0.01,
        },
        "same_name_distractors": {"share_ge_4": same_rate},
        "n_names": 220,
    }
    decision = fe.candidate_regime_decision(summary)
    assert decision["branch"] == branch
    if branch == "same_category_primary":
        assert "same-name" in decision["primary_hard_regime"]
    else:
        assert "official level" in decision["primary_hard_regime"]
        assert "random K=5" in decision["primary_hard_regime"]


def test_thresholds_match_the_pre_registered_protocol() -> None:
    """The F4 criteria come from the instruction, not from the observed numbers."""
    assert (fe.GO_RECALL_MIN, fe.GO_AVAILABILITY_MIN, fe.STOP_RECALL_MAX) == (0.90, 0.90, 0.80)
    assert fe.SAME_CATEGORY_PRIMARY_MIN_AVAILABILITY == fe.GO_AVAILABILITY_MIN
    assert fe.MIN_GROUP_FOR_TUPLE_TYPE == 300
    assert fe.INVALID_CROP_MIN_SIDE == 4.0
    assert fe.RECALL_IOUS == (0.5, 0.7)


# ---------------------------------------------------------------------------
# 12 - the negative branch is counted, never modelled
# ---------------------------------------------------------------------------
def _write_all_split(ann_dir: Path, n_pos: int = 3, n_text: int = 4, n_image: int = 5) -> None:
    rows: Dict[str, Any] = {}
    for i in range(n_pos):
        rows[str(i)] = {"id": i, "level": 1 + (i % 3), "tuple_type": "0_hop", "image_id": str(230 + i)}
    for i in range(n_text):
        rows[f"t{i}"] = {
            "id": i, "level": 1, "tuple_type": "1_hop", "image_id": str(230 + i),
            "negative_cate": "text", "negative_type": "attribute", "negative_level": 1,
        }
    for i in range(n_image):
        rows[f"i{i}"] = {
            "id": i, "level": 2, "tuple_type": "0_hop", "image_id": f"neg_{i}",
            "negative_cate": "image", "negative_type": "object", "negative_level": 2,
        }
    (ann_dir / fc.VANILLA_ALL_FILE).write_text(json.dumps(rows), encoding="utf-8")


def test_negative_counts_never_build_expressions(tmp_path) -> None:
    ann_dir = tmp_path / "finecops"
    ann_dir.mkdir()
    _write_all_split(ann_dir)
    counts = fc.negative_feasibility_counts(ann_dir)
    assert counts["n_positive"] == 3
    assert counts["n_negative_text"] == 4 and counts["n_negative_image"] == 5
    assert counts["n_images_negative_image"] == 5        # neg_1..neg_5 are separate images
    assert counts["negative_type"] == {"attribute": 4, "object": 5, "None": 3}
    assert counts["negative_cate"] == {"None": 3, "image": 5, "text": 4}
    assert "bbox" not in json.dumps(counts)              # no geometry, nothing to model
    assert counts["positive_level"] == {"1": 1, "2": 1, "3": 1}


def test_negative_rows_cannot_enter_the_positive_loader(cohort, tmp_path) -> None:
    ann_dir, graph_path = cohort
    _write_all_split(ann_dir)
    doc = json.loads((ann_dir / fc.COCO_POS_FILE).read_text(encoding="utf-8"))
    neg_image = {"id": 99, "file_name": "images/neg_1.jpg", "width": 500, "height": 375,
                 "caption": "", "dataset_name": "ref"}
    doc["images"].append(neg_image)
    doc["annotations"].append(_annotation(99, 99, [0, 0, 10, 10]))
    (ann_dir / fc.COCO_POS_FILE).write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric image id"):
        fc.load_test_expressions(ann_dir, graph_path)


# ---------------------------------------------------------------------------
# F0b - image geometry bookkeeping
# ---------------------------------------------------------------------------
def test_image_dim_rows_flag_mismatch(cohort) -> None:
    expressions = _expressions(cohort)
    dims = {_IMAGE_A: (500, 375), _IMAGE_B: (400, 500)}   # B was resized somewhere
    rows = fc.image_dim_rows(expressions, dims)
    assert [r["image_id"] for r in rows] == [_IMAGE_A, _IMAGE_B]
    assert rows[0]["ann_matches_actual"] is True and rows[0]["ann_matches_graph"] is True
    assert rows[1]["ann_matches_actual"] is False
    assert rows[0]["n_expressions"] == 2                  # both COCO image rows map to GQA 230
    assert rows[1]["n_expressions"] == 1


def test_boxes_and_overlap_reports_are_bounded(cohort) -> None:
    expressions = _expressions(cohort)
    geometry = fc.boxes_inside_image(expressions)
    assert geometry["n_expressions"] == 3
    assert 0.0 <= geometry["share_box_outside_frame"] <= 1.0
    assert geometry["box_area_share_median"] > 0.0
    overlap = fc.target_pair_overlaps(expressions)
    assert overlap["n_images"] == 2 and overlap["n_same_image_pairs"] == 1
    assert overlap["n_overlapping_target_pairs"] == 0     # the two dogs do not overlap


def test_member_name_and_local_path_mapping() -> None:
    assert gi.member_name(230) == "images/230.jpg"
    assert gi.IMAGE_ENTRY_TEMPLATE.format(gqa_image_id="230") == "images/230.jpg"
    assert gi.local_image_path(Path("data/raw/gqa/images"), 230).name == "230.jpg"
    report = gi.ImageFetchReport(requested=3, fetched=2, skipped_existing=1)
    report.failed.append(("7", "range_fetch", "HTTPError: 503"))
    payload = report.to_dict()
    assert payload["present_after_run"] == 3 and payload["n_failed"] == 1
    assert payload["failed"][0][0] == "7" and "503" in payload["failed"][0][1]


# ---------------------------------------------------------------------------
# F0b - range reader: one member per request, verified against the index
# ---------------------------------------------------------------------------
class _FakeRange:
    """Stand-in for :class:`gi.RangeConnection` over an in-memory archive."""

    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.n_requests = 0
        self.n_bytes = 0
        self.reset_calls = 0

    def get_range(self, start: int, end: int) -> bytes:
        self.n_requests += 1
        part = self.buf[start : end + 1]
        self.n_bytes += len(part)
        return part

    def reset(self) -> None:
        self.reset_calls += 1

    def close(self) -> None:
        pass


def _archive(payloads: Dict[str, bytes], *, stored: bool = False) -> bytes:
    buf = io.BytesIO()
    mode = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buf, "w", mode) as zf:
        for name, data in payloads.items():
            zf.writestr(name, data)
    return buf.getvalue()


_LOCAL = gi._LOCAL_HEADER


def _index(raw: bytes, only: Any = None) -> Dict[str, gi.MemberLocation]:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        return {
            info.filename: gi.MemberLocation(
                name=info.filename,
                header_offset=int(info.header_offset),
                compress_size=int(info.compress_size),
                file_size=int(info.file_size),
                compress_type=int(info.compress_type),
                crc=int(info.CRC),
                name_len=len(info.filename.encode("utf-8")),
            )
            for info in zf.infolist()
            if only is None or info.filename in only
        }


def test_member_locations_read_the_central_directory_only(monkeypatch) -> None:
    payloads = {gi.member_name(i): bytes([i % 251]) * 3000 for i in (230, 231, 232)}
    raw = _archive(payloads)
    opened = {"n": 0}

    def fake_open(url: str, *, timeout: float = 120.0):
        opened["n"] += 1
        return zipfile.ZipFile(io.BytesIO(raw))

    monkeypatch.setattr(gi, "open_remote_zip", fake_open)
    table = gi.member_locations("https://example.invalid/images.zip", only=[gi.member_name(230)])
    assert opened["n"] == 1                       # one tail read, not one per member
    assert set(table) == {gi.member_name(230)}    # and the table stays small
    location = table[gi.member_name(230)]
    assert location.file_size == 3000 and location.name_len == len(gi.member_name(230))


def test_fetch_member_returns_exact_bytes_in_one_request() -> None:
    payload_a = b"compressible " * 4000                       # deflated
    payload_b = np.random.default_rng(1).standard_normal(200).tobytes()  # incompressible
    payload_c = b"x" * 5000                                   # stored
    raw = _archive({gi.member_name(230): payload_a, gi.member_name(231): payload_b})
    stored = _archive({gi.member_name(232): payload_c}, stored=True)
    cases = [
        (raw, gi.member_name(230), payload_a),
        (raw, gi.member_name(231), payload_b),
        (stored, gi.member_name(232), payload_c),
    ]
    for buf, name, want in cases:
        conn = _FakeRange(buf)
        assert gi.fetch_member(conn, _index(buf)[name]) == want
        assert conn.n_requests == 1               # no zipfile read amplification


def test_fetch_member_retries_a_span_that_came_back_short() -> None:
    class _ShortRange(_FakeRange):
        """Host truncates range answers (a long local extra field, or a short read)."""

        def __init__(self, buf: bytes, *, keep: int, first_only: bool = True) -> None:
            super().__init__(buf)
            self.keep = int(keep)
            self.first_only = first_only

        def get_range(self, start: int, end: int) -> bytes:
            body = super().get_range(start, end)
            if self.first_only and self.n_requests > 1:
                return body
            return body[: self.keep]

    want = b"gqa jpeg bytes" * 250                # stored -> csize == usize == 3500
    raw = _archive({gi.member_name(230): want}, stored=True)
    conn = _ShortRange(raw, keep=200)
    assert gi.fetch_member(conn, _index(raw)[gi.member_name(230)]) == want
    assert conn.n_requests == 2                   # one grow-and-refetch retry

    forever = _ShortRange(raw, keep=200, first_only=False)
    with pytest.raises(ValueError):
        gi.fetch_member(forever, _index(raw)[gi.member_name(230)], attempts=3)
    assert forever.n_requests == 3                # bounded, never an infinite read


def test_decode_member_span_rejects_a_wrong_header_offset() -> None:
    raw = _archive({gi.member_name(230): b"payload " * 500, gi.member_name(231): b"other" * 500})
    table = _index(raw)
    wrong = table[gi.member_name(230)]
    body = raw[wrong.header_offset + 8 : wrong.header_offset + 8 + wrong.span()]
    with pytest.raises(ValueError, match="bad local header signature"):
        gi.decode_member_span(body, wrong)


def test_decode_member_span_rejects_corruption_and_truncation() -> None:
    raw = bytearray(_archive({gi.member_name(232): b"x" * 5000}, stored=True))
    table = _index(bytes(raw))
    location = table[gi.member_name(232)]
    body = bytearray(raw[location.header_offset : location.header_offset + location.span()])
    body[_LOCAL.size + location.name_len + 10] ^= 0xFF   # flip one data byte
    with pytest.raises(ValueError, match="CRC mismatch"):
        gi.decode_member_span(bytes(body), location)
    good = bytes(body)
    with pytest.raises(ValueError):
        gi.decode_member_span(good[: len(good) // 2], location)   # truncated
    with pytest.raises(ValueError, match="shorter than a local header"):
        gi.decode_member_span(b"", location)


def test_range_connection_refuses_a_non_https_url() -> None:
    with pytest.raises(ValueError, match="https"):
        gi.RangeConnection("http://example.invalid/images.zip")


def test_fetch_images_writes_verified_files_and_is_idempotent(monkeypatch, tmp_path) -> None:
    ids = ["230", "231", "999"]
    payloads = {gi.member_name(i): b"body-" + i.encode() for i in ids[:2]}
    raw = _archive(payloads, stored=True)
    monkeypatch.setattr(
        gi, "member_locations",
        lambda url, *, timeout=120.0, only=None: _index(raw, only),
    )
    monkeypatch.setattr(gi, "RangeConnection", lambda url, *, timeout=120.0: _FakeRange(raw))

    report = gi.fetch_images(ids, tmp_path, url="https://example.invalid/images.zip", workers=2)
    assert report.fetched == 2
    assert report.missing_in_archive == ["999"]     # never faked, always reported
    assert report.bytes_written == sum(len(v) for v in payloads.values())
    assert report.bytes_transferred >= report.bytes_written
    assert report.requests == 2
    assert (tmp_path / "230.jpg").read_bytes() == payloads[gi.member_name("230")]
    assert list(tmp_path.glob("*.part")) == []      # atomic replace left no debris

    again = gi.fetch_images(ids, tmp_path, url="https://example.invalid/images.zip", workers=2)
    assert (again.skipped_existing, again.fetched) == (2, 0)     # re-runs only fill the gaps
    payload = again.to_dict()
    assert payload["n_missing_in_archive"] == 1
    assert payload["present_after_run"] == 2


def test_target_size_buckets_partition_the_audit_rows() -> None:
    """The report-only size stratification must be exhaustive and non-overlapping.

    A bucket that silently drops or double-counts rows would misread the section 32
    item-10 domain-shift diagnosis, so the labels are pinned at the boundaries too.
    """
    sys.path.insert(0, str(_REPO / "scripts"))
    import run_phase1e_feasibility as driver  # noqa: WPS433

    assert [driver._size_bucket(s) for s in (0.0, 16.0, 31.9, 32.0, 63.9, 64.0,  # noqa: SLF001
                                             127.9, 128.0, 255.9, 256.0, 900.0)] == [
        "<32px", "<32px", "<32px", "<64px", "<64px", "<128px",
        "<128px", "<256px", "<256px", ">=256px", ">=256px",
    ]

    sides = [20.0, 50.0, 100.0, 200.0, 400.0] + [90.0] * 7
    rows = [
        fe.ProposalAuditRow(
            expr_id=i, image_id=1, level=1, tuple_type="0_hop", target_name="dog",
            n_proposals=64, target_best_iou=0.8, target_present=True, n_equivalent=1,
            n_valid_distractors=60, n_invalid_crop=0, n_same_name_distractors=0,
            available={5: True, 10: True}, same_name_available={5: False, 10: False},
        )
        for i in range(len(sides))
    ]
    areas = {r.expr_id: (s, 0.05) for r, s in zip(rows, sides)}
    table = driver._by_target_size(rows, areas)  # noqa: SLF001 - white-box contract check

    assert sum(int(r["n_rows"]) for r in table) == len(rows)
    assert len({r["target_side_bucket"] for r in table}) == len(table)
    by_bucket = {r["target_side_bucket"]: r for r in table}
    assert by_bucket["<128px"]["n_rows"] == 8          # 100.0 plus the seven 90.0 sides
    assert abs(by_bucket["<128px"]["share_of_rows"] - 8 / len(rows)) < 1e-6  # rounded to 6 dp
    assert by_bucket[">=256px"]["recall_at_05"] == 1.0
    assert by_bucket["<32px"]["k5_same_name_availability"] == 0.0
