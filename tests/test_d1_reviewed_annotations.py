"""D1 (V2-D) reviewed RefCOCO+ annotation-robustness tests.

Covers the protocol section 25 checklist:

- reviewed source checksum              (test_reviewed_source_checksums_agree)
- mapping deterministic                 (test_build_mapping_deterministic_and_classifies)
- original dataset untouched            (test_original_dataset_untouched)
- no ambiguous row in clean cohort      (test_clean_cohort_has_no_ambiguous_or_removed_row)
- target identity consistency           (test_proposal_audit_target_identity_consistency)
- proposal mapping recomputed when box
  changes                               (test_box_change_breaks_verification_and_identity,
                                         audit recomputed=True in the artifact)
- same-category construction valid      (test_same_category_construction_rule)
- no retraining path                    (test_stage2_has_no_retraining_path)
- frozen checkpoint hashes identical    (test_frozen_input_hashes_identical)
- paired bootstrap image-level          (test_paired_bootstrap_is_image_level)

Synthetic-fixture tests always run; artifact tests skip when the (large) D1
result directory is absent so the suite stays runnable on a fresh clone.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
ARTIFACTS = REPO / "results" / "v2_data_robustness" / "d1_reviewed_annotations"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_d1_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_d1_reviewed = _load_script("run_d1_reviewed")
needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "metadata.json").exists(),
    reason="D1 artifacts not present (run scripts/run_d1_reviewed.py first)",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# synthetic fixtures for the pure mapping functions
# ---------------------------------------------------------------------------
@pytest.fixture()
def mapping_fixture():
    refs = [
        {
            "split": "val", "image_id": 100, "ann_id": 10, "ref_id": 1,
            "sentences": [
                {"sent_id": 1, "sent": "the red cat"},
                {"sent_id": 2, "sent": "a red cat"},
            ],
        },
        {
            "split": "val", "image_id": 101, "ann_id": 11, "ref_id": 2,
            "sentences": [{"sent_id": 3, "sent": "the blue dog"}],
        },
        {
            "split": "testA", "image_id": 102, "ann_id": 12, "ref_id": 3,
            "sentences": [{"sent_id": 4, "sent": "the left person"}],
        },
    ]
    reviewed = {
        "val": {
            (100, 10): [
                {"caption": "the red cat", "bbox": [10.0, 20.0, 30.0, 40.0], "quality": 1},
                {"caption": "a red cat", "bbox": [10.0, 20.0, 30.0, 40.0], "quality": 1},
            ],
            (101, 11): [
                {"caption": "the blue dog", "bbox": [1.0, 2.0, 3.0, 4.0], "quality": 0},
            ],
        },
        "testA": {},  # ref 3 has no reviewed rows at all -> UNMATCHED
    }
    bbox = {10: [10.0, 20.0, 30.0, 40.0], 11: [1.0, 2.0, 3.0, 4.0], 12: [5.0, 6.0, 7.0, 8.0]}
    return refs, reviewed, bbox


def test_build_mapping_deterministic_and_classifies(mapping_fixture):
    refs, reviewed, bbox = mapping_fixture
    rows_a = run_d1_reviewed.build_mapping(refs, reviewed, bbox)
    rows_b = run_d1_reviewed.build_mapping(list(reversed(refs)), reviewed, bbox)
    key = lambda row: (row["split"], row["sent_id"])  # noqa: E731
    assert [key(r) for r in sorted(rows_a, key=key)] == [key(r) for r in sorted(rows_b, key=key)]
    for row_a in rows_a:
        row_b = next(r for r in rows_b if key(r) == key(row_a))
        assert row_a == row_b  # identical payloads regardless of input order

    by = {r["sent_id"]: r for r in rows_a}
    assert by[1]["status"] == "UNCHANGED" and by[1]["box_match"] and by[1]["text_match"]
    assert by[2]["status"] == "UNCHANGED"
    assert by[3]["status"] == "REMOVED" and by[3]["caption_quality"] == 0
    assert by[4]["status"] == "UNMATCHED" and by[4]["caption_quality"] is None

    stats = run_d1_reviewed.mapping_statistics(rows_a)
    assert stats["all"]["unchanged"] == 2
    assert stats["all"]["removed"] == 1
    assert stats["all"]["unmatched_expressions"] == 1
    # text_mismatch_rows counts every non-verified row incl. the UNMATCHED group;
    # no *mapped* row may have a text mismatch (kept rows all text_match=True)
    assert stats["all"]["text_mismatch_rows"] == 1
    kept = [r for r in rows_a if r["status"] in ("UNCHANGED", "REMOVED")]
    assert all(r["text_match"] for r in kept)


def test_box_change_breaks_verification_and_identity(mapping_fixture):
    """A reviewed row whose bbox differs from instances.json is never silently
    trusted: box_match flips to False and the group text identity is intact
    (the clean-cohort definition requires the box verification separately)."""
    refs, reviewed, bbox = mapping_fixture
    reviewed["val"][(100, 10)][0]["bbox"] = [10.0, 20.0, 31.0, 40.0]  # 1px wider
    rows = run_d1_reviewed.build_mapping(refs, reviewed, bbox)
    by = {r["sent_id"]: r for r in rows}
    assert by[1]["status"] == "UNCHANGED" and not by[1]["box_match"]
    assert by[2]["box_match"]
    stats = run_d1_reviewed.mapping_statistics(rows)
    assert stats["all"]["box_mismatch_rows"] == 1
    assert stats["all"]["unchanged_verified"] == 1  # only sent 2 survives verification


def test_text_group_mismatch_reports_unmatched_not_force_aligned(mapping_fixture):
    refs, reviewed, bbox = mapping_fixture
    reviewed["val"][(100, 10)][1]["caption"] = "a red kitten"  # text no longer matches
    rows = run_d1_reviewed.build_mapping(refs, reviewed, bbox)
    by = {r["sent_id"]: r for r in rows}
    assert by[1]["status"] == "UNMATCHED" and by[2]["status"] == "UNMATCHED"
    assert not by[1]["text_match"]


def test_status_taxonomy_is_honest(mapping_fixture):
    """Section 18: the reviewed source supports only UNCHANGED/REMOVED/UNMATCHED."""
    refs, reviewed, bbox = mapping_fixture
    rows = run_d1_reviewed.build_mapping(refs, reviewed, bbox)
    assert {row["status"] for row in rows} <= {"UNCHANGED", "REMOVED", "UNMATCHED"}
    if (ARTIFACTS / "feasibility.json").exists():
        feas = json.loads((ARTIFACTS / "feasibility.json").read_text(encoding="utf-8"))
        supported = feas["status_taxonomy_supported"]
        blob = json.dumps(supported)
        assert "CORRECTED_BOX" in blob and "not supported" in blob.lower()


def test_same_category_construction_rule():
    """The frozen candidate-audit rule: target category from its own proposal,
    unknown (-1) never matches, target and removed rows excluded."""
    from ccg.data.audit import same_category_counts

    categories = np.array([5, 5, 7, 5, -1, -1])
    # target at idx 0 (cat 5); distractors 1 (5), 2 (7), 3 (5), 4/5 unknown.
    assert same_category_counts(categories, target_idx=0, to_remove_idx=np.array([3])) == 1
    assert same_category_counts(categories, target_idx=0, to_remove_idx=np.array([])) == 2
    # unknown target category -> 0 ("no matchable category"), never a silent count
    assert same_category_counts(categories, target_idx=4, to_remove_idx=np.array([])) == 0
    assert same_category_counts(categories, target_idx=None, to_remove_idx=np.array([])) == 0


def test_assign_target_recomputed_when_box_changes():
    """Section 9: the target mapping is an IoU recomputation against the frozen
    bank, not a reused index - a shifted box lands on a different row."""
    from ccg.data.proposals import assign_target

    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]], dtype=np.float32)
    bank = SimpleNamespace(image_id=1, boxes=boxes, objectness=np.array([0.9, 0.8], dtype=np.float32))
    a = assign_target(bank, np.array([0.5, 0.5, 9.5, 9.5], dtype=np.float32))
    b = assign_target(bank, np.array([100.5, 100.5, 109.5, 109.5], dtype=np.float32))
    assert a.target_proposal_idx == 0 and b.target_proposal_idx == 1
    miss = assign_target(bank, np.array([500.0, 500.0, 510.0, 510.0], dtype=np.float32))
    assert miss.is_miss and miss.target_proposal_idx is None


def test_gate_functions_follow_frozen_criteria():
    assert run_d1_reviewed.proposal_gate(0.95)["verdict"] == "CONTINUE"
    assert run_d1_reviewed.proposal_gate(0.92)["verdict"] == "REVIEWED-PROPOSAL GRAY"
    assert run_d1_reviewed.proposal_gate(0.89)["verdict"] == "STOP"
    assert run_d1_reviewed.size_gate(3000, 500)["verdict"] == "FULL_ROBUSTNESS_AUDIT"
    assert run_d1_reviewed.size_gate(1500, 300)["verdict"] == "LIMITED_ROBUSTNESS_AUDIT"
    assert run_d1_reviewed.size_gate(999, 100)["verdict"] == "UNDERPOWERED"
    assert not run_d1_reviewed.size_gate(999, 100)["proceed"]


# ---------------------------------------------------------------------------
# artifact-level tests
# ---------------------------------------------------------------------------
@needs_artifacts
def test_reviewed_source_checksums_agree():
    manifest = json.loads((ARTIFACTS / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_sha256"] == run_d1_reviewed.EXPECTED_SHA256
    for name, recorded in manifest["files"].items():
        assert recorded["sha256"] == manifest["expected_sha256"][name], f"manifest drift: {name}"
        path = run_d1_reviewed.RAW_DIR / name
        if path.exists():
            assert _sha256(path) == recorded["sha256"], f"reviewed source {name} changed on disk"


@needs_artifacts
def test_original_dataset_untouched():
    """Section 5: the V1 originals are byte-identical to what stage 1 audited."""
    meta = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    frozen = meta["frozen_inputs_sha256"]
    for label, path in (
        ("data/raw/refcoco+/refcoco+/refs(unc).p", run_d1_reviewed.REFS_PATH),
        ("data/raw/refcoco+/refcoco+/instances.json", run_d1_reviewed.INSTANCES_PATH),
    ):
        assert _sha256(path) == frozen[label], f"original dataset file changed: {label}"


@needs_artifacts
def test_clean_cohort_has_no_ambiguous_or_removed_row():
    mapping = {
        (row["split"], int(row["sent_id"])): row
        for row in csv.DictReader((ARTIFACTS / "mapping_audit.csv").open(encoding="utf-8", newline=""))
    }
    clean = list(csv.DictReader((ARTIFACTS / "clean_manifest.csv").open(encoding="utf-8", newline="")))
    assert clean, "clean_manifest.csv is empty"
    seen = set()
    for row in clean:
        key = (row["split"], int(row["sent_id"]))
        assert key not in seen, f"duplicate (split, sent_id) in clean cohort: {key}"
        seen.add(key)
        source = mapping[key]
        assert source["status"] == "UNCHANGED"
        assert source["text_match"] == "True" and source["box_match"] == "True"


@needs_artifacts
def test_proposal_audit_target_identity_consistency():
    audit = json.loads((ARTIFACTS / "proposal_audit.json").read_text(encoding="utf-8"))
    assert audit["recomputed_target_mapping"] is True
    assert audit["manifest_consistency"]["target_index_mismatches"] == 0
    assert audit["manifest_consistency"]["target_max_iou_mismatches"] == 0
    assert audit["manifest_consistency"]["n_same_category_mismatches"] == 0
    assert audit["manifest_consistency"]["target_max_iou_max_abs_delta"] == 0.0
    assert audit["proposal_gate"]["verdict"] == "CONTINUE"
    assert audit["box_changed_by_review"] is False


@needs_artifacts
def test_frozen_input_hashes_identical():
    meta = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    for label, sha in meta["frozen_inputs_sha256"].items():
        path = REPO / label
        assert path.exists(), f"frozen input missing: {label}"
        assert _sha256(path) == sha, f"frozen input changed: {label}"


def test_stage2_has_no_retraining_path():
    """Section 20: the stage-2 script must not import or call any trainer /
    refit entry point (static source scan)."""
    source = (SCRIPTS / "run_d1_c1_c4.py").read_text(encoding="utf-8")
    forbidden = [
        r"\bb3_training\b", r"\btrain_b3\b", r"\bIndependentMLPScorer\b",
        r"\bfit_temperature\b", r"\brefit\b", r"\b\.fit\s*\(", r"\btorch\b",
        r"\bnn\.Module\b", r"\boptimizer\b", r"\.backward\s*\(",
    ]
    for pattern in forbidden:
        assert not re.search(pattern, source), f"stage-2 source matches training pattern {pattern!r}"
    assert "new_training_parameters\": 0" in source or 'new_training_parameters": 0' in source


@needs_artifacts
def test_paired_bootstrap_is_image_level():
    boot_path = ARTIFACTS / "bootstrap.csv"
    if not boot_path.exists():
        pytest.skip("stage-2 bootstrap.csv not present yet")
    rows = list(csv.DictReader(boot_path.open(encoding="utf-8", newline="")))
    assert rows, "bootstrap.csv is empty"
    for row in rows:
        assert row["resample_unit"] == "image"
        assert int(float(row["n_replicates"])) >= 5000
        assert row["ci_level"] in ("0.95", "0.950000")
    # every reported cluster count must be <= the cohort image count and > 1
    clean = list(csv.DictReader((ARTIFACTS / "clean_manifest.csv").open(encoding="utf-8", newline="")))
    n_images = len({row["image_id"] for row in clean if row["in_phase0b_cohort"] == "True"})
    for row in rows:
        if row["n_clusters"] not in ("", None):
            n_clusters = int(float(row["n_clusters"]))
            assert 1 < n_clusters <= n_images
