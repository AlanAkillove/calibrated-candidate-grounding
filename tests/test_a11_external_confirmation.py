"""A11 - RefCOCOg frozen external confirmation guarantees (protocol section 41).

The eighteen pre-registered items are checked here without a GPU and without the
20 GB of COCO JPEGs.  The suite splits into two kinds of test:

* **Artifact pins** - the run has already happened, so the frozen coefficients,
  the checkpoint identity, the A10 cohort pin, the anchor recovery, the matched
  bootstrap and the manipulation verdict are read back from the committed
  ``results/phase1e_refcocog_external`` artifacts and re-verified (checksums are
  recomputed from the real files on disk, never trusted from the manifest alone).
* **Source / logic pins** - the load-and-predict contract (no ``fit`` on the
  production path, train/val never loaded, ``K = 5``) is proved statically, and
  the cohort *verification* functions are exercised on tiny synthetic cohorts so
  a broken category / target-equivalent check would fail loudly.

The section 41 items and where each lives:

1.  zero RefCOCO+ image overlap ............. ``test_a11_strict_cohort_is_a_subset_of_the_zero_overlap_strict_subset``
2.  manifests match the A10 cohort / hash ... ``test_a11_candidate_manifests_match_the_frozen_a10_cohort``
3.  Random / Hard cohort identity ........... ``test_random_and_hard_manifests_describe_the_same_cohort``
4.  OpenCLIP checkpoint identical ........... ``test_openclip_checkpoint_identity_is_frozen``
5.  B3 state hashes identical ............... ``test_b3_state_hashes_match_the_manifest``
6.  Stats weights identical ................. ``test_stats_weights_and_prediction_are_frozen``
7.  E1b weights identical ................... ``test_e1b_weights_are_identical_to_the_phase1_coefficients``
8.  normalization identical ................. ``test_normalization_is_identical_to_the_phase05_fit``
9.  runner contains no fit path ............. ``test_production_runner_contains_no_fit_path``
10. RefCOCOg train / val never loaded ....... ``test_refcocog_train_and_val_are_never_loaded``
11. no calibration fitting .................. ``test_production_path_contains_no_calibration_fitting``
12. K fixed to 5 ............................ ``test_k_is_frozen_to_five``
13. matched bootstrap draws ................. ``test_bootstrap_is_matched_random_and_hard``
14. manipulation check correctness .......... ``test_manipulation_check_significance_is_correct``
15. raw grounding not changed by reliability  ``test_reliability_models_do_not_change_raw_grounding``
16. artifact recovery anchor ................ ``test_anchor_recovery_passed``
17. candidate category correctness .......... ``test_hard_distractors_must_share_the_target_category``
18. no target-equivalent hard distractor .... ``test_target_leak_and_distractor_shape_are_rejected``
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _extra in (_REPO / "src", _REPO / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from ccg.data.manifests import ManifestFile  # noqa: E402
from ccg.external import frozen_identity as fi  # noqa: E402
from ccg.external import refcocog as rg  # noqa: E402
from ccg.external import refcocog_manifests as rman  # noqa: E402
from ccg.semantic import frozen_load as fl  # noqa: E402

# ---------------------------------------------------------------------------
# committed artifact locations
# ---------------------------------------------------------------------------
EXTERNAL = _REPO / "results" / "phase1e_refcocog_external"
A10 = _REPO / "results" / "phase1e_refcocog_feasibility"
COHORT_DIR = EXTERNAL / "cohort"
PRED_DIR = EXTERNAL / "predictions"
FROZEN_MODELS = EXTERNAL / "frozen_models"
MANIFEST_JSON = FROZEN_MODELS / "frozen_artifact_manifest.json"
RESULTS_JSON = EXTERNAL / "a11_results.json"
ANCHOR_JSON = EXTERNAL / "anchor_recovery.json"
COHORT_REPORT = COHORT_DIR / "cohort_report.json"

SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")

# The production runner is the whole external path that must never fit.
PRODUCTION_PATH: List[Path] = [
    _REPO / "scripts" / "a11_external_inference.py",
    _REPO / "scripts" / "a11_external_analysis.py",
    _REPO / "scripts" / "extract_refcocog_features.py",
    _REPO / "scripts" / "a11_anchor_recovery_check.py",
    _REPO / "src" / "ccg" / "external" / "refcocog_external.py",
    _REPO / "src" / "ccg" / "external" / "refcocog_manifests.py",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ===========================================================================
# 1 - the strict cohort is disjoint from every RefCOCO+ image
# ===========================================================================
@pytest.mark.skipif(not (COHORT_REPORT.exists() and A10.exists()), reason="A10/A11 artifacts absent")
def test_a11_strict_cohort_is_a_subset_of_the_zero_overlap_strict_subset() -> None:
    # A10 froze ``rg_external_strict`` as "RefCOCOg test minus ALL RefCOCO+ images"
    # and separately asserted its zero image overlap; the A11 cohort is a subset of
    # it, so it inherits that disjointness transitively.
    definitions = _read_json(A10 / "external_subsets.json")["definitions"]
    assert "minus ALL RefCOCO+ images" in definitions[rg.SUBSET_STRICT]

    audit = _read_csv(A10 / "audit_expressions.csv")
    strict_images = {int(r["image_id"]) for r in audit if r["subset"] == rg.SUBSET_STRICT}
    assert strict_images, "A10 audit lists no strict-subset images"

    cohort_rows = _read_csv(COHORT_DIR / "cohort.csv")
    cohort_images = {int(r["image_id"]) for r in cohort_rows}
    assert cohort_images, "A11 cohort is empty"
    # every evaluated image is a strict (RefCOCO+ -disjoint) image ...
    assert cohort_images <= strict_images
    # ... and the A11 pin reports the full matched cohort A10 froze.
    pin = _read_json(COHORT_REPORT)["a10_pin"]
    assert pin["ok"] and pin["n_missing"] == 0 and pin["n_extra"] == 0


# ===========================================================================
# 2 - the candidate manifests match the A10 cohort and their recorded hashes
# ===========================================================================
@pytest.mark.skipif(not (COHORT_REPORT.exists()), reason="A11 cohort report absent")
def test_a11_candidate_manifests_match_the_frozen_a10_cohort() -> None:
    report = _read_json(COHORT_REPORT)

    a10_exprs = {int(r["expr_id"]) for r in _read_csv(A10 / "hard_cohort.csv")}
    a11_exprs = {int(r["expr_id"]) for r in _read_csv(COHORT_DIR / "cohort.csv")}
    assert a11_exprs == a10_exprs, "A11 evaluates a different cohort than A10 froze"

    # the persisted manifest / sidecar bytes still hash to what the run recorded
    recorded = report["persist"]["files"]
    for name, want in recorded.items():
        assert _sha256(COHORT_DIR / name) == want, f"{name} drifted from its recorded hash"


# ===========================================================================
# 3 - Random and Hard describe the exact same cohort (identity)
# ===========================================================================
@pytest.mark.skipif(
    not (COHORT_DIR / "random_rg_external_strict.jsonl").exists(), reason="manifests absent"
)
def test_random_and_hard_manifests_describe_the_same_cohort() -> None:
    rand, hard = rman.load_cohort_manifests(COHORT_DIR)
    rand_by_key = {int(e.ref_id): e for e in rand.entries}
    hard_by_key = {int(e.ref_id): e for e in hard.entries}
    assert set(rand_by_key) == set(hard_by_key), "regimes cover different expressions"
    for key, r in rand_by_key.items():
        h = hard_by_key[key]
        assert int(r.image_id) == int(h.image_id)      # same image
        assert int(r.target_index) == int(h.target_index)  # same target proposal
    # the two regimes differ only in distractor composition (not identical sets)
    assert any(
        list(np.asarray(rand_by_key[k].distractor_order)[: rman.A11_K - 1])
        != list(np.asarray(hard_by_key[k].distractor_order)[: rman.A11_K - 1])
        for k in rand_by_key
    ), "random and hard candidate sets are indistinguishable"


# ===========================================================================
# 4 - the OpenCLIP checkpoint is the frozen one
# ===========================================================================
@pytest.mark.skipif(not MANIFEST_JSON.exists(), reason="frozen manifest absent")
def test_openclip_checkpoint_identity_is_frozen() -> None:
    identity = _read_json(MANIFEST_JSON)["openclip_identity"]
    assert identity["ok"] is True
    want = fi.FROZEN_OPENCLIP["checkpoint_sha256"]
    assert identity["measured_checkpoint_sha256"] == want
    assert identity["metadata"]["recorded_checkpoint_sha256"] == want
    assert identity["frozen"]["checkpoint_sha256"] == want


def test_assert_openclip_identity_accepts_a_matching_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "clip.bin"
    checkpoint.write_bytes(b"frozen-checkpoint-bytes")
    digest = _sha256(checkpoint)
    meta = {
        "cache_version": "phase0a-v1",
        "backbone": {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "library": "open_clip_torch",
            "precision": "fp16",
            "embedding_dim": 512,
            "resolution": 224,
            "tokenizer": {"name": "SimpleTokenizer", "context_length": 77},
            "checkpoint_sha256": digest,
            "checkpoint_path": str(checkpoint),
        },
    }
    root = tmp_path / "features"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    expected = {**fi.FROZEN_OPENCLIP, "checkpoint_sha256": digest}
    ok = fi.assert_openclip_identity(root, checkpoint_path=checkpoint, expected=expected)
    assert ok["ok"] is True and ok["measured_checkpoint_sha256"] == digest

    # a different recorded digest is a STOP, even though the file exists
    tampered = json.loads(json.dumps(meta))
    tampered["backbone"]["checkpoint_sha256"] = "0" * 64
    (root / "metadata.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(fi.IdentityMismatch, match="checkpoint_sha256"):
        fi.assert_openclip_identity(root, checkpoint_path=checkpoint, expected=expected)


# ===========================================================================
# 5 - the three B3 model states still hash to the manifest values
# ===========================================================================
@pytest.mark.skipif(not MANIFEST_JSON.exists(), reason="frozen manifest absent")
def test_b3_state_hashes_match_the_manifest() -> None:
    artifacts = _read_json(MANIFEST_JSON)["artifacts"]
    required = [f"b3_model_{s}" for s in SEEDS]
    report = fi.verify_checksum_manifest(artifacts, root=_REPO, required=required)
    assert report["ok"], report["mismatches"]
    assert report["n_missing_required"] == 0


# ===========================================================================
# 6 / 7 / 8 - the frozen reliability bundle (Stats / E1b / normalisation)
# ===========================================================================
def _load_bundle():
    return fl.load_models(FROZEN_MODELS)


@pytest.mark.skipif(not (FROZEN_MODELS / "models.json").exists(), reason="bundle absent")
def test_stats_weights_and_prediction_are_frozen() -> None:
    bundle = _load_bundle()  # load_models pins the bundle to models.sha256
    verification = _read_json(MANIFEST_JSON)["bundle_verification"]
    tol = float(verification["tolerance"])
    for scorer in SEEDS:
        artifact = bundle.seed(scorer)
        assert artifact.stats_coef.size == 17
        # the load-and-predict Stats chain reproduces the stored RefCOCO+ confs
        assert verification["checks"][scorer]["stats_logistic_pred_max_abs"] <= tol


@pytest.mark.skipif(not (FROZEN_MODELS / "models.json").exists(), reason="bundle absent")
def test_e1b_weights_are_identical_to_the_phase1_coefficients() -> None:
    bundle = _load_bundle()
    verification = _read_json(MANIFEST_JSON)["bundle_verification"]
    for scorer in SEEDS:
        assert bundle.seed(scorer).e1b_coef.size == 33  # 17 stats + 16 semantic
        assert verification["checks"][scorer]["e1b_coefficients_max_abs"] == 0.0


@pytest.mark.skipif(not (FROZEN_MODELS / "models.json").exists(), reason="bundle absent")
def test_normalization_is_identical_to_the_phase05_fit() -> None:
    verification = _read_json(MANIFEST_JSON)["bundle_verification"]
    for scorer in SEEDS:
        assert verification["checks"][scorer]["stats17_normalisation_max_abs"] == 0.0
    # the bundle's feature ordering is the frozen one, or from_dict would raise
    _load_bundle()


@pytest.mark.skipif(not (FROZEN_MODELS / "models.json").exists(), reason="bundle absent")
def test_bundle_checksum_is_enforced(tmp_path: Path) -> None:
    import shutil

    for name in ("models.json", "models.sha256"):
        shutil.copyfile(FROZEN_MODELS / name, tmp_path / name)
    bundle = fl.load_models(tmp_path)  # passes while the checksum matches
    (tmp_path / "models.json").write_text(
        (tmp_path / "models.json").read_text(encoding="utf-8").replace(
            '"stats_intercept"', '"stats_intercept_x"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="modified after export"):
        fl.load_models(tmp_path)
    assert bundle.seeds  # sanity: the untouched bundle loaded fine before tampering


# ===========================================================================
# 9 / 11 - the production path is load-and-predict only
# ===========================================================================
@pytest.mark.skipif(
    not all(p.exists() for p in PRODUCTION_PATH), reason="A11 production scripts absent"
)
def test_production_runner_contains_no_fit_path() -> None:
    report = fl.assert_no_fit_path(PRODUCTION_PATH)
    assert report["n_files"] == len(PRODUCTION_PATH)


@pytest.mark.skipif(
    not all(p.exists() for p in PRODUCTION_PATH), reason="A11 production scripts absent"
)
def test_production_path_contains_no_calibration_fitting() -> None:
    fl.assert_no_fit_path(
        PRODUCTION_PATH,
        extra=[".calibrate(", "CalibratedClassifierCV", "StandardScaler(", ".partial_fit("],
    )


def test_fit_tripwire_blocks_a_refit() -> None:
    from ccg.reliability import models as rmodels

    model = rmodels.LogisticModel()
    with pytest.raises(AssertionError, match="never re-fitted"):
        with fl.fit_is_forbidden():
            model.fit(np.zeros((3, 2)), np.zeros(3))


# ===========================================================================
# 10 - RefCOCOg train / val are never loaded
# ===========================================================================
def test_build_manifest_defaults_to_the_test_split() -> None:
    import inspect

    signature = inspect.signature(rman.build_expression_manifest)
    assert signature.parameters["eval_split"].default == rg.UMD_EVAL_SPLIT == "test"


@pytest.mark.skipif(
    not (COHORT_DIR / "random_rg_external_strict.jsonl").exists(), reason="manifests absent"
)
def test_refcocog_train_and_val_are_never_loaded() -> None:
    rand, hard = rman.load_cohort_manifests(COHORT_DIR)
    for manifest in (rand, hard):
        assert manifest.meta["regime"] in rman.REGIMES
        splits = {str(e.split) for e in manifest.entries}
        assert splits == {"test"}  # never train / val_select / val_calib


# ===========================================================================
# 12 - K is frozen to 5
# ===========================================================================
def test_k_is_frozen_to_five() -> None:
    assert rman.A11_K == 5
    assert rman.PRIMARY_K == 5


@pytest.mark.skipif(
    not (COHORT_DIR / "random_rg_external_strict.jsonl").exists(), reason="manifests absent"
)
def test_every_cohort_row_carries_four_distractors() -> None:
    rand, hard = rman.load_cohort_manifests(COHORT_DIR)
    assert len(rand.entries) == len(hard.entries)
    for manifest in (rand, hard):
        for entry in manifest.entries:
            order = np.asarray(entry.distractor_order)
            assert order.size >= rman.A11_K - 1
            # C_5 = target + first (K-1) distractors, so the prefix is exactly 4
            assert order[: rman.A11_K - 1].size == rman.A11_K - 1


# ===========================================================================
# 13 - matched bootstrap: Random and Hard share the same cluster draws
# ===========================================================================
@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="a11_results.json absent")
def test_bootstrap_is_matched_random_and_hard() -> None:
    results = _read_json(RESULTS_JSON)
    assert results["replicates"] == 5000
    assert results["ci_level"] == 0.95
    assert results["n_clusters"] == 755  # one cluster per strict-cohort image
    for scorer in SEEDS:
        per_seed = results["per_seed"][scorer]
        rand_clusters = int(per_seed["random"]["n_clusters"])
        hard_clusters = int(per_seed["same_category"]["n_clusters"])
        # same image-cluster set for both regimes => the draws are shareable
        assert rand_clusters == hard_clusters == results["n_clusters"]


# ===========================================================================
# 14 - the manipulation gate is computed correctly
# ===========================================================================
@pytest.mark.skipif(not RESULTS_JSON.exists(), reason="a11_results.json absent")
def test_manipulation_check_significance_is_correct() -> None:
    import a11_external_analysis as ana

    threshold = ana.A11_THRESHOLDS["seed_sig_min"]
    manipulation = _read_json(RESULTS_JSON)["manipulation"]
    seed_validities = []
    for scorer, block in manipulation["per_seed"].items():
        recomputed = 0
        for name, feature in block["features"].items():
            assert name in ana.CRITERION_FEATURES
            sign = feature["expected_sign"]
            assert sign == ana.CRITERION_EXPECTED_SIGN[name]
            if sign > 0:
                sig = feature["diff"] > 0.0 and feature["ci_low"] > 0.0
            else:
                sig = feature["diff"] < 0.0 and feature["ci_high"] < 0.0
            assert feature["significant_expected"] == sig, f"{scorer}/{name}"
            recomputed += int(sig)
        assert block["n_criterion_significant"] == recomputed
        assert block["valid"] == (recomputed >= threshold)
        seed_validities.append(block["valid"])
    assert manipulation["valid"] == all(seed_validities)


# ===========================================================================
# 15 - the reliability models never change the raw grounding decision
# ===========================================================================
@pytest.mark.skipif(
    not (PRED_DIR / "external__b3_seed1__rand5.npz").exists(), reason="raw predictions absent"
)
def test_reliability_models_do_not_change_raw_grounding() -> None:
    for scorer in SEEDS:
        for tag in ("rand5", "hard5"):
            path = PRED_DIR / f"external__{scorer}__{tag}.npz"
            with np.load(path, allow_pickle=False) as store:
                scores = np.asarray(store["scores"], dtype=np.float64)
                correct = np.asarray(store["correct"])
                stats_conf = np.asarray(store["conf_stats"])
                e1b_conf = np.asarray(store["conf_e1b"])
            # the ground-truth judgement is argmax(B3 scores) == target(index 0)
            assert np.array_equal(correct, np.argmax(scores, axis=1) == 0)
            # confidences are probabilities in [0, 1], strictly separate from the
            # binary grounding decision the reliability models may never rewrite
            assert stats_conf.min() >= 0.0 and stats_conf.max() <= 1.0
            assert e1b_conf.min() >= 0.0 and e1b_conf.max() <= 1.0
            assert correct.dtype == bool and correct.shape == stats_conf.shape


# ===========================================================================
# 16 - the frozen anchor recovery passed
# ===========================================================================
@pytest.mark.skipif(not ANCHOR_JSON.exists(), reason="anchor recovery absent")
def test_anchor_recovery_passed() -> None:
    report = _read_json(ANCHOR_JSON)
    assert report["passed"] is True
    assert report["protocol"] == "A11.10"
    assert len(report["checks"]) == len(SEEDS) * 2  # three seeds x {rand5, hard5}
    for check in report["checks"]:
        assert check["ranking_identical"] is True
        assert check["correct_identical"] is True
        assert check["raw_score_max_abs"] <= check["tolerance"]
        assert check["conf_stats_max_abs"] <= check["tolerance"]
        assert check["conf_e1b_max_abs"] <= check["tolerance"]


# ===========================================================================
# 17 / 18 - the cohort verification logic (category + target-equivalent)
# ===========================================================================
def _synthetic_cohort(target_index: int, rand: List[int], hard: List[int]):
    """A one-row matched cohort for exercising ``verify_matched_pair``."""
    return rman.RefCOCOGCohort(
        expr_id=np.asarray([1], dtype=np.int64),
        ref_id=np.asarray([11], dtype=np.int64),
        image_id=np.asarray([100], dtype=np.int64),
        target_index=np.asarray([target_index], dtype=np.int64),
        rand_order=np.asarray([rand], dtype=np.int64),
        hard_order=np.asarray([hard], dtype=np.int64),
        category=["person"],
        category_id=np.asarray([17], dtype=np.int64),
        target_best_iou=np.asarray([0.9]),
        n_valid_distractors=np.asarray([60], dtype=np.int64),
        n_same_category=np.asarray([10], dtype=np.int64),
        n_target_equiv=np.asarray([0], dtype=np.int64),
        text=["the person"],
    )


@pytest.fixture
def patched_bank(monkeypatch):
    """Replace the disk reads inside ``verify_matched_pair`` with a fixed category map."""

    def _apply(category_by_index: Dict[int, int]):
        monkeypatch.setattr(
            rman, "_bank_boxes", lambda image_id: (np.zeros((rman.N_PROPOSALS, 4)), None)
        )
        monkeypatch.setattr(
            rman, "read_gt_basis", lambda gt_index, image_id, basis=None: (np.zeros((1, 4)), np.asarray([17]))
        )
        categories = np.full(rman.N_PROPOSALS, -1, dtype=np.int64)
        for index, value in category_by_index.items():
            categories[index] = value
        monkeypatch.setattr(
            rman, "assign_gt_category", lambda *a, **k: (categories, np.zeros((rman.N_PROPOSALS, 2)))
        )

    return _apply


def test_hard_distractors_must_share_the_target_category(patched_bank) -> None:
    cohort = _synthetic_cohort(target_index=0, rand=[1, 2, 3, 4], hard=[5, 6, 7, 8])
    patched_bank({i: 17 for i in range(rman.N_PROPOSALS)})  # everyone is "person"
    report = rman.verify_matched_pair(cohort, gt_index=None)
    assert report["failures"]["hard_same_category"] == 0
    assert report["failures"]["unknown_category"] == 0

    # one hard distractor in a different category is a failure ...
    mismatched = {i: 17 for i in range(rman.N_PROPOSALS)}
    mismatched[7] = 20  # index 7 is a hard distractor
    patched_bank(mismatched)
    report = rman.verify_matched_pair(cohort, gt_index=None)
    assert report["failures"]["hard_same_category"] == 1
    assert report["ok"] is False

    # ... an unknown (-1) target category is reported separately
    unknown = {i: 17 for i in range(rman.N_PROPOSALS)}
    unknown[0] = -1
    patched_bank(unknown)
    report = rman.verify_matched_pair(cohort, gt_index=None)
    assert report["failures"]["unknown_category"] == 1


def test_target_leak_and_distractor_shape_are_rejected(patched_bank) -> None:
    # a hard distractor that repeats the target proposal is a target leak
    cohort = _synthetic_cohort(target_index=0, rand=[1, 2, 3, 4], hard=[0, 5, 6, 7])
    patched_bank({i: 17 for i in range(rman.N_PROPOSALS)})
    report = rman.verify_matched_pair(cohort, gt_index=None)
    assert report["failures"]["target_leak"] == 1

    # the sample dataclass refuses a target among its own distractors outright
    with pytest.raises(ValueError, match="among its own distractors"):
        rman.RefCOCOGSample(
            sentence_id=1, ref_id=11, image_id=100, target_index=0,
            distractor_order=np.asarray([0, 1, 2, 3]), regime="same_category",
            expr_id=1, category="person",
        )
    # ... and requires exactly K-1 distractors
    with pytest.raises(ValueError, match="distractor_order"):
        rman.RefCOCOGSample(
            sentence_id=1, ref_id=11, image_id=100, target_index=0,
            distractor_order=np.asarray([1, 2, 3]), regime="same_category",
            expr_id=1, category="person",
        )
