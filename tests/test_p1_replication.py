"""V2-P1 P1-F4 / P1-F5 proposal-family replication invariants (protocol section 24).

Offline-decidable checks always run: the P1-A0 candidate-ordering freeze, the
simultaneous F4+F5 config freeze, nested-K cohort identity, the frozen gate logic
(G3 route A/B, G4 + manipulation -> NOT_ASSESSABLE-first), the section-22 verdict
matrix, the deterministic proposal-family intersection, the frozen-stack hash /
temperature invariants, the shared-bootstrap determinism, and the static zero-fit
scan of the shipped drivers.  The row-identity checks read the P1 prediction npz
files and self-skip while inference is in flight (``results/v2_proposal_robustness/
predictions/*.npz``).  No test here fits anything or touches the GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ccg.data.manifests import ManifestFile
from ccg.external import frozen_identity as fi
from ccg.semantic import frozen_load as fl
from ccg.semantic import hard_eval as heval
from ccg.v2 import semantic_features as v2feat

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import p1_replication_core as core  # noqa: E402

CONFIG_FREEZE = _ROOT / "results/v2_proposal_robustness/p1_f4_f5_config_freeze.json"
FROZEN_MANIFEST = _ROOT / "results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json"
F3_IDENTITY = _ROOT / "results/v2_p1_f3_route_f/frozen_identity_check.json"
PRED_DIR = _ROOT / "results/v2_proposal_robustness/predictions"

CONFIG = json.loads(CONFIG_FREEZE.read_text(encoding="utf-8")) if CONFIG_FREEZE.exists() else {}
HAS_CONFIG = bool(CONFIG)
HAS_F3_IDENTITY = F3_IDENTITY.exists()


class _Rec:
    """Minimal SentenceRecord duck-type for the pure cohort helpers."""

    def __init__(self, sid, ref_id=0, image_id=0, target_index=0, order_size=64,
                 eligible=None):
        self.sentence_id = sid
        self.ref_id = ref_id
        self.image_id = image_id
        self.target_index = target_index
        self.distractor_order = np.arange(order_size, dtype=np.int64)
        self.eligible = eligible


def _pred_exists(family: str, job: str, scorer: str) -> bool:
    return core.pred_path(PRED_DIR, family, job, scorer).exists()


def _all_ids(family: str, job: str, scorer: str):
    """Every sentence_id stored in one prediction npz (the full cohort)."""
    with np.load(core.pred_path(PRED_DIR, family, job, scorer), allow_pickle=False) as z:
        return {int(i) for i in z["sentence_id"]}


HAS_PREDS = all(
    _pred_exists(f, j, f"b3_seed{s}")
    for f in ("RPN", "DETR")
    for j in ("random_k5", "random_k10", "random_k20", "random_k50", "hard_k5")
    for s in (1, 2, 3)
)
NEEDS_PREDS = pytest.mark.skipif(not HAS_PREDS, reason="P1 predictions not written yet")
NEEDS_CONFIG = pytest.mark.skipif(not HAS_CONFIG, reason="config freeze not present")


# ===========================================================================
# P1-A0 + simultaneous config freeze
# ===========================================================================
@NEEDS_CONFIG
def test_p1_a0_candidate_ordering_frozen():
    ordering = CONFIG["candidate_ordering"]
    assert ordering["amendment"] == "P1-A0"
    assert int(ordering["manifest_seed"]) == 20260927 == core.MANIFEST_SEED
    rule = ordering["primary_random_rule"].lower()
    assert "seeded-random" in rule and "slot 0" in rule and "nested" in rule
    assert "provenance" in ordering["detr_confidence_ordering_role"]
    assert any("detr confidence top-k" in f.lower() for f in CONFIG["forbidden"])


@NEEDS_CONFIG
def test_f4_f5_configuration_frozen_simultaneously():
    assert CONFIG["frozen_before_results"] is True
    assert CONFIG["classification"] == "CONFIRMATORY"
    assert "p1_f4_c1" in CONFIG and "p1_f5_c4" in CONFIG
    shared = CONFIG["shared_bootstrap"]
    assert (shared["replicates"], shared["seed"], shared["ci"]) == (5000, 0, 0.95)
    assert shared["resample_unit"] == "image" and shared["paired"] is True
    assert (core.BOOTSTRAP_REPLICATES, core.BOOTSTRAP_SEED, core.BOOTSTRAP_CI) == (5000, 0, 0.95)
    assert CONFIG["p1_f4_c1"]["statistics"]["replicates"] == 5000
    assert CONFIG["p1_f5_c4"]["manipulation_check"]["statistics"]["replicates"] == 5000


# ===========================================================================
# nested-K cohort identity (pure)
# ===========================================================================
def test_nested_k_eligibility_is_monotone_prefix():
    # a record eligible at K50 is eligible at every smaller K (prefix property)
    full = _Rec(1, order_size=64)
    for k in core.KS:
        assert core.is_eligible(full, k)
    mid = _Rec(2, order_size=15)
    assert core.is_eligible(mid, 5) and core.is_eligible(mid, 10)
    assert not core.is_eligible(mid, 20) and not core.is_eligible(mid, 50)
    none = _Rec(3, target_index=None)
    assert not core.is_eligible(none, 5)


def test_nested_cohort_ids_subset_chain():
    recs = [_Rec(i, order_size=s) for i, s in
            enumerate([64, 64, 20, 20, 8, 8, 4, 0], start=1)]
    sets = [core.nested_cohort_ids(recs, k) for k in core.KS]
    for a, b in zip(core.KS[:-1], core.KS[1:]):
        assert sets[core.KS.index(b)] <= sets[core.KS.index(a)]  # C_{k+} subset C_{k-}
    # eligibility needs >= K-1 distractors: size-8 records pass K5 only, size-4 too
    assert sets[0] == {1, 2, 3, 4, 5, 6, 7}
    assert sets[1] == {1, 2, 3, 4}
    assert sets[2] == {1, 2, 3, 4}
    assert sets[3] == {1, 2}


# ===========================================================================
# C1 gate (pure)
# ===========================================================================
def _mk_stage(auc=(0.04, 0.01), eaurc=(-0.3, -0.2, 0.5, 0.9), rer=(0.2, 0.1),
              rer80=None):
    stage = {
        ("auroc_correct|absolute", 50): {
            f"seed{s}": {"diff": auc[0], "ci_low": auc[1], "ci_high": 0.9,
                         "mean_a": 0.8, "mean_b": 0.76} for s in core.B3_SEEDS},
        ("e_aurc|relative", 50): {
            f"seed{s}": {"diff": eaurc[0], "ci_low": eaurc[1], "ci_high": eaurc[2],
                         "mean_a": eaurc[3], "mean_b": eaurc[4] if len(eaurc) > 4 else 1.0}
            for s in core.B3_SEEDS},
        ("rer_at_50|absolute", 50): {
            f"seed{s}": {"diff": rer[0], "ci_low": rer[1], "ci_high": 0.9,
                         "mean_a": 0.8, "mean_b": 0.6} for s in core.B3_SEEDS},
    }
    if rer80 is not None:
        stage[("rer_at_80|absolute", 50)] = {
            f"seed{s}": {"diff": rer80[0], "ci_low": rer80[1], "ci_high": 0.9,
                         "mean_a": 0.5, "mean_b": 0.3} for s in core.B3_SEEDS}
    return stage


def test_c1_gate_route_a():
    gate = core.c1_gate(_mk_stage(auc=(0.04, 0.01), eaurc=(-0.05, -0.1, 0.0, 0.5, 0.55),
                                  rer=(0.02, -0.01)))
    assert gate["route_a_passed"] and not gate["route_b_passed"]
    assert gate["replicated"]


def test_c1_gate_route_b():
    gate = core.c1_gate(_mk_stage(auc=(0.01, -0.005), eaurc=(-0.30, -0.4, -0.2, 0.5, 0.72),
                                  rer=(0.15, 0.1)))
    assert not gate["route_a_passed"] and gate["route_b_passed"]
    assert gate["replicated"]


def test_c1_gate_neither_route():
    gate = core.c1_gate(_mk_stage(auc=(0.01, -0.01), eaurc=(-0.05, -0.1, 0.02, 0.5, 0.55),
                                  rer=(0.02, -0.01), rer80=(0.01, -0.02)))
    assert not gate["route_a_passed"] and not gate["route_b_passed"]
    assert not gate["replicated"]
    assert gate["rer80_drop_mean"] == pytest.approx(0.01)


def test_c1_gate_rer80_optional_key_absent():
    gate = core.c1_gate(_mk_stage())
    assert np.isnan(gate["rer80_drop_mean"])


def test_worsening_from_rel_mapping():
    # r = (a-b)/b with a=K5 baseline, b=K50; worsening w = (b-a)/a
    # t(x) = -x/(1+x) is monotone decreasing, so the CI endpoints are re-sorted.
    # here r = (2-3)/3 = -1/3 -> w = 0.5; ratio CI (-0.5, -0.2) straddles r.
    w, lo, hi = core._worsening_from_rel(2.0, 3.0, -0.5, -0.2)
    assert w == pytest.approx(0.5)           # (3-2)/2
    assert lo == pytest.approx(0.25)         # t(-0.2)
    assert hi == pytest.approx(1.0)          # t(-0.5)
    assert lo <= w <= hi


# ===========================================================================
# C4 manipulation + gate (pure)
# ===========================================================================
def test_manipulation_valid_rules_frozen():
    assert core.manipulation_valid_seed(2, 1) is True
    assert core.manipulation_valid_seed(3, 0) is False   # needs >=1 CI excluding 0
    assert core.manipulation_valid_seed(1, 1) is False   # needs >=2/3 directions
    assert core.manipulation_aggregate_ok(2, 3) is True
    assert core.manipulation_aggregate_ok(1, 3) is False
    cfg = CONFIG.get("p1_f5_c4", {}).get("manipulation_check", {}).get("valid_rule", {})
    if cfg:
        assert "2/3" in cfg["per_seed"] and "2/3" in cfg["aggregate"]


@NEEDS_CONFIG
def test_manipulation_feature_definitions_frozen():
    expected = (("winner_competitor_max_cos", "up"), ("winner_top2_cos", "up"),
                ("q_margin12", "down"))
    assert tuple(core.MANIPULATION) == expected
    assert tuple(v2feat.MANIPULATION_METRICS) == expected
    dirs = {f["name"]: f["direction"] for f in CONFIG["p1_f5_c4"]["manipulation_check"]["features"]}
    assert tuple(dirs.items()) == expected
    assert set(core.V2_SEM_INDEX) >= {n for n, _ in expected}
    assert all(core.V2_SEM_INDEX[n] < len(v2feat.V2_PRIMARY_SEMANTIC_NAMES)
               for n, _ in expected)


def _ps(dh=0.03, dh_lo=0.02, amp=0.02, amp_lo=0.01, dr=0.001):
    return {f"seed{s}": {"delta_hard": dh, "delta_hard_ci_low": dh_lo,
                          "delta_rand": dr, "amplification": amp,
                          "amplification_ci_low": amp_lo} for s in core.B3_SEEDS}


def test_c4_gate_not_assessable_is_never_fail():
    gate = core.c4_hard_gate(_ps(), manipulation_ok=False)
    assert gate["verdict"] == "NOT_ASSESSABLE"
    assert core.c4_verdict_label(gate) == "NOT_ASSESSABLE"
    assert "FAIL" not in core.c4_verdict_label(gate)


def test_c4_gate_replicated_and_not_replicated():
    ok = core.c4_hard_gate(_ps(0.02, 0.01, 0.015, 0.005), manipulation_ok=True)
    assert ok["verdict"] == "HARD_SEMANTIC_REPLICATED"
    assert core.c4_verdict_label(ok) == "YES"
    no = core.c4_hard_gate(_ps(dh=0.005), manipulation_ok=True)
    assert no["verdict"] == "NOT_REPLICATED" and core.c4_verdict_label(no) == "NO"
    no2 = core.c4_hard_gate(_ps(0.03, 0.02, 0.002, -0.001), manipulation_ok=True)
    assert no2["verdict"] == "NOT_REPLICATED"


# ===========================================================================
# section 22 overall verdict matrix (pure)
# ===========================================================================
def test_overall_p1_verdict_matrix():
    a = core.overall_p1_verdict(True, "YES")
    assert a["label"] == "CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST" and a["value"] == "YES"
    b = core.overall_p1_verdict(True, "NOT_ASSESSABLE")
    assert "CARDINALITY_PROPOSAL_ROBUST" in b["label"] and "NOT_ASSESSABLE" in b["label"]
    assert core.overall_p1_verdict(True, "NO")["label"] == "PARTIAL_PROPOSAL_ROBUSTNESS"
    assert core.overall_p1_verdict(False, "YES")["label"] == "PARTIAL_PROPOSAL_ROBUSTNESS"
    w = core.overall_p1_verdict(False, "NO")
    assert w["label"] == "PROPOSAL_FAMILY_SENSITIVITY_WARNING" and w["value"] == "NO"


# ===========================================================================
# shared bootstrap draws (determinism)
# ===========================================================================
def test_shared_bootstrap_draws_are_deterministic():
    rng = np.random.default_rng(7)
    n = 240
    clusters = rng.integers(0, 24, size=n)
    a, b = rng.normal(size=n), rng.normal(size=n)
    r1 = heval.paired_shift_bootstrap(a, b, clusters, n_replicates=500, seed=0, ci=0.95,
                                      name="x")
    r2 = heval.paired_shift_bootstrap(a, b, clusters, n_replicates=500, seed=0, ci=0.95,
                                      name="x")
    assert r1["diff"] == r2["diff"] and r1["ci_low"] == r2["ci_low"]
    # the drivers pass the SAME cluster array + seed for hard and random (shared draws)
    assert core.BOOTSTRAP_SEED == CONFIG.get("shared_bootstrap", {}).get("seed", 0)


# ===========================================================================
# frozen stack: hashes / temperature unchanged
# ===========================================================================
def test_frozen_r1_e1b_hashes_unchanged():
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]
    keys = [k for k in arts
            if k.startswith(("e1b_", "stats_", "frozen_loader_code", "stat_feature_code",
                             "semantic_feature_code", "b3_model_", "temperature_"))]
    assert keys, "manifest carries no frozen-head artifacts"
    for key in keys:
        rec = arts[key]
        assert fi.sha256_file(_ROOT / rec["path"]) == rec["sha256"], f"{key} changed"


def test_per_seed_temperature_unchanged():
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    for seed in core.B3_SEEDS:
        meta = _ROOT / "results/phase0b_independent" / f"seed_{seed}" / "eval_metadata.json"
        recorded = manifest["artifacts"][f"temperature_b3_seed{seed}"]
        assert fi.sha256_file(meta) == recorded["sha256"]
        temperature = json.loads(meta.read_text(encoding="utf-8"))["temperature_corrected"]
        assert float(temperature) > 0.0
        if HAS_F3_IDENTITY:
            f3 = json.loads(F3_IDENTITY.read_text(encoding="utf-8"))
            tc = f3["temperatures"][f"b3_seed{seed}"]
            assert tc["unchanged"] is True
            assert tc["bundle_temperature"] == pytest.approx(float(temperature), abs=1e-12)


@pytest.mark.skipif(not HAS_F3_IDENTITY, reason="F3 frozen-identity artifact not present")
def test_b3_checkpoint_hashes_unchanged_since_f3():
    f3 = json.loads(F3_IDENTITY.read_text(encoding="utf-8"))
    for scorer, row in f3["b3_checkpoints"].items():
        assert row["unchanged"] is True
        current = fi.sha256_file(_ROOT / f"results/phase0b_independent/seed_{scorer[-1]}/model.npz")
        assert current == row["recorded_sha256"]


def test_no_fit_path_in_shipped_p1_code():
    fl.assert_no_fit_path([
        _ROOT / "scripts/p1_replication_core.py",
        _ROOT / "scripts/p1_frozen_inference.py",
        _ROOT / "scripts/p1_f4_cardinality.py",
        _ROOT / "scripts/p1_f5_hard_semantic.py",
    ])


# ===========================================================================
# frozen RPN references loaded from artifacts (never hardcoded)
# ===========================================================================
def test_rpn_c1_reference_loaded_from_artifacts():
    ref = core.load_rpn_c1_reference()
    assert "auroc_correct|absolute" in ref and "e_aurc|relative" in ref
    assert ref["auroc_correct|absolute"]["diff_mean"] > 0.0
    assert "5" in ref["point_by_k"] and "50" in ref["point_by_k"]
    # independent re-parse of the frozen bootstrap.csv must reproduce the loader
    vals = []
    for seed in core.B3_SEEDS:
        path = _ROOT / core.V1_PHASE0B_BOOTSTRAP.format(seed=seed)
        with path.open(encoding="utf-8", newline="") as fh:
            for row in __import__("csv").DictReader(fh):
                if (row["eval_split"] == core.POOLED
                        and row["variant"] == "global_T_corrected"
                        and int(row["K_a"]) == core.K_BASELINE
                        and int(row["K_b"]) == core.PRIMARY_KB
                        and row["metric"] == "auroc_correct"):
                    vals.append(float(row["diff"]))
    assert ref["auroc_correct|absolute"]["diff_mean"] == pytest.approx(float(np.mean(vals)))
    assert len(vals) == 3


def test_rpn_c4_reference_loaded_from_gate_json():
    ref = core.load_rpn_c4_reference()
    gate = json.loads((_ROOT / "results/phase1f_hard_semantic/gate.json").read_text(encoding="utf-8"))
    assert ref["amplification"]["diff"] == pytest.approx(gate["diff_of_diffs_auroc"]["diff"])
    assert ref["delta_hard"] == pytest.approx(gate["cell_samecat_k5"]["delta_auroc"])
    assert ref["delta_rand"] == pytest.approx(
        gate["cell_samecat_k5"]["delta_auroc"] - gate["diff_of_diffs_auroc"]["diff"])


# ===========================================================================
# proposal-family intersection (deterministic) + manifest target identity
# ===========================================================================
def test_proposal_family_intersection_deterministic():
    a = {5, 3, np.int64(9), 1}
    b = {9, 2, np.int64(5), 3}
    i1 = core.matched_expression_ids(a, b)
    i2 = core.matched_expression_ids(b, a)
    assert i1 == i2 == {3, 5, 9}
    assert all(isinstance(v, int) for v in i1)
    assert i1 <= {int(x) for x in a} and i1 <= {int(x) for x in b}
    assert core.matched_expression_ids(set(), a) == set()


@pytest.mark.parametrize("family", ["RPN", "DETR"])
def test_same_target_proposal_between_random_and_hard_manifests(family):
    root = core.FAMILIES[family]["manifests_root"]
    rand = {int(e.ref_id): e for e in
            ManifestFile.load(core._resolve_manifest_path(root, "random", "testA")).entries}
    hard = {int(e.ref_id): e for e in
            ManifestFile.load(core._resolve_manifest_path(root, "same_category", "testA")).entries}
    common = set(rand) & set(hard)
    assert len(common) > 100
    assert all(rand[r].target_index == hard[r].target_index for r in common)


@pytest.mark.parametrize("family", ["RPN", "DETR"])
def test_same_category_assignment_identical_to_v1_rule(family):
    root = core.FAMILIES[family]["manifests_root"]
    path = core._resolve_manifest_path(root, "same_category", "testA")
    manifest = ManifestFile.load(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["regime"] == "same_category"
    assert float(meta["iou_thresh"]) == 0.5          # target matching threshold unchanged
    assert int(meta["seed"]) == core.MANIFEST_SEED   # seeded construction unchanged
    checked = 0
    for e in manifest.entries:
        hf = e.hard_fraction_by_K
        frac = hf.get(5, hf.get("5")) if isinstance(hf, dict) else None
        if int(e.n_same_category_available) >= core.HARD_SAME_CUTOFF and frac is not None:
            assert float(frac) == pytest.approx(1.0)  # K5 = target + 4 same-category (A8.3)
            checked += 1
    assert checked > 100


# ===========================================================================
# prediction-row identity contracts (need the P1 npz files)
# ===========================================================================
@NEEDS_PREDS
@pytest.mark.parametrize("family", ["RPN", "DETR"])
def test_common_k50_rows_identical_across_nested_k(family):
    scorer = "b3_seed1"
    keep = _all_ids(family, "random_k50", scorer)
    assert len(keep) > 1000
    base50 = core.load_pred(PRED_DIR, family, "random_k50", scorer, keep)
    slices = {k: core.load_pred(PRED_DIR, family, f"random_k{k}", scorer, keep)
              for k in core.KS}
    for k in core.KS:
        assert np.array_equal(slices[k]["sentence_id"], slices[50]["sentence_id"])
        assert np.array_equal(slices[k]["ref_id"], slices[50]["ref_id"])
        assert np.array_equal(slices[k]["image_id"], slices[50]["image_id"])
    # nested candidate sets are strict prefixes: identical rows, K-only difference
    assert slices[5]["conf_msp"].shape == slices[50]["conf_msp"].shape


@NEEDS_PREDS
def test_rows_identical_across_seeds():
    for family, job in (("RPN", "random_k5"), ("DETR", "random_k50"), ("DETR", "hard_k5")):
        ids = _all_ids(family, job, "b3_seed1")
        sids = [core.load_pred(PRED_DIR, family, job, f"b3_seed{s}", ids)["sentence_id"]
                for s in core.B3_SEEDS]
        assert np.array_equal(sids[0], sids[1]) and np.array_equal(sids[1], sids[2])


@NEEDS_PREDS
@pytest.mark.parametrize("family", ["RPN", "DETR"])
def test_hard_random_row_identity(family):
    for s in core.B3_SEEDS:
        scorer = f"b3_seed{s}"
        keep = _all_ids(family, "hard_k5", scorer)
        hard_all = core.load_pred(PRED_DIR, family, "hard_k5", scorer, keep)
        rand = core.load_pred(PRED_DIR, family, "random_k5", scorer, keep)
        assert np.array_equal(hard_all["sentence_id"], rand["sentence_id"])
        assert np.array_equal(hard_all["ref_id"], rand["ref_id"])
        assert np.array_equal(hard_all["image_id"], rand["image_id"])


def test_pred_column_contract():
    for col in ("sentence_id", "correct", "conf_msp", "conf_stats", "conf_e1b",
                "sem14", "raw_top1", "raw_margin12"):
        assert col in core.PRED_COLUMNS
    assert core.job_name("random", 10) == "random_k10"
    assert core.job_name("same_category", 5) == "hard_k5"
    p = core.pred_path(Path("x"), "DETR", "hard_k5", "b3_seed1")
    assert p.name == "p1__DETR__hard_k5__b3_seed1.npz"
