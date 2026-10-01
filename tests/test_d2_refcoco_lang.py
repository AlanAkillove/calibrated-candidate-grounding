"""V2-D2 RefCOCO language-distribution-transfer tests.

Covers the pieces that must not silently drift:

* :class:`~ccg.external.refcoco_lang.LangSample` / :class:`LangCohort` geometry
  (the ``DMAX - 1`` ordering, the target-never-a-distractor invariant, aligned
  ``samples_for`` slicing, ``subset`` round-tripping);
* the inference row-selection safety net - every row a job scores is inside the
  re-encoded union (``encoded_rows``), which is what fixed the missing-text bug;
* the two frozen gates replayed on synthetic inputs: the V2-G C1 cardinality
  route logic and the V2-G C4 hard gate (including the ``NOT ASSESSABLE`` branch,
  which must never be reported as a FAIL);
* the on-disk D2 artifacts, when present (skipped on a fresh clone).

Synthetic tests always run; artifact tests skip when the D2 result directory is
absent so the suite stays runnable without the large frozen inputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccg.external import refcoco_lang as L  # noqa: E402

ARTIFACTS = REPO / "results" / "v2_d2_refcoco_lang"
DMAX = L.DMAX
ORDER_LEN = DMAX - 1


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_d2_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d2_infer = _load_script("d2_frozen_inference")
d2_analysis = _load_script("run_d2_c1_c4")

needs_artifacts = pytest.mark.skipif(
    not (ARTIFACTS / "c1_c4_verdict.json").exists(),
    reason="D2 artifacts not present (run the D2 pipeline first)",
)


def _order(exclude: int, start: int = 0) -> np.ndarray:
    """A valid ``DMAX - 1`` distractor ordering that never contains ``exclude``."""
    vals = [v for v in range(start, start + ORDER_LEN + 1) if v != exclude][:ORDER_LEN]
    return np.asarray(vals, dtype=np.int64)


def _synthetic_cohort(n: int = 6) -> L.LangCohort:
    rand = np.stack([_order(exclude=0, start=1) for _ in range(n)])
    hard = np.stack([_order(exclude=0, start=1) for _ in range(n)])
    n_valid = np.asarray([DMAX - 1] * n, dtype=np.int64)
    n_same = np.asarray([4] * n, dtype=np.int64)
    return L.LangCohort(
        expr_id=np.arange(100, 100 + n, dtype=np.int64),
        ref_id=np.arange(200, 200 + n, dtype=np.int64),
        image_id=np.asarray([i % 2 for i in range(n)], dtype=np.int64),
        ann_id=np.arange(300, 300 + n, dtype=np.int64),
        target_index=np.zeros(n, dtype=np.int64),
        rand_order=rand,
        hard_order=hard,
        category=["cat"] * n,
        category_id=np.full(n, 7, dtype=np.int64),
        target_best_iou=np.full(n, 0.8),
        n_valid_distractors=n_valid,
        n_same_category=n_same,
        n_target_equiv=np.zeros(n, dtype=np.int64),
        text=[f"the cat on {i}" for i in range(n)],
        split=["testA"] * n,
    )


# ---------------------------------------------------------------------------
# LangSample / LangCohort geometry
# ---------------------------------------------------------------------------
def test_langsample_requires_full_ordering():
    with pytest.raises(ValueError):
        L.LangSample(
            sentence_id=1, ref_id=1, image_id=1, target_index=0,
            distractor_order=np.arange(1, 10, dtype=np.int64),  # far short of DMAX-1
            regime="random", expr_id=1, category="cat",
        )


def test_langsample_rejects_target_among_distractors():
    order = _order(exclude=0, start=1)
    order[0] = 0  # force the target into the distractor list
    with pytest.raises(ValueError):
        L.LangSample(
            sentence_id=1, ref_id=1, image_id=1, target_index=0,
            distractor_order=order, regime="random", expr_id=1, category="cat",
        )


def test_samples_for_is_aligned_and_sliced():
    cohort = _synthetic_cohort(6)
    rows = [0, 2, 4]
    samples = cohort.samples_for("random", rows)
    assert [s.expr_id for s in samples] == [int(cohort.expr_id[r]) for r in rows]
    for s, r in zip(samples, rows):
        assert s.distractor_order.size == ORDER_LEN
        assert int(cohort.target_index[r]) not in set(int(v) for v in s.distractor_order)


def test_subset_roundtrips_length_and_ids():
    cohort = _synthetic_cohort(6)
    sub = cohort.subset([1, 3, 5])
    assert len(sub) == 3
    assert list(sub.expr_id) == [int(cohort.expr_id[i]) for i in (1, 3, 5)]


def test_cohort_rejects_mismatched_matrix_width():
    cohort = _synthetic_cohort(4)
    bad = np.zeros((3, ORDER_LEN), dtype=np.int64)  # row count disagrees with expr_id
    with pytest.raises(ValueError):
        L.LangCohort(
            expr_id=cohort.expr_id, ref_id=cohort.ref_id, image_id=cohort.image_id,
            ann_id=cohort.ann_id, target_index=cohort.target_index,
            rand_order=bad, hard_order=cohort.hard_order,
            category=cohort.category, category_id=cohort.category_id,
            target_best_iou=cohort.target_best_iou,
            n_valid_distractors=cohort.n_valid_distractors,
            n_same_category=cohort.n_same_category, n_target_equiv=cohort.n_target_equiv,
            text=cohort.text, split=cohort.split,
        )


# ---------------------------------------------------------------------------
# row-selection safety net (the missing-text bug fix)
# ---------------------------------------------------------------------------
def test_job_rows_never_leave_the_encoded_union():
    cohort = _synthetic_cohort(6)
    enc = set(int(r) for r in d2_infer.encoded_rows(cohort))
    for regime in ("random", "same_category"):
        for k in (5, 10, 20, 50):
            for row in d2_infer.job_rows(cohort, regime, k):
                assert int(row) in enc


def test_job_rows_respect_distractor_supply():
    cohort = _synthetic_cohort(6)  # n_valid = DMAX-1 -> eligible for every random K
    assert d2_infer.job_rows(cohort, "random", 50).size == len(cohort)
    # a row short on same-category competitors must be dropped from the hard arm
    cohort2 = _synthetic_cohort(6)
    object.__setattr__(cohort2, "n_same_category",
                       np.asarray([4, 4, 1, 4, 0, 4], dtype=np.int64))
    hard_rows = set(int(r) for r in d2_infer.job_rows(cohort2, "same_category", 5))
    assert hard_rows == {0, 1, 3, 5}


# ---------------------------------------------------------------------------
# V2-G C1 cardinality gate (synthetic)
# ---------------------------------------------------------------------------
def _c1_stage(auc_diff: float, eaurc_mean_a: float, eaurc_mean_b: float,
              rer_diff: float, *, ci_sign: float) -> dict:
    seeds = {f"seed{s}": {
        "diff": auc_diff, "ci_low": ci_sign, "ci_high": ci_sign + 0.05,
        "mean_a": eaurc_mean_a, "mean_b": eaurc_mean_b} for s in (1, 2, 3)}
    stage = {("auroc_correct|absolute", d2_analysis.PRIMARY_KB): seeds}
    stage[("e_aurc|relative", d2_analysis.PRIMARY_KB)] = {
        f"seed{s}": {"diff": 0.0, "ci_low": -0.1, "ci_high": -0.02,
                     "mean_a": eaurc_mean_a, "mean_b": eaurc_mean_b} for s in (1, 2, 3)}
    stage[("rer_at_50|absolute", d2_analysis.PRIMARY_KB)] = {
        f"seed{s}": {"diff": rer_diff, "ci_low": ci_sign, "ci_high": ci_sign + 0.05,
                     "mean_a": 0.0, "mean_b": 0.0} for s in (1, 2, 3)}
    return stage


def test_c1_gate_route_a_passes_on_strong_degradation():
    stage = _c1_stage(auc_diff=0.05, eaurc_mean_a=0.30, eaurc_mean_b=0.40,
                      rer_diff=0.02, ci_sign=0.01)
    gate = d2_analysis.c1_gate(stage)
    assert gate["route_a_passed"] is True
    assert gate["replicated"] is True


def test_c1_gate_route_a_fails_when_ci_touches_zero():
    stage = _c1_stage(auc_diff=0.05, eaurc_mean_a=0.30, eaurc_mean_b=0.40,
                      rer_diff=0.02, ci_sign=-0.01)
    # ci_low <= 0 for every seed -> no seed "excludes 0 positive"
    stage[("auroc_correct|absolute", d2_analysis.PRIMARY_KB)] = {
        f"seed{s}": {"diff": 0.05, "ci_low": -0.01, "ci_high": 0.09,
                     "mean_a": 0.30, "mean_b": 0.40} for s in (1, 2, 3)}
    gate = d2_analysis.c1_gate(stage)
    assert gate["route_a_passed"] is False


def test_worsening_from_rel_sign():
    # e_aurc rises from K5 (a=0.30) to K50 (b=0.40): worsening (baseline K5) > 0
    w, lo, hi = d2_analysis._worsening_from_rel(0.30, 0.40, -0.4, -0.2)
    assert w > 0.0 and lo <= hi


# ---------------------------------------------------------------------------
# V2-G C4 hard gate (synthetic)
# ---------------------------------------------------------------------------
def _c4_per_seed(delta_hard: float, delta_hard_ci: float,
                 amp: float, amp_ci: float) -> dict:
    return {
        "b3_seed1": {"delta_hard": delta_hard, "delta_hard_ci_low": delta_hard_ci,
                     "delta_rand": 0.0, "amplification": amp, "amplification_ci_low": amp_ci},
        "b3_seed2": {"delta_hard": delta_hard, "delta_hard_ci_low": delta_hard_ci,
                     "delta_rand": 0.001, "amplification": amp, "amplification_ci_low": amp_ci},
        "b3_seed3": {"delta_hard": delta_hard, "delta_hard_ci_low": delta_hard_ci,
                     "delta_rand": -0.001, "amplification": amp, "amplification_ci_low": amp_ci},
    }


def test_c4_gate_replicated():
    gate = d2_analysis.d2_hard_gate(_c4_per_seed(0.03, 0.02, 0.03, 0.02), True)
    assert gate["verdict"] == "HARD_SEMANTIC_REPLICATED"
    assert gate["pass_delta_hard"] and gate["pass_amplification"]


def test_c4_gate_invalid_manipulation_is_not_assessable_not_fail():
    gate = d2_analysis.d2_hard_gate(_c4_per_seed(0.03, 0.02, 0.03, 0.02), False)
    assert gate["verdict"] == "NOT_ASSESSABLE"
    assert gate["manipulation_ok"] is False


def test_c4_gate_below_threshold_is_not_replicated():
    gate = d2_analysis.d2_hard_gate(_c4_per_seed(0.005, 0.001, 0.002, 0.001), True)
    assert gate["verdict"] == "NOT_REPLICATED"


def test_overall_verdict_labels_not_assessable_branch():
    v = d2_analysis.overall_verdict({"replicated": True}, {"verdict": "NOT_ASSESSABLE"})
    assert v["overall"] == "CARDINALITY ROBUST; HARD AMPLIFICATION NOT ASSESSABLE"
    assert "NOT cross-visual-domain" in v["boundary"]


# ---------------------------------------------------------------------------
# artifact integration (skipped on a fresh clone)
# ---------------------------------------------------------------------------
@needs_artifacts
def test_verdict_json_is_coherent():
    import json

    payload = json.loads((ARTIFACTS / "c1_c4_verdict.json").read_text(encoding="utf-8"))
    verdict = payload["verdict"]
    assert payload["models"]["new_parameters"] == 0
    assert verdict["c1_cross_dataset"] in (
        "C1 CROSS-DATASET REPLICATED", "C1 NOT REPLICATED")
    assert verdict["c4_cross_dataset"].startswith(("C4 CROSS-DATASET", "C4 NOT"))
    # the boundary statement must never claim cross-visual-domain generalisation
    assert "NOT cross-visual-domain generalisation" in verdict["boundary"]


@needs_artifacts
def test_environment_metadata_recorded():
    import json

    env = json.loads((ARTIFACTS / "c1_c4_verdict.json").read_text(encoding="utf-8"))["environment"]
    for key in ("python", "torch", "numpy", "sklearn", "transformers"):
        assert key in env and env[key]


@needs_artifacts
def test_c1_predictions_are_matched_across_k_on_common_cohort():
    # the K5..K50 sentence sets must be identical (paired design) on the common cohort
    pred_dir = ARTIFACTS / "predictions"
    base = np.load(pred_dir / "d2__random_k5__b3_seed1.npz")["sentence_id"]
    k50 = np.load(pred_dir / "d2__random_k50__b3_seed1.npz")["sentence_id"]
    common = set(int(s) for s in k50)
    assert common.issubset(set(int(s) for s in base))
