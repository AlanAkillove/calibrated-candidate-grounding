"""V2-P2-C1 result-layer discipline tests.

These assert the *outcome-side* invariants of the C1_GDINO run: the freeze scope
was actually honoured on disk, the frozen RPN/DETR numbers did not drift, and the
stop condition was met.  They read committed artifacts only, so they run on a
fresh clone; the two checks that would need the gitignored caches/predictions are
skipped when those are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_VERDICT = _ROOT / "results" / "v2_proposal_robustness" / "p2_c1_gdino" / "p2_c1_verdict.json"
_PREDICTIONS = _ROOT / "results" / "v2_proposal_robustness" / "predictions"
_GDINO_MANIFESTS = _ROOT / "cache" / "gdino_stage" / "manifests"

MIN_COHORT_ROWS = 8000  # frozen stop condition, p2_c1_gdino_config_freeze.json


@pytest.fixture(scope="module")
def verdict():
    if not _VERDICT.exists():
        pytest.skip("C1_GDINO verdict not produced yet")
    return json.loads(_VERDICT.read_text(encoding="utf-8"))


def test_verdict_is_confirmatory_and_zero_fit(verdict):
    assert verdict["classification"] == "CONFIRMATORY"
    assert verdict["new_training_parameters"] == 0
    assert verdict["protocol"].startswith("V2-P2-C1")
    assert verdict["config_freeze"].replace("/", "\\").endswith(
        "p2_c1_gdino_config_freeze.json")


def test_all_three_families_replicated(verdict):
    assert verdict["gdino_C1_PROPOSAL_FAMILY_REPLICATED"] == "YES"
    assert verdict["c1_verdicts"] == {"RPN": "YES", "DETR": "YES", "GDINO": "YES"}
    for family, row in verdict["c1_by_family"].items():
        gate = row["gate"]
        assert gate["route_a_passed"] and gate["route_b_passed"], family
        assert gate["auc_all_seeds_ci_exclude_0"], family


def test_gdino_cohort_clears_the_frozen_stop_condition(verdict):
    rows = verdict["c1_by_family"]["GDINO"]["cohort_rows"]
    assert rows >= MIN_COHORT_ROWS, f"{rows} < frozen {MIN_COHORT_ROWS}: must be reported underpowered"


def test_frozen_rpn_detr_rows_did_not_drift(verdict):
    assert verdict["rpn_detr_reuse_verbatim_check"].startswith("PASSED")


def test_no_gdino_c4_artifact_in_predictions(verdict):
    if not _PREDICTIONS.exists():
        pytest.skip("prediction files are gitignored; run the frozen inference to produce them")
    offenders = sorted(
        p.name for p in _PREDICTIONS.glob("p1__GDINO__*.npz")
        if "same_category" in p.name or "hard" in p.name
    )
    assert offenders == [], f"GDINO C4/hard predictions exist, violating the C1-only freeze: {offenders}"
    # and the C1 jobs that must exist are all there
    assert len(list(_PREDICTIONS.glob("p1__GDINO__random_k*__b3_seed*.npz"))) == 12


def test_no_gdino_same_category_manifest_on_disk():
    if not _GDINO_MANIFESTS.exists():
        pytest.skip("GDINO manifests cache is gitignored")
    offenders = sorted(p.name for p in _GDINO_MANIFESTS.glob("same_category*"))
    assert offenders == [], f"GDINO same_category manifests exist: {offenders}"
