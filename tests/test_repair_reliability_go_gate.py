from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccg.repairs.reliability_go_gate import build_reliability_go_decisions


_A_JOB = "phase0a___all_common__"
_B_JOB = "phase0b___all_common__"
_NAMES = {
    "e": "crossK_relative_eaurc_worsening__global_T",
    "r": "crossK_K5_minus_K50__global_T::rer_at_50",
    "a": "crossK_K5_minus_K50__global_T::auroc_correct",
}


def _estimate(name: str, point: float, ci_low: float, ci_high: float, seeds: list[str]) -> dict:
    per_seed = {
        seed: {
            "point": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_replicates": 5000,
            "valid_replicates": 5000,
            "invalid_replicates": 0,
            "gate_eligible": True,
        }
        for seed in seeds
    }
    return {
        "name": name,
        "scope": "test_scope",
        "cohort": "__all_common__",
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "valid_replicates": 5000,
        "invalid_replicates": 0,
        "n_replicates": 5000,
        "gate_eligible": True,
        "resample_unit": "image_cluster",
        "ci_level": 0.95,
        "n_rows": 20_799,
        "n_images": 2_981,
        "per_seed": per_seed,
        "raw_replicates": "raw/paired.npz",
        "source_artifacts": [{"path": "old/predictions.npz", "sha256": "fixture"}],
    }


def _job(job_id: str, seeds: list[str], *, e: tuple[float, float, float] = (0.25, 0.10, 0.35),
         r: tuple[float, float, float] = (0.12, 0.03, 0.18),
         a: tuple[float, float, float] = (0.04, 0.01, 0.07)) -> dict:
    values = {"e": e, "r": r, "a": a}
    estimates = [
        _estimate(_NAMES[key], *effect, seeds)
        for key, effect in values.items()
    ]
    estimates.extend([
        _estimate("K5__global_T::rer_at_80", 0.20, 0.18, 0.22, seeds),
        _estimate("K50__global_T::rer_at_80", 0.05, 0.04, 0.06, seeds),
    ])
    return {
        "job_id": job_id,
        "scope": "legacy_phase0_scope",
        "cohort": "__all_common__",
        "legacy_cohort_classification": "ALL_COMMON_INCLUDES_VALIDATION_NO_TRAIN",
        "status": "RECOMPUTED_5000",
        "config": {"n_replicates": 5000, "seed": 0, "ci_level": 0.95, "ci_method": "percentile"},
        "estimates": estimates,
    }


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "docs").mkdir()
    (root / "docs/research_protocol.md").write_text(
        "### A5.4 Reliability GO (replication) Criterion\n"
        "Route A E-AURC relative worsening >= 20% and 95% CI excludes zero; "
        "RER@50 or RER@80 drops >= 10 个百分点.\n"
        "Route B AUROC_correct drop >= 0.03 and 95% CI excludes zero, plus corrected map 稳定 shift.\n"
        "### A5.5 Gate standing\n",
        encoding="utf-8",
    )
    (root / "docs/phase0a_interpretation.md").write_text("Phase0A map interpretation\n", encoding="utf-8")
    a_entry = """```yaml
# ===== FORMAL ENTRY — REAL RESULT =====
experiment_id: p0a1-corrected-metrics-20260928-01
amendment_gate:
  route_A_selective: true
  route_B_correctness: false # ΔAUROC 未达 0.03；reliability shift 未达稳定阈值
```
"""
    b_entry = """```yaml
# ===== FORMAL ENTRY — REAL RESULT =====
experiment_id: p0b-b3-independent-20260928-01
amendment_gate:
  route_A_selective: true
  route_B_correctness: true # AUROC gate; corrected global-T reliability map 稳定 shift（保守方向，三 seed 一致）
notes: §26 Case A：GO
```
"""
    (root / "docs/experiment_log.md").write_text(a_entry + b_entry, encoding="utf-8")
    paths = [
        "results/phase0a_corrected/reliability_bins_globalT.csv",
        "results/phase0a_corrected/metadata.json",
        "results/phase0b_independent/seed_1/reliability_bins.csv",
        "results/phase0b_independent/seed_2/reliability_bins.csv",
        "results/phase0b_independent/seed_3/reliability_bins.csv",
        "results/phase0b_independent/metadata.json",
    ]
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps({"fixture": True}), encoding="utf-8")
        else:
            path.write_text("bin,confidence,accuracy,n\n", encoding="utf-8")
    return root


def _summary() -> dict:
    return {
        "jobs": [
            _job(_A_JOB, ["B0"]),
            _job(_B_JOB, ["b3_seed1", "b3_seed2", "b3_seed3"], e=(2.2, 2.0, 2.5), r=(0.45, 0.42, 0.48), a=(0.052, 0.044, 0.061)),
        ]
    }


def _by_gate(rows: list[dict]) -> dict[str, dict]:
    return {row["gate_id"]: row for row in rows}


def test_route_a_uses_fraction_units_and_historic_map_judgment_does_not_supply_go(tmp_path: Path) -> None:
    rows = _by_gate(build_reliability_go_decisions(_fixture_root(tmp_path), _summary()))

    cosine = rows["A5_4_RELIABILITY_GO_COSINE"]
    b3 = rows["A5_4_RELIABILITY_GO_B3"]
    composite = rows["A5_4_RELIABILITY_GO_REPLICATION"]

    assert cosine["old_decision"] == "GO"
    assert cosine["old_route_decisions"] == {"route_A": True, "route_B": False}
    assert cosine["new_result"]["route_A"]["status"] == "PASS"
    assert cosine["new_result"]["route_A"]["eaurc_relative_worsening"]["point"] == 0.25
    assert cosine["new_result"]["route_A"]["rer50_drop"]["point"] == 12.0
    assert cosine["new_result"]["route_A"]["rer_witness_selection"].startswith("RER@50 is the tested witness")
    assert cosine["new_decision"] == "GO_ROUTE_A"

    assert b3["old_decision"] == "GO"
    assert b3["old_route_decisions"] == {"route_A": True, "route_B": True}
    assert b3["new_result"]["route_A"]["status"] == "PASS"
    assert b3["new_result"]["route_B"]["numeric_status"] == "PASS"
    assert b3["new_result"]["route_B"]["status"] == "UNVERIFIABLE_MANUAL_MAP_REVIEW_REQUIRED"
    assert b3["new_result"]["route_B"]["map_component"]["numeric_threshold"] is None
    assert b3["new_result"]["route_B"]["map_component"]["new_manual_review"] == "NOT_PERFORMED"

    assert composite["old_decision"] == "GO_CASE_A"
    assert composite["new_decision"] == "GO_ROUTE_A_REPLICATED"
    assert composite["decision_basis"].startswith("Both the frozen B1 cosine and independent B3")
    assert "does not imply that cosine AUROC_correct declined" in composite["interpretation_limit"]
    assert {ref["estimate_name"] for ref in composite["new_estimates"]} == set(_NAMES.values()) | {
        "K5__global_T::rer_at_80", "K50__global_T::rer_at_80",
    }


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("config", "seed"), 1),
        (("config", "ci_level"), 0.90),
        (("config", "ci_method"), "basic"),
        (("config", "n_replicates"), 4999),
        (("estimate", "resample_unit"), "sentence"),
    ],
)
def test_wrong_bootstrap_method_cannot_make_route_a_gate_eligible(
    tmp_path: Path, field_path: tuple[str, str], replacement: object,
) -> None:
    summary = _summary()
    job = summary["jobs"][0]
    if field_path[0] == "config":
        job["config"][field_path[1]] = replacement
    else:
        job["estimates"][0][field_path[1]] = replacement
    rows = _by_gate(build_reliability_go_decisions(_fixture_root(tmp_path), summary))
    cosine = rows["A5_4_RELIABILITY_GO_COSINE"]
    assert cosine["new_result"]["route_A"]["status"] == "UNCERTAINTY_INSUFFICIENT"
    assert cosine["new_decision"] == "UNCERTAINTY_INSUFFICIENT"
    assert rows["A5_4_RELIABILITY_GO_REPLICATION"]["new_decision"] == "UNCERTAINTY_INSUFFICIENT"


def test_unknown_route_b_does_not_block_two_source_route_a_go(tmp_path: Path) -> None:
    summary = _summary()
    cosine_job = summary["jobs"][0]
    cosine_auroc = next(item for item in cosine_job["estimates"] if item["name"] == _NAMES["a"])
    cosine_auroc["resample_unit"] = "expression_cluster"

    rows = _by_gate(build_reliability_go_decisions(_fixture_root(tmp_path), summary))
    cosine = rows["A5_4_RELIABILITY_GO_COSINE"]
    assert cosine["new_result"]["route_A"]["status"] == "PASS"
    assert cosine["new_result"]["route_B"]["status"] == "UNCERTAINTY_INSUFFICIENT"
    assert cosine["new_decision"] == "GO_ROUTE_A"
    assert rows["A5_4_RELIABILITY_GO_REPLICATION"]["new_decision"] == "GO_ROUTE_A_REPLICATED"


def test_route_b_uncertainty_is_not_misreported_as_no_go_when_route_a_fails(tmp_path: Path) -> None:
    summary = _summary()
    cosine_job = summary["jobs"][0]
    for item in cosine_job["estimates"]:
        if item["name"] == _NAMES["e"]:
            item.update(point=0.10, ci_low=0.01, ci_high=0.19)
            for seed in item["per_seed"].values():
                seed.update(point=0.10, ci_low=0.01, ci_high=0.19)
        elif item["name"] == _NAMES["r"]:
            item.update(point=0.05, ci_low=0.01, ci_high=0.08)
            for seed in item["per_seed"].values():
                seed.update(point=0.05, ci_low=0.01, ci_high=0.08)
        elif item["name"] == _NAMES["a"]:
            item["resample_unit"] = "sentence"

    rows = _by_gate(build_reliability_go_decisions(_fixture_root(tmp_path), summary))
    cosine = rows["A5_4_RELIABILITY_GO_COSINE"]
    assert cosine["new_result"]["route_A"]["status"] == "FAIL"
    assert cosine["new_result"]["route_B"]["numeric_status"] == "UNCERTAINTY_INSUFFICIENT"
    assert cosine["new_decision"] == "UNCERTAINTY_INSUFFICIENT"
    assert rows["A5_4_RELIABILITY_GO_REPLICATION"]["status"] == "UNCERTAINTY_INSUFFICIENT"


def test_rer80_alternate_point_pass_without_paired_ci_stays_undecided(tmp_path: Path) -> None:
    summary = _summary()
    cosine_job = summary["jobs"][0]
    for item in cosine_job["estimates"]:
        if item["name"] == _NAMES["r"]:
            item.update(point=0.08, ci_low=0.03, ci_high=0.12)
            for seed in item["per_seed"].values():
                seed.update(point=0.08, ci_low=0.03, ci_high=0.12)

    rows = _by_gate(build_reliability_go_decisions(_fixture_root(tmp_path), summary))
    cosine = rows["A5_4_RELIABILITY_GO_COSINE"]
    route_a = cosine["new_result"]["route_A"]
    assert route_a["rer80_alternative"]["observed_drop_pp"] == pytest.approx(15.0)
    assert route_a["rer80_alternative"]["status"] == "POINT_PASSES_BUT_PAIRED_CI_UNAVAILABLE"
    assert route_a["rer80_alternative"]["paired_ci"] is None
    assert route_a["rer80_alternative"]["endpoint_cis_used_for_difference"] is False
    assert route_a["status"] == "UNCERTAINTY_INSUFFICIENT"
    assert cosine["new_decision"] == "UNCERTAINTY_INSUFFICIENT"


def test_negative_signed_effects_do_not_pass_and_needed_unreviewed_map_is_unverifiable(tmp_path: Path) -> None:
    summary = _summary()
    cosine_job = summary["jobs"][0]
    for item in cosine_job["estimates"]:
        if item["name"] == _NAMES["e"]:
            item.update(point=-0.25, ci_low=-0.35, ci_high=-0.10)
            for seed in item["per_seed"].values():
                seed.update(point=-0.25, ci_low=-0.35, ci_high=-0.10)
        elif item["name"] == _NAMES["r"]:
            item.update(point=-0.12, ci_low=-0.18, ci_high=-0.03)
            for seed in item["per_seed"].values():
                seed.update(point=-0.12, ci_low=-0.18, ci_high=-0.03)
        elif item["name"] == _NAMES["a"]:
            item.update(point=0.04, ci_low=0.01, ci_high=0.07)
    rows = _by_gate(build_reliability_go_decisions(_fixture_root(tmp_path), summary))
    cosine = rows["A5_4_RELIABILITY_GO_COSINE"]
    assert cosine["new_result"]["route_A"]["status"] == "FAIL"
    assert cosine["new_result"]["route_B"]["numeric_status"] == "PASS"
    assert cosine["new_decision"] == "UNVERIFIABLE_ROUTE_B_MANUAL_MAP_REVIEW_REQUIRED"
    assert cosine["status"] == "UNVERIFIABLE"
    assert rows["A5_4_RELIABILITY_GO_REPLICATION"]["status"] == "UNVERIFIABLE"


def test_route_a_threshold_boundaries_are_inclusive_and_units_are_not_double_scaled(tmp_path: Path) -> None:
    summary = _summary()
    for item in summary["jobs"][0]["estimates"]:
        if item["name"] == _NAMES["e"]:
            item.update(point=0.20, ci_low=0.01, ci_high=0.31)
            for seed in item["per_seed"].values():
                seed.update(point=0.20, ci_low=0.01, ci_high=0.31)
        elif item["name"] == _NAMES["r"]:
            item.update(point=0.10, ci_low=0.01, ci_high=0.15)
            for seed in item["per_seed"].values():
                seed.update(point=0.10, ci_low=0.01, ci_high=0.15)
    rows = _by_gate(build_reliability_go_decisions(_fixture_root(tmp_path), summary))
    route_a = rows["A5_4_RELIABILITY_GO_COSINE"]["new_result"]["route_A"]
    assert route_a["status"] == "PASS"
    assert route_a["eaurc_relative_worsening"]["point"] == 0.20
    assert route_a["rer50_drop"]["point"] == 10.0
