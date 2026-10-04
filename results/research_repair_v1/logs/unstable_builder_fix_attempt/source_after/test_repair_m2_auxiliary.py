from __future__ import annotations

import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.repairs.m2_auxiliary import (
    M2_RELATIVE_EAURC_NAME,
    M2_RELATIVE_EAURC_SPEC,
    M2_SEEDS,
    derive_m2_relative_eaurc_reduction,
)
from ccg.repairs.statistics import joint_prediction_bootstrap
from scripts import derive_m2_gate_estimate as m2_cli


def _valid_fixture():
    e1b = {
        "b3_seed1": np.array([2.0, 1.0, 2.0, 2.0]),
        "b3_seed2": np.array([1.0, 2.0, 3.0, 4.0]),
        "b3_seed3": np.array([3.0, 3.0, 3.0, 3.0]),
    }
    lcr = {
        "b3_seed1": np.array([1.0, 0.5, 1.0, 1.8]),
        "b3_seed2": np.array([2.0, 1.0, 3.0, 2.0]),
        "b3_seed3": np.array([3.0, 1.5, 0.0, 1.5]),
    }
    points_e1b = {"b3_seed1": 2.0, "b3_seed2": 4.0, "b3_seed3": 3.0}
    points_lcr = {"b3_seed1": 1.8, "b3_seed2": 2.0, "b3_seed3": 1.5}
    old = {"b3_seed1": 0.1, "b3_seed2": 0.5, "b3_seed3": 0.5}
    return e1b, lcr, points_e1b, points_lcr, old


def test_m2_auxiliary_uses_same_draw_per_seed_ratio_before_seed_mean():
    e1b, lcr, points_e1b, points_lcr, old = _valid_fixture()
    estimate, raw = derive_m2_relative_eaurc_reduction(
        e1b_draws=e1b,
        lcr_draws=lcr,
        e1b_points=points_e1b,
        lcr_points=points_lcr,
        old_seed_points=old,
        old_mean_point=float(np.mean(list(old.values()))),
        expected_replicates=4,
    )

    per_seed = np.stack([(e1b[seed] - lcr[seed]) / e1b[seed] for seed in M2_SEEDS])
    expected_aggregate = np.mean(per_seed, axis=0)
    np.testing.assert_allclose(raw[f"aggregate__{M2_RELATIVE_EAURC_NAME}"], expected_aggregate)
    for seed in M2_SEEDS:
        np.testing.assert_allclose(raw[f"seed_{seed}__{M2_RELATIVE_EAURC_NAME}"], per_seed[M2_SEEDS.index(seed)])

    ratio_of_mean_endpoints = (
        np.mean(np.stack([e1b[seed] for seed in M2_SEEDS]), axis=0)
        - np.mean(np.stack([lcr[seed] for seed in M2_SEEDS]), axis=0)
    ) / np.mean(np.stack([e1b[seed] for seed in M2_SEEDS]), axis=0)
    assert not np.isclose(expected_aggregate[0], ratio_of_mean_endpoints[0])
    assert estimate["point"] == pytest.approx(np.mean([0.1, 0.5, 0.5]))
    assert estimate["old_point"] == pytest.approx(estimate["point"])
    assert estimate["valid_replicates"] == 4
    assert estimate["invalid_replicates"] == 0
    assert estimate["gate_eligible"] is True
    assert estimate["per_seed"]["b3_seed2"]["valid_replicates"] == 4


def test_m2_zero_and_near_zero_denominators_are_invalid_not_zero_filled():
    e1b, lcr, points_e1b, points_lcr, old = _valid_fixture()
    e1b["b3_seed1"] = np.array([1.0, 0.0, 1e-12, 2.0])
    lcr["b3_seed1"] = np.array([0.5, 0.0, 0.0, 1.0])
    estimate, raw = derive_m2_relative_eaurc_reduction(
        e1b_draws=e1b,
        lcr_draws=lcr,
        e1b_points=points_e1b,
        lcr_points=points_lcr,
        old_seed_points=old,
        old_mean_point=float(np.mean(list(old.values()))),
        expected_replicates=4,
    )

    seed_values = raw[f"seed_b3_seed1__{M2_RELATIVE_EAURC_NAME}"]
    aggregate = raw[f"aggregate__{M2_RELATIVE_EAURC_NAME}"]
    assert np.isnan(seed_values[1]) and np.isnan(seed_values[2])
    assert np.isnan(aggregate[1]) and np.isnan(aggregate[2])
    assert estimate["invalid_replicates"] == 2
    assert estimate["valid_replicates"] == 2
    assert estimate["gate_eligible"] is False
    assert estimate["uncertainty_status"] == "UNCERTAINTY_INSUFFICIENT"
    assert estimate["per_seed"]["b3_seed1"]["invalid_replicates"] == 2


def test_m2_auxiliary_rejects_historical_point_anchor_mismatch():
    e1b, lcr, points_e1b, points_lcr, old = _valid_fixture()
    old["b3_seed1"] += 2e-9
    with pytest.raises(ValueError, match="relative E-AURC anchor failed"):
        derive_m2_relative_eaurc_reduction(
            e1b_draws=e1b,
            lcr_draws=lcr,
            e1b_points=points_e1b,
            lcr_points=points_lcr,
            old_seed_points=old,
            old_mean_point=float(np.mean(list(old.values()))),
            expected_replicates=4,
        )


def test_m2_auxiliary_requires_fixed_seed_identity_and_shared_draw_lengths():
    e1b, lcr, points_e1b, points_lcr, old = _valid_fixture()
    with pytest.raises(ValueError, match="fixed seeds"):
        derive_m2_relative_eaurc_reduction(
            e1b_draws={seed: e1b[seed] for seed in M2_SEEDS[:2]},
            lcr_draws={seed: lcr[seed] for seed in M2_SEEDS[:2]},
            e1b_points={seed: points_e1b[seed] for seed in M2_SEEDS[:2]},
            lcr_points={seed: points_lcr[seed] for seed in M2_SEEDS[:2]},
            old_seed_points={seed: old[seed] for seed in M2_SEEDS[:2]},
            old_mean_point=0.3,
            expected_replicates=4,
        )
    lcr["b3_seed2"] = lcr["b3_seed2"][:-1]
    with pytest.raises(ValueError, match="aligned 1D vectors"):
        derive_m2_relative_eaurc_reduction(
            e1b_draws=e1b,
            lcr_draws=lcr,
            e1b_points=points_e1b,
            lcr_points=points_lcr,
            old_seed_points=old,
            old_mean_point=float(np.mean(list(old.values()))),
            expected_replicates=4,
        )


def test_m2_relative_spec_is_a_prediction_estimate_without_auxiliary_cells():
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    seed_cells = {
        seed: {
            "m8__E1b": (np.array([0.9, 0.8, 0.7, 0.6]), correct),
            "m8__LCR": (np.array([0.9, 0.7, 0.8, 0.6]), correct),
        }
        for seed in M2_SEEDS
    }
    result = joint_prediction_bootstrap(
        np.array([11, 11, 22, 22]),
        seed_cells,
        estimates=[M2_RELATIVE_EAURC_SPEC],
        metrics=("e_aurc",),
        n_replicates=8,
        seed=0,
        ci=0.95,
    )
    assert M2_RELATIVE_EAURC_NAME in result["estimates"]
    assert set(result["estimates"][M2_RELATIVE_EAURC_NAME]["seed_estimates"]) == set(M2_SEEDS)


def test_m2_cli_requires_both_endpoints_to_share_the_same_raw_archive():
    names = ("m8__E1b::e_aurc", "m8__LCR::e_aurc")
    estimates = {
        names[0]: {"raw_replicates": "results/repair/raw.npz"},
        names[1]: {"raw_replicates": "results/other.npz"},
    }
    with pytest.raises(ValueError, match="same raw replicate archive"):
        m2_cli._shared_endpoint_archive(estimates, names)


def test_m2_cli_preflight_does_not_write_derived_output(tmp_path, monkeypatch):
    stat_dir = tmp_path / "results" / "research_repair_v1" / "statistics"
    raw_dir = stat_dir / "raw_replicates"
    raw_dir.mkdir(parents=True)
    raw_rel = "results/research_repair_v1/statistics/raw_replicates/m2.npz"
    raw_path = tmp_path / raw_rel
    endpoint_names = ("m8__E1b::e_aurc", "m8__LCR::e_aurc")
    raw_arrays = {}
    for seed in M2_SEEDS:
        raw_arrays[f"seed_{seed}__{endpoint_names[0]}"] = np.full(5000, 2.0)
        raw_arrays[f"seed_{seed}__{endpoint_names[1]}"] = np.full(5000, 1.8)
    raw_arrays[f"aggregate__{endpoint_names[0]}"] = np.full(5000, 2.0)
    raw_arrays[f"aggregate__{endpoint_names[1]}"] = np.full(5000, 1.8)
    np.savez(raw_path, **raw_arrays)

    gate_path = tmp_path / m2_cli.GATE_PATH
    gate_path.parent.mkdir(parents=True)
    gate_data = {
        "gates": {"selective_metrics": {
            "e_aurc_reduction": 0.1,
            "per_seed": {str(index): {"e_aurc_reduction": 0.1} for index in (1, 2, 3)},
        }},
    }
    gate_path.write_text(json.dumps(gate_data), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    estimates = []
    for name, value in zip(endpoint_names, (2.0, 1.8), strict=True):
        estimates.append({
            "name": name,
            "raw_replicates": raw_rel,
            "raw_replicates_sha256": raw_sha,
            "seed_estimates": {seed: value for seed in M2_SEEDS},
        })
    summary = {
        "bootstrap": {
            "replicates": 5000, "seed": 0, "ci_level": 0.95,
            "method": "percentile", "resample_unit": "image_cluster",
            "validity_sufficiency_rule": {"minimum_valid_fraction_for_exploratory_ci": 0.95},
        },
        "jobs": [{
            "job_id": m2_cli.JOB_ID,
            "scope": "V2M_M2_curriculum",
            "cohort": "same8_m0_m2_m4_m8",
            "status": "RECOMPUTED_5000",
            "config": {"n_replicates": 5000, "seed": 0, "ci_level": 0.95, "ci_method": "percentile"},
            "result": {"n_replicates": 5000, "seed": 0, "ci_level": 0.95, "method": "percentile",
                       "resample_unit": "image_cluster", "n_rows": 9, "n_images": 3},
            "source_artifacts": [{"path": m2_cli.GATE_PATH, "sha256": gate_sha}],
            "estimates": estimates,
        }],
    }
    summary_path = stat_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_bytes = json.dumps(summary).encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    (stat_dir / "status.json").write_text(json.dumps({"status": "PARTIAL_SCOPE_COMPLETE"}), encoding="utf-8")
    monkeypatch.setattr(m2_cli, "ROOT", tmp_path)
    monkeypatch.setattr(m2_cli, "STAT_DIR", stat_dir)

    result = m2_cli.derive_and_write(apply=False)

    assert result["status"] == "PREFLIGHT_ONLY"
    assert result["derived_point"] == pytest.approx(0.1)
    assert result["valid_replicates"] == 5000
    assert result["invalid_replicates"] == 0
    assert not (stat_dir / "raw_replicates" / f"{m2_cli.JOB_ID}__m2_relative_auxiliary.npz").exists()
    assert summary_path.read_bytes() == summary_bytes
