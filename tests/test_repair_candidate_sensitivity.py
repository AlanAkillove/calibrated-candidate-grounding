from __future__ import annotations

import numpy as np

from ccg.repairs import candidate_sensitivity as sensitivity


def _dose_seed_cells(n: int = 12) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    correct = np.tile(np.asarray([1, 0, 1, 0, 1, 0], dtype=np.float64), n // 6)
    confidence = np.linspace(0.15, 0.95, n, dtype=np.float64)
    cells = {seed: {} for seed in ("b3_seed1", "b3_seed2", "b3_seed3")}
    for seed_index, seed in enumerate(cells):
        for level_index, level in enumerate(sensitivity.DOSE_LEVELS):
            for condition_index, condition in enumerate(("old", "random", sensitivity.VARIANT)):
                for head_index, head in enumerate(sensitivity.HEADS):
                    roll = level_index + condition_index + head_index + seed_index
                    conf = np.roll(confidence, roll)
                    cells[seed][f"{condition}_{level}_{head}"] = (conf, correct)
    return cells


def test_dose_estimates_share_joint_draws_and_compute_macro_from_each_seed() -> None:
    ids = np.repeat(np.arange(4, dtype=np.int64), 3)
    result = sensitivity._run_one_group(
        _dose_seed_cells(), ids, cell_tag="dose", dose_levels=sensitivity.DOSE_LEVELS,
        n_replicates=32, seed=7, ci=0.95,
    )
    assert result["n_rows"] == 12
    assert result["n_images"] == 4
    estimates = result["estimates"]
    old_arm = estimates["old_dose_minus_old_m0__expb_m2__stats__auroc_correct"]
    corrected_arm = estimates["corrected_dose_minus_corrected_m0__expb_m2__stats__auroc_correct"]
    fixed_original = estimates["corrected_dose_minus_fixed_original_m0__expb_m2__stats__auroc_correct"]
    effect = estimates["dose_vs_m0_interaction__expb_m2__stats__auroc_correct"]
    for seed, value in effect["seed_estimates"].items():
        assert np.isclose(old_arm["seed_estimates"][seed] - corrected_arm["seed_estimates"][seed], value,
                          rtol=0.0, atol=1e-12)
        assert np.allclose(
            old_arm["per_seed_replicates"][seed] - corrected_arm["per_seed_replicates"][seed],
            effect["per_seed_replicates"][seed],
            rtol=0.0, atol=1e-12, equal_nan=True,
        )
    assert "fixed original Phase 1F expb_m0" in sensitivity._estimate_description(
        "corrected_dose_minus_fixed_original_m0__expb_m2__stats__auroc_correct"
    )["formula"]
    assert np.isfinite(fixed_original["point"])

    component_names = [
        f"old_minus_new__{level}__e1b__e_aurc" for level in sensitivity.DOSE_LEVELS
    ]
    macro = estimates["dose_macro_old_minus_new__dose_macro__e1b__e_aurc"]
    for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
        expected = np.mean([estimates[name]["seed_estimates"][seed] for name in component_names])
        assert np.isclose(macro["seed_estimates"][seed], expected, equal_nan=True)
        expected_reps = np.mean(
            [estimates[name]["per_seed_replicates"][seed] for name in component_names], axis=0
        )
        assert np.allclose(macro["per_seed_replicates"][seed], expected_reps,
                           rtol=0.0, atol=1e-12, equal_nan=True)
    assert macro["valid_replicates"] + macro["invalid_replicates"] == 32
    assert "dose_vs_m0_interaction__expb_m8__stats__e_aurc" in estimates
    assert "hard_random_interaction__expb_m2__stats__auroc_correct" not in estimates


def test_source_group_completion_reads_final_status_map() -> None:
    expected = [{
        "family": "RPN", "candidate_source": "phase1f", "cell": "hard5", "eval_split": split,
        "status": "PENDING",
    } for split in ("testA", "testB", "pooled_testA_testB")]
    statuses = {
        ("RPN", "phase1f", "hard5", "testA"): "PILOT",
        ("RPN", "phase1f", "hard5", "testB"): "PILOT",
        ("RPN", "phase1f", "hard5", "pooled_testA_testB"): "PILOT",
    }
    accepted = ("PILOT", "COMPONENT_INCLUDED_IN_DOSE_JOINT_GROUP_PILOT")
    assert sensitivity._source_groups_complete(expected, statuses, accepted)
    del statuses[("RPN", "phase1f", "hard5", "testB")]
    assert not sensitivity._source_groups_complete(expected, statuses, accepted)


def test_candidate_sensitivity_pilot_is_explicit_and_never_claims_complete(tmp_path, monkeypatch) -> None:
    from ccg.repairs import candidate_forward

    n = 12
    ids = np.arange(n, dtype=np.int64)
    correct = np.tile(np.asarray([1, 0, 1, 0, 0, 1], dtype=bool), 2)
    archive = {
        "sentence_id": ids, "ref_id": ids + 100, "image_id": np.repeat(np.arange(4), 3),
        "ann_id": ids + 200, "eval_split": np.asarray(["testA"] * n),
    }
    for seed_i, seed in enumerate(("b3_seed1", "b3_seed2", "b3_seed3")):
        for condition_i, condition in enumerate(("old", "random", sensitivity.VARIANT)):
            archive[f"{condition}__{seed}__correct"] = np.roll(correct, seed_i + condition_i)
            for head_i, head in enumerate(sensitivity.HEADS):
                archive[f"{condition}__{seed}__conf_{head}"] = np.roll(
                    np.linspace(0.1, 0.9, n), condition_i + seed_i + head_i
                )

    forward_dir = tmp_path / "audit_run_x" / "forward"
    forward_dir.mkdir(parents=True)
    (forward_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (forward_dir / "predictions_manifest.json").write_text("{}\n", encoding="utf-8")
    fake = {
        "audit": {"forward_dir": forward_dir, "pointer": {"audit_run_id": "x"},
                  "summary_path": forward_dir.parent / "summary.json", "summary_sha256": "audit-sha"},
        "forward_summary": {"predictions_manifest": {"path": "results/fake/predictions_manifest.json"}},
        "manifest_path": forward_dir / "predictions_manifest.json",
    }
    entry = {
        "family": "RPN", "candidate_source": "phase1f_frozen_candidate_archive", "cell": "hard5",
        "_key": ("RPN", "phase1f_frozen_candidate_archive", "hard5"),
        "_status": "FORWARDED_WITH_OLD_ANCHOR_PASS", "_arrays": archive,
    }
    monkeypatch.setattr(sensitivity, "_forward_archives", lambda pointer: (fake, [entry]))
    original_bootstrap = sensitivity.joint_prediction_bootstrap
    seen = []

    def recording_bootstrap(image_ids, seed_cells, **kwargs):
        seen.append(np.asarray(image_ids).copy())
        return original_bootstrap(image_ids, seed_cells, **kwargs)

    monkeypatch.setattr(sensitivity, "joint_prediction_bootstrap", recording_bootstrap)
    report = sensitivity.run_candidate_sensitivity_bootstrap(n_replicates=12, seed=3, ci=0.90)
    assert report["status"] == "PILOT_PARTIAL"
    assert report["formal"] is False
    assert report["bootstrap_contract"]["n_replicates"] == 12
    assert len(seen) == 2
    assert seen[0].size == 12 and np.unique(seen[0]).size == 4
    assert seen[1].size == 12 and np.unique(seen[1]).size == 4
    expected = report["expected_source_cohort_split_groups"]
    assert len(expected) == 3
    assert {row["eval_split"] for row in expected} == {"pooled_testA_testB", "testA", "testB"}
    assert next(row for row in expected if row["eval_split"] == "testB")["status"] == "UNAVAILABLE_EMPTY_EVAL_SPLIT"
    assert report["groups"][0]["n_rows"] == 12
    assert report["groups"][0]["n_images"] == 4
    output = forward_dir / next(path.name for path in forward_dir.iterdir() if path.name.startswith("pilot_sensitivity_"))
    saved = candidate_forward._read_json(output / "summary.json")
    assert saved["status"] == "PILOT_PARTIAL"
    assert not (forward_dir / "sensitivity_bootstrap").exists()
