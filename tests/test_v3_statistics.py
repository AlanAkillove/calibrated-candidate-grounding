"""Weighted tie groups must equal explicit repeated-expression resampling."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccg.metrics.discrimination import auroc_correct
from ccg.v3.evaluation import fractional_boundary_risk
from ccg.v3.statistics import WeightedReliability, controlled_inference, export_inference, load_prediction_grid, natural_inference


def test_weighted_metrics_equal_explicit_repeated_rows_and_ties():
    scores = np.array([.8, .8, .2, .3, .3])
    correct = np.array([1, 0, 1, 0, 1])
    weights = np.array([2, 0, 3, 1, 2])
    indices = np.repeat(np.arange(5), weights)
    result = WeightedReliability(scores, correct).evaluate(weights)
    assert result["brier_binary"] == pytest.approx(np.mean((scores[indices] - correct[indices]) ** 2))
    assert result["auroc_correct"] == pytest.approx(auroc_correct(scores[indices], correct[indices]))
    for c, key in ((.5, "risk50"), (.8, "risk80")):
        assert result[key] == pytest.approx(fractional_boundary_risk(scores[indices], correct[indices], c))


def test_zero_denominator_and_single_class_stay_undefined():
    profile = WeightedReliability([.3, .6], [1, 1])
    assert np.isnan(profile.evaluate([1, 1])["auroc_correct"])
    assert profile.evaluate([1, 1])["risk50"] == 0
    assert all(np.isnan(value) for value in profile.evaluate([0, 0]).values())
    with pytest.raises(ValueError, match="binary"):
        WeightedReliability([.2], [2])


def grid_fixture():
    order = ["r1", "r2", "r3", "r4"]
    images = np.array([1, 1, 2, 3])  # repeated expressions must remain clustered
    grid = {}
    for mode in ("controlled", "natural"):
        for backbone in ("b0", "b16"):
            for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
                for k in (5, 10, 20, 50):
                    grid[mode, backbone, seed, k] = {}
                    for i, record in enumerate(order):
                        correct = bool(i % 2) if backbone == "b16" else bool(1 - i % 2)
                        confidence = {name: float([.9, .1, .8, .2][i]) for name in ("MSP", "S", "S+Q", "S+V", "SDS_small", "SDS_large")}
                        confidence["Full"] = float([.1, .9, .2, .8][i])
                        grid[mode, backbone, seed, k][record] = {"record_id": record, "image_id": int(images[i]),
                            "source_split": "val", "mode": mode, "backbone": backbone, "seed": seed,
                            "requested_k": k, "k_eff": k, "correct": correct, "target_present": True, "bank_target_coverage": True,
                            "missing_reason": None, "confidence": confidence}
    return order, images, grid


def test_primary_contrasts_use_each_backbone_labels_and_save_same_draws(tmp_path):
    order, images, grid = grid_fixture()
    result = controlled_inference(order, images, grid, n_replicates=100)
    assert result["estimates"]["C2/auroc_correct"]["point"] == -1
    assert result["estimates"]["C3/auroc_correct"]["point"] == 1
    summary = export_inference(result, tmp_path, primary=True)
    saved = np.load(tmp_path / "draws.npz")
    item = summary["estimates"]["C3/auroc_correct"]
    assert "per_seed_replicates" not in item
    keys = summary["raw_draws"]["keys"]["C3/auroc_correct"]
    draws = saved[keys["mean"]]
    assert np.allclose(draws, np.mean([saved[key] for key in keys["per_seed"].values()], axis=0), equal_nan=True)
    valid = draws[np.isfinite(draws)]
    assert item["bonferroni_9875"]["ci_low"] == pytest.approx(np.quantile(valid, .00625))
    assert item["invalid_replicates"] > 0
    assert "bonferroni_9875" not in summary["estimates"]["C3/risk50"]


def test_prediction_grid_preserves_expressions_and_rejects_duplicates(tmp_path):
    order, images, grid = grid_fixture()
    path = tmp_path / "predictions.jsonl"
    rows = [row for group in grid.values() for row in group.values()]
    for row in rows:
        row["effective_k"] = row.pop("k_eff")
        row["confidences"] = row.pop("confidence")
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    loaded, loaded_images, _ = load_prediction_grid(path)
    assert loaded == order and loaded_images[0] == loaded_images[1]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + json.dumps(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        load_prediction_grid(path)


def test_natural_empty_and_missing_rows_keep_all_accuracy_denominator():
    order, images, grid = grid_fixture()
    for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
        row = grid["natural", "b0", seed, 5]["r1"]
        row.update(k_eff=0, correct=None, confidence={}, target_present=False)
        grid["natural", "b0", seed, 5]["r2"]["confidence"]["Full"] = None
    result = natural_inference(order, images, grid, n_replicates=20)
    flow = result["flow"]["b0/b3_seed1/K5"]
    assert flow["n_empty"] == 1
    assert flow["n_all"] == 4
    assert flow["model_denominators"]["Full"]["overall"] == 2
    assert flow["paired_denominators"]["overall"] == 2
    assert result["estimates"]["b0/K5/flow/accuracy_all"]["point"] == .25


def test_omission_ranking_accounting_is_signed_and_exact():
    order, images, grid = grid_fixture()
    for seed in ("b3_seed1", "b3_seed2", "b3_seed3"):
        grid["natural", "b0", seed, 5]["r2"]["target_present"] = False
    result = natural_inference(order, images, grid, n_replicates=20)
    values = result["estimates"]
    gain = values["b0/K5/paired_overall/Full_minus_SQ/auroc_correct"]["point"]
    covered = values["b0/K5/ranking_audit/covered_ranking_contribution"]["point"]
    uncovered = values["b0/K5/ranking_audit/uncovered_ranking_contribution"]["point"]
    assert gain == pytest.approx(covered + uncovered)
    assert values["b0/K5/ranking_audit/uncovered_negative_weight"]["point"] == .5
