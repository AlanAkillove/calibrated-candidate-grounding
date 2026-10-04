"""Small synthetic end-to-end checks for B information bootstrap assembly."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import repair_information_bootstrap as iboot  # noqa: E402
from ccg.repairs import atomic as atomic_io  # noqa: E402
from ccg.repairs.statistics import metric_bundle  # noqa: E402


SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
GROUPS = ("S", "S+Q", "S+V", "Full")
KS = (5, 10, 20, 50)


def _make_data(seed_index: int, cell_index: int, k: int, correctness: np.ndarray, model_index: int = 0) -> dict[str, np.ndarray]:
    n = correctness.size
    row = np.arange(n)
    # Every image owns two adjacent expressions; testA and testB each have four images.
    image_id = 100 + row // 2
    eval_split = np.where(image_id < 104, 0, 1).astype(np.int8)
    rank = (row * 5 + seed_index * 3 + cell_index + (k // 5) + model_index * 2) % n
    confidence = (rank + 1.0) / (n + 1.0)
    return {
        "sentence_id": 1000 + row,
        "ref_id": 2000 + row,
        "image_id": image_id,
        "eval_split": eval_split,
        "K": np.full(n, k, dtype=np.int16),
        "grounding_correct": correctness.astype(np.int8),
        "correct": correctness.astype(np.int8),
        "confidence": confidence.astype(np.float64),
    }


def _synthetic_tables():
    semantic = {}
    cross = {}
    cells = tuple(f"randomK{k}" for k in KS) + ("rand5", "hard5", "expb_m0", "expb_m2", "expb_m4", "expb_m8")
    cell_indices = {name: index for index, name in enumerate(cells)}
    for seed_index, seed in enumerate(SEEDS):
        for cell in cells:
            k = 5 if cell in ("rand5", "hard5") else 10 if cell.startswith("expb_") else int(cell.removeprefix("randomK"))
            # Labels vary by seed, K, and sampling cell. They stay fixed only
            # across reliability feature groups within one seed/cell/K.
            cell_offset = cell_indices[cell]
            rows = np.arange(16)
            correct = ((rows + seed_index + cell_offset + (k // 5)) % 3 != 0).astype(np.int8)
            local_s = _make_data(seed_index, cell_offset, k, correct, 0)
            for group_index, group in enumerate(GROUPS):
                semantic[iboot._key(seed, cell, k, group)] = _make_data(
                    seed_index, cell_offset, k, correct, group_index,
                )
            if cell.startswith("randomK"):
                for model_index, (model, regime) in enumerate((
                    ("StatsLogistic_smallK5_10", "smallK5_10"),
                    ("StatsLogistic_largeK20_50", "largeK20_50"),
                    ("ScoreDeepSets_smallK5_10", "smallK5_10"),
                    ("ScoreDeepSets_largeK20_50", "largeK20_50"),
                    ("corrected_score_MSP", "score_only"),
                ), start=1):
                    cross[iboot._key(seed, cell, k, model, regime)] = _make_data(
                        seed_index, cell_offset, k, local_s["grounding_correct"], model_index,
                    )
    return semantic, cross


def test_alignment_ignores_cross_k_and_cross_seed_correctness_but_can_enforce_within_cell_labels():
    ref = {
        "sentence_id": np.asarray([1, 2, 3]),
        "ref_id": np.asarray([11, 12, 13]),
        "image_id": np.asarray([21, 22, 23]),
        "eval_split": np.asarray([0, 0, 1]),
        "K": np.asarray([5, 5, 5]),
        "grounding_correct": np.asarray([1, 1, 0]),
    }
    cross_k = {key: value.copy() for key, value in ref.items()}
    cross_k["K"][:] = 50
    cross_k["grounding_correct"] = np.asarray([0, 1, 0])
    iboot._verify_alignment(ref, cross_k, label="nested-K-fixture", expected_k=50)
    with pytest.raises(ValueError, match="grounding correctness differs"):
        iboot._verify_alignment(ref, cross_k, label="same-cell-groups", expected_k=50, same_correctness=True)


def test_low_replicate_full_information_bootstrap_fixture_checks_pairing_sign_and_dod():
    semantic, cross = _synthetic_tables()
    flat, calls, raw = iboot._compute_bootstrap(semantic, cross, n_replicates=2, evidence_path="fixture.npz")

    assert len(calls) == 9  # random(all four K), joint matched pair, joint dose cohort; testA/B/pooled
    assert {scope: sum(call["scope"] == scope for call in calls)
            for scope in ("random", "matched_same4", "dose_same8")} == {
                "random": 3, "matched_same4": 3, "dose_same8": 3,
            }
    assert raw
    main = next(row for row in flat if row["scope"] == "random" and row["cell"] == "randomK5"
                and row["K"] == "5" and row["eval_split"] == "testA"
                and row["contrast"] == "Full_minus_S_plus_Q" and row["metric"] == "auroc_correct")
    rer_gain = next(row for row in flat if row["scope"] == "random" and row["cell"] == "randomK5"
                    and row["K"] == "5" and row["eval_split"] == "testA"
                    and row["contrast"] == "Full_minus_S_plus_Q" and row["metric"] == "rer_at_50_gain_pp")
    eaurc_gain = next(row for row in flat if row["scope"] == "random" and row["cell"] == "randomK5"
                      and row["K"] == "5" and row["eval_split"] == "testA"
                      and row["contrast"] == "Full_minus_S_plus_Q" and row["metric"] == "e_aurc_reduction")

    expected = {"auroc_correct": [], "rer_at_50_gain_pp": [], "e_aurc_reduction": []}
    for seed in SEEDS:
        full = semantic[iboot._key(seed, "randomK5", 5, "Full")]
        sq = semantic[iboot._key(seed, "randomK5", 5, "S+Q")]
        rows = full["eval_split"] == 0
        fm = metric_bundle(full["confidence"][rows], full["grounding_correct"][rows], metrics=iboot.METRICS)
        sm = metric_bundle(sq["confidence"][rows], sq["grounding_correct"][rows], metrics=iboot.METRICS)
        expected["auroc_correct"].append(fm["auroc_correct"] - sm["auroc_correct"])
        # RER is a skill score (larger is better), so the gain for
        # Full_minus_S_plus_Q must be Full minus S+Q, matching the label.
        expected["rer_at_50_gain_pp"].append(100.0 * (fm["rer_at_50"] - sm["rer_at_50"]))
        expected["e_aurc_reduction"].append(sm["e_aurc"] - fm["e_aurc"])
    assert main["point"] == pytest.approx(np.mean(expected["auroc_correct"]))
    assert rer_gain["point"] == pytest.approx(np.mean(expected["rer_at_50_gain_pp"]))
    assert rer_gain["point"] == pytest.approx(22.22222222222223)
    assert eaurc_gain["point"] == pytest.approx(np.mean(expected["e_aurc_reduction"]))
    assert "fixture.npz::" in rer_gain["per_seed_evidence_path"]

    matched_call = next(call for call in calls if call["scope"] == "matched_same4" and call["eval_split"] == "testA")
    assert "effect::Hard_minus_Rand_Full::rer_at_50_gain_pp" in matched_call["estimates"]
    hard_rand = next(row for row in flat if row["scope"] == "matched_same4" and row["eval_split"] == "testA"
                     and row["contrast"] == "Hard_minus_Rand_Full" and row["metric"] == "auroc_correct")
    expected_hard_rand = []
    for seed in SEEDS:
        hard = semantic[iboot._key(seed, "hard5", 5, "Full")]
        rand = semantic[iboot._key(seed, "rand5", 5, "Full")]
        hm = metric_bundle(hard["confidence"][hard["eval_split"] == 0], hard["grounding_correct"][hard["eval_split"] == 0], metrics=iboot.METRICS)
        rm = metric_bundle(rand["confidence"][rand["eval_split"] == 0], rand["grounding_correct"][rand["eval_split"] == 0], metrics=iboot.METRICS)
        expected_hard_rand.append(hm["auroc_correct"] - rm["auroc_correct"])
    assert hard_rand["point"] == pytest.approx(np.mean(expected_hard_rand))

    dose_call = next(call for call in calls if call["scope"] == "dose_same8" and call["eval_split"] == "testB")
    assert "effect::expb_m8_minus_m0_Full::e_aurc_reduction" in dose_call["estimates"]
    assert dose_call["n_images"] == len(np.unique(_make_data(0, 0, 10, np.ones(16, dtype=np.int8))["image_id"][
        _make_data(0, 0, 10, np.ones(16, dtype=np.int8))["eval_split"] == 1]))

    paired = next(row for row in flat if row["scope"] == "random" and row["eval_split"] == "testA"
                  and row["contrast"] == "large_minus_small_StatsLogistic_K20"
                  and row["metric"] == "auroc_difference")
    dod = next(row for row in flat if row["scope"] == "random" and row["eval_split"] == "testA"
               and row["contrast"] == "K50_vs_K5_regime_DoD_StatsLogistic"
               and row["metric"] == "auroc_change_DoD")
    expected_dod = []
    for seed in SEEDS:
        def auc(model: str, regime: str, k: int) -> float:
            data = cross[iboot._key(seed, f"randomK{k}", k, f"{model}_{regime}", regime)]
            rows = data["eval_split"] == 0
            return metric_bundle(data["confidence"][rows], data["grounding_correct"][rows], metrics=iboot.METRICS)["auroc_correct"]
        expected_dod.append((auc("StatsLogistic", "largeK20_50", 50) - auc("StatsLogistic", "largeK20_50", 5))
                            - (auc("StatsLogistic", "smallK5_10", 50) - auc("StatsLogistic", "smallK5_10", 5)))
    assert dod["point"] == pytest.approx(np.mean(expected_dod))
    assert paired["n_replicates"] == 2


def test_call_checkpoints_are_persisted_and_reused_only_for_matching_sources(tmp_path):
    semantic, cross = _synthetic_tables()
    source = {"runner_sha256": "fixture-runner", "metric_sha256": "fixture-metrics"}
    kwargs = {
        "n_replicates": 2,
        "checkpoint_dir": tmp_path,
        "source_fingerprint": source,
    }
    first_rows, first_calls, first_raw = iboot._compute_bootstrap(semantic, cross, **kwargs)
    first_call_log = (tmp_path / "call_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(first_calls) == 9
    assert len(first_call_log) == 18  # one STARTED and one COMPLETED record per call
    assert len(list(tmp_path.glob("*.json"))) == 9
    assert len(list(tmp_path.glob("*.npz"))) == 9
    checkpoint = json.loads(next(tmp_path.glob("random__random_allK__testA.*.json")).read_text(encoding="utf-8"))
    assert checkpoint["metadata"]["source_sha256"] == source
    assert checkpoint["call"]["elapsed_seconds"] >= 0
    assert checkpoint["call"]["cohort_counts"]["n_images"] > 0

    second_rows, second_calls, second_raw = iboot._compute_bootstrap(semantic, cross, **kwargs)
    assert first_rows == second_rows
    assert [call["call_id"] for call in first_calls] == [call["call_id"] for call in second_calls]
    assert first_raw.keys() == second_raw.keys()
    for key in first_raw:
        np.testing.assert_array_equal(first_raw[key], second_raw[key])
    all_events = [json.loads(line) for line in (tmp_path / "call_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(event.get("completion_mode") == "CHECKPOINT_REUSE" for event in all_events) == 9


def test_npz_writer_retries_two_transient_windows_locks_and_succeeds(tmp_path, monkeypatch):
    target = tmp_path / "bootstrap_raw.npz"
    target.write_bytes(b"previous-complete-artifact")
    real_replace = atomic_io.os.replace
    attempts = []
    delays = []

    def transient_twice(source, destination):
        attempts.append((Path(source), Path(destination)))
        if len(attempts) <= 2:
            error = PermissionError("synthetic transient Windows sharing lock")
            error.winerror = 32
            raise error
        return real_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", transient_twice)
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)
    iboot._write_npz_atomic(target, {"draws": np.asarray([0.25, 0.75], dtype=np.float64)})

    assert len(attempts) == 3
    assert attempts[-1][1] == target
    assert len(delays) == 2
    with np.load(target, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["draws"], [0.25, 0.75])
    assert not list(tmp_path.glob("*.tmp"))
