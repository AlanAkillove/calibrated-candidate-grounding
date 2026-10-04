"""Mathematical invariants for the shared Repair v1 bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.repairs.bootstrap import assert_row_alignment, shared_image_cluster_bootstrap


def test_shared_draws_mean_seed_effects_and_preserves_raw_replicates() -> None:
    # Unequal rows per image ensure the draw expands whole clusters, rather
    # than resampling expression rows or forcing equal image sizes.
    image_ids = np.asarray([10, 10, 20, 30, 30, 30, 40])
    seed_a = np.asarray([0.0, 2.0, 4.0, 2.0, 4.0, 6.0, 10.0])
    seed_b = np.asarray([2.0, 4.0, 8.0, 6.0, 8.0, 10.0, 14.0])
    n_seen = 0

    def statistic(rows: np.ndarray) -> dict[str, dict[int, float]]:
        nonlocal n_seen
        n_seen += 1
        a = float(np.mean(seed_a[rows]))
        b = float(np.mean(seed_b[rows]))
        return {"delta": {1: a, 2: b}, "same_draw_copy": {1: a, 2: b}}

    result = shared_image_cluster_bootstrap(image_ids, statistic, n_replicates=200, seed=0)
    delta = result["estimates"]["delta"]
    assert n_seen == 201  # point plus one callback for each shared draw
    assert delta["point"] == pytest.approx(np.mean([seed_a.mean(), seed_b.mean()]))
    assert delta["seed_standard_deviation"] == pytest.approx(np.std([seed_a.mean(), seed_b.mean()], ddof=1))
    assert delta["replicates"].shape == (200,)
    assert delta["per_seed_replicates"][1].shape == (200,)
    assert np.array_equal(delta["replicates"], result["estimates"]["same_draw_copy"]["replicates"])
    assert np.allclose(
        delta["replicates"],
        (delta["per_seed_replicates"][1] + delta["per_seed_replicates"][2]) / 2.0,
    )
    assert delta["valid_replicates"] == 200 and delta["invalid_replicates"] == 0
    assert delta["ci_low"] <= delta["point"] <= delta["ci_high"]


def test_nonfinite_seed_draws_are_counted_and_never_filled_with_zero() -> None:
    image_ids = np.asarray([1, 2, 3, 4])
    labels = np.asarray([0, 0, 1, 1])

    def statistic(rows: np.ndarray) -> dict[str, dict[str, float]]:
        sampled = labels[rows]
        # Simulate AUROC becoming undefined on single-class bootstrap draws.
        auroc = float("nan") if np.unique(sampled).size < 2 else 0.75
        # Simulate an undefined relative effect when its denominator is zero.
        relative = float("nan") if int(np.sum(sampled)) == 0 else 0.25
        return {"auroc": {"s1": auroc, "s2": 0.5}, "relative": {"s1": relative, "s2": 0.5}}

    result = shared_image_cluster_bootstrap(image_ids, statistic, n_replicates=200, seed=4)
    auroc = result["estimates"]["auroc"]
    relative = result["estimates"]["relative"]
    assert auroc["invalid_replicates"] > 0
    assert auroc["valid_replicates"] + auroc["invalid_replicates"] == 200
    assert auroc["per_seed"]["s1"]["invalid_replicates"] > 0
    assert auroc["per_seed_replicates"]["s1"].dtype == np.float64
    assert np.isnan(auroc["per_seed_replicates"]["s1"]).any()
    assert relative["invalid_replicates"] > 0
    assert np.isnan(relative["replicates"]).any()


def test_too_few_valid_draws_keep_conditional_ci_but_block_gate_use() -> None:
    image_ids = np.arange(8)
    calls = 0

    def statistic(rows: np.ndarray) -> dict[str, dict[str, float]]:
        nonlocal calls
        calls += 1
        value = float(np.mean(rows)) if calls <= 3 else float("nan")
        return {"sparse": {"seed": value}}

    estimate = shared_image_cluster_bootstrap(
        image_ids, statistic, n_replicates=20, seed=0,
    )["estimates"]["sparse"]
    assert estimate["valid_replicates"] == 2
    assert estimate["invalid_replicates"] == 18
    assert estimate["interval_status"] == "AVAILABLE_VALID_REPLICATES"
    assert estimate["uncertainty_status"] == "UNCERTAINTY_INSUFFICIENT"
    assert estimate["ci_conditioning"] == "conditional_on_finite_draws"
    assert estimate["gate_eligible"] is False
    assert estimate["minimum_valid_replicates_for_exploratory_ci"] == 19
    assert estimate["ci_low"] is not None and estimate["ci_high"] is not None


def test_reproducible_and_rejects_malformed_image_cohort() -> None:
    image_ids = np.asarray(["a", "a", "b", "c", "c"])
    values = np.arange(image_ids.size, dtype=np.float64)

    def statistic(rows: np.ndarray) -> dict[str, dict[int, float]]:
        return {"mean": {0: float(np.mean(values[rows]))}}

    first = shared_image_cluster_bootstrap(image_ids, statistic, n_replicates=100, seed=21)
    again = shared_image_cluster_bootstrap(image_ids, statistic, n_replicates=100, seed=21)
    assert np.array_equal(first["estimates"]["mean"]["replicates"], again["estimates"]["mean"]["replicates"])
    assert np.array_equal(
        first["estimates"]["mean"]["per_seed_replicates"][0],
        again["estimates"]["mean"]["per_seed_replicates"][0],
    )
    with pytest.raises(ValueError, match="at least two distinct images"):
        shared_image_cluster_bootstrap(np.asarray([1, 1, 1]), statistic)
    with pytest.raises(ValueError, match="one-dimensional"):
        shared_image_cluster_bootstrap(np.asarray([[1, 2], [3, 4]]), statistic)


def test_relative_dod_and_severity_macro_are_formed_inside_each_seed_draw() -> None:
    image_ids = np.repeat(np.arange(1, 9), 2)
    # Two deliberately different seed scales make mean(seed-relative-effect)
    # differ from a ratio computed after averaging seed-level inputs.
    rng = np.random.default_rng(7)
    base = rng.uniform(1.0, 5.0, image_ids.size)
    score_a = {1: base + rng.uniform(0.1, 1.0, image_ids.size), 2: 2.5 * base + rng.uniform(1.0, 4.0, image_ids.size)}
    score_b = {1: base, 2: 2.5 * base}
    rand = {1: base - rng.uniform(0.1, 1.0, image_ids.size), 2: 2.5 * base - rng.uniform(0.2, 2.0, image_ids.size)}
    hard = {1: base + rng.uniform(0.1, 1.1, image_ids.size), 2: 2.5 * base + rng.uniform(0.4, 2.5, image_ids.size)}
    severity_delta = {
        1: [rng.uniform(-0.2, 0.3, image_ids.size) for _ in range(4)],
        2: [rng.uniform(-0.5, 0.6, image_ids.size) for _ in range(4)],
    }

    def statistic(rows: np.ndarray) -> dict[str, dict[int, float]]:
        rel: dict[int, float] = {}
        dod: dict[int, float] = {}
        macro: dict[int, float] = {}
        for model_seed in (1, 2):
            a = float(np.mean(score_a[model_seed][rows]))
            b = float(np.mean(score_b[model_seed][rows]))
            rel[model_seed] = (a - b) / b if b != 0.0 else float("nan")
            hard_delta = float(np.mean(hard[model_seed][rows] - rand[model_seed][rows]))
            reference_delta = float(np.mean(score_a[model_seed][rows] - score_b[model_seed][rows]))
            dod[model_seed] = hard_delta - reference_delta
            macro[model_seed] = float(
                np.mean([np.mean(level[rows]) for level in severity_delta[model_seed]])
            )
        return {"relative": rel, "difference_of_differences": dod, "severity_macro": macro}

    result = shared_image_cluster_bootstrap(image_ids, statistic, n_replicates=400, seed=13)
    relative = result["estimates"]["relative"]
    assert np.allclose(
        relative["replicates"],
        (relative["per_seed_replicates"][1] + relative["per_seed_replicates"][2]) / 2.0,
    )
    # Re-aggregating numerator/denominator across seeds changes the estimand.
    pooled_seed_ratio = (
        np.mean(score_a[1] + score_a[2]) / 2.0 - np.mean(score_b[1] + score_b[2]) / 2.0
    ) / (np.mean(score_b[1] + score_b[2]) / 2.0)
    assert abs(relative["point"] - pooled_seed_ratio) > 1e-3
    for name in ("difference_of_differences", "severity_macro"):
        estimate = result["estimates"][name]
        assert np.allclose(
            estimate["replicates"],
            (estimate["per_seed_replicates"][1] + estimate["per_seed_replicates"][2]) / 2.0,
        )
        assert estimate["valid_replicates"] == 400
    assert relative["ci_low"] == pytest.approx(np.quantile(relative["replicates"], 0.025))
    assert relative["ci_high"] == pytest.approx(np.quantile(relative["replicates"], 0.975))


def test_legacy_rows_reconstruct_sentence_ids_only_after_ordered_anchor_match() -> None:
    reference = {
        "sentence_id": np.asarray([101, 102, 103, 104]),
        "ref_id": np.asarray([7, 7, 8, 8]),  # ref_id is deliberately repeated
        "image_id": np.asarray([40, 40, 41, 42]),
        "eval_split": np.asarray(["testB", "testB", "testB", "testA"]),
        "K": np.asarray([5, 5, 10, 50]),
        "grounding_correct": np.asarray([1, 0, 1, 0]),
    }
    legacy = {key: value.copy() for key, value in reference.items() if key != "sentence_id"}
    recovered = assert_row_alignment(
        reference,
        legacy,
        required_fields=("ref_id", "image_id", "eval_split", "K", "grounding_correct"),
        reference_name="canonical",
        candidate_name="legacy_csv",
    )
    assert np.array_equal(recovered, reference["sentence_id"])
    assert recovered.size == 4  # repeated ref rows are retained

    shuffled = {key: value.copy() for key, value in legacy.items()}
    shuffled["grounding_correct"][[0, 1]] = shuffled["grounding_correct"][[1, 0]]
    with pytest.raises(ValueError, match="canonical row 0"):
        assert_row_alignment(
            reference,
            shuffled,
            required_fields=("ref_id", "image_id", "eval_split", "K", "grounding_correct"),
            candidate_name="shuffled_legacy_csv",
        )

    wrong_seed_order = {key: value.copy() for key, value in reference.items()}
    wrong_seed_order["sentence_id"][[1, 2]] = wrong_seed_order["sentence_id"][[2, 1]]
    with pytest.raises(ValueError, match="sentence_id"):
        assert_row_alignment(
            reference,
            wrong_seed_order,
            required_fields=("ref_id", "image_id", "eval_split", "K", "grounding_correct"),
            candidate_name="seed_2",
        )

