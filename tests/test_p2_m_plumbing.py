"""V2-P2-M plumbing and estimator-contract tests.

The mechanism diagnostic computes every decision quantity by itself, so the two
places a silent drift could hide are (i) the decision constants and (ii) the
Spearman / cluster-bootstrap estimator.  Every expected value is read from the
pre-result config freeze
(``results/v2_proposal_robustness/p2_m_mechanism_config_freeze.json``) or from
``scipy``, never re-specified here, so the script cannot drift from the frozen
protocol and still pass.

The last three tests are result-layer discipline checks and skip until the
diagnostic has been run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FREEZE = ROOT / "results" / "v2_proposal_robustness" / "p2_m_mechanism_config_freeze.json"
OUT_DIR = ROOT / "results" / "v2_proposal_robustness" / "p2_m_mechanism"
POINT = OUT_DIR / "mechanism_point.csv"
VERDICT = OUT_DIR / "p2_m_verdict.json"
ASSOCIATION = OUT_DIR / "mechanism_association.csv"
STRATA = OUT_DIR / "mechanism_strata.csv"


def _mod():
    import p2_m_mechanism as m

    return m


def _freeze() -> dict:
    return json.loads(FREEZE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# protocol identity
# ---------------------------------------------------------------------------
def test_freeze_is_the_pre_result_descriptive_protocol():
    f = _freeze()
    assert f["protocol"] == "V2-P2-M"
    assert f["authored_before_any_result"] is True
    assert f["classification"] == "DESCRIPTIVE_MECHANISM"
    assert "honesty_clause" in f["decision_labels"]
    # a descriptive protocol may not create/strengthen/weaken an existence claim
    assert "may not create" in f["classification_meaning"]


def test_decision_constants_are_the_frozen_numbers():
    m = _mod()
    f = _freeze()
    sc = f["stop_conditions"]
    assert f"< {m.MIN_COHORT_ROWS} expressions" in sc["cohort"]
    assert f"< {m.MIN_BIN_ROWS} expressions" in sc["strata"]
    assert "> 20%" in sc["unknown_category"] and m.MAX_UNKNOWN_SHARE == 0.20
    assert f">= {m.SHRINKAGE_MIN:.2f}" in f["hypotheses"]["H2_gap_explained_by_composition"]
    assert f"at least {m.RHO_FAMILIES_REQUIRED} of the 3 families" in (
        f["decision_labels"]["MECHANISM_COMPOSITION_LINKED"])
    assert m.N_BINS == 5 and "quintile" in sc["strata"]


def test_a1_grid_is_the_measure_set_the_freeze_named():
    m = _mod()
    a1 = _freeze()["statistics"]["A1_within_family_association"]
    harm = "(" + ", ".join(name.split("_")[0] for name in m.A1_HARM_MEASURES) + ")"
    red = "(" + ", ".join(
        {"R1_same_class_distractors": "R1", "R2_unmatched_fraction": "R2",
         "R4_frac_pairs_gt_mid": "R4-mid", "R4_frac_pairs_gt_high": "R4-high",
         "R5_query_cos_spread": "query_cos_spread"}[name]
        for name in m.REDUNDANCY_MEASURES) + ")"
    assert harm in a1, harm
    assert red in a1, red
    # Class II winner-anchored columns are reported, never decided on
    for name in m.SEM_CLASS_II:
        assert name in _freeze()["measures"]["R3_clip_similarity"]


def test_script_stays_zero_fit_and_forward_pass_free():
    m = _mod()
    src = Path(m.__file__).read_text(encoding="utf-8")
    for forbidden in ("torch", "cuda", "extract_proposals", "materialise_examples",
                      "build_hard_samples", "same_category", "hard_k5"):
        assert forbidden not in src, forbidden
    assert "new_training_parameters" in src and "model_forward_passes" in src


# ---------------------------------------------------------------------------
# estimator contract
# ---------------------------------------------------------------------------
def test_spearman_matches_scipy_with_ties_and_nan():
    m = _mod()
    rng = np.random.default_rng(7)
    for size in (50, 500):
        a = rng.normal(size=size)
        b = rng.normal(size=size)
        assert m.spearman(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-12)
    # the harm measures are discrete, so tie handling is not academic
    x = rng.integers(0, 4, size=400).astype(float)
    y = rng.integers(-1, 2, size=400).astype(float)
    assert m.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)
    # non-finite rows are dropped pairwise, not zero-filled
    xm = np.concatenate([x, [np.nan, np.nan]])
    ym = np.concatenate([y, [1.0, 2.0]])
    assert m.spearman(xm, ym) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)
    assert np.isnan(m.spearman(np.ones(10), rng.normal(size=10)))


def test_rho_block_is_the_same_functional_as_scipy():
    m = _mod()
    rng = np.random.default_rng(11)
    A = rng.normal(size=(3, 200))
    B = np.vstack([rng.normal(size=(1, 200)), rng.integers(0, 5, size=(1, 200)).astype(float)])
    got = m.rho_block(m._rank_std(A), m._rank_std(B))
    want = np.array([[spearmanr(b, a).statistic for a in A] for b in B])
    assert np.allclose(got, want, atol=1e-12)
    assert got.shape == (2, 3)  # [redundancy rows, harm rows]


def _synthetic(n=600, seed=3):
    rng = np.random.default_rng(seed)
    clusters = rng.integers(0, 60, size=n)
    shared = rng.normal(size=n)
    harm_cols = {
        name: [shared + 0.4 * rng.normal(size=n) for _ in (1, 2, 3)]
        for name in _mod().A1_HARM_MEASURES
    }
    red_cols = {
        "R2_unmatched_fraction": shared + 0.6 * rng.normal(size=n),
        "R4_frac_pairs_gt_mid": rng.normal(size=n),
        "R4_frac_pairs_gt_high": rng.normal(size=n),
        "R5_query_cos_spread": shared + 0.6 * rng.normal(size=n),
    }
    return harm_cols, red_cols, clusters


def test_association_group_replicates_reuse_the_point_functional():
    m = _mod()
    harm_cols, red_cols, clusters = _synthetic()
    mask = np.ones(clusters.size, dtype=bool)
    key = "H1c_net_harm|R2_unmatched_fraction"
    point = m.association_group(harm_cols, red_cols, clusters, mask, replicates=0)
    assert np.isnan(point[key]["ci_low"]) and np.isnan(point[key]["ci_high"])
    assert point[key]["rho_mean_of_seeds"] > 0.3  # shared factor is positive
    boot = m.association_group(harm_cols, red_cols, clusters, mask, replicates=200)
    again = m.association_group(harm_cols, red_cols, clusters, mask, replicates=200)
    for est in (boot, again):
        assert est[key]["rho_mean_of_seeds"] == pytest.approx(
            point[key]["rho_mean_of_seeds"], abs=1e-12)
        assert est[key]["ci_low"] < est[key]["rho_mean_of_seeds"] < est[key]["ci_high"]
    # frozen seed => the CI is reproducible bit for bit
    assert boot[key]["ci_low"] == again[key]["ci_low"]
    assert boot[key]["ci_high"] == again[key]["ci_high"]
    assert len(boot[key]["per_seed_rho"]) == 3
    assert boot[key]["n"] == clusters.size


def test_association_group_drops_rows_that_are_not_finite():
    m = _mod()
    harm_cols, red_cols, clusters = _synthetic(n=300)
    full = np.ones(300, dtype=bool)
    r1 = np.linspace(-2.0, 2.0, 300)
    r1[:40] = np.nan
    cells = m.association_group(harm_cols, {m.R1_NAME: r1}, clusters, full, replicates=0)
    key = f"H1c_net_harm|{m.R1_NAME}"
    assert cells[key]["n"] == 260  # the non-finite head is dropped, not imputed
    assert cells[key]["rho_mean_of_seeds"] == pytest.approx(
        np.mean([spearmanr(harm_cols["H1c_net_harm"][s][40:], r1[40:]).statistic
                 for s in (0, 1, 2)]), abs=1e-12)
    # a redundancy column that is one harm column gives rho 1 for that seed, and the
    # mean over seeds stays high because the seeds share the same underlying factor
    perfect = m.association_group(
        harm_cols, {m.R1_NAME: harm_cols["H1c_net_harm"][0]}, clusters, full, replicates=0)
    assert perfect[f"H1c_net_harm|{m.R1_NAME}"]["per_seed_rho"][0] == pytest.approx(1.0)
    assert perfect[f"H1c_net_harm|{m.R1_NAME}"]["rho_mean_of_seeds"] > 0.85
    assert perfect[f"H1c_net_harm|{m.R1_NAME}"]["rho_std_of_seeds"] > 0.0  # seeds differ


# ---------------------------------------------------------------------------
# stratification rule
# ---------------------------------------------------------------------------
def test_stratification_shrinkage_arithmetic_follows_the_freeze():
    m = _mod()
    rng = np.random.default_rng(5)
    n = 4000
    # overlapping location families so that no quintile falls under the bin minimum
    r1 = {f: rng.normal(loc=loc, scale=2.0, size=n)
          for f, loc in zip(m.FAMILIES, (8.0, 9.0, 10.0))}
    # the whole family difference is carried by composition (an R1 location shift),
    # so comparing inside matched R1 strata must remove nearly all of the gap
    harm = {f: 0.09 * r1[f] for f in m.FAMILIES}
    out = m.stratify(r1, harm)
    assert out["matched_gap_H1c"] is not None and out["bins_dropped"] == 0
    assert out["unmatched_gap_H1c"]["GDINO_minus_RPN"] > 0
    assert out["shrinkage"]["GDINO_minus_RPN"] > 0.5
    assert out["shrinkage"]["GDINO_minus_DETR"] > 0.5
    # harm independent of R1 -> matching cannot shrink anything
    flat = {f: np.full(n, off) for f, off in zip(m.FAMILIES, (0.0, 0.1, 0.9))}
    out2 = m.stratify(r1, flat)
    assert out2["shrinkage"]["GDINO_minus_RPN"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# result layer (skips until the diagnostic has been run)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not VERDICT.exists(), reason="P2-M has not been run yet")
def test_verdict_is_descriptive_and_zero_fit():
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    assert v["classification"] == "DESCRIPTIVE_MECHANISM"
    assert v["new_training_parameters"] == 0
    assert v["model_forward_passes"] == 0
    labels = set(_freeze()["decision_labels"]) - {"honesty_clause"}
    assert v["verdict"] in labels
    assert v["bootstrap"]["replicates"] == 5000
    assert v["honesty_clause"] == _freeze()["decision_labels"]["honesty_clause"]
    assert v["out_of_scope_confirmed"] == _freeze()["out_of_scope"]


@pytest.mark.skipif(not (ASSOCIATION.exists() and VERDICT.exists()),
                    reason="P2-M has not been run yet")
def test_association_table_is_the_full_frozen_grid_with_cohorts():
    import csv

    m = _mod()
    cohorts = json.loads(VERDICT.read_text(encoding="utf-8"))["cohorts"]
    with open(ASSOCIATION, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(m.FAMILIES) * len(m.A1_HARM_MEASURES) * len(m.REDUNDANCY_MEASURES)
    for row in rows:
        # the unknown-category stop condition allows at most a 20 % row loss
        assert int(row["n"]) >= int(0.8 * cohorts[row["family"]])
        assert np.isfinite(float(row["rho_mean_of_seeds"]))
        assert np.isfinite(float(row["ci_low"])) and np.isfinite(float(row["ci_high"]))


@pytest.mark.skipif(not (STRATA.exists() and VERDICT.exists()),
                    reason="P2-M has not been run yet")
def test_strata_report_the_drop_and_the_min_bin_rule():
    import csv

    m = _mod()
    with open(STRATA, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == m.N_BINS
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    rule_b = v["rule_b_matched_strata"]
    assert int(rule_b["bins_dropped"]) == sum(1 for r in rows if r["dropped"] == "True")
    if int(rule_b["bins_dropped"]) > 1:
        assert rule_b["matched_gap_H1c"] is None and rule_b["shrinkage"] is None
        assert rule_b["passed"] is False
    else:
        assert rule_b["shrinkage"] is not None


@pytest.mark.skipif(not POINT.exists(), reason="P2-M has not been run yet")
def test_nested_pool_identity_is_visible_in_the_committed_predictions():
    """Nested pools + a per-candidate scorer make H1b structurally zero.

    If a gained flip ever appeared, the K5 -> K50 comparison would no longer be a
    pure pool-growth contrast and H1c would stop being a clean harm measure.
    """
    import csv

    m = _mod()
    with open(POINT, newline="", encoding="utf-8") as fh:
        means = {(r["family"], r["measure"]): float(r["mean"]) for r in csv.DictReader(fh)}
    for family in m.FAMILIES:
        assert means[(family, "H1b_gained_flip")] == 0.0
        assert means[(family, "H1c_net_harm")] == pytest.approx(
            means[(family, "H1a_lost_flip")], abs=1e-12)
        assert means[(family, "H1c_net_harm")] == pytest.approx(
            means[(family, "accuracy_K5")] - means[(family, "accuracy_K50")], abs=1e-9)


@pytest.mark.skipif(not (ASSOCIATION.exists() and VERDICT.exists()),
                    reason="P2-M has not been run yet")
def test_rule_a_is_recomputable_from_the_association_table():
    """The gate cannot be detached from the numbers it claims to read."""
    import csv

    m = _mod()
    v = json.loads(VERDICT.read_text(encoding="utf-8"))
    with open(ASSOCIATION, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["harm"] == "H1c_net_harm" and r["redundancy"] == m.R1_NAME]
    assert len(rows) == len(m.FAMILIES)
    passing = [r["family"] for r in rows
               if float(r["rho_mean_of_seeds"]) > 0 and float(r["ci_low"]) > 0]
    rule = v["rule_a_within_family_rho"]
    assert passing == rule["families_passing"]
    assert rule["passed"] is (len(passing) >= m.RHO_FAMILIES_REQUIRED)
    for row in rows:
        cell = rule["per_family"][row["family"]]
        assert cell["rho_mean_of_seeds"] == pytest.approx(
            float(row["rho_mean_of_seeds"]), abs=1e-9)
        assert cell["n"] == int(row["n"])
