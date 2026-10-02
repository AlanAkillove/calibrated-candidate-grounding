"""RQ4-M1 transition-confidence decomposition tests (protocol RQ4-M1, freeze round).

Two tiers, mirroring ``results/v2_rq4_mechanism/m1_transition_confidence/protocol_freeze.json``
section ``tests``:

* **algebra tier** - runs anywhere, on synthetic arrays only.  It never opens a frozen
  RPN / DETR / GDINO prediction file, so it is legal in the freeze round, before any
  RQ4-M1 number exists (``freeze_round_rule``).  It checks that the two frozen identities
  are *exact*, that the ``G`` group invariant really fails loudly, that ties use the
  sklearn 0.5 convention, that shares are never clipped, and that the primary path
  imports nothing that could fit or forward anything.
* **artifact tier** - reads the written RQ4-M1 CSV/JSON artifacts and compares them
  against the already-published V2-P C1 artifacts.  It is skipped while no result exists
  and never recomputes a frozen row: the reproduction check it enforces is the one the
  driver performs at computation time (``StopCondition``), verified here traceably.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import sys
import tokenize
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _extra in (_ROOT / "src", _ROOT / "scripts", _ROOT):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.metrics.selective import selective_accuracy  # noqa: E402
from ccg.rq4 import decomposition as dec  # noqa: E402

import rq4_m1_decomposition as drv  # noqa: E402  (scripts/ driver, freeze-gated)

RESULT_DIR = _ROOT / "results" / "v2_rq4_mechanism" / "m1_transition_confidence"
FREEZE = RESULT_DIR / "protocol_freeze.json"
C1_POINT = _ROOT / "results" / "v2_proposal_robustness" / "p2_c1_gdino" / "c1_point.csv"

ALGEBRA_TOLERANCE = dec.IDENTITY_TOLERANCE
FORBIDDEN_VERDICT_LABELS = ("GO", "NO_GO", "SUCCESS", "FAIL", "MECHANISM_FOUND",
                            "MECHANISM_PROVED", "PREREGISTERED_HYPOTHESIS_CONFIRMED")
#: previously observed candidate properties: allowed only as exclusion declarations,
#: never as a quantity the primary decomposition reads (§15 of the user protocol).
EXCLUDED_MECHANISM_TOKENS = ("same_class", "unmatched_fraction", "geometric_redundancy",
                             "query_cos_spread", "h2a", "delta_margin", "tail_pressure")
#: nothing in the primary path may touch a model, a trainer or a calibration fit.
FORBIDDEN_IMPORT_ROOTS = ("torch", "transformers", "open_clip", "h5py", "PIL", "ccg.data",
                          "ccg.features", "ccg.models", "ccg.reliability", "ccg.lcr")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def nested_cohort(seed: int, n: int = 1200, base_rate: float = 0.78,
                  flip_rate: float = 0.3, separate_scores: bool = True):
    """A synthetic aligned cohort that satisfies the nested-K structure (``G == 0``)."""
    rng = np.random.default_rng(seed)
    r5 = (rng.random(n) < base_rate).astype(np.float64)
    pos = np.nonzero(r5 == 1.0)[0]
    flip = rng.random(pos.shape[0]) < flip_rate
    r50 = r5.copy()
    r50[pos[flip]] = 0.0
    p5 = np.clip(rng.normal(0.74, 0.16, n), 1e-4, 1.0) * (0.55 + 0.45 * r5)
    if separate_scores:
        p50 = np.clip(rng.normal(0.60, 0.21, n), 1e-4, 1.0) * (0.55 + 0.45 * r50)
    else:
        p50 = p5.copy()
    return {"r5": r5, "r50": r50, "p5": p5, "p50": p50,
            "image_id": rng.integers(0, 180, n).astype(np.int64)}


def _close(a: float, b: float, tol: float = ALGEBRA_TOLERANCE) -> bool:
    return abs(float(a) - float(b)) <= tol


def _module_identifiers(path: Path) -> set:
    """Every ``Name``/attribute token used as code in *path*, string literals excluded."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            used.add(node.attr.lower())
        elif isinstance(node, ast.Import):
            for alias in node.names:
                used.add(alias.name.lower())
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                used.add(node.module.lower())
            for alias in node.names:
                used.add(alias.name.lower())
    return used


def _doc_slice(path: Path, marker: str, end_marker: str | None = None) -> str:
    """The appended record only: from the first occurrence of *marker*, stopping at
    *end_marker* when given (the freeze records and the later result records are separate
    appended sections, and a result record is allowed to carry result numbers)."""
    if not path.exists():
        pytest.skip(f"{path.name} absent")
    text = path.read_text(encoding="utf-8")
    index = text.find(marker)
    if index < 0:
        return ""
    if end_marker is not None:
        end = text.find(end_marker, index + len(marker))
        return text[index:] if end < 0 else text[index:end]
    return text[index:]


#: the append-only boundary between the RQ4-M1 freeze record and the later result record
LOG_RESULT_RECORD = "# ===== RQ4-M1 result record"


def _module_string_constants(path: Path) -> list:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)]


# ===========================================================================
# algebra tier - transition groups and the structural invariant
# ===========================================================================
def test_transition_groups_are_exhaustive_and_disjoint():
    for seed in (1, 2, 3, 4):
        arr = nested_cohort(seed)
        groups = dec.transition_groups(arr["r5"], arr["r50"])
        counts = groups.counts
        assert counts["S"] + counts["F"] + counts["E"] + counts["G"] == groups.n
        masks = [groups.stable_correct, groups.lost, groups.stable_error, groups.gained]
        stacked = np.stack(masks)
        assert np.array_equal(stacked.sum(axis=0), np.ones(groups.n, dtype=np.int64))
        pair_overlap = [(m1 & m2).sum() for m1 in masks for m2 in masks if m1 is not m2]
        assert all(int(v) == 0 for v in pair_overlap)


def test_gained_group_is_exactly_zero_on_nested_cohorts():
    arr = nested_cohort(11)
    groups = dec.transition_groups(arr["r5"], arr["r50"])
    assert groups.counts["G"] == 0
    report = dec.check_structural_invariant(groups, family="SYNTH", seed="seed1")
    assert report["gained_zero"] is True
    assert report["partition_exhaustive"] is True


def test_structural_invariant_failure_raises_and_names_the_family():
    r5 = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    r50 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])          # row 1 gained correctness -> G = 1
    groups = dec.transition_groups(r5, r50)
    assert groups.counts["G"] == 1
    with pytest.raises(dec.StructuralInvariantError, match="STRUCTURAL_INVARIANT_FAILURE"):
        dec.check_structural_invariant(groups, family="GDINO", seed="seed1")
    p5 = np.array([0.9, 0.8, 0.3, 0.7, 0.2])
    with pytest.raises(dec.StructuralInvariantError):
        dec.group_weight_reconstruction(p5, r5, p5, r50, groups)


def test_groups_reject_non_binary_or_misaligned_labels():
    arr = nested_cohort(3)
    with pytest.raises(ValueError):
        dec.transition_groups(arr["r5"], arr["r50"] * 0.5)
    with pytest.raises(ValueError):
        dec.transition_groups(arr["r5"], arr["r50"][:-1])
    with pytest.raises(ValueError):
        dec.transition_groups(arr["r5"], np.full_like(arr["r5"], np.nan))


# ===========================================================================
# algebra tier - AUROC corners, ties and the Shapley identity
# ===========================================================================
def test_corner_aurocs_match_direct_auroc_correct_calls():
    arr = nested_cohort(21)
    a00, a10, a01, a11 = dec.corner_aurocs(arr["p5"], arr["r5"], arr["p50"], arr["r50"])
    assert _close(a00, float(auroc_correct(arr["p5"], arr["r5"])))
    assert _close(a10, float(auroc_correct(arr["p5"], arr["r50"])))
    assert _close(a01, float(auroc_correct(arr["p50"], arr["r5"])))
    assert _close(a11, float(auroc_correct(arr["p50"], arr["r50"])))


def test_shapley_identity_is_exact_over_many_random_cohorts():
    for seed in range(12):
        arr = nested_cohort(seed, n=400 + 37 * seed, flip_rate=0.05 + 0.07 * seed)
        point = dec.decompose(arr["p5"], arr["r5"], arr["p50"], arr["r50"])
        assert abs(point.shapley_residual) <= ALGEBRA_TOLERANCE, point.shapley_residual
        assert _close(point.d_label + point.d_conf, point.d_total)
        assert _close(point.d_total, point.a00 - point.a11)
        assert _close(point.delta_total, -point.d_total)
        assert _close(point.d_label, -0.5 * ((point.a10 - point.a00) + (point.a11 - point.a01)))
        assert _close(point.d_conf, -0.5 * ((point.a01 - point.a00) + (point.a11 - point.a10)))


def test_single_class_and_misaligned_inputs_are_not_silently_dropped():
    flat = np.full(40, 0.5)
    all_correct = np.ones(40)
    # a single-class corner has no ROC curve: the frozen estimator answers nan, and the
    # decomposition refuses to build a Shapley split out of it
    a00 = dec.corner_aurocs(flat, all_correct, flat, all_correct)[0]
    assert math.isnan(a00)
    with pytest.raises(ValueError):
        dec.shapley_decomposition(0.8, 0.7, 0.6, float("nan"))
    with pytest.raises(ValueError):
        dec.corner_aurocs(flat, all_correct, flat[:-1], all_correct[:-1])
    with pytest.raises(ValueError):
        dec.group_weight_reconstruction(flat, np.arange(40) % 2, flat, np.ones(40) * 0.5)


def test_group_weight_reconstruction_is_exact_for_all_four_corners():
    for seed in (31, 32, 33):
        arr = nested_cohort(seed, n=900)
        groups = dec.transition_groups(arr["r5"], arr["r50"])
        recon = dec.group_weight_reconstruction(arr["p5"], arr["r5"], arr["p50"], arr["r50"],
                                                groups)
        assert recon["max_abs_residual"] <= ALGEBRA_TOLERANCE, recon["residual"]
        for corner in ("A00", "A01", "A10", "A11"):
            assert _close(recon["reconstructed"][corner], recon["direct"][corner])


def test_reconstruction_uses_the_sklearn_half_tie_convention():
    # every score tied: every pairwise AUC must be exactly 0.5, so every corner is 0.5
    p5 = np.full(60, 0.4)
    p50 = np.full(60, 0.4)
    r5 = (np.arange(60) < 40).astype(np.float64)
    r50 = (np.arange(60) < 20).astype(np.float64)
    recon = dec.group_weight_reconstruction(p5, r5, p50, r50)
    for corner in ("A00", "A01", "A10", "A11"):
        assert _close(recon["reconstructed"][corner], 0.5)
    assert _close(recon["pairwise"]["AUC_SF_p50"], 0.5)


def test_pairwise_auc_equals_auroc_correct_on_the_same_two_groups():
    rng = np.random.default_rng(7)
    pos = rng.normal(0.7, 0.1, 200)
    neg = rng.normal(0.5, 0.1, 150)
    pooled = float(auroc_correct(np.concatenate((pos, neg)),
                                 np.concatenate((np.ones(200), np.zeros(150)))))
    assert _close(dec.pairwise_auc(pos, neg), pooled)
    tied = np.concatenate((np.full(50, 0.3), np.full(50, 0.3)))
    assert _close(dec.pairwise_auc(tied[:50], tied[50:]), 0.5, tol=1e-15)
    with pytest.raises(ValueError):                     # an empty group is never a silent nan
        dec.pairwise_auc(np.array([1.0]), np.array([]))


# ===========================================================================
# algebra tier - sign policy, shares, selective composition, gap identity
# ===========================================================================
def test_components_may_be_negative_or_exceed_total_and_are_never_clipped():
    # confidence improves (D_conf < 0) while labels degrade more: D_label > D_total > 0
    a00, a10, a01, a11 = 0.80, 0.70, 0.83, 0.75
    point = dec.shapley_decomposition(a00, a10, a01, a11)
    assert point.d_conf < 0.0
    assert point.d_label > point.d_total > 0.0
    assert abs(point.shapley_residual) <= ALGEBRA_TOLERANCE
    shares = point.shares(floor=1e-12)
    assert shares["share_conf"] is not None and shares["share_conf"] < 0.0
    assert shares["share_label"] > 1.0
    assert _close(shares["share_label"] + shares["share_conf"], 1.0, tol=1e-12)


def test_signed_share_suppresses_a_degenerate_denominator_only():
    assert dec.signed_share(0.01, 0.02, floor=1e-3) == pytest.approx(0.5)
    assert dec.signed_share(0.01, 5e-4, floor=1e-3) is None
    assert dec.signed_share(-0.2, 0.4, floor=1e-3) == pytest.approx(-0.5)
    assert dec.signed_share(0.1, float("nan"), floor=1e-3) is None


def test_decomposition_shares_follow_the_same_denominator_rule():
    arr = nested_cohort(41)
    point = dec.decompose(arr["p5"], arr["r5"], arr["p50"], arr["r50"])
    shares = point.shares()
    assert shares["share_denominator_ok"] is True          # a real degradation is present
    assert _close(shares["share_label"] + shares["share_conf"], 1.0, tol=1e-12)
    flat = dec.shapley_decomposition(0.8, 0.8 - 1e-6, 0.8, 0.8 - 2e-6)
    empty = flat.shares()
    assert empty["share_label"] is None and empty["share_conf"] is None


def test_selective_error_sources_reproduce_the_frozen_acceptance_convention():
    arr = nested_cohort(51, n=777)
    groups = dec.transition_groups(arr["r5"], arr["r50"])
    rows = dec.selective_error_sources(arr["p50"], groups, coverage_levels=(0.5, 0.8))
    assert [r["coverage"] for r in rows] == [0.5, 0.8]
    for row in rows:
        n_keep = row["n_accepted"]
        assert n_keep == max(1, math.ceil(row["coverage"] * groups.n))
        assert row["accepted_S"] + row["accepted_F_new_errors"] + \
            row["accepted_E_persistent_errors"] + row["accepted_G"] == n_keep
        assert row["accepted_G"] == 0
        errors = row["accepted_F_new_errors"] + row["accepted_E_persistent_errors"]
        assert errors == n_keep - row["accepted_S"]
        expected = 1.0 - selective_accuracy(arr["p50"], arr["r50"], row["coverage"])
        assert _close(row["risk"], expected, tol=1e-12)
        if errors:
            assert _close(row["fraction_of_accepted_errors_from_F"] +
                          row["fraction_of_accepted_errors_from_E"], 1.0)
    with pytest.raises(ValueError):
        dec.selective_error_sources(arr["p50"], groups, coverage_levels=(0.0,))


def test_cross_family_gap_identity_is_exact_and_antisymmetric():
    left = dec.shapley_decomposition(0.84, 0.79, 0.83, 0.59)
    right = dec.shapley_decomposition(0.83, 0.80, 0.82, 0.79)
    gap = dec.cross_family_gap("GDINO", left.as_dict(), "RPN", right.as_dict())
    assert abs(gap["gap_residual"]) <= ALGEBRA_TOLERANCE
    assert _close(gap["gap_total_degradation"],
                  gap["gap_label_component"] + gap["gap_conf_component"])
    assert _close(gap["gap_total_degradation"], left.d_total - right.d_total)
    flipped = dec.cross_family_gap("RPN", right.as_dict(), "GDINO", left.as_dict())
    for key in ("gap_total_degradation", "gap_label_component", "gap_conf_component"):
        assert _close(gap[key], -flipped[key])
    assert gap["heavier_component"] != "GO"


def test_component_ordering_is_descriptive_only():
    assert dec.component_ordering(dec.shapley_decomposition(0.8, 0.7, 0.79, 0.74)) == \
        "TRANSITION_HEAVIER"
    assert dec.component_ordering(dec.shapley_decomposition(0.8, 0.79, 0.72, 0.71)) == \
        "CONFIDENCE_CHANGE_HEAVIER"
    tie = dec.shapley_decomposition(0.8, 0.75, 0.75, 0.70)
    assert dec.component_ordering(tie) == "EXACT_TIE"


# ===========================================================================
# algebra tier - bootstrap design
# ===========================================================================
def test_bootstrap_resamples_whole_image_clusters_and_is_deterministic():
    arr = nested_cohort(61, n=800)
    captured = []

    def statistic(draw):
        captured.append(np.asarray(draw).copy())
        point = dec.decompose(arr["p5"][draw], arr["r5"][draw],
                              arr["p50"][draw], arr["r50"][draw])
        return {"D_total": point.d_total, "D_label": point.d_label, "D_conf": point.d_conf}

    out = dec.image_cluster_bootstrap(statistic, arr["image_id"], n_replicates=25, seed=0)
    full = statistic(np.arange(arr["r5"].shape[0]))
    assert _close(out["D_total"]["point"], full["D_total"])
    assert out["D_total"]["resample_unit"] == "image"
    assert out["D_total"]["n_replicates"] == 25
    assert out["D_total"]["seed"] == 0
    assert out["D_total"]["ci_low"] <= out["D_total"]["ci_high"]

    cluster_size = np.bincount(arr["image_id"])
    for draw in captured[1:]:
        counts = np.bincount(arr["image_id"][draw], minlength=cluster_size.shape[0])
        present = counts > 0
        assert np.all(counts[present] % cluster_size[present] == 0)   # whole clusters only
        assert draw.shape[0] > 0
        assert len(np.unique(arr["image_id"][draw])) <= out["D_total"]["n_clusters"]

    again = dec.image_cluster_bootstrap(statistic, arr["image_id"], n_replicates=25, seed=0)
    assert _close(again["D_total"]["ci_low"], out["D_total"]["ci_low"])
    assert _close(again["D_total"]["ci_high"], out["D_total"]["ci_high"])


def test_bootstrap_carries_every_input_of_a_family_on_one_shared_draw():
    arr = nested_cohort(62, n=600)
    seen = []

    def statistic(draw):
        idx = np.asarray(draw)
        seen.append((idx.shape[0],))
        point = dec.decompose(arr["p5"][idx], arr["r5"][idx], arr["p50"][idx], arr["r50"][idx])
        groups = dec.transition_groups(arr["r5"][idx], arr["r50"][idx])
        recon = dec.group_weight_reconstruction(arr["p5"][idx], arr["r5"][idx],
                                                arr["p50"][idx], arr["r50"][idx], groups)
        return {"D_total": point.d_total, "max_abs_residual": recon["max_abs_residual"],
                "G": float(groups.counts["G"])}

    out = dec.image_cluster_bootstrap(statistic, arr["image_id"], n_replicates=15, seed=0,
                                      keep_replicates=True)
    reps = out["_replicates"]
    assert reps["G"].shape[0] == 15
    assert np.all(reps["G"] == 0.0)                       # the invariant holds per replicate
    assert np.all(np.abs(reps["max_abs_residual"]) <= ALGEBRA_TOLERANCE)


def test_aligned_cross_family_gap_ci_needs_matching_replicate_shapes():
    a = {"D_total": np.arange(10, dtype=np.float64)}
    b = {"D_total": np.arange(10, dtype=np.float64) * 0.5}
    ci = dec.aligned_cross_family_gap_ci(a, b, key="D_total")
    assert ci["gap_mean"] == pytest.approx(2.25)
    assert ci["gap_ci_low"] <= ci["gap_mean"] <= ci["gap_ci_high"]
    with pytest.raises(ValueError):
        dec.aligned_cross_family_gap_ci(a, {"D_total": np.arange(4)}, key="D_total")


# ===========================================================================
# algebra tier - driver plumbing on synthetic rows (never a frozen row)
# ===========================================================================
def test_driver_synthetic_selftest_passes_and_writes_nothing(monkeypatch, tmp_path):
    ghost = tmp_path / "must_not_be_created"
    monkeypatch.setattr(drv, "OUT_DIR", ghost)
    args = SimpleNamespace(n=900, reps=12)
    report = drv.run_synthetic_selftest(args)
    assert report["classification"] == "DESCRIPTIVE_MECHANISM_DECOMPOSITION"
    assert report["identities_within_tolerance"] is True
    assert report["matched_classification"] == "SECONDARY_MATCHED_DIAGNOSTIC"
    assert report["max_abs_group_reconstruction_residual"] <= ALGEBRA_TOLERANCE
    assert not ghost.exists(), "the freeze-round smoke wrote into the results directory"


def test_driver_reproduction_gate_fires_when_a_published_corner_is_off():
    arrays = drv.synthetic_arrays(n=800, seed=11)
    ref = drv.synthetic_reference(arrays, "SYNTH")
    key = ("SYNTH", "seed1", int(drv.core.K_BASELINE))
    ref[key] = dict(ref[key], auroc_correct=ref[key]["auroc_correct"] - 1e-6)
    with pytest.raises(drv.StopCondition, match="does not match"):
        drv.analyze_family("SYNTH", arrays, ref)


def test_matched_intersection_is_deterministic_and_antisymmetric():
    arrays = {"SYNTH": drv.synthetic_arrays(n=800, seed=11),
              "SYNTH2": drv.synthetic_arrays(n=800, seed=23, flip_rate=0.15)}
    ids = {name: {int(v) for v in per_seed[0]["sentence_id"]}
           for name, per_seed in arrays.items()}
    first = drv.matched_pair("SYNTH", "SYNTH2", ids, arrays, replicates=8)
    second = drv.matched_pair("SYNTH", "SYNTH2", ids, arrays, replicates=8)
    assert json.dumps(first["family_mean"], sort_keys=True, default=float) == \
        json.dumps(second["family_mean"], sort_keys=True, default=float)
    flipped = drv.matched_pair("SYNTH2", "SYNTH", ids, arrays, replicates=8)
    assert first["n_expressions"] == flipped["n_expressions"]
    assert first["n_expressions"] == 800                    # synthetic cohorts coincide
    columns = {"D_total": "gap_total_degradation", "D_label": "gap_label_component",
               "D_conf": "gap_conf_component"}
    for key, column in columns.items():
        assert _close(first["gap"][column], -flipped["gap"][column], tol=1e-12), key
    for name in ("SYNTH", "SYNTH2"):
        assert _close(first["family_mean"][name]["D_total"],
                      flipped["family_mean"][name]["D_total"])
    assert first["classification"] == "SECONDARY_MATCHED_DIAGNOSTIC"
    assert "SHARED image-cluster draws" in first["design"]


# ===========================================================================
# algebra tier - static freeze guarantees
# ===========================================================================
def test_primary_modules_import_no_model_data_or_fit_path():
    for path in (drv.ROOT / "src" / "ccg" / "rq4" / "decomposition.py", drv.__file__):
        used = _module_identifiers(path)
        for forbidden in FORBIDDEN_IMPORT_ROOTS:
            assert not any(token.startswith(forbidden.lower()) for token in used), \
                f"{path.name} references {forbidden!r}"
        for token in ("fit_is_forbidden", "forward", "train", "softmax", "temperature_fit"):
            assert token not in used, f"{path.name} calls {token!r}"


def test_excluded_mechanism_variables_never_enter_the_primary_computation():
    module = drv.ROOT / "src" / "ccg" / "rq4" / "decomposition.py"
    for token in EXCLUDED_MECHANISM_TOKENS:
        assert token not in _module_identifiers(module), f"{token!r} used in the algebra module"
    # in the driver the tokens may only appear inside declared exclusion strings
    for text in _module_string_constants(drv.__file__):
        for token in EXCLUDED_MECHANISM_TOKENS:
            if token in text.lower():
                assert ("no R1" in text or "not reported" in text or "NOT USED" in text
                        or "duplicate-of-existing" in text or "forbidden" in text.lower()
                        or "no added-candidate" in text), f"undeclared use in {text[:80]!r}"


def test_freeze_round_gates_the_real_data_path():
    with pytest.raises(drv.FreezeViolation):
        drv.run_real(SimpleNamespace(allow_real_data=False))
    src = Path(drv.__file__).read_text(encoding="utf-8")
    assert "allow_real_data" in src and "ls-files" in src
    with Path(drv.__file__).open(encoding="utf-8") as handle:
        tokens = {tok.string for tok in tokenize.generate_tokens(handle.readline)
                  if tok.type == tokenize.STRING}
    assert any("--allow-real-data" in text for text in tokens)


def test_algebra_module_is_pure_and_reads_no_files():
    module = drv.ROOT / "src" / "ccg" / "rq4" / "decomposition.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    allowed = {"math", "dataclasses", "typing", "numpy", "numpy.typing",
               "ccg.metrics.discrimination", "ccg.experiment.phase0a", "__future__"}
    assert imports <= allowed, sorted(imports - allowed)
    assert "ccg.metrics.discrimination" in imports         # the single frozen AUROC primitive
    assert "ccg.experiment.phase0a" in imports              # the frozen cluster sampler
    assert not any(node.lineno for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id in ("open", "loadtxt", "save"))


# ===========================================================================
# freeze tier - the protocol file itself
# ===========================================================================
def test_freeze_file_exists_and_declares_pre_result_authorship():
    freeze = drv.load_freeze()
    assert freeze["protocol"] == "RQ4-M1"
    assert freeze["authored_before_any_result"] is True
    assert freeze["classification"] == "DESCRIPTIVE_MECHANISM_DECOMPOSITION"
    assert "causal mechanism proof" in json.dumps(freeze["not"])
    assert freeze["data"]["new_parameters"] == 0
    assert freeze["data"]["gpu_required"] is False


def test_freeze_constants_match_the_code():
    freeze = drv.load_freeze()
    assert "1e-12" in freeze["decomposition"]["residual_rule"]
    assert ALGEBRA_TOLERANCE == 1e-12
    safety = freeze["decomposition"]["shares"]["denominator_safety"]
    floor = float(safety.split("|D_total| < ")[1].split()[0].rstrip("."))
    assert floor == dec.SHARE_DENOM_FLOOR
    boot = freeze["statistics"]["bootstrap"]
    assert int(boot["replicates"]) == 5000 and int(boot["seed"]) == 0
    assert boot["unit"].startswith("image")
    assert boot["sampler"].startswith("ccg.experiment.phase0a")
    assert boot["ci"].startswith("95%")
    assert drv.COVERAGES == tuple(freeze["selective_error_sources"]["coverages"])
    assert drv.FAMILIES == tuple(freeze["data"]["families"])
    assert drv.SEEDS == tuple(freeze["data"]["seeds"])
    assert set(freeze["verdict"]["forbidden_labels"]) == set(FORBIDDEN_VERDICT_LABELS)


def test_freeze_formula_tokens_are_the_ones_implemented():
    formulas = drv.load_freeze()["decomposition"]["formulas"]
    assert formulas["L"].replace(" ", "") == "0.5*((A10-A00)+(A11-A01))"
    assert formulas["C"].replace(" ", "") == "0.5*((A01-A00)+(A11-A10))"
    assert "L + C == A11 - A00" in formulas["identity"]
    assert formulas["D_label"] == "-L" and formulas["D_conf"] == "-C"


def test_freeze_declares_three_figures_and_the_full_artifact_set():
    outputs = drv.load_freeze()["outputs"]
    assert len(outputs["figures_exactly_three"]) == 3
    required = {"point_decomposition.csv", "bootstrap_decomposition.csv",
                "transition_groups.csv", "pairwise_auc_components.csv",
                "selective_error_sources.csv", "cross_family_gap_decomposition.csv",
                "matched_intersection_decomposition.csv", "verdict.json", "metadata.json",
                "protocol_freeze.json", "input_artifact_manifest.csv"}
    listed = " ".join(outputs["files"])
    assert required.issubset({name for name in required if name in listed})


# ===========================================================================
# freeze tier - the frozen input-artifact manifest (hashes only, no row payload)
# ===========================================================================
@pytest.fixture(scope="module")
def manifest_rows_read():
    import csv as _csv
    if not drv.MANIFEST_PATH.exists():
        pytest.skip("input_artifact_manifest.csv not written yet")
    with drv.MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        return list(_csv.DictReader(handle))


def test_input_manifest_registers_exactly_18_prediction_artifacts(manifest_rows_read):
    prediction = [r for r in manifest_rows_read if r["role"] == drv.ROLE_PREDICTION]
    assert len(prediction) == len(drv.FAMILIES) * len(drv.SEEDS) * 2 == 18
    keys = {(r["family"], int(float(r["K"])), r["seed"]) for r in prediction}
    expected = {(family, int(k), f"seed{seed}")
                for family in drv.FAMILIES
                for k in (drv.core.K_BASELINE, drv.core.PRIMARY_KB)
                for seed in drv.SEEDS}
    assert keys == expected
    paths = {r["path"] for r in prediction}
    assert all(p.startswith("results/v2_proposal_robustness/predictions/") for p in paths)
    assert len(paths) == 18                                    # no file listed twice
    reference = [r for r in manifest_rows_read if r["role"] == drv.ROLE_REFERENCE]
    assert [r["path"] for r in reference] == \
        [drv.POINT_REF.relative_to(drv.ROOT).as_posix()]


def test_input_manifest_hash_fields_are_nonempty_and_sizes_match(manifest_rows_read):
    for row in manifest_rows_read:
        assert len(row["sha256"]) == 64, row
        assert all(c in "0123456789abcdef" for c in row["sha256"]), row
        path = drv.ROOT / row["path"]
        assert path.exists(), row["path"]
        assert int(row["file_size"]) == path.stat().st_size, row
        assert row["row_count_source"], row                     # provenance always stated


def test_protocol_input_manifest_sha256_matches_the_committed_file(manifest_rows_read):
    freeze = drv.load_freeze()
    assert freeze["input_manifest_sha256"] == drv._sha256(drv.MANIFEST_PATH)
    assert freeze["input_artifact_manifest"]["path"] == \
        drv.MANIFEST_PATH.relative_to(drv.ROOT).as_posix()
    assert "exactly these hashed artifacts" in \
        freeze["input_artifact_manifest"]["consumption_rule"]


def test_verify_input_manifest_passes_on_the_frozen_artifacts():
    freeze = drv.load_freeze()
    report = drv.verify_input_manifest(freeze)
    assert report["n_prediction_verified"] == 18
    assert report["n_verified"] == 19
    assert report["manifest_sha256"] == freeze["input_manifest_sha256"]


def test_verify_input_manifest_refuses_a_missing_or_rehashed_artifact(tmp_path, monkeypatch):
    freeze = drv.load_freeze()
    monkeypatch.setattr(drv, "MANIFEST_PATH", tmp_path / "absent.csv")
    with pytest.raises(drv.FreezeViolation):
        drv.verify_input_manifest(freeze)


def test_tail_pressure_is_not_used_because_its_inputs_are_absent():
    tp = drv.load_freeze()["excluded_from_primary"]["added_candidate_tail_pressure"]
    reason = " ".join(str(v) for v in tp.values()).lower()
    assert tp["decision"].startswith("NOT USED")
    assert "per-candidate" in tp["reason"]
    assert "new model forward" in tp["reason"]
    assert "forbidden" in tp["reason"]
    assert "raw_top1" in tp["evidence"] and "raw_margin12" in tp["evidence"]
    # inability to compute is not recorded as duplication
    assert tp["duplicate_of_existing_claim"].startswith("NOT MADE")
    assert "not made" in reason
    drv_status = " ".join(str(v) for v in tp.values())
    assert "duplicate-of-existing" not in drv_status.replace("NOT MADE", "")


def test_freeze_and_docs_records_carry_the_freeze_classification_and_zero_delta_attestations():
    freeze = drv.load_freeze()
    assert freeze["classification"] == "DESCRIPTIVE_MECHANISM_DECOMPOSITION"
    assert freeze["real_results_seen_before_freeze"] is False
    assert freeze["authored_before_any_result"] is True
    data = freeze["data"]
    assert data["new_parameters"] == 0 and data["gpu_required"] is False
    assert data["new_training_parameters"] == 0
    assert data["new_model_forward"] == 0
    assert data["new_features"] == 0
    records = {
        "protocol_freeze.json": FREEZE.read_text(encoding="utf-8"),
        "research_protocol.md#RQ4-A1": _doc_slice(
            _ROOT / "docs" / "research_protocol.md", "RQ4-A1"),
        "experiment_log.md#RQ4-M1": _doc_slice(
            _ROOT / "docs" / "experiment_log.md", "RQ4-M1（", LOG_RESULT_RECORD),
    }
    for name, text in records.items():
        assert text, f"{name}: record not found"
        assert "DESCRIPTIVE_MECHANISM_DECOMPOSITION" in text, name
        for token in ("real_results_seen_before_freeze",
                      "new_training_parameters", "new_model_forward", "new_features"):
            assert token in text, f"{name} lacks the {token} attestation"
        # the attestation must state the value, not merely name the key
        assert re.search(r"real_results_seen_before_freeze[\"']?\s*[:=]\s*false", text,
                         re.IGNORECASE), f"{name} does not assert real_results_seen_before_freeze"
        for token in ("new_training_parameters", "new_model_forward", "new_features"):
            assert re.search(rf"{token}[\"']?\s*[:=]\s*0", text), \
                f"{name} does not state {token}: 0"


def test_freeze_documents_hold_no_real_decomposition_numbers():
    pattern = re.compile(r"D_(?:total|label|conf)\s*[=:]\s*[-+]?0?\.\d")
    scanned = {
        "protocol_freeze.json": FREEZE.read_text(encoding="utf-8"),
        "research_protocol.md": _doc_slice(
            _ROOT / "docs" / "research_protocol.md", "RQ4-A1"),
        # only the pre-result freeze record: the result record registered afterwards is
        # supposed to carry these numbers, so the scan must stop at its append-only boundary
        "experiment_log.md": _doc_slice(
            _ROOT / "docs" / "experiment_log.md", "RQ4-M1（", LOG_RESULT_RECORD),
    }
    assert all(scanned.values()), "a freeze record slice came back empty"
    for name, text in scanned.items():
        hits = pattern.findall(text)
        assert not hits, f"{name} already contains a computed degradation number: {hits[:3]}"
    # once the result round has run, the same records must still not carry any of its numbers:
    # that is the durable form of the freeze-round guarantee (the docs were written blind and
    # were never back-filled after the results existed)
    point = RESULT_DIR / "point_decomposition.csv"
    if not point.exists():
        return
    rendered = set()
    with point.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("D_total", "D_label", "D_conf"):
                value = row.get(key, "")
                if value in ("", None):
                    continue
                rendered.add(str(value))
                rendered.add(f"{float(value):+.6f}")
                rendered.add(f"{float(value):.6f}")
    assert rendered, "the result file carried no decomposition number"
    for name, text in scanned.items():
        leaked = {value for value in rendered if value in text}
        assert not leaked, f"{name} was back-filled with result numbers: {sorted(leaked)[:3]}"


# ===========================================================================
# rendering tier - the figure / report / CSV writers, on synthetic arrays only
# ===========================================================================
#: the real run crashed in a matplotlib call once; these tests keep the whole rendering
#: layer executable without touching a frozen row, so a plotting regression surfaces in
#: pytest instead of 25 minutes into a result round.
@pytest.fixture(scope="module")
def synthetic_pipeline():
    names = ("SYN_A", "SYN_B")
    arrays = {"SYN_A": drv.synthetic_arrays(n=1200, seed=11),
              "SYN_B": drv.synthetic_arrays(n=1200, seed=23, flip_rate=0.15)}
    families, boots, ids = {}, {}, {}
    for name in names:
        per_seed = arrays[name]
        family = drv.analyze_family(name, per_seed, drv.synthetic_reference(per_seed, name))
        family["cohort_rows"] = int(family["n"])
        families[name] = family
        boots[name] = dec.image_cluster_bootstrap(
            drv.family_statistic(per_seed), per_seed[0]["image_id"],
            n_replicates=25, seed=drv.core.BOOTSTRAP_SEED, ci=drv.core.BOOTSTRAP_CI,
            keep_replicates=True)
        ids[name] = {int(v) for v in per_seed[0]["sentence_id"]}
    gaps = drv.gap_rows(families, boots, pairs=(("SYN_B", "SYN_A"),))
    matched = [drv.matched_pair("SYN_B", "SYN_A", ids, arrays, replicates=25)]
    freeze = drv.load_freeze()
    verdict = {
        "primary_verdict": "DECOMPOSITION_REPORTED",
        "interpretation_boundary": freeze["interpretation_boundary"],
        "identity_checks": {"max_abs_shapley_residual": 0.0,
                            "max_abs_group_reconstruction_residual": 0.0,
                            "tolerance": dec.IDENTITY_TOLERANCE, "all_within_tolerance": True},
        "out_of_scope_confirmed": ["no new threshold"],
    }
    return {"names": names, "families": families, "arrays": arrays, "boots": boots,
            "gaps": gaps, "matched": matched, "freeze": freeze, "verdict": verdict}


def test_figures_render_on_synthetic_arrays(tmp_path, synthetic_pipeline):
    written = drv.make_figures(synthetic_pipeline["families"], synthetic_pipeline["arrays"],
                              synthetic_pipeline["gaps"], tmp_path)
    assert len(written) == 3
    for name in written:
        path = drv.ROOT / name
        assert path.exists() and path.stat().st_size > 1000, name


def test_report_and_csv_writers_render_on_synthetic_arrays(tmp_path, monkeypatch,
                                                           synthetic_pipeline):
    monkeypatch.setattr(drv, "OUT_DIR", tmp_path)
    names = synthetic_pipeline["names"]
    monkeypatch.setattr(drv, "FAMILIES", names)
    drv._write_point_rows(synthetic_pipeline["families"], synthetic_pipeline["gaps"])
    drv._write_support_rows(synthetic_pipeline["families"], synthetic_pipeline["arrays"],
                            synthetic_pipeline["boots"], synthetic_pipeline["gaps"],
                            synthetic_pipeline["matched"])
    report = drv.write_report(synthetic_pipeline["freeze"], synthetic_pipeline["families"],
                              synthetic_pipeline["boots"], synthetic_pipeline["gaps"],
                              synthetic_pipeline["matched"], synthetic_pipeline["verdict"],
                              ["figures/fig1.png", "figures/fig2.png", "figures/fig3.png"],
                              names)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.endswith("report.md")
    assert "DESCRIPTIVE_MECHANISM_DECOMPOSITION" in text
    assert "not paired" in text.replace("**not** paired", "not paired")   # naming discipline kept
    for sentence in synthetic_pipeline["freeze"]["interpretation_boundary"]["allowed"]:
        assert sentence.split(".")[0][:40] in text
    # the report renders numbers it was given and invents no causal wording of its own:
    # causal phrases may only appear quoted inside the frozen boundary section (section 8)
    causal = re.compile(r"\bcaus(?:e|es|ed|ing|al|ality)\b", re.IGNORECASE)
    body, _, boundary_section = text.partition("## 8. Interpretation boundary")
    assert boundary_section, "the frozen interpretation boundary section is missing"
    assert not causal.search(body), causal.search(body)
    for sentence in synthetic_pipeline["freeze"]["interpretation_boundary"]["forbidden"]:
        assert sentence in boundary_section
    for family in names:
        assert f"{synthetic_pipeline['families'][family]['mean']['D_total']:+0.6f}" in text
    assert (tmp_path / "point_decomposition.csv").exists()
    for name in ("transition_groups.csv", "pairwise_auc_components.csv",
                 "selective_error_sources.csv", "bootstrap_decomposition.csv",
                 "cross_family_gap_decomposition.csv", "matched_intersection_decomposition.csv"):
        assert (tmp_path / name).exists(), name


# ===========================================================================
# artifact tier - skipped until the result round has written anything
# ===========================================================================
def _require_result(name: str) -> Path:
    path = RESULT_DIR / name
    if not path.exists():
        pytest.skip(f"RQ4-M1 result round has not written {name} yet")
    return path


@pytest.fixture(scope="module")
def point_rows():
    import csv as _csv
    path = _require_result("point_decomposition.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(_csv.DictReader(handle))


@pytest.fixture(scope="module")
def published_c1():
    import csv as _csv
    if not C1_POINT.exists():
        pytest.skip("published C1 point CSV absent")
    with C1_POINT.open(encoding="utf-8", newline="") as handle:
        return {(row["family"], row["seed"], int(row["K"])): row for row in _csv.DictReader(handle)}


def test_written_corners_reproduce_the_published_c1_auroc(point_rows, published_c1):
    checked = 0
    for row in point_rows:
        if row["seed"] in ("mean3", "gap"):
            continue
        for k, corner in ((drv.core.K_BASELINE, "A00"), (drv.core.PRIMARY_KB, "A11")):
            published = float(published_c1[(row["family"], row["seed"], int(k))]["auroc_correct"])
            assert abs(float(row[corner]) - published) <= ALGEBRA_TOLERANCE, (row, k, corner)
            checked += 1
    assert checked == 2 * len(drv.FAMILIES) * len(drv.SEEDS)


def test_written_decomposition_identity_and_cohort_sizes_hold(point_rows):
    per_seed = 0
    for row in point_rows:
        if row["seed"] == "gap":
            continue
        d_total, d_label, d_conf = (float(row["D_total"]), float(row["D_label"]),
                                    float(row["D_conf"]))
        assert abs(d_label + d_conf - d_total) <= ALGEBRA_TOLERANCE, row
        assert 0.0 <= float(row["A00"]) <= 1.0 and 0.0 <= float(row["A11"]) <= 1.0
        if row["S"] == "":                                       # family mean row
            assert row["seed"] == "mean3"
            continue
        per_seed += 1
        assert int(float(row["G"])) == 0, row
        assert abs(float(row["S"]) + float(row["F"]) + float(row["E"])
                   + float(row["G"]) - float(row["n"])) <= 1e-9
    assert per_seed == len(drv.FAMILIES) * len(drv.SEEDS)


def test_written_gap_rows_are_exact_and_match_the_family_means(point_rows):
    means = {row["family"]: row for row in point_rows if row["seed"] == "mean3"}
    gaps = [row for row in point_rows if row["seed"] == "gap"]
    assert gaps, "no gap rows written"
    assert len(gaps) == len(drv.PAIRS)
    for gap in gaps:
        first, second = gap["family"].split("_minus_")
        for key, column in (("gap_total_degradation", "D_total"),
                            ("gap_label_component", "D_label"),
                            ("gap_conf_component", "D_conf")):
            expected = float(means[first][column]) - float(means[second][column])
            assert abs(float(gap[key]) - expected) <= 1e-9, (gap["family"], key)
        assert abs(float(gap["gap_residual"])) <= ALGEBRA_TOLERANCE


def test_result_artifacts_are_all_present_with_three_figures():
    for name in ("point_decomposition.csv", "bootstrap_decomposition.csv",
                 "transition_groups.csv", "pairwise_auc_components.csv",
                 "selective_error_sources.csv", "cross_family_gap_decomposition.csv",
                 "matched_intersection_decomposition.csv", "verdict.json", "metadata.json"):
        _require_result(name)
    figure_dir = RESULT_DIR / "figures"
    if not figure_dir.is_dir():
        pytest.skip("RQ4-M1 figures not written yet")
    figures = sorted(figure_dir.glob("*.png"))
    assert len(figures) == 3, [f.name for f in figures]
    assert len({f.name for f in figures}) == 3


def test_verdict_is_a_decomposition_report_with_no_success_label_or_causal_wording():
    path = _require_result("verdict.json")
    verdict = json.loads(path.read_text(encoding="utf-8"))
    assert verdict["classification"] == "DESCRIPTIVE_MECHANISM_DECOMPOSITION"
    assert verdict["primary_verdict"] == "DECOMPOSITION_REPORTED"
    assert verdict["identity_checks"]["all_within_tolerance"] is True
    assert verdict["stop_conditions_triggered"] == []
    assert verdict["forbidden_labels_used"] == []
    assert verdict["identity_checks"]["tolerance"] == ALGEBRA_TOLERANCE
    for label in FORBIDDEN_VERDICT_LABELS:
        assert f'"{label}"' not in json.dumps(verdict), label
    assert verdict["per_family_ordering"] and set(verdict["per_family_ordering"].values()) <= {
        "TRANSITION_HEAVIER", "CONFIDENCE_CHANGE_HEAVIER", "EXACT_TIE"}
    # the interpretation boundary *quotes* the forbidden causal phrasings, so the causal
    # wording check runs on everything else in the artifact
    scan = json.dumps({k: v for k, v in verdict.items() if k != "interpretation_boundary"})
    lowered = scan.lower()
    for phrase in ("causes the degradation", "is the causal mechanism",
                   "the true mechanism", "is the cause"):
        assert phrase not in lowered, phrase
    boundary = verdict["interpretation_boundary"]
    assert boundary["phrase_rule"].startswith("use 'accounts for")
    assert any("causes" in line for line in boundary["forbidden"])
    assert any("exactly decomposed" in line for line in boundary["allowed"])


def test_metadata_attests_zero_fit_and_only_frozen_prediction_inputs():
    path = _require_result("metadata.json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["classification"] == "DESCRIPTIVE_MECHANISM_DECOMPOSITION"
    assert meta["new_training_parameters"] == 0
    assert meta["model_forward_passes"] == 0
    assert meta["gpu_used"] is False
    assert meta["temperature_refit"] == 0
    assert meta["confidence_column"].startswith("conf_msp")
    inputs = set(meta["inputs_read"])
    assert len(inputs) == len(drv.FAMILIES) * len(drv.SEEDS) * 2
    # the metadata records OS-native paths, so compare on forward slashes
    assert all(p.replace("\\", "/").startswith("results/v2_proposal_robustness/predictions/")
               for p in inputs)
    assert len(meta["input_sha256"]) == len(inputs)
    assert meta["bootstrap"]["replicates"] == 5000 and meta["bootstrap"]["seed"] == 0


def test_matched_secondary_is_labelled_secondary_and_keeps_the_primary_meaning():
    import csv as _csv
    path = _require_result("matched_intersection_decomposition.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(_csv.DictReader(handle))
    assert rows
    assert {r["classification"] for r in rows} == {"SECONDARY_MATCHED_DIAGNOSTIC"}
    assert len({r["pair"] for r in rows}) == len(drv.PAIRS)
    for row in rows:
        if row["level"] == "paired_gap":
            assert abs(float(row["gap_residual"])) <= ALGEBRA_TOLERANCE
