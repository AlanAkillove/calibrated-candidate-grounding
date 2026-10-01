"""V2-P1 P1-F3 Route F frozen-scorer-compatibility invariants (protocol section 11).

The checks decidable *offline* (no GPU, no completed DETR extraction) always run:
the deterministic score-distribution / confidence-collapse detector, the primary
accuracy gate boundaries, the A/B/C/D interpretation matrix, the zero-fit guards
(static scan + runtime tripwire around the closed-form predict), and the frozen
V1-identity self-checks (B3 checkpoint hashes, per-seed temperature, bundle
checksum, OpenCLIP encoder).  The checks that need the DETR artefacts are guarded
by file existence so the default full-pytest run never fails while the region
extraction is still materialising.  The live GPU scoring is opt-in.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from ccg.external import frozen_identity as fi
from ccg.features import cache as fcache
from ccg.models.b3_data import B3Corpus
from ccg.semantic import frozen_load as fl
from ccg.semantic import hard_scores as hscores

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import p1_f3_route_f_compatibility as mod  # noqa: E402

# ---------------------------------------------------------------------------
# artefact availability
# ---------------------------------------------------------------------------
DETR_REGION = _ROOT / "cache/features_detr_r50/region_features.h5"
DETR_TEXT = _ROOT / "cache/features_detr_r50/text_features.h5"
DETR_MANIFESTS = _ROOT / "cache/manifests_detr/manifests/random_testA.jsonl"
HAS_DETR_CACHE = DETR_REGION.exists() and DETR_TEXT.exists()
HAS_DETR_MANIFESTS = DETR_MANIFESTS.exists()

NEEDS_DETR_CACHE = pytest.mark.skipif(not HAS_DETR_CACHE, reason="DETR feature cache not built yet")
NEEDS_DETR_MANIFESTS = pytest.mark.skipif(not HAS_DETR_MANIFESTS, reason="DETR manifests not built yet")

HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    pass
RUN_LIVE = pytest.mark.skipif(
    not (HAS_CUDA and HAS_DETR_CACHE and os.environ.get("CCG_RUN_P1F3_LIVE") == "1"),
    reason="set CCG_RUN_P1F3_LIVE=1 (with CUDA + DETR cache) to run live Route F scoring",
)


# ===========================================================================
# pure helpers (always run)
# ===========================================================================
def test_gate_boundaries():
    assert mod.gate_label(0.79) == "SCORER_COMPATIBLE"
    assert mod.gate_label(0.50) == "SCORER_COMPATIBLE"
    assert mod.gate_label(0.4999) == "SCORER_DEGRADED_BUT_USABLE"
    assert mod.gate_label(0.35) == "SCORER_DEGRADED_BUT_USABLE"
    assert mod.gate_label(0.3499) == "SCORER_DOMAIN_FAILURE"
    assert mod.gate_label(0.20) == "SCORER_DOMAIN_FAILURE"


def test_interpretation_matrix_cases():
    a = mod.interpretation_case(0.60, collapse=False)
    assert a["case"] == "A" and a["verdict"] == "ROUTE_F_FULLY_USABLE"
    b = mod.interpretation_case(0.60, collapse=True)
    assert b["case"] == "B" and "RELIABILITY_HEAD_SHIFT_WARNING" in b["verdict"]
    c = mod.interpretation_case(0.40, collapse=False)
    assert c["case"] == "C" and c["verdict"] == "SCORER_DEGRADED_BUT_USABLE"
    d = mod.interpretation_case(0.10, collapse=False)
    assert d["case"] == "D" and d["verdict"] == "SCORER_DOMAIN_FAILURE"


def test_std_collapse_boundary():
    assert mod.std_collapse(1e-4) is True
    assert mod.std_collapse(0.5) is False
    assert mod.std_collapse(float("nan")) is False


def test_densest_interval_fraction_known():
    # 96% of mass inside one 0.02-wide window at 0.50
    probs = np.concatenate([np.full(96, 0.50), np.arange(4) * 0.1 + 0.05])
    frac, lo = mod.densest_interval_fraction(probs, width=0.02)
    assert abs(frac - 0.96) < 1e-9
    assert lo == pytest.approx(0.50)
    # uniform -> small densest fraction
    uni = np.linspace(0.0, 1.0, 1000)
    ufrac, _ = mod.densest_interval_fraction(uni, width=0.02)
    assert ufrac <= 0.05


def test_confidence_collapse_collapse_true_on_constant_head():
    n = 500
    msp = np.linspace(0.2, 0.9, n)
    r1 = np.full(n, 0.5)          # std == 0  -> collapse
    e1b = np.linspace(0.1, 0.2, n)
    out = mod.confidence_collapse(msp=msp, r1=r1, e1b=e1b)
    assert out["collapse"] is True
    assert any("r1.std" in reason for reason in out["reasons"])


def test_confidence_collapse_healthy_is_false():
    rng = np.random.default_rng(0)
    msp = rng.uniform(0.3, 0.99, 4000)
    r1 = rng.uniform(0.05, 0.95, 4000)
    e1b = rng.uniform(0.05, 0.95, 4000)
    out = mod.confidence_collapse(msp=msp, r1=r1, e1b=e1b)
    assert out["collapse"] is False


def test_confidence_collapse_detector_is_deterministic():
    rng = np.random.default_rng(123)
    a = rng.uniform(0, 1, 2000)
    b = rng.uniform(0.4, 0.6, 2000)  # pile-up -> interval collapse
    c = rng.uniform(0, 1, 2000)
    first = mod.confidence_collapse(msp=a, r1=b, e1b=c)
    second = mod.confidence_collapse(msp=a, r1=b, e1b=c)
    assert json.dumps(first, sort_keys=True, default=float) == json.dumps(
        second, sort_keys=True, default=float
    )


def test_mean_std_empty_and_populated():
    mean0, std0 = mod._mean_std([])
    assert np.isnan(mean0) and np.isnan(std0)
    mean, std = mod._mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std([1.0, 2.0, 3.0]))


# ===========================================================================
# zero-fit guards
# ===========================================================================
def test_driver_source_has_no_forbidden_fit_calls():
    # assert_no_fit_path raises AssertionError on any forbidden call site; the
    # scanner strips strings/comments, so prose mentioning them is fine.
    info = fl.assert_no_fit_path([mod.__file__])
    assert info["n_files"] == 1
    assert set(info["needles"]) == set(fl.FORBIDDEN_PRODUCTION_CALLS)


def test_r1_e1b_predict_does_not_fit():
    bundle = fl.load_models(_ROOT / "results/phase1e_refcocog_external/frozen_models", verify_checksum=True)
    stats17 = np.random.default_rng(0).normal(size=(16, 17))
    sem16 = np.random.default_rng(1).normal(size=(16, 16))
    with fl.fit_is_forbidden() as tripwire:
        r1, e1b = bundle.predict("b3_seed1", stats17, sem16)
    assert tripwire.hits == []
    assert r1.shape == (16,) and e1b.shape == (16,)
    assert np.all((r1 >= 0.0) & (r1 <= 1.0)) and np.all((e1b >= 0.0) & (e1b <= 1.0))


# ===========================================================================
# frozen V1 identity (no DETR artefacts needed)
# ===========================================================================
def test_openclip_encoder_identity_is_frozen():
    report = fi.assert_openclip_identity(_ROOT / "cache/features")
    assert isinstance(report, dict)


def test_b3_checkpoint_hashes_unchanged():
    manifest = json.loads(
        (_ROOT / "results/phase1e_refcocog_external/frozen_models/frozen_artifact_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    arts = manifest["artifacts"]
    for seed in mod.B3_SEEDS:
        rec = arts[f"b3_model_b3_seed{seed}"]
        current = fi.sha256_file(_ROOT / rec["path"])
        assert current == rec["sha256"], f"seed {seed} B3 checkpoint hash drifted"


def test_bundle_temperature_matches_eval_metadata():
    bundle = fl.load_models(_ROOT / "results/phase1e_refcocog_external/frozen_models", verify_checksum=True)
    for seed in mod.B3_SEEDS:
        meta = json.loads(
            (_ROOT / f"results/phase0b_independent/seed_{seed}/eval_metadata.json").read_text(encoding="utf-8")
        )
        assert float(bundle.seed(f"b3_seed{seed}").temperature) == pytest.approx(
            float(meta["temperature_corrected"]), abs=1e-12
        )


def test_bundle_feature_contract_and_seeds():
    bundle = fl.load_models(_ROOT / "results/phase1e_refcocog_external/frozen_models", verify_checksum=True)
    bundle.check_feature_contract()
    for seed in mod.B3_SEEDS:
        assert f"b3_seed{seed}" in bundle.seeds


def test_pooled_test_scope_is_two_eval_splits():
    assert set(mod.POOLED_TEST) == {"testA", "testB"}
    assert "val_select" not in mod.POOLED_TEST and "val_calib" not in mod.POOLED_TEST


# ===========================================================================
# DETR manifest semantics (need manifests only, no region cache)
# ===========================================================================
def _detr_records():
    cfg = mod.FAMILIES["DETR"]
    corpus = B3Corpus(
        cfg["features_root"], cfg["manifests_root"], mod.REFS_PATH, cfg["bank"],
        image_sizes_path=mod.IMAGE_SIZES_PATH, ks=(mod.K,), regime="random", preload=False,
    )
    try:
        return corpus, mod.pooled_test_samples(corpus)
    except Exception:
        corpus.close()
        raise


@pytest.mark.parametrize("family", ["RPN", "DETR"])
def test_target_slot_is_local_index_zero(family):
    cfg = mod.FAMILIES[family]
    corpus = B3Corpus(
        cfg["features_root"], cfg["manifests_root"], mod.REFS_PATH, cfg["bank"],
        image_sizes_path=mod.IMAGE_SIZES_PATH, ks=(mod.K,), regime="random", preload=False,
    )
    try:
        samples = mod.pooled_test_samples(corpus)
        assert samples, f"{family}: no pooled-test rows"
        for s in samples[:50]:
            cand = B3Corpus.candidate_bank_indices(s, mod.K)
            assert cand.shape == (mod.K,)
            assert int(cand[0]) == int(s.target_index)
            head = np.asarray(s.distractor_order, dtype=np.int64)[: mod.K - 1]
            assert int(s.target_index) not in set(int(x) for x in head)
    finally:
        corpus.close()


@NEEDS_DETR_MANIFESTS
def test_detr_random_k5_row_identity_across_seeds():
    # A single frozen manifest -> identical cohort rows for every scorer seed.
    corpus, samples = _detr_records()
    try:
        ids = [int(s.sentence_id) for s in samples]
        assert len(ids) == len(set(ids)), "duplicate sentence_id in cohort"
        again = mod.pooled_test_samples(corpus)
        assert [int(s.sentence_id) for s in again] == ids
    finally:
        corpus.close()


@NEEDS_DETR_MANIFESTS
def test_detr_k5_candidate_count_is_five():
    corpus, samples = _detr_records()
    try:
        for s in samples[:50]:
            assert int(np.asarray(s.distractor_order).size) >= mod.K - 1
            cand = B3Corpus.candidate_bank_indices(s, mod.K)
            assert len(set(int(x) for x in cand)) == mod.K  # target + 4 distinct distractors
    finally:
        corpus.close()


# ===========================================================================
# DETR region cache (needs the completed extraction)
# ===========================================================================
@NEEDS_DETR_CACHE
def test_detr_region_cache_complete_and_dims():
    meta_path = _ROOT / "cache/features_detr_r50/metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    import h5py

    with h5py.File(DETR_REGION, "r") as h:
        feats = h["features"]
        assert int(feats.shape[1]) == 512
        assert feats.dtype == np.float16
    assert int(meta.get("n_images", 0)) > 0
    assert int(meta.get("invalid_crop_count", -1)) == 0


@NEEDS_DETR_CACHE
def test_detr_text_cache_is_the_frozen_v1_copy():
    v1 = fi.sha256_file(_ROOT / "cache/features/text_features.h5")
    detr = fi.sha256_file(DETR_TEXT)
    assert v1 == detr, "DETR text cache diverged from the frozen V1 text cache"


# ===========================================================================
# live Route F (opt-in GPU)
# ===========================================================================
@RUN_LIVE
def test_live_route_f_smoke_and_gate():
    out = _ROOT / "results/v2_p1_f3_route_f/_smoke"
    import argparse

    args = argparse.Namespace(
        b3_root=_ROOT / "results/phase0b_independent",
        frozen_models=_ROOT / "results/phase1e_refcocog_external/frozen_models",
        out_dir=out,
        device="cuda",
        batch_size=64,
        verify_rpn=True,
    )
    summary = mod.run(args)
    assert summary["gate"] in {
        "SCORER_COMPATIBLE",
        "SCORER_DEGRADED_BUT_USABLE",
        "SCORER_DOMAIN_FAILURE",
    }
    assert (out / "scorer_compatibility.json").exists()
    assert (out / "frozen_identity_check.json").exists()
