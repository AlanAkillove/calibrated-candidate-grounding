"""CPU end-to-end + guard tests for :mod:`ccg.experiment.phase0b` (B3 eval).

The dataset is built with the *real* frozen writers (the tiny-dataset recipe of
``tests/test_phase0a_pipeline.py``) plus the B3 extras:

* ``ccg.features.cache.write_feature_cache`` (512-d storage contract),
* ``ccg.data.manifests.ManifestFile`` (jsonl + sibling meta json),
* ``ccg.data.bank.write_bank_entry`` (``cache/proposals.h5`` group layout),
* an ``image_sizes.npz`` ``{image_id: (W, H)}`` cache built by hand,
* one randomly-initialised :class:`ccg.models.independent.IndependentMLPScorer`
  checkpoint (``model.npz`` + ``training.json``).

Layout mirrors Phase 0A: 8 images x 12 proposals x 512-d unit vectors, 16
sentences (2 per ref); ``val`` image 1 -> ``val_select``, image 2 ->
``val_calib``; ``testA`` images 3-4; ``testB`` images 5-6; images 7-8 are
``train`` (never evaluated) -> **12 evaluated sentences**.  A random MLP keeps
the three hard checks (score invariance / rank & accuracy monotonicity) true by
construction, so the end-to-end path exercises the full artifact set.

The guard tests then pin the two contractual STOP paths and the two data-join
contracts:

* a tampered gather hook must raise ``SystemExit(2)`` *and* write
  ``seed_{s}/VALIDATION_FAILURE.json`` before any result artifact exists;
* the corrected temperature fit is isolated to ``val_calib`` - feeding it
  ``testA`` rows must raise ``CalibrationIsolationError``;
* ``paired_vs_cosine.csv`` must join on ``sentence_id`` regardless of the row
  order of either file and carry the frozen schema;
* ``raw_scores/K{K}.npz`` rows must equal the common-cohort sentence count.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ccg.data.manifests import SCHEMA_VERSION, ManifestEntry, ManifestFile
from ccg.experiment import phase0a, phase0b
from ccg.experiment.phase0a import (
    CalibrationIsolationError,
    ScoreBundle,
    SentenceRecord,
)
from ccg.experiment.phase0b import (
    fit_corrected_global_temperature,
    paired_vs_cosine_rows,
    run_b3_eval,
)
from ccg.models.independent import IndependentMLPScorer

N_PROPOSALS = 12
FEATURES_DIM = 512
GEO_DIM = 6
KS = (5, 10)
SPLITS = ("val_select", "val_calib", "testA", "testB")
N_EVAL_SENTENCES = 12  # images 1-6 x 2 sentences
SEED = 1
B3_VARIANTS = ("native", "global_T_corrected", "oracle_T_K")

PER_SEED_ARTIFACTS = (
    "ranking_metrics.csv",
    "calibration_metrics.csv",
    "normalized_selective_metrics.csv",
    "reliability_bins.csv",
    "diagnostics.csv",
    "bootstrap.csv",
    "per_sentence_predictions.npz",
    "eval_metadata.json",
)
TOP_LEVEL_ARTIFACTS = ("aggregate.csv", "paired_vs_cosine.csv", "metadata.json")


# ---------------------------------------------------------------------------
# dataset builders
# ---------------------------------------------------------------------------
def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    v = rng.normal(size=(n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def _make_entry(rng: np.random.Generator, ref_id, image_id, split, *, eval_split=None):
    target = int(rng.integers(0, N_PROPOSALS))
    order = [i for i in range(N_PROPOSALS) if i != target]
    return ManifestEntry(
        ref_id=int(ref_id),
        image_id=int(image_id),
        split=str(split),
        target_index=target,
        distractor_order=np.asarray(order, dtype=np.int32),
        n_valid_distractors=len(order),
        n_target_equiv_removed=0,
        n_same_category_available=0,
        eligible={5: True, 10: True},
        n_same_used_by_K={5: 0, 10: 0},
        hard_fraction_by_K={5: 0.0, 10: 0.0},
        target_max_iou=0.8,
        eval_split=None if eval_split is None else str(eval_split),
    )


def _write_cache(root, region, text, global_, metadata):
    """Write the tiny feature cache; prefer the frozen writer (fallback clone)."""
    from ccg.features.cache import write_feature_cache

    try:
        write_feature_cache(root, region, text, global_, metadata)
        return
    except NameError:
        _write_cache_compat(root, region, text, global_, metadata)


def _write_cache_compat(root, region, text, global_, metadata):
    """Local clone of the frozen cache layout (h5 files + metadata json)."""
    import h5py

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    image_ids = sorted(int(key) for key in region)
    blocks = [
        np.ascontiguousarray(np.asarray(region[image_id]), dtype=np.float16)
        for image_id in image_ids
    ]
    counts = np.asarray([block.shape[0] for block in blocks], dtype=np.int64)
    offsets = np.zeros((len(image_ids), 3), dtype=np.int64)
    offsets[:, 0] = image_ids
    offsets[:, 1] = np.concatenate([[0], np.cumsum(counts)[:-1]])
    offsets[:, 2] = counts
    region_rows = (
        np.concatenate(blocks, axis=0) if blocks else np.zeros((0, FEATURES_DIM), np.float16)
    )
    with h5py.File(str(root / "region_features.h5"), "w") as handle:
        handle.create_dataset("features", data=region_rows)
        handle.create_dataset("image_offsets", data=offsets)
        handle.attrs["counts"] = counts
        handle.attrs["cache_version"] = "phase0a-v1"

    sentence_ids = sorted(int(key) for key in text)
    text_rows = np.asarray(
        [np.asarray(text[sid], dtype=np.float16) for sid in sentence_ids], dtype=np.float16
    )
    sentence_offsets = np.zeros((len(sentence_ids), 2), dtype=np.int64)
    sentence_offsets[:, 0] = sentence_ids
    sentence_offsets[:, 1] = np.arange(len(sentence_ids), dtype=np.int64)
    with h5py.File(str(root / "text_features.h5"), "w") as handle:
        handle.create_dataset("features", data=text_rows)
        handle.create_dataset("sentence_offsets", data=sentence_offsets)
        handle.create_dataset(
            "texts",
            data=np.asarray(["" for _ in sentence_ids], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        handle.attrs["cache_version"] = "phase0a-v1"

    global_ids = sorted(int(key) for key in global_)
    global_rows = np.asarray(
        [np.asarray(global_[image_id], dtype=np.float16) for image_id in global_ids],
        dtype=np.float16,
    )
    with h5py.File(str(root / "global_features.h5"), "w") as handle:
        handle.create_dataset("features", data=global_rows)
        handle.create_dataset("image_ids", data=np.asarray(global_ids, dtype=np.int64))
        handle.attrs["cache_version"] = "phase0a-v1"

    payload = dict(metadata or {})
    payload.setdefault("cache_version", "phase0a-v1")
    payload["n_crops"] = int(counts.sum())
    payload["n_images"] = int(len(image_ids))
    payload["n_sentences"] = int(len(sentence_ids))
    (root / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (root / "text_index.csv").write_text(
        "sentence_id,ref_id,sent_id,image_id,split,text\n", encoding="utf-8"
    )


def _write_bank(path, rng, image_ids):
    """Synthetic ``cache/proposals.h5``: 12 valid ``xyxy`` boxes + objectness/image."""
    import h5py

    from ccg.data.bank import write_bank_entry

    with h5py.File(str(path), "w") as handle:
        for image_id in image_ids:
            top_left = rng.uniform(0.0, 500.0, size=(N_PROPOSALS, 2)).astype(np.float32)
            extent = rng.uniform(5.0, 120.0, size=(N_PROPOSALS, 2)).astype(np.float32)
            boxes = np.concatenate([top_left, top_left + extent], axis=1).astype(np.float32)
            objectness = rng.uniform(0.0, 1.0, size=(N_PROPOSALS,)).astype(np.float32)
            write_bank_entry(handle, int(image_id), boxes, objectness)


def _write_image_sizes(path, image_ids, width=640, height=480):
    payload = {
        str(int(image_id)): np.asarray([int(width), int(height)], dtype=np.int64)
        for image_id in image_ids
    }
    np.savez(str(path), **payload)


def _write_model(seed_dir, *, seed=SEED):
    """Random B3 checkpoint: ``model.npz`` (state dict) + ``training.json``."""
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    model = IndependentMLPScorer(
        feature_dim=FEATURES_DIM,
        hidden_dim=128,
        geo_dim=GEO_DIM,
        temperature=1.0,
        device="cpu",
        seed=seed,
    )
    model.save(seed_dir / "model.npz")
    payload = {
        "status": "complete",
        "feature_dim": FEATURES_DIM,
        "hidden_dim": 128,
        "geo_dim": GEO_DIM,
        "temperature": 1.0,
        "seed": int(seed),
        "lr": 1e-3,
        "num_parameters": int(model.num_parameters()),
    }
    (seed_dir / "training.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return int(model.num_parameters())


def _build_tiny_dataset(root: Path):
    """Write ``features/`` + ``manifests/`` + bank + sizes under ``root``."""
    features_root = root / "features"
    rng = np.random.default_rng(0)
    region = {image_id: _unit_rows(rng, N_PROPOSALS, FEATURES_DIM) for image_id in range(1, 9)}
    text = {sid: _unit_rows(rng, 1, FEATURES_DIM)[0] for sid in range(16)}
    global_ = {image_id: _unit_rows(rng, 1, FEATURES_DIM)[0] for image_id in range(1, 9)}
    _write_cache(
        features_root,
        region,
        text,
        global_,
        {"native_logit_scale": 100.0, "source": "synthetic-phase0b-test"},
    )

    # two sentences per ref; ref_id == image_id; sent_ids are archive-unique
    refs = [
        {
            "ref_id": image_id,
            "image_id": image_id,
            "sent_ids": [2 * (image_id - 1), 2 * (image_id - 1) + 1],
        }
        for image_id in range(1, 9)
    ]

    ms_rng = np.random.default_rng(1)
    meta = {"schema_version": SCHEMA_VERSION, "regime": "random", "source": "tiny-phase0b-test"}
    val_entries = [
        _make_entry(ms_rng, 1, 1, "val", eval_split="val_select"),
        _make_entry(ms_rng, 2, 2, "val", eval_split="val_calib"),
    ]
    test_a = [_make_entry(ms_rng, 3, 3, "testA"), _make_entry(ms_rng, 4, 4, "testA")]
    test_b = [_make_entry(ms_rng, 5, 5, "testB"), _make_entry(ms_rng, 6, 6, "testB")]
    man_dir = root / "manifests"
    ManifestFile(regime="random", meta=meta, entries=val_entries).save(
        man_dir / "random_val.jsonl"
    )
    ManifestFile(regime="random", meta=meta, entries=test_a).save(
        man_dir / "random_testA.jsonl"
    )
    ManifestFile(regime="random", meta=meta, entries=test_b).save(
        man_dir / "random_testB.jsonl"
    )

    bank_path = root / "proposals.h5"
    _write_bank(bank_path, np.random.default_rng(3), range(1, 9))
    image_sizes_path = root / "image_sizes.npz"
    _write_image_sizes(image_sizes_path, range(1, 9))

    return {
        "root": root,
        "features": features_root,
        "manifests": man_dir,
        "bank": bank_path,
        "refs": refs,
        "image_sizes": image_sizes_path,
    }


def _run_b3(ds, out_dir, *, seeds=(SEED,), cosine_predictions=None, bootstrap_replicates=40):
    return run_b3_eval(
        features_root=ds["features"],
        manifests_root=ds["manifests"],
        bank_path=ds["bank"],
        refs=ds["refs"],
        out_dir=out_dir,
        seeds=tuple(int(s) for s in seeds),
        ks=KS,
        splits=SPLITS,
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=0,
        bootstrap_ci=0.95,
        device="cpu",
        image_sizes_path=ds["image_sizes"],
        cosine_predictions=cosine_predictions,
        log=lambda message: None,
    )


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """One shared end-to-end CPU run over the tiny dataset (1 seed, K={5,10})."""
    root = tmp_path_factory.mktemp("phase0b_e2e")
    ds = _build_tiny_dataset(root)
    out_dir = root / "results"
    params = _write_model(out_dir / f"seed_{SEED}")
    result = _run_b3(ds, out_dir)
    return {"ds": ds, "root": root, "out_dir": out_dir, "result": result, "params": params}


# ---------------------------------------------------------------------------
# 1. end-to-end artifact set + cohort / invariance contracts
# ---------------------------------------------------------------------------
def test_run_b3_eval_writes_full_artifact_set(tiny_run):
    out_dir = tiny_run["out_dir"]
    seed_dir = out_dir / f"seed_{SEED}"

    for name in TOP_LEVEL_ARTIFACTS:
        path = out_dir / name
        assert path.exists() and path.stat().st_size > 0, name
    for name in PER_SEED_ARTIFACTS:
        path = seed_dir / name
        assert path.exists() and path.stat().st_size > 0, name
    assert not (seed_dir / "VALIDATION_FAILURE.json").exists()
    for k in KS:
        assert (seed_dir / "raw_scores" / f"K{k}.npz").exists()

    assert tiny_run["result"]["n_records"] == N_EVAL_SENTENCES

    # hard checks all passed (score invariance is enforced during scoring)
    meta = json.loads((seed_dir / "eval_metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete"
    assert meta["seed"] == SEED
    assert meta["ks"] == list(KS)
    for check in ("score_invariance", "rank_monotonic", "accuracy_monotonic"):
        assert meta["hard_checks"][check]["status"] == "passed", check
    assert meta["temperature_corrected"] > 0.0
    assert meta["corrected_fit"]["n_sets"] == 4  # 2 val_calib sentences x K in {5, 10}
    assert meta["n_rows_common"]["5"] == N_EVAL_SENTENCES
    assert meta["n_rows_common"]["10"] == N_EVAL_SENTENCES


def test_raw_scores_rows_match_cohort_and_dtypes(tiny_run):
    seed_dir = tiny_run["out_dir"] / f"seed_{SEED}"
    for k in KS:
        with np.load(seed_dir / "raw_scores" / f"K{k}.npz", allow_pickle=False) as npz:
            assert set(npz.files) == {
                "scores",
                "sentence_id",
                "ref_id",
                "image_id",
                "eval_split",
                "target_local",
            }
            scores = npz["scores"]
            # row count == common-cohort sentence count (never re-sampled)
            assert scores.shape == (N_EVAL_SENTENCES, k)
            assert scores.dtype == np.float32
            assert npz["sentence_id"].dtype == np.int64
            assert npz["ref_id"].dtype == np.int64
            assert npz["image_id"].dtype == np.int64
            assert npz["target_local"].dtype == np.int32
            assert np.unique(npz["sentence_id"]).size == N_EVAL_SENTENCES
            assert np.all(np.isfinite(scores))


def test_per_sentence_predictions_columns(tiny_run):
    seed_dir = tiny_run["out_dir"] / f"seed_{SEED}"
    with np.load(seed_dir / "per_sentence_predictions.npz", allow_pickle=False) as npz:
        files = set(npz.files)
        assert {"sentence_ids", "ref_ids", "image_ids", "eval_split"} <= files
        for k in KS:
            assert f"correct_K{k}" in files
            assert f"target_rank_K{k}" in files
            assert f"predicted_index_K{k}" in files
            for variant in B3_VARIANTS:
                assert f"confidence_{variant}_K{k}" in files
        assert npz["sentence_ids"].shape == (N_EVAL_SENTENCES,)


def test_aggregate_includes_selective_metrics(tiny_run):
    out_dir = tiny_run["out_dir"]
    with (out_dir / "aggregate.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    # ranking metrics are variant-invariant; calibration/selective carry variants
    assert any(
        row["metric"] == "top1" and row["variant"] == "(invariant)" for row in rows
    )
    assert any(
        row["metric"] == "e_aurc" and row["variant"] == "global_T_corrected"
        for row in rows
    ), "selective metrics must survive into aggregate.csv"
    # a metric reported by two tables (``accuracy``) must not inflate n_seeds
    assert all(int(row["n_seeds"]) == 1 for row in rows)
    # a robust metric stays finite even on the degenerate tiny sample
    top1_rows = [row for row in rows if row["metric"] == "top1"]
    assert top1_rows
    assert all(float(row["mean"]) == float(row["mean"]) for row in top1_rows)


def test_metadata_and_stubbed_paired_table(tiny_run):
    out_dir = tiny_run["out_dir"]
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_records"] == N_EVAL_SENTENCES
    assert metadata["config"]["device"] == "cpu"
    assert metadata["config"]["ks"] == list(KS)
    assert metadata["config"]["seeds"] == [SEED]
    assert "1" in metadata["per_seed"]
    assert metadata["per_seed"]["1"]["status"] == "complete"
    # cosine file was never supplied -> explicit stub rows, never a silent skip
    assert metadata["paired_vs_cosine"]["status"] == ["cosine_missing"]
    with (out_dir / "paired_vs_cosine.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert paired
    assert all(row["status"] == "cosine_missing" for row in paired)


# ---------------------------------------------------------------------------
# 2. tampered gather hook -> VALIDATION_FAILURE + SystemExit(2)
# ---------------------------------------------------------------------------
def test_tampered_gather_stops_with_validation_failure(tmp_path, monkeypatch):
    ds = _build_tiny_dataset(tmp_path)
    out_dir = tmp_path / "tampered"
    seed_dir = out_dir / f"seed_{SEED}"
    _write_model(seed_dir)

    original = phase0b.gather_superset_for_k

    def tampered(superset_scores, superset_candidates, k):
        scores, candidates = original(superset_scores, superset_candidates, k)
        if int(k) >= 10:  # break bit-identity of the shared candidates
            scores = (scores * np.float32(0.5)).astype(np.float32)
        return scores, candidates

    monkeypatch.setattr(phase0b, "gather_superset_for_k", tampered)

    with pytest.raises(SystemExit) as excinfo:
        _run_b3(ds, out_dir)
    assert excinfo.value.code == 2

    failure_path = seed_dir / "VALIDATION_FAILURE.json"
    assert failure_path.exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["status"] == "STOP"
    assert payload["check"] == "score_invariance"
    assert payload["violations"]

    # the evaluation STOPPED before any result artifact was written
    assert not (seed_dir / "raw_scores").exists()
    assert not (seed_dir / "ranking_metrics.csv").exists()
    assert not (seed_dir / "per_sentence_predictions.npz").exists()
    assert not (seed_dir / "eval_metadata.json").exists()
    assert not (out_dir / "aggregate.csv").exists()
    assert not (out_dir / "metadata.json").exists()


# ---------------------------------------------------------------------------
# 3. temperature fit isolation (val_calib only)
# ---------------------------------------------------------------------------
def _hand_bundle():
    """A minimal hand-built :class:`ScoreBundle`: 2 val_calib + 2 testA sentences."""
    ks = (5, 10)
    rng = np.random.default_rng(0)
    specs = [("val_calib", 10), ("val_calib", 11), ("testA", 20), ("testA", 21)]
    records = [
        SentenceRecord(
            sentence_id=sid,
            ref_id=sid,
            image_id=sid,
            eval_split=split,
            target_index=0,
            distractor_order=np.arange(1, N_PROPOSALS, dtype=np.int32),
            target_max_iou=0.8,
            eligible={5: True, 10: True},
        )
        for split, sid in specs
    ]
    raw_scores, candidate_indices, target_local = {}, {}, {}
    for k in ks:
        raw_scores[k] = rng.normal(size=(len(records), k)).astype(np.float32)
        candidate_indices[k] = np.tile(
            np.arange(k, dtype=np.int32), (len(records), 1)
        )
        target_local[k] = np.zeros(len(records), dtype=np.int32)
    return ScoreBundle(
        ks=ks,
        records_by_k={k: list(records) for k in ks},
        raw_scores=raw_scores,
        candidate_indices=candidate_indices,
        target_local=target_local,
        n_skipped_target_missing={k: 0 for k in ks},
        n_skipped_insufficient={k: 0 for k in ks},
    )


def test_temperature_fit_accepts_val_calib():
    bundle = _hand_bundle()
    fit = fit_corrected_global_temperature(bundle, ks=(5, 10))
    assert fit.temperature > 0.0
    assert fit.n_sets == 4  # the two val_calib sentences pooled over K in {5, 10}


def test_temperature_fit_rejects_testA_rows():
    bundle = _hand_bundle()
    test_rows = np.asarray([2, 3], dtype=np.int64)  # the two testA sentences
    with pytest.raises(CalibrationIsolationError):
        fit_corrected_global_temperature(
            bundle, ks=(5, 10), rows_by_k={5: test_rows, 10: test_rows}
        )


# ---------------------------------------------------------------------------
# 4. paired_vs_cosine: schema + row alignment independent of file order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cos_id_key", ["sentence_id", "sentence_ids"])
def test_paired_vs_cosine_schema_and_alignment(tiny_run, tmp_path, cos_id_key):
    out_dir = tiny_run["out_dir"]
    seed_dir = out_dir / f"seed_{SEED}"
    with np.load(seed_dir / "per_sentence_predictions.npz", allow_pickle=False) as store:
        sids = np.asarray(store["sentence_ids"]).reshape(-1)
        correct5 = np.asarray(store["correct_K5"]).reshape(-1).astype(np.float64)

    # a fake cosine table stored in REVERSED row order (must still join by id).
    # Regression (live crash, 2026-09-28): the corrected cosine audit writes the
    # *singular* ``sentence_id`` while the B3 writer emits plurals - the reader
    # must accept both spellings, so the test runs over both.
    conf_fake = np.linspace(0.51, 0.99, sids.size).astype(np.float32)
    cos_path = tmp_path / "cosine_predictions.npz"
    np.savez(
        str(cos_path),
        **{cos_id_key: sids[::-1].copy()},
        confidence_global_T_K5=conf_fake[::-1].copy(),
        correct_K5=correct5.astype(np.float32)[::-1].copy(),
    )

    summaries = [{"seed": SEED, "status": "complete"}]
    rows = paired_vs_cosine_rows(
        out_dir,
        summaries,
        cosine_predictions=cos_path,
        ks=(5, 20, 50),
        bootstrap_replicates=30,
        log=lambda message: None,
    )
    assert rows
    for row in rows:
        assert set(row.keys()) == set(phase0b.PAIRED_FIELDS)

    k5_rows = [row for row in rows if int(row["K"]) == 5]
    assert len(k5_rows) == len(phase0b._PAIRED_METRICS)
    assert all(row["status"] == "ok" for row in k5_rows)
    for row in k5_rows:
        assert int(row["n"]) == N_EVAL_SENTENCES
        assert int(row["n_clusters"]) == 6  # 6 distinct images
        # degenerate tiny samples can make a resample degenerate (e.g. risk 0 at
        # full coverage) -> NaN; the interval must still be ordered when finite
        lo, hi = float(row["delta_ci_low"]), float(row["delta_ci_high"])
        if lo == lo and hi == hi:
            assert lo <= hi

    # the cosine column is rebuilt from the id-sorted join, so it equals the
    # metric of the fake stats in sentence_id order
    order = np.argsort(sids, kind="stable")
    expected_conf = conf_fake[order].astype(np.float64)
    expected_correct = correct5[order]
    expected = phase0a.accuracy_metric(
        phase0a.SampleStats.from_conf_correct(expected_conf, expected_correct)
    )
    accuracy_row = next(row for row in k5_rows if row["metric"] == "accuracy")
    assert accuracy_row["cosine"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert np.isfinite(float(accuracy_row["delta_ci_low"]))
    assert float(accuracy_row["delta_ci_low"]) <= float(accuracy_row["delta_ci_high"])

    # K=20/50 columns are absent -> explicit stubs, not silent omissions
    for k in (20, 50):
        k_rows = [row for row in rows if int(row["K"]) == k]
        assert k_rows
        assert all(row["status"] == f"cosine_missing_k{k}" for row in k_rows)
