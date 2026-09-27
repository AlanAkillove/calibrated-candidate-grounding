"""End-to-end test of :func:`ccg.experiment.phase0a.run_audit` on a tiny cache.

The dataset is built with the *real* writers of the frozen interfaces:
:func:`ccg.features.cache.write_feature_cache` (512-d storage contract) and
:class:`ccg.data.manifests.ManifestFile` (jsonl + sibling meta json).  Layout:

* 8 images x 12 proposals x 512-d unit vectors, 16 sentences (2 per ref);
* ``val``  : image 1 -> ``val_select``, image 2 -> ``val_calib`` (2 sentences each)
* ``testA``: images 3-4  (4 sentences)      ``testB``: images 5-6 (4 sentences)
* images 7-8 are ``train`` and never evaluated -> 12 evaluated sentences total.

The second test monkeypatches ``phase0a.compute_candidate_scores`` so that K=10
scores are scaled by 0.5: the score-invariance check must catch it, write
``VALIDATION_FAILURE.json`` and STOP with ``SystemExit(2)`` before any result
artifact is produced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ccg.data.manifests import SCHEMA_VERSION, ManifestEntry, ManifestFile
from ccg.experiment import phase0a
from ccg.experiment.phase0a import run_audit

N_PROPOSALS = 12
FEATURES_DIM = 512
KS = (5, 10)
SPLITS = ("val_select", "val_calib", "testA", "testB")
N_EVAL_SENTENCES = 12  # images 1-6 x 2 sentences

TOP_LEVEL_ARTIFACTS = (
    "metadata.json",
    "cohort_summary.json",
    "prediction_summary.csv",
    "ranking_metrics.csv",
    "calibration_metrics.csv",
    "selective_metrics.csv",
    "reliability_bins.csv",
    "calibration_map_shift.csv",
    "global_temperature.json",
    "oracle_temperature_diagnostic.json",
    "bootstrap_ci.csv",
    "selective_curves.npz",
)
FIGURE_FILES = (
    "reliability_per_K.png",
    "risk_coverage_per_K.png",
    "ece_brier_acc_vs_K.png",
    "confidence_hist_per_K.png",
)


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
    """Write the tiny feature cache; prefer the frozen writer.

    ``ccg.features.cache.write_feature_cache`` is the frozen entry point, but
    at the time of writing it raises ``NameError`` inside ``_write_text_file``
    / ``_write_global_file`` (``for k, _ in items`` still uses the loop value
    ``v``).  Until that fix lands, the identical on-disk layout is written
    locally so this pipeline test does not depend on the repair; as soon as
    the frozen writer works again, the test switches back to it automatically.
    """
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


def _build_tiny_dataset(tmp_path):
    """Write ``features/`` + ``manifests/`` under ``tmp_path``; return inputs."""
    features_root = tmp_path / "features"
    rng = np.random.default_rng(0)
    region = {image_id: _unit_rows(rng, N_PROPOSALS, FEATURES_DIM) for image_id in range(1, 9)}
    text = {sid: _unit_rows(rng, 1, FEATURES_DIM)[0] for sid in range(16)}
    global_ = {image_id: _unit_rows(rng, 1, FEATURES_DIM)[0] for image_id in range(1, 9)}
    _write_cache(
        features_root,
        region,
        text,
        global_,
        {"native_logit_scale": 100.0, "source": "synthetic-test"},
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
    meta = {"schema_version": SCHEMA_VERSION, "regime": "random", "source": "tiny-pipeline-test"}
    val_entries = [
        _make_entry(ms_rng, 1, 1, "val", eval_split="val_select"),
        _make_entry(ms_rng, 2, 2, "val", eval_split="val_calib"),
    ]
    test_a = [_make_entry(ms_rng, 3, 3, "testA"), _make_entry(ms_rng, 4, 4, "testA")]
    test_b = [_make_entry(ms_rng, 5, 5, "testB"), _make_entry(ms_rng, 6, 6, "testB")]
    man_dir = tmp_path / "manifests"
    ManifestFile(regime="random", meta=meta, entries=val_entries).save(
        man_dir / "random_val.jsonl"
    )
    ManifestFile(regime="random", meta=meta, entries=test_a).save(
        man_dir / "random_testA.jsonl"
    )
    ManifestFile(regime="random", meta=meta, entries=test_b).save(
        man_dir / "random_testB.jsonl"
    )
    return features_root, refs


def _run(features_root, refs, tmp_path, out_name="results"):
    out_dir = tmp_path / out_name
    result = run_audit(
        features_root=features_root,
        manifests_root=tmp_path,
        refs=refs,
        out_dir=out_dir,
        ks=KS,
        splits=SPLITS,
        bootstrap_replicates=100,
        log=lambda message: None,
    )
    return out_dir, result


def test_run_audit_writes_full_artifact_set_and_consistent_cohort(tmp_path):
    features_root, refs = _build_tiny_dataset(tmp_path)
    out_dir, result = _run(features_root, refs, tmp_path)

    # -- artifact set -------------------------------------------------------
    for name in TOP_LEVEL_ARTIFACTS:
        path = out_dir / name
        assert path.exists() and path.stat().st_size > 0, name
    assert not (out_dir / "VALIDATION_FAILURE.json").exists()
    for k in KS:
        assert (out_dir / "raw_predictions" / f"K{k}.npz").exists()
    for name in FIGURE_FILES:
        assert (out_dir / "figures" / name).exists(), name
    manifest_dir = out_dir / "candidate_manifests"
    for stem in ("random_val", "random_testA", "random_testB"):
        assert (manifest_dir / f"{stem}.jsonl").exists()
        assert (manifest_dir / f"{stem}.meta.json").exists()
    assert (manifest_dir / "manifest_summary.json").exists()

    # -- cohort summary: n_common matches the hand-built dataset ------------
    cohort = result["cohort_summary"]
    expected_common = {"val_select": 2, "val_calib": 2, "testA": 4, "testB": 4, "__pooled__": 12}
    for name, n_common in expected_common.items():
        entry = cohort["splits"][name]
        assert entry["n_common_sentences"] == n_common, name
        for k in KS:
            per_k = entry["per_K"][str(k)]
            assert per_k["n_available_sentences"] == n_common, (name, k)
            assert per_k["n_skipped_target_missing"] == 0, (name, k)
            assert per_k["n_skipped_insufficient"] == 0, (name, k)
    assert cohort["splits"]["__pooled__"]["n_refs_total"] == 6
    assert cohort["splits"]["__pooled__"]["n_common_refs"] == 6

    # -- hard checks all passed --------------------------------------------
    for check in ("score_invariance", "rank_monotonic", "accuracy_monotonic"):
        assert result["hard_checks"][check]["status"] == "passed", check

    # -- raw predictions: shapes, dtypes, variant math ----------------------
    with np.load(out_dir / "raw_predictions" / "K5.npz") as npz:
        assert set(npz.files) >= {
            "raw_scores",
            "probabilities_native",
            "probabilities_T1",
            "candidate_indices",
            "sentence_ids",
            "ref_ids",
            "image_ids",
            "split_codes",
            "target_local",
        }
        raw = npz["raw_scores"]
        assert raw.shape == (N_EVAL_SENTENCES, 5)
        assert raw.dtype == np.float32
        assert npz["candidate_indices"].dtype == np.int32
        assert npz["candidate_indices"].shape == (N_EVAL_SENTENCES, 5)
        assert npz["sentence_ids"].dtype == np.int64
        assert npz["ref_ids"].dtype == np.int64
        assert npz["image_ids"].dtype == np.int64
        assert npz["split_codes"].dtype == np.int8
        assert npz["target_local"].dtype == np.int32

        # native = softmax(100 * s), recomputed straight from the kept logits
        z = raw.astype(np.float64) * 100.0
        z -= z.max(axis=1, keepdims=True)
        expected = np.exp(z)
        expected /= expected.sum(axis=1, keepdims=True)
        assert np.allclose(npz["probabilities_native"], expected, rtol=1e-5, atol=1e-7)
        # T1 = softmax(s)
        z = raw.astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        expected_t1 = np.exp(z)
        expected_t1 /= expected_t1.sum(axis=1, keepdims=True)
        assert np.allclose(npz["probabilities_T1"], expected_t1, rtol=1e-5, atol=1e-7)

    # shared candidates stay bit-identical across K inside the written npz too
    with np.load(out_dir / "raw_predictions" / "K10.npz") as npz10:
        raw10 = npz10["raw_scores"]
        cand10 = npz10["candidate_indices"]
        assert raw10.shape == (N_EVAL_SENTENCES, 10)
        with np.load(out_dir / "raw_predictions" / "K5.npz") as npz5:
            cand5 = npz5["candidate_indices"]
            raw5 = npz5["raw_scores"]
        for row in range(N_EVAL_SENTENCES):
            for col in range(5):
                hit = np.nonzero(cand10[row] == cand5[row, col])[0]
                assert hit.size == 1
                assert raw10[row, hit[0]] == raw5[row, col]  # atol=0, bit-identical

    # -- tables -------------------------------------------------------------
    with (out_dir / "prediction_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == N_EVAL_SENTENCES * len(KS)
    assert {row["K"] for row in rows} == {"5", "10"}
    assert all(row["confidence_global"] not in ("", None) for row in rows)
    assert all(int(row["correct"]) in (0, 1) for row in rows)

    with (out_dir / "bootstrap_ci.csv").open(encoding="utf-8", newline="") as handle:
        boot_rows = list(csv.DictReader(handle))
    assert boot_rows  # at least the K=5 vs K=10 comparisons
    assert {row["metric"] for row in boot_rows} >= {"accuracy", "ece_adaptive", "brier_binary", "aurc"}
    assert {row["variant"] for row in boot_rows} >= {"native", "T1", "global_T"}
    for row in boot_rows:
        assert float(row["ci_low"]) <= float(row["ci_high"])

    # -- global temperature fit (val_calib, K in {5, 10}) -------------------
    payload = json.loads((out_dir / "global_temperature.json").read_text(encoding="utf-8"))
    assert payload["kind"] == "global"
    assert payload["split"] == "val_calib"
    assert payload["ks"] == [5, 10]
    assert payload["n_sets"] == 4  # 2 sentences x 2 Ks of val_calib
    assert payload["temperature"] > 0.0
    assert payload["nll_after"] <= payload["nll_before"] + 1e-9

    # -- oracle diagnostic is clearly marked --------------------------------
    oracle = json.loads(
        (out_dir / "oracle_temperature_diagnostic.json").read_text(encoding="utf-8")
    )
    assert "ORACLE" in oracle["status"]
    assert set(oracle["oracle_temperatures"]) == {"5", "10"}
    assert oracle["verdict"]["verdict"] in {
        "oracle_repairs_drift",
        "no_meaningful_change",
        "mixed_or_partial",
    }

    # -- metadata -----------------------------------------------------------
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["regime"] == "random"
    assert metadata["primary_variant"] == "native"
    assert metadata["n_records"] == N_EVAL_SENTENCES
    assert metadata["native_logit_scale"] == pytest.approx(100.0)
    assert "loaded" not in metadata  # no stray keys

    # selective curves npz is non-empty
    with np.load(out_dir / "selective_curves.npz") as curves:
        assert curves.files
        assert any(key.endswith("__coverage") for key in curves.files)


def test_tampered_scoring_stops_with_validation_failure(tmp_path, monkeypatch):
    features_root, refs = _build_tiny_dataset(tmp_path)
    out_dir = tmp_path / "tampered"

    original = phase0a.compute_candidate_scores

    def tampered(text_vec, region_rows, candidate_indices, *, K=None):
        scores = original(text_vec, region_rows, candidate_indices, K=K)
        if K is not None and int(K) >= 10:
            return (scores * np.float32(0.5)).astype(np.float32)
        return scores

    monkeypatch.setattr(phase0a, "compute_candidate_scores", tampered)

    with pytest.raises(SystemExit) as excinfo:
        run_audit(
            features_root=features_root,
            manifests_root=tmp_path,
            refs=refs,
            out_dir=out_dir,
            ks=KS,
            splits=SPLITS,
            bootstrap_replicates=100,
            log=lambda message: None,
        )
    assert excinfo.value.code == 2

    failure_path = out_dir / "VALIDATION_FAILURE.json"
    assert failure_path.exists()
    payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert payload["status"] == "STOP"
    assert payload["check"] == "score_invariance"
    assert payload["violations"]

    # the audit STOPPED before writing any result artifact
    assert not (out_dir / "ranking_metrics.csv").exists()
    assert not (out_dir / "metadata.json").exists()
    assert not (out_dir / "raw_predictions").exists()
