"""Deterministic seeding, plus the config / logging / io utilities next to it."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from ccg.utils.config import (
    Config,
    apply_overrides,
    flatten,
    load_config,
    merge_configs,
    save_config,
)
from ccg.utils.io import (
    cache_path,
    ensure_dir,
    file_sha256,
    iter_chunks,
    load_npz,
    read_json,
    require_file,
    save_npz,
    write_json,
)
from ccg.utils.logging import (
    EXPERIMENT_RECORD_FIELDS,
    ExperimentRecord,
    PredictionRecord,
    jsonable,
    load_jsonl,
    load_predictions_npz,
    new_experiment_id,
    resolve_git_commit,
    save_jsonl,
    save_predictions_npz,
    save_records,
    utc_now_iso,
)
from ccg.utils.seed import SEEDS, default_rng, set_deterministic

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------
def test_set_deterministic_makes_the_numpy_stream_reproducible():
    set_deterministic(0)
    first = (np.random.rand(5), random.random())
    set_deterministic(0)
    second = (np.random.rand(5), random.random())
    np.testing.assert_allclose(first[0], second[0])
    assert first[1] == second[1]

    set_deterministic(1)
    other = np.random.rand(5)
    assert not np.allclose(other, first[0]), "a different seed must change the stream"
    assert set_deterministic(7) == 7, "the seed is returned so call sites can log it"


def test_set_deterministic_touches_torch_when_available():
    assert set_deterministic(3, cuda_deterministic=True) == 3
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        pytest.skip("torch not installed in this environment")
    value = torch.rand(4).numpy()
    set_deterministic(3)
    np.testing.assert_allclose(torch.rand(4).numpy(), value)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_set_deterministic_environment_and_validation(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    set_deterministic(0)
    assert "OMP_NUM_THREADS" not in __import__("os").environ, "off by default"
    set_deterministic(0, env_threads=True)
    assert __import__("os").environ["OMP_NUM_THREADS"] == "1"
    assert __import__("os").environ["PYTHONHASHSEED"] == "0"
    with pytest.raises(ValueError, match="non-negative"):
        set_deterministic(-1)


def test_default_rng_is_independent_of_the_global_stream():
    a = default_rng(0).normal(size=4)
    b = default_rng(0).normal(size=4)
    np.testing.assert_allclose(a, b)
    set_deterministic(0)
    np.random.rand(10)
    np.testing.assert_allclose(default_rng(0).normal(size=4), a), "global state must not leak"
    assert SEEDS == (0, 1, 2)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_default_and_phase0_configs_load_and_inherit():
    default = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert default.get("dataset.name") == "refcoco+"
    assert default.get("proposal_n") == 64 and default.get("iou_threshold") == 0.5
    assert default.get("Ks_test") == [5, 10, 20, 50]
    assert default.get("statistics.bootstrap_replicates") == 5000
    assert default.get("backbone.model_name") == "ViT-B/32"
    assert default.get("does.not.exist", "fallback") == "fallback"
    assert "seed" in default and "models.b3_independent_mlp.hidden_dim" in default

    phase0 = load_config(REPO_ROOT / "configs" / "phase0.yaml")
    assert phase0.get("baselines") == ["b0_random", "b1_cosine", "b2_temp", "b3_mlp"]
    # _base_ inheritance: everything frozen stays identical to default.yaml
    for key in ("dataset.name", "backbone.model_name", "proposal_n", "iou_threshold", "seed"):
        assert phase0.get(key) == default.get(key), key
    assert phase0.get("Ks_test") == [5, 10, 20, 50]
    assert "_base_" not in phase0 and phase0.get("_base_", None) is None


def test_config_dot_access_and_mutation():
    cfg = Config({"a": {"b": {"c": 1}}, "list": [10, 20]})
    assert cfg["a.b.c"] == 1 and cfg.get("a.b") == {"c": 1}
    assert cfg["list.1"] == 20
    cfg["a.b.d"] = 2
    assert cfg["a.b.d"] == 2
    cfg.set("x.y", 3)
    assert cfg["x.y"] == 3
    del cfg["x.y"]
    assert "x.y" not in cfg
    with pytest.raises(KeyError):
        cfg["a.missing"]
    with pytest.raises(KeyError):
        cfg.require("a.missing")
    with pytest.raises(TypeError):
        cfg["a.b.c.deeper"] = 1, "cannot descend into a scalar"
    assert sorted(flatten({"a": {"b": 1}})) == ["a.b"]
    assert cfg.to_dict()["a"]["b"]["c"] == 1
    assert cfg.to_dict() is not cfg._data, "to_dict must hand out a copy"


def test_merge_and_override_semantics():
    base = {"a": {"b": 1, "c": 2}, "list": [1, 2, 3]}
    merged = merge_configs(base, {"a": {"c": 9}, "list": [4]})
    assert merged == {"a": {"b": 1, "c": 9}, "list": [4]}, "lists replace, dicts merge"
    cfg = Config(base)
    out = apply_overrides(cfg, ["a.b=5", "new.key=[1,2]", "flag=true"])
    assert out["a.b"] == 5 and out["new.key"] == [1, 2] and out["flag"] is True
    assert cfg["a.b"] == 1, "overrides never mutate the source config"
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(cfg, ["nope"])


def test_save_load_roundtrip_and_base_cycle(tmp_path):
    cfg = Config({"a": {"b": [1, 2]}, "c": 0.5}, source="mem")
    path = save_config(cfg, tmp_path / "cfg" / "x.yaml")
    assert load_config(path)["a.b"] == [1, 2] and load_config(path)["c"] == 0.5
    (tmp_path / "base.yaml").write_text("k: 1\nshared: {a: 1, b: 2}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\nk: 2\nshared: {b: 3}\n", encoding="utf-8")
    child = load_config(tmp_path / "child.yaml")
    assert child["k"] == 2 and child["shared"] == {"a": 1, "b": 3}
    assert load_config(tmp_path / "child.yaml", follow_base=False)["shared"] == {"b": 3}
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic"):
        load_config(tmp_path / "a.yaml")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(tmp_path / "list.yaml")


# ---------------------------------------------------------------------------
# experiment records (section 28)
# ---------------------------------------------------------------------------
def test_experiment_record_schema_and_json_safety():
    record = ExperimentRecord(
        experiment_id="phase0-test",
        K=np.int64(10),
        seed=0,
        metrics={"acc": np.float32(0.5), "per_image": np.arange(3)},
        training_config={"lr": 1e-3},
    )
    payload = record.to_dict()
    assert list(payload) == list(EXPERIMENT_RECORD_FIELDS) + ["artefacts"]
    assert payload["K"] == 10 and payload["metrics"]["acc"] == pytest.approx(0.5)
    assert payload["metrics"]["per_image"] == [0, 1, 2]
    assert payload["dataset"] == "refcoco+" and payload["backbone"].startswith("open_clip")
    text = json.dumps(payload)  # must be serialisable without a custom encoder
    assert json.loads(text)["experiment_id"] == "phase0-test"
    with pytest.raises(ValueError, match="target_presence"):
        ExperimentRecord(target_presence="sometimes")
    assert ExperimentRecord().target_presence == "present"
    assert isinstance(resolve_git_commit(REPO_ROOT), (str, type(None)))
    identifier = new_experiment_id("phase0")
    assert identifier.startswith("phase0-") and len(identifier.split("-")) == 3
    assert utc_now_iso().endswith("+00:00") or "T" in utc_now_iso()
    assert jsonable(Path("x/y")) == str(Path("x/y")) and jsonable(np.bool_(True)) is True


def test_prediction_record_validations():
    ok = PredictionRecord(
        ref_id=1, K=3, candidate_indices=[7, 8, 9], logits=[2.0, 0.0, 1.0],
        probabilities=[0.6, 0.2, 0.2], predicted_index=0, target_index=0, confidence=0.6,
    )
    assert ok.correct is True and ok.abstained is False
    assert ok.candidate_indices.dtype == np.int64 and ok.logits.dtype == np.float32
    abstain = PredictionRecord(
        ref_id=2, K=3, target_present=False, target_index=None, predicted_index=None,
        candidate_indices=[1, 2, 3],
    )
    assert abstain.correct is None and abstain.abstained is True
    assert abstain.to_dict()["target_present"] is False
    with pytest.raises(ValueError, match="logits for K=3"):
        PredictionRecord(ref_id=3, K=3, logits=[1.0, 2.0])
    with pytest.raises(ValueError, match="candidate ids for K=3"):
        PredictionRecord(ref_id=4, K=3, candidate_indices=[1, 2])
    with pytest.raises(ValueError, match="requires a target_index"):
        PredictionRecord(ref_id=5, K=2, target_present=True)
    with pytest.raises(ValueError, match="target-absent sets"):
        PredictionRecord(ref_id=6, K=2, target_present=False, target_index=1)
    wrong = PredictionRecord(
        ref_id=7, K=2, candidate_indices=[1, 2], predicted_index=1, target_index=0
    )
    assert wrong.correct is False


def test_jsonl_roundtrip_appends(tmp_path):
    path = tmp_path / "logs" / "runs.jsonl"
    records = [
        ExperimentRecord(experiment_id="a", K=5, metrics={"acc": 0.4}),
        ExperimentRecord(experiment_id="b", K=10, metrics={"acc": 0.3}),
    ]
    save_records(records, path)
    assert len(load_jsonl(path)) == 2
    save_jsonl([ExperimentRecord(experiment_id="c", K=20)], path)  # append mode
    rows = load_jsonl(path)
    assert [row["experiment_id"] for row in rows] == ["a", "b", "c"]
    typed = load_jsonl(path, ExperimentRecord)
    assert isinstance(typed[0], ExperimentRecord) and typed[0].experiment_id == "a"
    assert all(json.loads(line)["K"] for line in path.read_text(encoding="utf-8").splitlines())


def test_prediction_npz_dump_is_ragged_safe(tmp_path):
    path = tmp_path / "preds.npz"
    save_predictions_npz(
        path,
        ref_ids=[1, 2],
        candidate_indices=[[3, 4], [5, 6, 7]],
        logits=[[1.0, 0.0], [2.0, 1.0, 0.0]],
        probabilities=[[0.7, 0.3], [0.6, 0.3, 0.1]],
        correct=[True, False],
        metadata={"model": "b1_clip_cosine", "K": np.int64(5)},
    )
    data = load_predictions_npz(path)
    assert data["ref_ids"].tolist() == [1, 2]
    assert [len(row) for row in data["candidate_indices"]] == [2, 3]
    assert data["metadata"] == {"model": "b1_clip_cosine", "K": 5}
    assert data["correct"].tolist() == [True, False]
    with pytest.raises(ValueError, match="same length"):
        save_predictions_npz(
            path, ref_ids=[1], candidate_indices=[[1], [2]], logits=[[1.0]],
            probabilities=[[1.0]], correct=[True],
        )


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------
def test_io_helpers(tmp_path):
    path = save_npz(tmp_path / "a" / "b.npz", arr=np.zeros(3), scalar=2, mapping={"x": 1})
    loaded = load_npz(path, allow_pickle=True)
    assert loaded["arr"].tolist() == [0.0, 0.0, 0.0] and int(loaded["scalar"]) == 2
    assert loaded["mapping"].item() == {"x": 1}
    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_npz(path), "object arrays need explicit trust"
    json_path = write_json({"b": 1, "a": np.float32(0.5)}, tmp_path / "x.json")
    assert read_json(json_path) == {"a": 0.5, "b": 1}
    assert json.dumps(read_json(json_path))  # numpy scalars were converted
    digest = file_sha256(json_path)
    assert len(digest) == 64 and file_sha256(json_path) == digest
    json_path.write_text('{"changed": true}', encoding="utf-8")
    assert file_sha256(json_path) != digest
    assert ensure_dir(tmp_path / "new" / "dir").is_dir()
    assert cache_path(tmp_path / "cache", "features.h5").name == "features.h5"
    assert require_file(json_path) == json_path
    with pytest.raises(FileNotFoundError, match="Phase 0 checklist step"):
        require_file(tmp_path / "missing.h5")
    with pytest.raises(FileNotFoundError, match="custom hint"):
        require_file(tmp_path / "missing.h5", hint="custom hint")
    chunks = list(iter_chunks(range(5), 2))
    assert chunks == [[0, 1], [2, 3], [4]]
    assert list(iter_chunks([], 3)) == []
    with pytest.raises(ValueError, match="chunk_size"):
        list(iter_chunks([1], 0))
