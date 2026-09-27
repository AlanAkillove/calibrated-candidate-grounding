"""Utilities: seeding, configuration, experiment logging, cache IO."""

from __future__ import annotations

from .config import Config, apply_overrides, load_config, merge_configs, save_config
from .io import (
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
from .logging import (
    ExperimentRecord,
    PredictionRecord,
    load_jsonl,
    load_predictions_npz,
    new_experiment_id,
    resolve_git_commit,
    save_jsonl,
    save_predictions_npz,
    save_records,
    utc_now_iso,
)
from .seed import SEEDS, default_rng, set_deterministic

__all__ = [
    # seed
    "set_deterministic",
    "default_rng",
    "SEEDS",
    # config
    "Config",
    "load_config",
    "save_config",
    "merge_configs",
    "apply_overrides",
    # logging
    "ExperimentRecord",
    "PredictionRecord",
    "save_jsonl",
    "load_jsonl",
    "save_records",
    "save_predictions_npz",
    "load_predictions_npz",
    "new_experiment_id",
    "utc_now_iso",
    "resolve_git_commit",
    # io
    "ensure_dir",
    "require_file",
    "save_npz",
    "load_npz",
    "write_json",
    "read_json",
    "file_sha256",
    "cache_path",
    "iter_chunks",
]
