"""Small IO helpers shared by the cache writers and the scripts.

Deliberately thin: the cache formats themselves are owned by
:mod:`ccg.data.proposals` (HDF5 preferred, NPZ fallback).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

__all__ = [
    "ensure_dir",
    "save_npz",
    "load_npz",
    "write_json",
    "read_json",
    "file_sha256",
    "cache_path",
    "iter_chunks",
    "require_file",
]


def ensure_dir(path: str | Path) -> Path:
    """Create (and return) a directory, tolerating an existing one."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_file(path: str | Path, *, hint: str = "") -> Path:
    """Return ``path`` or raise with the Phase 0 checklist context attached."""
    file_path = Path(path)
    default_hint = (
        "This artefact is produced by an earlier Phase 0 checklist step "
        "(data download -> proposals -> features -> candidate sets)."
    )
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist. {hint or default_hint}")
    return file_path


def save_npz(path: str | Path, **arrays: Any) -> Path:
    """Compressed ``.npz`` writer (dicts/lists become pickled object arrays)."""
    path = Path(path)
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            payload[key] = value
        elif isinstance(value, (dict, list, tuple)):
            payload[key] = np.asarray(value, dtype=object)
        else:
            payload[key] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return path


def load_npz(path: str | Path, *, allow_pickle: bool = False) -> Dict[str, Any]:
    with np.load(str(path), allow_pickle=allow_pickle) as store:
        return {key: store[key] for key in store.files}


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False, sort_keys=True, default=float)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Checksum of a cache file, recorded so a stale cache is detectable."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def cache_path(cache_root: str | Path, name: str) -> Path:
    """Canonical location for a cache artefact (``cache/`` stays out of git)."""
    root = Path(cache_root)
    ensure_dir(root)
    return root / name


def iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterable[List[Any]]:
    """Yield consecutive chunks - the batch loop of every extraction script."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    items = list(items)
    for start in range(0, len(items), int(chunk_size)):
        yield items[start : start + int(chunk_size)]
