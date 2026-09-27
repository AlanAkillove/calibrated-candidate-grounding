"""Feature-cache storage for the frozen OpenCLIP embeddings (protocol §13).

Cache layout (paths relative to the cache *root* directory, e.g.
``cache/features``)::

    region_features.h5   "features"          [total_crops, 512] float16
                         "image_offsets"     [M, 3] int64  (image_id, start, count)
                         attr: "cache_version" (per-image counts live in image_offsets)
    text_features.h5     "features"          [n_sentences, 512] float16
                         "sentence_offsets"  [n_sentences, 2] int64 (sentence_id, row)
                         "texts"             variable-length utf-8 (one per row)
    global_features.h5   "features"          [n_images, 512] float16
                         "image_ids"         [n_images] int64
    metadata.json        provenance + extraction stats (frozen schema, §5 brief)
    text_index.csv       sentence_id, ref_id, sent_id, image_id, split, text

Design rules (frozen multi-agent interface, 2026-09-27):

* the row order of ``region_features.h5`` for one image is exactly the
  proposal-bank order of that image; a crop that round->clamp leaves smaller
  than 1 px ("invalid") is stored as an all-zero row (never silently dropped,
  never geometrically expanded);
* :class:`FeatureCache` keeps the HDF5 handles plus ``id -> offset``
  dictionaries in memory, so every lookup is O(1) (one seek + one contiguous
  block read);
* every writer is atomic: data goes to ``<name>.tmp`` first and only a
  finished file is promoted with :func:`os.replace`;
* the streaming writers exist so a 1.3 GB region extraction can be resumed
  after a crash without re-encoding finished images (``--resume``).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..utils.io import read_json, write_json
from .clip_encoder import FEATURE_DIM

__all__ = [
    "CACHE_VERSION",
    "REGION_FILENAME",
    "TEXT_FILENAME",
    "GLOBAL_FILENAME",
    "METADATA_FILENAME",
    "TEXT_INDEX_FILENAME",
    "TEXT_INDEX_COLUMNS",
    "FeatureCache",
    "StreamingRegionWriter",
    "StreamingGlobalWriter",
    "write_feature_cache",
    "write_region_cache",
    "write_text_cache",
    "write_global_cache",
    "write_text_index",
    "read_text_index",
    "update_metadata",
    "merge_extraction_stats",
    "read_metadata",
    "feature_health",
    "atomic_replace",
    "write_json_atomic",
    "write_csv_atomic",
]

#: Frozen cache schema version (Phase 0a).
CACHE_VERSION = "phase0a-v1"

REGION_FILENAME = "region_features.h5"
TEXT_FILENAME = "text_features.h5"
GLOBAL_FILENAME = "global_features.h5"
METADATA_FILENAME = "metadata.json"
TEXT_INDEX_FILENAME = "text_index.csv"

#: Frozen column order of ``text_index.csv``.
TEXT_INDEX_COLUMNS: Tuple[str, ...] = (
    "sentence_id",
    "ref_id",
    "sent_id",
    "image_id",
    "split",
    "text",
)


# ---------------------------------------------------------------------------
# small IO helpers (atomic writes; h5py imported lazily)
# ---------------------------------------------------------------------------


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "h5py is required for the feature cache; it is declared in pyproject.toml "
            "(pip install h5py)"
        ) from exc
    return h5py


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def atomic_replace(tmp_path: str | Path, final_path: str | Path) -> Path:
    """Promote a fully written ``.tmp`` file onto its final name (atomic)."""
    tmp = Path(tmp_path)
    final = Path(final_path)
    if not tmp.exists():
        raise FileNotFoundError(f"{tmp} does not exist; nothing to promote")
    os.replace(str(tmp), str(final))
    return final


def write_json_atomic(path: str | Path, data) -> Path:
    """``json.dump`` to ``<path>.tmp`` then rename (never a half-written json)."""
    path = Path(path)
    tmp = _tmp_path(path)
    write_json(data, tmp)
    return atomic_replace(tmp, path)


def write_csv_atomic(path: str | Path, fieldnames: Sequence[str], rows: Iterable[Mapping]) -> Path:
    """``csv.DictWriter`` to ``<path>.tmp`` then rename (fixed column order)."""
    path = Path(path)
    tmp = _tmp_path(path)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return atomic_replace(tmp, path)


def _feature_matrix(value, name: str) -> np.ndarray:
    """Validate one ``[K, FEATURE_DIM]`` block and cast to float16 (storage dtype)."""
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected a 2-d array, got shape {arr.shape}")
    if arr.shape[1] != FEATURE_DIM:
        raise ValueError(f"{name}: expected {FEATURE_DIM} columns, got {arr.shape[1]}")
    return np.ascontiguousarray(arr, dtype=np.float16)


def _feature_row(value, name: str) -> np.ndarray:
    """Validate one ``[FEATURE_DIM]`` vector and cast to float16."""
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.shape[0] != FEATURE_DIM:
        raise ValueError(f"{name}: expected shape ({FEATURE_DIM},), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float16)


def _int_key(key, name: str) -> int:
    try:
        return int(key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: non-integer id {key!r}") from exc


# ---------------------------------------------------------------------------
# one-shot writers (per part + the frozen all-in-one entry point)
# ---------------------------------------------------------------------------


def _write_region_file(path_tmp: Path, region: Mapping[int, np.ndarray]) -> np.ndarray:
    h5py = _h5py()
    items = sorted(((int(k), v) for k, v in region.items()), key=lambda kv: kv[0])
    blocks = [_feature_matrix(v, f"region[{k}]") for k, v in items]
    counts = np.asarray([b.shape[0] for b in blocks], dtype=np.int64)
    offsets = np.zeros((len(items), 3), dtype=np.int64)
    if len(items):
        offsets[:, 0] = [k for k, _ in items]
        offsets[:, 1] = np.concatenate([[0], np.cumsum(counts)[:-1]])
        offsets[:, 2] = counts
    features = (
        np.concatenate(blocks, axis=0) if blocks else np.zeros((0, FEATURE_DIM), np.float16)
    )
    path_tmp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path_tmp), "w") as handle:
        handle.create_dataset("features", data=features.astype(np.float16, copy=False))
        handle.create_dataset("image_offsets", data=offsets)
        handle.attrs["cache_version"] = CACHE_VERSION
    return counts


def _write_text_file(
    path_tmp: Path, text: Mapping[int, np.ndarray], texts: Optional[Sequence[str]] = None
) -> None:
    h5py = _h5py()
    items = sorted(((int(k), v) for k, v in text.items()), key=lambda kv: kv[0])
    rows = np.asarray([_feature_row(v, f"text[{k}]") for k, v in items], dtype=np.float16)
    features = rows if rows.size else np.zeros((0, FEATURE_DIM), np.float16)
    offsets = np.zeros((len(items), 2), dtype=np.int64)
    if len(items):
        offsets[:, 0] = [k for k, _ in items]
        offsets[:, 1] = np.arange(len(items), dtype=np.int64)
    if texts is None:
        text_values: List[str] = ["" for _ in items]
    else:
        text_values = list(texts)
        if len(text_values) != len(items):
            raise ValueError(f"texts: got {len(text_values)} strings for {len(items)} vectors")
    path_tmp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path_tmp), "w") as handle:
        handle.create_dataset("features", data=features.astype(np.float16, copy=False))
        handle.create_dataset("sentence_offsets", data=offsets)
        handle.create_dataset(
            "texts",
            data=np.asarray(text_values, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        handle.attrs["cache_version"] = CACHE_VERSION


def _write_global_file(path_tmp: Path, global_: Mapping[int, np.ndarray]) -> None:
    h5py = _h5py()
    items = sorted(((int(k), v) for k, v in global_.items()), key=lambda kv: kv[0])
    rows = np.asarray([_feature_row(v, f"global[{k}]") for k, v in items], dtype=np.float16)
    features = rows if rows.size else np.zeros((0, FEATURE_DIM), np.float16)
    image_ids = np.asarray([k for k, _ in items], dtype=np.int64)
    path_tmp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path_tmp), "w") as handle:
        handle.create_dataset("features", data=features.astype(np.float16, copy=False))
        handle.create_dataset("image_ids", data=image_ids)
        handle.attrs["cache_version"] = CACHE_VERSION


def write_region_cache(
    root: str | Path, region: Mapping[int, np.ndarray], metadata_updates: Optional[dict] = None
) -> Path:
    """Atomically write ``region_features.h5`` from ``{image_id -> [K, 512]}``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / REGION_FILENAME
    _write_region_file(_tmp_path(final), region)
    atomic_replace(_tmp_path(final), final)
    if metadata_updates:
        update_metadata(root, metadata_updates)
    return final


def write_text_cache(
    root: str | Path,
    text: Mapping[int, np.ndarray],
    texts: Optional[Sequence[str]] = None,
    metadata_updates: Optional[dict] = None,
) -> Path:
    """Atomically write ``text_features.h5`` from ``{sentence_id -> [512]}``.

    ``texts`` (if given) must be aligned with the *sorted* sentence ids and is
    stored in the ragged-string dataset of the same file.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / TEXT_FILENAME
    _write_text_file(_tmp_path(final), text, texts)
    atomic_replace(_tmp_path(final), final)
    if metadata_updates:
        update_metadata(root, metadata_updates)
    return final


def write_global_cache(
    root: str | Path, global_: Mapping[int, np.ndarray], metadata_updates: Optional[dict] = None
) -> Path:
    """Atomically write ``global_features.h5`` from ``{image_id -> [512]}``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / GLOBAL_FILENAME
    _write_global_file(_tmp_path(final), global_)
    atomic_replace(_tmp_path(final), final)
    if metadata_updates:
        update_metadata(root, metadata_updates)
    return final


def write_feature_cache(
    root: str | Path,
    region: Dict[int, np.ndarray],
    text: Dict[int, np.ndarray],
    global_: Dict[int, np.ndarray],
    metadata: dict,
) -> None:
    """Frozen entry point: write the whole cache layout at once (tests / small data).

    * ``region``: ``{image_id -> [K, 512]}``, row order == proposal bank order;
    * ``text``: ``{sentence_id -> [512]}``;
    * ``global_``: ``{image_id -> [512]}``;
    * ``metadata``: provenance dict; ``cache_version`` is injected when absent
      and the counts (``n_images``, ``n_crops``, ``n_sentences``) are always
      recomputed from the arrays so the json cannot drift from the files.

    ``text_index.csv`` is created header-only here: the real rows are written
    by :func:`write_text_index` (the CLI always does).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    region = dict(region or {})
    text = dict(text or {})
    global_ = dict(global_ or {})

    meta = dict(metadata or {})
    meta.setdefault("cache_version", CACHE_VERSION)
    n_crops = int(sum(np.asarray(v).reshape(-1).size // FEATURE_DIM for v in region.values()))
    meta["n_crops"] = n_crops
    meta["n_images"] = int(len(region) if region else len(global_))
    meta["n_sentences"] = int(len(text))

    _write_region_file(_tmp_path(root / REGION_FILENAME), region)
    _write_text_file(_tmp_path(root / TEXT_FILENAME), text, None)
    _write_global_file(_tmp_path(root / GLOBAL_FILENAME), global_)
    write_text_index(root, [])
    for final in (REGION_FILENAME, TEXT_FILENAME, GLOBAL_FILENAME):
        atomic_replace(_tmp_path(root / final), root / final)
    write_json_atomic(root / METADATA_FILENAME, meta)


# ---------------------------------------------------------------------------
# text index / metadata
# ---------------------------------------------------------------------------


def write_text_index(root: str | Path, rows: Iterable[Mapping]) -> Path:
    """Write ``root/text_index.csv`` (atomic, frozen column order)."""
    normalized: List[dict] = []
    for row in rows:
        missing = [c for c in TEXT_INDEX_COLUMNS if c not in row]
        if missing:
            raise ValueError(f"text index row misses {missing}: {dict(row)}")
        normalized.append(
            {
                "sentence_id": int(row["sentence_id"]),
                "ref_id": int(row["ref_id"]),
                "sent_id": int(row["sent_id"]),
                "image_id": int(row["image_id"]),
                "split": str(row["split"]),
                "text": str(row["text"]),
            }
        )
    return write_csv_atomic(Path(root) / TEXT_INDEX_FILENAME, TEXT_INDEX_COLUMNS, normalized)


def read_text_index(root: str | Path) -> List[dict]:
    """Read ``root/text_index.csv`` back (ints parsed, file order preserved)."""
    path = Path(root) / TEXT_INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist - the text cache has no index")
    out: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TEXT_INDEX_COLUMNS:
            raise ValueError(
                f"{path}: columns {reader.fieldnames} != frozen {list(TEXT_INDEX_COLUMNS)}"
            )
        for row in reader:
            out.append(
                {
                    "sentence_id": int(row["sentence_id"]),
                    "ref_id": int(row["ref_id"]),
                    "sent_id": int(row["sent_id"]),
                    "image_id": int(row["image_id"]),
                    "split": row["split"],
                    "text": row["text"],
                }
            )
    return out


def read_metadata(root: str | Path) -> dict:
    return read_json(Path(root) / METADATA_FILENAME)


def update_metadata(root: str | Path, updates: Mapping) -> dict:
    """Merge ``updates`` into ``metadata.json`` (atomic; creates the file)."""
    path = Path(root) / METADATA_FILENAME
    current: dict = read_json(path) if path.exists() else {}
    current.update(dict(updates))
    write_json_atomic(path, current)
    return current


def merge_extraction_stats(root: str | Path, updates: Mapping) -> dict:
    """Merge ``updates`` into ``metadata.json["extraction_stats"]`` (atomic).

    Used by the sub-CLIs so a standalone ``extract_text`` / ``extract_regions``
    / ``extract_image`` run cannot clobber the statistics another of them (or
    the orchestrator) already recorded.
    """
    path = Path(root) / METADATA_FILENAME
    meta: dict = read_json(path) if path.exists() else {}
    stats = dict(meta.get("extraction_stats") or {})
    stats.update(dict(updates))
    meta["extraction_stats"] = stats
    write_json_atomic(path, meta)
    return stats


# ---------------------------------------------------------------------------
# embedding health (used by the §9 audit; pure numpy)
# ---------------------------------------------------------------------------


def feature_health(features) -> dict:
    """NaN / Inf / zero-norm row counters for one ``[N, D]`` feature matrix.

    ``zero_norm_rows`` counts exactly all-zero rows - these are *expected* for
    the region cache only, where an invalid (degenerate) crop is stored as a
    zero row; the audit cross-checks their count against
    ``metadata.invalid_crop_count``.
    """
    arr = np.asarray(features)
    if arr.ndim != 2:
        raise ValueError(f"feature_health: expected a 2-d array, got shape {arr.shape}")
    total = int(arr.shape[0])
    if total == 0:
        return {"total_rows": 0, "nan_rows": 0, "inf_rows": 0, "zero_norm_rows": 0}
    as_f32 = arr.astype(np.float32, copy=False)
    nan_rows = int(np.isnan(as_f32).any(axis=1).sum())
    inf_rows = int(np.isinf(as_f32).any(axis=1).sum())
    zero_norm_rows = int((np.linalg.norm(as_f32, axis=1) == 0).sum())
    return {
        "total_rows": total,
        "nan_rows": nan_rows,
        "inf_rows": inf_rows,
        "zero_norm_rows": zero_norm_rows,
    }


# ---------------------------------------------------------------------------
# streaming writers (crash-resumable extraction of the big caches)
# ---------------------------------------------------------------------------


class _StreamingBase:
    """Shared append-only machinery: features [rows, 512] + an own id index."""

    #: dataset name of the per-row id index written by :meth:`flush`
    index_name = ""

    def __init__(self, path: str | Path, *, resume: bool = False, flush_every: int = 64) -> None:
        self.path = Path(path)
        self.tmp_path = _tmp_path(self.path)
        self.flush_every = max(1, int(flush_every))
        self._tmp_handle = None
        self._final_handle = None
        self.processed: Dict[int, Tuple[int, int]] = {}
        self._append_order: List[int] = []
        self.source = "fresh"

        if resume and self.tmp_path.exists():
            self._open_tmp_for_append()
            self.source = "tmp"
        elif resume and self.path.exists():
            self.processed = self._read_index(self.path)
            self.source = "final"
        else:
            if self.tmp_path.exists():
                self.tmp_path.unlink()
            self._create_tmp()

    # -- index layout ------------------------------------------------------
    def _read_index(self, file_path: Path) -> Dict[int, Tuple[int, int]]:
        raise NotImplementedError

    def _index_array(self) -> np.ndarray:
        """The full index to write into the tmp file (rows in append order)."""
        raise NotImplementedError

    def _mark_written(self, image_id: int, rows: int) -> None:
        raise NotImplementedError

    def _validate_rows(self, image_id: int, rows) -> int:
        raise NotImplementedError

    # -- tmp lifecycle -----------------------------------------------------
    def _create_tmp(self) -> None:
        h5py = _h5py()
        self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
        handle = h5py.File(str(self.tmp_path), "w")
        handle.create_dataset(
            "features",
            shape=(0, FEATURE_DIM),
            maxshape=(None, FEATURE_DIM),
            dtype=np.float16,
            chunks=(8192, FEATURE_DIM),
        )
        self._create_index_dataset(handle)
        handle.attrs["cache_version"] = CACHE_VERSION
        self._tmp_handle = handle

    def _create_index_dataset(self, handle) -> None:
        raise NotImplementedError

    def _open_tmp_for_append(self) -> None:
        h5py = _h5py()
        handle = h5py.File(str(self.tmp_path), "r+")
        processed = self._read_index_from_handle(handle)
        committed = 0
        for image_id, (start, count) in processed.items():
            committed = max(committed, start + count)
        features = handle["features"]
        if features.shape[0] != committed:
            # drop rows appended after the last index flush (crash recovery)
            features.resize(committed, axis=0)
        handle.flush()
        self.processed = processed
        self._append_order = [image_id for image_id, _ in sorted(processed.items(), key=lambda kv: kv[1][0])]
        self._tmp_handle = handle

    def _read_index_from_handle(self, handle) -> Dict[int, Tuple[int, int]]:
        raise NotImplementedError

    # -- public API --------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return int(self._tmp_handle["features"].shape[0]) if self._tmp_handle is not None else 0

    def covers(self, image_ids: Iterable[int]) -> bool:
        return all(int(i) in self.processed for i in image_ids)

    def append(self, image_id: int, rows) -> np.ndarray:
        """Append the (validated) feature block of one image; returns stored rows."""
        if self._tmp_handle is None:
            raise RuntimeError("writer is not in append mode (adopt_final() or abandon() first)")
        image_id = int(image_id)
        if image_id in self.processed:
            raise ValueError(f"image_id {image_id} is already in the cache")
        if self._append_order and image_id <= self._append_order[-1]:
            raise ValueError(
                f"append order must be ascending by image_id: {image_id} after "
                f"{self._append_order[-1]} (would corrupt resume semantics)"
            )
        stored = self._validate_rows(image_id, rows)
        count = int(stored.shape[0])
        start = self.n_rows
        features = self._tmp_handle["features"]
        if count:
            features.resize(start + count, axis=0)
            features[start : start + count] = stored
        self._mark_written(image_id, count)
        self.processed[image_id] = (start, count)
        self._append_order.append(image_id)
        if len(self._append_order) % self.flush_every == 0:
            self.flush()
        return stored

    def flush(self) -> None:
        """Persist the id index so a crashed run can resume from this point."""
        if self._tmp_handle is None or not self._append_order:
            return
        handle = self._tmp_handle
        index = self._index_array()
        dataset = handle[self.index_name]
        dataset.resize(index.shape[0], axis=0)
        dataset[...] = index
        # NB: never mirror the per-image counts into a file *attribute* - HDF5
        # object-header attributes cap out near 64 KiB, so a "counts" array dies
        # at ~8192 images (seen live: "object header message is too large").
        # The resumable index is this dataset; counts are the third column of
        # image_offsets and implicit (all ones) in image_ids.
        handle.flush()

    def finish(self) -> Path:
        """Flush, close and atomically promote ``.tmp`` onto the final name."""
        if self._tmp_handle is not None:
            self.flush()
            self._tmp_handle.close()
            self._tmp_handle = None
        if self._final_handle is not None:
            self._final_handle.close()
            self._final_handle = None
        return atomic_replace(self.tmp_path, self.path)

    def abandon(self) -> None:
        """Close handles without promoting the tmp file (skip path of ``--resume``)."""
        for attr in ("_tmp_handle", "_final_handle"):
            handle = getattr(self, attr)
            if handle is not None:
                handle.close()
                setattr(self, attr, None)

    def adopt_final(self) -> None:
        """Copy every block of an existing *final* file into a fresh tmp, then append."""
        if self.source != "final":
            raise RuntimeError(f"adopt_final() only valid when resuming from final (got {self.source})")
        h5py = _h5py()
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self._create_tmp()
        with h5py.File(str(self.path), "r") as source:
            order = sorted(self.processed.items(), key=lambda kv: kv[1][0])
            blocks = []
            for image_id, (start, count) in order:
                blocks.append((image_id, source["features"][start : start + count]))
        self.processed = {}
        self._append_order = []
        for image_id, block in blocks:
            self.append(image_id, block)
        self.flush()
        self.source = "adopted"

    def __enter__(self):  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc):  # pragma: no cover - convenience
        self.abandon()


class StreamingRegionWriter(_StreamingBase):
    """Incremental ``region_features.h5`` writer (resume-aware, atomic)."""

    index_name = "image_offsets"

    def _create_index_dataset(self, handle) -> None:
        handle.create_dataset(
            "image_offsets", shape=(0, 3), maxshape=(None, 3), dtype=np.int64
        )

    def _read_index_from_handle(self, handle) -> Dict[int, Tuple[int, int]]:
        return _offsets_to_index(handle["image_offsets"][...])

    def _read_index(self, file_path: Path) -> Dict[int, Tuple[int, int]]:
        with _h5py().File(str(file_path), "r") as handle:
            return _offsets_to_index(handle["image_offsets"][...])

    def _validate_rows(self, image_id: int, rows) -> np.ndarray:
        return _feature_matrix(rows, f"region[{image_id}]")

    def _mark_written(self, image_id: int, count: int) -> None:  # index rebuilt from processed
        return None

    def _index_array(self) -> np.ndarray:
        out = np.zeros((len(self._append_order), 3), dtype=np.int64)
        for position, image_id in enumerate(self._append_order):
            start, count = self.processed[image_id]
            out[position] = (image_id, start, count)
        return out


class StreamingGlobalWriter(_StreamingBase):
    """Incremental ``global_features.h5`` writer (resume-aware, atomic)."""

    index_name = "image_ids"

    def _create_index_dataset(self, handle) -> None:
        handle.create_dataset("image_ids", shape=(0,), maxshape=(None,), dtype=np.int64)

    def _read_index_from_handle(self, handle) -> Dict[int, Tuple[int, int]]:
        ids = np.asarray(handle["image_ids"][...], dtype=np.int64)
        return {int(image_id): (row, 1) for row, image_id in enumerate(ids)}

    def _read_index(self, file_path: Path) -> Dict[int, Tuple[int, int]]:
        with _h5py().File(str(file_path), "r") as handle:
            return self._read_index_from_handle(handle)

    def _validate_rows(self, image_id: int, rows) -> np.ndarray:
        arr = np.asarray(rows, dtype=np.float16)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (1, FEATURE_DIM):
            raise ValueError(f"global[{image_id}]: expected one ({FEATURE_DIM},) row, got {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.float16)

    def _mark_written(self, image_id: int, count: int) -> None:
        return None

    def _index_array(self) -> np.ndarray:
        return np.asarray(self._append_order, dtype=np.int64)


def _offsets_to_index(offsets: np.ndarray) -> Dict[int, Tuple[int, int]]:
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 3)
    index: Dict[int, Tuple[int, int]] = {}
    for image_id, start, count in offsets.tolist():
        if image_id in index:
            raise ValueError(f"duplicate image_id {image_id} in image_offsets")
        index[int(image_id)] = (int(start), int(count))
    return index


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------


class FeatureCache:
    """O(1) random access to one feature-cache root (see module docstring).

    All three ``.h5`` files and ``metadata.json`` are opened read-only and
    stay open until :meth:`close` (or the ``with`` block ends).  Missing ids
    raise :class:`KeyError`; a missing file raises :class:`FileNotFoundError`
    with the file name - partial caches are never silently tolerated by the
    accessor that needs the missing part.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._h5_files: Dict[str, object] = {}
        self._region_index: Dict[int, Tuple[int, int]] = {}
        self._sentence_index: Dict[int, int] = {}
        self._global_index: Dict[int, int] = {}

    # -- construction ------------------------------------------------------
    @classmethod
    def open(cls, root: str | Path) -> "FeatureCache":
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"{root} does not exist - nothing to open")
        metadata_path = root / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(f"{metadata_path} is missing - not a feature cache root")
        cache = cls(root)
        cache.metadata = read_json(metadata_path)

        region_path = root / REGION_FILENAME
        if region_path.exists():
            handle = _h5py().File(str(region_path), "r")
            cache._h5_files["region"] = handle
            offsets = np.asarray(handle["image_offsets"][...], dtype=np.int64).reshape(-1, 3)
            cache._region_index = _offsets_to_index(offsets)
            _validate_region_offsets(offsets, int(handle["features"].shape[0]), region_path)
        text_path = root / TEXT_FILENAME
        if text_path.exists():
            handle = _h5py().File(str(text_path), "r")
            cache._h5_files["text"] = handle
            offsets = np.asarray(handle["sentence_offsets"][...], dtype=np.int64).reshape(-1, 2)
            cache._sentence_index = _sentences_to_index(offsets, int(handle["features"].shape[0]))
        global_path = root / GLOBAL_FILENAME
        if global_path.exists():
            handle = _h5py().File(str(global_path), "r")
            cache._h5_files["global"] = handle
            ids = np.asarray(handle["image_ids"][...], dtype=np.int64).reshape(-1)
            if ids.size != int(handle["features"].shape[0]):
                raise ValueError(
                    f"{global_path}: image_ids has {ids.size} entries for "
                    f"{int(handle['features'].shape[0])} feature rows"
                )
            cache._global_index = {int(image_id): row for row, image_id in enumerate(ids)}
            if len(cache._global_index) != ids.size:
                raise ValueError(f"{global_path}: duplicate image_ids")
        return cache

    def _file(self, part: str, filename: str):
        handle = self._h5_files.get(part)
        if handle is None:
            raise FileNotFoundError(
                f"{self.root / filename} is not part of this cache (partial extraction?)"
            )
        return handle

    # -- properties --------------------------------------------------------
    @property
    def n_images(self) -> int:
        """Images with region features (falls back to the global cache)."""
        if "region" in self._h5_files:
            return len(self._region_index)
        if "global" in self._h5_files:
            return len(self._global_index)
        return 0

    @property
    def n_sentences(self) -> int:
        return len(self._sentence_index)

    @property
    def region_offsets(self) -> np.ndarray:
        handle = self._file("region", REGION_FILENAME)
        return np.asarray(handle["image_offsets"][...], dtype=np.int64).reshape(-1, 3)

    @property
    def sentence_offsets(self) -> np.ndarray:
        handle = self._file("text", TEXT_FILENAME)
        return np.asarray(handle["sentence_offsets"][...], dtype=np.int64).reshape(-1, 2)

    @property
    def global_image_ids(self) -> np.ndarray:
        handle = self._file("global", GLOBAL_FILENAME)
        return np.asarray(handle["image_ids"][...], dtype=np.int64).reshape(-1)

    # -- random access (O(1)) ---------------------------------------------
    def region_features(self, image_id: int) -> np.ndarray:
        """``[N_img, 512]`` float16 rows of one image, in proposal-bank order."""
        handle = self._file("region", REGION_FILENAME)
        image_id = int(image_id)
        if image_id not in self._region_index:
            raise KeyError(f"image_id {image_id} not in {self.root / REGION_FILENAME}")
        start, count = self._region_index[image_id]
        return np.asarray(handle["features"][start : start + count], dtype=np.float16)

    def text_features(self, sentence_id: int) -> np.ndarray:
        """``[512]`` float16 embedding of one sentence."""
        handle = self._file("text", TEXT_FILENAME)
        sentence_id = int(sentence_id)
        if sentence_id not in self._sentence_index:
            raise KeyError(f"sentence_id {sentence_id} not in {self.root / TEXT_FILENAME}")
        row = self._sentence_index[sentence_id]
        return np.asarray(handle["features"][row], dtype=np.float16)

    def global_feature(self, image_id: int) -> np.ndarray:
        """``[512]`` float16 whole-image embedding."""
        handle = self._file("global", GLOBAL_FILENAME)
        image_id = int(image_id)
        if image_id not in self._global_index:
            raise KeyError(f"image_id {image_id} not in {self.root / GLOBAL_FILENAME}")
        row = self._global_index[image_id]
        return np.asarray(handle["features"][row], dtype=np.float16)

    def region_ids(self) -> np.ndarray:
        """Sorted int64 array of the image ids covered by the region cache."""
        return np.sort(np.asarray(sorted(self._region_index), dtype=np.int64))

    def sentence_ids(self) -> np.ndarray:
        return np.sort(np.asarray(sorted(self._sentence_index), dtype=np.int64))

    def global_ids(self) -> np.ndarray:
        return np.sort(np.asarray(sorted(self._global_index), dtype=np.int64))

    def text(self, sentence_id: int) -> str:
        """The stored raw text of one sentence (``""`` when not recorded)."""
        handle = self._file("text", TEXT_FILENAME)
        sentence_id = int(sentence_id)
        if sentence_id not in self._sentence_index:
            raise KeyError(f"sentence_id {sentence_id} not in {self.root / TEXT_FILENAME}")
        row = self._sentence_index[sentence_id]
        value = handle["texts"][row]
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        for handle in self._h5_files.values():
            handle.close()
        self._h5_files.clear()

    def __enter__(self) -> "FeatureCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _validate_region_offsets(offsets: np.ndarray, n_rows: int, path: Path) -> None:
    if offsets.size == 0:
        if n_rows:
            raise ValueError(f"{path}: {n_rows} feature rows but an empty image_offsets")
        return
    starts = offsets[:, 1]
    counts = offsets[:, 2]
    if (counts < 0).any():
        raise ValueError(f"{path}: negative count in image_offsets")
    expected = np.concatenate([[0], np.cumsum(counts)[:-1]])
    if not np.array_equal(starts, expected):
        raise ValueError(f"{path}: image_offsets are not contiguous (start != cumsum)")
    if int(starts[-1] + counts[-1]) != n_rows:
        raise ValueError(f"{path}: image_offsets cover {int(starts[-1] + counts[-1])} of {n_rows} rows")


def _sentences_to_index(offsets: np.ndarray, n_rows: int) -> Dict[int, int]:
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
    if offsets.shape[0] != n_rows:
        raise ValueError(f"sentence_offsets has {offsets.shape[0]} entries for {n_rows} rows")
    if not np.array_equal(offsets[:, 1], np.arange(offsets.shape[0], dtype=np.int64)):
        raise ValueError("sentence_offsets rows must be 0..n-1 (contiguous row numbers)")
    index: Dict[int, int] = {}
    for sentence_id, row in offsets.tolist():
        if sentence_id in index:
            raise ValueError(f"duplicate sentence_id {sentence_id} in sentence_offsets")
        index[int(sentence_id)] = int(row)
    return index
