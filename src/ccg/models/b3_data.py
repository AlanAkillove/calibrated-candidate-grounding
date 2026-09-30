"""Phase 0B / B3 data pipeline: candidate-blind training + evaluation examples.

This module owns *everything data-side* for the B3 independent-MLP study
(:class:`ccg.models.independent.IndependentMLPScorer`).  It re-uses the frozen
Phase 0A interfaces read-only and never re-samples a candidate set:

* the candidate sets are exactly ``C_K = [target] + distractor_order[:K-1]``
  of the frozen manifests - the target is always the *local* index ``0``
  (:data:`TARGET_LOCAL_INDEX`), the same definition Phase 0A scores;
* the region / text embeddings come from :class:`ccg.features.cache.FeatureCache`
  (row order per image == proposal-bank order);
* the proposal boxes / objectness come from the bank written by
  :mod:`ccg.data.bank` (``cache/proposals.h5``);
* the evaluation row set and its order are produced *by* Phase 0A itself
  (:func:`ccg.experiment.phase0a.load_cosine_inputs`), never re-derived here -
  so a B3 eval pass iterates the identical ``SentenceRecord`` list that
  ``phase0a.score_sets`` consumes.

Independence guarantee (the whole point of B3)
----------------------------------------------
Every input row is a function of *one* candidate only::

    x_i = [ z_q , z_i , z_q * z_i , cos(z_q, z_i) , g_i ]

with ``g_i`` the geometry of candidate ``i`` alone.  Concretely:

* no input row carries ``K`` - there is no candidate-set-size feature;
* no input row carries any aggregate (mean / max / softmax / rank / std) of the
  other candidates;
* ``g_i`` is normalised by the *true* image size ``(W, H)`` read from the image
  file, never by a statistic of the set (see
  :func:`geometry_features_centered`); a missing size is an error, never a
  silent default;
* the same ``(query, candidate)`` pair yields bit-identical
  ``candidate_features`` / ``geometry`` rows whatever ``K`` they are sliced
  into (they are cast from the *same* stored ``float16`` block), so the only
  ``K`` that ever reaches the model is the batch *shape*.

Documentation of the training/eval split: training iterators only ever emit
``random_train`` rows; evaluation iterators only ever emit the frozen
``val_select`` / ``val_calib`` / ``testA`` / ``testB`` rows of Phase 0A.
"""

from __future__ import annotations

import csv
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..data.manifests import ManifestFile
from ..features.cache import (
    REGION_FILENAME,
    TEXT_FILENAME,
    FeatureCache,
    read_text_index,
)

__all__ = [
    "FEATURE_DIM",
    "GEO_DIM",
    "GEO_FIELDS",
    "TARGET_LOCAL_INDEX",
    "TRAIN_KS",
    "EVAL_KS",
    "DEFAULT_REGION_CACHE_SIZE",
    "DEFAULT_BANK_CACHE_SIZE",
    "IMAGE_SIZES_FILENAME",
    "TRAIN_MANIFEST_NAME",
    "B3Example",
    "TrainSample",
    "B3Corpus",
    "build_image_sizes",
    "load_image_sizes",
    "geometry_features_centered",
    "read_image_manifest_rows",
]

#: OpenCLIP ViT-B/32 joint embedding size (frozen, protocol §3).
FEATURE_DIM = 512
#: Geometry block of B3: centred ``x, y, w, h`` + relative area + objectness.
GEO_DIM = 6
GEO_FIELDS: Tuple[str, ...] = ("cx", "cy", "w", "h", "area", "objectness")
#: The target always sits at local candidate index 0 (``C_K`` layout).
TARGET_LOCAL_INDEX = 0
#: The two candidate-set sizes mixed 50:50 during B3 training.
TRAIN_KS: Tuple[int, ...] = (5, 10)
#: The primary candidate-set sizes of the audit (Phase 0A frozen set).
EVAL_KS: Tuple[int, ...] = (5, 10, 20, 50)
#: Per-process LRU capacity for region feature blocks (number of images).
DEFAULT_REGION_CACHE_SIZE = 2048
#: Per-process LRU capacity for proposal-bank blocks (number of images).
DEFAULT_BANK_CACHE_SIZE = 4096
#: Derived cache file holding ``{image_id: (W, H)}``.
IMAGE_SIZES_FILENAME = "image_sizes.npz"
#: Training manifest filename under the manifests root.
TRAIN_MANIFEST_NAME = "random_train.jsonl"


# ---------------------------------------------------------------------------
# image sizes (derived cache)
# ---------------------------------------------------------------------------
def read_image_manifest_rows(manifest_csv: Union[str, Path]) -> List[Tuple[int, str]]:
    """``data/full_image_manifest.csv`` -> ``[(image_id, "<split>/<file_name>")]``.

    The relative path is exactly the on-disk layout under ``data/raw/mscoco``;
    the columns ``image_id`` / ``file_name`` / ``coco_split`` must be present
    (a malformed row raises instead of being skipped).
    """
    path = Path(manifest_csv)
    if not path.exists():
        raise FileNotFoundError(f"image manifest {path} not found")
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [
            key
            for key in ("image_id", "file_name", "coco_split")
            if key not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"{path}: manifest misses columns {missing}")
        for line_no, row in enumerate(reader, start=2):
            try:
                image_id = int(row["image_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: non-integer image_id {row['image_id']!r}") from exc
            relative = f"{row['coco_split']}/{row['file_name']}"
            rows.append((image_id, relative))
    if not rows:
        raise ValueError(f"{path}: no image rows")
    return rows


def _read_pil_size(path: Path) -> Tuple[int, int]:
    """``(W, H)`` of one image read from its header (PIL ``Image.size``)."""
    from PIL import Image  # local import: PIL is only needed here / by the encoder

    if not path.exists():
        raise FileNotFoundError(f"image {path} not found")
    with Image.open(path) as handle:
        width, height = handle.size
    if width <= 0 or height <= 0:
        raise ValueError(f"image {path} reports a degenerate size {(width, height)}")
    return int(width), int(height)


def build_image_sizes(
    images_root: Union[str, Path],
    manifest_csv: Union[str, Path],
    out_path: Union[str, Path] = f"cache/{IMAGE_SIZES_FILENAME}",
    *,
    workers: int = 8,
    force: bool = False,
    log: Optional[Any] = None,
) -> Path:
    """Build (or return) the derived ``{image_id: (W, H)}`` cache as an ``.npz``.

    Reads ``manifest_csv`` (``image_id, file_name, coco_split``), resolves each
    image as ``images_root/<coco_split>/<file_name>`` and stores its *header*
    size with PIL.  The ``.npz`` keys are the ``image_id`` strings, the values
    are ``int64[2] == [W, H]``.  Atomic write (``.tmp`` then ``os.replace``).

    Idempotent: when ``out_path`` already exists and ``force`` is false the
    existing file is returned untouched.  A missing/unreadable image aborts
    (never a silent default), so the cache can never silently drop a size.
    """
    out = Path(out_path)
    if out.exists() and not force:
        return out
    root = Path(images_root)
    rows = read_image_manifest_rows(manifest_csv)
    workers = max(1, int(workers))

    sizes: Dict[int, Tuple[int, int]] = {}
    errors: List[Tuple[int, str]] = []

    def _one(item: Tuple[int, str]) -> Tuple[int, Tuple[int, int]]:
        image_id, relative = item
        return image_id, _read_pil_size(root / relative)

    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, row): row[0] for row in rows}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="image-sizes", unit="img", disable=None
        ):
            image_id = futures[future]
            try:
                key, value = future.result()
            except Exception as exc:  # noqa: BLE001 - collected, raised below
                errors.append((image_id, str(exc)))
                continue
            sizes[int(key)] = value

    if errors:
        shown = ", ".join(f"{i}({e})" for i, e in errors[:5])
        raise RuntimeError(
            f"{len(errors)} of {len(rows)} images could not be measured under {root}: {shown}"
            + (" ..." if len(errors) > 5 else "")
        )
    if len(sizes) != len(rows):
        raise RuntimeError(
            f"measured {len(sizes)} images but the manifest lists {len(rows)} "
            "(duplicate image_id in the manifest?)"
        )

    if log is not None:
        log(f"[build_image_sizes] measured {len(sizes)} images under {root}")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    payload = {str(k): np.asarray(v, dtype=np.int64) for k, v in sorted(sizes.items())}
    with open(tmp, "wb") as handle:
        np.savez(handle, **payload)
    os.replace(str(tmp), str(out))
    return out


def load_image_sizes(path: Union[str, Path] = f"cache/{IMAGE_SIZES_FILENAME}") -> Dict[int, Tuple[int, int]]:
    """Read :func:`build_image_sizes` output back as ``{image_id: (W, H)}``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"image-size cache {path} not found; build it with build_image_sizes() "
            "(scripts/train_b3.py does this automatically when the file is absent)"
        )
    out: Dict[int, Tuple[int, int]] = {}
    with np.load(path, allow_pickle=False) as store:
        for key in store.files:
            arr = np.asarray(store[key]).reshape(-1)
            if arr.size < 2:
                raise ValueError(f"{path}: key {key!r} does not hold (W, H)")
            out[int(key)] = (int(arr[0]), int(arr[1]))
    return out


# ---------------------------------------------------------------------------
# geometry (candidate-independent by construction)
# ---------------------------------------------------------------------------
def geometry_features_centered(
    boxes: np.ndarray,
    image_size: Optional[Sequence[float]],
    objectness: Optional[np.ndarray] = None,
) -> np.ndarray:
    """B3 geometry block ``[K, 6]`` of ``xyxy`` boxes: ``[cx/W, cy/H, w/W, h/H, area/(W*H), obj]``.

    ``cx, cy`` are the box *centre*; ``w, h`` the box extent; ``area`` the
    relative box area.  Normalisation uses the **true** image size
    ``image_size=(W, H)`` - a missing (``None``) or non-positive size raises,
    there is deliberately no default side (a set-derived rescaling would couple
    the candidates and destroy B3's independence).

    ``objectness`` (the detector score, one per box) is the sixth column, or
    ``0`` when not supplied.  Everything is float32 and per-row: row ``i``
    depends on box ``i`` (and ``objectness[i]``) only.
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"boxes must be [K,4] xyxy, got {arr.shape}")
    if image_size is None:
        raise ValueError(
            "image_size=(W,H) is required for geometry_features_centered; a missing "
            "true image size must be an error (never a silent default)"
        )
    width = float(image_size[0])
    height = float(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive (W,H), got {tuple(image_size)}")

    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]
    centre_x = (x1 + x2) * np.float32(0.5)
    centre_y = (y1 + y2) * np.float32(0.5)
    box_w = x2 - x1
    box_h = y2 - y1

    out = np.empty((arr.shape[0], GEO_DIM), dtype=np.float32)
    out[:, 0] = centre_x / width
    out[:, 1] = centre_y / height
    out[:, 2] = box_w / width
    out[:, 3] = box_h / height
    out[:, 4] = (box_w * box_h) / (width * height)
    if objectness is None:
        out[:, 5] = np.float32(0.0)
    else:
        scores = np.asarray(objectness, dtype=np.float32).reshape(-1)
        if scores.size != arr.shape[0]:
            raise ValueError(f"objectness has {scores.size} entries, expected {arr.shape[0]}")
        out[:, 5] = scores
    return out


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass
class B3Example:
    """One target-present B3 instance: a query and its ``C_K`` candidate set.

    ``candidate_features`` / ``geometry`` are ``[K, ...]`` and *locally*
    ordered so that the target is row ``target_index`` (always ``0``).  The
    arrays carry no set-level information whatsoever.
    """

    ref_id: int
    sentence_id: int
    image_id: int
    split: str
    query_feature: np.ndarray  # [512] float32
    candidate_features: np.ndarray  # [K, 512] float32
    geometry: np.ndarray  # [K, 6] float32
    target_index: int = TARGET_LOCAL_INDEX

    @property
    def K(self) -> int:
        return int(np.asarray(self.candidate_features).shape[0])

    @property
    def target_present(self) -> bool:
        return self.target_index is not None


@dataclass(eq=False)
class TrainSample:
    """A target-present training *row* (one sentence of one ``random_train`` ref).

    Holds only the frozen metadata needed to materialise a :class:`B3Example`
    for any ``K``: nothing about the query/candidates is loaded yet (lazy).
    """

    sentence_id: int
    ref_id: int
    image_id: int
    split: str
    target_index: int
    distractor_order: np.ndarray
    eligible: Mapping[int, bool] = field(default_factory=dict)


def _sample_split(sample: Any) -> str:
    """Split label of a training sample (``split``) or a Phase 0A record (``eval_split``)."""
    for attr in ("split", "eval_split"):
        value = getattr(sample, attr, None)
        if value is not None:
            return str(value)
    return ""


def _choose_train_k(seed: int, epoch: int, sentence_id: int, pool: Sequence[int]) -> int:
    """Deterministic 50:50 draw of the training ``K`` (per sentence, per epoch).

    ``np.random.default_rng([seed, epoch, sentence_id])`` makes the choice
    reproducible inside one epoch and *different* across epochs; the two
    candidates are drawn with probability 0.5 each (``pool`` is the 2-element
    ``(5, 10)`` pool by default).
    """
    pool = tuple(int(k) for k in pool)
    if not pool:
        raise ValueError("train K pool is empty")
    if len(pool) == 1:
        return int(pool[0])
    rng = np.random.default_rng([int(seed), int(epoch), int(sentence_id)])
    return int(pool[0]) if float(rng.random()) < 0.5 else int(pool[1])


def _train_pool(ks: Sequence[int]) -> Tuple[int, ...]:
    """The training K pool: the subset of ``ks`` inside :data:`TRAIN_KS` (default ``(5, 10)``)."""
    pool = tuple(sorted({int(k) for k in ks if int(k) in TRAIN_KS}))
    return pool if pool else TRAIN_KS


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
class B3Corpus:
    """Lazy B3 example factory over the frozen feature cache / manifests / bank.

    Construction is cheap; the feature cache, the image-size cache, the refs
    archive and the training index are all opened/derived on first use.  Region
    feature blocks and proposal-bank blocks are kept in bounded per-process
    LRUs (``region_cache_size`` / ``bank_cache_size`` images).
    """

    def __init__(
        self,
        features_root: Union[str, Path],
        manifests_root: Union[str, Path],
        refs: Any,
        bank_path: Union[str, Path],
        *,
        image_sizes_path: Union[str, Path] = f"cache/{IMAGE_SIZES_FILENAME}",
        region_cache_size: int = DEFAULT_REGION_CACHE_SIZE,
        bank_cache_size: int = DEFAULT_BANK_CACHE_SIZE,
        ks: Sequence[int] = EVAL_KS,
        regime: str = "random",
        text_index_path: Optional[Union[str, Path]] = None,
        preload: bool = False,
    ) -> None:
        self.features_root = Path(features_root)
        self.manifests_root = Path(manifests_root)
        self.refs = refs
        self.bank_path = Path(bank_path)
        self.image_sizes_path = Path(image_sizes_path)
        self.region_cache_size = max(1, int(region_cache_size))
        self.bank_cache_size = max(1, int(bank_cache_size))
        self.ks = tuple(sorted({int(k) for k in ks}))
        self.regime = str(regime)
        self._text_index_path = None if text_index_path is None else Path(text_index_path)

        #: RAM-resident stores set by :meth:`preload`; when present the matching
        #: LRU / on-demand path is bypassed (see :meth:`preload` for why).
        self._region_full: Optional[np.ndarray] = None
        self._region_slices: Optional[Dict[int, Tuple[int, int]]] = None
        self._bank_full: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None
        self._text_full: Optional[np.ndarray] = None
        self._text_rows: Optional[Dict[int, int]] = None

        self._cache: Optional[FeatureCache] = None
        self._region_lru: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._bank_lru: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._image_sizes: Optional[Dict[int, Tuple[int, int]]] = None
        self._refs_records: Optional[List[Mapping[str, Any]]] = None
        self._train_samples: Optional[List[TrainSample]] = None
        self._text_index_by_ref: Optional[Dict[int, List[Tuple[int, int]]]] = None
        self._eval_cache: Dict[Tuple[Tuple[str, ...], Tuple[int, ...]], List[Any]] = {}
        self._bucket_cache: Dict[Tuple[Tuple[int, ...], int, int], Dict[int, List[TrainSample]]] = {}

        if preload:
            self.preload()

    # -- lazy handles --------------------------------------------------------
    def _feature_cache(self) -> FeatureCache:
        if self._cache is None:
            self._cache = FeatureCache.open(self.features_root)
        return self._cache

    @property
    def feature_dim(self) -> int:
        """Joint embedding width of the backing feature cache (512 or 768, ...).

        Resolved from the opened datasets (or the RAM-resident preloaded store
        when :meth:`preload` ran), never from the frozen V1 module constant, so
        a SigLIP cache trains a 768-d B3 without any caller override.
        """
        if self._text_full is not None:
            return int(np.asarray(self._text_full).shape[-1])
        if self._region_full is not None:
            return int(np.asarray(self._region_full).shape[-1])
        return int(self._feature_cache().feature_dim)

    def _sizes(self) -> Dict[int, Tuple[int, int]]:
        if self._image_sizes is None:
            self._image_sizes = load_image_sizes(self.image_sizes_path)
        return self._image_sizes

    def _refs_list(self) -> List[Mapping[str, Any]]:
        if self._refs_records is None:
            if isinstance(self.refs, (str, Path)):
                from ..data.refcoco import load_refs_pickle

                self._refs_records = list(load_refs_pickle(self.refs))
            else:
                self._refs_records = list(self.refs)
        return self._refs_records

    # -- LRU accessors -------------------------------------------------------
    def _region(self, image_id: int) -> np.ndarray:
        image_id = int(image_id)
        if self._region_full is not None:
            slices = self._region_slices or {}
            if image_id not in slices:
                raise KeyError(
                    f"image {image_id} not in the preloaded region store ({self.features_root})"
                )
            start, count = slices[image_id]
            return self._region_full[start : start + count]
        hit = self._region_lru.get(image_id)
        if hit is not None:
            self._region_lru.move_to_end(image_id)
            return hit
        block = np.ascontiguousarray(self._feature_cache().region_features(image_id), dtype=np.float16)
        self._region_lru[image_id] = block
        if len(self._region_lru) > self.region_cache_size:
            self._region_lru.popitem(last=False)
        return block

    def _bank(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        image_id = int(image_id)
        if self._bank_full is not None:
            if image_id not in self._bank_full:
                raise KeyError(f"image {image_id} not in the preloaded bank ({self.bank_path})")
            return self._bank_full[image_id]
        hit = self._bank_lru.get(image_id)
        if hit is not None:
            self._bank_lru.move_to_end(image_id)
            return hit
        from ..data.bank import read_bank_image

        boxes, objectness = read_bank_image(self.bank_path, image_id)
        entry = (
            np.ascontiguousarray(boxes, dtype=np.float32),
            np.ascontiguousarray(objectness, dtype=np.float32),
        )
        self._bank_lru[image_id] = entry
        if len(self._bank_lru) > self.bank_cache_size:
            self._bank_lru.popitem(last=False)
        return entry

    def _text(self, sentence_id: int) -> np.ndarray:
        """Query embedding of one sentence (RAM store when preloaded, else the cache)."""
        sentence_id = int(sentence_id)
        if self._text_full is not None:
            rows = self._text_rows or {}
            if sentence_id not in rows:
                raise KeyError(
                    f"sentence {sentence_id} not in the preloaded text store ({self.features_root})"
                )
            return self._text_full[rows[sentence_id]]
        return self._feature_cache().text_features(sentence_id)

    def image_size(self, image_id: int) -> Tuple[int, int]:
        """True ``(W, H)`` of one image; raises when the size cache does not cover it."""
        image_id = int(image_id)
        sizes = self._sizes()
        if image_id not in sizes:
            raise KeyError(
                f"image {image_id} has no entry in {self.image_sizes_path}; "
                "rebuild cache/image_sizes.npz with build_image_sizes()"
            )
        return sizes[image_id]

    # -- candidate layout ----------------------------------------------------
    @staticmethod
    def candidate_bank_indices(sample: Any, K: int) -> np.ndarray:
        """The frozen bank rows of ``C_K`` for one sample: ``[target] + order[:K-1]``."""
        target = getattr(sample, "target_index", None)
        if target is None:
            raise ValueError("candidate_bank_indices: sample has no target (target-absent row)")
        k = int(K)
        if k < 2:
            raise ValueError(f"K must be >= 2, got {k}")
        order = np.asarray(getattr(sample, "distractor_order"), dtype=np.int64)
        if order.size < k - 1:
            raise ValueError(
                f"stored distractor ordering has {order.size} rows, need {k - 1} for K={k}"
            )
        cand = np.empty(k, dtype=np.int64)
        cand[0] = int(target)
        cand[1:] = order[: k - 1]
        return cand

    def example_for(self, sample: Any, K: int) -> B3Example:
        """Materialise one :class:`B3Example` for ``C_K`` of ``sample``.

        ``sample`` is anything exposing ``sentence_id`` / ``ref_id`` /
        ``image_id`` / ``target_index`` / ``distractor_order`` (a
        :class:`TrainSample` or a Phase 0A ``SentenceRecord``).  The candidate
        rows are gathered from the *same* stored region block, so the shared
        candidates are bit-identical across ``K``.
        """
        k = int(K)
        cand = self.candidate_bank_indices(sample, k)
        image_id = int(getattr(sample, "image_id"))
        sentence_id = int(getattr(sample, "sentence_id"))
        region = self._region(image_id)
        if int(cand.max(initial=-1)) >= int(region.shape[0]):
            raise IndexError(
                f"candidate index {int(cand.max())} exceeds the {int(region.shape[0])} region rows "
                f"of image {image_id}"
            )
        candidate_features = np.ascontiguousarray(region[cand], dtype=np.float32)
        query_feature = np.ascontiguousarray(self._text(sentence_id), dtype=np.float32)
        boxes, objectness = self._bank(image_id)
        geometry = geometry_features_centered(
            boxes[cand], self.image_size(image_id), objectness[cand]
        )
        return B3Example(
            ref_id=int(getattr(sample, "ref_id")),
            sentence_id=sentence_id,
            image_id=image_id,
            split=_sample_split(sample),
            query_feature=query_feature,
            candidate_features=candidate_features,
            geometry=geometry,
            target_index=TARGET_LOCAL_INDEX,
        )

    # -- training index ------------------------------------------------------
    def _text_index_grouped(self) -> Dict[int, List[Tuple[int, int]]]:
        if self._text_index_by_ref is None:
            root = self._text_index_path if self._text_index_path is not None else self.features_root
            rows = read_text_index(root)
            grouped: Dict[int, List[Tuple[int, int]]] = {}
            for row in rows:
                if str(row["split"]) != "train":
                    continue
                grouped.setdefault(int(row["ref_id"]), []).append(
                    (int(row["sentence_id"]), int(row["image_id"]))
                )
            self._text_index_by_ref = grouped
        return self._text_index_by_ref

    def _train_manifest(self) -> ManifestFile:
        path = self.manifests_root / TRAIN_MANIFEST_NAME
        if not path.exists():
            alt = self.manifests_root / "manifests" / TRAIN_MANIFEST_NAME
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(
                    f"training manifest {TRAIN_MANIFEST_NAME} not found under {self.manifests_root}"
                )
        return ManifestFile.load(path)

    def train_samples(self) -> List[TrainSample]:
        """All target-present ``random_train`` rows (sentences from ``text_index.csv``)."""
        if self._train_samples is not None:
            return self._train_samples
        grouped = self._text_index_grouped()
        manifest = self._train_manifest()
        samples: List[TrainSample] = []
        for entry in manifest.entries:
            if entry.target_index is None:
                continue
            sentences = grouped.get(int(entry.ref_id))
            if not sentences:
                continue
            for sentence_id, image_id in sentences:
                if int(image_id) != int(entry.image_id):
                    raise ValueError(
                        f"ref {entry.ref_id}: text_index says image {image_id}, "
                        f"manifest says {entry.image_id}"
                    )
                samples.append(
                    TrainSample(
                        sentence_id=int(sentence_id),
                        ref_id=int(entry.ref_id),
                        image_id=int(image_id),
                        split="train",
                        target_index=int(entry.target_index),
                        distractor_order=np.asarray(entry.distractor_order, dtype=np.int64),
                        eligible=dict(entry.eligible),
                    )
                )
        self._train_samples = samples
        return samples

    def _train_buckets(self, ks: Sequence[int], seed: int, epoch: int) -> Dict[int, List[TrainSample]]:
        key = (tuple(int(k) for k in ks), int(seed), int(epoch))
        cached = self._bucket_cache.get(key)
        if cached is not None:
            return cached
        pool = _train_pool(ks)
        buckets: Dict[int, List[TrainSample]] = {}
        for sample in self.train_samples():
            k = _choose_train_k(seed, epoch, sample.sentence_id, pool)
            if not bool(sample.eligible.get(int(k), False)):
                continue
            if sample.distractor_order.size < k - 1:
                continue
            buckets.setdefault(k, []).append(sample)
        self._bucket_cache = {key: buckets}  # keep only the current epoch's layout
        return buckets

    def train_bucket_sizes(self, ks: Sequence[int], *, seed: int, epoch: int) -> Dict[int, int]:
        """``{K: n_samples}`` of one deterministic training epoch (for planning/tqdm)."""
        buckets = self._train_buckets(ks, seed, epoch)
        return {int(k): len(v) for k, v in buckets.items()}

    def iter_train(self, ks: Sequence[int], *, seed: int, epoch: int) -> Iterator[B3Example]:
        """Yield one training epoch's examples, grouped by ``K`` (shuffled in-bucket).

        The ``K`` of a sentence is drawn deterministically from
        ``[seed, epoch, sentence_id]``; the in-bucket order is a deterministic
        shuffle from ``[seed, epoch]``.  Only ``random_train`` rows are ever
        produced.
        """
        buckets = self._train_buckets(ks, seed, epoch)
        rng = np.random.default_rng([int(seed), int(epoch)])
        for k in sorted(buckets):
            items = buckets[k]
            order = np.arange(len(items), dtype=np.int64)
            if len(items) > 1:
                rng.shuffle(order)
            for position in order:
                yield self.example_for(items[int(position)], k)

    # -- evaluation rows (Phase 0A authority) --------------------------------
    def eval_records(self, splits: Any, ks: Sequence[int] = EVAL_KS) -> List[Any]:
        """Phase 0A ``SentenceRecord`` list for ``splits`` (row set and order frozen).

        Delegates to :func:`ccg.experiment.phase0a.load_cosine_inputs` so the B3
        evaluation rows are *identical* to the Phase 0A audit's.
        """
        from ..experiment.phase0a import load_cosine_inputs

        names = (splits,) if isinstance(splits, str) else tuple(str(s) for s in splits)
        key = (tuple(names), tuple(int(k) for k in ks))
        cached = self._eval_cache.get(key)
        if cached is None:
            cached = load_cosine_inputs(
                self.features_root,
                self.manifests_root,
                self._refs_list(),
                names if len(names) > 1 else names[0],
                regime=self.regime,
                ks=tuple(int(k) for k in ks),
            )
            self._eval_cache[key] = cached
        return cached

    def eval_available_counts(self, splits: Any, ks: Sequence[int] = EVAL_KS) -> Dict[int, int]:
        """``{K: n_rows}`` of the Phase 0A row set (target present and long enough)."""
        records = self.eval_records(splits, ks)
        counts = {int(k): 0 for k in ks}
        for record in records:
            order_size = int(np.asarray(record.distractor_order).size)
            for k in ks:
                k = int(k)
                if record.target_index is None or order_size < k - 1:
                    continue
                counts[k] += 1
        return counts

    def iter_eval(self, splits: Any, ks: Sequence[int] = EVAL_KS) -> Iterator[B3Example]:
        """Yield ``(record, K)`` evaluation examples for the requested splits/Ks.

        The iteration order is the Phase 0A record order, ``K`` ascending within
        a record; rows whose target is absent or whose stored ordering is
        shorter than ``K-1`` are skipped (exactly as ``score_sets`` does).
        """
        names = (splits,) if isinstance(splits, str) else tuple(str(s) for s in splits)
        records = self.eval_records(names if len(names) > 1 else names[0], ks)
        for record in records:
            if record.target_index is None:
                continue
            order_size = int(np.asarray(record.distractor_order).size)
            for k in ks:
                k = int(k)
                if order_size < k - 1:
                    continue
                yield self.example_for(record, k)

    # -- unified iterator ----------------------------------------------------
    def iter_examples(
        self, split: Any, ks: Sequence[int], *, seed: int = 0, epoch: Optional[int] = None
    ) -> Iterator[B3Example]:
        """Unified iterator: ``train`` -> one epoch; an eval split -> the frozen rows.

        ``ks`` is mandatory (the training pool is intersected with
        :data:`TRAIN_KS`; evaluation uses it verbatim).  ``train`` may not be
        mixed with evaluation splits.
        """
        ks = tuple(int(k) for k in ks)
        names = (split,) if isinstance(split, str) else tuple(str(s) for s in split)
        if "train" in names:
            if len(names) != 1:
                raise ValueError("'train' cannot be mixed with evaluation splits in one iterator")
            yield from self.iter_train(ks, seed=int(seed), epoch=0 if epoch is None else int(epoch))
            return
        yield from self.iter_eval(names if len(names) > 1 else names[0], ks)

    # -- preload (bulk RAM residency) ----------------------------------------
    def preload(
        self,
        *,
        region: bool = True,
        bank: bool = True,
        text: bool = True,
        log: Optional[Any] = None,
    ) -> "B3Corpus":
        """Bulk-load the region / text / bank stores into RAM (one sequential read each).

        The region h5 is chunked at 8192 rows (8 MiB), so a *random* per-image
        read pulls a whole chunk - under the shuffled training order that is
        ~15 ms per image on the on-demand path.  Reading each store once and
        serving :meth:`example_for` from RAM makes random access effectively
        free.  The region block is the big one (~1.2 GiB, float16); text/bank
        are ~150 MiB / ~25 MiB.
        """
        import h5py

        if region:
            path = self.features_root / REGION_FILENAME
            if log is not None:
                log(f"[B3Corpus.preload] region <- {path}")
            with h5py.File(str(path), "r") as handle:
                offsets = np.asarray(handle["image_offsets"][:], dtype=np.int64)
                self._region_full = np.asarray(handle["features"][:], dtype=np.float16)
            self._region_slices = {int(row[0]): (int(row[1]), int(row[2])) for row in offsets}
            self._region_lru.clear()

        if text:
            path = self.features_root / TEXT_FILENAME
            if log is not None:
                log(f"[B3Corpus.preload] text   <- {path}")
            with h5py.File(str(path), "r") as handle:
                offsets = np.asarray(handle["sentence_offsets"][:], dtype=np.int64)
                self._text_full = np.asarray(handle["features"][:], dtype=np.float16)
            self._text_rows = {int(row[0]): int(row[1]) for row in offsets}

        if bank:
            from ..data.bank import iter_bank

            if log is not None:
                log(f"[B3Corpus.preload] bank   <- {self.bank_path}")
            store: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for image_id, boxes, objectness in iter_bank(self.bank_path):
                store[int(image_id)] = (
                    np.ascontiguousarray(boxes, dtype=np.float32),
                    np.ascontiguousarray(objectness, dtype=np.float32),
                )
            self._bank_full = store
            self._bank_lru.clear()
        return self

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None
        self._region_lru.clear()
        self._bank_lru.clear()
        self._region_full = None
        self._region_slices = None
        self._bank_full = None
        self._text_full = None
        self._text_rows = None

    def __enter__(self) -> "B3Corpus":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
