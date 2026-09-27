"""COCO 2014 annotation index (instances_*.json) - geometry and file names only.

The proposal audit needs three things from COCO: the GT object boxes + category
ids *per image* (recall, category assignment), the ``file_name``/size of an
image (proposal extraction, downloading single smoke images) and which COCO
split an image belongs to (so that an audit subset can state its provenance).

Parsing is kept out of the loaders' way: :func:`build_index` works on an
already-decoded json dict, so unit tests can build an index without a 330 MB
file, while :func:`load_coco_index` handles the on-disk case.

Determinism: annotations of an image are stored sorted by ascending ``id``, so
``GtObjects`` rows are identical across runs and across machines - the tie-break
rules in :mod:`ccg.data.proposals` / :mod:`ccg.data.audit` depend on that.

Format note (verified against the local download): COCO 2014 ``bbox`` is
``[x, y, w, h]`` with float pixels and ``image_id``/``category_id`` are ints;
``iscrowd`` annotations are *kept* (they are real objects for a proposal recall
computation) but flagged in ``crowd_flags`` so a report can exclude them if it
wants the standard COCO metrics instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CocoImage",
    "GtObjects",
    "CocoIndex",
    "build_index",
    "load_coco_index",
    "split_label_for_annotation_file",
    "coco_image_url",
]

_COCO_URL_TEMPLATE = "http://images.cocodataset.org/{folder}/{file_name}"


@dataclass(frozen=True)
class CocoImage:
    """Minimal image metadata needed to locate and size one COCO image."""

    image_id: int
    file_name: str
    height: int
    width: int
    split: str = ""

    @property
    def folder(self) -> str:
        """``COCO_train2014_*.jpg`` -> ``train2014`` (the URL/ directory name)."""
        parts = self.file_name.split("_")
        for part in parts:
            if part.endswith("2014") or part.endswith("2017"):
                return part
        return self.split or "unknown"

    @property
    def url(self) -> str:
        return coco_image_url(self.file_name)


@dataclass
class GtObjects:
    """The GT objects of one image, rows sorted by ascending annotation id.

    ``boxes`` are ``xyxy`` float32 in original-image pixels, ``categories`` are
    COCO category ids (``int64``), ``object_ids`` the annotation ids and
    ``crowd_flags`` marks ``iscrowd == 1`` rows.
    """

    image_id: int
    boxes: np.ndarray
    categories: np.ndarray
    object_ids: np.ndarray
    crowd_flags: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )

    def __post_init__(self) -> None:
        self.image_id = int(self.image_id)
        self.boxes = np.ascontiguousarray(np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4))
        n = self.boxes.shape[0]
        self.categories = np.asarray(self.categories, dtype=np.int64).reshape(-1)
        self.object_ids = np.asarray(self.object_ids, dtype=np.int64).reshape(-1)
        self.crowd_flags = np.asarray(self.crowd_flags, dtype=bool).reshape(-1)
        for name, arr in (("categories", self.categories), ("object_ids", self.object_ids)):
            if arr.size != n:
                raise ValueError(f"{name} has {arr.size} rows, expected {n}")
        if self.crowd_flags.size not in (0, n):
            raise ValueError(f"crowd_flags has {self.crowd_flags.size} rows, expected {n}")
        if self.crowd_flags.size == 0:
            self.crowd_flags = np.zeros(n, dtype=bool)

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def subset(self, mask: Sequence[bool] | np.ndarray) -> "GtObjects":
        """Rows where ``mask`` is true (used to drop crowd objects)."""
        sel = np.asarray(mask, dtype=bool).reshape(-1)
        if sel.size != len(self):
            raise ValueError(f"mask of size {sel.size} cannot select {len(self)} objects")
        return GtObjects(
            image_id=self.image_id,
            boxes=self.boxes[sel],
            categories=self.categories[sel],
            object_ids=self.object_ids[sel],
            crowd_flags=self.crowd_flags[sel],
        )

    def non_crowd(self) -> "GtObjects":
        return self.subset(~self.crowd_flags)


class CocoIndex:
    """``image_id`` -> metadata / GT objects, for one or more COCO files."""

    def __init__(
        self,
        images: Mapping[int, CocoImage],
        objects: Mapping[int, GtObjects],
        *,
        category_names: Optional[Mapping[int, str]] = None,
        sources: Sequence[str] = (),
    ) -> None:
        self.images: Dict[int, CocoImage] = {int(k): v for k, v in images.items()}
        self.objects_by_image: Dict[int, GtObjects] = {int(k): v for k, v in objects.items()}
        self.category_names: Dict[int, str] = {
            int(k): str(v) for k, v in (category_names or {}).items()
        }
        self.sources: Tuple[str, ...] = tuple(str(s) for s in sources)
        #: Annotation rows dropped because their image is not in ``images``
        #: (reported, never silently filtered - see :func:`build_index`).
        self.num_orphan_annotations: int = 0

    # -- queries --------------------------------------------------------------
    def __contains__(self, image_id: object) -> bool:
        return int(image_id) in self.images  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.images)

    @property
    def image_ids(self) -> np.ndarray:
        return np.asarray(sorted(self.images), dtype=np.int64)

    def image_ids_of_split(self, split: str) -> np.ndarray:
        wanted = str(split)
        return np.asarray(
            sorted(i for i, meta in self.images.items() if meta.split == wanted), dtype=np.int64
        )

    def image(self, image_id: int) -> CocoImage:
        try:
            return self.images[int(image_id)]
        except KeyError as exc:
            raise KeyError(f"image {int(image_id)} is not in this COCO index") from exc

    def file_name(self, image_id: int) -> str:
        return self.image(image_id).file_name

    def objects(self, image_id: int) -> GtObjects:
        """GT objects of an image; an empty :class:`GtObjects` when it has none."""
        image_id = int(image_id)
        if image_id not in self.images:
            raise KeyError(f"image {image_id} is not in this COCO index")
        return self.objects_by_image.get(
            image_id,
            GtObjects(
                image_id=image_id,
                boxes=np.zeros((0, 4), dtype=np.float32),
                categories=np.zeros(0, dtype=np.int64),
                object_ids=np.zeros(0, dtype=np.int64),
            ),
        )

    def num_annotations(self) -> int:
        return int(sum(len(o) for o in self.objects_by_image.values()))

    def image_path(self, image_id: int, root: str | Path) -> Path:
        """``root/<folder>/<file_name>`` - the standard COCO 2014 layout."""
        meta = self.image(image_id)
        return Path(root) / meta.folder / meta.file_name

    def missing_images(self, image_ids: Iterable[int], root: str | Path) -> List[int]:
        """Which of ``image_ids`` are not on disk under ``root`` (report, never skip)."""
        out: List[int] = []
        for image_id in image_ids:
            if not self.image_path(int(image_id), root).exists():
                out.append(int(image_id))
        return out


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def build_index(
    payload: Mapping[str, Any],
    *,
    split: str = "",
    source: str = "",
    only_image_ids: Optional[Iterable[int]] = None,
    index: Optional[CocoIndex] = None,
) -> CocoIndex:
    """Turn one decoded COCO ``instances_*.json`` dict into a :class:`CocoIndex`.

    ``only_image_ids`` prunes the (potentially huge) annotation list down to the
    images actually being audited - the reason the audit can run on a laptop
    without holding 860k boxes in memory twice.  ``index`` appends into an
    existing index (id collisions across COCO files raise, they never silently
    overwrite).
    """
    for key in ("images", "annotations"):
        if not isinstance(payload.get(key), list):
            raise ValueError(
                f"not a COCO instances json: expected list fields 'images'/'annotations', "
                f"got keys {sorted(payload)[:8]}"
            )
    wanted = None if only_image_ids is None else {int(i) for i in only_image_ids}
    images: Dict[int, CocoImage] = {}
    for meta in payload["images"]:
        image_id = int(meta["id"])
        if wanted is not None and image_id not in wanted:
            continue
        images[image_id] = CocoImage(
            image_id=image_id,
            file_name=str(meta["file_name"]),
            height=int(meta["height"]),
            width=int(meta["width"]),
            split=str(split or meta.get("coco_split") or ""),
        )

    grouped: Dict[int, List[Tuple[int, float, float, float, float, int, int]]] = {}
    dropped_without_image = 0
    for ann in payload["annotations"]:
        image_id = int(ann["image_id"])
        if wanted is not None and image_id not in wanted:
            continue
        if image_id not in images:
            # annotation whose image row is missing from this file: counted, not
            # silently swallowed (see CocoIndex.num_orphan_annotations)
            dropped_without_image += 1
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        grouped.setdefault(image_id, []).append(
            (
                int(ann["id"]),
                x,
                y,
                x + w,
                y + h,
                int(ann["category_id"]),
                int(ann.get("iscrowd", 0)),
            )
        )

    objects: Dict[int, GtObjects] = {}
    orphan = 0
    for image_id, rows in grouped.items():
        rows.sort(key=lambda row: row[0])  # deterministic: ascending annotation id
        array = np.asarray(rows, dtype=np.float64)
        objects[image_id] = GtObjects(
            image_id=image_id,
            boxes=array[:, 1:5].astype(np.float32),
            categories=array[:, 5].astype(np.int64),
            object_ids=array[:, 0].astype(np.int64),
            crowd_flags=array[:, 6].astype(np.int64).astype(bool),
        )
    orphan += int(dropped_without_image)

    names = {int(c["id"]): str(c.get("name", "")) for c in payload.get("categories", []) or []}
    if index is None:
        built = CocoIndex(
            images, objects, category_names=names, sources=(source,) if source else ()
        )
        built.num_orphan_annotations = orphan
        return built

    clash = set(index.images) & set(images)
    if clash:
        raise ValueError(
            f"image ids present in more than one source file ({len(clash)} clashes, e.g. "
            f"{sorted(clash)[:5]}); refusing to guess which COCO split an image belongs to"
        )
    merged_images = dict(index.images)
    merged_images.update(images)
    merged_objects = dict(index.objects_by_image)
    merged_objects.update(objects)
    merged_names = dict(index.category_names)
    merged_names.update(names)
    merged = CocoIndex(
        merged_images,
        merged_objects,
        category_names=merged_names,
        sources=index.sources + ((source,) if source else ()),
    )
    merged.num_orphan_annotations = index.num_orphan_annotations + orphan
    return merged


def split_label_for_annotation_file(path: str | Path) -> str:
    """``instances_val2014.json`` -> ``val2014`` (used as the ``coco_split`` label)."""
    stem = Path(path).stem
    parts = stem.split("_")
    return parts[-1] if len(parts) > 1 else stem


def load_coco_index(
    annotation_files: Sequence[str | Path],
    *,
    only_image_ids: Optional[Iterable[int]] = None,
) -> CocoIndex:
    """Load one or more ``instances_*.json`` files into a single index.

    Files are labelled by their name (``instances_train2014.json`` ->
    ``split="train2014"``).  A missing file raises with the checklist hint
    instead of returning a partial index.
    """
    files = [Path(p) for p in annotation_files]
    index: Optional[CocoIndex] = None
    for path in files:
        if not path.exists():
            raise FileNotFoundError(
                f"COCO annotation {path} not found - unpack annotations_trainval2014.zip into "
                "data/raw/annotations/ (Phase 0 checklist step)"
            )
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        built = build_index(
            payload,
            split=split_label_for_annotation_file(path),
            source=str(path),
            only_image_ids=only_image_ids,
            index=index,
        )
        index = built
    if index is None:
        raise ValueError("load_coco_index() called with an empty list of annotation files")
    return index


def coco_image_url(file_name: str) -> str:
    """Public COCO 2014 image URL for a ``file_name`` (single-image downloads only)."""
    name = Path(str(file_name)).name
    if not name.startswith("COCO_"):
        raise ValueError(f"{file_name!r} is not a COCO file name (expected COCO_<split>2014_*.jpg)")
    folder = name.split("_")[1]
    return _COCO_URL_TEMPLATE.format(folder=folder, file_name=name)
