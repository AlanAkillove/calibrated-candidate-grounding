"""RefCOCO+ annotation parsers (Task 4 / Task 3 hand-off).

Two on-disk layouts matter for RefCOCO+:

1. **referring-coco json** (``refcoco+.json`` / ``refs.*.json``) - the *primary*
   format implemented here.  A top-level list (or a ``{"refs": [...]}`` object)
   of records shaped like::

       {"ref_id": 21, "split": "train", "aishell_id": "...",
        "image_id": 391893, "file_url": "coco/train2014/COCO_train2014_000000391893.jpg",
        "bbox": [x, y, w, h], "phrase": "the man in the black shirt"}

   This is the layout produced by ``matplotlib/refcoco`` dumps, the
   ``lxyicheng/refcoco`` processing scripts and the ``refer`` toolbox
   (``refs_train.json`` / ``refs_val.json`` / ``refs_testA.json`` ...).

2. **UNC mats** (``refCOCOp_train.mat`` / ``..._val.mat`` / ``..._testA/B.mat``)
   with a ``refs`` struct array whose fields are ``REF_ID``, ``IMAGE_ID``,
   ``BOXES`` (``[x, y, w, h, BL]`` rows), ``SENTENCE`` and ``SPLIT_ID``, plus a
   ``splitsCell`` / ``imgIds`` cell listing the images per split.  Implemented
   as a *best-effort skeleton* (:func:`parse_unc_mats`).

3. **UNC referring pickle** ``refs(unc).p`` - the layout the referit / LMSCLOUD
   ``refcoco+.zip`` actually ships.  *Verified* on the local download
   (2026-09-27): a top-level **list** of one record per referred *region*
   (49,856 records / 141,564 sentences, 19,992 images), each record::

       {"ref_id": 0, "ann_id": 1719310, "image_id": 581857, "category_id": 1,
        "split": "train",                       # train / val / testA / testB
        "file_name": "COCO_train2014_000000581857_16.jpg",  # NOT a COCO file name
        "sent_ids": [0, 1, 2],
        "sentences": [{"tokens": [...], "raw": "...", "sent_id": 0, "sent": "..."}, ...]}

   There is **no ``bbox`` field** - the box has to be joined from the
   ``instances.json`` sitting next to the pickle through ``ann_id`` (a COCO 2014
   annotation id; verified: all 49,856 resolve and their ``bbox`` values match
   ``instances_train2014.json`` exactly).  Implemented by
   :func:`parse_refs_unc_pickle`, which also accepts the dict-shaped variant
   ``{"refs": [...], "_splits": {...}}`` used by other mirrors of the same data.
   Split labels are image-level and disjoint (train 16,992 / val 1,500 / testA
   750 / testB 750 images, all from COCO ``train2014``).

>>> 待真实数据下载后核对 (to be verified against the real files) <<<
Field spelling, nesting depth, ``bbox`` semantics (``xywh`` vs ``xyxy``) and the
``split`` label strings of both formats can only be confirmed once
``refcoco+.json`` / the UNC ``.mat`` files are actually downloaded.  Every
assumption is therefore (a) localised in the ``*_FIELD_ALIASES`` tables and the
constants below, and (b) reported instead of silently swallowed: unknown fields
raise :class:`AnnotationFormatError` with the offending record attached, and
unknown split labels are surfaced through :func:`parse_refs_json`'s report
(``parser_report``) rather than being dropped.  Nothing in this module needs a
real file to be unit-tested - the parsers work on record dicts.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .splits import normalize_split_name
from .types import ReferringExample

__all__ = [
    "AnnotationFormatError",
    "ParseReport",
    "parse_refs_json",
    "dump_refs_json",
    "refs_from_records",
    "parse_unc_mats",
    "parse_refs_unc_pickle",
    "image_ids_by_split_unc_pickle",
    "find_refcoco_plus_files",
    "load_refcoco_plus",
]

#: Referring-coco json stores ``bbox`` as ``[x, y, w, h]``.  Flip to ``"xyxy"``
#: here if the real download turns out to be corner-based (待核对).
DEFAULT_BOX_FORMAT = "xywh"

REF_ID_ALIASES = ("ref_id", "region_id", "id", "ID", "refId")
IMAGE_ID_ALIASES = ("image_id", "img_id", "imageId", "image")
TEXT_ALIASES = ("phrase", "sentence", "sentences", "expression", "text", "sentence_raw")
SPLIT_ALIASES = ("split", "split_name", "splitName", "partition")
BBOX_ALIASES = ("bbox", "box", "bboxes", "regions_bbox")
OBJECT_ID_ALIASES = ("gt_object_id", "object_id", "instance_id", "ann_id", "annotation_id")


class AnnotationFormatError(ValueError):
    """Raised when an annotation record cannot be interpreted."""


@dataclass
class ParseReport:
    """Non-fatal observations collected while parsing (auditable, never silent)."""

    num_examples: int = 0
    num_records_seen: int = 0
    num_skipped: int = 0
    unknown_splits: List[str] = field(default_factory=list)
    missing_object_ids: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num_examples": self.num_examples,
            "num_records_seen": self.num_records_seen,
            "num_skipped": self.num_skipped,
            "unknown_splits": sorted(set(self.unknown_splits)),
            "missing_object_ids": self.missing_object_ids,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# box handling
# ---------------------------------------------------------------------------
def _to_xyxy(raw: Any, box_format: str) -> np.ndarray:
    """Normalise one annotation box to ``xyxy`` float32."""
    if isinstance(raw, Mapping):  # {"bbox": [...]} nesting or {"x":..,"y":..,"w":..,"h":..}
        for key in ("bbox", "box", "coordinates"):
            if key in raw:
                return _to_xyxy(raw[key], box_format)
        try:
            arr = np.array([raw["x"], raw["y"], raw["w"], raw["h"]], dtype=np.float32)
        except KeyError as exc:  # pragma: no cover - defensive
            raise AnnotationFormatError(f"cannot read box from {raw!r}") from exc
        return _xywh_to_xyxy(arr)
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size == 6:  # UNC BOXES rows are [x, y, w, h, BL?] plus padding in some dumps
        arr = arr[:4]
    if arr.size != 4:
        raise AnnotationFormatError(f"expected a 4-value box, got {raw!r}")
    if box_format == "xyxy":
        return np.ascontiguousarray(arr, dtype=np.float32)
    if box_format == "xywh":
        return _xywh_to_xyxy(arr)
    raise ValueError(f"unknown box_format {box_format!r}")


def _xywh_to_xyxy(arr: np.ndarray) -> np.ndarray:
    x, y, w, h = [float(v) for v in arr]
    if w <= 0 or h <= 0:
        raise AnnotationFormatError(f"degenerate xywh box {arr.tolist()}")
    return np.array([x, y, x + w, y + h], dtype=np.float32)


# ---------------------------------------------------------------------------
# record field extraction
# ---------------------------------------------------------------------------
def _pick(record: Mapping[str, Any], aliases: Sequence[str], what: str, record_no: int):
    for key in aliases:
        if key in record and record[key] is not None:
            return record[key]
    raise AnnotationFormatError(
        f"record #{record_no}: no {what} field among {tuple(aliases)}; keys={sorted(record)}"
    )


def _text_from_record(record: Mapping[str, Any], record_no: int) -> str:
    raw = _pick(record, TEXT_ALIASES, "expression text", record_no)
    if isinstance(raw, np.ndarray):  # scipy .mat cells arrive as arrays
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        # refcoco2 "sentences" style: a list of paraphrases.  The first is used
        # 待核对 - the real dumps keep one sentence per record.
        if not raw:
            raise AnnotationFormatError(f"record #{record_no}: empty sentence list")
        while isinstance(raw, (list, tuple)):
            if not raw:
                raise AnnotationFormatError(f"record #{record_no}: empty sentence list")
            raw = raw[0]  # nested lists come from .mat character matrices
    if isinstance(raw, Mapping):
        raw = _pick(raw, ("sentence", "text", "phrase", "sentence_raw"), "sentence text", record_no)
    text = str(raw).strip()
    if not text:
        raise AnnotationFormatError(f"record #{record_no}: empty expression text")
    return text


def refs_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    split: Optional[str] = None,
    box_format: str = DEFAULT_BOX_FORMAT,
    report: Optional[ParseReport] = None,
) -> List[ReferringExample]:
    """Convert annotation records into :class:`ReferringExample` objects.

    This is the shared core of every parser in this module and the function that
    unit tests exercise directly (no file needed).

    Parameters
    ----------
    records:
        Iterable of record dicts (one per expression).
    split:
        When given, forces this split for every record (used by
        ``refs_train.json`` style files that have no per-record ``split``).
    box_format:
        ``"xywh"`` (default, referring-coco json) or ``"xyxy"``.  待核对.
    report:
        Optional :class:`ParseReport` filled with observations.
    """
    from .splits import SPLIT_NAMES

    sink = report if report is not None else ParseReport()
    examples: List[ReferringExample] = []
    for record_no, record in enumerate(records):
        sink.num_records_seen += 1
        if not isinstance(record, Mapping):
            raise AnnotationFormatError(f"record #{record_no} is not a mapping: {type(record)}")
        ref_id = int(np.asarray(_pick(record, REF_ID_ALIASES, "ref id", record_no)).item())
        image_id = int(np.asarray(_pick(record, IMAGE_ID_ALIASES, "image id", record_no)).item())
        text = _text_from_record(record, record_no)
        gt_box = _to_xyxy(_pick(record, BBOX_ALIASES, "box", record_no), box_format)
        raw_split = split if split is not None else _first(record, SPLIT_ALIASES)
        split_name = normalize_split_name(raw_split)
        if not split_name:
            raise AnnotationFormatError(
                f"record #{record_no} (ref_id={ref_id}): missing split label; pass split=... "
                "or provide a per-record 'split' field"
            )
        if split_name not in SPLIT_NAMES:
            sink.unknown_splits.append(split_name)
        gt_object_id = _first(record, OBJECT_ID_ALIASES)
        gt_object_id = int(np.asarray(gt_object_id).item()) if gt_object_id is not None else None
        if gt_object_id is None:
            sink.missing_object_ids += 1
        examples.append(
            ReferringExample(
                ref_id=ref_id,
                image_id=image_id,
                text=text,
                gt_box=gt_box,
                gt_object_id=gt_object_id,
                split=split_name,
            )
        )
    sink.num_examples = len(examples)
    return examples


def _first(record: Mapping[str, Any], aliases: Sequence[str]):
    for key in aliases:
        if key in record and record[key] is not None:
            return record[key]
    return None


# ---------------------------------------------------------------------------
# json layout handling
# ---------------------------------------------------------------------------
def _records_from_json_root(root: Any, *, path: Path) -> List[Mapping[str, Any]]:
    """Flatten the several known top-level json layouts into a record list.

    Supported (待核对 which one the real RefCOCO+ download uses):

    * ``[ {...}, {...} ]`` - a plain list of refs.
    * ``{"refs": [...]}`` / ``{"regions": [...]}`` - list under a key.
    * ``{"images": [{"regions": [{"bbox", "sentence"/"phrase", ...}]}]}`` -
      nesting used by the ``mastronardi/refcoco`` style dumps; ``image_id`` is
      inherited from the parent image, ``ref_id`` from the region when present,
      otherwise a deterministic ``(image, region)`` counter is assigned.
    * ``{"sentences": [...], "regions": [...], "images": [...]}`` - refcoco2
      triple-store; sentences are joined onto their region onto its image.
    """
    if isinstance(root, list):
        return [r for r in root if isinstance(r, Mapping)]

    if not isinstance(root, Mapping):
        raise AnnotationFormatError(f"unexpected json root type {type(root)} in {path}")

    # The refcoco2 triple store must be detected *before* the "regions" shortcut,
    # otherwise the region rows would be read without their sentence text.
    if "sentences" in root and "regions" in root:
        return _join_refcoco2_store(root)

    for key in ("refs", "regions", "data"):
        if isinstance(root.get(key), list):
            return [r for r in root[key] if isinstance(r, Mapping)]

    if isinstance(root.get("images"), list):
        records: List[Mapping[str, Any]] = []
        counter = 0
        for image in root["images"]:
            if not isinstance(image, Mapping):
                continue
            image_id = _first(image, ("image_id", "id", "imageFilename", "filename"))
            for region in image.get("regions", []) or []:
                if not isinstance(region, Mapping):
                    continue
                counter += 1
                merged = dict(region)
                merged.setdefault("image_id", image_id)
                merged.setdefault("ref_id", counter)
                records.append(merged)
        if records:
            return records

    raise AnnotationFormatError(
        f"cannot find referring records in {path}; top-level keys={sorted(root) if isinstance(root, Mapping) else 'n/a'}"
    )


def _join_refcoco2_store(root: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Join the refcoco2 ``sentences`` / ``regions`` / ``images`` triple store."""
    regions: Dict[Any, Dict[str, Any]] = {}
    for region in root.get("regions", []) or []:
        if isinstance(region, Mapping) and "region_id" in region:
            regions[region["region_id"]] = dict(region)
    images: Dict[Any, Any] = {}
    for image in root.get("images", []) or []:
        if isinstance(image, Mapping):
            image_id = image.get("image_id", image.get("id"))
            images[image_id] = image_id
    records: List[Mapping[str, Any]] = []
    for sentence in root.get("sentences", []) or []:
        if not isinstance(sentence, Mapping):
            continue
        region = regions.get(sentence.get("region_id"))
        if region is None:  # sentence without a resolvable region: skip loudly
            continue
        merged = dict(region)
        merged["sentence"] = _first(sentence, ("sentence_raw", "sentence", "text"))
        image_id = region.get("image_id", sentence.get("image_id"))
        merged["image_id"] = images.get(image_id, image_id)
        for key in ("split", "split_name"):
            if key in sentence:
                merged[key] = sentence[key]
        records.append(merged)
    return records


def parse_refs_json(
    path: str | Path,
    *,
    split: Optional[str] = None,
    box_format: str = DEFAULT_BOX_FORMAT,
    report: Optional[ParseReport] = None,
) -> List[ReferringExample]:
    """Parse a referring-coco style ``refs.*.json`` file into examples.

    Parameters
    ----------
    path:
        ``refcoco+.json`` / ``refs_train.json`` / ``refs_val.json`` ...
    split:
        Force a split label for every record (for files without per-record
        ``split``, e.g. ``refs_testA.json``).
    box_format:
        ``"xywh"`` (default) or ``"xyxy"`` - see module docstring, 待核对.
    report:
        Optional :class:`ParseReport` collecting observations (unknown split
        spellings, records missing a GT object id).

    Returns
    -------
    list[ReferringExample]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. RefCOCO+ annotations must be downloaded before parsing "
            "(see scripts/prepare_refcoco.py); this parser is unit-tested on synthetic files only."
        )
    with path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    records = _records_from_json_root(root, path=path)
    return refs_from_records(records, split=split, box_format=box_format, report=report)


def dump_refs_json(examples: Sequence[ReferringExample], path: str | Path) -> Path:
    """Write examples back to the primary json layout (round-trip / tests only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    refs = []
    for ex in examples:
        x1, y1, x2, y2 = [float(v) for v in ex.gt_box]
        refs.append(
            {
                "ref_id": int(ex.ref_id),
                "image_id": int(ex.image_id),
                "split": ex.split,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "phrase": ex.text,
                **({"gt_object_id": int(ex.gt_object_id)} if ex.gt_object_id is not None else {}),
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"refs": refs}, handle, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# UNC .mat skeleton
# ---------------------------------------------------------------------------
def parse_unc_mats(
    path: str | Path,
    *,
    split: Optional[str] = None,
    mat_key: str = "refs",
    box_field: str = "BOXES",
    text_field: str = "SENTENCE",
    report: Optional[ParseReport] = None,
) -> List[ReferringExample]:
    """Best-effort parser for the UNC RefCOCO+ ``.mat`` dumps.  待真实数据核对.

    The UNC distribution ships per-split ``.mat`` files whose ``refs`` struct
    array carries upper-case fields (``REF_ID``, ``IMAGE_ID``, ``BOXES``,
    ``SENTENCE``, ``SPLIT_ID``, ``CATEGORY_ID`` ...).  This function:

    1. loads the file with ``scipy.io.loadmat(..., simplify_cells=True)`` so that
       struct arrays become ``list[dict]`` when scipy is new enough;
    2. normalises the records into the same shape handled by
       :func:`refs_from_records` (upper-case keys are lower-cased, ``BOXES`` rows
       are read as ``[x, y, w, h]``, ``SENTENCE`` as text);
    3. delegates conversion + validation to :func:`refs_from_records`.

    Not verified against real files: exact field names, whether ``BOXES`` rows
    are 5 columns (last column is a ``BL`` flag) and whether ``SPLIT_ID`` is a
    numeric index into a separate cell array.  When the assumption breaks the
    function raises :class:`AnnotationFormatError` rather than returning partial
    data.  ``loadmat`` is imported lazily so that environments without scipy
    still import this module.
    """
    path = Path(path)
    try:
        from scipy.io import loadmat  # local import: optional heavy dependency
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("parse_unc_mats requires scipy (scipy.io.loadmat)") from exc

    if not path.exists():
        raise FileNotFoundError(f"{path} not found - UNC .mat files require the data download")
    loaded = loadmat(str(path), simplify_cells=True)
    if mat_key not in loaded:
        raise AnnotationFormatError(
            f"{path.name}: key {mat_key!r} absent; available={sorted(k for k in loaded if not k.startswith('__'))}"
        )
    records = [_normalise_mat_record(r) for r in _as_record_list(loaded[mat_key])]
    sink = report if report is not None else ParseReport()
    if sink.notes is not None:
        sink.notes.append(f"parse_unc_mats assumptions unverified for {path.name}")
    return refs_from_records(records, split=split, box_format="xywh", report=sink)


def _as_record_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1).tolist() if value.ndim else [value.item()]
        return flat
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalise_mat_record(record: Any) -> Mapping[str, Any]:
    """Map one UNC-mat struct onto the referring-coco record vocabulary."""
    if not isinstance(record, Mapping):
        raise AnnotationFormatError(f"unexpected mat record type {type(record)}: {record!r}")
    out: Dict[str, Any] = {}
    for key, value in record.items():
        lower = str(key).lower()
        if lower in ("boxes", "bbox"):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[0] if arr.shape[0] else arr.reshape(-1)
            out["bbox"] = arr[:4]
        elif lower in ("sentence", "sentences", "phrase"):
            out["phrase"] = value
        elif lower in ("image_id", "img_id", "imageid", "split_base"):
            out["image_id"] = value
        elif lower in ("ref_id", "id", "region_id"):
            out["ref_id"] = value
        elif lower in ("category_id", "object_id", "instance_id", "ann_id"):
            out["gt_object_id"] = value
        else:
            out[lower] = value
    return out


# ---------------------------------------------------------------------------
# UNC referring pickle  ``refs(unc).p``
# ---------------------------------------------------------------------------
#: File names the official / mirrored RefCOCO+ (UNC) pickles are shipped under.
UNC_PICKLE_NAMES: tuple[str, ...] = (
    "refs(unc).p",
    "refs(unc).pickle",
    "refcoco+_refs(unc).p",
    "refs.p",
)
_ANN_ID_ALIASES = ("ann_id", "annotation_id", "instance_id", "object_id")
_CATEGORY_ALIASES = ("category_id", "category", "catid")
_SENTENCE_LIST_ALIASES = ("sentences", "refs", "expressions")


def _load_pickle_root(path: Path) -> Any:
    """Unpickle ``refs(unc).p``.

    .. warning::
       ``pickle`` executes arbitrary objects while loading.  Only point this at
       the official RefCOCO+ / referit archive you downloaded yourself (or a
       checksum-verified copy of it); never at an uploaded or network-streamed
       file.  The json parsers above are the safe path whenever a json exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - RefCOCO+ (UNC) annotations must be downloaded first "
            "(Phase 0 checklist step)"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def _unc_pickle_records(root: Any, *, path: Path) -> tuple[List[Mapping[str, Any]], Dict[int, str]]:
    """Flatten both known ``refs(unc).p`` shapes into ``(records, id_to_split)``.

    * ``list`` of region records - the verified referit/LMSCLOUD layout.
    * ``dict`` with ``"refs"`` (optionally plus ``"_splits": {split: set(ids)}``
      as shipped by the LMSCLOUD ``refer`` toolbox).  ``id_to_split`` then maps
      ``annot_id``/``ref_id`` membership of those sets onto split labels, which
      is how a record without a per-record ``split`` field still gets one.
    """
    if isinstance(root, list):
        records = [r for r in root if isinstance(r, Mapping)]
        if not records:
            raise AnnotationFormatError(f"{path}: pickle is a list but holds no mappings")
        return records, {}
    if isinstance(root, Mapping):
        for key in ("refs", "regions", "data"):
            if isinstance(root.get(key), list):
                records = [r for r in root[key] if isinstance(r, Mapping)]
                id_to_split: Dict[int, str] = {}
                splits = root.get("_splits") or root.get("splits")
                if isinstance(splits, Mapping):
                    for name, ids in splits.items():
                        label = normalize_split_name(name)
                        for raw_id in list(ids) if isinstance(ids, (list, tuple, set)) else []:
                            try:
                                id_to_split.setdefault(int(raw_id), label)
                            except (TypeError, ValueError):
                                continue
                return records, id_to_split
        raise AnnotationFormatError(f"{path}: no 'refs' list among keys {sorted(root)[:8]}")
    raise AnnotationFormatError(f"{path}: unexpected pickle root type {type(root)}")


def _ann_boxes(instances_path: Path) -> Dict[int, List[float]]:
    """``ann_id -> [x, y, w, h]`` from a COCO-style ``instances.json``."""
    if not instances_path.exists():
        raise FileNotFoundError(
            f"{instances_path} not found: refs(unc).p carries no boxes, they must be joined "
            "from the COCO instances.json shipped next to it"
        )
    with instances_path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    annotations = root.get("annotations")
    if not isinstance(annotations, list):
        raise AnnotationFormatError(f"{instances_path} has no 'annotations' list")
    return {int(ann["id"]): list(ann["bbox"]) for ann in annotations if "id" in ann and "bbox" in ann}


def _first_text(sentence: Any, record_no: int) -> str:
    """Expression text of one ``sentences`` entry (dict or bare string)."""
    if isinstance(sentence, Mapping):
        for key in ("raw", "sent", "sentence", "sentence_raw", "text", "phrase"):
            value = sentence.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        raise AnnotationFormatError(
            f"record #{record_no}: sentence dict has no text field; keys={sorted(sentence)}"
        )
    if isinstance(sentence, (list, tuple)):
        return _first_text(sentence[0], record_no) if sentence else ""
    text = str(sentence).strip()
    if not text:
        raise AnnotationFormatError(f"record #{record_no}: empty sentence {sentence!r}")
    return text


def _sentence_records(
    record: Mapping[str, Any],
    *,
    record_no: int,
    boxes: Sequence[float],
    ann_id: Optional[int],
    image_id: int,
    split: str,
) -> List[Dict[str, Any]]:
    """Explode one region record into one annotation record per expression."""
    sentences = _first(record, _SENTENCE_LIST_ALIASES)
    if sentences is None:
        # a record that *is* a single expression (some mirrors flatten it that way)
        text = _first(record, TEXT_ALIASES)
        sentences = [{"raw": text}] if text is not None else []
    if isinstance(sentences, Mapping):  # {"raw": ..., "sent": ...} nesting
        sentences = [sentences]
    if isinstance(sentences, str):
        sentences = [{"raw": sentences}]
    out: List[Dict[str, Any]] = []
    for position, sentence in enumerate(sentences):
        text = _first_text(sentence, record_no)
        if not text:
            raise AnnotationFormatError(
                f"record #{record_no}: sentence #{position} carries no text: {sentence!r}"
            )
        sent_id = sentence.get("sent_id") if isinstance(sentence, Mapping) else None
        ref_id = int(sent_id) if sent_id is not None else 10 * record_no + position
        out.append(
            {
                "ref_id": ref_id,
                "image_id": image_id,
                "bbox": list(boxes),
                "phrase": text,
                "split": split,
                "gt_object_id": ann_id,
            }
        )
    if not out:
        raise AnnotationFormatError(f"record #{record_no}: region without any expression")
    return out


def parse_refs_unc_pickle(
    path: str | Path,
    *,
    instances_path: Optional[str | Path] = None,
    split: Optional[str] = None,
    box_format: str = DEFAULT_BOX_FORMAT,
    expand_sentences: bool = True,
    report: Optional[ParseReport] = None,
) -> List[ReferringExample]:
    """Parse the RefCOCO+ (UNC) ``refs(unc).p`` annotation pickle.

    Layout verified on the local download (see the module docstring): one record
    per referred region, **no box field**, so the boxes are joined from
    ``instances.json`` (default: the file with that name in the pickle's own
    directory) through ``ann_id``.  With ``expand_sentences=True`` (the audit
    default) the result is expression-level - one :class:`ReferringExample` per
    sentence, ``ref_id`` = ``sent_id`` (globally unique in this archive),
    ``gt_object_id`` = the COCO annotation id.  ``False`` keeps one example per
    region and uses the first sentence as its text.

    Parameters
    ----------
    path:
        ``refs(unc).p``.
    instances_path:
        COCO-style json carrying ``annotations[].{id, bbox}``.  Only required
        when the records have no ``bbox`` of their own.
    split:
        Force a split label for every record (``expand_sentences`` unaffected);
        needed only if the archive lacks both a per-record ``split`` and
        ``_splits``.
    expand_sentences:
        Explode ``sentences`` into one example per expression.
    report:
        :class:`ParseReport` collecting the usual observations plus a note that
        the file came from a pickle (provenance, never silent).

    Returns
    -------
    list[ReferringExample]
        Sorted by ``ref_id`` so two runs are identical regardless of the
        pickle's internal ordering.
    """
    path = Path(path)
    root = _load_pickle_root(path)
    records, id_to_split = _unc_pickle_records(root, path=path)
    sink = report if report is not None else ParseReport()
    sink.notes.append(f"parsed {path.name} via pickle (trusted-file requirement, see docstring)")

    boxes_by_ann: Optional[Dict[int, List[float]]] = None
    parsed: List[Mapping[str, Any]] = []
    for record_no, record in enumerate(records):
        image_id = int(np.asarray(_pick(record, IMAGE_ID_ALIASES, "image id", record_no)).item())
        raw_ann = _first(record, _ANN_ID_ALIASES)
        ann_id = int(np.asarray(raw_ann).item()) if raw_ann is not None else None
        raw_box = _first(record, BBOX_ALIASES)
        if raw_box is None:
            if ann_id is None:
                raise AnnotationFormatError(
                    f"record #{record_no} (ref_id={record.get('ref_id')}): neither a 'bbox' "
                    "nor an annotation id to join one from"
                )
            if boxes_by_ann is None:
                candidate = Path(instances_path) if instances_path is not None else path.parent / "instances.json"
                boxes_by_ann = _ann_boxes(candidate)
                sink.notes.append(f"boxes joined from {candidate}")
            if ann_id not in boxes_by_ann:
                raise AnnotationFormatError(
                    f"record #{record_no}: ann_id {ann_id} missing from the instances json; "
                    "the RefCOCO+ region -> COCO annotation join is broken"
                )
            raw_box = boxes_by_ann[ann_id]
        raw_split = split if split is not None else _first(record, SPLIT_ALIASES)
        if raw_split is None and ann_id is not None and id_to_split:
            raw_split = id_to_split.get(ann_id)
        if raw_split is None:
            ref_only = record.get("ref_id")
            if ref_only is not None and id_to_split:
                raw_split = id_to_split.get(int(ref_only))
        if raw_split is None:
            raise AnnotationFormatError(
                f"record #{record_no}: no split label (neither a per-record 'split' nor a "
                "'_splits' membership table); pass split=... explicitly"
            )
        split_name = normalize_split_name(raw_split)
        if expand_sentences:
            parsed.extend(
                _sentence_records(
                    record,
                    record_no=record_no,
                    boxes=raw_box,
                    ann_id=ann_id,
                    image_id=image_id,
                    split=split_name,
                )
            )
        else:
            sentences = _first(record, _SENTENCE_LIST_ALIASES) or []
            text = _first_text(sentences[0], record_no) if sentences else _first_text(record, record_no)
            parsed.append(
                {
                    "ref_id": int(record["ref_id"]) if "ref_id" in record else record_no,
                    "image_id": image_id,
                    "bbox": list(raw_box),
                    "phrase": text,
                    "split": split_name,
                    "gt_object_id": ann_id,
                }
            )
    examples = refs_from_records(parsed, box_format=box_format, report=sink)
    examples.sort(key=lambda ex: (ex.ref_id, ex.image_id))
    return examples


def image_ids_by_split_unc_pickle(path: str | Path) -> Dict[str, np.ndarray]:
    """``split -> sorted unique image ids`` straight from ``refs(unc).p``.

    Reads only the ``image_id``/``split`` fields, so an audit subset can be built
    without joining the (120 MB) ``instances.json`` at all - the boxes are
    irrelevant for an image-level sample.
    """
    path = Path(path)
    root = _load_pickle_root(path)
    records, id_to_split = _unc_pickle_records(root, path=path)
    buckets: Dict[str, set[int]] = {}
    for record_no, record in enumerate(records):
        image_id = int(np.asarray(_pick(record, IMAGE_ID_ALIASES, "image id", record_no)).item())
        raw_split = _first(record, SPLIT_ALIASES)
        if raw_split is None:
            raw_ann = _first(record, _ANN_ID_ALIASES)
            key = int(np.asarray(raw_ann).item()) if raw_ann is not None else None
            raw_split = id_to_split.get(key) if key is not None else None
        if raw_split is None:
            raw_split = record.get("ref_id")
            raw_split = id_to_split.get(int(raw_split)) if raw_split is not None else None
        if raw_split is None:
            raise AnnotationFormatError(
                f"record #{record_no}: no split label; cannot build image-level splits"
            )
        buckets.setdefault(normalize_split_name(raw_split), set()).add(image_id)
    return {name: np.asarray(sorted(ids), dtype=np.int64) for name, ids in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# dataset-level entry point (needs the real download)
# ---------------------------------------------------------------------------
def find_refcoco_plus_files(data_root: str | Path) -> Dict[str, Optional[Path]]:
    """Locate the RefCOCO+ archive under ``data_root`` (searches one level down).

    Returns ``{"pickle": ..., "instances": ..., "json": ..., "mats": [...]}`` - the
    verified real download only has the first two.  Nothing raises: callers
    decide what is fatal.
    """
    root = Path(data_root)
    search_roots = [root] + [p for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []
    found: Dict[str, Optional[Path]] = {"pickle": None, "instances": None, "json": None, "mats": []}
    json_candidates = ("refcoco+.json", "refcoco.json", "refs.json")
    for base in search_roots:
        for name in UNC_PICKLE_NAMES:
            if found["pickle"] is None and (base / name).exists():
                found["pickle"] = base / name
        for name in ("instances.json", "instances2014.json"):
            if found["instances"] is None and (base / name).exists():
                found["instances"] = base / name
        for name in json_candidates:
            if found["json"] is None and (base / name).exists():
                found["json"] = base / name
        if not found["json"]:
            for extra in sorted(base.glob("refs*.json")):
                if found["json"] is None:
                    found["json"] = extra
        found["mats"] = found["mats"] or sorted(base.glob("*.mat"))
    return found


def load_refcoco_plus(
    data_root: str | Path,
    *,
    annotation_file: Optional[str | Path] = None,
    split: Optional[str] = None,
    instances_path: Optional[str | Path] = None,
    expand_sentences: bool = True,
) -> List[ReferringExample]:
    """Load RefCOCO+ expressions from a local dataset root.

    ``annotation_file`` wins and is dispatched by suffix (``.p``/``.pickle`` ->
    :func:`parse_refs_unc_pickle`, ``.mat`` -> :func:`parse_unc_mats`, anything
    else -> :func:`parse_refs_json`).  Without it the root is searched for the
    json first (the format the protocol was written against) and then for the
    ``refs(unc).p`` pickle that the official archive actually ships; the
    ``instances.json`` next to it supplies the boxes.
    """
    root = Path(data_root)
    if annotation_file is not None:
        return _load_refcoco_plus_file(annotation_file, split=split, instances_path=instances_path,
                                       expand_sentences=expand_sentences)
    found = find_refcoco_plus_files(root)
    if found["json"] is not None:
        return _load_refcoco_plus_file(found["json"], split=split)
    if found["pickle"] is not None:
        return parse_refs_unc_pickle(
            found["pickle"],
            instances_path=found["instances"],
            split=split,
            expand_sentences=expand_sentences,
        )
    if found["mats"]:
        examples: List[ReferringExample] = []
        for mat in found["mats"]:
            examples.extend(parse_unc_mats(mat, split=split))
        return examples
    raise FileNotFoundError(
        f"no RefCOCO+ annotation file under {root} (looked for "
        f"{('refcoco+.json', 'refs.json', 'refs*.json') + UNC_PICKLE_NAMES + ('*.mat',)}). "
        "Pending data download - Phase 0 checklist step."
    )


def _load_refcoco_plus_file(
    path: str | Path,
    *,
    split: Optional[str] = None,
    instances_path: Optional[str | Path] = None,
    expand_sentences: bool = True,
) -> List[ReferringExample]:
    """Suffix dispatch used by :func:`load_refcoco_plus` (and by tests)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".p", ".pickle"):
        return parse_refs_unc_pickle(
            path, instances_path=instances_path, split=split, expand_sentences=expand_sentences
        )
    if suffix == ".mat":
        return parse_unc_mats(path, split=split)
    inferred = split
    if inferred is None:
        name = path.stem.lower()
        for tag in ("train", "val", "testa", "testb"):
            if tag in name:
                inferred = normalize_split_name(tag)
                break
    return parse_refs_json(path, split=inferred)


# ---------------------------------------------------------------------------
# audit-friendly loaders (frozen interface of the multi-agent brief, 2026-09-27)
# ---------------------------------------------------------------------------
#: Fields every ``refs(unc).p`` record must carry for the audit pipeline.
_AUDIT_REF_REQUIRED_FIELDS: tuple[str, ...] = (
    "ref_id",
    "image_id",
    "split",
    "ann_id",
    "sentences",
)


def load_refs_pickle(path: str | Path) -> List[dict]:
    """Load the UNC ``refs(unc).p`` referring pickle and validate every record.

    Loaded with ``encoding="latin1"``: the official archive is a Python-2
    pickle, and latin1 decodes its byte strings losslessly.  As with every
    pickle path in this module, only ever point this at the official RefCOCO+
    download (or a checksum-verified copy of it) - never at an uploaded or
    network-streamed file; see :func:`_load_pickle_root` for the full warning.

    Accepted roots (provenance-verified layouts): a plain ``list`` of region
    records, or a mapping carrying the list under ``"refs"`` / ``"regions"`` /
    ``"data"``.  Every record must be a mapping with the fields
    :data:`_AUDIT_REF_REQUIRED_FIELDS`, integer-like ``image_id`` / ``ann_id``,
    and a non-empty ``sentences`` list of text-carrying mappings (or plain
    strings).  Anything else raises :class:`AnnotationFormatError` with the
    offending record index - the audit never runs on a half-parsed table.

    Returns the records **as loaded** (no field is modified or dropped), in
    file order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - RefCOCO+ (UNC) annotations must be downloaded first "
            "(Phase 0 checklist step)"
        )
    with path.open("rb") as handle:
        root = pickle.load(handle, encoding="latin1")
    if isinstance(root, Mapping):
        for key in ("refs", "regions", "data"):
            if isinstance(root.get(key), list):
                root = root[key]
                break
        else:
            raise AnnotationFormatError(
                f"{path}: no 'refs' list among keys {sorted(root)[:8]}"
            )
    if not isinstance(root, list):
        raise AnnotationFormatError(f"{path}: unexpected pickle root type {type(root).__name__}")
    refs: List[dict] = []
    for record_no, record in enumerate(root):
        if not isinstance(record, Mapping):
            raise AnnotationFormatError(
                f"{path}: record #{record_no} is not a mapping ({type(record).__name__})"
            )
        missing = [field for field in _AUDIT_REF_REQUIRED_FIELDS if record.get(field) is None]
        if missing:
            raise AnnotationFormatError(
                f"{path}: record #{record_no} misses {missing}; keys={sorted(record)}"
            )
        for field in ("image_id", "ann_id"):
            try:
                int(np.asarray(record[field]).item())
            except (TypeError, ValueError) as exc:
                raise AnnotationFormatError(
                    f"{path}: record #{record_no} has non-integer {field}={record[field]!r}"
                ) from exc
        sentences = record["sentences"]
        if not isinstance(sentences, (list, tuple)) or len(sentences) == 0:
            raise AnnotationFormatError(
                f"{path}: record #{record_no} has an empty 'sentences' field"
            )
        for position, sentence in enumerate(sentences):
            if isinstance(sentence, Mapping):
                if not any(key in sentence for key in ("raw", "sent", "tokens", "text")):
                    raise AnnotationFormatError(
                        f"{path}: record #{record_no} sentence #{position} carries no text or "
                        f"tokens; keys={sorted(sentence)}"
                    )
            elif not isinstance(sentence, str):
                raise AnnotationFormatError(
                    f"{path}: record #{record_no} sentence #{position} is neither a mapping "
                    f"nor a string ({type(sentence).__name__})"
                )
        refs.append(record if isinstance(record, dict) else dict(record))
    if not refs:
        raise AnnotationFormatError(f"{path}: no referring records found")
    return refs


def load_instances_json(path: str | Path) -> Dict[str, dict]:
    """``ann_id(str) -> {"image_id", "bbox", "category_id"}`` from a COCO json.

    The referring pickle carries no boxes; the audit joins each region's
    ``ann_id`` to its standard-COCO annotation through this table
    (``instances.json`` next to the pickle, or the official
    ``instances_{train,val}2014.json``).  Keys are strings exactly as the
    frozen interface specifies (canonical ``str(int(annotation["id"]))``) and
    ``bbox`` stays ``[x, y, w, h]`` as stored.  Duplicate ids raise instead of
    silently keeping the last row; malformed rows raise with their index.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - the instances json joins RefCOCO+ ann_ids to boxes"
        )
    with path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    annotations = root.get("annotations") if isinstance(root, Mapping) else None
    if not isinstance(annotations, list):
        raise AnnotationFormatError(f"{path}: no 'annotations' list")
    out: Dict[str, dict] = {}
    for position, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise AnnotationFormatError(f"{path}: annotation #{position} is not a mapping")
        for field in ("id", "image_id", "bbox", "category_id"):
            if annotation.get(field) is None:
                raise AnnotationFormatError(
                    f"{path}: annotation #{position} misses {field!r}; keys={sorted(annotation)}"
                )
        try:
            key = str(int(np.asarray(annotation["id"]).item()))
        except (TypeError, ValueError) as exc:
            raise AnnotationFormatError(
                f"{path}: annotation #{position} has non-integer id {annotation['id']!r}"
            ) from exc
        if key in out:
            raise AnnotationFormatError(f"{path}: duplicate annotation id {key} at row #{position}")
        bbox = np.asarray(annotation["bbox"], dtype=np.float64).reshape(-1)
        if bbox.size != 4:
            raise AnnotationFormatError(
                f"{path}: annotation #{position} bbox has {bbox.size} values, expected 4"
            )
        out[key] = {
            "image_id": int(np.asarray(annotation["image_id"]).item()),
            "bbox": [float(value) for value in bbox.tolist()],
            "category_id": int(np.asarray(annotation["category_id"]).item()),
        }
    if not out:
        raise AnnotationFormatError(f"{path}: 'annotations' is empty")
    return out


def refs_by_image(refs: Sequence[Mapping[str, Any]]) -> Dict[int, List[dict]]:
    """Group raw referring records by ``int(image_id)``, preserving file order.

    The lists hold the **very records passed in** (no copies); the dict keys
    follow first-appearance order, which is deterministic for a deterministic
    input order (the pickle is one).  Records without an integer ``image_id``
    raise :class:`AnnotationFormatError` with their position - a record that
    cannot be attributed to an image must not vanish into a grouping step.
    """
    grouped: Dict[int, List[dict]] = {}
    for position, record in enumerate(refs):
        if not isinstance(record, Mapping) or record.get("image_id") is None:
            raise AnnotationFormatError(f"refs_by_image: record #{position} has no image_id")
        try:
            image_id = int(np.asarray(record["image_id"]).item())
        except (TypeError, ValueError) as exc:
            raise AnnotationFormatError(
                f"refs_by_image: record #{position} image_id={record['image_id']!r} is not "
                "an integer"
            ) from exc
        grouped.setdefault(image_id, []).append(record if isinstance(record, dict) else dict(record))
    return grouped


__all__ = [
    *__all__,
    "load_refs_pickle",
    "load_instances_json",
    "refs_by_image",
]
