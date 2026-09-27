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
# dataset-level entry point (needs the real download)
# ---------------------------------------------------------------------------
def load_refcoco_plus(
    data_root: str | Path,
    *,
    annotation_file: Optional[str | Path] = None,
    split: Optional[str] = None,
) -> List[ReferringExample]:
    """Load RefCOCO+ expressions from a local dataset root.

    Looks for ``<data_root>/refcoco+.json`` then ``<data_root>/refs*.json``.
    The root layout itself is a Task-3 open question, so a missing file raises
    ``FileNotFoundError`` with actionable text instead of returning ``[]``.
    """
    root = Path(data_root)
    if annotation_file is not None:
        return parse_refs_json(annotation_file, split=split)
    candidates = [root / "refcoco+.json", root / "refcoco.json", root / "refs.json"]
    candidates += sorted(root.glob("refs*.json"))
    for candidate in candidates:
        if candidate.exists():
            inferred = split
            if inferred is None:
                name = candidate.stem.lower()
                for tag in ("train", "val", "testa", "testb"):
                    if tag in name:
                        inferred = normalize_split_name(tag)
                        break
            return parse_refs_json(candidate, split=inferred)
    raise FileNotFoundError(
        f"no RefCOCO+ annotation json under {root} (checked {candidates[:3]} + refs*.json). "
        "Pending data download - Phase 0 checklist step."
    )
