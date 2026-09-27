"""Run the Phase 0 proposal audit on the frozen image subset.

Pipeline (one GPU pass, then pure numpy):

1. read ``data/audit_subset.csv`` (image_id, file_name, ...);
2. per image: load pixels -> :func:`ccg.data.rpn.extract_proposals` (top-N,
   objectness-sorted, original-image xyxy) -> cache ``cache/<image_id>.npz``
   (``boxes``/``objectness`` float32 + ``meta_json``); an existing cache entry
   is reused as is - the detector is *never* re-run for it;
3. analyse N in {64, 128} by truncating the cached 128-bank to its first N rows;
4. aggregate with :mod:`ccg.data.audit_report` and write the artifacts:

       results/proposal_audit/
           summary.json
           recall_by_N.csv
           candidate_availability_by_N.csv
           same_category_availability.csv
           natural_omission.csv
           proposal_iou_statistics.csv
           figures/*.png                (--figures)

Missing images are skipped and *counted* (``skipped_missing``) - the CSCOCO
download may still be running while the audit is prepared.  The frozen
interfaces (``ccg.data.rpn`` / ``ccg.data.audit`` / ``ccg.data.refcoco``) are
implemented by parallel agents; every access is wrapped so an unfinished
module produces one explicit error instead of an obscure traceback.

Usage (from the repo root, ``PYTHONPATH=src``; conda env ``deepminer``):

    python scripts/run_proposal_audit.py --max-images 5 --device cuda --figures
    python scripts/run_proposal_audit.py                      # full 1500-image run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from ccg.data.audit_report import (  # noqa: E402  (path fixed above; pure module)
    DEFAULT_K_LEVELS,
    DEFAULT_RECALL_THRESHOLDS,
    DEFAULT_SAME_CATEGORY_LEVELS,
    aggregate_audit,
    group_refs_by_image,
    make_expression_record,
    make_image_record,
    render_figures,
    write_csv_artifacts,
    write_summary_json,
)

__all__ = [
    "FrozenInterfaceError",
    "DEFAULT_LEVELS",
    "resolve_image_path",
    "load_subset_rows",
    "load_or_extract",
    "build_parser",
    "run",
    "main",
]

#: Truncation levels audited out of one cached 128-bank (spec: N in {64, 128}).
DEFAULT_LEVELS: tuple[int, ...] = (64, 128)


class FrozenInterfaceError(RuntimeError):
    """A module/function of the frozen interface is not available (yet)."""


# ---------------------------------------------------------------------------
# frozen-interface wrappers (single choke point, monkeypatch-friendly)
# ---------------------------------------------------------------------------
def build_rpn_model_for(device: str) -> Any:
    """Load the COCO-pretrained Faster R-CNN used for its RPN stage."""
    try:
        from ccg.data.rpn import build_rpn_model
    except ImportError as exc:
        raise FrozenInterfaceError(
            f"ccg.data.rpn is not importable ({exc}); the proposal extraction module is "
            "still being implemented by the parallel agent - re-run once it lands"
        ) from exc
    return build_rpn_model(weights="COCO_V1", device=device)


def extract_for_image(image_path: Path, model: Any, top_n: int, autocast: bool):
    """``extract_proposals`` wrapper: returns ``(boxes, objectness, meta)``.

    The frozen ``ccg.data.rpn`` contract returns an ``RPNProposals`` dataclass
    (``boxes`` / ``objectness`` / ``meta``); a plain tuple return is accepted
    too so minimal test stubs keep working.
    """
    try:
        from ccg.data.rpn import extract_proposals
    except ImportError as exc:
        raise FrozenInterfaceError(
            f"ccg.data.rpn is not importable ({exc}); proposal extraction unavailable"
        ) from exc
    result = extract_proposals(image_path, model, top_n=int(top_n), autocast=bool(autocast))
    if hasattr(result, "boxes") and hasattr(result, "objectness"):
        return result.boxes, result.objectness, dict(getattr(result, "meta", None) or {})
    boxes, objectness, meta = result  # tuple-style return (test stubs)
    return boxes, objectness, meta


def load_gt_index(annotation_file: Path, image_ids: Sequence[int]):
    """COCO GT index restricted to the audited images (iscrowd rows are kept but
    the audit drops them per spec via ``GtObjects.non_crowd()``)."""
    try:
        from ccg.data.coco import load_coco_index
    except ImportError as exc:
        raise FrozenInterfaceError(f"ccg.data.coco is not importable ({exc})") from exc
    return load_coco_index([Path(annotation_file)], only_image_ids=list(image_ids))


def _first_sentence_text(sentences: Any) -> str:
    """First non-empty expression text of a raw RefCOCO+ region record."""
    if isinstance(sentences, str):
        return sentences.strip()
    for sentence in sentences or []:
        text = ""
        if isinstance(sentence, str):
            text = sentence.strip()
        elif isinstance(sentence, Mapping):
            for key in ("raw", "sent", "sentence", "sentence_raw", "text", "phrase"):
                value = sentence.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
        if text:
            return text
    return ""


#: Join failures collected by the last :func:`load_refcoco_refs` call (notes).
_REF_LOAD_SKIPS: List[str] = []


def load_refcoco_refs(pickle_path: Path, instances_path: Optional[Path]) -> List[Any]:
    """RefCOCO+ **region** records as :class:`ccg.data.types.ReferringExample`.

    Uses the frozen trio of the interface brief: ``load_refs_pickle``
    (validated latin1 region records) + ``load_instances_json``
    (``ann_id -> bbox/category_id`` of the standard COCO annotations) +
    ``refs_by_image`` (per-image grouping).  One example per *region* - the
    unit the subset's ``ref_count`` column counts - with the region's first
    sentence as text and the ``ann_id`` box (xywh -> xyxy) as its target.
    Regions whose join is broken are skipped and counted in
    ``_REF_LOAD_SKIPS``; they never abort the audit.
    """
    try:
        from ccg.data.refcoco import load_instances_json, load_refs_pickle
        from ccg.data.refcoco import refs_by_image as group_records_by_image
        from ccg.data.types import ReferringExample
    except ImportError as exc:
        raise FrozenInterfaceError(
            f"ccg.data.refcoco is not importable ({exc}); RefCOCO+ loading unavailable"
        ) from exc
    records = load_refs_pickle(Path(pickle_path))
    instances_file = (
        Path(instances_path)
        if instances_path is not None and Path(instances_path).exists()
        else Path(pickle_path).parent / "instances.json"
    )
    instances = load_instances_json(instances_file)
    _REF_LOAD_SKIPS.clear()
    examples: List[Any] = []
    for image_id, image_records in group_records_by_image(records).items():
        for record in image_records:
            ref_id = int(record["ref_id"])
            ann_id = int(record["ann_id"])
            info = instances.get(str(ann_id))
            if info is None:
                _REF_LOAD_SKIPS.append(
                    f"image {image_id} region {ref_id}: ann_id {ann_id} absent from "
                    f"{instances_file.name}"
                )
                continue
            text = _first_sentence_text(record.get("sentences"))
            x, y, width, height = (float(value) for value in info["bbox"])
            if not text or width <= 0.0 or height <= 0.0:
                _REF_LOAD_SKIPS.append(
                    f"image {image_id} region {ref_id}: unjoinable text/box"
                )
                continue
            examples.append(
                ReferringExample(
                    ref_id=ref_id,
                    image_id=int(image_id),
                    text=text,
                    gt_box=np.asarray([x, y, x + width, y + height], dtype=np.float64),
                    gt_object_id=ann_id,
                    split=str(record.get("split", "")),
                )
            )
    examples.sort(key=lambda example: (example.image_id, example.ref_id))
    return examples


_AUDIT_FNS: Optional[SimpleNamespace] = None


def _audit_fns() -> SimpleNamespace:
    """The frozen per-image audit primitives (imported once, lazily).

    Migrated to the frozen ten-function contract of ``ccg.data.audit``
    (2026-09-27): ``assign_gt_category`` / ``same_category_counts`` instead of
    the draft's ``same_category_availability``, ``proposal_recall`` +
    ``remaining_candidate_count`` + ``natural_omission_rate`` instead of
    ``max_iou_per_object``, and the new ``redundancy_stats`` return shape.
    """
    global _AUDIT_FNS
    if _AUDIT_FNS is None:
        try:
            from ccg.data.audit import (
                assign_gt_category,
                proposal_recall,
                remaining_candidate_count,
                redundancy_stats,
                same_category_counts,
                target_availability,
            )
            from ccg.data.proposals import assign_target, iou_matrix
            from ccg.data.types import ProposalBank
        except ImportError as exc:
            raise FrozenInterfaceError(
                f"ccg.data.audit / ccg.data.proposals are not importable ({exc}); "
                "the audit statistics module is still being implemented"
            ) from exc
        _AUDIT_FNS = SimpleNamespace(
            assign_gt_category=assign_gt_category,
            proposal_recall=proposal_recall,
            remaining_candidate_count=remaining_candidate_count,
            redundancy_stats=redundancy_stats,
            same_category_counts=same_category_counts,
            target_availability=target_availability,
            assign_target=assign_target,
            iou_matrix=iou_matrix,
            ProposalBank=ProposalBank,
        )
    return _AUDIT_FNS


# ---------------------------------------------------------------------------
# subset / paths / cache
# ---------------------------------------------------------------------------
def load_subset_rows(path: str | Path) -> List[Dict[str, Any]]:
    """Read the subset CSV; ``image_id``/``file_name`` required, extras kept."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audit subset {path} not found (run build_audit_subset.py first)")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for required in ("image_id", "file_name"):
            if required not in fields:
                raise ValueError(f"{path}: missing column {required!r}; got {fields}")
        for raw in reader:
            row: Dict[str, Any] = {
                "image_id": int(raw["image_id"]),
                "file_name": str(raw["file_name"]).strip(),
            }
            for extra, cast in (("ref_count", int), ("refcoco_split", str), ("coco_split", str)):
                value = raw.get(extra)
                if value not in (None, ""):
                    row[extra] = cast(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def resolve_image_path(images_root: str | Path, file_name: str) -> Optional[Path]:
    """Locate an image under ``images_root`` (flat or split-folder layout).

    The verified layout is ``data/raw/mscoco/train2014/<file_name>``; a flat
    ``images_root/<file_name>`` is accepted too.  Returns ``None`` when the
    file is absent - the caller counts it as ``skipped_missing`` (the download
    may legitimately still be running).
    """
    root = Path(images_root)
    for candidate in (root / file_name, root / "train2014" / file_name):
        if candidate.exists():
            return candidate
    return None


def load_or_extract(
    image_id: int,
    image_path: Path,
    cache_dir: Path,
    top_n: int,
    get_model: Callable[[], Any],
    autocast: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], bool]:
    """Cache-first proposal loader: ``(boxes, objectness, meta, from_cache)``."""
    cache_file = cache_dir / f"{int(image_id)}.npz"
    if cache_file.exists():
        try:
            with np.load(cache_file, allow_pickle=False) as store:
                boxes = np.asarray(store["boxes"], dtype=np.float32).reshape(-1, 4)
                objectness = np.asarray(store["objectness"], dtype=np.float32).reshape(-1)
                raw_meta = store["meta_json"]
                meta = json.loads(str(raw_meta)) if raw_meta.size else {}
            if boxes.shape[0] and boxes.shape[0] == objectness.shape[0]:
                return boxes, objectness, meta, True
        except Exception:
            pass  # corrupt entry: recompute below (counted as an extraction)
    boxes, objectness, meta = extract_for_image(image_path, get_model(), int(top_n), autocast)
    boxes = np.ascontiguousarray(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
    objectness = np.ascontiguousarray(np.asarray(objectness, dtype=np.float32).reshape(-1))
    if boxes.shape[0] == 0:
        raise RuntimeError(f"RPN returned no proposals for image {image_id}")
    if boxes.shape[0] != objectness.shape[0]:
        raise RuntimeError(
            f"image {image_id}: boxes/objectness length mismatch "
            f"({boxes.shape[0]} vs {objectness.shape[0]})"
        )
    np.savez_compressed(
        cache_file,
        boxes=boxes,
        objectness=objectness,
        meta_json=json.dumps(meta, default=str),
    )
    return boxes, objectness, meta, False


# ---------------------------------------------------------------------------
# per-image auditing
# ---------------------------------------------------------------------------
def audit_level(
    image_id: int,
    bank_boxes: np.ndarray,
    bank_scores: np.ndarray,
    gt_boxes: np.ndarray,
    gt_categories: np.ndarray,
    refs: Sequence[Any],
    n_cap: int,
    iou_thresh: float,
    k_levels: Sequence[int],
    recall_thresholds: Sequence[float] = DEFAULT_RECALL_THRESHOLDS,
) -> Dict[str, Any]:
    """One image record at truncation ``n_cap`` (frozen primitives only)."""
    fns = _audit_fns()
    bank = fns.ProposalBank(image_id=int(image_id), boxes=bank_boxes, objectness=bank_scores)
    gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_cats = np.asarray(gt_categories, dtype=np.int64).reshape(-1)
    num_props = int(bank.boxes.shape[0])

    # per-GT-object max IoU (the raw sample behind every GT-object statistic)
    if gt.shape[0] and num_props:
        gt_max_ious = fns.iou_matrix(gt, bank.boxes).max(axis=1)  # [M]
    else:
        gt_max_ious = np.zeros(gt.shape[0], dtype=np.float32)

    # GT-object recall counts from the frozen proposal_recall; make_image_record
    # cross-checks them against gt_max_ious so the two views always agree
    gt_recall_counts: Dict[str, List[int]] = {}
    for threshold in recall_thresholds:
        rate = float(fns.proposal_recall(bank.boxes, gt, float(threshold)))
        gt_recall_counts[f"{float(threshold):g}"] = [
            int(round(rate * gt.shape[0])),
            int(gt.shape[0]),
        ]

    redundancy = fns.redundancy_stats(bank.boxes)
    categories, _best_ious = fns.assign_gt_category(
        bank.boxes, gt, gt_cats, iou_thresh=float(iou_thresh)
    )

    expressions: List[Dict[str, Any]] = []
    for example in refs:
        assignment = fns.assign_target(bank, example.gt_box, iou_thresh=float(iou_thresh))
        target_idx = assignment.target_proposal_idx
        to_remove = assignment.to_remove
        availability = fns.target_availability(
            bank.boxes, target_idx, to_remove, Ks=list(k_levels)
        )
        remaining = int(fns.remaining_candidate_count(bank.boxes, target_idx, to_remove))
        same_category = int(fns.same_category_counts(categories, target_idx, to_remove))
        unknown = bool(target_idx is not None and int(categories[int(target_idx)]) == -1)
        expressions.append(
            make_expression_record(
                ref_id=int(example.ref_id),
                max_iou=float(assignment.best_iou),
                is_miss=bool(assignment.is_miss),
                num_remaining_candidates=remaining,
                num_equivalent_removed=(0 if to_remove is None else int(np.size(to_remove))),
                num_same_category_distractors=same_category,
                available=availability,
                target_category_unknown=unknown,
            )
        )
    return make_image_record(
        image_id=image_id,
        N_cap=int(n_cap),
        num_proposals=num_props,
        gt_max_ious=gt_max_ious,
        gt_recall_counts=gt_recall_counts,
        redundancy=redundancy,
        expressions=expressions,
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the audit described by ``args``; writes artifacts; returns summary."""
    started = time.perf_counter()
    subset_path = Path(args.subset)
    out_dir = Path(args.out)
    cache_dir = Path(args.cache)
    images_root = Path(args.images_root)
    levels = [int(n) for n in DEFAULT_LEVELS if int(n) <= int(args.topn)]
    if not levels:
        raise ValueError(f"--topn {args.topn} is below the smallest audited level {DEFAULT_LEVELS[0]}")

    print(f"[proposal-audit] subset      : {subset_path}")
    rows = load_subset_rows(subset_path)
    print(f"[proposal-audit] rows        : {len(rows)}")

    print(f"[proposal-audit] refcoco+    : {args.refcoco_pickle}")
    refs = load_refcoco_refs(Path(args.refcoco_pickle), Path(args.refcoco_instances))
    refs_by_image = group_refs_by_image(refs)
    print(
        f"[proposal-audit] refs        : {len(refs)} regions on {len(refs_by_image)} images"
        + (f" ({len(_REF_LOAD_SKIPS)} region(s) skipped)" if _REF_LOAD_SKIPS else "")
    )

    print(f"[proposal-audit] gt          : {args.coco_annotations}")
    gt_index = load_gt_index(Path(args.coco_annotations), [r["image_id"] for r in rows])
    print(f"[proposal-audit] gt index    : {len(gt_index)} images (subset-restricted)")

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_holder: List[Any] = [None]

    def get_model() -> Any:
        if model_holder[0] is None:
            print(f"[proposal-audit] loading RPN weights (device={args.device}) ...")
            model_holder[0] = build_rpn_model_for(str(args.device))
        return model_holder[0]

    peak_mem_mb: Optional[float] = None
    torch_mod: Any = None
    if str(args.device).startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch_mod = torch
        except Exception:
            torch_mod = None

    records: List[Dict[str, Any]] = []
    n_cached = n_extracted = n_failed = n_skipped = 0
    failure_messages: List[str] = []
    max_images = int(args.max_images)
    processed = 0

    iterator, print_plain = _progress_iterator(rows)
    for index, row in enumerate(iterator):
        if max_images > 0 and processed >= max_images:
            break
        image_id = int(row["image_id"])
        image_path = resolve_image_path(images_root, row["file_name"])
        if image_path is None:
            n_skipped += 1
            continue
        try:
            boxes, objectness, _meta, from_cache = load_or_extract(
                image_id, image_path, cache_dir, int(args.topn), get_model, bool(args.autocast)
            )
        except Exception as exc:  # one bad image must not kill the audit
            n_failed += 1
            message = f"image {image_id}: {type(exc).__name__}: {exc}"
            if len(failure_messages) < 10:
                failure_messages.append(message)
            processed += 1
            continue
        if from_cache:
            n_cached += 1
        else:
            n_extracted += 1
        processed += 1

        gt = gt_index.objects(image_id)
        gt_nc = gt.non_crowd()  # spec: GT-object stats exclude iscrowd == 1
        image_refs = refs_by_image.get(image_id, [])
        for n_cap in levels:
            n_used = min(int(n_cap), boxes.shape[0])
            records.append(
                audit_level(
                    image_id=image_id,
                    bank_boxes=boxes[:n_used],
                    bank_scores=objectness[:n_used],
                    gt_boxes=gt_nc.boxes,
                    gt_categories=gt_nc.categories,
                    refs=image_refs,
                    n_cap=n_cap,
                    iou_thresh=float(args.iou_thresh),
                    k_levels=DEFAULT_K_LEVELS,
                )
            )
        if print_plain and (index + 1) % 100 == 0:
            print(
                f"[proposal-audit] progress    : rows={index + 1} analyzed={processed} "
                f"skipped_missing={n_skipped} failed={n_failed}"
            )

    if torch_mod is not None:
        try:
            peak_mem_mb = float(torch_mod.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        except Exception:
            peak_mem_mb = None

    analyzed = processed - n_failed
    agg: Dict[str, Any] = (
        aggregate_audit(
            records,
            k_levels=DEFAULT_K_LEVELS,
            same_category_levels=DEFAULT_SAME_CATEGORY_LEVELS,
            iou_thresh=float(args.iou_thresh),
        )
        if records
        else {"levels": [], "per_N": {}}
    )

    notes = _build_notes(
        rows, refs_by_image, records, agg, failure_messages, ref_load_skips=_REF_LOAD_SKIPS
    )
    cache_size_mb = _dir_size_mb(cache_dir)
    summary = {
        "task": "proposal_audit",
        "config": {
            "subset": str(subset_path),
            "out_dir": str(out_dir),
            "cache_dir": str(cache_dir),
            "images_root": str(images_root),
            "coco_annotations": str(args.coco_annotations),
            "refcoco_pickle": str(args.refcoco_pickle),
            "refcoco_instances": str(args.refcoco_instances),
            "top_n": int(args.topn),
            "levels": levels,
            "batch_size": int(args.batch_size),
            "device": str(args.device),
            "autocast": bool(args.autocast),
            "max_images": max_images,
            "figures": bool(args.figures),
            "iou_thresh": float(args.iou_thresh),
            "K_levels": list(DEFAULT_K_LEVELS),
            "same_category_levels": list(DEFAULT_SAME_CATEGORY_LEVELS),
        },
        "versions": _versions(),
        "n_images": {
            "subset_rows": len(rows),
            "analyzed": analyzed,
            "skipped_missing": n_skipped,
            "failed": n_failed,
            "cached_reused": n_cached,
            "extracted": n_extracted,
        },
        "failed_examples": failure_messages,
        "cache_size_mb": cache_size_mb,
        "runtime_seconds": float(time.perf_counter() - started),
        "gpu_peak_mem_mb": peak_mem_mb,
        "per_N": agg["per_N"],
        "notes": notes,
    }

    write_csv_artifacts(out_dir, agg)
    summary_path = write_summary_json(out_dir / "summary.json", summary)
    print(f"[proposal-audit] summary     : {summary_path}")
    for name in ("recall_by_N.csv", "candidate_availability_by_N.csv",
                 "same_category_availability.csv", "natural_omission.csv",
                 "proposal_iou_statistics.csv"):
        print(f"[proposal-audit] wrote       : {out_dir / name}")

    if args.figures:
        if records:
            figures = render_figures(records, agg, out_dir / "figures")
            for name, path in sorted(figures.items()):
                print(f"[proposal-audit] figure      : {path}")
        else:
            print("[proposal-audit] figures skipped: no records to plot")

    _print_headline(summary)
    return summary


def _progress_iterator(rows: Sequence[Mapping[str, Any]]):
    """tqdm when available, plain iteration (caller prints every 100) otherwise."""
    try:
        from tqdm import tqdm

        return tqdm(list(rows), desc="proposal-audit", unit="img"), False
    except ImportError:  # pragma: no cover - tqdm is a listed dependency
        return rows, True


def _versions() -> Dict[str, Any]:
    try:
        import torch
        import torchvision

        return {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        }
    except Exception:
        return {"python": sys.version.split()[0], "numpy": np.__version__, "torch": None,
                "torchvision": None}


def _dir_size_mb(directory: Path) -> float:
    total = 0
    for path in directory.glob("*.npz"):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return float(total / (1024.0 * 1024.0))


def _build_notes(
    rows: Sequence[Mapping[str, Any]],
    refs_by_image: Mapping[int, Sequence[Any]],
    records: Sequence[Mapping[str, Any]],
    agg: Mapping[str, Any],
    failure_messages: Sequence[str],
    ref_load_skips: Sequence[str] = (),
) -> List[str]:
    notes = [
        "wilson_ci comes from the frozen ccg.data.audit contract (re-exported by "
        "ccg.data.audit_report); every rate carries a Wilson 95% interval and "
        "empty populations report 0.0 with (0.0, 1.0).",
        "candidate availability follows the frozen rule "
        "available[K] = (N - |{target} U to_remove|) >= K - 1.",
        "RefCOCO+ targets loaded at the *region* level via the frozen ccg.data.refcoco "
        "trio load_refs_pickle + load_instances_json + refs_by_image; each region joins "
        "instances.json through ann_id (xywh -> xyxy) and carries its first sentence as "
        "text - the same unit the subset's ref_count column counts.",
        "GT-object statistics use instances_train2014.json with iscrowd==1 rows excluded.",
    ]
    with_ref_count = [r for r in rows if "ref_count" in r]
    if with_ref_count:
        mismatches = [
            int(r["image_id"])
            for r in with_ref_count
            if int(r["ref_count"]) != len(refs_by_image.get(int(r["image_id"]), []))
        ]
        notes.append(
            f"ref_count column cross-check (informational): {len(with_ref_count)} rows, "
            f"{len(mismatches)} mismatches vs loaded RefCOCO+ regions"
            + (f", e.g. {mismatches[:5]}" if mismatches else "")
        )
    no_refs = [
        i for i in sorted({int(r["image_id"]) for r in records}) if not refs_by_image.get(i)
    ]
    if no_refs:
        notes.append(
            f"{len(no_refs)} audited image(s) have no RefCOCO+ regions (GT-only stats), "
            f"e.g. {no_refs[:5]}"
        )
    for key, level in sorted((agg.get("per_N") or {}).items(), key=lambda kv: int(kv[0])):
        short = int(level.get("num_images_short_bank", 0))
        if short:
            notes.append(
                f"N={key}: {short} image(s) had fewer than {key} cached proposals "
                "(bank trimmed by the detector); statistics use the actual bank size."
            )
    for message in ref_load_skips[:5]:
        notes.append(f"ref join skip: {message}")
    for message in failure_messages[:10]:
        notes.append(f"extraction failure: {message}")
    return notes


def _print_headline(summary: Mapping[str, Any]) -> None:
    counts = summary["n_images"]
    print(
        f"[proposal-audit] images      : analyzed={counts['analyzed']} "
        f"cached={counts['cached_reused']} extracted={counts['extracted']} "
        f"skipped_missing={counts['skipped_missing']} failed={counts['failed']}"
    )
    for key, level in sorted((summary.get("per_N") or {}).items(), key=lambda kv: int(kv[0])):
        gt = level["recall"]["gt_object"]["by_threshold"]
        ref = level["recall"]["ref_target"]["by_threshold"]
        omit_expr = level["natural_omission"]["expression"]["rate"]
        omit_gt = level["natural_omission"]["gt_object"]["rate"]
        avail10 = level["candidate_availability"].get("10", {}).get("frac_available")
        print(
            f"[proposal-audit] N={key:<4} gt_recall@0.5={gt['0.5']['recall']:.3f} "
            f"@0.7={gt['0.7']['recall']:.3f} | ref_recall@0.5={ref['0.5']['recall']:.3f} "
            f"@0.7={ref['0.7']['recall']:.3f} | omission expr={omit_expr:.3f} "
            f"gt={omit_gt:.3f} | avail(K=10)={avail10:.3f}"
        )
    print(f"[proposal-audit] runtime     : {summary['runtime_seconds']:.1f}s"
          + (f", gpu peak {summary['gpu_peak_mem_mb']:.0f} MB"
             if summary.get("gpu_peak_mem_mb") is not None else ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset", type=Path, default=_REPO_ROOT / "data" / "audit_subset.csv")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "results" / "proposal_audit")
    parser.add_argument("--cache", type=Path, default=_REPO_ROOT / "cache" / "proposal_audit")
    parser.add_argument("--images-root", type=Path, default=_REPO_ROOT / "data" / "raw" / "mscoco")
    parser.add_argument(
        "--coco-annotations",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "annotations" / "instances_train2014.json",
    )
    parser.add_argument(
        "--refcoco-pickle",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "refs(unc).p",
    )
    parser.add_argument(
        "--refcoco-instances",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "instances.json",
    )
    parser.add_argument("--topn", type=int, default=128, help="cached bank size (>= 64)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="images per progress chunk; the frozen extract_proposals API is "
        "single-image, so this only bounds progress granularity",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=0, help="0 = all images")
    parser.add_argument("--figures", action="store_true", help="render diagnostic figures")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument(
        "--no-autocast", action="store_false", dest="autocast", help="disable FP16 autocast"
    )
    parser.set_defaults(autocast=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_images) < 0:
        print("[proposal-audit] --max-images must be >= 0")
        return 2
    if int(args.topn) < DEFAULT_LEVELS[0]:
        print(f"[proposal-audit] --topn must be >= {DEFAULT_LEVELS[0]}, got {args.topn}")
        return 2
    try:
        summary = run(args)
    except FrozenInterfaceError as exc:
        print(f"[proposal-audit] FATAL: {exc}")
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"[proposal-audit] FATAL: {exc}")
        return 1
    analyzed = summary["n_images"]["analyzed"]
    if analyzed == 0:
        print(
            "[proposal-audit] WARNING: no image could be analysed "
            f"(skipped_missing={summary['n_images']['skipped_missing']}, "
            f"failed={summary['n_images']['failed']}); artifacts contain empty aggregates"
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
