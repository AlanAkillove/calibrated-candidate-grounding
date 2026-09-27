"""Extract the frozen class-agnostic RPN proposal bank for every RefCOCO+ image.

Pipeline (protocol section 4 / Amendment A2, the ``bank-v1`` contract)
----------------------------------------------------------------------
``I -> P(I) = {p_1 .. p_N}``, ``N = 64``, produced by **one frozen detector**
(torchvision ``fasterrcnn_resnet50_fpn`` COCO_V1, its RPN stage only) that
never sees the referring expression.  One GPU pass over the image manifest:

1. read the image manifest (``image_id, file_name, coco_split, ref_count``);
2. with ``--resume`` skip every image already stored in the target h5;
3. prefetch PIL decodes on a thread pool (bounded in-flight window);
4. on the main thread: ``ccg.data.rpn.extract_proposals`` -> append one
   ``bank-v1`` group via :func:`ccg.data.bank.write_bank_entry` (h5py ``"a"``
   mode; every 200 images the file is flushed and a progress + throughput line
   is printed);
5. after the loop: :func:`ccg.data.bank.finalize_bank` (root attrs) plus
   ``<out>_stats.json`` with the distribution summaries and, verbatim, every
   failed image ([{image_id, reason}] - failures are counted, never silent).

Robustness contract
-------------------
* Half-written groups (missing ``boxes``/``objectness``, wrong shape, empty)
  are detected up front by :func:`ccg.data.bank.find_corrupt_images`; with
  ``--repair`` they are deleted and re-extracted, without it the run aborts
  with the offending image ids (no silent reuse of broken data).
* A corrupt bank built with a different ``--top-n`` is refused: the bank size
  is part of the experiment identity.
* Any per-image load/decode/inference failure is recorded and counted; the
  run continues.  Missing files (download still running) show up in the same
  list.

Usage (repo root, conda env ``deepminer``; ``PYTHONPATH=src`` not needed -
the script inserts ``src`` itself)::

    python scripts/extract_proposals.py \
        --images-root data/raw/mscoco \
        --manifest data/full_image_manifest.csv \
        --out cache/proposals.h5 --top-n 64 --device cuda \
        --resume --prefetch-workers 8

    python scripts/extract_proposals.py --max-images 20 ...   # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from ccg.data.bank import (  # noqa: E402  (path fixed above)
    BANK_SCHEMA_VERSION,
    bank_attrs,
    finalize_bank,
    find_corrupt_images,
    image_ids,
    repair_bank,
    write_bank_entry,
)
from ccg.data.rpn import MODEL_NAME, build_rpn_model, extract_proposals  # noqa: E402

__all__ = [
    "DEFAULT_TOP_N",
    "DEFAULT_WEIGHTS",
    "PROGRESS_EVERY",
    "build_parser",
    "load_manifest",
    "resolve_image_path",
    "load_image_rgb",
    "run",
    "main",
]

DEFAULT_TOP_N = 64
DEFAULT_WEIGHTS = "COCO_V1"
#: flush + progress line cadence (images)
PROGRESS_EVERY = 200


# ---------------------------------------------------------------------------
# manifest / image IO
# ---------------------------------------------------------------------------
def load_manifest(path: str | Path) -> Tuple[List[Dict[str, Any]], int]:
    """Read the image manifest; returns ``(rows, num_duplicate_rows)``.

    Required columns: ``image_id``, ``file_name``.  ``coco_split`` and
    ``ref_count`` are carried along when present.  Duplicate ``image_id`` rows
    are dropped (first wins) and **counted** - the file is a generated
    artefact, a duplicate means the generator must be checked.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"image manifest {path} not found - it is generated once the COCO image "
            "download completes (data/full_image_manifest.csv)"
        )
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    duplicates = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for required in ("image_id", "file_name"):
            if required not in fields:
                raise ValueError(f"{path}: missing column {required!r}; got {fields}")
        for raw in reader:
            image_id = int(raw["image_id"])
            if image_id in seen:
                duplicates += 1
                continue
            seen.add(image_id)
            row: Dict[str, Any] = {
                "image_id": image_id,
                "file_name": str(raw["file_name"]).strip(),
            }
            for extra, cast in (("coco_split", str), ("ref_count", int)):
                value = raw.get(extra)
                if value not in (None, ""):
                    row[extra] = cast(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows, duplicates


def resolve_image_path(images_root: str | Path, file_name: str) -> Optional[Path]:
    """Locate an image under ``images_root`` (flat or ``train2014/`` layout).

    The verified layout is ``data/raw/mscoco/train2014/<file_name>``; returns
    ``None`` when the file is absent (the download may still be running - the
    caller records it as a failure).
    """
    root = Path(images_root)
    for candidate in (root / file_name, root / "train2014" / file_name):
        if candidate.exists():
            return candidate
    return None


def load_image_rgb(path: Path) -> Any:
    """Fully decoded RGB PIL image (runs on the prefetch workers)."""
    from PIL import Image

    with Image.open(path) as handle:
        return handle.convert("RGB")


# ---------------------------------------------------------------------------
# distribution summaries for the stats artefact
# ---------------------------------------------------------------------------
def _summary_stats(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "p5": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _version_block() -> Dict[str, Any]:
    block: Dict[str, Any] = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import torch
        import torchvision

        block["torch"] = torch.__version__
        block["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover - torch is a hard environment dependency
        block["torch"] = None
        block["torchvision"] = None
    return block


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the extraction described by ``args``; returns the stats summary."""
    started = time.perf_counter()
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    images_root = Path(args.images_root)
    top_n = int(args.top_n)
    workers = int(args.prefetch_workers)
    resume = bool(args.resume)

    print(f"[extract] manifest    : {manifest_path}")
    rows, duplicates = load_manifest(manifest_path)
    if duplicates:
        print(f"[extract] WARNING     : {duplicates} duplicate image_id row(s) dropped")
    print(f"[extract] manifest    : {len(rows)} images")

    # --- integrity of an existing bank -------------------------------------
    existing_ids: set = set()
    if out_path.exists():
        corrupt = find_corrupt_images(out_path)
        if corrupt:
            if args.repair:
                removed = repair_bank(out_path)
                print(
                    f"[extract] repair      : removed {len(removed)} corrupt group(s) "
                    f"e.g. {removed[:5]}"
                )
            else:
                head = ", ".join(str(i) for i in corrupt[:10])
                more = f" (+{len(corrupt) - 10} more)" if len(corrupt) > 10 else ""
                raise RuntimeError(
                    f"{len(corrupt)} corrupt (half-written or malformed) group(s) in "
                    f"{out_path}: {head}{more}; re-run with --repair to delete and re-extract them"
                )
        existing_ids = set(int(i) for i in image_ids(out_path).tolist())
        attrs = bank_attrs(out_path)
        if attrs and attrs.get("top_n") is not None and int(attrs["top_n"]) != top_n:
            raise RuntimeError(
                f"{out_path} was built with top_n={attrs['top_n']} but --top-n is {top_n}; "
                "bank size is part of the experiment identity - use a different --out"
            )
        print(f"[extract] bank        : {out_path} already holds {len(existing_ids)} image(s)")

    # --- work list ----------------------------------------------------------
    todo: List[Dict[str, Any]] = []
    skipped_resume = 0
    for row in rows:
        if resume and row["image_id"] in existing_ids:
            skipped_resume += 1
            continue
        todo.append(row)
    if int(args.max_images) > 0:
        todo = todo[: int(args.max_images)]
    print(
        f"[extract] to process  : {len(todo)} images "
        f"(skipped_resume={skipped_resume}, out={out_path}, top_n={top_n})"
    )

    # --- model (lazily: a pure resume run with nothing to do needs no GPU) --
    model = None
    peak_vram_mb: Optional[float] = None
    device_name = str(args.device)
    torch_mod = None
    if todo:
        print(f"[extract] loading     : {MODEL_NAME} ({DEFAULT_WEIGHTS}) on {args.device} ...")
        model = build_rpn_model(device=str(args.device), weights=DEFAULT_WEIGHTS)
        try:
            import torch

            torch_mod = torch
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                device_name = torch.cuda.get_device_name(torch.device(args.device))
        except Exception:
            torch_mod = None

    # --- prefetch + extract + write -----------------------------------------
    failures: List[Dict[str, Any]] = []
    n_done = 0
    n_failed = 0
    n_crops = 0
    processed_t0: Optional[float] = None

    # tqdm auto-disables itself when the stream is not a TTY (disable=None).
    processing_bar = (
        tqdm(total=len(todo), desc="rpn", unit="img", disable=None) if todo else None
    )

    def submit_more(executor: ThreadPoolExecutor, iterator: Iterator[Dict[str, Any]], pending) -> None:
        while len(pending) < workers:
            try:
                row = next(iterator)
            except StopIteration:
                return
            pending.append(
                (row, executor.submit(load_image_rgb, resolve_image_path(images_root, row["file_name"])))
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import h5py

    with h5py.File(out_path, "a") as handle_write:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending: deque = deque()
            iterator = iter(todo)
            submit_more(executor, iterator, pending)
            while pending:
                row, future = pending.popleft()
                image_id = int(row["image_id"])
                if processed_t0 is None:
                    processed_t0 = time.perf_counter()
                try:
                    image_path = resolve_image_path(images_root, row["file_name"])
                    if image_path is None:
                        raise FileNotFoundError(
                            f"image file not found under {images_root}: {row['file_name']}"
                        )
                    image = future.result()
                    result = extract_proposals(image, model, top_n=top_n)
                    boxes = result.boxes
                    objectness = result.objectness
                    meta = result.meta
                    write_bank_entry(
                        handle_write,
                        image_id,
                        boxes,
                        objectness,
                        {"n_raw_post_nms": int(meta.get("n_raw_post_nms", boxes.shape[0]))},
                        overwrite=not resume,
                    )
                except Exception as exc:  # one bad image must never kill the run
                    n_failed += 1
                    failures.append(
                        {"image_id": image_id, "reason": f"{type(exc).__name__}: {exc}"}
                    )
                    if n_failed <= 10:
                        tqdm.write(f"[extract] FAILED      : image {image_id}: {exc}")
                else:
                    n_done += 1
                    n_crops += int(boxes.shape[0])
                    if n_done % PROGRESS_EVERY == 0:
                        handle_write.flush()
                        elapsed = time.perf_counter() - processed_t0
                        rate = n_done / elapsed if elapsed > 0 else 0.0
                        tqdm.write(
                            f"[extract] progress    : {n_done}/{len(todo)} "
                            f"({rate:.2f} img/s) failed={n_failed}"
                        )
                submit_more(executor, iterator, pending)
                if processing_bar is not None:
                    elapsed = time.perf_counter() - processed_t0 if processed_t0 else 0.0
                    postfix: Dict[str, str] = {
                        "imgs/s": f"{n_done / elapsed:.2f}" if elapsed > 0 else "0.00",
                        "crops/s": f"{n_crops / elapsed:.2f}" if elapsed > 0 else "0.00",
                    }
                    if torch_mod is not None and str(args.device).startswith("cuda"):
                        try:
                            peak_mb_now = torch_mod.cuda.max_memory_allocated() / (
                                1024.0 * 1024.0
                            )
                            postfix["peak_vram"] = f"{peak_mb_now:.0f} MB"
                        except Exception:  # VRAM reporting is best-effort only
                            pass
                    processing_bar.set_postfix(postfix, refresh=False)
                    processing_bar.update(1)
            handle_write.flush()
    if processing_bar is not None:
        processing_bar.close()
    process_elapsed = (time.perf_counter() - processed_t0) if processed_t0 else 0.0

    if torch_mod is not None:
        try:
            peak_vram_mb = float(torch_mod.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        except Exception:
            peak_vram_mb = None

    # --- finalise + stats ---------------------------------------------------
    import torchvision

    finalize_bank(
        out_path,
        {
            "model_name": MODEL_NAME,
            "weights": DEFAULT_WEIGHTS,
            "proposal_type": "class-agnostic post-NMS RPN",
            "top_n": top_n,
            "torchvision_version": torchvision.__version__,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "images_root": str(images_root),
        },
    )

    all_ids = [int(i) for i in image_ids(out_path).tolist()]
    k_values: List[int] = []
    n_raw_values: List[int] = []
    with h5py.File(out_path, "r") as handle_read:
        for image_id in all_ids:
            group = handle_read[f"image_{image_id}"]
            k = int(group["boxes"].shape[0])
            k_values.append(k)
            n_raw_values.append(int(group.attrs.get("n_raw_post_nms", k)))

    histogram: Dict[str, int] = {}
    for k in k_values:
        histogram[str(k)] = histogram.get(str(k), 0) + 1

    runtime = time.perf_counter() - started
    summary: Dict[str, Any] = {
        "task": "extract_proposals",
        "config": {
            "images_root": str(images_root),
            "manifest": str(manifest_path),
            "out": str(out_path),
            "top_n": top_n,
            "device": str(args.device),
            "max_images": int(args.max_images),
            "resume": resume,
            "repair": bool(args.repair),
            "prefetch_workers": workers,
            "schema_version": BANK_SCHEMA_VERSION,
        },
        "versions": _version_block(),
        # frozen field names of the multi-agent brief
        "n_images": len(all_ids),
        "n_crops_total": int(sum(k_values)),
        "n_raw_post_nms_summary": _summary_stats(n_raw_values),
        "per_image_K": {**_summary_stats(k_values), "histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0])))},
        "runtime_seconds": float(runtime),
        "images_per_second": float(n_done / process_elapsed) if process_elapsed > 0 else 0.0,
        "crops_per_second": float(n_crops / process_elapsed) if process_elapsed > 0 else 0.0,
        "peak_vram_mb": peak_vram_mb,
        "device_name": device_name,
        "failures": failures,
        # bookkeeping extras (additive, never replacing the frozen names)
        "n_manifest_rows": len(rows),
        "n_duplicate_rows_dropped": duplicates,
        "n_images_this_run": n_done,
        "n_images_skipped_resume": skipped_resume,
        "n_images_failed": n_failed,
        "n_crops_this_run": int(n_crops),
        "process_seconds": float(process_elapsed),
        "notes": [
            "bank-v1: one group image_<id> per image; datasets boxes [K,4] float32 xyxy "
            "(original pixels), objectness [K] float32 descending; group attr n_raw_post_nms.",
            "failures list every image that could not be extracted (missing file / decode / "
            "inference error) - they are counted, never silently dropped.",
            "resume skips images already present; --repair first removes half-written groups.",
        ],
    }
    stats_path = out_path.with_name(f"{out_path.stem}_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(
        f"[extract] images      : bank={summary['n_images']} "
        f"this_run={n_done} skipped_resume={skipped_resume} failed={n_failed}"
    )
    print(
        f"[extract] crops       : total={summary['n_crops_total']} "
        f"(this run={n_crops}, mean K={summary['per_image_K']['mean']})"
    )
    print(
        f"[extract] runtime     : {runtime:.1f}s "
        f"({summary['images_per_second']:.2f} img/s"
        + (f", peak vram {peak_vram_mb:.0f} MB" if peak_vram_mb is not None else "")
        + ")"
    )
    if failures:
        print(
            f"[extract] WARNING     : {len(failures)} image(s) failed "
            f"(first: image {failures[0]['image_id']}: {failures[0]['reason']})"
        )
    print(f"[extract] stats       : {stats_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "mscoco",
        help="COCO image root (flat or with a train2014/ sub-folder)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "data" / "full_image_manifest.csv",
        help="image manifest: image_id,file_name,coco_split,ref_count",
    )
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "cache" / "proposals.h5", help="bank-v1 h5 file"
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="bank size N (frozen: 64)")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--max-images", type=int, default=0, help="0 = every manifest row")
    parser.add_argument(
        "--resume", action="store_true", help="skip images already stored in the h5"
    )
    parser.add_argument(
        "--repair", action="store_true", help="delete half-written groups before processing"
    )
    parser.add_argument(
        "--prefetch-workers", type=int, default=8, help="thread-pool size / max in-flight images"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_images) < 0:
        print("[extract] --max-images must be >= 0")
        return 2
    if int(args.top_n) < 1:
        print("[extract] --top-n must be >= 1")
        return 2
    if int(args.prefetch_workers) < 1:
        print("[extract] --prefetch-workers must be >= 1")
        return 2
    try:
        summary = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[extract] FATAL: {exc}")
        return 2
    if summary["failures"] and summary["n_images"] == 0:
        print("[extract] WARNING: every image failed; the bank is empty")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
