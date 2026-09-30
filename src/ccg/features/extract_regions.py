"""Region-crop feature extraction for the frozen proposal bank (protocol §6/§7).

Every proposal of every image in ``cache/proposals.h5`` is cropped with the
**frozen crop policy** - exact proposal crop: xyxy float32 -> ``np.rint`` ->
clamp to the image; a crop thinner than 1 px in either axis is *invalid* and
stored as an all-zero row (never silently expanded, never silently dropped).
Valid crops go through the open_clip val transform of the frozen ViT-B/32.

Crops are accumulated across images into GPU batches (``--batch-size``), while
the row order per image stays exactly the bank order and images are appended
to the cache in ascending ``image_id`` order - the resume semantics of
:class:`~ccg.features.cache.StreamingRegionWriter` depend on it.

``--resume`` continues a crashed/tmp or partial final cache; the already
cached ids must form a *prefix* of the requested bank slice (anything else
raises instead of silently mixing ranges).  Invalid crops are logged to
``<out>/invalid_crops.csv`` per completed image, so the row set always mirrors
the images actually committed to the cache.

CLI::

    python -m ccg.features.extract_regions --bank cache/proposals.h5 \
        --images-root data/raw/mscoco --out cache/features \
        --batch-size 128 --precision fp16 --resume
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from ..data.bank import BANK_SCHEMA_VERSION, bank_attrs, image_ids as bank_image_ids, iter_bank
from ..utils.logging import utc_now_iso
from .cache import (
    REGION_FILENAME,
    StreamingRegionWriter,
    merge_extraction_stats,
    update_metadata,
)
from .clip_encoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    ClipEncoder,
    build_encoder,
    crop_image_at_boxes,
    load_rgb_image,
)

__all__ = [
    "INVALID_CROP_COLUMNS",
    "InvalidCropLog",
    "read_invalid_crop_rows",
    "load_image_manifest",
    "resolve_image_path",
    "require_images",
    "collect_target_ids",
    "resume_missing_ids",
    "iter_region_entries",
    "run_region_extraction",
    "build_parser",
    "main",
]

#: Frozen column order of ``invalid_crops.csv`` (box corners are the rounded,
#: clamped integer xyxy actually used by the crop policy).
INVALID_CROP_COLUMNS: Tuple[str, ...] = (
    "image_id",
    "proposal_index",
    "x1",
    "y1",
    "x2",
    "y2",
)

#: Warn when the invalid-crop rate exceeds this (still never aborts).
INVALID_CROP_WARN_RATE = 1e-3

_MANIFEST_RELATIVE_KEYS = ("file_name", "coco_split")


# ---------------------------------------------------------------------------
# image location
# ---------------------------------------------------------------------------
def load_image_manifest(path: str | Path) -> Dict[int, Tuple[str, str]]:
    """``data/full_image_manifest.csv`` -> ``{image_id: (relative_path, split)}``.

    Columns: ``image_id,file_name,coco_split,ref_count`` (written by
    ``tools/download_all_images.py``).  Only the columns needed to build the
    ``<split>/<file_name>`` relative path are required; a malformed row raises.
    """
    path = Path(path)
    out: Dict[int, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [key for key in ("image_id", *_MANIFEST_RELATIVE_KEYS) if key not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: manifest misses columns {missing}")
        for row in reader:
            image_id = int(row["image_id"])
            relative = f"{row['coco_split']}/{row['file_name']}"
            out[image_id] = (relative, row["coco_split"])
    return out


def resolve_image_path(
    images_root: str | Path,
    image_id: int,
    manifest: Optional[Dict[int, Tuple[str, str]]] = None,
) -> Path:
    """Locate one COCO image on disk (manifest first, then standard layout)."""
    image_id = int(image_id)
    root = Path(images_root)
    candidates: List[Path] = []
    if manifest is not None and image_id in manifest:
        relative, split = manifest[image_id]
        candidates.append(root / relative)
        candidates.append(root / split / f"COCO_{split}_{image_id:012d}.jpg")
    candidates.extend(
        [
            root / "train2014" / f"COCO_train2014_{image_id:012d}.jpg",
            root / "val2014" / f"COCO_val2014_{image_id:012d}.jpg",
            root / f"COCO_train2014_{image_id:012d}.jpg",
            root / f"COCO_val2014_{image_id:012d}.jpg",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"COCO image {image_id} not found under {root} (tried "
        + ", ".join(str(c) for c in candidates)
        + "); run tools/download_all_images.py or point --images-root at the image folder"
    )


def require_images(
    images_root: str | Path,
    image_ids: Iterable[int],
    manifest: Optional[Dict[int, Tuple[str, str]]] = None,
) -> Dict[int, Path]:
    """Resolve **every** id up front - a missing image aborts before GPU work."""
    wanted = [int(i) for i in image_ids]
    paths: Dict[int, Path] = {}
    missing: List[int] = []
    for image_id in wanted:
        try:
            paths[image_id] = resolve_image_path(images_root, image_id, manifest=manifest)
        except FileNotFoundError:
            missing.append(image_id)
    if missing:
        shown = ", ".join(str(i) for i in missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} of {len(wanted)} COCO images are not on disk under {images_root}: "
            f"{shown}{more}. Nothing was extracted; download the images first "
            "(tools/download_all_images.py) or limit the run with --max-images."
        )
    return paths


# ---------------------------------------------------------------------------
# resume / entry planning
# ---------------------------------------------------------------------------
def collect_target_ids(bank_path: str | Path, max_images: int = 0) -> List[int]:
    """Ascending bank image ids, truncated to the first ``max_images`` (0 = all)."""
    ids = [int(i) for i in bank_image_ids(bank_path)]
    if max_images and max_images > 0:
        ids = ids[: int(max_images)]
    return ids


def resume_missing_ids(cached_ids: Sequence[int], expected_ids: Sequence[int]) -> List[int]:
    """The ids still to process when ``cached_ids`` must be a prefix of ``expected_ids``.

    Silent range mixing is forbidden: a cached set that is not exactly the
    prefix raises with the first diverging position.
    """
    cached = [int(i) for i in cached_ids]
    expected = [int(i) for i in expected_ids]
    if len(cached) > len(expected) or cached != expected[: len(cached)]:
        for position, cached_id in enumerate(cached):
            if position >= len(expected) or expected[position] != cached_id:
                raise RuntimeError(
                    f"cannot resume: cached image ids are not a prefix of the requested "
                    f"bank slice (cached {len(cached)} ids, expected {len(expected)}; "
                    f"first mismatch at position {position}: cache has {cached_id}, "
                    f"requested {expected[position] if position < len(expected) else '<end>'}). "
                    "Re-run without --resume to rebuild the cache, or widen --max-images."
                )
        raise RuntimeError(
            f"cannot resume: the cache holds {len(cached)} images but the requested bank "
            f"slice has only {len(expected)}; refusing to silently drop cached data."
        )
    return expected[len(cached) :]


def iter_region_entries(
    bank_path: str | Path,
    images_root: str | Path,
    *,
    only_ids: Optional[Iterable[int]] = None,
    paths: Optional[Dict[int, Path]] = None,
    manifest: Optional[Dict[int, Tuple[str, str]]] = None,
) -> Iterator[Tuple[int, np.ndarray, Path]]:
    """Stream ``(image_id, boxes, image_path)`` from the bank in ascending id order."""
    wanted = None if only_ids is None else {int(i) for i in only_ids}
    for image_id, boxes, _objectness in iter_bank(bank_path):
        image_id = int(image_id)
        if wanted is not None and image_id not in wanted:
            continue
        if paths is not None and image_id in paths:
            path = Path(paths[image_id])
        else:
            path = resolve_image_path(images_root, image_id, manifest=manifest)
        yield image_id, boxes, path


# ---------------------------------------------------------------------------
# invalid-crop log (mirrors exactly the images committed to the cache)
# ---------------------------------------------------------------------------
def read_invalid_crop_rows(path: str | Path) -> List[Tuple[int, int, int, int, int, int]]:
    """Read ``invalid_crops.csv`` back as integer tuples (validated header)."""
    path = Path(path)
    rows: List[Tuple[int, int, int, int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration:
            return rows
        if header != INVALID_CROP_COLUMNS:
            raise ValueError(f"{path}: columns {header} != frozen {list(INVALID_CROP_COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            if len(row) != len(INVALID_CROP_COLUMNS):
                raise ValueError(f"{path}:{line}: {len(row)} fields, expected 6")
            rows.append(tuple(int(value) for value in row))  # type: ignore[arg-type]
    return rows


class InvalidCropLog:
    """Append-only ``invalid_crops.csv``; flushed per completed image.

    ``resume_keep_ids`` (the ids already committed to the cache) filters the
    rows of a previous run so the log can never contain rows for images the
    cache does not have.  Rows are written only via :meth:`add_image`, which
    the extraction calls when an image is fully appended to the cache.
    """

    def __init__(self, path: str | Path, *, resume_keep_ids: Optional[Iterable[int]] = None) -> None:
        self.path = Path(path)
        self._kept: List[Tuple[int, int, int, int, int, int]] = []
        if resume_keep_ids is not None and self.path.exists():
            keep = {int(i) for i in resume_keep_ids}
            self._kept = [row for row in read_invalid_crop_rows(self.path) if row[0] in keep]
        self.n_rows = len(self._kept)
        self._handle = None
        self._writer = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(INVALID_CROP_COLUMNS)
            for row in self._kept:
                writer.writerow(row)
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)

    def add_image(self, image_id: int, rows: Sequence[Tuple[int, int, int, int, int]]) -> None:
        if self._writer is None:
            raise RuntimeError("InvalidCropLog.start() must be called first")
        for proposal_index, x1, y1, x2, y2 in rows:
            self._writer.writerow(
                (int(image_id), int(proposal_index), int(x1), int(y1), int(x2), int(y2))
            )
        self.n_rows += len(rows)
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def __enter__(self) -> "InvalidCropLog":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# extraction core (shared by this CLI and scripts/extract_features.py)
# ---------------------------------------------------------------------------
def run_region_extraction(
    encoder: ClipEncoder,
    entries: Iterable[Tuple[int, np.ndarray, Path]],
    writer: StreamingRegionWriter,
    *,
    batch_size: int,
    on_invalid: Optional[Callable[[int, Sequence[Tuple[int, int, int, int, int]]], None]] = None,
    log: Callable[[str], None] = print,
    progress_every: int = 250,
    total: Optional[int] = None,
    progress: bool = True,
) -> dict:
    """Encode every proposal crop of ``entries`` into ``writer`` (ascending ids).

    Crops are batched across images up to ``batch_size``; an image's rows are
    appended only once all its proposals are assigned, so the cache row order
    per image equals the bank order and the append order stays ascending.
    ``on_invalid(image_id, rows)`` fires right before an image is appended with
    ``rows = [(proposal_index, x1, y1, x2, y2), ...]`` of its invalid crops.

    ``total`` is the expected number of entries: when given (and ``progress``
    is true) the loop runs under a ``region`` progress bar; milestone lines go
    through :func:`tqdm.write` so they never interleave with the bar.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    pending_crops: List = []
    pending_meta: List[Tuple[int, int]] = []
    buffers: Dict[int, np.ndarray] = {}
    invalid_stash: Dict[int, List[Tuple[int, int, int, int, int]]] = {}
    remaining: Dict[int, int] = {}
    order: List[int] = []
    next_flush = 0
    n_images = 0
    n_crops_total = 0
    n_crops_encoded = 0
    n_invalid = 0
    started = time.perf_counter()
    bar = None
    if progress and total is not None:
        bar = tqdm(entries, total=total, desc="region", unit="img", disable=None)
        entries = bar

    def flush_batches(*, force: bool) -> None:
        nonlocal n_crops_encoded
        while len(pending_crops) >= batch_size or (force and pending_crops):
            take = min(batch_size, len(pending_crops))
            features = encoder.encode_images(pending_crops[: take], batch_size=batch_size)
            if features.shape != (take, encoder.feature_dim):
                raise RuntimeError(
                    f"encoder returned {features.shape} for {take} crops; "
                    f"expected {(take, encoder.feature_dim)}"
                )
            for row in range(take):
                image_id, box_index = pending_meta[row]
                buffers[image_id][box_index] = features[row]
                remaining[image_id] -= 1
            n_crops_encoded += take
            del pending_crops[:take]
            del pending_meta[:take]
        if bar is not None:
            elapsed = time.perf_counter() - started
            bar.set_postfix(
                crops_per_s=f"{n_crops_encoded / max(elapsed, 1e-9):.0f}",
                invalid=n_invalid,
                refresh=False,
            )

    def drain_ready() -> None:
        nonlocal n_images, next_flush
        while next_flush < len(order):
            image_id = order[next_flush]
            if remaining[image_id] != 0:
                break
            stash = invalid_stash.pop(image_id, [])
            if stash and on_invalid is not None:
                on_invalid(image_id, stash)
            writer.append(image_id, buffers.pop(image_id))
            remaining.pop(image_id)
            n_images += 1
            next_flush += 1
            if progress_every and n_images % progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = n_crops_encoded / max(elapsed, 1e-9)
                tqdm.write(
                    f"  {n_images} images appended ({n_crops_encoded} crops encoded, "
                    f"{n_invalid} invalid, {rate:.0f} crops/s)"
                )
                if bar is not None:
                    bar.set_postfix(crops_per_s=f"{rate:.0f}", invalid=n_invalid)

    try:
        for image_id, boxes, path in entries:
            image_id = int(image_id)
            if image_id in buffers:
                raise ValueError(f"image_id {image_id} appears twice in the entry stream")
            image = load_rgb_image(path)
            crops, valid, int_boxes = crop_image_at_boxes(image, boxes)
            k = int(int_boxes.shape[0])
            buffers[image_id] = np.zeros((k, encoder.feature_dim), dtype=np.float16)
            invalid_rows = [
                (int(i), *[int(v) for v in int_boxes[i]]) for i in np.flatnonzero(~valid)
            ]
            n_invalid += len(invalid_rows)
            n_crops_total += k
            order.append(image_id)
            remaining[image_id] = k - len(invalid_rows)
            if invalid_rows:
                invalid_stash[image_id] = invalid_rows
            if len(crops):
                pending_crops.extend(crops)
                pending_meta.extend((image_id, int(i)) for i in np.flatnonzero(valid))
            flush_batches(force=False)
            drain_ready()

        flush_batches(force=True)
        drain_ready()
        if next_flush != len(order) or buffers:
            raise RuntimeError(
                f"internal error: {len(order) - next_flush} image(s) were never appended "
                f"({len(buffers)} buffers open)"
            )
    finally:
        if bar is not None:
            bar.close()
    seconds = time.perf_counter() - started
    stats = {
        "n_images_encoded": n_images,
        "n_crops_total": n_crops_total,
        "region_crops_encoded": n_crops_encoded,
        "invalid_crop_count": n_invalid,
        "invalid_crop_rate": round(n_invalid / n_crops_total, 6) if n_crops_total else 0.0,
        "region_extract_seconds": round(seconds, 3),
        "crops_per_second": round(n_crops_encoded / seconds, 1) if seconds > 0 else 0.0,
        "region_encoded_utc": utc_now_iso(),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_regions",
        description="Extract frozen OpenCLIP crop embeddings for every proposal (bank order).",
    )
    parser.add_argument("--bank", type=Path, default=Path("cache/proposals.h5"))
    parser.add_argument("--images-root", type=Path, default=Path("data/raw/mscoco"))
    parser.add_argument("--out", type=Path, default=Path("cache/features"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional data/full_image_manifest.csv for image path resolution",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--checkpoint", type=Path, default=None, help="local weights (skip download)")
    parser.add_argument(
        "--hf-cache-dir", type=Path, default=Path("cache/hf_hub"), help="huggingface cache"
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-images", type=int, default=0, help="first N bank images (0 = all)")
    parser.add_argument("--resume", action="store_true", help="continue an existing cache")
    parser.add_argument("--flush-every", type=int, default=64, help="writer index flush cadence")
    parser.add_argument("--download-attempts", type=int, default=10)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = print
    bank_path = Path(args.bank)
    if not bank_path.exists():
        log(f"proposal bank {bank_path} does not exist - run scripts/extract_proposals.py first")
        return 2
    attrs = bank_attrs(bank_path)
    if not attrs:
        log(f"WARNING: {bank_path} has no root attributes (finalize_bank not run yet)")
    else:
        log(f"bank schema: {attrs.get('schema_version', '<unknown>')} (frozen: {BANK_SCHEMA_VERSION})")

    out_root = Path(args.out)
    target_ids = collect_target_ids(bank_path, args.max_images)
    log(f"target images: {len(target_ids)} (max-images={args.max_images or 'all'})")
    manifest = load_image_manifest(args.manifest) if args.manifest else None

    writer = StreamingRegionWriter(out_root / REGION_FILENAME, resume=args.resume)
    cached = sorted(writer.processed)
    missing = resume_missing_ids(cached, target_ids)
    log(f"cached: {len(cached)} images; to process: {len(missing)}")
    if not missing:
        writer.abandon()
        log("region cache already complete for the requested slice - nothing to do")
        return 0
    if writer.source == "final":
        writer.adopt_final()

    paths = require_images(args.images_root, missing, manifest=manifest)
    encoder = build_encoder(
        device=args.device,
        precision=args.precision,
        cache_dir=args.hf_cache_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        download_attempts=args.download_attempts,
    )

    invalid_log = InvalidCropLog(
        out_root / "invalid_crops.csv",
        resume_keep_ids=set(cached) if args.resume else None,
    )
    invalid_log.start()
    try:
        stats = run_region_extraction(
            encoder,
            iter_region_entries(
                bank_path, args.images_root, only_ids=set(missing), paths=paths
            ),
            writer,
            batch_size=args.batch_size,
            on_invalid=invalid_log.add_image,
            log=log,
        )
        writer.finish()
    except BaseException:
        writer.abandon()
        invalid_log.close()
        raise
    invalid_log.close()

    n_crops = int(sum(count for _start, count in writer.processed.values()))
    rate = invalid_log.n_rows / n_crops if n_crops else 0.0
    if rate > INVALID_CROP_WARN_RATE:
        log(
            f"WARNING: invalid crop rate {rate:.4%} exceeds {INVALID_CROP_WARN_RATE:.2%} - "
            f"check {out_root / 'invalid_crops.csv'}"
        )
    merge_extraction_stats(out_root, {"region": stats})
    update_metadata(
        out_root,
        {
            "backbone": encoder.metadata(),
            "native_logit_scale": encoder.logit_scale,
            "n_images": len(writer.processed),
            "n_crops": n_crops,
            "invalid_crop_count": invalid_log.n_rows,
            "invalid_crop_rate": round(rate, 6),
        },
    )
    log(
        f"region extraction done: {stats['n_images_encoded']} images, "
        f"{stats['region_crops_encoded']} crops in {stats['region_extract_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
