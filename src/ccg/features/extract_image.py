"""Global whole-image feature extraction with the frozen OpenCLIP encoder.

One L2-normalised 512-d embedding per bank image (the open_clip val transform
of the whole image, no crop), appended to ``global_features.h5`` in ascending
``image_id`` order through the crash-resumable
:class:`~ccg.features.cache.StreamingGlobalWriter`.

CLI::

    python -m ccg.features.extract_image --bank cache/proposals.h5 \
        --images-root data/raw/mscoco --out cache/features --batch-size 128 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from ..data.bank import BANK_SCHEMA_VERSION, bank_attrs, iter_bank
from ..utils.logging import utc_now_iso
from .cache import (
    CACHE_VERSION,
    GLOBAL_FILENAME,
    StreamingGlobalWriter,
    merge_extraction_stats,
    read_metadata,
    update_metadata,
)
from .clip_encoder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    ClipEncoder,
    build_encoder,
    load_rgb_image,
)
from .extract_regions import (
    collect_target_ids,
    load_image_manifest,
    require_images,
    resolve_image_path,
    resume_missing_ids,
)

__all__ = [
    "iter_global_entries",
    "run_global_extraction",
    "build_parser",
    "main",
]


def iter_global_entries(
    bank_path: str | Path,
    images_root: str | Path,
    *,
    only_ids: Optional[Iterable[int]] = None,
    paths: Optional[Dict[int, Path]] = None,
    manifest: Optional[Dict[int, Tuple[str, str]]] = None,
) -> Iterator[Tuple[int, Path]]:
    """Stream ``(image_id, image_path)`` from the bank in ascending id order."""
    wanted = None if only_ids is None else {int(i) for i in only_ids}
    for image_id, _boxes, _objectness in iter_bank(bank_path):
        image_id = int(image_id)
        if wanted is not None and image_id not in wanted:
            continue
        if paths is not None and image_id in paths:
            path = Path(paths[image_id])
        else:
            path = resolve_image_path(images_root, image_id, manifest=manifest)
        yield image_id, path


def run_global_extraction(
    encoder: ClipEncoder,
    entries: Iterable[Tuple[int, Path]],
    writer: StreamingGlobalWriter,
    *,
    batch_size: int,
    log: Callable[[str], None] = print,
    progress_every: int = 250,
    total: Optional[int] = None,
    progress: bool = True,
) -> dict:
    """Encode whole images of ``entries`` into ``writer`` (ascending ids).

    ``total`` is the expected number of entries: when given (and ``progress``
    is true) the loop runs under a ``global`` progress bar; milestone lines go
    through :func:`tqdm.write` so they never interleave with the bar.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    started = time.perf_counter()
    n_images = 0
    batch_ids: List[int] = []
    batch_images: List = []
    bar = None
    if progress and total is not None:
        bar = tqdm(entries, total=total, desc="global", unit="img", disable=None)
        entries = bar

    def flush() -> None:
        nonlocal n_images
        if not batch_images:
            return
        features = encoder.encode_images(batch_images, batch_size=batch_size)
        if features.shape != (len(batch_images), encoder.feature_dim):
            raise RuntimeError(
                f"encoder returned {features.shape} for {len(batch_images)} images; "
                f"expected {(len(batch_images), encoder.feature_dim)}"
            )
        for row, image_id in enumerate(batch_ids):
            writer.append(image_id, features[row])
            n_images += 1
            if progress_every and n_images % progress_every == 0:
                elapsed = time.perf_counter() - started
                rate = n_images / max(elapsed, 1e-9)
                tqdm.write(f"  {n_images} global features appended ({rate:.1f}/s)")
                if bar is not None:
                    bar.set_postfix(imgs_per_s=f"{rate:.1f}")
        if bar is not None:
            elapsed = time.perf_counter() - started
            bar.set_postfix(imgs_per_s=f"{n_images / max(elapsed, 1e-9):.1f}", refresh=False)
        batch_ids.clear()
        batch_images.clear()

    try:
        for image_id, path in entries:
            batch_ids.append(int(image_id))
            batch_images.append(load_rgb_image(path))
            if len(batch_images) >= batch_size:
                flush()
        flush()
    finally:
        if bar is not None:
            bar.close()

    seconds = time.perf_counter() - started
    stats = {
        "n_images_encoded": n_images,
        "global_seconds": round(seconds, 3),
        "images_per_second": round(n_images / seconds, 1) if seconds > 0 else 0.0,
        "global_encoded_utc": utc_now_iso(),
    }
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_image",
        description="Extract frozen OpenCLIP global image features into the feature cache.",
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

    writer = StreamingGlobalWriter(out_root / GLOBAL_FILENAME, resume=args.resume)
    cached = sorted(writer.processed)
    missing = resume_missing_ids(cached, target_ids)
    log(f"cached: {len(cached)} images; to process: {len(missing)}")
    if not missing:
        writer.abandon()
        log("global cache already complete for the requested slice - nothing to do")
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
    try:
        stats = run_global_extraction(
            encoder,
            iter_global_entries(bank_path, args.images_root, only_ids=set(missing), paths=paths),
            writer,
            batch_size=args.batch_size,
            log=log,
        )
        writer.finish()
    except BaseException:
        writer.abandon()
        raise

    updates = {
        "cache_version": CACHE_VERSION,
        "backbone": encoder.metadata(),
        "native_logit_scale": encoder.logit_scale,
        "n_global_images": len(writer.processed),
    }
    try:
        existing_n_images = read_metadata(out_root).get("n_images")
    except FileNotFoundError:
        existing_n_images = None
    if not existing_n_images:
        updates["n_images"] = len(writer.processed)
    merge_extraction_stats(out_root, {"global": stats})
    update_metadata(out_root, updates)
    log(
        f"global extraction done: {stats['n_images_encoded']} images in "
        f"{stats['global_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
