"""Extract the frozen OpenCLIP ViT-B/32 (laion2b_s34b_b79k) feature caches.

One command builds the three caches of the frozen layout (protocol §5/§7):

* ``region_features.h5`` - one crop embedding per proposal, row order ==
  proposal-bank order (frozen exact-crop policy, invalid crops stored as
  zero rows and logged to ``invalid_crops.csv``);
* ``text_features.h5``   - one query embedding per RefCOCO+ sentence
  (``sentence_id`` = global counter over ``ref_id`` asc, ``sent_id`` asc);
* ``global_features.h5`` - one whole-image embedding per bank image.

Every part is crash-resumable (``--resume``): the streaming writers keep a
``.tmp`` file with a monotonically updated id index, so a killed run resumes
without re-encoding finished images - and only ever resumes a *prefix* of the
requested bank slice (anything else raises instead of silently mixing ranges).

The run records the full provenance (metadata.json): backbone fingerprint,
checkpoint sha256, mirror endpoint, crop policy, invalid-crop accounting and
``extraction_stats.json`` (GPU seconds, crops/s, peak VRAM, cache bytes).

Usage
-----
    python scripts/extract_features.py --bank cache/proposals.h5 \
        --images-root data/raw/mscoco --out cache/features \
        --batch-size 128 --precision fp16 --resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.bank import BANK_SCHEMA_VERSION, bank_attrs  # noqa: E402  (pure)
from ccg.features import extract_image, extract_regions, extract_text  # noqa: E402  (no heavy imports)
from ccg.features.cache import (  # noqa: E402
    CACHE_VERSION,
    GLOBAL_FILENAME,
    REGION_FILENAME,
    TEXT_FILENAME,
    StreamingGlobalWriter,
    StreamingRegionWriter,
    merge_extraction_stats,
    read_metadata,
    update_metadata,
    write_json_atomic,
)
from ccg.features.clip_encoder import (  # noqa: E402
    CROP_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
)
from ccg.features.backbone import (  # noqa: E402
    BACKEND_OPEN_CLIP,
    BACKEND_SIGLIP,
    build_encoder,
)
from ccg.utils.logging import utc_now_iso  # noqa: E402

__all__ = ["build_parser", "main"]

STATS_FILENAME = "extraction_stats.json"
INVALID_CROPS_FILENAME = "invalid_crops.csv"
INVALID_CROP_WARN_RATE = extract_regions.INVALID_CROP_WARN_RATE


def _torch():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError:  # pragma: no cover - torch is a declared dependency
        return None


def _start_vram_window(torch) -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _vram_report(torch) -> Tuple[Optional[float], Optional[str]]:
    if torch is None or not torch.cuda.is_available():
        return None, None
    peak = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
    return round(peak, 1), str(torch.cuda.get_device_name(0))


def _region_counts(path: Path) -> Tuple[int, int]:
    """``(n_images, n_crops)`` stored in an existing region file, else ``(0, 0)``."""
    if not path.exists():
        return 0, 0
    import h5py

    with h5py.File(str(path), "r") as handle:
        offsets = handle["image_offsets"][...] if "image_offsets" in handle else None
    if offsets is None:
        return 0, 0
    return int(offsets.shape[0]), int(offsets[:, 2].sum()) if offsets.size else 0


def _global_count(path: Path) -> int:
    if not path.exists():
        return 0
    import h5py

    with h5py.File(str(path), "r") as handle:
        return int(handle["image_ids"].shape[0])


def _text_count(path: Path) -> int:
    if not path.exists():
        return 0
    import h5py

    with h5py.File(str(path), "r") as handle:
        return int(handle["features"].shape[0])


def _cache_bytes(root: Path) -> int:
    total = 0
    for name in (
        REGION_FILENAME,
        TEXT_FILENAME,
        GLOBAL_FILENAME,
        "metadata.json",
        "text_index.csv",
        INVALID_CROPS_FILENAME,
        STATS_FILENAME,
    ):
        path = root / name
        if path.exists():
            total += int(path.stat().st_size)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_features",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bank", type=Path, default=Path("cache/proposals.h5"))
    parser.add_argument("--images-root", type=Path, default=Path("data/raw/mscoco"))
    parser.add_argument("--out", type=Path, default=Path("cache/features"))
    parser.add_argument(
        "--refs",
        type=Path,
        default=Path("data/raw/refcoco+/refcoco+/refs(unc).p"),
        help="UNC referring pickle for the text cache",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="optional data/full_image_manifest.csv for image path resolution",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="FROZEN: ViT-B/32")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="FROZEN: laion2b_s34b_b79k")
    parser.add_argument(
        "--backend",
        default=BACKEND_OPEN_CLIP,
        choices=(BACKEND_OPEN_CLIP, BACKEND_SIGLIP),
        help="encoder backend: open_clip (V1 default) or siglip (transformers)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="local weights (skip download)")
    parser.add_argument(
        "--hf-cache-dir", type=Path, default=Path("cache/hf_hub"), help="huggingface cache dir"
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="override the HF endpoint (default: HF_ENDPOINT env or https://hf-mirror.com)",
    )
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-images", type=int, default=0, help="first N bank images (0 = all)")
    parser.add_argument(
        "--max-sentences", type=int, default=0, help="first N sentences (0 = all)"
    )
    parser.add_argument("--resume", action="store_true", help="skip/continue finished parts")
    parser.add_argument("--flush-every", type=int, default=64, help="writer index flush cadence")
    parser.add_argument("--download-attempts", type=int, default=10)
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-regions", action="store_true")
    parser.add_argument("--skip-global", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = print
    run_started = time.perf_counter()
    run_started_utc = utc_now_iso()

    bank_path = Path(args.bank)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if not bank_path.exists():
        log(f"ERROR: proposal bank {bank_path} does not exist - run scripts/extract_proposals.py first")
        return 2
    attrs = bank_attrs(bank_path)
    if attrs:
        log(
            f"bank: {bank_path} schema={attrs.get('schema_version', '<unknown>')} "
            f"(frozen {BANK_SCHEMA_VERSION}) n_images={attrs.get('n_images', '?')}"
        )
    else:
        log(f"WARNING: {bank_path} has no root attributes (finalize_bank not run yet)")

    target_ids = extract_regions.collect_target_ids(bank_path, args.max_images)
    if not target_ids:
        log("ERROR: the proposal bank holds no images - nothing to extract")
        return 2
    log(
        f"target: {len(target_ids)} images (id {target_ids[0]}..{target_ids[-1]}), "
        f"out={out_root}, device={args.device}, batch={args.batch_size}, precision={args.precision}"
    )
    manifest = extract_regions.load_image_manifest(args.manifest) if args.manifest else None

    torch = _torch()
    _start_vram_window(torch)

    needs_encoder = not (args.skip_text and args.skip_regions and args.skip_global)
    encoder = None
    if needs_encoder:
        encoder = build_encoder(
            device=args.device,
            precision=args.precision,
            cache_dir=args.hf_cache_dir,
            backend=args.backend,
            model_name=args.model_name,
            pretrained=args.pretrained,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            hf_endpoint=args.hf_endpoint,
            download_attempts=args.download_attempts,
        )
        log(
            f"encoder: {args.backend} {encoder.config.library_version}, "
            f"{encoder.config.model_name}/{encoder.config.pretrained} on {encoder.config.device} "
            f"(feature_dim={encoder.feature_dim})"
        )
        log(f"checkpoint: {encoder.config.checkpoint_path}")
        log(f"checkpoint sha256: {encoder.config.checkpoint_sha256}")
        log(
            f"hf endpoint: {encoder.config.hf_endpoint} ({encoder.config.hf_endpoint_source}); "
            f"native logit scale {encoder.logit_scale:.3f}"
        )
        log(f"crop policy: {CROP_POLICY}")

    stats: dict = {
        "device": str(encoder.config.device) if encoder else f"{args.device} (unused)",
        "device_name": _vram_report(torch)[1],
        "batch_size": int(args.batch_size),
        "precision": args.precision,
        "max_images": int(args.max_images),
        "max_sentences": int(args.max_sentences),
        "resumed": bool(args.resume),
        "n_target_images": len(target_ids),
        "started_utc": run_started_utc,
    }
    if encoder is not None:
        stats["backbone_config"] = {
            "backend": args.backend,
            "model_name": encoder.config.model_name,
            "pretrained": encoder.config.pretrained,
            "library_version": encoder.config.library_version,
            "feature_dim": int(encoder.feature_dim),
            "checkpoint_sha256": encoder.config.checkpoint_sha256,
            "hf_endpoint": encoder.config.hf_endpoint,
            "hf_endpoint_source": encoder.config.hf_endpoint_source,
        }

    # ------------------------------------------------------------------ text
    if not args.skip_text:
        if encoder is None:
            log("text phase needs the encoder - skipped")
        elif not Path(args.refs).exists():
            log(f"ERROR: referring pickle {args.refs} not found - cannot build the text cache")
            return 2
        else:
            log(f"[text] corpus from {args.refs}")
            corpus = extract_text.build_text_corpus(Path(args.refs))
            if args.max_sentences and args.max_sentences > 0:
                corpus = corpus[: args.max_sentences]
                log(f"[text] truncated to {len(corpus)} sentences (--max-sentences)")
            text_stats = extract_text.run_text_extraction(
                encoder,
                corpus,
                out_root,
                batch_size=args.batch_size,
                resume=args.resume,
                log=log,
            )
            stats["n_sentences"] = text_stats["n_sentences"]
            stats["text_seconds"] = text_stats["text_seconds"]
            stats["texts_per_second"] = text_stats.get("texts_per_second", 0.0)
            stats["text_skipped_resume"] = bool(text_stats.get("skipped"))

    # --------------------------------------------------------------- regions
    region_writer = None
    invalid_total: Optional[int] = None
    if not args.skip_regions:
        if encoder is None:
            log("region phase needs the encoder - skipped")
        else:
            region_writer = StreamingRegionWriter(
                out_root / REGION_FILENAME, resume=args.resume, flush_every=args.flush_every,
                feature_dim=encoder.feature_dim,
            )
            cached = sorted(region_writer.processed)
            missing = extract_regions.resume_missing_ids(cached, target_ids)
            log(f"[region] cached={len(cached)} to_process={len(missing)}")
            if not missing:
                region_writer.abandon()
                region_writer = None
                log("[region] cache already complete for the requested slice")
            else:
                if region_writer.source == "final":
                    region_writer.adopt_final()
                paths = extract_regions.require_images(args.images_root, missing, manifest=manifest)
                invalid_log = extract_regions.InvalidCropLog(
                    out_root / INVALID_CROPS_FILENAME,
                    resume_keep_ids=set(cached) if args.resume else None,
                )
                invalid_log.start()
                try:
                    region_stats = extract_regions.run_region_extraction(
                        encoder,
                        extract_regions.iter_region_entries(
                            bank_path, args.images_root, only_ids=set(missing), paths=paths
                        ),
                        region_writer,
                        batch_size=args.batch_size,
                        on_invalid=invalid_log.add_image,
                        log=log,
                        total=len(missing),
                    )
                    region_writer.finish()
                except BaseException:
                    region_writer.abandon()
                    invalid_log.close()
                    raise
                invalid_log.close()
                invalid_total = invalid_log.n_rows
                stats["region_crops_encoded_this_run"] = region_stats["region_crops_encoded"]
                stats["region_crops_total_this_run"] = region_stats["n_crops_total"]
                stats["region_images_encoded_this_run"] = region_stats["n_images_encoded"]
                stats["region_extract_seconds"] = region_stats["region_extract_seconds"]
                stats["crops_per_second"] = region_stats["crops_per_second"]
            n_region_images, n_region_crops = (
                _region_counts(out_root / REGION_FILENAME) if region_writer is None else
                (len(region_writer.processed), int(sum(c for _s, c in region_writer.processed.values())))
            )
            stats["region_images"] = n_region_images
            stats["region_crops_encoded"] = n_region_crops
            stats["region_crops_encoded_this_run"] = stats.get("region_crops_encoded_this_run", 0)
            stats["region_crops_total_this_run"] = stats.get("region_crops_total_this_run", 0)
            stats["region_extract_seconds"] = stats.get("region_extract_seconds", 0.0)
            stats["crops_per_second"] = stats.get("crops_per_second", 0.0)

    # ---------------------------------------------------- invalid-crop audit
    invalid_path = out_root / INVALID_CROPS_FILENAME
    if invalid_total is None and invalid_path.exists():
        invalid_total = len(extract_regions.read_invalid_crop_rows(invalid_path))
    if invalid_total is not None:
        stats["invalid_crop_count"] = int(invalid_total)
        n_crops_final = stats.get("region_crops_encoded") or 0
        stats["invalid_crop_rate"] = (
            round(invalid_total / n_crops_final, 6) if n_crops_final else 0.0
        )
        if n_crops_final and stats["invalid_crop_rate"] > INVALID_CROP_WARN_RATE:
            log(
                f"WARNING: invalid crop rate {stats['invalid_crop_rate']:.4%} exceeds "
                f"{INVALID_CROP_WARN_RATE:.2%} ({invalid_total}/{n_crops_final}); "
                f"details in {invalid_path} (run continues)"
            )

    # ---------------------------------------------------------------- global
    if not args.skip_global:
        if encoder is None:
            log("global phase needs the encoder - skipped")
        else:
            global_writer = StreamingGlobalWriter(
                out_root / GLOBAL_FILENAME, resume=args.resume, flush_every=args.flush_every,
                feature_dim=encoder.feature_dim,
            )
            cached = sorted(global_writer.processed)
            missing = extract_regions.resume_missing_ids(cached, target_ids)
            log(f"[global] cached={len(cached)} to_process={len(missing)}")
            if not missing:
                global_writer.abandon()
                global_writer = None
                log("[global] cache already complete for the requested slice")
            else:
                if global_writer.source == "final":
                    global_writer.adopt_final()
                paths = extract_regions.require_images(args.images_root, missing, manifest=manifest)
                try:
                    global_stats = extract_image.run_global_extraction(
                        encoder,
                        extract_image.iter_global_entries(
                            bank_path, args.images_root, only_ids=set(missing), paths=paths
                        ),
                        global_writer,
                        batch_size=args.batch_size,
                        log=log,
                        total=len(missing),
                    )
                    global_writer.finish()
                except BaseException:
                    global_writer.abandon()
                    raise
                stats["global_images_encoded_this_run"] = global_stats["n_images_encoded"]
                stats["global_seconds"] = global_stats["global_seconds"]
                stats["images_per_second"] = global_stats["images_per_second"]
            stats["global_images"] = (
                _global_count(out_root / GLOBAL_FILENAME)
                if global_writer is None
                else len(global_writer.processed)
            )

    # ------------------------------------------------------------- provenance
    peak_vram_mb, device_name = _vram_report(torch)
    stats["peak_vram_mb"] = peak_vram_mb
    if device_name:
        stats["device_name"] = device_name
    stats["wall_seconds"] = round(time.perf_counter() - run_started, 3)
    stats["finished_utc"] = utc_now_iso()
    stats["cache_bytes"] = _cache_bytes(out_root)
    stats.setdefault("n_sentences", _text_count(out_root / TEXT_FILENAME))
    stats.setdefault("global_images", _global_count(out_root / GLOBAL_FILENAME))
    if "region_crops_encoded" not in stats:
        stats["region_crops_encoded"] = _region_counts(out_root / REGION_FILENAME)[1]
    if "region_images" not in stats:
        stats["region_images"] = _region_counts(out_root / REGION_FILENAME)[0]

    if encoder is not None:
        n_images_meta = stats.get("region_images") or stats.get("global_images") or 0
        meta_updates = {
            "backbone": encoder.metadata(),
            "native_logit_scale": encoder.logit_scale,
            "n_images": int(n_images_meta),
            "n_crops": int(stats.get("region_crops_encoded") or 0),
            "n_sentences": int(stats.get("n_sentences") or 0),
            "invalid_crop_count": stats.get("invalid_crop_count"),
            "invalid_crop_rate": stats.get("invalid_crop_rate"),
            "updated_utc": utc_now_iso(),
            "extraction_args": {
                "bank": str(bank_path.resolve()),
                "images_root": str(Path(args.images_root).resolve()),
                "refs": str(Path(args.refs).resolve()),
                "batch_size": int(args.batch_size),
                "precision": args.precision,
                "max_images": int(args.max_images),
                "max_sentences": int(args.max_sentences),
                "resumed": bool(args.resume),
            },
        }
    else:
        meta_updates = {"updated_utc": utc_now_iso()}
    try:
        current = read_metadata(out_root)
    except FileNotFoundError:
        current = {}
    if "created_utc" not in current:
        meta_updates["created_utc"] = run_started_utc
    if "cache_version" not in current:
        meta_updates["cache_version"] = CACHE_VERSION
    update_metadata(out_root, meta_updates)

    write_json_atomic(out_root / STATS_FILENAME, stats)
    merge_extraction_stats(out_root, stats)

    # ------------------------------------------------------------------ report
    log("")
    log("[extract_features] summary")
    for key in (
        "region_images",
        "region_crops_encoded",
        "region_crops_encoded_this_run",
        "invalid_crop_count",
        "invalid_crop_rate",
        "region_extract_seconds",
        "crops_per_second",
        "n_sentences",
        "text_seconds",
        "global_images",
        "global_seconds",
        "peak_vram_mb",
        "cache_bytes",
        "wall_seconds",
    ):
        if key in stats and stats[key] is not None:
            log(f"  {key}: {stats[key]}")
    log(f"  cache root: {out_root.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
