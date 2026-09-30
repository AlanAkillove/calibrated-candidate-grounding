"""A11 frozen OpenCLIP feature extraction for the RefCOCOg external cohort (A11.5).

The encoder is *never* refit or re-resolved: this script pins the exact frozen
ViT-B/32 (laion2b_s34b_b79k) checkpoint (sha256) **before** encoding a single
crop (protocol A11.5 "a mismatch stops the run"), then reuses the repository's
frozen extraction primitives (``ccg.features.extract_regions`` /
``extract_text`` / ``cache``) to write an independent namespace

    cache/refcocog_external/region_features.h5   (one crop per bank proposal)
    cache/refcocog_external/text_features.h5      (one query per expression, keyed by expr_id)
    cache/refcocog_external/metadata.json

Nothing here trains or calibrates a model; it only produces the CLIP embeddings
the loaded reliability bundle predicts on (b7).  The run records extraction
runtime, throughput, peak VRAM and cache bytes (A11.38).

    conda activate deepminer
    # smoke (validate layout + round-trip, tiny GPU cost):
    python scripts/extract_refcocog_features.py --max-images 5 --limit-expressions 8
    # full external extraction:
    python scripts/extract_refcocog_features.py

``--skip-identity-check`` exists only for tests; production refuses to encode
against an unverified checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.external import frozen_identity as fi  # noqa: E402
from ccg.external import refcocog_external as rx  # noqa: E402
from ccg.features import extract_regions, extract_text  # noqa: E402
from ccg.features.cache import (  # noqa: E402
    CACHE_VERSION,
    REGION_FILENAME,
    TEXT_FILENAME,
    StreamingRegionWriter,
    merge_extraction_stats,
    read_metadata,
    update_metadata,
    write_json_atomic,
)
from ccg.features.clip_encoder import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    build_encoder,
)

DEFAULT_BANK = Path("cache/phase1e_refcocog/proposals_refcocog.h5")
DEFAULT_OUT = Path("results/phase1e_refcocog_external")
STATS_FILENAME = "extraction_stats.json"
INVALID_CROPS_FILENAME = "invalid_crops.csv"


def _load_cohort(sidecar: Path, log) -> Tuple[Any, Dict[int, Tuple[int, int]]]:
    """Read the frozen cohort sidecar if present, else rebuild + persist it."""
    if (Path(sidecar) / "cohort.csv").exists():
        log(f"[extract] cohort sidecar found at {sidecar}")
    cohort, image_sizes, report = rx.load_refcocog_cohort(persist_dir=sidecar, log=log)
    if not (report["a10_pin"]["ok"] and report["matched_pair"]["ok"]):
        raise AssertionError("A11 cohort pin failed - refusing to extract features")
    return cohort, image_sizes


def _text_corpus(
    cohort: Any, limit: int, restrict_ids: Optional[set] = None
) -> List[dict]:
    """One text-query row per expression, keyed by ``expr_id``.

    ``restrict_ids`` (the target-image set actually region-extracted) keeps the
    text slice aligned with the region slice: a smoke run that encodes only the
    first N cohort images must encode only the expressions living in those
    images, or the ``(image_id, expr_id)`` round-trip intersection is empty and
    the layout cannot be verified.  For the full run the restriction is the
    whole cohort, so it is a no-op.
    """
    rows = [
        i
        for i in range(len(cohort))
        if restrict_ids is None or int(cohort.image_id[i]) in restrict_ids
    ]
    if limit and limit > 0:
        rows = rows[:limit]
    corpus = []
    for row in rows:
        text = str(cohort.text[row]).strip()
        if not text:
            raise AssertionError(f"expr {int(cohort.expr_id[row])}: empty expression text")
        corpus.append(
            {
                "sentence_id": int(cohort.expr_id[row]),  # the A11 row key
                "ref_id": int(cohort.ref_id[row]),
                "sent_id": 0,
                "image_id": int(cohort.image_id[row]),
                "split": "test",
                "text": text,
            }
        )
    return corpus


def _torch():
    import torch

    return torch


def run_extraction(args: argparse.Namespace, log) -> Dict[str, Any]:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cohort, image_sizes = _load_cohort(args.cohort_dir, log)
    all_image_ids = sorted({int(i) for i in cohort.image_id})
    if args.max_images and args.max_images > 0:
        target_ids = all_image_ids[: int(args.max_images)]
        log(f"[extract] SMOKE: first {len(target_ids)} of {len(all_image_ids)} cohort images")
    else:
        target_ids = all_image_ids
    target_set = set(target_ids)

    # ---- 1: pin the frozen encoder identity BEFORE any crop is encoded ------
    identity: Dict[str, Any] = {}
    if not args.skip_identity_check:
        identity = fi.assert_openclip_identity(args.features_root)
        checkpoint = Path(identity["checkpoint_path"])
        log(f"[extract] OpenCLIP identity verified: {identity['frozen']['model_name']}/"
            f"{identity['frozen']['pretrained']} sha256 {identity['measured_checkpoint_sha256'][:16]}...")
    else:
        checkpoint = args.checkpoint
        log("[extract] WARNING: identity check skipped (tests only)")

    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    encoder = build_encoder(
        device=args.device,
        precision=args.precision,
        cache_dir=args.hf_cache_dir,
        model_name=DEFAULT_MODEL_NAME,
        pretrained=DEFAULT_PRETRAINED,
        checkpoint_path=checkpoint,
        batch_size=args.batch_size,
    )
    measured = getattr(encoder.config, "checkpoint_sha256", None)
    if not args.skip_identity_check and measured != fi.FROZEN_OPENCLIP["checkpoint_sha256"]:
        raise AssertionError(
            f"built encoder checkpoint sha256 {measured} != frozen "
            f"{fi.FROZEN_OPENCLIP['checkpoint_sha256']} - refusing to extract"
        )
    log(f"[extract] encoder on {encoder.config.device}; sha256 {str(measured)[:16]}...")

    stats: Dict[str, Any] = {
        "device": str(encoder.config.device),
        "precision": args.precision,
        "batch_size": int(args.batch_size),
        "n_target_images": len(target_ids),
        "max_images": int(args.max_images or 0),
        "resumed": bool(args.resume),
        "checkpoint_sha256": measured,
    }

    # ---- 2: region crops (bank proposals of the cohort images) -------------
    bank = Path(args.bank)
    started = time.perf_counter()
    writer = StreamingRegionWriter(out_root / REGION_FILENAME, resume=args.resume,
                                   feature_dim=encoder.feature_dim)
    cached = sorted(writer.processed)
    missing = extract_regions.resume_missing_ids(cached, target_ids)
    log(f"[region] cached={len(cached)} to_process={len(missing)}")
    if not missing:
        writer.abandon()
        log("[region] cache already complete for the requested slice")
    else:
        if writer.source == "final":
            writer.adopt_final()
        paths = extract_regions.require_images(args.images_root, missing)
        invalid_log = extract_regions.InvalidCropLog(
            out_root / INVALID_CROPS_FILENAME,
            resume_keep_ids=set(cached) if args.resume else None,
        )
        invalid_log.start()
        try:
            region_stats = extract_regions.run_region_extraction(
                encoder,
                extract_regions.iter_region_entries(bank, args.images_root, only_ids=set(missing), paths=paths),
                writer,
                batch_size=args.batch_size,
                on_invalid=invalid_log.add_image,
                log=log,
                total=len(missing),
            )
            writer.finish()
        except BaseException:
            writer.abandon()
            invalid_log.close()
            raise
        invalid_log.close()
        stats["region_crops_encoded"] = region_stats["n_crops_total"]
        stats["region_images_encoded"] = region_stats["n_images_encoded"]
        stats["region_seconds"] = round(time.perf_counter() - started, 2)
        stats["crops_per_second"] = region_stats["crops_per_second"]
        stats["invalid_crop_count"] = invalid_log.n_rows
    if "region_crops_encoded" not in stats:
        stats["region_crops_encoded"] = sum(int(c) for _s, c in writer.processed.values()) if writer else 0
        stats["region_images_encoded"] = len(writer.processed) if writer else len(cached)

    # ---- 3: text queries (one per expression, keyed by expr_id) ------------
    corpus = _text_corpus(cohort, args.limit_expressions, restrict_ids=target_set)
    text_started = time.perf_counter()
    text_stats = extract_text.run_text_extraction(
        encoder, corpus, out_root, batch_size=args.batch_size, resume=args.resume, log=log
    )
    stats["n_sentences"] = text_stats["n_sentences"]
    stats["text_seconds"] = round(time.perf_counter() - text_started, 2)

    # ---- 4: provenance metadata (FeatureCache.open requires metadata.json) --
    n_images = stats.get("region_images_encoded") or len(target_ids)
    n_crops = stats.get("region_crops_encoded") or 0
    meta_updates = {
        "backbone": encoder.metadata(),
        "native_logit_scale": encoder.logit_scale,
        "n_images": int(n_images),
        "n_crops": int(n_crops),
        "n_sentences": int(stats.get("n_sentences") or 0),
        "dataset": "refcocog_external",
        "cohort_rows": len(corpus),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        current = read_metadata(out_root)
    except FileNotFoundError:
        current = {}
    if "created_utc" not in current:
        meta_updates["created_utc"] = meta_updates["updated_utc"]
    if "cache_version" not in current:
        meta_updates["cache_version"] = CACHE_VERSION
    update_metadata(out_root, meta_updates)

    if torch.cuda.is_available():
        stats["peak_vram_mb"] = round(float(torch.cuda.max_memory_allocated()) / (1024 ** 2), 1)
        stats["device_name"] = torch.cuda.get_device_name(0)
    stats["cache_bytes"] = _cache_bytes(out_root)
    stats["wall_seconds"] = round(time.perf_counter() - started, 2)
    write_json_atomic(out_root / STATS_FILENAME, stats)
    merge_extraction_stats(out_root, stats)

    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, stats)
    log("[extract] summary: " + ", ".join(f"{k}={v}" for k, v in stats.items() if v is not None))
    return stats


def _cache_bytes(root: Path) -> int:
    total = 0
    for name in (REGION_FILENAME, TEXT_FILENAME, "metadata.json", "text_index.csv", INVALID_CROPS_FILENAME):
        path = root / name
        if path.exists():
            total += int(path.stat().st_size)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--images-root", type=Path, default=rx.IMAGE_ROOT)
    parser.add_argument("--out", type=Path, default=rx.EXTERNAL_CACHE_ROOT)
    parser.add_argument("--cohort-dir", type=Path, default=rx.COHORT_ARTIFACTS)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_OUT / "feature_extraction_stats.json")
    parser.add_argument("--features-root", type=Path, default=fi.DEFAULT_FEATURES_ROOT,
                        help="the RefCOCO+ cache whose metadata pins the frozen identity")
    parser.add_argument("--checkpoint", type=Path, default=None, help="override (identity-skip only)")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("cache/hf_hub"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=0, help="smoke: first N cohort images (0 = all)")
    parser.add_argument("--limit-expressions", type=int, default=0, help="smoke: first N expressions (0 = all)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-identity-check", action="store_true", help="tests only")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = lambda message: print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)  # noqa: E731
    run_extraction(args, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
