"""Extract the frozen OpenCLIP ViT-B/32 embeddings (checklist step 3).

Three independent caches are produced, all keyed by the same ids the data layer
uses, so a candidate-set rebuild never touches the GPU again (protocol
section 7 - "cheap decision experiments on an expensive, frozen perception
layer"):

* ``text_embeddings``   - one row per ``ref_id`` (query side),
* ``region_embeddings`` - ``[N, 512]`` per ``image_id``, stored next to the
  proposal bank and referenced through ``ProposalBank.feature_ref``,
* ``image_embeddings``  - whole-image embedding, only needed by the ablation
  that asks whether a *global* image feature explains the K effect.

:mod:`ccg.features.clip_encoder` contains the encoder skeleton; the CLI below
refuses to run until the checkpoint is provided locally - it never downloads
weights.

Usage
-----
    python scripts/extract_features.py --mode regions --proposals cache/proposals.h5 \
        --images data/coco --checkpoint /models/clip_ViT-B-32_laion2b.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.proposals import hdf5_available  # noqa: E402  (pure)
from ccg.features.clip_encoder import (  # noqa: E402  (no heavy import at module level)
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    FEATURE_DIM,
)

__all__ = ["PENDING_MESSAGE", "MODES", "build_parser", "main"]

PENDING_MESSAGE = "pending data download — Phase 0 checklist step"
MODES = ("text", "regions", "image")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=MODES, default="regions", help="which cache to build")
    parser.add_argument("--annotations", type=Path, default=None, help="prepared ref records (text)")
    parser.add_argument("--images", type=Path, default=Path("data/coco"), help="COCO image root")
    parser.add_argument("--proposals", type=Path, default=None, help="proposal cache (regions)")
    parser.add_argument("--out", type=Path, default=None, help="output cache file")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="FROZEN: ViT-B/32")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="FROZEN: laion2b_s34b_b79k")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="local weights; required because downloading is forbidden",
    )
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="crops or texts per batch")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "fp32"),
        help="embedding precision (cache dtype stays float16 either way)",
    )
    parser.add_argument(
        "--crop-mode",
        default="bbox",
        choices=("bbox", "context", "square"),
        help="crop recipe for regions (>>> 待真实数据核对: recipe is fixed in the "
        "checklist before any feature is written <<<)",
    )
    parser.add_argument("--context-pad", type=float, default=0.0, help="relative padding for --crop-mode context")
    parser.add_argument("--seed", type=int, default=0, help="determinism seed")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit 0")
    return parser


def default_out(mode: str, backend_hdf5: bool) -> Path:
    """Canonical cache names, one file per mode (never a file per image)."""
    suffix = "h5" if backend_hdf5 else "npz"
    return Path("cache") / f"{mode}_embeddings_{FEATURE_DIM}.{suffix}"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    hdf5 = hdf5_available()
    out = args.out or default_out(args.mode, hdf5)
    print(f"[extract_features] mode       : {args.mode}")
    print(f"[extract_features] backbone   : open_clip {args.model_name} / {args.pretrained}")
    print(f"[extract_features] checkpoint : {args.checkpoint or 'MISSING (downloads forbidden)'}")
    print(f"[extract_features] out        : {out}")
    if args.dry_run:
        print("[extract_features] dry run, nothing computed")
        return 0
    raise NotImplementedError(PENDING_MESSAGE)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
