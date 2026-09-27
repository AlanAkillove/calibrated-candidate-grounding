"""CLI stub: extract OpenCLIP region-crop features for every proposal (N~64/image).

Writes the ``proposal_features [N, 512]`` part of the cache layout (protocol
section 7) next to the proposal boxes created by ``scripts/extract_proposals.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clip_encoder import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, FEATURE_DIM

TODO_MESSAGE = (
    "TODO: region-crop feature extraction is not enabled yet - it needs the proposal bank "
    "cache, the COCO images and a GPU pass (Phase 0 checklist: extract_features)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_regions",
        description="Extract frozen OpenCLIP crop embeddings for all proposals.",
    )
    parser.add_argument("--proposals", type=Path, required=True, help="proposal bank cache (.h5/.npz)")
    parser.add_argument("--images", type=Path, required=True, help="root of the COCO image folder")
    parser.add_argument(
        "--out", type=Path, default=Path("cache/proposal_features.h5"), help="output feature cache"
    )
    parser.add_argument(
        "--crop-mode",
        default="context",
        choices=["plain", "context"],
        help="'context' keeps surrounding pixels (CLIP is spatially weak); crop-context loss is "
        "a known risk recorded in the protocol",
    )
    parser.add_argument("--context-pad", type=float, default=0.25, help="relative box expansion")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--embed-dim", type=int, default=FEATURE_DIM)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(TODO_MESSAGE)
    print(f"parsed arguments: {vars(args)}")
    raise NotImplementedError("Phase 0 pipeline: to be enabled after data download")


if __name__ == "__main__":
    sys.exit(main())
