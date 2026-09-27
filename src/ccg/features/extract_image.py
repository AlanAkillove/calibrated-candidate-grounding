"""CLI stub: extract global image features with the frozen OpenCLIP encoder.

Phase 0 checklist step - enabled only after the COCO images and the open_clip
checkpoint are available locally.  The module is importable and ``--help``
works without torch / open_clip / GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clip_encoder import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, FEATURE_DIM

TODO_MESSAGE = (
    "TODO: global image feature extraction is not enabled yet - it needs the COCO images, "
    "the local open_clip checkpoint and a GPU pass (Phase 0 checklist: extract_features)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_image",
        description="Extract frozen OpenCLIP global image features into the feature cache.",
    )
    parser.add_argument("--images", type=Path, required=True, help="root of the COCO image folder")
    parser.add_argument(
        "--image-ids",
        type=Path,
        default=None,
        help="optional file with the image ids to process (default: all used by the split)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cache/image_features.h5"),
        help="output cache path (.h5 preferred, .npz supported)",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--checkpoint", type=Path, default=None, help="local weight file (no download)")
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--batch-size", type=int, default=64, help="start at 64, drop to 32 on OOM")
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
