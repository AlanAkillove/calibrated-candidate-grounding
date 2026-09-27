"""CLI stub: extract OpenCLIP text (query) features, one per referring expression.

Already wired to the parts that do not need a GPU: the expression list is read
with :func:`ccg.data.refcoco.parse_refs_json` when the annotation file exists, so
the number of query embeddings to cache is known before any GPU pass.  The
encoding step itself stays disabled until the checkpoint is local.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..data.refcoco import parse_refs_json
from .clip_encoder import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, FEATURE_DIM

TODO_MESSAGE = (
    "TODO: query text feature extraction is not enabled yet - the encoder needs a local "
    "open_clip checkpoint and (for the full corpus) a GPU pass (Phase 0 checklist)."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_text",
        description="Extract frozen OpenCLIP query embeddings for RefCOCO+ expressions.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="referring-coco style refs json (e.g. refcoco+.json / refs_val.json)",
    )
    parser.add_argument("--split", default=None, help="force a split label when the file lacks one")
    parser.add_argument(
        "--out", type=Path, default=Path("cache/query_features.h5"), help="output feature cache"
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=FEATURE_DIM)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def count_queries(annotations: Path, split: str | None = None) -> int:
    """How many query embeddings the cache must hold (pure-CPU, no weights)."""
    examples = parse_refs_json(annotations, split=split)
    return len(examples)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(TODO_MESSAGE)
    if Path(args.annotations).exists():
        print(f"expressions to encode: {count_queries(Path(args.annotations), args.split)}")
    else:
        print(f"annotation file {args.annotations} not found - pending data download")
    print(f"parsed arguments: {vars(args)}")
    raise NotImplementedError("Phase 0 pipeline: to be enabled after data download")


if __name__ == "__main__":
    sys.exit(main())
