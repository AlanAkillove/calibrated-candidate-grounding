"""Build the frozen, query-independent proposal bank per image (checklist step 2).

Contract (protocol section 4)
-----------------------------
``I -> P(I) = {p_1 .. p_N}``, ``N ~= 64``, produced by **one frozen detector**
that never sees the referring expression.  Each image yields a
:class:`ccg.data.types.ProposalBank` (boxes ``xyxy``, objectness, IoU with the
nearest GT object, that object's id) written into one HDF5 cache with a group
per image (:func:`ccg.data.proposals.write_banks`).

The unique-target rule is applied here as well:
:func:`ccg.data.proposals.assign_target` picks the ``argmax`` IoU proposal when
``max IoU >= 0.5`` and lists the *equivalent* ``IoU >= 0.5`` proposals that must
be removed so a candidate set can never contain two correct answers.

Detector weights are not part of this repository and must not be downloaded
from a script - hence the ``NotImplementedError`` below.

Usage
-----
    python scripts/extract_proposals.py --images data/coco/val2014 \
        --annotations data/coco/instances_val2014.json --out cache/proposals.h5
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

__all__ = ["PENDING_MESSAGE", "DEFAULT_PROPOSAL_N", "cache_path_for", "build_parser", "main"]

PENDING_MESSAGE = "pending data download — Phase 0 checklist step"
DEFAULT_PROPOSAL_N = 64
DEFAULT_IOU_THRESH = 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", type=Path, default=Path("data/coco"), help="COCO image root")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="COCO instances_*.json used for gt_ious / gt_assignment",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "testA", "testB"],
        help="which splits to process (images are deduplicated)",
    )
    parser.add_argument("--out", type=Path, default=Path("cache/proposals.h5"), help="cache file")
    parser.add_argument(
        "--proposal-n",
        type=int,
        default=DEFAULT_PROPOSAL_N,
        help="bank size N, FROZEN to 64 (default: %(default)s)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=DEFAULT_IOU_THRESH,
        help="unique-target IoU threshold, FROZEN to 0.5 (default: %(default)s)",
    )
    parser.add_argument(
        "--detector",
        # >>> 待真实数据核对 <<<: the exact frozen detector checkpoint is picked
        # during the Phase 0 checklist and must then be recorded in the run log.
        default="frozen-rpn-or-rcnn",
        help="identifier of the frozen proposal generator",
    )
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="images per forward pass")
    parser.add_argument("--seed", type=int, default=0, help="determinism seed")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "fp32"),
        help="inference precision of the frozen detector",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit 0")
    return parser


def cache_path_for(out: Path, proposal_n: int) -> Path:
    """``proposals.h5`` -> ``proposals_n64.h5``: the bank size is part of the key.

    A cache built with a different ``N`` is a different experiment, so ``N`` can
    never be a silent mismatch between writer and reader.
    """
    out = Path(out)
    if out.stem.endswith(f"_n{int(proposal_n)}"):
        return out
    return out.with_name(f"{out.stem}_n{int(proposal_n)}{out.suffix}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    target = cache_path_for(args.out, args.proposal_n)
    print(f"[extract_proposals] out        : {target}")
    print(f"[extract_proposals] N          : {args.proposal_n}")
    print(f"[extract_proposals] iou thresh : {args.iou_thresh}")
    print(f"[extract_proposals] backend    : {'h5py' if hdf5_available() else 'npz fallback'}")
    if args.dry_run:
        print("[extract_proposals] dry run, nothing computed")
        return 0
    raise NotImplementedError(PENDING_MESSAGE)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
