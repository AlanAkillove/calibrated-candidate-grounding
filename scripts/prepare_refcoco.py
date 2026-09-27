"""Stage the RefCOCO+ / COCO annotations (Phase 0 checklist step 1).

What this script will do once the data is on disk
-------------------------------------------------
1. read the released ``refcoco+.json`` (or the per-split ``refs.*.json``) with
   :func:`ccg.data.refcoco.parse_refs_json`,
2. read the UNC ``*.mat`` releases with :func:`ccg.data.refcoco.parse_unc_mats`
   and check them against the json on ``ref_id`` / box level,
3. cut the UNC ``val`` split into ``val_select`` / ``val_calib`` at *image* level
   (:func:`ccg.data.splits.plan_val_split`) and freeze the resulting
   :class:`ccg.data.splits.SplitPlan` as json,
4. write a normalised record cache (one NPZ / parquet file, not a file per
   expression) plus a parse report.

Steps 1-2 are pure logic that already exists in :mod:`ccg.data.refcoco`; the
script wires them and only refuses to run while the annotation files are missing.

Usage
-----
    python scripts/prepare_refcoco.py --data-root data/refcoco+ --out data/prepared
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Make ``src/`` importable when executed as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.refcoco import parse_refs_json  # noqa: E402  (pure, no IO deps)
from ccg.data.splits import plan_val_split  # noqa: E402

__all__ = ["PENDING_MESSAGE", "build_parser", "annotation_candidates", "prepare", "main"]

#: Every Phase 0 script refuses with this exact message while the inputs are absent.
PENDING_MESSAGE = "pending data download — Phase 0 checklist step"

DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "refcoco+"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="directory holding refcoco+.json / refs.*.json (default: %(default)s)",
    )
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=None,
        help="explicit annotation json; auto-detected from --data-root when omitted",
    )
    parser.add_argument(
        "--box-format",
        default="xywh",
        choices=("xywh", "xyxy"),
        help="coordinate convention of the raw boxes (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data" / "prepared",
        help="output directory for the normalised records + split plan",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="keep only this split (train/val/testA/testB); default keeps all",
    )
    parser.add_argument(
        "--calib-fraction",
        type=float,
        default=0.5,
        help="image-level fraction of val reserved for val_calib (default: %(default)s)",
    )
    parser.add_argument("--split-seed", type=int, default=0, help="fixed split seed")
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=_REPO_ROOT / "data" / "coco",
        help="COCO root used to resolve image file names (informational here)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit 0")
    return parser


def annotation_candidates(data_root: Path) -> List[Path]:
    """Plausible annotation file names, most-likely first.

    >>> 待真实数据下载后核对 <<<: the released file is usually called
    ``refcoco+.json`` (Liu et al. releases) or ``refs_*.json`` per split; the
    exact spelling has to be confirmed against the actual download.
    """
    names = (
        "refcoco+.json",
        "refcocoplus.json",
        "refs_refcoco+.json",
        "refs_val.json",
        "refs_train.json",
        "refs_testA.json",
        "refs_testB.json",
    )
    found = [data_root / name for name in names if (data_root / name).exists()]
    if not found and data_root.exists():
        found = sorted(data_root.glob("*.json"))
    return found


def prepare(
    annotation_file: Path,
    *,
    out: Path,
    box_format: str = "xywh",
    split: Optional[str] = None,
    calib_fraction: float = 0.5,
    split_seed: int = 0,
) -> dict:
    """Run the pure part of the preparation and return a summary dict.

    Raises ``FileNotFoundError`` when ``annotation_file`` does not exist - the
    caller (``main``) turns that into the Phase 0 ``NotImplementedError``.
    """
    examples = parse_refs_json(annotation_file, split=split, box_format=box_format)
    plan = plan_val_split(examples, seed=split_seed, calib_fraction=calib_fraction)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_examples": len(examples),
        "splits": sorted({ex.split for ex in examples}),
        "split_plan": plan.to_dict(),
        "source": str(annotation_file),
        "box_format": box_format,
    }
    (out / "split_plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = annotation_candidates(args.data_root)
    if args.annotation_file is not None:
        candidates.insert(0, Path(args.annotation_file))
    print(f"[prepare_refcoco] data root : {args.data_root}")
    print(f"[prepare_refcoco] found     : {[str(p.name) for p in candidates] or 'nothing'}")
    if args.dry_run:
        print("[prepare_refcoco] dry run, nothing written")
        return 0
    if not candidates:
        raise NotImplementedError(PENDING_MESSAGE)
    summary = prepare(
        candidates[0],
        out=args.out,
        box_format=args.box_format,
        split=args.split,
        calib_fraction=args.calib_fraction,
        split_seed=args.split_seed,
    )
    print(f"[prepare_refcoco] parsed {summary['num_examples']} examples")
    raise NotImplementedError(
        f"{PENDING_MESSAGE}: normalised record cache + UNC cross-check still "
        "require the full COCO image set"
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
