"""Audit the proposal bank before any decision experiment (checklist step 5).

Pure numpy, no GPU, no model: this is the gate that tells us whether the
perception layer is good enough to interpret RQ1-RQ4 at all (protocol section
4/6).  It reports

* proposal recall @ 32 and @ 64 (``max_i IoU(p_i, b*) >= 0.5``),
* the **natural miss rate** - expressions whose target is not in the bank, which
  is the realistic target-absent population and must be kept apart from
  synthetic omission,
* the distribution of ``max_i IoU``,
* how many proposals are *equivalent* (``IoU >= 0.5``) for one expression, i.e.
  how many rows the unique-target rule has to delete.

All of it comes from :mod:`ccg.data.proposals`; the script just points it at the
caches and dumps the json.

Usage
-----
    python scripts/audit_proposals.py --proposals cache/proposals_n64.h5 \
        --annotations data/prepared/refs.json --out results/phase0/proposal_audit.json
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

from ccg.data.proposals import (  # noqa: E402  (pure numpy + h5py reader)
    audit_examples,
    dump_audit_json,
    natural_miss_rate,
    proposal_recall,
    read_all_banks,
    summarize_proposal_quality,
)
from ccg.data.refcoco import parse_refs_json  # noqa: E402  (pure)

__all__ = [
    "PENDING_MESSAGE",
    "DEFAULT_TOP_KS",
    "audit_arrays",
    "audit_cache",
    "format_summary",
    "build_parser",
    "main",
]

PENDING_MESSAGE = "pending data download — Phase 0 checklist step"
#: FROZEN by the protocol: recall is reported at exactly these two truncation
#: levels of the objectness-ranked bank.
DEFAULT_TOP_KS = (32, 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--proposals", type=Path, default=Path("cache/proposals_n64.h5"))
    parser.add_argument("--annotations", type=Path, default=None, help="prepared refs json")
    parser.add_argument("--out", type=Path, default=Path("results/phase0/proposal_audit.json"))
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOP_KS),
        help="objectness truncation levels to audit (default: %(default)s)",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="FROZEN: 0.5")
    parser.add_argument(
        "--split",
        default=None,
        help="restrict the audit to one split (train/val_select/val_calib/testA/testB)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit 0")
    return parser


def audit_arrays(
    gt_boxes: List, 
    bank_boxes: List, 
    *, 
    iou_thresh: float = 0.5, 
    top_ks: List[int] = DEFAULT_TOP_KS,
) -> dict:
    """Audit two aligned python lists of arrays - the entry point used by unit
    tests and notebooks when no cache file exists yet.

    Wraps :func:`ccg.data.proposals.summarize_proposal_quality` and keeps the
    single-number answers (``proposal_recall`` / ``natural_miss_rate``) beside
    the full report so a reader never has to recompute them.
    """
    report = dict(summarize_proposal_quality(gt_boxes, bank_boxes, iou_thresh=iou_thresh, top_ks=top_ks))
    report["recall_at_full_bank"] = proposal_recall(gt_boxes, bank_boxes, iou_thresh=iou_thresh)
    report["natural_miss_at_full_bank"] = natural_miss_rate(gt_boxes, bank_boxes, iou_thresh=iou_thresh)
    return report


def audit_cache(
    proposals: Path,
    annotations: Path,
    *,
    iou_thresh: float = 0.5,
    top_ks: List[int] = DEFAULT_TOP_KS,
    split: Optional[str] = None,
) -> dict:
    """Run :func:`ccg.data.proposals.audit_examples` on the on-disk caches."""
    banks = read_all_banks(proposals)
    examples = parse_refs_json(annotations, split=split)
    return audit_examples(examples, banks, iou_thresh=iou_thresh, top_ks=top_ks)


def format_summary(report) -> str:
    """One-line-per-K rendering of an audit dict (human-readable stdout)."""
    lines: List[str] = []
    recall = report.get("recall_by_k", {}) if isinstance(report, dict) else {}
    miss = report.get("natural_miss_rate_by_k", {}) if isinstance(report, dict) else {}
    for k in recall:
        lines.append(
            f"  K={k:<4} recall={float(recall[k]):.4f}  natural_miss={float(miss.get(k, float('nan'))):.4f}"
        )
    if isinstance(report, dict) and "num_examples" in report:
        lines.insert(0, f"  examples={report['num_examples']}  missing_banks={report.get('missing_banks')}")
    return "\n".join(lines) if lines else "  (empty report)"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[audit_proposals] proposals  : {args.proposals} (exists: {args.proposals.exists()})")
    print(f"[audit_proposals] annotations: {args.annotations}")
    print(f"[audit_proposals] iou thresh : {args.iou_thresh}   top_ks: {args.top_ks}")
    if args.dry_run:
        print("[audit_proposals] dry run, nothing computed")
        return 0
    if not args.proposals.exists() or args.annotations is None or not Path(args.annotations).exists():
        raise NotImplementedError(PENDING_MESSAGE)
    report = audit_cache(
        args.proposals,
        Path(args.annotations),
        iou_thresh=args.iou_thresh,
        top_ks=args.top_ks,
        split=args.split,
    )
    print(format_summary(report))
    dump_audit_json(report, args.out)
    print(f"[audit_proposals] wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
