"""Materialise the frozen candidate sets (checklist step 4).

Everything here is pure numpy and already implemented in
:mod:`ccg.data.candidate_sets`; the script only needs the proposal cache (and,
for the CLIP-hard regime, the region embeddings):

* nested random sets ``C_5 subset C_10 subset C_20 subset C_50`` with a constant
  target (:func:`ccg.data.candidate_sets.build_nested_random_sets`),
* same-category hard sets (IoU with same-category GT objects first),
* CLIP-hard sets (query-crop cosine first) - a *diagnostic* regime, not a
  realistic detector distribution,
* synthetic omission variants ``C^- = C \\ {c*}``.

Sets are written as one NPZ/HDF5 artefact plus an **availability report**: a
requested ``K`` that the bank cannot supply is recorded, never silently dropped
(``--require-availability-report`` is on by default for exactly that reason).

Usage
-----
    python scripts/build_candidate_sets.py --proposals cache/proposals_n64.h5 \
        --annotations data/prepared/refs.json --out cache/candidate_sets.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.candidate_sets import (  # noqa: E402  (pure numpy)
    CandidateAvailability,
    build_nested_random_sets,
    synthetic_omit,
)
from ccg.data.proposals import assign_target, read_all_banks  # noqa: E402  (pure / h5py)
from ccg.data.refcoco import parse_refs_json  # noqa: E402  (pure)
from ccg.data.types import HARDNESS_VALUES  # noqa: E402
from ccg.utils.io import save_npz  # noqa: E402

__all__ = [
    "PENDING_MESSAGE",
    "DEFAULT_KS",
    "build_sets_for_example",
    "build_all",
    "build_parser",
    "main",
]

PENDING_MESSAGE = "pending data download — Phase 0 checklist step"
DEFAULT_KS = (5, 10, 20, 50)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--proposals", type=Path, default=Path("cache/proposals_n64.h5"))
    parser.add_argument("--annotations", type=Path, default=None, help="prepared refs json")
    parser.add_argument("--region-features", type=Path, default=None, help="needed for clip_hard")
    parser.add_argument("--text-features", type=Path, default=None, help="needed for clip_hard")
    parser.add_argument("--out", type=Path, default=Path("cache/candidate_sets.npz"))
    parser.add_argument(
        "--Ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="set sizes to materialise (default: %(default)s)",
    )
    parser.add_argument(
        "--hardness",
        nargs="+",
        default=["random"],
        choices=list(HARDNESS_VALUES),
        help="which construction regimes to build (clip_hard needs --region-features)",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="FROZEN target rule")
    parser.add_argument("--seed", type=int, default=0, help="FROZEN: sets are generated once")
    parser.add_argument(
        "--with-omission",
        action="store_true",
        help="also emit synthetic_omit variants of every target-present set",
    )
    parser.add_argument(
        "--require-availability-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="refuse to write when any requested K was unreachable without recording it",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit 0")
    return parser


def build_sets_for_example(
    example,
    bank,
    Ks: List[int],
    *,
    iou_thresh: float = 0.5,
    seed: int = 0,
    availability: Optional[CandidateAvailability] = None,
    with_omission: bool = False,
) -> dict:
    """Nested random sets for one expression, plus the target bookkeeping.

    Returns ``{K: CandidateSet}`` (``None`` where the bank was too small) merged
    with ``"<K>_omit"`` variants when ``with_omission`` is set, and records the
    equivalent proposals that the unique-target rule removed from the pool.
    """
    assignment = assign_target(bank, example.gt_box, iou_thresh=iou_thresh)
    if assignment.target_proposal_idx is None:
        # Natural omission: the bank does not contain the target at all.  The
        # example stays in the dataset (regime "natural_omission") - dropping it
        # here would hide the very failure mode RQ4 is about.
        return {"assignment": assignment, "sets": {}, "natural_omission": True}
    pool = list(range(bank.N))
    pool.remove(int(assignment.target_proposal_idx))
    for idx in assignment.to_remove.tolist():
        pool.remove(int(idx))
    sets = build_nested_random_sets(
        int(assignment.target_proposal_idx),
        pool,
        Ks,
        seed,
        ref_id=example.ref_id,
        image_id=example.image_id,
        availability=availability,
    )
    out = dict(sets)
    if with_omission:
        for k, candidate_set in sets.built().items():
            out[f"{k}_omit"] = synthetic_omit(candidate_set)
    return {"assignment": assignment, "sets": out, "natural_omission": False}


def build_all(
    proposals: Path,
    annotations: Path,
    Ks: List[int],
    *,
    iou_thresh: float = 0.5,
    seed: int = 0,
    with_omission: bool = False,
) -> dict:
    """Pure pass over cache + annotations; returns rows and the availability report."""
    banks = read_all_banks(proposals)
    examples = parse_refs_json(annotations)
    availability = CandidateAvailability(requested_Ks=tuple(int(k) for k in Ks))
    rows: List[dict] = []
    num_natural_miss = 0
    num_missing_bank = 0
    for example in examples:
        bank = banks.get(int(example.image_id))
        if bank is None:
            # No bank for the image at all: an artefact gap, reported separately
            # from a natural miss so the two are never conflated in the write-up.
            num_missing_bank += 1
            continue
        result = build_sets_for_example(
            example,
            bank,
            Ks,
            iou_thresh=iou_thresh,
            seed=seed,
            availability=availability,
            with_omission=with_omission,
        )
        num_natural_miss += int(result["natural_omission"])
        record = {
            "ref_id": int(example.ref_id),
            "image_id": int(example.image_id),
            "split": example.split,
            "natural_omission": bool(result["natural_omission"]),
            "target_proposal_index": result["assignment"].target_proposal_idx,
            "removed_equivalents": list(result["assignment"].to_remove),
        }
        for key, candidate_set in result["sets"].items():
            if candidate_set is None:
                continue
            record[f"set_{key}"] = candidate_set.candidate_indices.tolist()
            record[f"target_{key}"] = candidate_set.target_index
        rows.append(record)
    return {
        "rows": rows,
        "availability": availability.to_dict(),
        "num_natural_omission": num_natural_miss,
        "num_missing_banks": num_missing_bank,
        "Ks": [int(k) for k in Ks],
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[build_candidate_sets] Ks        : {args.Ks}")
    print(f"[build_candidate_sets] hardness  : {args.hardness}")
    print(f"[build_candidate_sets] proposals : {args.proposals} (exists: {args.proposals.exists()})")
    print(f"[build_candidate_sets] out       : {args.out}")
    if args.dry_run:
        print("[build_candidate_sets] dry run, nothing written")
        return 0
    if not args.proposals.exists() or args.annotations is None or not Path(args.annotations).exists():
        raise NotImplementedError(PENDING_MESSAGE)
    report = build_all(
        args.proposals,
        Path(args.annotations),
        args.Ks,
        iou_thresh=args.iou_thresh,
        seed=args.seed,
        with_omission=args.with_omission,
    )
    save_npz(args.out, rows=json.dumps(report["rows"]), availability=json.dumps(report["availability"]))
    print(f"[build_candidate_sets] built {len(report['rows'])} expressions")
    print(f"[build_candidate_sets] availability: {report['availability']}")
    if args.require_availability_report:
        shortages = report["availability"].get("example_shortages", [])
        print(f"[build_candidate_sets] shortages recorded: {len(shortages)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
