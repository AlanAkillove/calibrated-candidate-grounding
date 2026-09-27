"""Materialise the frozen candidate manifests (multi-agent brief, 2026-09-27).

This script is a thin CLI over :func:`ccg.data.manifests.build_manifests`: it
builds one manifest per construction regime (``random`` / ``same_category``),
cuts it into the four UNC split files and writes

    <out>/manifests/<regime>_<split>.jsonl          (+ sibling .meta.json)

Every row freezes the target position inside the N=64 proposal bank plus ONE
full distractor ordering; the C_K candidate sets are the prefixes
``C_K = target + order[:K-1]`` for K in {5, 10, 20, 50}.  The ``val`` file
carries the ``eval_split`` column (``val_select`` / ``val_calib``, cut by
image with the frozen seed); :func:`ccg.data.manifests.filter_entries`
selects it back.

The proposal bank is produced by the main work stream
(``cache/proposals.h5``); while it is not built yet this script exits with a
clear FATAL message instead of producing anything.

Usage
-----
    python scripts/build_candidate_sets.py \
        --bank cache/proposals.h5 \
        --refs "data/raw/refcoco+/refcoco+/refs(unc).p" \
        --instances data/raw/refcoco+/refcoco+/instances.json \
        --coco data/raw/annotations/instances_train2014.json \
        --out cache --regime both --seed 20260927
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.manifests import (  # noqa: E402  (pure numpy / lazy h5py)
    FILE_SPLITS,
    MANIFEST_SEED,
    PRIMARY_KS,
    REGIMES,
    ManifestEntry,
    build_manifests,
    common_cohort,
    manifest_path,
)
from ccg.data.splits import SPLIT_VAL_CALIB, SPLIT_VAL_SELECT  # noqa: E402

__all__ = [
    "DEFAULT_BANK",
    "DEFAULT_REFS",
    "DEFAULT_INSTANCES",
    "DEFAULT_COCO",
    "DEFAULT_OUT",
    "summarize_entries",
    "build_parser",
    "run",
    "main",
]

DEFAULT_BANK = Path("cache/proposals.h5")
DEFAULT_REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
DEFAULT_INSTANCES = Path("data/raw/refcoco+/refcoco+/instances.json")
DEFAULT_COCO = Path("data/raw/annotations/instances_train2014.json")
DEFAULT_OUT = Path("cache")

#: The largest primary K; the common cohort is `eligible[50]` (protocol section 29).
_COHORT_K = max(PRIMARY_KS)


def summarize_entries(
    entries: Sequence[ManifestEntry], Ks: Sequence[int] = PRIMARY_KS
) -> Dict[str, Any]:
    """Per-split statistics of one regime's entries (pure, no I/O)."""
    n_refs = len(entries)
    n_target_present = sum(1 for entry in entries if entry.target_index is not None)
    n_eligible = {
        str(int(K)): sum(1 for entry in entries if bool(entry.eligible.get(int(K), False)))
        for K in Ks
    }
    n_common = int(common_cohort(entries, K=_COHORT_K).size)
    same_counts = np.asarray(
        [int(entry.n_same_category_available) for entry in entries], dtype=np.int64
    )
    if same_counts.size:
        same_summary = {
            "min": int(same_counts.min()),
            "median": float(np.median(same_counts)),
            "max": int(same_counts.max()),
            "mean": float(same_counts.mean()),
        }
    else:
        same_summary = {"min": 0, "median": 0.0, "max": 0, "mean": 0.0}
    return {
        "n_refs": n_refs,
        "n_target_present": n_target_present,
        "n_eligible": n_eligible,
        "n_common_cohort": n_common,
        "same_category_available": same_summary,
    }


def _print_summary(regime: str, split: str, summary: Dict[str, Any], out_path: Path) -> None:
    eligible = " ".join(
        f"K={K}:{summary['n_eligible'][str(int(K))]}" for K in PRIMARY_KS
    )
    same = summary["same_category_available"]
    tqdm.write(f"[build_candidate_sets] {regime}/{split}: n_refs={summary['n_refs']} "
          f"n_target_present={summary['n_target_present']} "
          f"common_cohort(K={_COHORT_K})={summary['n_common_cohort']}")
    tqdm.write(f"[build_candidate_sets]   eligible: {eligible}")
    tqdm.write(f"[build_candidate_sets]   n_same_category_available: "
          f"min={same['min']} median={same['median']} max={same['max']} mean={same['mean']:.2f}")
    tqdm.write(f"[build_candidate_sets]   -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK,
                        help="frozen proposal bank cache/proposals.h5 (written by the "
                             "main work stream; FATAL when missing)")
    parser.add_argument("--refs", type=Path, default=DEFAULT_REFS,
                        help="UNC refs(unc).p pickle")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES,
                        help="RefCOCO+ instances.json (ann_id -> target bbox)")
    parser.add_argument("--coco", type=Path, default=DEFAULT_COCO,
                        help="COCO train2014 instances json (same-category GT objects)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output root; manifests land in <out>/manifests/")
    parser.add_argument("--regime", choices=["random", "same_category", "both"],
                        default="both", help="which ordering regimes to build")
    parser.add_argument("--seed", type=int, default=MANIFEST_SEED,
                        help="frozen manifest seed (default: %(default)s)")
    return parser


def _require_file(path: Path, what: str, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}\n  {hint}")


def run(args: argparse.Namespace) -> int:
    """Build the requested regimes and write the per-split manifest files."""
    _require_file(
        args.refs, "RefCOCO+ refs pickle",
        "expected data/raw/refcoco+/refcoco+/refs(unc).p (see data/README.md)",
    )
    _require_file(
        args.instances, "RefCOCO+ instances.json",
        "expected data/raw/refcoco+/refcoco+/instances.json (ships with the refs pickle)",
    )
    _require_file(
        args.bank, "proposal bank (cache/proposals.h5)",
        "the frozen RPN bank is generated by the main work stream first; "
        "re-run this script once cache/proposals.h5 lands",
    )

    regimes = list(REGIMES) if args.regime == "both" else [args.regime]
    coco: Optional[Path] = args.coco
    if coco is not None and not coco.exists():
        if "same_category" in regimes:
            raise FileNotFoundError(
                f"COCO GT annotations not found: {coco}\n  needed for the same_category "
                "proposal categories; expected "
                "data/raw/annotations/instances_train2014.json"
            )
        print(f"[build_candidate_sets] WARNING: {coco} missing - random regime will "
              "record n_same_category_available=0 (metadata unavailable)")
        coco = None

    print(f"[build_candidate_sets] bank      : {args.bank}")
    print(f"[build_candidate_sets] regimes   : {regimes}")
    print(f"[build_candidate_sets] seed      : {args.seed}")
    print(f"[build_candidate_sets] out       : {args.out}")

    # Outer bar over the regime x split files; tqdm auto-disables on a non-TTY
    # stream (disable=None).
    manifests_bar = tqdm(
        total=len(regimes) * len(FILE_SPLITS), desc="manifests", unit="split", disable=None
    )
    for regime in regimes:
        manifest = build_manifests(
            args.bank, args.refs, args.instances, regime,
            seed=int(args.seed), coco=coco,
        )
        n_missing = int(manifest.meta.get("n_missing_bank", 0))
        if n_missing:
            tqdm.write(f"[build_candidate_sets] WARNING: {n_missing} ref image(s) have no bank "
                  "group (first ids: "
                  f"{manifest.meta.get('missing_bank_images', [])[:8]}); those refs are absent")
        tqdm.write(f"[build_candidate_sets] regime={regime}: built {manifest.meta['n_refs']} refs "
              f"over {manifest.meta['n_images']} images, "
              f"bank_fingerprint={manifest.meta['bank_fingerprint'][:16]}...")
        for split in FILE_SPLITS:
            entries = [entry for entry in manifest.entries if entry.split == split]
            subset = manifest.subset(entries)
            out_path = manifest_path(args.out, regime, split)
            # Inner bar over the split's refs: the per-ref work of this split
            # (jsonl write + statistics) is covered by its true ref count.
            with tqdm(
                total=len(entries), desc=f"{regime}/{split}", unit="ref", disable=None
            ) as ref_bar:
                subset.save(out_path)
                summary = summarize_entries(entries)
                ref_bar.update(len(entries))
            _print_summary(regime, split, summary, out_path)
            if split == "val":
                for eval_split in (SPLIT_VAL_SELECT, SPLIT_VAL_CALIB):
                    n_eval = sum(1 for entry in entries if entry.eval_split == eval_split)
                    n_eval_common = int(
                        common_cohort(
                            [entry for entry in entries if entry.eval_split == eval_split],
                            K=_COHORT_K,
                        ).size
                    )
                    tqdm.write(f"[build_candidate_sets]   {eval_split}: n_refs={n_eval} "
                          f"common_cohort={n_eval_common}")
            manifests_bar.update(1)
    manifests_bar.close()
    print("[build_candidate_sets] done")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
