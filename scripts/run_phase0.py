"""Run the Phase 0 baseline grid (checklist step 6) - the actual GO/NO-GO study.

Planned pipeline, in the order the protocol fixes it:

1. load ``configs/phase0.yaml`` (which inherits ``default.yaml`` through
   ``_base_``) and freeze every knob into the experiment record,
2. read the frozen candidate sets + feature caches (no re-extraction, no
   re-sampling: the sets were generated once with a fixed seed),
3. score with the four baselines
   ``b0_random`` / ``b1_cosine`` / ``b2_temp`` / ``b3_mlp``,
4. fit calibration **on ``val_calib`` only** (global T, then the K-aware form
   ``T(K)=a+b*log K``; ``K=20/50`` are extrapolated through the form, never
   fitted), fit abstention thresholds on the same split,
5. hand the saved raw logits to ``ccg.metrics`` (accuracy / top-label ECE / AURC
   / risk@coverage, image-level paired bootstrap >= 5000 replicates) and evaluate
   Gate Q1 / Q2 / Q3,
6. write one :class:`ccg.utils.logging.ExperimentRecord` per (model, K, hardness,
   target-presence) cell plus the raw prediction dump.

Nothing in step 3-6 can run before the caches exist, hence the
``NotImplementedError``; steps 1 and the environment report below already work.

Usage
-----
    python scripts/run_phase0.py --config configs/phase0.yaml --dry-run
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

from ccg.data.splits import CALIBRATION_SPLITS, SELECTION_SPLITS  # noqa: E402  (pure)
from ccg.utils.config import load_config  # noqa: E402  (yaml only)
from ccg.utils.logging import new_experiment_id  # noqa: E402

__all__ = [
    "PENDING_MESSAGE",
    "BASELINES",
    "environment_report",
    "summarise_plan",
    "build_parser",
    "main",
]

PENDING_MESSAGE = "pending data download — Phase 0 checklist step"
#: FROZEN Phase 0 model set; a fifth baseline is a protocol amendment.
BASELINES = ("b0_random", "b1_cosine", "b2_temp", "b3_mlp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase0.yaml"))
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(BASELINES),
        choices=list(BASELINES),
        help="subset of the frozen baseline grid (default: all four)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="override the config seeds; every learned model must run 3 seeds",
    )
    parser.add_argument("--splits", nargs="+", default=None, help="override which splits to evaluate")
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--out", type=Path, default=Path("results/phase0"))
    parser.add_argument("--device", default="cuda", help="cuda / cpu (B3 training only)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an experiment id found in the JSONL log instead of starting a new one",
    )
    parser.add_argument("--experiment-id", default=None, help="explicit id (default: generated)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit 0")
    return parser


def environment_report() -> dict:
    """Which optional backends are importable - never raises, CPU-safe."""
    report: dict = {}
    for module in ("numpy", "torch", "h5py", "yaml", "scipy", "open_clip"):
        try:
            __import__(module)
            report[module] = True
        except Exception:  # pragma: no cover - environment dependent
            report[module] = False
    return report


def summarise_plan(config, baselines: List[str], seeds: Optional[List[int]]) -> dict:
    """Flat, checkable description of the run (also printed to stdout)."""
    return {
        "phase": config.get("phase", 0),
        "dataset": config.get("dataset.name"),
        "backbone": f"{config.get('backbone.library')} {config.get('backbone.model_name')} "
        f"{config.get('backbone.pretrained')}",
        "baselines": list(baselines),
        "Ks_test": config.get("Ks_test"),
        "Ks_train": config.get("Ks_train"),
        "hardness_test": config.get("hardness_test"),
        "iou_threshold": config.get("iou_threshold"),
        "proposal_n": config.get("proposal_n"),
        "seeds": list(seeds) if seeds else config.get("seeds", [config.get("seed", 0)]),
        "calibration_split": list(CALIBRATION_SPLITS),
        "selection_split": list(SELECTION_SPLITS),
        "bootstrap_replicates": config.get("statistics.bootstrap_replicates"),
        "param_budget": config.get("models.b3_independent_mlp.param_budget"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    experiment_id = args.experiment_id or new_experiment_id("phase0")
    print(f"[run_phase0] experiment : {experiment_id}")
    for key, value in summarise_plan(config, args.baselines, args.seeds).items():
        print(f"[run_phase0]   {key:<20}: {value}")
    print(f"[run_phase0]   environment       : {environment_report()}")
    missing = [
        name
        for name in ("proposals", "candidate_sets", "region_embeddings", "text_embeddings")
        if not (args.cache_root / name).exists() and not list(args.cache_root.glob(f"{name}*"))
    ] if args.cache_root.exists() else ["<cache root missing>"]
    print(f"[run_phase0]   missing caches    : {missing or 'none'}")
    if args.dry_run:
        print("[run_phase0] dry run, nothing computed")
        return 0
    raise NotImplementedError(PENDING_MESSAGE)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
