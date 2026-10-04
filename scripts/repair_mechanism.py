from __future__ import annotations

import os

# Set CPU math limits before importing NumPy, SciPy, or torch through project modules.
for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "2"

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccg.repairs.mechanism import OUT_DIR, run_mechanism_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Research Repair v1 four-corner mechanism accounting.")
    parser.add_argument("--compute", action="store_true", help="run formal analysis (fixed 5000/seed0/95%% CI)")
    parser.add_argument("--pilot", action="store_true", help="run a clearly labeled pilot in a separate directory")
    parser.add_argument("--pilot-replicates", type=int, default=100)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    if args.compute and args.pilot:
        parser.error("choose either --compute or --pilot")
    if not args.compute and not args.pilot:
        parser.print_help()
        return 0
    if args.compute:
        out_dir = OUT_DIR
        summary = run_mechanism_analysis(
            out_dir=out_dir, n_replicates=5000, seed=0, ci=0.95, formal_run=True
        )
    else:
        run_id = args.run_id or f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
            parser.error("run-id may contain only ASCII letters, digits, underscores, and hyphens")
        out_dir = OUT_DIR / f"pilot_{run_id}"
        if out_dir.exists():
            parser.error(f"pilot output directory already exists; refusing overwrite: {out_dir}")
        if args.pilot_replicates < 1 or args.pilot_replicates >= 5000:
            parser.error("pilot-replicates must be between 1 and 4999")
        summary = run_mechanism_analysis(
            out_dir=out_dir, n_replicates=args.pilot_replicates, seed=0, ci=0.95,
            formal_run=False,
        )
    print(f"status={summary['status']} output_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
