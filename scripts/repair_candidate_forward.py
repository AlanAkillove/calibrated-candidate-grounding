"""Research Repair v1 C: frozen inference over corrected candidate intersections."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccg.repairs.candidate_forward import (  # noqa: E402
    CURRENT_AUDIT_PATH,
    _write_json,
    resolve_current_audit,
    run_category_sensitivity_forward,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-pointer", type=Path, default=CURRENT_AUDIT_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--preflight-rows", type=int, default=None,
                        help="check only the first N canonical rows per source and write an isolated preflight report")
    parser.add_argument("--detr-feature-manifest", type=Path, default=None,
                        help="reuse and verify an existing immutable DETR feature-cache snapshot")
    args = parser.parse_args()

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    try:
        report = run_category_sensitivity_forward(
            pointer_path=args.audit_pointer, device=args.device,
            batch_size=args.batch_size, row_chunk=args.row_chunk,
            preflight_rows=args.preflight_rows,
            detr_feature_manifest_path=args.detr_feature_manifest, log=log,
        )
    except Exception as exc:
        try:
            audit = resolve_current_audit(args.audit_pointer)
            if args.preflight_rows is None:
                output_dir = audit["forward_dir"]
            else:
                output_dir = audit["forward_dir"].parent / (
                    f"forward_preflight_rows{int(args.preflight_rows)}_"
                    f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
                )
            if output_dir.is_dir() and not (output_dir / "summary.json").exists():
                _write_json(output_dir / "failure.json", {
                    "schema": "research-repair-v1-candidate-forward-failure-v1",
                    "status": "FAILED",
                    "audit_run_id": audit["pointer"]["audit_run_id"],
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                progress_path = output_dir / "progress.json"
                if progress_path.is_file():
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    progress["status"] = "FAILED"
                    progress["failure"] = "failure.json"
                    _write_json(progress_path, progress)
        except Exception as write_exc:
            log(f"failed to write forward failure record: {type(write_exc).__name__}: {write_exc}")
        raise
    log(
        f"candidate sensitivity forward {report['status']}: "
        f"{len(report['cohort_contract']['families_and_cells'])} source cohorts -> "
        f"{report['candidate_audit_pointer']['summary_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
