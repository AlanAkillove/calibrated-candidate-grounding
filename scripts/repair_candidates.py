"""Research Repair v1 C: immutable candidate-input preflight and audit runner."""
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

from ccg.repairs.candidates import (  # noqa: E402
    OUT_DIR,
    candidate_audit_run_dir,
    prepare_supplemental_input_manifest,
    run_candidate_audit,
    verify_supplemental_input_manifest,
    write_current_audit_pointer,
)
from ccg.repairs.atomic import AtomicReplaceError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-input-manifest", action="store_true",
        help="snapshot/verify COCO GT identity before the audit reads it",
    )
    parser.add_argument("--verify-input-manifest", action="store_true")
    parser.add_argument("--audit", action="store_true", help="run the full CPU candidate audit")
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="write audit into a new candidates/audit_run_<ID> directory; existing runs are never overwritten",
    )
    args = parser.parse_args()
    if not (args.prepare_input_manifest or args.verify_input_manifest or args.audit):
        parser.error("choose --prepare-input-manifest, --verify-input-manifest, or --audit")
    if args.prepare_input_manifest:
        manifest = prepare_supplemental_input_manifest()
        print(f"supplemental manifest ready: {manifest['inputs'][0]}", flush=True)
        print(f"path gap: {manifest['source_path_gap']}", flush=True)
    if args.verify_input_manifest:
        manifest = verify_supplemental_input_manifest()
        print(f"supplemental manifest verified: {manifest['inputs'][0]}", flush=True)
    if args.audit:
        # This is immutable if present; first-run creation is still before COCO load.
        prepare_supplemental_input_manifest()
        run_id = args.run_id or f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        out_dir = candidate_audit_run_dir(run_id, base_dir=OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=False)
        stage = "audit"
        try:
            summary = run_candidate_audit(out_dir=out_dir)
            stage = "current_pointer_update"
            pointer = write_current_audit_pointer(
                out_dir / "summary.json", base_dir=OUT_DIR, repo_root=ROOT
            )
        except Exception as exc:
            failure = {
                "schema": "research-repair-v1-candidate-audit-failure-v1",
                "status": (
                    "FAILED_CURRENT_POINTER" if stage == "current_pointer_update"
                    else "FAILED_OUTPUT_REPLACE" if isinstance(exc, AtomicReplaceError)
                    else "FAILED"
                ),
                "stage": stage,
                "audit_run_id": run_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if isinstance(exc, AtomicReplaceError):
                failure.update({
                    "target_path": exc.target.relative_to(OUT_DIR.resolve()).as_posix(),
                    "temporary_artifact_path": exc.source.relative_to(OUT_DIR.resolve()).as_posix(),
                    "winerror": exc.winerror, "errno": exc.errno,
                    "replace_attempts": exc.attempts,
                })
            with (out_dir / "audit_failure.json").open("x", encoding="utf-8", newline="") as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            raise
        print(
            f"candidate audit complete: {summary['status']} "
            f"mismatch={summary['mismatch']['total_mismatches']}/"
            f"{summary['mismatch']['total_expressions']} output={out_dir} pointer={pointer['summary_path']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
