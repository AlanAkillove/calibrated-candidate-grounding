"""Research Repair v1 C: shared image-cluster bootstrap for candidate sensitivity."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccg.repairs.candidate_forward import CURRENT_AUDIT_PATH, resolve_current_audit, _write_json  # noqa: E402
from ccg.repairs.candidate_sensitivity import run_candidate_sensitivity_bootstrap  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-pointer", type=Path, default=CURRENT_AUDIT_PATH)
    args = parser.parse_args()
    try:
        report = run_candidate_sensitivity_bootstrap(pointer_path=args.audit_pointer)
    except Exception as exc:
        try:
            audit = resolve_current_audit(args.audit_pointer)
            output_dir = audit["forward_dir"] / "sensitivity_bootstrap"
            if output_dir.is_dir() and not (output_dir / "summary.json").exists():
                _write_json(output_dir / "failure.json", {
                    "schema": "research-repair-v1-candidate-sensitivity-failure-v1",
                    "status": "FAILED", "audit_run_id": audit["pointer"]["audit_run_id"],
                    "exception_type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        except Exception:
            pass
        raise
    print(
        f"candidate sensitivity bootstrap {report['status']}: "
        f"{report['n_estimates']} estimates from {report['bootstrap_contract']['n_replicates']} draws",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
