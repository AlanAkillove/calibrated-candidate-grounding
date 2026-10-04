"""Run the final existing-weight suite and bind its result to source snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# These two opt-in tests generate new proposals/features, outside this repair.
os.environ["CCG_RUN_DETR_LIVE"] = "0"
os.environ["CCG_RUN_P1F3_LIVE"] = "0"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_repair_v1/logs"
sys.path.insert(0, str(ROOT / "src"))


def source_snapshot() -> dict[str, str]:
    paths = [path for directory in ("src", "scripts", "tests") for path in (ROOT / directory).rglob("*.py")]
    paths += [ROOT / "pyproject.toml", ROOT / "pytest.ini", ROOT / "conftest.py"]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def main() -> int:
    import pytest
    import torch
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    before = source_snapshot()
    started = datetime.now(timezone.utc).isoformat()
    elapsed = time.perf_counter()
    result = pytest.main(["tests", "-q", f"--junitxml={OUT / 'pytest_full_final.xml'}"])
    after = source_snapshot()
    record = {"schema": "research-repair-v1-final-test-source-snapshot-v1", "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "duration_seconds": time.perf_counter() - elapsed, "pytest_exit_code": int(result),
              "source_sha256": before, "source_unchanged_during_test_run": before == after,
              "changed_during_run": sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key)),
              "blas_threads": 2, "torch_threads": 2, "offline": True,
              "opt_in_extraction_tests": "not enabled; two ordinary suite skips expected"}
    (OUT / "final_test_sources.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return int(result) if before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
