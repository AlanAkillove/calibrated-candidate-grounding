"""Run the final existing-weight suite and bind its result to source snapshots."""
from __future__ import annotations

import argparse
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


def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        help="directory for the JUnit XML and source snapshot (default: the original logs directory)",
    )
    parser.add_argument(
        "--basetemp",
        help="fresh pytest temporary directory; TMP/TEMP/TMPDIR use its parent directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = _resolve_from_root(args.evidence_dir) if args.evidence_dir else OUT
    junit_path = out / "pytest_full_final.xml"
    snapshot_path = out / "final_test_sources.json"
    if args.evidence_dir and (junit_path.exists() or snapshot_path.exists()):
        raise FileExistsError(f"refusing to overwrite closeout evidence in {out}")
    out.mkdir(parents=True, exist_ok=True)

    pytest_args = ["tests", "-q", f"--junitxml={junit_path}"]
    basetemp = None
    tmp_root = None
    if args.basetemp:
        basetemp = _resolve_from_root(args.basetemp)
        # pytest removes/recreates --basetemp. Require a new child so no prior
        # user or review files can be removed by pytest's startup cleanup.
        if basetemp.exists():
            raise FileExistsError(f"pytest --basetemp must be a fresh path: {basetemp}")
        tmp_root = basetemp.parent
        tmp_root.mkdir(parents=True, exist_ok=True)
        for name in ("TMP", "TEMP", "TMPDIR"):
            os.environ[name] = str(tmp_root)
        pytest_args.append(f"--basetemp={basetemp}")

    import pytest
    import torch
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    before = source_snapshot()
    started = datetime.now(timezone.utc).isoformat()
    elapsed = time.perf_counter()
    result = pytest.main(pytest_args)
    after = source_snapshot()
    record = {"schema": "research-repair-v1-final-test-source-snapshot-v1", "started_utc": started,
              "finished_utc": datetime.now(timezone.utc).isoformat(),
              "duration_seconds": time.perf_counter() - elapsed, "pytest_exit_code": int(result),
              "source_sha256": before, "source_unchanged_during_test_run": before == after,
              "changed_during_run": sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key)),
              "blas_threads": 2, "torch_threads": 2, "offline": True,
              "evidence_dir": str(out), "junit_xml": str(junit_path),
              "pytest_basetemp": str(basetemp) if basetemp is not None else None,
              "python_temp_root": str(tmp_root) if tmp_root is not None else None,
              "opt_in_extraction_tests": "not enabled; two ordinary suite skips expected"}
    snapshot_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return int(result) if before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
