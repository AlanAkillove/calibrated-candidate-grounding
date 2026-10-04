"""Run the repository suite and bind actual test results to tested source bytes."""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
from ccg.v3.protocol import sha256


def main():
    import pytest
    out = ROOT / "results/v3_final_validation"
    paths = sorted(set((ROOT / "src").rglob("*.py")) | set((ROOT / "tests").rglob("*.py")) | set((ROOT / "scripts").glob("v3_*.py")))
    before = {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}
    started = datetime.now(timezone.utc).isoformat()
    junit = out / "tests_before_confirmation.xml"
    exit_code = pytest.main([str(ROOT / "tests"), "-o", "addopts=", "-q", f"--junitxml={junit}"])
    changed = [relative for relative, digest in before.items() if not (ROOT / relative).is_file() or sha256(ROOT / relative) != digest]
    counts = {name: 0 for name in ("tests", "errors", "failures", "skipped")}
    if junit.exists():
        tree = ET.parse(junit).getroot()
        suites = [tree] if tree.tag == "testsuite" else list(tree.findall("testsuite"))
        for suite in suites:
            for name in counts:
                counts[name] += int(suite.get(name, "0"))
    record = {"status": "PASS" if int(exit_code) == 0 and not changed else "FAIL_OR_TESTED_SOURCE_CHANGED",
              "command": "python -B scripts/v3_test_acceptance.py", "executable": sys.executable,
              "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "exit_code": int(exit_code), "counts": counts, "tested_source_inputs": before,
              "changed_during_tests": changed, "junit_path": junit.relative_to(ROOT).as_posix(),
              "junit_sha256": sha256(junit) if junit.exists() else None,
              "limitations": "Unit/repository tests do not replace actual exposure, dev, source-anchor and result acceptance."}
    (out / "tests_before_confirmation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in ("status", "exit_code", "counts", "changed_during_tests")}), flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
