"""Independent review entry point; writes only to this review directory."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CCG_RUN_DETR_LIVE"] = "0"
os.environ["CCG_RUN_P1F3_LIVE"] = "0"
ROOT = Path(__file__).resolve().parents[2]
REVIEW = Path(__file__).resolve().parent
OUT = ROOT / "results/research_repair_v1"
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def snapshot():
    files = [p for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py")]
    files += [ROOT / n for n in ("pyproject.toml", "pytest.ini", "conftest.py")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files) if p.is_file()}


def inputs():
    expected, failures, manifests = {}, [], []
    paths = [OUT / "input_manifest.json", *sorted(OUT.rglob("supplemental_input_manifest.json"))]
    for path in paths:
        data = read(path)
        manifests.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "n_files": len(data["inputs"])})
        if "protocol_sha256" in data and sha(ROOT / data["protocol"]) != data["protocol_sha256"]:
            failures.append("frozen repair protocol changed")
        for r in data["inputs"]:
            identity = (r["size_bytes"], r["sha256"])
            if r["path"] in expected and expected[r["path"]] != identity:
                failures.append("conflicting baseline " + r["path"])
            expected[r["path"]] = identity
    for i, (name, (size, digest)) in enumerate(expected.items(), 1):
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != size or sha(path) != digest:
            failures.append("input changed/missing " + name)
        if i % 100 == 0:
            print(f"Independent input hashes {i}/{len(expected)}", flush=True)
    initial = read(OUT / "logs/initial_input_verification.json")
    if sha(paths[0]) != initial["baseline_manifest_sha256"]:
        failures.append("baseline manifest changed")
    previous = read(OUT / "logs/final_test_sources.json")["source_sha256"]
    current = snapshot()
    source_changed = [n for n in set(previous) | set(current) if previous.get(n) != current.get(n)]
    failures.extend("source changed after claimed tests " + n for n in source_changed)
    base = read(paths[0])["base_commit"]
    prefixes = {}
    for name in ("docs/research_protocol.md", "docs/experiment_log.md"):
        original = subprocess.check_output(["git", "show", f"{base}:{name}"], cwd=ROOT)
        prefixes[name] = (ROOT / name).read_bytes().replace(b"\r\n", b"\n").startswith(original.replace(b"\r\n", b"\n"))
        if not prefixes[name]:
            failures.append("history prefix changed " + name)
    return {"status": "PASS" if not failures else "FAIL", "failures": failures,
            "n_inputs": len(expected), "manifests": manifests, "source_count": len(current),
            "source_changed_since_claimed_tests": source_changed, "historical_prefixes_preserved": prefixes}


def numerical():
    # This checks stored arithmetic using the inspected project verifier, not
    # independent sample-to-metric computation. Independent draws are separate.
    return module("verify_repair_estimates").check(["statistics", "information", "candidates", "mechanism"])


def tests():
    # Short fresh workspace path avoids Windows legacy path-length limits in
    # deeply nested synthetic fixtures. Never reuse/delete an existing path.
    temp_root = ROOT / ".rqa4"
    if temp_root.exists():
        raise RuntimeError("Refusing to reuse/delete prior reviewer temporary directory")
    temp_root.mkdir()
    for name in ("TEMP", "TMP", "TMPDIR"):
        os.environ[name] = str(temp_root)
    os.environ["MPLCONFIGDIR"] = str(temp_root / "matplotlib")
    import pytest
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    os.chdir(ROOT)
    before = snapshot()
    result = pytest.main(["tests", "-q", "--basetemp=" + str(temp_root / "pytest"), "--junitxml=" + str(REVIEW / "pytest_independent.xml")])
    after = snapshot()
    return {"status": "PASS" if result == 0 and before == after else "FAIL", "pytest_exit_code": int(result),
            "source_count": len(before), "source_sha256": before, "source_unchanged": before == after,
            "skips": "two original opt-in extraction tests disabled per approved scope", "temp_root": str(temp_root)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inputs", "numerical", "tests", "project_acceptance"))
    mode = parser.parse_args().mode
    start = time.time()
    result = module("check_repair_acceptance").check() if mode == "project_acceptance" else globals()[mode]()
    result["review_run_started_unix"] = start
    result["review_run_finished_unix"] = time.time()
    (REVIEW / (mode + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"INDEPENDENT REVIEW {mode}: {result['status']}", flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)
