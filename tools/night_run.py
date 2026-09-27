"""Unattended overnight driver: extraction -> cache audit -> Phase 0A -> tests.

Every stage is resume-capable or idempotent, so retrying a crashed stage never
redoes finished work:

    1. scripts/extract_features.py  --resume ...        (GPU, ~1-2 h)
    2. scripts/audit_cache.py       --sample 100        (GPU, minutes)
    3. scripts/run_phase0a.py       --regime random ... (CPU, minutes)
    4. pytest tests -q                                  (CPU, ~1 min)

Artefacts (all under ``logs_night/``):

    STATUS.json   machine-readable state, atomically replaced on every change
    master.log    this driver's own timeline (stdout; the detached launcher
                  appends here through the same file)
    stepN_*.log   raw stdout+stderr of each stage (append mode, one line
                  marking every attempt)
    SUMMARY.json  digest of the final results (written when the chain ends)
    PID.txt       the driver's pid

Retry policy: bounded retries with a fixed wait; ``run_phase0a`` exit code 2
(hard-sanity STOP) is *not* retried - the protocol freezes that a violated
nested-set property must be inspected, never patched automatically.

Pre-flight: refuses to start if another writer may still hold the region cache
(``cache/features/*.tmp`` newer than ``--fresh-window`` seconds), unless
``--force`` is given.  Dry run: ``--dry-run`` prints the plan and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOGS = ROOT / "logs_night"
STATUS = LOGS / "STATUS.json"
SUMMARY = LOGS / "SUMMARY.json"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

HEARTBEAT_S = 120.0

STEPS: List[Dict[str, Any]] = [
    {
        "name": "extract_features",
        "log": "step1_extract.log",
        "retries": 8,
        "timeout_s": 8 * 3600,
        "retry_wait_s": 60,
        "abort_codes": set(),
        "cmd": [
            PY, "-u", "scripts/extract_features.py",
            "--bank", "cache/proposals.h5",
            "--images-root", "data/raw/mscoco",
            "--manifest", "data/full_image_manifest.csv",
            "--refs", "data/raw/refcoco+/refcoco+/refs(unc).p",
            "--out", "cache/features",
            "--batch-size", "128",
            "--precision", "fp16",
            "--device", "cuda",
            "--resume",
        ],
    },
    {
        "name": "audit_cache",
        "log": "step2_audit.log",
        "retries": 3,
        "timeout_s": 2 * 3600,
        "retry_wait_s": 30,
        "abort_codes": set(),
        "cmd": [
            PY, "-u", "scripts/audit_cache.py",
            "--features", "cache/features",
            "--bank", "cache/proposals.h5",
            "--images-root", "data/raw/mscoco",
            "--manifest", "data/full_image_manifest.csv",
            "--sample", "100",
            "--device", "cuda",
        ],
    },
    {
        "name": "run_phase0a",
        "log": "step3_phase0a.log",
        "retries": 2,
        "timeout_s": 6 * 3600,
        "retry_wait_s": 30,
        "abort_codes": {2},  # hard-sanity STOP: inspect, never auto-retry
        "cmd": [
            PY, "-u", "scripts/run_phase0a.py",
            "--features", "cache/features",
            "--manifests", "cache/manifests",
            "--refs", "data/raw/refcoco+/refcoco+/refs(unc).p",
            "--instances", "data/raw/refcoco+/refcoco+/instances.json",
            "--out", "results/phase0a_cosine",
            "--regime", "random",
            "--bootstrap-replicates", "5000",
        ],
    },
    {
        "name": "pytest",
        "log": "step4_pytest.log",
        "retries": 1,
        "timeout_s": 1 * 3600,
        "retry_wait_s": 10,
        "abort_codes": set(),
        "cmd": [PY, "-m", "pytest", "tests", "-q", "-p", "no:warnings"],
    },
]

STATE: Dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def _write_status() -> None:
    STATE["updated_utc"] = utc_now()
    LOGS.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(STATE, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, STATUS)


def _tail_line(path: Path, limit: int = 600) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            lines = handle.read().decode("utf-8", "replace").strip().splitlines()
        return lines[-1][:200] if lines else "<empty>"
    except OSError:
        return "<unreadable>"


def _run_once(name: str, cmd: List[str], log_path: Path, timeout_s: float) -> int:
    """Run one attempt; returns the exit code (-9 on timeout kill)."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as handle:
        handle.write(f"\n===== {utc_now()} attempt start: {' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        )
        started = time.time()
        next_hb = started + HEARTBEAT_S
        while True:
            rc = proc.poll()
            if rc is not None:
                return int(rc)
            now = time.time()
            if now >= next_hb:
                next_hb = now + HEARTBEAT_S
                log(
                    f"[hb] {name} running {(now - started) / 60:.1f} min | "
                    f"last: {_tail_line(log_path)}"
                )
            if now - started > timeout_s:
                log(f"[watchdog] {name}: timeout after {timeout_s:.0f}s -> killing pid {proc.pid}")
                proc.kill()
                proc.wait()
                return -9
            time.sleep(10)


def _run_step(step: Dict[str, Any]) -> str:
    """Run one stage with bounded retries; returns 'ok' | 'stopped' | 'failed'."""
    name = step["name"]
    log_path = LOGS / step["log"]
    record = {
        "name": name,
        "status": "running",
        "attempts": 0,
        "started_utc": utc_now(),
        "log": str(log_path.relative_to(ROOT)),
        "cmd": step["cmd"],
    }
    STATE.setdefault("steps", []).append(record)
    _write_status()
    log(f"[step] {name}: start (retries={step['retries']})")

    rc: Optional[int] = None
    for attempt in range(1, int(step["retries"]) + 1):
        record["attempts"] = attempt
        _write_status()
        started = time.time()
        rc = _run_once(name, step["cmd"], log_path, float(step["timeout_s"]))
        took = time.time() - started
        log(f"[step] {name}: attempt {attempt} exit={rc} in {took / 60:.1f} min")
        if rc == 0:
            record.update(status="ok", exit_code=0, seconds=round(took, 1), ended_utc=utc_now())
            _write_status()
            return "ok"
        if rc in step["abort_codes"]:
            record.update(
                status="stopped", exit_code=int(rc), seconds=round(took, 1), ended_utc=utc_now()
            )
            _write_status()
            log(f"[step] {name}: STOP marker detected (exit={rc}) - no retry by design")
            return "stopped"
        if attempt < int(step["retries"]):
            wait = float(step["retry_wait_s"])
            log(f"[step] {name}: retrying in {wait:.0f}s (resume semantics, no rework)")
            time.sleep(wait)

    record.update(status="failed", exit_code=int(rc if rc is not None else -1), ended_utc=utc_now())
    _write_status()
    return "failed"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _head_lines(path: Path, n: int = 6) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [next(handle).rstrip("\n") for _ in range(n)]
    except (OSError, StopIteration):
        return []


def _write_summary() -> None:
    results = ROOT / "results" / "phase0a_cosine"
    payload: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "status": STATE.get("status"),
        "files": {},
        "cohort_summary": None,
        "metadata": None,
        "global_temperature": None,
        "oracle_temperature": None,
        "integrity_report_passed": None,
        "validation_failure": None,
        "csv_heads": {},
    }
    if results.exists():
        for path in sorted(results.rglob("*")):
            if path.is_file():
                payload["files"][str(path.relative_to(ROOT))] = path.stat().st_size
        payload["metadata"] = _read_json(results / "metadata.json")
        payload["cohort_summary"] = _read_json(results / "cohort_summary.json")
        payload["global_temperature"] = _read_json(results / "global_temperature.json")
        payload["oracle_temperature"] = _read_json(results / "oracle_temperature_diagnostic.json")
        payload["validation_failure"] = _read_json(results / "VALIDATION_FAILURE.json")
        for name in (
            "ranking_metrics.csv",
            "calibration_metrics.csv",
            "selective_metrics.csv",
            "bootstrap_ci.csv",
            "calibration_map_shift.csv",
            "reliability_bins.csv",
        ):
            payload["csv_heads"][name] = _head_lines(results / name, 6)
    report = _read_json(ROOT / "cache" / "features" / "integrity_report.json")
    if isinstance(report, dict):
        for key in ("passed", "ok", "all_passed"):
            if key in report:
                payload["integrity_report_passed"] = report[key]
                break
    SUMMARY.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log(f"[summary] wrote {SUMMARY.relative_to(ROOT)}")


def _preflight(fresh_window: float, force: bool) -> Optional[str]:
    """Return an error string when another writer may still be active."""
    feature_dir = ROOT / "cache" / "features"
    now = time.time()
    if feature_dir.exists():
        for tmp in feature_dir.glob("*.tmp"):
            age = now - tmp.stat().st_mtime
            log(f"[preflight] {tmp.name}: {tmp.stat().st_size / 1e6:.1f} MB, mtime {age:.0f}s ago")
            if age < fresh_window and not force:
                return (
                    f"{tmp.name} was touched {age:.0f}s ago - another extraction "
                    "may still hold it; stop it or pass --force"
                )
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--force", action="store_true", help="skip the active-writer preflight")
    parser.add_argument(
        "--fresh-window",
        type=float,
        default=180.0,
        help="a *.tmp touched less than this many seconds ago blocks the start",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(f"python : {PY}")
        for step in STEPS:
            print(f"\n[{step['name']}] retries={step['retries']} timeout={step['timeout_s']}s")
            print("  " + " ".join(step["cmd"]))
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "PID.txt").write_text(str(os.getpid()), encoding="utf-8")
    STATE.update(
        {
            "driver_pid": os.getpid(),
            "root": str(ROOT),
            "started_utc": utc_now(),
            "status": "running",
            "steps": [],
        }
    )
    _write_status()
    log(f"[driver] start pid={os.getpid()} python={PY}")

    error = _preflight(float(args.fresh_window), bool(args.force))
    if error is not None:
        log(f"[driver] preflight failed: {error}")
        STATE["status"] = "preflight_failed"
        STATE["error"] = error
        _write_status()
        return 2

    exit_code = 0
    for step in STEPS:
        outcome = _run_step(step)
        if outcome != "ok":
            STATE["status"] = "stopped" if outcome == "stopped" else "failed"
            STATE["failed_step"] = step["name"]
            exit_code = 3 if outcome == "stopped" else 1
            break
    else:
        STATE["status"] = "complete"

    STATE["finished_utc"] = utc_now()
    _write_status()
    _write_summary()
    log(f"[driver] {STATE['status']} (exit={exit_code}); logs in {LOGS.relative_to(ROOT)}")
    return exit_code


if __name__ == "__main__":  # pragma: no cover - ops entry point
    raise SystemExit(main())
