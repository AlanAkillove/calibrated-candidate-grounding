"""Content-bound freeze verification and a single confirmation-run ledger.

This is a procedural barrier, not a claim that a researcher cannot read files.
All inputs and acceptance reports must be bound before confirmation inference.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "v3-freeze-v1"
REQUIRED_ROLES = frozenset({"code", "model", "cohort", "config", "exposure"})
REQUIRED_ACCEPTANCE = ("exposure", "development", "replication")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Frozen input outside workspace: {relative}")
    return path


def _check(payload: Mapping[str, Any], root: Path) -> None:
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported V3 freeze schema")
    inputs = payload.get("inputs", [])
    if not inputs:
        raise ValueError("Freeze has no inputs")
    roles = set()
    seen = set()
    for record in inputs:
        relative = record["path"]
        if relative in seen:
            raise ValueError(f"Duplicate frozen path: {relative}")
        seen.add(relative)
        roles.add(record["role"])
        path = _inside(root, relative)
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen input missing or changed: {relative}")
    if not REQUIRED_ROLES.issubset(roles):
        raise ValueError(f"Missing freeze roles: {sorted(REQUIRED_ROLES - roles)}")
    cohort = payload.get("cohort_stage", {})
    if cohort.get("name") != "confirmation" or cohort.get("split") != "val":
        raise ValueError("Freeze must identify the official val confirmation cohort")
    cohort_paths = {item["path"] for item in inputs if item["role"] == "cohort"}
    if cohort.get("manifest") not in cohort_paths:
        raise ValueError("Confirmation cohort manifest must be content-bound as cohort")
    accepted = payload.get("acceptance", {})
    for stage in REQUIRED_ACCEPTANCE:
        item = accepted.get(stage, {})
        if item.get("status") != "PASS" or item.get("path") not in seen:
            raise ValueError(f"Acceptance not passed and content-bound: {stage}")
        report = json.loads(_inside(root, item["path"]).read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise ValueError(f"Acceptance report does not itself pass: {stage}")
    if payload.get("primary_comparisons") != ["C1", "C2", "C3", "C4"]:
        raise ValueError("Four primary comparisons must be fixed")
    statistics = payload.get("statistics", {})
    if statistics.get("replicates") != 5000 or statistics.get("bootstrap_seed") != 0:
        raise ValueError("Formal bootstrap must use 5000 draws, seed 0")
    if statistics.get("primary_ci") != 0.9875 or statistics.get("marginal_ci") != 0.95:
        raise ValueError("Predeclared confidence levels changed")
    if statistics.get("risk_ties") != "fractional_boundary_tie":
        raise ValueError("Risk tie convention must be frozen")


def verify_freeze(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Verify every bound file; refuse partial or merely self-labelled freezes."""
    workspace = Path(root).resolve()
    freeze = Path(path).resolve()
    if not freeze.is_relative_to(workspace):
        raise ValueError("Freeze outside workspace")
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    _check(payload, workspace)
    return payload


def seal_freeze(destination: str | Path, payload: Mapping[str, Any], *, root: str | Path) -> str:
    """Write once, only after actual content-bound acceptance checks pass."""
    workspace = Path(root).resolve()
    path = Path(destination).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError("Freeze destination outside workspace")
    content = dict(payload)
    content["schema"] = SCHEMA
    content["sealed_utc"] = datetime.now(timezone.utc).isoformat()
    _check(content, workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation means a later run cannot silently overwrite a freeze.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return sha256(path)


def claim_confirmation(freeze: str | Path, ledger: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Claim exactly one run before inference; re-entry needs a correction record."""
    verify_freeze(freeze, root=root)
    workspace = Path(root).resolve()
    path = Path(ledger).resolve()
    if not path.is_relative_to(workspace):
        raise ValueError("Confirmation ledger outside workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "v3-confirmation-run-v1",
        "freeze_sha256": sha256(freeze),
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "corrections": [],
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return record
