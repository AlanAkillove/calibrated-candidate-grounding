"""Verify the frozen repair baseline and every versioned supplemental input list."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_repair_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict:
    manifests = [OUT / "input_manifest.json"] + sorted(OUT.rglob("supplemental_input_manifest.json"))
    checked = {}
    report = {"status": "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "manifests": [], "changed": [], "unique_files": 0}
    initial_record = OUT / "logs/initial_input_verification.json"
    if initial_record.exists():
        initial = json.loads(initial_record.read_text(encoding="utf-8"))
        expected_manifest = initial.get("baseline_manifest_sha256")
        if expected_manifest and sha256(manifests[0]) != expected_manifest:
            report["changed"].append({"path": "results/research_repair_v1/input_manifest.json",
                                      "reason": "baseline manifest identity changed"})
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest["inputs"]
        if "protocol_sha256" in manifest:
            protocol = ROOT / manifest["protocol"]
            if not protocol.is_file() or sha256(protocol) != manifest["protocol_sha256"]:
                report["changed"].append({"path": manifest["protocol"],
                                          "reason": "frozen repair protocol changed"})
        report["manifests"].append({"path": manifest_path.relative_to(ROOT).as_posix(),
                                    "sha256": sha256(manifest_path), "n_files": len(records)})
        for record in records:
            path = (ROOT / record["path"]).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError(f"Input outside the workspace: {record['path']}")
            expected = (record["size_bytes"], record["sha256"])
            if path in checked:
                if checked[path] != expected:
                    report["changed"].append({"path": record["path"], "reason": "conflicting snapshots"})
                continue
            checked[path] = expected
            if not path.is_file() or path.stat().st_size != expected[0] or sha256(path) != expected[1]:
                report["changed"].append({"path": record["path"], "reason": "missing or bytes changed"})
        print(f"Verified {manifest_path.relative_to(OUT)} ({len(records)} inputs)", flush=True)
    report["unique_files"] = len(checked)
    report["append_only_git_prefix_checks"] = []
    if (ROOT / ".git").exists():
        base_commit = json.loads(manifests[0].read_text(encoding="utf-8")).get("base_commit")
        if base_commit:
            for relative in ("docs/research_protocol.md", "docs/experiment_log.md"):
                original = subprocess.check_output(["git", "show", f"{base_commit}:{relative}"], cwd=ROOT)
                current = (ROOT / relative).read_bytes()
                # Git stores LF; Windows worktree files can retain CRLF.
                normalized_original = original.replace(b"\r\n", b"\n")
                preserved = current.replace(b"\r\n", b"\n").startswith(normalized_original)
                report["append_only_git_prefix_checks"].append({
                    "path": relative, "base_commit": base_commit,
                    "git_blob_sha256": hashlib.sha256(original).hexdigest(),
                    "git_prefix_preserved_after_newline_normalization": preserved})
                if not preserved:
                    report["changed"].append({"path": relative, "reason": "original Git document prefix changed"})
    report["status"] = "FAIL" if report["changed"] else "PASS"
    (OUT / "logs/final_input_verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = verify()
    print(f"{result['status']}: {result['unique_files']} unique baseline/supplemental inputs")
    raise SystemExit(0 if result["status"] == "PASS" else 1)
