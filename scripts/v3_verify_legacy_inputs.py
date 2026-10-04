"""Verify the repair's frozen input manifests without writing historical results."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.protocol import sha256


def verify(destination: Path) -> dict:
    out = ROOT / "results/research_repair_v1"
    manifests = [out / "input_manifest.json"] + sorted(out.rglob("supplemental_input_manifest.json"))
    initial = json.loads((out / "logs/initial_input_verification.json").read_text(encoding="utf-8"))
    changed = []
    if sha256(manifests[0]) != initial["baseline_manifest_sha256"]:
        changed.append({"path": str(manifests[0].relative_to(ROOT)), "reason": "baseline changed"})
    checked = {}
    manifest_records = []
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_records.append({"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256(manifest_path)})
        if "protocol_sha256" in manifest:
            protocol = ROOT / manifest["protocol"]
            if not protocol.is_file() or sha256(protocol) != manifest["protocol_sha256"]:
                changed.append({"path": manifest["protocol"], "reason": "frozen protocol changed"})
        for record in manifest["inputs"]:
            path = (ROOT / record["path"]).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Input outside workspace")
            expected = (record["size_bytes"], record["sha256"])
            if path in checked:
                if checked[path] != expected:
                    changed.append({"path": record["path"], "reason": "conflicting identities"})
                continue
            checked[path] = expected
            if not path.is_file() or path.stat().st_size != expected[0] or sha256(path) != expected[1]:
                changed.append({"path": record["path"], "reason": "missing or changed"})
        print(f"Verified {manifest_path.relative_to(out)}", flush=True)
    base = json.loads(manifests[0].read_text(encoding="utf-8"))["base_commit"]
    prefixes = []
    for relative in ("docs/research_protocol.md", "docs/experiment_log.md"):
        original = subprocess.check_output(["git", "show", f"{base}:{relative}"], cwd=ROOT)
        current = (ROOT / relative).read_bytes()
        preserved = current.replace(b"\r\n", b"\n").startswith(original.replace(b"\r\n", b"\n"))
        prefixes.append({"path": relative, "preserved": preserved, "base_blob_sha256": hashlib.sha256(original).hexdigest()})
        if not preserved:
            changed.append({"path": relative, "reason": "historical prefix changed"})
    report = {"status": "FAIL" if changed else "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(),
              "unique_files": len(checked), "changed": changed, "manifests": manifest_records,
              "prefix_checks": prefixes}
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "results/v3_final_validation").resolve()):
        raise ValueError("Verification output must be in V3 namespace")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/v3_final_validation/legacy_inputs_initial.json")
    result = verify(parser.parse_args().out)
    print(f"{result['status']}: {result['unique_files']} frozen files; {len(result['changed'])} changed")
    raise SystemExit(result["status"] != "PASS")
