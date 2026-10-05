"""Read-only V3/repair identity audit; write evidence exclusively into V4."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "a203744cba2a44b12514012a966a753f028ab1e2"
OUT = ROOT / "results/v4_targeted_strengthening"
PROTECTED = ("results/v3_final_validation", "results/research_repair_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("initial", "final"), default="initial")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"baseline_audit_{args.stage}.json"
    if destination.exists():
        raise FileExistsError("Preserve existing audit evidence; use a new version explicitly")
    memo = {}
    errors = []

    def identity(relative):
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Input outside workspace")
        key = path.relative_to(ROOT).as_posix()
        if key not in memo:
            if not path.is_file():
                errors.append({"path": key, "reason": "missing"})
                return None
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            memo[key] = {"path": key, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        return memo[key]

    def verify(record, source):
        actual = identity(record["path"])
        if actual is not None and (actual["sha256"] != record["sha256"] or
                                  ("size_bytes" in record and actual["size_bytes"] != record["size_bytes"])):
            errors.append({"path": record["path"], "reason": "identity differs", "source": source})

    tracked = subprocess.run(["git", "diff", "--quiet", BASELINE, "--", *PROTECTED], cwd=ROOT)
    if tracked.returncode != 0:
        errors.append({"reason": "protected tracked state differs from baseline", "exit_code": tracked.returncode})
    inventory = {}
    for directory in PROTECTED:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file():
                item = identity(path.relative_to(ROOT))
                inventory[item["path"]] = item
        print(f"Protected inventory: {directory}; {len(inventory)} cumulative files", flush=True)
    if args.stage == "final":
        first = json.loads((OUT / "baseline_audit_initial.json").read_text(encoding="utf-8"))
        if inventory != first["protected_inventory"]:
            errors.append({"reason": "protected file inventory/bytes changed since V4 start"})
    freeze_path = ROOT / "results/v3_final_validation/protocol_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    for index, record in enumerate(freeze["inputs"], 1):
        verify(record, "V3 formal freeze")
        if index % 500 == 0:
            print(f"V3 formal frozen inputs: {index}/{len(freeze['inputs'])}", flush=True)
    repair = ROOT / "results/research_repair_v1"
    manifests = [repair / "input_manifest.json"] + sorted(repair.rglob("supplemental_input_manifest.json"))
    repair_paths = set()
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "protocol_sha256" in manifest:
            verify({"path": manifest["protocol"], "sha256": manifest["protocol_sha256"]}, str(path.relative_to(ROOT)))
        for record in manifest["inputs"]:
            verify(record, str(path.relative_to(ROOT)))
            repair_paths.add(record["path"])
    tests = json.loads((ROOT / "results/v3_final_validation/tests_before_confirmation.json").read_text(encoding="utf-8"))
    for name, digest in tests["tested_source_inputs"].items():
        verify({"path": name, "sha256": digest}, "V3 tested source snapshot")
    ledger = json.loads((ROOT / "results/v3_final_validation/confirmation_run.json").read_text(encoding="utf-8"))
    if ledger["freeze_sha256"] != identity(freeze_path.relative_to(ROOT))["sha256"] or ledger["corrections"]:
        errors.append({"reason": "V3 run identity/correction state mismatch"})
    for record in (ledger["predictions"], ledger["summary"]):
        verify(record, "V3 completed inference")
    index = json.loads((ROOT / "results/v3_final_validation/evidence_index.json").read_text(encoding="utf-8"))
    for record in index["sources"]:
        verify({**record, "path": "results/v3_final_validation/" + record["path"]}, "V3 publication evidence index")
    report = {"status": "FAIL" if errors else "PASS", "stage": args.stage,
              "checked_utc": datetime.now(timezone.utc).isoformat(), "baseline_commit": BASELINE,
              "protected_directories": list(PROTECTED), "protected_inventory": inventory,
              "protected_files": len(inventory), "v3_freeze_input_records": len(freeze["inputs"]),
              "repair_unique_input_paths": len(repair_paths), "repair_manifest_count": len(manifests),
              "v3_tested_source_files": len(tests["tested_source_inputs"]),
              "publication_sources": len(index["sources"]), "total_distinct_files_hashed": len(memo),
              "distinct_bytes_hashed": sum(item["size_bytes"] for item in memo.values()),
              "errors": errors, "scope": "Identity/provenance only; no model forward, metric recomputation, or V4 performance."}
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("status", "protected_files", "v3_freeze_input_records", "repair_unique_input_paths", "errors")}))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
