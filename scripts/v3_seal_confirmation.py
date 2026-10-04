"""Seal accepted preparation inputs before the first confirmation scorer call."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.protocol import seal_freeze, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/v3_final_validation/protocol_freeze.json")
    args = parser.parse_args()
    auth = json.loads(args.authorization.read_text(encoding="utf-8"))
    prep = json.loads(args.preparation.read_text(encoding="utf-8"))
    if auth.get("schema") != "v3-confirmation-prepare-v1" or auth.get("confirmation_performance_observed") is not False:
        raise ValueError("Preparation authorization is not an unobserved confirmation authorization")
    if prep.get("stage") != "confirmation" or prep.get("authorization_sha256") != sha256(args.authorization):
        raise ValueError("Preparation is not bound to this confirmation authorization")
    if any(prep.get(name) is not False for name in ("scorer_forward", "reliability_forward", "performance_aggregation")):
        raise ValueError("Confirmation preparation must not have observed performance")
    records = {item["path"]: item for item in auth["inputs"]}
    if len(records) != len(auth["inputs"]):
        raise ValueError("Duplicate authorized input")
    for item in records.values():
        source = (ROOT / item["path"]).resolve()
        if not source.is_relative_to(ROOT) or sha256(source) != item["sha256"]:
            raise ValueError(f"Accepted preparation input changed: {item['path']}")
    for item in prep["inputs"]:
        relative = item["path"]
        if relative in records:
            if records[relative]["sha256"] != item["sha256"]:
                raise ValueError(f"Preparation changed its accepted input: {relative}")
            continue
        name = Path(relative).name
        role = "proposals" if name in ("proposals.h5", "empty_proposal_images.jsonl") else "candidates" if name == "candidate_manifest.jsonl" else "features"
        records[relative] = {**item, "role": role}
    for path, role in ((args.authorization, "protocol"), (args.preparation, "provenance"),
                       (ROOT / "reviews/v3_protocol.md", "protocol"), (Path(__file__), "code"),
                       (ROOT / "results/v3_final_validation/environment.json", "environment")):
        relative = path.resolve().relative_to(ROOT).as_posix()
        records[relative] = {"path": relative, "role": role, "sha256": sha256(path), "size_bytes": path.stat().st_size}
    payload = {"inputs": sorted(records.values(), key=lambda item: item["path"]), "acceptance": auth["acceptance"],
               "cohort_stage": auth["cohort_stage"], "primary_comparisons": ["C1", "C2", "C3", "C4"],
               "statistics": {"replicates": 5000, "bootstrap_seed": 0, "marginal_ci": .95,
                              "primary_ci": .9875, "risk_ties": "fractional_boundary_tie"},
               "confirmation_performance_observed": False,
               "interpretation": "Independent within recorded project exposure screening; foundation-model pretraining exposure unassessed"}
    print(json.dumps({"status": "SEALED_BEFORE_CONFIRMATION", "sha256": seal_freeze(args.out, payload, root=ROOT), "inputs": len(records)}))


if __name__ == "__main__":
    main()
