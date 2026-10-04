"""Bind accepted development work before confirmation feature preparation.

This permits only RPN/CLIP/candidate construction. Scorer forward requires a
separate final freeze including every derived feature and candidate artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.inference import B0_ENCODER_CHECKPOINT, B16_ENCODER_CHECKPOINT, frozen_model_paths, source_code_paths
from ccg.v3.protocol import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "results/v3_final_validation/config.json")
    parser.add_argument("--exposure", type=Path, action="append", required=True)
    parser.add_argument("--exposure-acceptance", type=Path, required=True)
    parser.add_argument("--development-acceptance", type=Path, required=True)
    parser.add_argument("--replication-acceptance", type=Path, required=True)
    parser.add_argument("--tests-acceptance", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/v3_final_validation/confirmation_prepare_authorization.json")
    args = parser.parse_args()
    records = {}
    def bind(path, role):
        source = Path(path).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f"Missing or outside workspace input: {source}")
        relative = source.relative_to(ROOT).as_posix()
        record = {"path": relative, "role": role, "sha256": sha256(source), "size_bytes": source.stat().st_size}
        if relative in records and records[relative]["role"] != role:
            raise ValueError(f"Conflicting frozen role: {relative}")
        records[relative] = record
        return relative
    acceptance = {}
    for stage, path in (("exposure", args.exposure_acceptance), ("development", args.development_acceptance),
                        ("replication", args.replication_acceptance), ("tests", args.tests_acceptance)):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise ValueError(f"Actual acceptance not passed: {stage}")
        role = "exposure" if stage == "exposure" else "acceptance"
        acceptance[stage] = {"status": "PASS", "path": bind(path, role)}
    cohort = bind(args.cohort, "cohort")
    bind(args.config, "config")
    for path in args.exposure:
        bind(path, "exposure")
    # Every actual image source is bound before creating proposal/feature caches.
    n_expressions = 0
    with args.cohort.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_split") != "val":
                raise ValueError("Confirmation preparation requires official positive val only")
            bind(ROOT / row["actual_image_path"], "image")
            n_expressions += 1
    if not n_expressions:
        raise ValueError("Confirmation cohort is empty")
    encoder_paths = {B0_ENCODER_CHECKPOINT.resolve(), B16_ENCODER_CHECKPOINT.resolve()}
    for metadata in tuple(encoder_paths):
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        encoder_paths.add((ROOT / payload["backbone"]["checkpoint_path"].replace("\\", "/")).resolve())
    for path in frozen_model_paths():
        bind(path, "encoder_model" if path.resolve() in encoder_paths else "model")
    # Bind all repository source modules, avoiding an incomplete dependency list.
    for path in sorted((ROOT / "src").rglob("*.py")):
        bind(path, "code")
    for path in source_code_paths():
        bind(path, "code")
    for relative in ("scripts/v3_bootstrap_confirmation.py", "scripts/v3_freeze.py", "scripts/v3_prepare_authorization.py"):
        bind(ROOT / relative, "code")
    bind(ROOT / "pyproject.toml", "environment")
    spec = {"schema": "v3-confirmation-prepare-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "cohort_stage": {"name": "confirmation", "split": "val", "manifest": cohort}, "acceptance": acceptance,
            "allowed_operations": ["rpn_bank", "clip_features", "candidate_manifest"],
            "disallowed_operations": ["grounding_forward", "reliability_forward", "performance_aggregation"],
            "confirmation_performance_observed": False, "n_expressions": n_expressions,
            "inputs": sorted(records.values(), key=lambda item: item["path"])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(spec, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": "PREPARATION_ONLY_AUTHORIZED", "inputs": len(records), "sha256": sha256(args.out)}))


if __name__ == "__main__":
    main()
