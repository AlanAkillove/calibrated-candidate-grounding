"""Recompute frozen V3 statistics from saved predictions; never run a scorer."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.protocol import sha256, verify_freeze
from ccg.v3.statistics import controlled_inference, export_inference, load_prediction_grid, natural_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/v3_final_validation/pipeline/confirmation/bootstrap")
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze, root=ROOT)
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    if ledger.get("freeze_sha256") != sha256(args.freeze):
        raise ValueError("Prediction run ledger is not bound to this freeze")
    if ledger.get("status") not in ("INFERENCE_COMPLETE", "COMPLETE"):
        raise ValueError("Saved inference must finish before formal statistics")
    record = ledger.get("predictions", {})
    if record.get("sha256") != sha256(args.predictions):
        raise ValueError("Saved predictions differ from completed run ledger")
    order, images, grid = load_prediction_grid(args.predictions)
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"schema": "v3-formal-statistics-v1", "status": "RUNNING", "replicates": 5000, "bootstrap_seed": 0,
             "freeze_sha256": sha256(args.freeze), "predictions_sha256": sha256(args.predictions),
             "cohort_manifest": freeze["cohort_stage"]["manifest"],
             "cohort_sha256": sha256(ROOT / freeze["cohort_stage"]["manifest"]),
             "started_utc": datetime.now(timezone.utc).isoformat(), "stages": {}}
    def save():
        (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    save()
    for name, function in (("controlled", controlled_inference), ("natural", natural_inference)):
        print(f"Starting {name}: 5000 shared image draws, seed 0", flush=True)
        result = function(order, images, grid, n_replicates=5000)
        summary = export_inference(result, args.out / name, primary=name == "controlled")
        index["stages"][name] = {"status": summary["status"], "path": f"{name}/summary.json",
                                  "sha256": sha256(args.out / name / "summary.json")}
        save()
    index["status"] = "COMPLETE" if all(stage["status"] == "COMPLETE" for stage in index["stages"].values()) else "INCOMPLETE_UNDEFINED_COHORT"
    index["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(index), flush=True)


if __name__ == "__main__":
    main()
