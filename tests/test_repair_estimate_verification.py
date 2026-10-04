from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def test_saved_interval_and_per_draw_seed_mean_corruption_are_detected(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_repair_estimates.py"
    spec = importlib.util.spec_from_file_location("repair_estimate_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    directory = out / "statistics"
    directory.mkdir(parents=True)
    archive = directory / "toy.npz"
    first = np.linspace(-0.02, 0.06, 5000)
    second = np.linspace(0.01, 0.04, 5000)
    aggregate = (first + second) / 2
    def estimate(values, point):
        low, high = np.percentile(values, [2.5, 97.5])
        return {"point": point, "ci_low": low, "ci_high": high,
                "valid_replicates": 5000, "invalid_replicates": 0}
    metadata = dict(estimate(aggregate, 0.03), name="paired_effect",
                    raw_replicates=archive.relative_to(tmp_path).as_posix(),
                    per_seed={"one": estimate(first, 0.02), "two": estimate(second, 0.04)},
                    seed_standard_deviation=float(np.std([0.02, 0.04], ddof=1)))
    summary = {"jobs": [{"status": "RECOMPUTED_5000", "estimates": [metadata]}]}
    source = directory / "summary.json"
    source.write_text(json.dumps(summary))
    def write(mean):
        np.savez(archive, aggregate__paired_effect=mean,
                 seed_one__paired_effect=first, seed_two__paired_effect=second)
    write(aggregate)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    assert module.check(["statistics"])["status"] == "PASS"
    metadata["ci_low"] += 0.001
    source.write_text(json.dumps(summary))
    assert any("stored interval differs" in message
               for message in module.check(["statistics"])["failures"])
    # Preserve the aggregate's values and percentiles, but break draw pairing.
    metadata["ci_low"] -= 0.001
    source.write_text(json.dumps(summary))
    write(aggregate[::-1])
    result = module.check(["statistics"])
    assert any("not per-draw fixed-seed means" in message for message in result["failures"])
    assert not any("stored interval differs" in message for message in result["failures"])
