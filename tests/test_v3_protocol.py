"""Content changes, unfinished work and repeated confirmation must be refused."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccg.v3.protocol import claim_confirmation, seal_freeze, sha256, verify_freeze


def _spec(root: Path) -> dict:
    records = []
    for role in ("code", "model", "cohort", "config", "exposure", "development", "replication"):
        path = root / f"{role}.json"
        path.write_text(json.dumps({"status": "PASS", "role": role}), encoding="utf-8")
        records.append({"role": role, "path": path.name, "sha256": sha256(path)})
    return {
        "inputs": records,
        "cohort_stage": {"name": "confirmation", "split": "val", "manifest": "cohort.json"},
        "acceptance": {stage: {"status": "PASS", "path": f"{stage}.json"}
                       for stage in ("exposure", "development", "replication")},
        "primary_comparisons": ["C1", "C2", "C3", "C4"],
        "statistics": {"replicates": 5000, "bootstrap_seed": 0,
                       "primary_ci": .9875, "marginal_ci": .95,
                       "risk_ties": "fractional_boundary_tie"},
    }


def test_seal_verify_and_refuse_changed_model(tmp_path):
    payload = _spec(tmp_path)
    freeze = tmp_path / "freeze.json"
    seal_freeze(freeze, payload, root=tmp_path)
    assert len(verify_freeze(freeze, root=tmp_path)["inputs"]) == 7
    (tmp_path / "model.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_freeze(freeze, root=tmp_path)


def test_refuse_unfinished_or_falsely_labelled_acceptance(tmp_path):
    payload = _spec(tmp_path)
    payload["acceptance"]["development"]["status"] = "PENDING"
    with pytest.raises(ValueError, match="Acceptance"):
        seal_freeze(tmp_path / "freeze.json", payload, root=tmp_path)
    payload["acceptance"]["development"]["status"] = "PASS"
    path = tmp_path / "development.json"
    path.write_text('{"status":"PENDING"}', encoding="utf-8")
    for item in payload["inputs"]:
        if item["path"] == path.name:
            item["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="report"):
        seal_freeze(tmp_path / "freeze.json", payload, root=tmp_path)


def test_seal_and_confirmation_are_write_once(tmp_path):
    payload = _spec(tmp_path)
    freeze = tmp_path / "freeze.json"
    ledger = tmp_path / "run.json"
    seal_freeze(freeze, payload, root=tmp_path)
    with pytest.raises(FileExistsError):
        seal_freeze(freeze, payload, root=tmp_path)
    assert claim_confirmation(freeze, ledger, root=tmp_path)["status"] == "RUNNING"
    with pytest.raises(FileExistsError):
        claim_confirmation(freeze, ledger, root=tmp_path)


def test_paths_cannot_escape_workspace(tmp_path):
    payload = _spec(tmp_path)
    payload["inputs"][0]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="outside"):
        seal_freeze(tmp_path / "freeze.json", payload, root=tmp_path)


@pytest.mark.parametrize("cohort", [{}, {"name": "development", "split": "train", "manifest": "cohort.json"},
                                   {"name": "confirmation", "split": "val", "manifest": "model.json"}])
def test_confirmation_cohort_must_be_explicit_and_bound(tmp_path, cohort):
    payload = _spec(tmp_path)
    payload["cohort_stage"] = cohort
    with pytest.raises(ValueError, match="cohort"):
        seal_freeze(tmp_path / "freeze.json", payload, root=tmp_path)


@pytest.mark.parametrize("key,value", [("replicates", 20), ("primary_ci", .95), ("risk_ties", "stable")])
def test_reject_nonformal_or_changed_statistics(tmp_path, key, value):
    payload = _spec(tmp_path)
    payload["statistics"][key] = value
    with pytest.raises(ValueError):
        seal_freeze(tmp_path / "freeze.json", payload, root=tmp_path)
