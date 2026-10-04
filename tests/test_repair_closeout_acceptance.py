"""A later closeout must select its own evidence and preserve earlier verdicts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_closeout_missing_evidence_cannot_fall_back_to_old_pass(tmp_path, monkeypatch):
    module = load_script("check_repair_acceptance")
    out = tmp_path / "results/research_repair_v1"
    old = out / "logs"
    old.mkdir(parents=True)
    # These previously passing records are deliberately present in the old location.
    for name in ("stored_estimate_verification.json", "final_input_verification.json", "final_test_sources.json"):
        (old / name).write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (old / "acceptance_check.json").write_text('{"status":"PASS","historical":true}', encoding="utf-8")
    preserved = {p: p.read_bytes() for p in old.iterdir()}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    import ccg.repairs.acceptance_candidates as candidates
    monkeypatch.setattr(candidates, "validate_candidate_chain", lambda *_: {"failures": [], "source_sha256": {}})
    selected = out / "independent_closeouts/new/logs"
    result = module.check(selected)
    assert result["status"] == "INCOMPLETE"
    assert result["evidence_directory"] == "independent_closeouts/new/logs"
    for name in ("stored_estimate_verification.json", "final_input_verification.json", "final_test_sources.json"):
        assert f"Missing independent_closeouts/new/logs/{name}" in result["failures"]
    assert all(p.read_bytes() == content for p, content in preserved.items())


@pytest.mark.parametrize("script", ["check_repair_acceptance", "verify_repair_inputs"])
def test_versioned_evidence_directory_cannot_escape_repair_output(tmp_path, monkeypatch, script):
    module = load_script(script)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "results/research_repair_v1")
    with pytest.raises(ValueError, match="inside results/research_repair_v1"):
        if script == "check_repair_acceptance":
            module.check(tmp_path / "elsewhere")
        else:
            module.verify(evidence_dir=tmp_path / "elsewhere")
    assert not (tmp_path / "elsewhere").exists()


def test_new_input_verification_keeps_previous_record_and_checks_current_bytes(tmp_path, monkeypatch):
    module = load_script("verify_repair_inputs")
    out = tmp_path / "results/research_repair_v1"
    (out / "logs").mkdir(parents=True)
    data = tmp_path / "input.bin"
    data.write_bytes(b"frozen input")
    manifest = {"inputs": [{"path": "input.bin", "size_bytes": data.stat().st_size, "sha256": module.sha256(data)}]}
    (out / "input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    previous = out / "logs/final_input_verification.json"
    previous.write_text('{"status":"PASS","historical":true}', encoding="utf-8")
    prior = previous.read_bytes()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    selected = out / "independent_closeouts/new/logs"
    assert module.verify(evidence_dir=selected)["status"] == "PASS"
    data.write_bytes(b"changed input")
    report = module.verify(evidence_dir=selected)
    assert report["status"] == "FAIL"
    assert report["changed"] == [{"path": "input.bin", "reason": "missing or bytes changed"}]
    assert previous.read_bytes() == prior
    assert json.loads((selected / "final_input_verification.json").read_text())["status"] == "FAIL"
