"""Evidence indexing must preserve sources and never manufacture completion."""
from __future__ import annotations

import importlib.util
import json
import hashlib
from pathlib import Path
import pytest


def _registry_module():
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_registry.py"
    spec = importlib.util.spec_from_file_location("repair_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_renders_formal_values_without_changing_sources_or_status(tmp_path, monkeypatch):
    module = _registry_module()
    out = tmp_path / "results/research_repair_v1"
    out.mkdir(parents=True)
    for axis in module.AXES:
        (out / axis).mkdir()
    manifest = {"base_commit": "frozen", "n_files": 2}
    status = {"status": "RUNNING", "completed": False,
              "stages": {axis: "IN_PROGRESS" for axis in module.AXES}}
    (out / "input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (out / "STATUS.json").write_text(json.dumps(status), encoding="utf-8")
    source = out / "statistics/coverage.csv"
    source.write_text("endpoint,new_point,new_ci_low,new_ci_high\nAUROC,0.123456,-0.01,0.23\n",
                      encoding="utf-8")
    baseline = tmp_path / "results/historical.json"
    baseline.write_text('{"value": 0.7}', encoding="utf-8")
    preserved = {p: p.read_bytes() for p in
                 (source, baseline, out / "STATUS.json", out / "input_manifest.json")}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    inventory = module.build()
    page = (out / "registry/statistics_tables.md").read_text(encoding="utf-8")
    assert "| AUROC | 0.123456 | -0.01 | 0.23 |" in page
    assert module.digest(source) in page
    assert inventory["status_snapshot"]["completed"] is False
    assert inventory["artifacts"][0]["n_rows"] == 1
    for path, value in preserved.items():
        assert path.read_bytes() == value


def test_large_prediction_tables_are_indexed_but_not_presented_as_headline_results(tmp_path, monkeypatch):
    module = _registry_module()
    out = tmp_path / "results/research_repair_v1"
    for axis in module.AXES:
        (out / axis).mkdir(parents=True)
    (out / "input_manifest.json").write_text('{"base_commit":"frozen","n_files":2}')
    (out / "STATUS.json").write_text(json.dumps({"status": "INCOMPLETE", "completed": False,
                                                "stages": dict.fromkeys(module.AXES, "NOT_STARTED")}))
    path = out / "information/predictions.csv"
    path.write_text("sentence_id,confidence\n77,0.987654321\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    result = module.build()
    assert any(r["path"].endswith("predictions.csv") for r in result["artifacts"])
    assert "0.987654321" not in (out / "registry/information_tables.md").read_text(encoding="utf-8")


def test_original_and_supplemental_hashes_reject_mutation_without_replacing_snapshot(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_repair_inputs.py"
    spec = importlib.util.spec_from_file_location("repair_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    (out / "logs").mkdir(parents=True)
    (out / "candidates").mkdir()
    source = tmp_path / "source.bin"
    extra = tmp_path / "annotation.bin"
    source.write_bytes(b"baseline")
    extra.write_bytes(b"annotation")
    def record(file):
        return {"path": file.name, "size_bytes": file.stat().st_size,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    manifest = out / "input_manifest.json"
    supplement = out / "candidates/supplemental_input_manifest.json"
    manifest.write_text(json.dumps({"inputs": [record(source)]}), encoding="utf-8")
    supplement.write_text(json.dumps({"inputs": [record(extra)]}), encoding="utf-8")
    snapshots = {p: p.read_bytes() for p in (manifest, supplement)}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    assert module.verify()["status"] == "PASS"
    extra.write_bytes(b"annotatioX")  # same size: the hash must detect it
    result = module.verify()
    assert result["status"] == "FAIL"
    assert result["changed"] == [{"path": "annotation.bin", "reason": "missing or bytes changed"}]
    for p, original in snapshots.items():
        assert p.read_bytes() == original


def test_publication_export_excludes_pilots_and_preserves_stored_effect_interval():
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimate = {"name": "Full_minus_S_plus_Q::auroc_correct", "metric": "auroc_correct",
                "point": 0.012, "ci_low": -0.004, "ci_high": 0.025,
                "seed_estimates": {"seed1": 0.005, "seed2": 0.015, "seed3": 0.016},
                "n_replicates": 5000, "valid_replicates": 4998, "invalid_replicates": 2}
    formal = {"job_id": "formal", "scope": "information", "cohort": "testA",
              "status": "RECOMPUTED_5000", "result": {"n_replicates": 5000, "seed": 0,
              "ci_level": 0.95, "resample_unit": "image_cluster"}, "estimates": [estimate]}
    pilot = dict(formal, job_id="pilot", status="PILOT")
    wrong_seed = dict(formal, job_id="other", result=dict(formal["result"], seed=7))
    rows = module.formal_statistics({"jobs": [pilot, wrong_seed, formal]})
    assert len(rows) == 1
    assert (rows[0]["point"], rows[0]["ci_low"], rows[0]["ci_high"]) == (0.012, -0.004, 0.025)
    assert rows[0]["n_seeds"] == 3
    assert rows[0]["invalid_replicates"] == 2


def test_publication_does_not_promote_point_training_or_mechanism_pilot_to_formal(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    for name in ("information", "mechanism", "publication"):
        (out / name).mkdir(parents=True)
    (out / "information/summary.json").write_text(json.dumps({
        "training_complete": True, "formal_bootstrap_complete": False, "formal_replicates": 0}))
    (out / "information/information_bootstrap.csv").write_text(
        "contrast,point,ci_low,ci_high,n_replicates\nFull_minus_S_plus_Q,0.1,0.05,0.15,5000\n")
    (out / "mechanism/summary.json").write_text(json.dumps({
        "status": "COMPLETE", "bootstrap": {"n_replicates": 2, "seed": 0, "ci_level": 0.95}}))
    (out / "mechanism/bootstrap_ci.csv").write_text("n_replicates,estimate\n2,interaction_I\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    assert module.information_evidence(out / "publication")["rows"] == 0
    assert module.mechanism_evidence(out / "publication")["rows"] == 0
    assert "0.15" not in (out / "publication/information.md").read_text(encoding="utf-8")
    assert "No completed formal source" in (out / "publication/mechanism.md").read_text(encoding="utf-8")
    # A stale completion flag cannot hide an incompatible resampling protocol.
    (out / "information/summary.json").write_text(json.dumps({
        "formal_bootstrap_complete": True, "formal_replicates": 5000}))
    (out / "information/information_bootstrap.json").write_text(json.dumps({
        "status": "COMPLETE", "n_replicates": 5000, "bootstrap_seed": 7,
        "ci_level": 0.95, "resample_unit": "image_cluster"}))
    assert module.information_evidence(out / "publication")["rows"] == 0


def test_candidate_publication_requires_matching_completed_pointer_and_keeps_sensitivity_pending(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    run = out / "candidates/audit_run_fixture"
    run.mkdir(parents=True)
    (out / "publication").mkdir()
    source = run / "summary.json"
    source.write_text(json.dumps({
        "status": "AUDIT_COMPLETE_MISMATCH_FOUND", "audit_run_id": "fixture",
        "mismatch": {"total_expressions": 10, "total_mismatches": 1, "overall_rate": 0.1,
                     "by_family": {"RPN": {"n_expressions": 10, "n_mismatches": 1, "mismatch_rate": 0.1}}},
        "historical_candidate_source_scope": {"sources": []}, "output_files": {}}))
    (out / "candidates/current_audit.json").write_text(json.dumps({
        "summary_path": source.relative_to(tmp_path).as_posix(), "audit_run_id": "fixture",
        "summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    result = module.candidate_evidence(out / "publication")
    assert result["audit_status"] == "AUDIT_COMPLETE_MISMATCH_FOUND"
    assert result["sensitivity_status"] == "PENDING"
    assert "sensitivity intervals remain pending" in (out / "publication/candidates.md").read_text()
    source.write_text(source.read_text() + "\n")  # same JSON values but a different frozen snapshot
    with pytest.raises(ValueError, match="matching snapshot"):
        module.candidate_evidence(out / "publication")


def test_registry_hash_and_render_share_one_snapshot_during_atomic_source_update(tmp_path, monkeypatch):
    module = _registry_module()
    out = tmp_path / "results/research_repair_v1"
    for axis in module.AXES:
        (out / axis).mkdir(parents=True)
    (out / "input_manifest.json").write_text('{"base_commit":"frozen","n_files":2}')
    (out / "STATUS.json").write_text(json.dumps({"status": "RUNNING", "completed": False,
                                                "stages": dict.fromkeys(module.AXES, "RUNNING")}))
    source = out / "statistics/coverage.csv"
    original = b"endpoint,new_point\nAUROC,0.123456\n"
    source.write_bytes(original)
    real_read = Path.read_bytes

    def read_then_publish_new_version(path):
        snapshot = real_read(path)
        if path == source:
            path.write_bytes(b"endpoint,new_point\nAUROC,0.987654\n")
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read_then_publish_new_version)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    result = module.build()
    page = (out / "registry/statistics_tables.md").read_text(encoding="utf-8")
    assert "0.123456" in page and "0.987654" not in page
    row = next(r for r in result["artifacts"] if r["path"].endswith("coverage.csv"))
    assert row["sha256"] == hashlib.sha256(original).hexdigest()
    assert row["size_bytes"] == len(original)


def test_formal_candidate_table_identifies_each_confidence_head(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report_candidate_heads", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    run = out / "candidates/audit_run_fixture"
    forward = run / "forward"
    sensitivity = forward / "sensitivity_bootstrap"
    sensitivity.mkdir(parents=True)
    destination = out / "publication"
    destination.mkdir()
    audit = run / "summary.json"
    audit.write_text('{}')
    forward_source = forward / "summary.json"
    forward_source.write_text('{"status":"COMPLETE"}')
    (sensitivity / "summary.json").write_text(json.dumps({
        "status": "COMPLETE", "audit_run_id": "fixture", "estimates_csv": "estimates.csv",
        "candidate_audit_summary_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "candidate_forward_summary_sha256": hashlib.sha256(forward_source.read_bytes()).hexdigest(),
        "bootstrap": {"n_replicates": 5000, "seed": 0, "ci_level": 0.95, "resample_unit": "image_cluster"},
    }))
    (sensitivity / "estimates.csv").write_text(
        "estimate,confidence_head,metric,contrast,point,ci_low,ci_high,n_replicates,seed_standard_deviation\n"
        "old_minus_new__msp,msp,auroc_correct,old_minus_new,0.001,0,0.003,5000,0.0002\n"
        "old_minus_new__e1b,e1b,auroc_correct,old_minus_new,0.002,0.001,0.004,5000,0.0003\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    result = module.candidate_sensitivity_evidence(audit, hashlib.sha256(audit.read_bytes()).hexdigest(), destination)
    assert result["sensitivity_rows"] == 2
    page = (destination / "candidate_sensitivity.md").read_text()
    assert "confidence_head" in page and "seed_standard_deviation" in page
    assert "| msp |" in page and "| e1b |" in page
    assert "do not establish equivalence" in page


def test_candidate_audit_counts_family_rows_sentence_ids_and_mismatch_object_pairs(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report_candidate_count_units", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    run = out / "candidates/audit_run_fixture"
    run.mkdir(parents=True)
    destination = out / "publication"
    destination.mkdir()
    audit_csv = run / "mismatch_by_expression.csv"
    audit_csv.write_text(
        "family,sentence_id,image_id,ann_id,category_mismatch\n"
        "RPN,s1,i1,a1,1\n"
        "RPN,s2,i1,a1,1\n"
        "DETR,s1,i1,a1,1\n"
        "DETR,s3,i1,b1,0\n", encoding="utf-8")
    mismatch = {
        "total_expressions": 4, "total_mismatches": 3, "overall_rate": 0.75,
        "by_family": {
            "RPN": {"n_expressions": 2, "n_mismatches": 2, "mismatch_rate": 1.0},
            "DETR": {"n_expressions": 2, "n_mismatches": 1, "mismatch_rate": 0.5},
        },
    }
    summary_path = run / "summary.json"
    summary_path.write_text(json.dumps({
        "status": "AUDIT_COMPLETE_MISMATCH_FOUND", "audit_run_id": "fixture",
        "mismatch": mismatch, "output_files": {"expression_audit": audit_csv.name},
        "historical_candidate_source_scope": {"sources": []},
    }), encoding="utf-8")
    summary_before = summary_path.read_bytes()
    csv_before = audit_csv.read_bytes()
    summary_hash = hashlib.sha256(summary_before).hexdigest()
    (out / "candidates/current_audit.json").write_text(json.dumps({
        "summary_path": summary_path.relative_to(tmp_path).as_posix(),
        "audit_run_id": "fixture", "summary_sha256": summary_hash,
    }), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)

    result = module.candidate_evidence(destination)
    counts = result["audit_count_evidence"]
    assert counts["status"] == "VERIFIED"
    assert counts["audit_record_unit"] == "family_expression_row"
    assert counts["audit_record_rows"] == 4
    assert counts["distinct_sentence_id_count"] == 3
    assert counts["mismatch_record_count"] == 3
    assert counts["mismatch_family_target_object_pair_count"] == 2
    assert counts["mismatch_pair_key_fields"] == ["family", "image_id", "ann_id"]
    assert counts["by_family"] == {
        "DETR": {"audit_record_rows": 2, "mismatch_record_rows": 1},
        "RPN": {"audit_record_rows": 2, "mismatch_record_rows": 2},
    }
    page = (destination / "candidates.md").read_text(encoding="utf-8")
    assert "3 mismatching rows among 4 family–expression audit records" in page
    assert "(RPN 2/2; DETR 1/2;" in page
    assert "3 distinct sentence_id values" in page
    assert "mismatches involve 2 family–target-object pairs" in page
    assert "4 audited expressions" not in page
    stored_counts = json.loads((destination / "candidate_audit_count_evidence.json").read_text(encoding="utf-8"))
    assert stored_counts == counts
    assert counts["expression_audit_sha256"] == hashlib.sha256(csv_before).hexdigest()
    assert summary_path.read_bytes() == summary_before
    assert audit_csv.read_bytes() == csv_before


def test_information_summary_quotes_conditional_v_values_from_formal_rows(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report_v_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    info = out / "information"
    destination = out / "publication"
    info.mkdir(parents=True)
    destination.mkdir()
    (info / "summary.json").write_text(json.dumps({
        "formal_bootstrap_complete": True, "formal_replicates": 5000,
    }), encoding="utf-8")
    (info / "information_bootstrap.json").write_text(json.dumps({
        "status": "COMPLETE", "n_replicates": 5000, "bootstrap_seed": 0,
        "ci_level": 0.95, "resample_unit": "image_cluster",
    }), encoding="utf-8")
    (info / "information_bootstrap.csv").write_text(
        "scope,cell,K,eval_split,contrast,metric,point,ci_low,ci_high,n_replicates,valid_replicates,invalid_replicates\n"
        "random,randomK50,50,__pooled_test__,Full_minus_S_plus_Q,auroc_correct,0.111111,0.101111,0.121111,5000,5000,0\n"
        "matched_same4,hard5,5,__pooled_test__,Full_minus_S_plus_Q,auroc_correct,0.222222,0.212222,0.232222,5000,5000,0\n"
        "dose_same8,expb_m8,10,__pooled_test__,Full_minus_S_plus_Q,auroc_correct,0.333333,0.323333,0.343333,5000,5000,0\n"
        "random,randomK5,5,__pooled_test__,Full_minus_S_plus_Q,auroc_correct,0.000321,-0.001000,0.001000,5000,5000,0\n"
        "dose_same8,expb_m0,10,__pooled_test__,Full_minus_S_plus_Q,auroc_correct,0.000654,-0.001000,0.001000,5000,5000,0\n",
        encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)

    result = module.information_evidence(destination)
    narrative = result["conditional_v_interpretation"]
    assert "random K=50 ΔAUROC=0.111111" in narrative
    assert "matched hard K=5 ΔAUROC=0.222222" in narrative
    assert "dose m=8 (K=10) ΔAUROC=0.333333" in narrative
    assert "random K=5 ΔAUROC=0.000321" in narrative
    assert "dose m=0 (K=10) ΔAUROC=0.000654" in narrative
    assert "not establish a universal V gain or identify a causal" in narrative
    page = (destination / "information.md").read_text(encoding="utf-8")
    assert narrative in page


def test_publication_explains_adaptive_ece_percentile_boundary(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report_ece_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path / "publication"
    destination.mkdir()
    module.main_tables([], {}, {}, {}, destination)
    page = (destination / "main_tables.md").read_text(encoding="utf-8")
    assert "non-smooth, especially near zero" in page
    assert "nominal coverage is not guaranteed" in page
    assert "a positive lower bound alone does not establish population miscalibration" in page
    assert "trapezoidal AURC minus the continuous oracle formula" in page
    assert "E-AURC can be negative" in page and "not clipped to zero" in page


def test_coverage_report_distinguishes_m3_replay_tolerance_from_protocol_cutoffs(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location("repair_report_m3_tolerance_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    (out / "statistics").mkdir(parents=True)
    destination = out / "publication"
    destination.mkdir()
    failure_path = out / "statistics/recovery/m3_failure.json"
    failure_path.parent.mkdir(parents=True)
    failure_record = {
        "schema": "research-repair-v1-recovery-failure-v1", "job_id": "m3_b1_severity_macro",
        "original_tolerance": {"phase_a_k5_store": 1e-9, "dose_rescore": 1e-4,
                               "committed_point_metrics": 1e-9, "m2_validation_stop": "original M2 STOP_TOL"},
        "observed_error": 1e-7, "failure_reason": "committed point differs",
    }
    failure_path.write_text(json.dumps(failure_record), encoding="utf-8")
    failure_before = failure_path.read_bytes()
    coverage = {
        "scopes": [{
            "scope_id": "m3", "status": "UNVERIFIABLE", "job_ids": [],
            "formal_jobs_complete": 0, "formal_jobs_required": 0,
            "recovery_method": "Original deterministic routes; original 1e-9 point-anchor tolerance retained.",
            "anchor_status": "STRICT_FAILURE_EVIDENCE",
            "unverifiable_jobs": [{
                "job_id": "m3_b1_severity_macro", "reason_code": "UNVERIFIABLE_RECOVERY",
                "evidence_path": failure_path.relative_to(tmp_path).as_posix(),
            }],
        }],
    }
    (out / "statistics/required_scope_coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)

    module.scope_and_gate_evidence(destination)
    page = (destination / "coverage.md").read_text(encoding="utf-8")
    assert "`original_tolerance` is the stored recovery-record field name" in page
    assert "repair wrapper's replay criterion" in page
    assert "not an established original protocol cutoff" in page
    assert "M3.1 frozen-store comparison uses 1e-9" in page
    assert "dose rescore uses 1e-4" in page
    assert "Recorded recovery tolerances:" in page
    assert "Original tolerances:" not in page
    assert "original 1e-9 point-anchor tolerance retained" not in page
    assert failure_path.read_bytes() == failure_before


@pytest.mark.parametrize("evidence_state", ["missing", "invalid_schema"])
def test_coverage_report_does_not_claim_tolerances_without_valid_recovery_record(
        evidence_state, tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/build_repair_report.py"
    spec = importlib.util.spec_from_file_location(f"repair_report_missing_m3_{evidence_state}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    (out / "statistics").mkdir(parents=True)
    destination = out / "publication"
    destination.mkdir()
    failure_path = out / "statistics/recovery/m3_failure.json"
    failure_path.parent.mkdir(parents=True)
    if evidence_state == "invalid_schema":
        failure_path.write_text(json.dumps({"schema": "unknown", "original_tolerance": {"fake": 1e-9}}),
                                encoding="utf-8")
    coverage = {"scopes": [{
        "scope_id": "m3", "status": "UNVERIFIABLE", "job_ids": [],
        "formal_jobs_complete": 0, "formal_jobs_required": 0,
        "recovery_method": "M3 recovery route.", "anchor_status": "UNVERIFIABLE",
        "unverifiable_jobs": [{
            "job_id": "m3_b0_severity_macro", "reason_code": "UNVERIFIABLE_RECOVERY",
            "evidence_path": failure_path.relative_to(tmp_path).as_posix(),
        }],
    }]}
    (out / "statistics/required_scope_coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)

    module.scope_and_gate_evidence(destination)
    page = (destination / "coverage.md").read_text(encoding="utf-8")
    assert "tolerance and anchor details are unavailable" in page
    assert "Recorded recovery tolerances:" not in page
    assert "fake" not in page


def test_acceptance_cannot_close_by_omitting_scopes_or_declaring_unverifiable_without_evidence(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/check_repair_acceptance.py"
    spec = importlib.util.spec_from_file_location("repair_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    (out / "statistics").mkdir(parents=True)
    (out / "input_manifest.json").write_text('{"inputs": []}')
    (out / "statistics/summary.json").write_text('{"status": "COMPLETE", "jobs": []}')
    coverage = out / "statistics/required_scope_coverage.json"
    coverage.write_text('{"scopes": []}')
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    omitted = module.check()
    assert omitted["status"] == "INCOMPLETE"
    assert any("Required scope coverage missing" in message for message in omitted["failures"])
    coverage.write_text(json.dumps({"scopes": [
        {"scope_id": scope, "required": True, "status": "UNVERIFIABLE", "unverifiable_evidence": []}
        for scope in module.REQUIRED_SCOPES
    ]}))
    unproven = module.check()
    assert unproven["status"] == "INCOMPLETE"
    assert sum("requires concrete recovery-failure evidence" in message
               for message in unproven["failures"]) == len(module.REQUIRED_SCOPES)
    assert not unproven["declares_completion"]


def test_mixed_recovery_completion_cannot_omit_backbones_or_skip_formal_jobs(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/check_repair_acceptance.py"
    spec = importlib.util.spec_from_file_location("repair_acceptance_mixed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    (out / "statistics").mkdir(parents=True)
    (out / "input_manifest.json").write_text('{"inputs": []}')
    (out / "statistics/summary.json").write_text('{"jobs": []}')
    coverage = out / "statistics/required_scope_coverage.json"
    failure_log = out / "recovery.log"
    failure_log.write_text("Original anchor discrepancy exceeds 1e-9")
    failure_record = out / "failure.json"
    failure_record.write_text(json.dumps({
        "schema": "research-repair-v1-recovery-failure-v1", "scope_id": "m3",
        "job_id": "m3_b0_severity_macro", "status": "UNVERIFIABLE",
        "attempted_route": ["original deterministic recovery"],
        "source_sha256": {"historical.json": "frozen"},
        "failure_reason": "Old point differs", "original_tolerance": 1e-9,
        "log_path": failure_log.relative_to(tmp_path).as_posix(),
        "log_sha256": hashlib.sha256(failure_log.read_bytes()).hexdigest(),
    }))
    scope = {"scope_id": "m3", "required": True,
             "status": "COMPLETE_WITH_UNVERIFIABLE_RECOVERY",
             "job_ids": ["m3_b1_severity_macro", "m3_b2_severity_macro"],
             "unverifiable_jobs": [{"job_id": "m3_b0_severity_macro", "required": True,
                 "status": "UNVERIFIABLE", "evidence_path": failure_record.relative_to(tmp_path).as_posix()}]}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    coverage.write_text(json.dumps({"scopes": [scope]}))
    result = module.check()
    assert not any("all B0/B1/B2" in item for item in result["failures"])
    assert any("formal jobs missing" in item for item in result["failures"])
    scope["job_ids"] = []
    coverage.write_text(json.dumps({"scopes": [scope]}))
    assert any("all B0/B1/B2" in item for item in module.check()["failures"])
    scope["job_ids"] = ["m3_b1_severity_macro", "m3_b2_severity_macro"]
    scope["status"] = "UNVERIFIABLE"
    scope["unverifiable_evidence"] = [failure_record.relative_to(tmp_path).as_posix()]
    coverage.write_text(json.dumps({"scopes": [scope]}))
    assert any("whole-scope UNVERIFIABLE cannot skip" in item for item in module.check()["failures"])


def test_acceptance_matches_verified_manifest_paths_and_detects_later_identity_change(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/check_repair_acceptance.py"
    spec = importlib.util.spec_from_file_location("repair_acceptance_manifests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results/research_repair_v1"
    (out / "logs").mkdir(parents=True)
    manifest = out / "input_manifest.json"
    manifest.write_text('{"inputs": []}')
    (out / "logs/final_input_verification.json").write_text(json.dumps({
        "status": "PASS", "changed": [],
        "manifests": [{"path": manifest.relative_to(tmp_path).as_posix(),
                       "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}],
        "append_only_git_prefix_checks": [{"git_prefix_preserved_after_newline_normalization": True}] * 2,
    }))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", out)
    result = module.check()
    assert not any("current manifest identities" in item for item in result["failures"])
    manifest.write_text('{"inputs": []}\n')  # unchanged data, changed verified snapshot
    result = module.check()
    assert any("current manifest identities" in item for item in result["failures"])
