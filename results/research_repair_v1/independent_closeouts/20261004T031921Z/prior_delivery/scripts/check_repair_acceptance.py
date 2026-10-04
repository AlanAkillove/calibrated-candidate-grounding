"""Check saved repair evidence; never train, resample, or mark a run complete."""
from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_repair_v1"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
REQUIRED_SCOPES = frozenset((
    "phase0a", "phase0b", "phase05", "phase1", "phase1f", "refcocog",
    "v2g_g3", "v2g_g4", "d1", "d2_c1", "d2_c4", "p", "m1", "m2", "m25", "m3",
))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check() -> dict:
    failures: list[str] = []
    sources: dict[str, str] = {}

    def read(relative: str) -> dict:
        path = OUT / relative
        if not path.is_file():
            failures.append(f"Missing {relative}")
            return {}
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode("utf-8-sig"))

    def require(condition: bool, explanation: str) -> None:
        if not condition:
            failures.append(explanation)

    def formal(method: dict, *, count: str = "n_replicates", seed: str = "seed") -> bool:
        return (method.get(count) == 5000 and method.get(seed) == 0
                and method.get("ci_level") == 0.95
                and method.get("resample_unit") == "image_cluster")

    def recovery_failure(scope_id: str, item: object, job_id: str | None = None) -> None:
        relative = item if isinstance(item, str) else item.get("path", "")
        path = (ROOT / relative).resolve()
        safe = bool(relative) and path.is_relative_to(OUT.resolve()) and path.is_file()
        require(safe, f"{scope_id}: missing repair-owned recovery evidence {relative}")
        if not safe:
            return
        try:
            record = read(path.relative_to(OUT).as_posix())
        except (ValueError, UnicodeDecodeError):
            require(False, f"{scope_id}: recovery evidence is not a structured failure record")
            return
        require(record.get("schema") == "research-repair-v1-recovery-failure-v1"
                and record.get("scope_id") == scope_id and record.get("status") == "UNVERIFIABLE"
                and bool(record.get("attempted_route")) and bool(record.get("source_sha256"))
                and bool(record.get("failure_reason")) and "original_tolerance" in record,
                f"{scope_id}: recovery record lacks failure provenance/identity")
        if job_id is not None:
            require(record.get("job_id") == job_id,
                    f"{scope_id}/{job_id}: recovery evidence belongs to a different job")
        log = (ROOT / record.get("log_path", "")).resolve()
        require(log.is_relative_to(OUT.resolve()) and log.is_file()
                and sha256(log) == record.get("log_sha256"),
                f"{scope_id}: recovery failure log missing or hash mismatch")
        for child in record.get("seed_failure_records", []):
            child_path = (ROOT / child.get("path", "")).resolve()
            safe_child = child_path.is_relative_to(OUT.resolve()) and child_path.is_file()
            require(safe_child and sha256(child_path) == child.get("sha256"),
                    f"{scope_id}: seed recovery failure record missing or hash mismatch")
            if safe_child:
                child_record = read(child_path.relative_to(OUT).as_posix())
                require(child_record.get("schema") == "research-repair-v1-recovery-failure-v1"
                        and child_record.get("scope_id") == child.get("scope_id")
                        and child_record.get("status") == "UNVERIFIABLE"
                        and bool(child_record.get("failure_reason"))
                        and bool(child_record.get("attempted_route")),
                        f"{scope_id}: invalid seed recovery failure provenance")
                child_log = (ROOT / child_record.get("log_path", "")).resolve()
                require(child_log.is_relative_to(OUT.resolve()) and child_log.is_file()
                        and sha256(child_log) == child_record.get("log_sha256"),
                        f"{scope_id}: seed recovery failure log missing or hash mismatch")
        inventory = record.get("no_checkpoint_inventory")
        if inventory:
            inventory_path = (ROOT / inventory.get("path", "")).resolve()
            require(inventory_path.is_relative_to(OUT.resolve()) and inventory_path.is_file()
                    and sha256(inventory_path) == inventory.get("sha256"),
                    f"{scope_id}: unavailable-checkpoint inventory missing or hash mismatch")

    statistics = read("statistics/summary.json")
    jobs = {job["job_id"]: job for job in statistics.get("jobs", [])}
    coverage = read("statistics/required_scope_coverage.json")
    scopes = coverage.get("scopes", [])
    if isinstance(scopes, dict):
        scopes = [dict(value, scope_id=key) for key, value in scopes.items()]
    ids = {scope.get("scope_id") for scope in scopes}
    require(REQUIRED_SCOPES <= ids, f"Required scope coverage missing: {sorted(REQUIRED_SCOPES - ids)}")
    require(len(ids) == len(scopes), "Required scope IDs must be unique")
    for scope in scopes:
        if scope.get("required", True) is False:
            require(scope.get("scope_id") not in REQUIRED_SCOPES, "A required scope was declared optional")
            continue
        scope_id, status = scope.get("scope_id"), scope.get("status")
        unavailable = scope.get("unverifiable_jobs", [])
        unavailable_ids = [row.get("job_id") for row in unavailable]
        formal_ids = set(scope.get("job_ids", []))
        require(len(set(unavailable_ids)) == len(unavailable_ids)
                and not formal_ids.intersection(unavailable_ids),
                f"{scope_id}: formal and unverifiable job identities overlap or repeat")
        if scope_id == "m3":
            require(formal_ids | set(unavailable_ids) == {
                "m3_b0_severity_macro", "m3_b1_severity_macro", "m3_b2_severity_macro"},
                "m3: all B0/B1/B2 jobs must be independently accounted for")
        for row in unavailable:
            job_id = row.get("job_id")
            require(row.get("required") is True and row.get("status") == "UNVERIFIABLE",
                    f"{scope_id}/{job_id}: unverifiable job is not an explicit required terminal")
            evidence = row.get("evidence_paths", [])
            if row.get("evidence_path"):
                evidence = [row["evidence_path"], *evidence]
            require(bool(evidence), f"{scope_id}/{job_id}: UNVERIFIABLE requires concrete recovery-failure evidence")
            for item in evidence:
                recovery_failure(scope_id, item, job_id)
        if status == "UNVERIFIABLE":
            evidence = scope.get("unverifiable_evidence", [])
            require(bool(evidence) or bool(unavailable),
                    f"{scope_id}: UNVERIFIABLE requires concrete recovery-failure evidence")
            for item in evidence:
                recovery_failure(scope_id, item)
            require(not formal_ids, f"{scope_id}: whole-scope UNVERIFIABLE cannot skip formal jobs")
            continue
        require(status in ("COMPLETE", "FORMAL_5000_COMPLETE", "VERIFIED_EXISTING_CORRECT_METHOD",
                           "COMPLETE_WITH_UNVERIFIABLE_RECOVERY"),
                f"{scope_id}: unfinished scope {status}")
        require((status == "COMPLETE_WITH_UNVERIFIABLE_RECOVERY") == bool(unavailable),
                f"{scope_id}: terminal status must disclose unverifiable jobs")
        require(bool(scope.get("job_ids")), f"{scope_id}: completed scope has no job evidence")
        missing_jobs = [job_id for job_id in scope.get("job_ids", []) if job_id not in jobs]
        require(not missing_jobs, f"{scope_id}: formal jobs missing {missing_jobs}")
        specification_rows = scope.get("job_estimator_ledger", scope.get("required_estimates", []))
        declared_jobs = {row.get("job_id") for row in specification_rows if isinstance(row, dict)}
        require(set(scope.get("job_ids", [])) <= declared_jobs,
                f"{scope_id}: required estimator ledger is not explicit for every job")
        for specification in specification_rows:
            if not isinstance(specification, dict) or specification.get("job_id") not in jobs:
                continue
            job = jobs[specification["job_id"]]
            declared = set(specification.get("estimate_names", [])) | set(specification.get("auxiliary_estimate_names", []))
            actual = {estimate["name"] for estimate in job.get("estimates", [])}
            require(bool(declared) and declared == actual,
                    f"{scope_id}/{specification['job_id']}: stored estimators differ from the required specification")
        for job_id in scope.get("job_ids", []):
            if job_id not in jobs:
                continue
            job = jobs.get(job_id, {})
            require(job.get("status") == "RECOMPUTED_5000" and formal(job.get("result", {})),
                    f"{scope_id}/{job_id}: missing compatible formal job")
            require(bool(job.get("estimates")), f"{job_id}: no estimates")
            for estimate in job.get("estimates", []):
                valid, invalid = estimate.get("valid_replicates"), estimate.get("invalid_replicates")
                require(estimate.get("n_replicates") == 5000
                        and estimate.get("resample_unit") == "image_cluster"
                        and estimate.get("ci_level") == 0.95,
                        f"{job_id}/{estimate.get('name')}: incompatible estimate method")
                require(isinstance(valid, int) and isinstance(invalid, int) and valid + invalid == 5000,
                        f"{job_id}/{estimate.get('name')}: replicate counts do not account for 5000 draws")
                require(isinstance(estimate.get("gate_eligible"), bool),
                        f"{job_id}/{estimate.get('name')}: gate eligibility must be explicit")
                require(not estimate.get("gate_eligible") or invalid == 0,
                        f"{job_id}/{estimate.get('name')}: invalid draws used for a gate")
                require(estimate.get("anchor_status") != "FAIL", f"{job_id}: failed numerical anchor")
                raw = estimate.get("raw_replicates")
                require(bool(raw) and (ROOT / str(raw).split("::", 1)[0]).is_file(),
                        f"{job_id}: raw replicate evidence missing")

    decisions = read("statistics/gate_decisions.json")
    gates = decisions.get("decisions", decisions.get("gates", []))
    require(bool(gates), "No old/new operational gate decisions")
    required_gates = {(scope.get("scope_id"), gate_id)
                      for scope in scopes for gate_id in scope.get("gate_ids", [])}
    actual_gates = {(gate.get("scope_id"), gate.get("gate_id")) for gate in gates}
    require(required_gates <= actual_gates,
            f"Missing required gate decisions: {sorted(required_gates - actual_gates)}")
    for gate in gates:
        verdict = gate.get("new_decision", gate.get("new_gate"))
        require(verdict is not None and not str(verdict).startswith("PENDING")
                and verdict != "THRESHOLD_REVIEW_REQUIRED",
                f"Unresolved gate {gate.get('gate_id')}")
        source = gate.get("predicate_source", gate.get("source", ""))
        require(bool(source) and not str(source).startswith("listed in required scope")
                and ("thresholds" in gate or "old_thresholds" in gate),
                f"Gate {gate.get('gate_id')} lacks original predicate/threshold provenance")
        require(gate.get("threshold_change", "NONE") == "NONE", f"Gate {gate.get('gate_id')} changed its historical threshold")
        if str(verdict).upper() != "NOT_APPLICABLE":
            require(gate.get("old_decision") is not None
                    and gate.get("old_thresholds", gate.get("thresholds")) is not None,
                    f"Gate {gate.get('gate_id')} retains unresolved historical decision/threshold placeholders")
        if str(verdict).upper() == "UNVERIFIABLE":
            gate_scope = next((scope for scope in scopes if scope.get("scope_id") == gate.get("scope_id")), {})
            require(bool(gate.get("unverifiable_evidence"))
                    or bool(gate_scope.get("unverifiable_jobs"))
                    or bool(gate_scope.get("unverifiable_evidence")),
                    f"Gate {gate.get('gate_id')} declares UNVERIFIABLE without recovery evidence")
        if str(verdict).upper() not in ("UNVERIFIABLE", "UNCERTAINTY_INSUFFICIENT", "NOT_APPLICABLE") \
                and not str(verdict).startswith("PENDING"):
            references = gate.get("new_estimates", [])
            require(bool(references), f"Gate {gate.get('gate_id')} has no concrete estimator references")
            for reference in references:
                if not isinstance(reference, dict):
                    require(False, f"Gate {gate.get('gate_id')} has an ambiguous estimator reference")
                    continue
                job = jobs.get(reference.get("job_id"), {})
                estimates = {estimate["name"]: estimate for estimate in job.get("estimates", [])}
                estimate = estimates.get(reference.get("estimate_name"), {})
                require(estimate.get("gate_eligible") is True and estimate.get("invalid_replicates") == 0
                        and estimate.get("valid_replicates") == 5000,
                        f"Gate {gate.get('gate_id')} consumes a missing or ineligible estimate {reference}")

    information = read("information/summary.json")
    method = read("information/information_bootstrap.json")
    require(information.get("status") == "COMPLETE" and information.get("formal_bootstrap_complete") is True,
            "Information axis unfinished")
    require(method.get("status") == "COMPLETE" and formal(method, seed="bootstrap_seed")
            and len(method.get("calls", [])) == 9, "Information axis lacks nine compatible formal calls")
    call_ids = [(call.get("scope"), call.get("eval_split")) for call in method.get("calls", [])]
    require(len(set(call_ids)) == len(call_ids), "Information axis repeats a formal call identity")
    for call in method.get("calls", []):
        require(formal(call, seed="bootstrap_seed"), f"Information call {call.get('scope')}/{call.get('eval_split')} has incompatible method")

    raw_verification = read("logs/stored_estimate_verification.json")
    require(raw_verification.get("status") == "PASS" and not raw_verification.get("failures")
            and set(raw_verification.get("axes", [])) == {"statistics", "information", "candidates", "mechanism"},
            "Complete raw-array/percentile/seed-mean verification missing or failed")
    expected_counts = {"statistics": sum(len(job.get("estimates", [])) for job in jobs.values()
                                          if job.get("status") == "RECOMPUTED_5000"),
                       "information": 1860, "candidates": 1404, "mechanism": 45}
    require(raw_verification.get("estimate_counts") == expected_counts,
            "Raw-array verification does not cover all declared formal estimates")
    require(len(raw_verification.get("estimate_source_sha256", {})) == 4
            and bool(raw_verification.get("raw_archives")), "Raw verification source/archive identities missing")
    for relative, expected in raw_verification.get("estimate_source_sha256", {}).items():
        path = (OUT / relative).resolve()
        require(path.is_relative_to(OUT.resolve()) and path.is_file() and sha256(path) == expected,
                f"Stored estimate verification source changed: {relative}")
    for archive in raw_verification.get("raw_archives", []):
        path = (ROOT / archive.get("path", "")).resolve()
        require(path.is_relative_to(OUT.resolve()) and path.is_file()
                and path.stat().st_size == archive.get("size_bytes")
                and sha256(path) == archive.get("sha256"),
                f"Raw archive changed after numerical verification: {archive.get('path')}")

    from ccg.repairs.acceptance_candidates import validate_candidate_chain
    candidate_chain = validate_candidate_chain(ROOT, OUT)
    failures.extend(candidate_chain["failures"])
    sources.update(candidate_chain["source_sha256"])

    mechanism = read("mechanism/summary.json")
    require(mechanism.get("status") == "COMPLETE" and formal(mechanism.get("bootstrap", {})),
            "Mechanism formal analysis unfinished or incompatible")

    verification = read("logs/final_input_verification.json")
    require(verification.get("status") == "PASS" and not verification.get("changed"), "Final immutable input verification missing or failed")
    require(len(verification.get("append_only_git_prefix_checks", [])) == 2
            and all(row.get("git_prefix_preserved_after_newline_normalization") is True
                    for row in verification.get("append_only_git_prefix_checks", [])), "Append-only document checks missing or failed")
    current_manifests = {path.relative_to(ROOT).as_posix(): sha256(path)
                         for path in [OUT / "input_manifest.json", *OUT.rglob("supplemental_input_manifest.json")]
                         if path.is_file()}
    verified_manifests = {row["path"]: row["sha256"] for row in verification.get("manifests", [])}
    require(current_manifests == verified_manifests, "Final input verification does not cover current manifest identities")

    junit = OUT / "logs/pytest_full_final.xml"
    tested_sources = read("logs/final_test_sources.json")
    require(tested_sources.get("pytest_exit_code") == 0
            and tested_sources.get("source_unchanged_during_test_run") is True
            and bool(tested_sources.get("source_sha256")), "Final tests lack a stable passing source snapshot")
    for relative, expected in tested_sources.get("source_sha256", {}).items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha256(path) == expected,
                f"Source changed after final full suite: {relative}")
    require(junit.is_file(), "Final full-suite JUnit record missing")
    if junit.is_file():
        sources[junit.relative_to(OUT).as_posix()] = sha256(junit)
        xml = ET.parse(junit).getroot()
        suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
        require(bool(suites) and sum(int(s.get("tests", 0)) for s in suites) >= 1175,
                "Final test run does not cover the full suite")
        require(sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites) == 0,
                "Final full suite has failures/errors")
        require(sum(int(s.get("skipped", 0)) for s in suites) == 2,
                "Final suite skip count differs from the two recorded opt-in live-extraction tests")
        skipped_names = {case.get("name") for case in xml.iter("testcase") if case.find("skipped") is not None}
        require(skipped_names == {"test_live_extract_detr_proposals_single_image", "test_live_route_f_smoke_and_gate"},
                "Final suite skipped tests differ from the two recorded opt-in live-extraction tests")

    return {"schema": "research-repair-v1-acceptance-v1", "status": "PASS" if not failures else "INCOMPLETE",
            "checked_utc": datetime.now(timezone.utc).isoformat(), "failures": failures,
            "source_sha256": sources, "declares_completion": False}


if __name__ == "__main__":
    result = check()
    destination = OUT / "logs/acceptance_check.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{result['status']}: {len(result['failures'])} unresolved acceptance checks")
    for failure in result["failures"]:
        print(f"- {failure}")
    raise SystemExit(0 if result["status"] == "PASS" else 1)
