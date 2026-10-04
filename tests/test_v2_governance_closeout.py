# Research Repair v1: legacy transcription is checked in its preserved appendix;
# current scientific conclusions are checked by test_repair_integration.py.
"""V2 governance close-out: the two living governance docs must agree with the artifacts.

Covered documents:

* ``docs/experiment_log.md`` — the append-only section ``# V2 post-A11 result records``;
* ``docs/research_protocol.md`` — the append-only section ``# V2 Post-A11 Program Result Record``.

No verdict label, gate value or boundary sentence is duplicated as a literal here: each is
read back from its own committed artifact and then required to appear in the governance
documents (and, for the published labels, in the material bank
``docs/final_result_summary.md``). Three structural properties are enforced as well:

* **append-only** — whatever is committed at HEAD must still be a line-for-line prefix of the
  working documents, so a close-out round can never rewrite history;
* **no invented provenance** — every path inside an ``artifact_paths`` field of the new log
  section must exist, and a field the artifacts do not carry (an M1 upper confidence bound)
  may not be asserted as if it did;
* **no retroactive preregistration** — the protocol section must declare itself retrospective,
  and wordings that would claim otherwise are rejected.

Reads committed files only, so it runs on a fresh clone.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_LOG = _ROOT / "docs" / "experiment_log.md"
_PROTOCOL = _ROOT / "docs" / "research_protocol.md"
_PAPER = _ROOT / "docs" / "appendix" / "final_result_summary_pre_repair.md"

LOG_SECTION = "# V2 post-A11 result records"
PROTOCOL_SECTION = "# V2 Post-A11 Program Result Record"

DISCLAIMER = "This section is a retrospective result record"
DISCLAIMER_2 = "It does not retroactively preregister any completed"
RQ4_QUESTION = ("which measurable property explains the cross-proposal-family difference in "
                "cardinality-induced reliability degradation")


# --------------------------------------------------------------------------- plumbing

def _json(*parts: str) -> dict:
    return json.loads(_ROOT.joinpath(*parts).read_text(encoding="utf-8"))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _norm(text: str) -> str:
    """Collapse markup and whitespace so a quoted artifact sentence can be matched across
    the line wraps, blockquote markers and emphasis that every governance record uses."""
    text = re.sub(r"[`*|>#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has(doc: str, needle: str) -> bool:
    return _norm(needle) in _norm(doc)


def _has_prefix(doc: str, needle: str, limit: int = 90) -> bool:
    """A record may quote the head of an artifact sentence and close it early; match that."""
    return _has(doc, needle[:limit])


@pytest.fixture(scope="module")
def log_all() -> str:
    return _text(_LOG)


@pytest.fixture(scope="module")
def log_new(log_all: str) -> str:
    assert LOG_SECTION in log_all, "experiment log has no V2 post-A11 result-record section"
    return log_all[log_all.index(LOG_SECTION):]


@pytest.fixture(scope="module")
def protocol_all() -> str:
    return _text(_PROTOCOL)


@pytest.fixture(scope="module")
def protocol_new(protocol_all: str) -> str:
    assert PROTOCOL_SECTION in protocol_all, "protocol has no V2 Post-A11 program result record"
    return protocol_all[protocol_all.index(PROTOCOL_SECTION):]


def _artifact_labels() -> dict[str, str]:
    """Every verdict label the V2 close-out must carry, read from its own artifact."""
    d1 = _json("results", "v2_data_robustness", "d1_reviewed_annotations", "verdict.json")
    d2 = _json("results", "v2_d2_refcoco_lang", "c1_c4_verdict.json")
    m25 = _json("results", "v2_local_competition", "m25_specialist_audit", "verdict.json")
    m3 = _json("results", "v2_local_competition", "m3_mixture", "conf", "cross_backbone_verdict.json")
    p1 = _json("results", "v2_proposal_robustness", "p1_f5_c4", "f5_verdict.json")
    pm = _json("results", "v2_proposal_robustness", "p2_m_mechanism", "p2_m_verdict.json")
    table = _text(_ROOT / "results" / "v2_proposal_robustness" / "proposal_family_final_table.md")
    third = re.search(r"P2 extension verdict: \*\*([A-Z0-9_]+)\*\*", table)
    assert third, "the final table no longer carries the P2 extension verdict label"
    return {
        "D1 C1": d1["c1_cardinality"]["verdict"]["label"],
        "D1 C4": d1["c4_hard_semantic"]["verdict"]["label"],
        "D1 overall": d1["combined_verdict"],
        "D2 C1": d2["verdict"]["c1_cross_dataset"],
        "D2 C4": d2["verdict"]["c4_cross_dataset"],
        "D2 overall": d2["verdict"]["overall"],
        "D2 boundary": d2["verdict"]["boundary"],
        "M2.5": m25["pattern"],
        "M3": m3["verdict"],
        "P1 overall": p1["overall_p1_verdict"]["label"],
        "P2 third family": third.group(1),
        "P2-M": pm["verdict"],
        "P2-M class": pm["classification"],
    }


_GUARDS = (
    "禁止", "不得", "不要", "不授权", "未授权", "从未", "不存在", "没有", "不是", "不含",
    "无", "而非", r"\bnot\b", r"\bno\b", r"\bnever\b", "forbid", "may not", "must not",
    "cannot", "out of scope", "out_of_scope", "NOT ASSESSABLE", "withdraw",
)


def _governed(window: str) -> bool:
    low = window.lower()
    for guard in _GUARDS:
        if guard.startswith("\\b"):
            if re.search(guard, low):
                return True
        elif guard in low:
            return True
    return False


def _assert_no_unqualified_claim(doc: str, patterns: tuple[str, ...], name: str) -> None:
    """A forbidden wording is legal only inside an explicit prohibition or boundary quote."""
    offenders = []
    for pattern in patterns:
        for m in re.finditer(pattern, doc):
            window = doc[max(0, m.start() - 340):m.end() + 200]
            if not _governed(window):
                offenders.append((pattern, _norm(window[-260:])))
    assert not offenders, f"{name} asserts a forbidden wording without a governing negation: {offenders}"


def _artifact_path_blocks(section: str) -> list[str]:
    blocks = re.findall(r"^artifact_paths: >\n((?:[ \t]+\S[^\n]*\n?)+)", section, re.M)
    assert blocks, "no artifact_paths blocks found in the new log section"
    return blocks


def _quoted_paths(block: str) -> list[str]:
    """Extract ``results/...`` tokens from one ``artifact_paths`` body, expanding brace groups."""
    text = re.sub(r"\s+", " ", block)
    tokens: list[str] = []
    for m in re.finditer(r"([^\s,{}]*\{[^{}]*\})", text):
        head = m.group(1)
        base, opts = head[:head.index("{")], head[head.index("{") + 1:-1]
        for opt in opts.split(","):
            opt = opt.strip()
            if opt:
                tokens.append(base + opt)
        text = text.replace(m.group(1), " ", 1)
    for m in re.finditer(r"results/[\w./-]*[\w.-]", text):
        tokens.append(m.group(0))
    return [t for t in tokens if t.startswith("results/")]


#: a log record may compress a run of same-shaped artifacts as ``fig1..fig3.png``, meaning
#: the figures whose names start with fig1 / fig2 / fig3; the shorthand is only accepted when
#: every index in the range actually resolves to at least one committed file
_RANGE_SHORTHAND = re.compile(
    r"^(?P<head>.*?)(?P<stem>[A-Za-z_-]*)(?P<first>\d+)\.\.(?P<stem2>[A-Za-z_-]*)"
    r"(?P<last>\d+)(?P<ext>\.\w+)$")


def _expand_range(rel: str) -> list[str]:
    m = _RANGE_SHORTHAND.match(rel)
    if not m or m.group("stem") != m.group("stem2"):
        return []
    first, last = int(m.group("first")), int(m.group("last"))
    if last < first or last - first > 20:
        return []
    return [f"{m.group('head')}{m.group('stem')}{i}*{m.group('ext')}"
            for i in range(first, last + 1)]


def _resolves(rel: str) -> bool:
    if "*" in rel:
        return bool(list(_ROOT.glob(rel)))
    if _RANGE_SHORTHAND.match(rel):
        patterns = _expand_range(rel)
        assert patterns, f"unparseable range shorthand in an artifact_paths block: {rel}"
        return all(bool(list(_ROOT.glob(pattern))) for pattern in patterns)
    return (_ROOT / rel).exists()


# ------------------------------------------------------------------ append-only history

def test_governance_docs_are_strictly_append_only_at_head() -> None:
    for path in (_LOG, _PROTOCOL):
        rel = path.relative_to(_ROOT).as_posix()
        proc = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(_ROOT),
                              capture_output=True, text=True, encoding="utf-8")
        if proc.returncode != 0:
            pytest.skip("git unavailable; append-only is checked by review instead")
        committed = proc.stdout.splitlines()
        while committed and not committed[-1].strip():
            committed.pop()
        current = _text(path).splitlines()
        assert current[:len(committed)] == committed, (
            f"{rel}: a governance close-out may only append, never rewrite committed lines")


def test_historical_v2g_and_v2a1_text_survives(log_all: str, protocol_all: str) -> None:
    """The pre-A11 V2-G entry and the V2-A1 'LCR 尚未实现' sentence are kept, not overwritten."""
    assert log_all.index("# ===== FORMAL ENTRY") < log_all.index(LOG_SECTION)
    assert "v2a1_result_record.json" in log_all[:log_all.index(LOG_SECTION)]
    historical = protocol_all[:protocol_all.index(PROTOCOL_SECTION)]
    assert "LCR 尚未实现" in historical, "the historical V2-A1 status sentence was deleted"
    assert "Amendment V2-A1" in historical


# ----------------------------------------------------------------------- axis coverage

def test_experiment_log_registers_every_closed_axis(log_new: str) -> None:
    for heading in ("## V2-D1", "## V2-D2", "## V2-M｜", "## V2-M3", "## V2-P｜",
                    "## V2-P2｜", "## V2-P2-M｜", "## P2-A0｜"):
        assert heading in log_new, f"the new log section is missing the {heading!r} record"
    assert "legacy / axis-specific result record" in log_new


def test_log_and_protocol_carry_the_artifact_labels(log_new: str, protocol_new: str) -> None:
    for name, label in _artifact_labels().items():
        assert _has(log_new, label), f"experiment log does not carry the {name} label {label!r}"
        assert _has(protocol_new, label), f"protocol does not carry the {name} label {label!r}"


def test_material_bank_still_agrees_with_the_artifact_labels() -> None:
    paper = _text(_PAPER)
    for name, label in _artifact_labels().items():
        assert _has(paper, label), f"material bank lost the {name} label {label!r}"


# ------------------------------------------------------------------------ D1 / D2 rules

def test_d1_is_cleanup_robustness_not_a_new_dataset(log_new: str, protocol_new: str) -> None:
    d1 = _json("results", "v2_data_robustness", "d1_reviewed_annotations", "verdict.json")
    assert d1["constraints"]["new_training_parameters"] == 0
    assert _has(log_new, d1["constraints"]["reviewed_annotation_namespace"])
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        assert "annotation cleanup" in doc.lower(), f"{name} does not call D1 cleanup robustness"
        assert "新数据集" in doc, f"{name} does not bound D1 against 'a new dataset'"


def test_d2_impossible_disjoint_premise_is_disclosed(log_new: str, protocol_new: str) -> None:
    audit = _json("results", "v2_d2_refcoco_lang", "phase1", "cohort_audit.json")
    assert audit["overlap"]["image_disjoint_premise_holds"] is False
    assert audit["overlap"]["strict_image_disjoint_survivors"] == 0
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        assert "strict image-disjoint" in doc, f"{name} hides the impossible disjoint premise"
        assert "redefinition" in doc.lower(), f"{name} does not label the protocol redefinition"
    assert _has(log_new, audit["cohort_redefinition"])


# -------------------------------------------------------------------------- V2-M rules

def test_m1_gate_carries_no_upper_ci_and_the_record_says_so(log_new: str) -> None:
    gate = _json("results", "v2_local_competition", "gate.json")
    m1 = gate["gates"]["M1_GO"]
    assert m1["verdict"] is False
    assert not any("ci_high" in k or "ci_upper" in k for k in m1), (
        "the M1 artifact now carries an upper bound; the record must be updated honestly")
    record = log_new[log_new.index("RESULT RECORD — M1"):log_new.index("RESULT RECORD — M2（")]
    assert "ci_high" in record, "the M1 record does not state which CI field the artifact lacks"
    assert not re.search(r"(CI 上界|upper)[^\n]{0,24}<\s*0", record), (
        "the M1 record claims an upper bound below zero")
    assert f"{m1['delta_auroc']:.6f}".lstrip("-") in record.replace(" ", "").replace("+", "")


def test_m2_is_a_negative_result_with_the_correct_reading(log_new: str, protocol_new: str) -> None:
    gate = _json("results", "v2_local_competition", "m2_curriculum", "gate.json")
    assert gate["gates"]["M2_GO"]["verdict"] is False
    assert gate["authorization"]["v2mg_not_authorized"] is True
    assert "V2-MG_NOT_AUTHORIZED" in gate["authorization"]["note"]
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        assert "V2-MG_NOT_AUTHORIZED" in doc, f"{name} omits the non-authorization"
        # the forbidden reading may only appear inside an explicit prohibition
        _assert_no_unqualified_claim(doc, (r"LCR learned nothing",), name)
    assert "Aggregate-MLP" in log_new and re.search(r"LCR\s*(?:<=|≤)\s*E1b", log_new), (
        "the log must record both relations of the M2 result")
    assert _has(log_new, "did not outperform the simple semantic-statistics baseline")


def test_m2_5_is_labelled_diagnostic(log_new: str, protocol_new: str) -> None:
    verdict = _json("results", "v2_local_competition", "m25_specialist_audit", "verdict.json")
    boundary = verdict["interpretation"]["boundary"]
    assert boundary.lower().startswith("diagnostic only")
    assert _has(log_new, boundary) and _has(protocol_new, verdict["pattern"])
    record = log_new[log_new.index("RESULT RECORD — M2.5"):]
    assert "diagnostic" in record[:700]


def test_m3_verdict_and_alpha_evidence_match_the_artifacts(log_new: str, protocol_new: str) -> None:
    verdict = _json("results", "v2_local_competition", "m3_mixture", "conf", "cross_backbone_verdict.json")
    audit = _json("results", "v2_local_competition", "m3_mixture", "conf", "m3_conf_audit.json")
    assert verdict["n_pass"] == 0 and verdict["stop"] is True
    assert f"n_pass = {verdict['n_pass']}" in log_new
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        assert _has(doc, verdict["verdict"]), f"{name} does not carry the M3 verdict"
        # "collapsed everywhere" is only legal inside the explicit prohibition
        _assert_no_unqualified_claim(doc, (r"collapsed everywhere",), name)
    for back in ("b1", "b2"):
        alpha = audit["headline"][back]["alpha_m8_minus_m0"]
        assert alpha > 0, f"{back}: the adaptive weight did not respond; re-read the record"
        assert f"{alpha:.6f}" in log_new.replace(" ", ""), f"{back} alpha response not quoted"
    assert "POST-HOC" in log_new and "CONFIRMATORY" in log_new


# -------------------------------------------------------------------------- V2-P rules

def test_v2p_freezes_are_named_per_lineage(log_new: str, protocol_new: str) -> None:
    f45 = _json("results", "v2_proposal_robustness", "p1_f4_f5_config_freeze.json")
    p2c1 = _json("results", "v2_proposal_robustness", "p2_c1_gdino_config_freeze.json")
    pm = _json("results", "v2_proposal_robustness", "p2_m_mechanism_config_freeze.json")
    assert f45["classification"] == "CONFIRMATORY" and f45["frozen_before_results"] is True
    assert pm["authored_before_any_result"] is True
    for commit in ("1285e95", "91f2758", "87009ac"):
        assert commit in log_new, f"freeze commit {commit} not recorded in the log"
        assert commit in protocol_new, f"freeze commit {commit} not recorded in the protocol"
    assert _has_prefix(log_new, p2c1["authorization"])
    assert _has_prefix(log_new, pm["classification_meaning"])
    assert _has_prefix(log_new, f45["note"])


def test_p2_lineages_are_registered_separately_not_merged(log_new: str) -> None:
    for heading in ("F0–F3", "P1-F4", "P1-F5", "V2-P2-C1", "V2-P2-M", "P2-A0"):
        assert heading in log_new, f"lineage {heading!r} is not registered separately"
    assert "不得合并" in log_new or "不得把它们合并" in log_new


def test_gdino_is_c1_only_and_the_ban_is_recorded_in_both_docs(log_new: str, protocol_new: str) -> None:
    probe = _json("results", "v2_proposal_robustness", "p2_gdino_probe", "probe_report.json")
    supply = probe["gates"]["same_category_ge4_frac"]
    assert supply["pass"] is False and supply["measured"] < supply["threshold"]
    assert probe["verdict"] == "FAIL"
    c1 = _json("results", "v2_proposal_robustness", "p2_c1_gdino", "p2_c1_verdict.json")
    assert c1["c1_verdicts"]["GDINO"] == "YES"
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        assert "0.6892" in doc, f"{name} omits the supply-gate measurement"
        assert "0.85" in doc, f"{name} omits the supply-gate threshold"
        assert "C4" in doc, f"{name} does not mention the C4 scope decision"
        assert _has(doc, "no C4 verdict may be computed or reported for GDINO"), (
            f"{name} does not carry the C4 prohibition")
    assert _has(log_new, c1["out_of_scope_confirmed"][0]), "log does not record the C4 absence"


GDINO_FORBIDDEN = (
    r"GDINO\s+C4\b(?!\s*(?:verdict|manifest))",
    r"GDINO[^\n]{0,45}hard[- ]semantic",
    r"GDINO[^\n]{0,45}same-?category result",
    r"GDINO[^\n]{0,45}hard[- ]regime claim",
    r"GDINO[^\n]{0,45}hard[- ]competition claim",
)


def test_no_unqualified_gdino_c4_claim_in_the_governance_docs(log_new: str, protocol_new: str) -> None:
    _assert_no_unqualified_claim(log_new, GDINO_FORBIDDEN, "experiment log")
    _assert_no_unqualified_claim(protocol_new, GDINO_FORBIDDEN, "protocol")
    _assert_no_unqualified_claim(_text(_PAPER), GDINO_FORBIDDEN, "material bank")


def test_no_unqualified_cross_visual_domain_claim() -> None:
    patterns = (r"[Cc]ross-visual-domain", r"跨视觉域")
    for path in (_LOG, _PROTOCOL, _PAPER):
        _assert_no_unqualified_claim(_text(path), patterns, path.name)


def test_p2_m_is_descriptive_and_does_not_close_the_question(log_new: str, protocol_new: str) -> None:
    pm = _json("results", "v2_proposal_robustness", "p2_m_mechanism", "p2_m_verdict.json")
    assert pm["classification"] == "DESCRIPTIVE_MECHANISM"
    assert pm["model_forward_passes"] == 0 and pm["new_training_parameters"] == 0
    assert pm["rule_a_within_family_rho"]["passed"] is True
    assert pm["rule_b_matched_strata"]["passed"] is False
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        assert "DESCRIPTIVE_MECHANISM" in doc
        assert _has(doc, "does not explain the between-family amplification"), (
            f"{name} lacks the safe P2-M wording")
        _assert_no_unqualified_claim(doc, (r"composition in general is ruled out",), name)
    assert "算术" in log_new and "N = 64" in log_new


# ------------------------------------------------------------------------ P2-A0 rule

def test_p2_a0_withdrawal_is_recorded_in_both_docs(log_new: str, protocol_new: str) -> None:
    with (_ROOT / "results" / "v2_proposal_robustness" / "p2_amendment_history.csv").open(
            encoding="utf-8-sig", newline="") as fh:
        rows = {r["amendment_id"]: r for r in csv.DictReader(fh)}
    assert "P2-A0" in rows, "the ledger lost the P2-A0 row"
    row = rows["P2-A0"]
    assert row["modified_original_gates"] == "no"
    assert "prose only" in row["classification"]
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        assert "P2-A0" in doc, f"{name} does not name the ledger item"
        assert "WITHDRAWN" in doc, f"{name} does not mark the withdrawn interpretation"
        assert _has(doc, row["classification"]), f"{name} does not quote the ledger classification"
        assert _has(doc, "all gates unchanged") and _has(doc, "all thresholds unchanged"), (
            f"{name} does not state that nothing numeric moved")
    report = _text(_ROOT / "results" / "v2_proposal_robustness" / "p2_m_mechanism" / "mechanism_report.md")
    assert "Withdrawn" in report


# ------------------------------------------------------- protocol retrospective discipline

def test_protocol_section_is_an_explicit_retrospective_record(protocol_new: str) -> None:
    assert DISCLAIMER in protocol_new and DISCLAIMER_2 in protocol_new


def test_no_retroactive_preregistration_wording(log_new: str, protocol_new: str) -> None:
    for doc, name in ((log_new, "experiment log"), (protocol_new, "protocol")):
        lowered = doc.lower()
        for banned in ("preregistered now", "frozen before v2 began",
                       "this protocol was frozen before"):
            assert banned not in lowered, f"{name} claims retroactive preregistration: {banned!r}"


def test_program_status_declares_every_axis_closed(protocol_new: str) -> None:
    for token in ("V2-G CLOSED", "V2-D CLOSED", "V2-M CLOSED", "V2-M3 CLOSED", "V2-P CLOSED",
                  "No active experiment."):
        assert token in protocol_new, f"program status is missing {token!r}"


def test_rq4_is_an_open_question_not_started_not_authorized(protocol_new: str) -> None:
    assert _has(protocol_new, RQ4_QUESTION), "the RQ4 question is not registered verbatim"
    assert "NOT STARTED" in protocol_new and "NOT AUTHORIZED" in protocol_new
    assert "pre-result frozen mechanism amendment" in protocol_new
    candidates = (r"unmatched COCO fraction is the mechanism",
                  r"margin collapse is the mechanism",
                  r"H2a is the mechanism")
    _assert_no_unqualified_claim(protocol_new, candidates, "protocol")
    assert "candidate observations" in protocol_new


# ------------------------------------------------------------- documentation-only round

def test_close_out_describes_itself_as_documentation_only(log_new: str, protocol_new: str) -> None:
    assert "documentation" in protocol_new.lower()
    for doc in (log_new, protocol_new):
        assert "0 new training parameters" in doc or "new_training_parameters" in doc
    assert "不新增任何" in log_new or "no new result number" in log_new.lower()


def test_every_artifact_path_quoted_in_the_new_log_records_exists(log_new: str) -> None:
    missing = []
    for block in _artifact_path_blocks(log_new):
        for rel in _quoted_paths(block):
            if not _resolves(rel):
                missing.append(rel)
    assert not missing, f"artifact_paths quote non-existent artifacts: {sorted(set(missing))}"
