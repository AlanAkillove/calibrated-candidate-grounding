"""V2-P program summary: traceability and overstatement tests.

`results/v2_proposal_robustness/v2_p_program_summary.md` is a *close-out index*: it is
allowed to restate published verdicts and published numbers, and nothing else. These
tests therefore check it against the artifacts rather than against prose taste:

* every verdict label and headline number it quotes equals the artifact it cites;
* every artifact path it mentions exists on disk;
* the withdrawn interpretation is recorded as withdrawn, with the same schema the P1
  amendment ledger uses;
* the boundary register is enforced on the document itself (it may not use the wordings
  this axis never licensed).

Reads committed artifacts only, so it runs on a fresh clone.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DIR = _ROOT / "results" / "v2_proposal_robustness"
_SUMMARY = _DIR / "v2_p_program_summary.md"
_AMENDMENTS = _DIR / "p2_amendment_history.csv"
_P1_AMENDMENTS = _DIR / "p1_amendment_history.csv"

FAMILIES = ("RPN", "DETR", "GDINO")


@pytest.fixture(scope="module")
def doc() -> str:
    if not _SUMMARY.exists():
        pytest.skip("V2-P program summary not written yet")
    return _SUMMARY.read_text(encoding="utf-8")


def _load(*parts: str) -> dict:
    return json.loads(_DIR.joinpath(*parts).read_text(encoding="utf-8"))


def _grouped(v: int) -> str:
    """10286 -> '10 286', the separator style used in the summary."""
    return f"{v:,}".replace(",", " ")


def _artifact(rel: str) -> dict:
    """Read a committed JSON artifact by repository-relative path."""
    return json.loads((_ROOT / rel).read_text(encoding="utf-8"))


def _require(doc: str, token: str) -> None:
    assert token in doc, f"summary does not quote {token!r}"


# ---------------------------------------------------------------- identity of the file


def test_summary_declares_itself_an_index_and_disambiguates_v2m(doc):
    assert "Status: axis closed" in doc
    assert "*index*, not a result" in doc
    # V2-P2-M (mechanism, this axis) must not be confused with the V2-M module.
    assert "Local Competition Reliability Module" in doc
    assert (_ROOT / "results" / "v2_local_competition").is_dir()


def test_summary_quotes_only_the_published_verdict_labels(doc):
    f5 = _load("p1_f5_c4", "f5_verdict.json")
    p1 = _load("p1_final_summary.json")
    c1 = _load("p2_c1_gdino", "p2_c1_verdict.json")
    pm = _load("p2_m_mechanism", "p2_m_verdict.json")

    label = f5["overall_p1_verdict"]["label"]
    assert label == p1["overall_p1_verdict"]["label"]
    _require(doc, label)
    assert f5["overall_p1_verdict"]["value"] == "YES"

    assert pm["classification"] == "DESCRIPTIVE_MECHANISM"
    _require(doc, pm["verdict"])
    for other in ("MECHANISM_COMPOSITION_LINKED", "MECHANISM_GAP_PERSISTS",
                  "MECHANISM_NOT_SUPPORTED"):
        assert other not in doc, f"{other} is not this axis's verdict"

    assert c1["gdino_C1_PROPOSAL_FAMILY_REPLICATED"] == "YES"
    assert c1["c1_verdicts"] == {"RPN": "YES", "DETR": "YES", "GDINO": "YES"}
    _require(doc, "gdino_C1_PROPOSAL_FAMILY_REPLICATED")
    assert "C4 verdict (GDINO): YES" not in doc


def test_summary_reports_the_frozen_rule_outcomes_verbatim(doc):
    pm = _load("p2_m_mechanism", "p2_m_verdict.json")
    a, b = pm["rule_a_within_family_rho"], pm["rule_b_matched_strata"]
    assert a["passed"] is True and b["passed"] is False
    assert a["families_passing"] == ["DETR", "GDINO"] and a["required"] == 2
    _require(doc, "rule a PASS 2/3, rule b FAIL")
    assert b["bins_dropped"] == 0 and b["shrinkage_min"] == 0.5
    assert pm["unknown_target_share"] == {f: 0.0 for f in FAMILIES}
    assert pm["new_training_parameters"] == 0 and pm["model_forward_passes"] == 0


# ---------------------------------------------------------------- headline numbers


def test_c1_headline_table_equals_the_frozen_verdicts(doc):
    c1 = _load("p2_c1_gdino", "p2_c1_verdict.json")
    expect = {
        "RPN": (0.7895, 0.4290),
        "DETR": (0.8806, 0.5849),
        "GDINO": (0.8463, 0.4448),
    }
    point = _mechanism_point()
    for family, row in c1["c1_by_family"].items():
        gate = row["gate"]
        _require(doc, f"{gate['auc_drop_mean']:+.4f}")
        _require(doc, f"{gate['eaurc_worsening_mean']:.3f}")
        _require(doc, f"{gate['rer50_drop_mean']:.4f}")
        _require(doc, _grouped(row["cohort_rows"]))
        # accuracies quoted in the same table come from mechanism_point.csv
        for key, token in zip(("accuracy_K5", "accuracy_K50"), expect[family]):
            value = point[(family, key)]
            assert round(value, 4) == token, family
            _require(doc, f"{token:.4f}")


def test_eaurc_worsening_is_a_relative_increase_not_a_multiplier(doc):
    """The Route-B quantity is (E-AURC_K50 - E-AURC_K5) / E-AURC_K5, averaged over seeds."""
    rows = list(csv.DictReader(
        (_DIR / "p2_c1_gdino" / "c1_point.csv").read_text(encoding="utf-8").splitlines()))
    by = {}
    for r in rows:
        by[(r["family"], r["seed"], int(r["K"]))] = r
    c1 = _load("p2_c1_gdino", "p2_c1_verdict.json")
    for family in FAMILIES:
        seeds = sorted({k[1] for k in by if k[0] == family and k[2] == 5})
        worsen = [
            (float(by[(family, s, 50)]["e_aurc"]) - float(by[(family, s, 5)]["e_aurc"]))
            / float(by[(family, s, 5)]["e_aurc"]) for s in seeds
        ]
        recomputed = sum(worsen) / len(worsen)
        published = c1["c1_by_family"][family]["gate"]["eaurc_worsening_mean"]
        assert abs(recomputed - published) < 1e-3, family
        # the multiplier reading of the same quantity would be ~3.26 / 5.55 / 9.65
        assert f"{published + 1:.3f}" not in doc
    assert "2.257x" not in doc and "8.654x" not in doc


def test_mechanism_numbers_equal_the_p2_m_artifacts(doc):
    pm = _load("p2_m_mechanism", "p2_m_verdict.json")
    for family in FAMILIES:
        cell = pm["rule_a_within_family_rho"]["per_family"][family]
        _require(doc, f"{cell['rho_mean_of_seeds']:+.4f}")
        _require(doc, f"[{cell['ci_low']:+.4f}, {cell['ci_high']:+.4f}]")
    point = _mechanism_point()
    for family, token in (("RPN", 11.51), ("DETR", 8.71), ("GDINO", 6.23)):
        assert round(point[(family, "R1_same_class_distractors")], 2) == token
        _require(doc, f"{token:.2f}")
    for family, token in (("RPN", 0.564), ("DETR", 0.611), ("GDINO", 0.718)):
        assert round(point[(family, "R2_unmatched_fraction")], 3) == token
        _require(doc, f"{token:.3f}")
    for family, token in (("RPN", 0.3605), ("DETR", 0.2957), ("GDINO", 0.4016)):
        assert round(point[(family, "H1c_net_harm")], 4) == token
        _require(doc, f"{token:.4f}")
    for family in FAMILIES:  # the structural identity of section P-09
        assert point[(family, "H1b_gained_flip")] == 0.0
    sh = pm["rule_b_matched_strata"]["shrinkage"]
    _require(doc, f"{sh['GDINO_minus_RPN']:.3f}")
    _require(doc, f"{sh['GDINO_minus_DETR']:.3f}")


def test_derived_ratios_are_arithmetic_of_published_means(doc):
    c1 = _load("p2_c1_gdino", "p2_c1_verdict.json")
    g = {f: c1["c1_by_family"][f]["gate"]["auc_drop_mean"] for f in FAMILIES}
    assert round(g["GDINO"] / g["DETR"], 2) == 2.78
    assert round(g["GDINO"] / g["RPN"], 2) == 4.58
    _require(doc, "GDINO/DETR = 2.78")
    _require(doc, "GDINO/RPN = 4.58")
    # the "~2.8x / ~4.6x" phrasing belongs to the final table, and is quoted as such
    final = (_DIR / "proposal_family_final_table.md").read_text(encoding="utf-8")
    assert "2.8x DETR, 4.6x RPN" in final


def test_secondary_intersection_numbers_match_the_verdict(doc):
    c1 = _load("p2_c1_gdino", "p2_c1_verdict.json")["intersections_secondary"]
    for pair, fam_a in (("GDINO_union_RPN", "RPN"), ("GDINO_union_DETR", "DETR")):
        blk = c1[pair]
        _require(doc, _grouped(blk["n_expressions"]))
        _require(doc, f"{blk[fam_a]['delta_auroc_K5_K50_mean']:.4f}")
        _require(doc, f"{blk['GDINO']['delta_auroc_K5_K50_mean']:.4f}")
        gap = [v for k, v in blk.items() if k.startswith("delta_auroc_K5_K50_gap")]
        _require(doc, f"+{gap[0]:.4f}")


def test_open_items_quote_real_artifact_values(doc):
    """Nothing in 'open and unclaimed' may be a number this axis never published."""
    rows = list(csv.DictReader(
        (_DIR / "p2_c1_gdino" / "c1_secondary_raw_auroc.csv").read_text(encoding="utf-8").splitlines()))
    vals = [float(r["raw_margin12_auroc"]) for r in rows
            if r["family"] == "GDINO" and r["K"] == "50"]
    assert round(sum(vals) / len(vals), 3) == 0.587
    _require(doc, "0.587")

    probe = json.dumps(_load("p2_gdino_probe", "probe_report.json"))
    assert "0.6892" in doc and "0.6892" in probe  # measured same-category supply
    assert "0.85" in doc  # the frozen supply threshold, from the P2-C1 freeze
    freeze = (_DIR / "p2_c1_gdino_config_freeze.json").read_text(encoding="utf-8")
    assert "0.6892 < 0.85" in freeze

    assoc = {"%s|%s|%s" % (r["family"], r["harm"], r["redundancy"]): r for r in csv.DictReader(
        (_DIR / "p2_m_mechanism" / "mechanism_association.csv").read_text(encoding="utf-8").splitlines())}
    gdino_r2 = assoc["GDINO|H1c_net_harm|R2_unmatched_fraction"]
    assert round(float(gdino_r2["rho_mean_of_seeds"]), 4) == 0.0036
    _require(doc, "+0.0036")
    gdino_r4 = assoc["GDINO|H2a_delta_margin|R4_frac_pairs_gt_mid"]
    assert round(float(gdino_r4["rho_mean_of_seeds"]), 3) == -0.251
    _require(doc, "-0.251")
    detr_r4 = assoc["DETR|H2a_delta_margin|R4_frac_pairs_gt_mid"]
    assert round(float(detr_r4["rho_mean_of_seeds"]), 3) == 0.175
    _require(doc, "+0.175")


def test_p1_frozen_artifact_hash_quoted_in_the_summary_is_the_real_one(doc):
    digest = hashlib.sha256((_DIR / "inference_report.json").read_bytes()).hexdigest()
    _require(doc, digest)
    assert f"{digest[:7]}..." in doc or digest[:7] in doc


# ---------------------------------------------------------------- paths and ledger


def test_every_cited_artifact_name_exists_on_disk(doc):
    names = set(re.findall(r"`([^`\s]+?\.(?:json|csv|md|py|png|h5))`", doc))
    assert names, "no artifact cited"
    present = set()
    for base in ("results", "tests", "scripts", "src", "docs"):
        present.update(p.name for p in (_ROOT / base).rglob("*") if p.is_file())
    missing = sorted(n for n in names if Path(n).name not in present)
    assert missing == [], f"summary cites artifacts that do not exist: {missing}"


def _amendment_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_amendment_ledger_uses_the_p1_schema_and_records_p2_a0(doc):
    if not _AMENDMENTS.exists():
        pytest.skip("P2 amendment ledger not written yet")
    header = _P1_AMENDMENTS.read_text(encoding="utf-8").splitlines()[0]
    assert _AMENDMENTS.read_text(encoding="utf-8").splitlines()[0] == header, \
        "the P2 ledger must use the same columns as the frozen P1 amendment history"
    rows = _amendment_rows(_AMENDMENTS)
    assert [r["amendment_id"] for r in rows] == ["P2-A0"]
    row = rows[0]
    assert row["linked_experiment"] == "V2-P2-M"
    assert row["modified_original_gates"] == "no"
    assert "post-result interpretation correction" in row["classification"]
    assert "87009ac" in row["gate_frozen_before_results"]
    _require(doc, "P2-A0")


def test_the_withdrawal_is_traceable_in_the_files_it_corrects(doc):
    line = [ln for ln in doc.splitlines() if "strongest evidence in the program so far" in ln]
    assert line, "the summary must quote the sentence it withdraws"
    assert "withdrawn" in line[0], "the quote may only appear inside the withdrawn ledger"

    gdino = (_DIR / "p2_c1_gdino" / "analysis_report.md").read_text(encoding="utf-8")
    assert "Superseded in part by V2-P2-M" in gdino
    assert "\n## 9." in gdino, "the addendum section must exist"
    final = (_DIR / "proposal_family_final_table.md").read_text(encoding="utf-8")
    assert "MECHANISM_PARTIAL" in final


def test_boundary_register_blocks_the_wordings_this_axis_never_licensed(doc):
    banned = [
        "proves that", "mechanism established", "cross-domain generalization",
        "detector leaderboard", "refutes the composition account", "GDINO is worse",
        "composition explains the amplification", "C1 does not replicate",
    ]
    found = [phrase for phrase in banned if phrase in doc]
    assert found == [], f"summary uses wording outside the axis boundary: {found}"
    for guard in ("No detector ranking", "No causal language", "No CI on the shrinkage ratio",
                  "No GDINO hard-regime / C4 statement"):
        assert guard in doc


def test_material_bank_now_states_the_axis_boundary_in_artifact_words(doc):
    """The preserved pre-repair material bank records the V2-P boundary, and every V2
    label the material bank carries equals the artifact that produced it (both language
    sections are checked, because the bank is written twice on purpose)."""
    paper = (_ROOT / "docs" / "appendix" / "final_result_summary_pre_repair.md").read_text(encoding="utf-8")

    # the stale limitation is gone from the bank, and the index no longer claims innocence
    for stale in ("single proposal family", "单一 proposal 家族"):
        assert stale not in paper, f"material bank still carries {stale!r}"
    assert "This file does not edit that document." not in doc
    _require(doc, "does not duplicate its numbers")

    labels = {
        "P1 overall": _load("p1_f5_c4", "f5_verdict.json")["overall_p1_verdict"]["label"],
        "D1": _artifact("results/v2_data_robustness/d1_reviewed_annotations/verdict.json")[
            "combined_verdict"],
        "D2": _artifact("results/v2_d2_refcoco_lang/c1_c4_verdict.json")["verdict"]["overall"],
        "M2.5": _artifact("results/v2_local_competition/m25_specialist_audit/verdict.json")["pattern"],
        "M3": _artifact(
            "results/v2_local_competition/m3_mixture/conf/cross_backbone_verdict.json")["verdict"],
    }
    for name, label in labels.items():
        assert label in paper, f"material bank does not carry the {name} label {label!r}"

    # the P2 extension label is published only in the final table, so read it from there
    final = (_DIR / "proposal_family_final_table.md").read_text(encoding="utf-8")
    third = re.search(r"P2 extension verdict: \*\*([A-Z0-9_]+)\*\*", final)
    assert third, "the final table must still publish its P2 extension verdict"
    assert third.group(1) in paper

    # backbone axis verdict and the GDINO scope gate, stated as their artifacts state them
    v2g = _artifact("results/v2_backbone_generalization/v2a1_result_record.json")
    assert v2g["verdicts"]["STRONG_CROSS_BACKBONE_GENERALITY"] == "YES"
    assert "STRONG_CROSS_BACKBONE_GENERALITY = YES" in paper
    assert "0.6892 < 0.85" in paper, "the GDINO same-category supply gate must stay non-claimable"

    # the withdrawal is registered on both sides of the bilingual document
    assert "已被撤回" in paper, "the Chinese section must state the withdrawal"
    assert "withdrawn" in paper, "the English section must state the withdrawal"


def _mechanism_point():
    rows = csv.DictReader(
        (_DIR / "p2_m_mechanism" / "mechanism_point.csv").read_text(encoding="utf-8").splitlines())
    return {(r["family"], r["measure"]): float(r["mean"]) for r in rows if r["mean"]}
