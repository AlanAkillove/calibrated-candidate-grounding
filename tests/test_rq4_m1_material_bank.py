# Research Repair v1: legacy transcription is checked in its preserved appendix;
# current scientific conclusions are checked by test_repair_integration.py.
"""RQ4-M1 material-bank close-out: the paper-facing summary must transcribe the artifacts.

``docs/final_result_summary.md`` is the material bank the paper will be drafted from. This
module holds it to the same standard as the governance documents: every RQ4-M1 label and
number is **read back from the committed result artifacts** (``verdict.json``,
``point_decomposition.csv``, ``bootstrap_decomposition.csv``, ``transition_groups.csv``,
``matched_intersection_decomposition.csv``, ``report.md``, ``protocol_freeze.json``) and then
required to appear in the material bank. Nothing here hardcodes a decomposition value
independently, so a re-run that moves a number moves the test instead of passing silently.

Enforced properties:

* **faithful transcription** — the headline table, the bootstrap intervals, the transition-group
  rates and the matched paired gaps in the material bank equal the artifact values;
* **the frozen identity** — ``D_total = D_label + D_conf`` holds on the artifact and on the
  rendered values a reader would copy out of the table;
* **label discipline** — ``DECOMPOSITION_REPORTED`` and the three ``CONFIDENCE_CHANGE_HEAVIER``
  orderings are quoted as published, and none of the frozen forbidden wordings appears;
* **non-causal boundary** — every causal vocabulary use in the material bank sits inside an
  explicit negation or an out-of-scope declaration;
* **significance discipline** — the D_label intervals all straddle zero, so no claim that a label
  component is significant may be stated unqualified;
* **program freeze** — the axes are recorded as CLOSED, ``RQ4-M1 CLOSED`` included, with no
  RQ4-M2 active, and ``results/final_registry/`` was not regenerated for V2 or RQ4.

Reads committed files only, so it runs on a fresh clone.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RESULT = _ROOT / "results" / "v2_rq4_mechanism" / "m1_transition_confidence"
_PAPER = _ROOT / "docs" / "appendix" / "final_result_summary_pre_repair.md"
_PROTOCOL = _ROOT / "docs" / "research_protocol.md"
_REGISTRY = _ROOT / "results" / "final_registry"

FAMILIES = ("RPN", "DETR", "GDINO")

# The wording the user requires in the material bank; checked as normalized substrings so a
# line wrap or an emphasis marker cannot break the traceability test.
BOUNDARY = "exact statistical accounting decomposition, not a causal identification"
RANKING_STATISTIC = "ranking statistic, not an error-rate statistic"
MATCHED_CLAIM = "the confidence-heavy ordering survives matched-expression restriction"

PROGRAM_TOKENS = (
    "V2-G CLOSED",
    "V2-D CLOSED",
    "V2-M CLOSED",
    "V2-M3 CLOSED",
    "V2-P CLOSED",
    "RQ4-M1 CLOSED",
    "EXPERIMENTAL PROGRAM FROZEN",
    "No active experiment.",
    "out of scope / future work",
)

CAUSAL = re.compile(r"\bcaus(?:e|es|ed|ing|al|ality)\b|因果", re.IGNORECASE)
_Guards = (
    "not", "no ", "never", "without", "out of scope", "unanswer",
    "禁止", "不得", "不要", "未", "不是", "而非", "只可", "仅作", "仍开放", "尚未",
)


def _skip_without_results() -> None:
    if not (_RESULT / "verdict.json").exists():
        pytest.skip("RQ4-M1 result artifacts are not present in this checkout")


# --------------------------------------------------------------------------- artifact readers

def _json(name: str) -> dict:
    return json.loads((_RESULT / name).read_text(encoding="utf-8"))


def _rows(name: str) -> list[dict[str, str]]:
    with (_RESULT / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _mean3(name: str) -> dict[str, dict[str, float]]:
    """family -> float columns, restricted to the three-seed mean rows."""
    out: dict[str, dict[str, float]] = {}
    for row in _rows(name):
        if row.get("seed") != "mean3":
            continue
        out[row["family"]] = {
            k: float(v) for k, v in row.items()
            if k in ("A00", "A10", "A01", "A11", "D_total", "D_label", "D_conf",
                     "shapley_residual", "share_label", "share_conf") and v not in ("", None)
        }
    return out


@pytest.fixture(scope="module")
def paper() -> str:
    _skip_without_results()
    return _PAPER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def verdict() -> dict:
    _skip_without_results()
    return _json("verdict.json")


@pytest.fixture(scope="module")
def point() -> dict[str, dict[str, float]]:
    _skip_without_results()
    return _mean3("point_decomposition.csv")


# ------------------------------------------------------------------------------- normalization

def _norm(text: str) -> str:
    """Fold markup, line wraps and the typographic minus so an artifact value can be matched
    against prose that was typeset by hand."""
    text = text.replace("\u2212", "-")
    text = re.sub(r"[`*|>#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has(doc: str, needle: str) -> bool:
    return _norm(needle) in _norm(doc)


def _table_norm(text: str) -> str:
    """Keep the table pipes but fold emphasis markers and the typographic minus, so a bolded
    cell such as ``**-0.009145**`` can still be parsed as the number it transcribes."""
    text = text.replace("\u2212", "-")
    return re.sub(r"[*`]+", "", text)


def _governed(window: str) -> bool:
    low = window.lower()
    return any(guard in low for guard in _Guards)


# --------------------------------------------------------------------- labels and the boundary

def test_primary_verdict_and_classification_are_transcribed(paper: str, verdict: dict) -> None:
    assert verdict["primary_verdict"] == "DECOMPOSITION_REPORTED"
    assert _has(paper, verdict["primary_verdict"]), "material bank lost the primary verdict"
    assert _has(paper, verdict["classification"]), "material bank lost the freeze classification"


def test_all_three_families_are_confidence_change_heavier(paper: str, verdict: dict, point: dict) -> None:
    ordering = verdict["per_family_ordering"]
    assert set(ordering) == set(FAMILIES)
    labels = [ordering[f] for f in FAMILIES]
    assert labels.count("CONFIDENCE_CHANGE_HEAVIER") == 3, labels
    for family in FAMILIES:
        assert _has(paper, family), f"{family} is missing from the material bank"
    # the same label must also be the one the point artifact recorded per family
    rows = {r["family"]: r for r in _rows("point_decomposition.csv") if r["seed"] == "mean3"}
    for family in FAMILIES:
        assert rows[family]["ordering"] == ordering[family]
    assert paper.count("CONFIDENCE_CHANGE_HEAVIER") >= 3


def test_frozen_forbidden_wordings_are_absent_from_the_material_bank(paper: str, verdict: dict) -> None:
    for phrase in verdict["interpretation_boundary"]["forbidden"]:
        assert _norm(phrase) not in _norm(paper), f"material bank asserts a frozen forbidden claim: {phrase!r}"


def test_material_bank_carries_the_non_causal_boundary(paper: str) -> None:
    assert _has(paper, BOUNDARY), "the exact-accounting / not-causal boundary sentence is missing"
    offenders = []
    for m in CAUSAL.finditer(paper):
        window = paper[max(0, m.start() - 340):m.end() + 200]
        if not _governed(window):
            offenders.append(_norm(window[-260:]))
    assert not offenders, f"unqualified causal wording in the material bank: {offenders}"


# ------------------------------------------------------------------------- numbers and identity

def _table_rows(doc: str) -> dict[str, tuple[float, float, float]]:
    """Every headline table row in the material bank: family -> (D_total, D_label, D_conf)."""
    pattern = re.compile(
        r"^\|\s*(RPN|DETR|GDINO)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(-?\d+\.\d+)\s*\|\s*(-?\d+\.\d+)\s*\|",
        re.M)
    found: dict[str, list[tuple[float, float, float]]] = {}
    for family, total, label, conf in pattern.findall(_table_norm(doc)):
        found.setdefault(family, []).append((float(total), float(label), float(conf)))
    assert set(found) == set(FAMILIES), "the headline table does not cover all three families"
    for family, rows in found.items():
        assert len(rows) >= 2, f"{family}: expected the table in both the Chinese and English record"
        assert all(r == rows[0] for r in rows), f"{family}: the two headline tables disagree"
    return {family: rows[0] for family, rows in found.items()}


def test_headline_table_equals_the_point_artifact(paper: str, point: dict) -> None:
    table = _table_rows(paper)
    for family in FAMILIES:
        artifact = point[family]
        expected = (round(artifact["D_total"], 6), round(artifact["D_label"], 6),
                    round(artifact["D_conf"], 6))
        assert table[family] == expected, f"{family}: table {table[family]} != artifact {expected}"
        for key in ("D_total", "D_label", "D_conf", "A00", "A11"):
            assert f"{artifact[key]:.6f}" in paper, f"{family}: {key}={artifact[key]:.6f} not transcribed"


def test_material_bank_identity_survives_rounding(paper: str, point: dict) -> None:
    for family, (total, label, conf) in _table_rows(paper).items():
        assert abs(total - (label + conf)) <= 5e-7, f"{family}: rounded D_total != D_label + D_conf"
        artifact = point[family]
        assert abs(artifact["D_total"] - (artifact["D_label"] + artifact["D_conf"])) <= 1e-12
        assert artifact["shapley_residual"] <= 1e-12
    assert _has(paper, "D_total = D_label + D_conf")


def test_cross_family_gaps_match_the_gap_artifact(paper: str) -> None:
    gaps = {r["comparison"]: r for r in _rows("cross_family_gap_decomposition.csv")}
    assert set(gaps) == {"GDINO_minus_RPN", "GDINO_minus_DETR", "DETR_minus_RPN"}
    for comparison, row in gaps.items():
        total = float(row["gap_total_degradation"])
        label = float(row["gap_label_component"])
        conf = float(row["gap_conf_component"])
        assert abs(total - (label + conf)) <= 1e-12, comparison
        assert row["heavier_component"] == "CONFIDENCE_CHANGE_HEAVIER", comparison
        pretty = comparison.replace("_minus_", "\u2212")
        needle = f"{pretty} {total:+.6f} = {label:+.6f} + {conf:.6f}"
        assert _has(paper, needle), f"material bank does not carry the exact gap: {needle}"


def test_bootstrap_intervals_are_transcribed_with_the_frozen_sign_discipline(paper: str) -> None:
    rows = {(r["scope"], r["quantity"]): r for r in _rows("bootstrap_decomposition.csv")}
    for family in FAMILIES:
        conf = rows[(family, "D_conf")]
        assert float(conf["ci_low"]) > 0, f"{family}: D_conf no longer excludes zero; re-read the record"
        assert _has(paper, f"[{float(conf['ci_low']):.6f}, {float(conf['ci_high']):.6f}]"), \
            f"{family}: D_conf interval not transcribed"
        label = rows[(family, "D_label")]
        lo, hi = float(label["ci_low"]), float(label["ci_high"])
        assert lo < 0 < hi, f"{family}: the D_label interval no longer straddles zero"
        assert _has(paper, f"[{lo:.6f}, {hi:.6f}]"), f"{family}: D_label interval not transcribed"
    # a label component may only be discussed inside an explicit non-significance statement
    claim = re.compile(r"(?:D_label|标签分量)[^\n]{0,60}?(?:显著|significant)", re.I)
    offenders = [_norm(paper[max(0, m.start() - 300):m.end() + 120]) for m in claim.finditer(paper)
                 if not _governed(paper[max(0, m.start() - 340):m.end() + 200])]
    assert not offenders, f"a straddling D_label interval is described as significant: {offenders}"


def test_flip_rate_note_is_supported_by_the_transition_group_artifact(paper: str) -> None:
    groups = _rows("transition_groups.csv")
    gdino_f = [r for r in groups if r["family"] == "GDINO" and r["group"] == "F" and r["seed"] != "mean3"]
    assert len(gdino_f) == 3
    for row in gdino_f:
        pct = f"{float(row['rate']) * 100:.1f} %"
        assert _has(paper, pct) or _has(paper, pct.replace(" ", "")), \
            f"GDINO F-group rate {row['rate']} not quoted in the material bank"
    # the note claims the flipped rows sit in the same confidence band as the always-wrong rows
    for seed in {r["seed"] for r in gdino_f}:
        f50 = float(next(r["mean_p50"] for r in gdino_f if r["seed"] == seed))
        e50 = float(next(r["mean_p50"] for r in groups
                         if r["family"] == "GDINO" and r["group"] == "E" and r["seed"] == seed))
        assert abs(f50 - e50) < 0.01, f"{seed}: F and E no longer share a band; the note must be rewritten"
        assert _has(paper, f"{f50:.4f}") and _has(paper, f"{e50:.4f}")
    assert _has(paper, RANKING_STATISTIC), "the ranking-vs-error-rate point is missing"


def test_matched_gaps_equal_the_matched_artifact_components(paper: str) -> None:
    means = [r for r in _rows("matched_intersection_decomposition.csv") if r["level"] == "family_mean"]
    by_pair: dict[str, dict[str, dict[str, float]]] = {}
    for row in means:
        by_pair.setdefault(row["pair"], {})[row["family"]] = {
            k: float(row[k]) for k in ("D_total", "D_label", "D_conf")}
    required = {
        "GDINO_intersection_RPN": ("GDINO", "RPN", ("D_total", "D_label", "D_conf")),
        "GDINO_intersection_DETR": ("GDINO", "DETR", ("D_total", "D_label", "D_conf")),
        # the material bank records only the total for the pair whose CI straddles zero
        "DETR_intersection_RPN": ("DETR", "RPN", ("D_total",)),
    }
    for pair, (first, second, quantities) in required.items():
        assert pair in by_pair and {first, second} <= set(by_pair[pair]), f"{pair} missing"
        gaps = {k: by_pair[pair][first][k] - by_pair[pair][second][k]
                for k in ("D_total", "D_label", "D_conf")}
        assert abs(gaps["D_total"] - (gaps["D_label"] + gaps["D_conf"])) <= 1e-12, pair
        for key in quantities:
            assert _has(paper, f"{gaps[key]:+.6f}"), f"{pair}: {key} gap {gaps[key]:+.6f} not transcribed"
        for family in (first, second):
            assert by_pair[pair][family]["D_total"] > 0
    # DETR vs RPN must not be presented as a significant magnitude difference
    report = (_RESULT / "report.md").read_text(encoding="utf-8")
    row = next(line for line in report.splitlines() if line.startswith("| DETR_intersection_RPN | +0.0"))
    cells = [c.strip() for c in row.split("|")]
    lo, hi = (float(x) for x in re.findall(r"-?\d+\.\d+", cells[3]))
    assert lo < 0 < hi, "the DETR-vs-RPN paired gap no longer crosses zero; re-read the record"
    assert _has(paper, "跨 0") or _has(paper, "crosses zero")
    assert _has(paper, MATCHED_CLAIM)


# ------------------------------------------------------------------------- program freeze

def test_program_status_block_is_recorded_in_the_material_bank(paper: str) -> None:
    for token in PROGRAM_TOKENS:
        assert token in paper, f"material bank program status is missing {token!r}"
    assert "RQ4-M1 COMPLETED" in paper
    assert "NO RQ4-M2 PLANNED" in paper


def test_program_status_block_is_appended_to_the_protocol() -> None:
    doc = _PROTOCOL.read_text(encoding="utf-8")
    for token in PROGRAM_TOKENS:
        assert token in doc, f"protocol program status is missing {token!r}"


def test_no_rq4_m2_is_active_and_the_freeze_still_forbids_it(verdict: dict) -> None:
    freeze = _json("protocol_freeze.json")
    banned = [item for item in freeze["out_of_scope"]["explicitly_not_authorized"] if "RQ4-M2" in item]
    assert banned, "the freeze no longer lists RQ4-M2 as not authorized"
    assert any("intervention" in item for item in freeze["out_of_scope"]["explicitly_not_authorized"])
    assert verdict["out_of_scope_confirmed"], "the result verdict does not restate the out-of-scope items"


def test_final_registry_was_not_regenerated_for_rq4() -> None:
    assert _REGISTRY.is_dir(), "the mainline registry directory disappeared"
    for path in sorted(_REGISTRY.glob("*.csv")) + [_REGISTRY / "tables.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "rq4" not in text.lower(), f"{path.name} now carries RQ4 values"
        assert "m1_transition_confidence" not in text, f"{path.name} references the RQ4-M1 artifacts"
    assert "final_registry" in _PAPER.read_text(encoding="utf-8")


def test_material_bank_says_v2_and_rq4_values_come_from_per_axis_artifacts(paper: str) -> None:
    assert _has(paper, "has not been regenerated for V2")
    assert _has(paper, "results/v2_rq4_mechanism/m1_transition_confidence/")
