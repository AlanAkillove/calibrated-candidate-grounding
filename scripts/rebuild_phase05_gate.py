"""Rebuild ``sufficiency_gate.json`` from the frozen Phase 0.5 artifacts.

Why this script exists
----------------------
The first full Phase 0.5 run stored the gate cells' ``rer50_drop`` values as
*fractions* while ``ccg.reliability.evaluate.sufficiency_verdict`` (Amendment
A6.6) compares them against ``rer_drop_pp`` in *percentage points*; Route A was
therefore unreachable and the reported ``route_A_cells`` list was empty even
though the RER@50 drops (29-47pp) and E-AURC worsenings (129-285%) clearly
satisfy it.  The driver fix (``_rer_pp`` in ``scripts/run_phase05.py``) converts
the RER fields to pp before they enter a gate cell.

The gate is a pure post-processing step of the published artifacts
(``reliability_metrics.csv`` + ``paired_bootstrap.csv``), so instead of
re-running the 71-minute 5000-replicate bootstrap this script reconstructs the
exact inputs of ``ccg.experiment``-driven ``_gate`` from those artifacts,
verifies that the reconstruction reproduces the published
``cross_k_degradation.csv`` bit-for-bit, then rewrites
``sufficiency_gate.json`` with the fixed units.  Metrics, bootstrap rows and
the verdict itself are unaffected (the verdict was already GO via Route B).

Usage:
    python scripts/rebuild_phase05_gate.py [--out results/phase05_score_sufficiency]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg.experiment import phase0a  # noqa: E402

import run_phase05  # noqa: E402

SCORERS = ("b3_seed1", "b3_seed2", "b3_seed3", "cosine")
_INTS = {"K", "n", "params", "K_lo", "K_hi", "n_clusters", "n_replicates"}
_FLOATS = {
    "tune_mean_auroc", "auroc_correct", "aurc", "aurc_oracle", "e_aurc",
    "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95", "ece_adaptive",
    "brier_binary", "nll_binary", "diff", "ci_low", "ci_high", "mean_lo",
    "mean_hi", "mean_a", "mean_b", "worsening", "worsening_ci_low",
    "worsening_ci_high", "ci_level", "auroc_drop", "auroc_drop_ci_low",
    "auroc_drop_ci_high", "e_aurc_worsening", "e_aurc_ci_low", "e_aurc_ci_high",
    "rer_at_50_drop", "rer50_ci_low", "rer50_ci_high", "rer_at_80_drop",
    "e_aurc_abs_hi",
}


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out: Dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    out[key] = None
                elif key in _INTS:
                    out[key] = int(float(value))
                elif key in _FLOATS:
                    out[key] = float(value)
                else:
                    out[key] = value
            rows.append(out)
    return rows


def _close(a: Optional[float], b: Optional[float], tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= tol


def _verify_degradation(rows: List[Dict[str, Any]], published: List[Dict[str, Any]]) -> None:
    key = lambda r: (r["scorer"], r["model"], r["eval_split"], int(r["K_hi"]))  # noqa: E731
    by_key = {key(r): r for r in published}
    assert len(by_key) == len(published), "duplicate keys in published degradation csv"
    assert len(rows) == len(published), f"row count mismatch: {len(rows)} vs {len(published)}"
    fields = [f for f in run_phase05.DEGRADATION_FIELDS if f not in {"scorer", "model", "eval_split", "K_hi", "scorer_kind"}]
    for row in rows:
        ref = by_key.get(key(row))
        assert ref is not None, f"missing published row for {key(row)}"
        assert row["scorer_kind"] == ref["scorer_kind"], f"scorer_kind mismatch {key(row)}"
        for field in fields:
            assert _close(row.get(field), ref.get(field)), (
                f"{key(row)} field {field}: {row.get(field)} != {ref.get(field)}")
    print(f"reconstruction check PASS: {len(rows)} degradation rows match the published csv "
          f"(values identical within 1e-9)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("results/phase05_score_sufficiency"))
    args = parser.parse_args(argv)
    out: Path = args.out

    metric_rows = _read_rows(out / "reliability_metrics.csv")
    boot = _read_rows(out / "paired_bootstrap.csv")
    cross_k = [r for r in boot if r["bootstrap_kind"] == "cross_k"]
    gate_old = json.loads((out / "sufficiency_gate.json").read_text(encoding="utf-8"))

    boot_fields = {"cross_k": cross_k, "pairwise": [r for r in boot
                                                    if r["bootstrap_kind"] == "model_vs_model"]}
    best = {"b3": {"family": gate_old["best_family_b3"],
                   "summary": gate_old["best_summary_b3"],
                   "tune_scores_mean": gate_old["tune_scores_b3_mean"]},
            "per_scorer": gate_old["per_scorer_best"]}

    rebuilt = run_phase05._degradation_rows(list(SCORERS), metric_rows, cross_k)
    published = _read_rows(out / "cross_k_degradation.csv")
    _verify_degradation(rebuilt, published)

    gate = run_phase05._gate(list(SCORERS), metric_rows, boot_fields, best, print)
    phase0a._write_json(out / "sufficiency_gate.json", gate)
    print("verdict:", gate["verdict_seed_mean"]["verdict"],
          "| route_A:", gate["verdict_seed_mean"]["route_A_cells"],
          "| route_B:", gate["verdict_seed_mean"]["route_B_cells"])
    print("rewritten:", out / "sufficiency_gate.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
