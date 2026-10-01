#!/usr/bin/env python
"""V2-M M2.5 consistency audit: recompute the specialist audit and compare.

Recompute chain (no training, no fitting, no hyper-parameter search, no new
candidate sampling):

    frozen M1 confidences (``Random-K5__E1b``) and frozen M2 confidences
    (``m{m}__E1b``) are re-derived read-only on the frozen K=10 severity cells
    by re-running the M2.5 pipeline into a scratch directory:
    point metrics -> paired image-cluster bootstrap (5000 reps, seed 0, shared
    draws) -> specialist pattern verdict.

Every recomputed value is then compared against the committed M2.5 artifacts
(``m25_specialist_audit/point_metrics.csv`` / ``paired_bootstrap.csv`` /
``expert_disagreement.csv`` / ``calibration.csv`` / ``verdict.json``):
m0/m2/m4/m8 random expert AUROC, curriculum expert AUROC, deltas, paired CIs,
and the ``SPECIALIST TRADEOFF PRESENT`` verdict.

Writes ``results/v2_local_competition/m25_specialist_audit/m25_audit.json``;
exit code 0 = PASS.

Usage
-----
    python scripts/audit_v2m_m25.py
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

__all__ = ["main"]

TOLERANCE = 1e-10
M25_DIR = Path("results/v2_local_competition/m25_specialist_audit")
RECOMPUTE_DIR = M25_DIR / "_audit_recompute"
RECOMPUTE_LOG = Path("logs_v2m_m25_audit_recompute.txt")

POINT_FIELDS = (
    "b3_accuracy", "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80",
    "ece_adaptive", "brier_binary",
)
BOOT_FLOAT_FIELDS = ("diff", "ci_low", "ci_high", "mean_a", "mean_b", "ci_level")
BOOT_INT_FIELDS = ("n", "n_clusters", "n_replicates")
DISAGREE_FIELDS = ("mean_abs_diff", "median_abs_diff", "p90_abs_diff")
CALIB_FIELDS = ("mean_pred", "accuracy", "ece_adaptive", "brier_binary")


def _load_module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Audit:
    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {name}" + (f" ({detail})" if detail else ""), flush=True)


def _leaf_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(
        b, (int, float, np.integer, np.floating)
    ):
        return abs(float(a) - float(b)) <= TOLERANCE
    return a == b


def _tree_diffs(
    prefix: str, recomputed: Any, stored: Any, out: List[str], limit: int = 40
) -> None:
    """Collect mismatching leaves between two nested JSON-like structures."""
    if len(out) >= limit:
        return
    if isinstance(recomputed, Mapping):
        if not isinstance(stored, Mapping):
            out.append(f"{prefix}: type mismatch ({type(recomputed)} vs {type(stored)})")
            return
        rec_map = {str(key): value for key, value in recomputed.items()}
        sto_map = {str(key): value for key, value in stored.items()}
        keys = sorted(set(rec_map) | set(sto_map))
        for key in keys:
            step = f"{prefix}.{key}" if prefix else key
            if key not in rec_map:
                out.append(f"{step}: missing in recomputed")
            elif key not in sto_map:
                out.append(f"{step}: missing in stored")
            else:
                _tree_diffs(step, rec_map[key], sto_map[key], out, limit)
        return
    if isinstance(recomputed, (list, tuple)):
        if not isinstance(stored, (list, tuple)):
            out.append(f"{prefix}: type mismatch ({type(recomputed)} vs {type(stored)})")
            return
        if len(recomputed) != len(stored):
            out.append(f"{prefix}: length {len(recomputed)} vs {len(stored)}")
            return
        for index, (item_r, item_s) in enumerate(zip(recomputed, stored)):
            _tree_diffs(f"{prefix}[{index}]", item_r, item_s, out, limit)
        return
    if not _leaf_equal(recomputed, stored):
        out.append(f"{prefix}: {recomputed!r} != {stored!r}")


def _compare_rows(
    audit: Audit,
    name: str,
    rec_rows: Sequence[Mapping[str, Any]],
    sto_rows: Sequence[Mapping[str, Any]],
    *,
    key_fn: Any,
    float_fields: Sequence[str],
    int_fields: Sequence[str] = (),
) -> float:
    rec_map = {key_fn(r): r for r in rec_rows}
    sto_map = {key_fn(r): r for r in sto_rows}
    if set(rec_map) != set(sto_map):
        audit.check(
            name, False,
            f"row universe differs: recomputed={len(rec_map)} stored={len(sto_map)}",
        )
        return float("inf")
    faults = 0
    max_delta = 0.0
    for key, rec in rec_map.items():
        sto = sto_map[key]
        for field in float_fields:
            delta = abs(float(rec[field]) - float(sto[field]))
            if delta > TOLERANCE:
                faults += 1
            max_delta = max(max_delta, delta)
        for field in int_fields:
            if int(rec[field]) != int(sto[field]):
                faults += 1
    audit.check(name, faults == 0, f"rows={len(rec_map)} faults={faults} max|delta|={max_delta:.3e}")
    return max_delta


def main() -> int:
    started = time.perf_counter()
    print("V2-M M2.5 consistency audit")
    m25 = _load_module("run_v2m_m25", "scripts/run_v2m_m25.py")

    # ---- recompute (read-only chain; identical frozen inputs) ----------------
    print("[recompute]")
    print(f"  scratch dir: {RECOMPUTE_DIR}")
    t0 = time.perf_counter()
    with open(_REPO / RECOMPUTE_LOG, "w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
        rc = m25.main(
            [
                "--out-dir", str(RECOMPUTE_DIR),
                "--bootstrap-replicates", "5000",
                "--bootstrap-seed", "0",
                "--ci", "0.95",
            ]
        )
    if rc != 0:
        print(f"  recompute failed with rc={rc}; see {RECOMPUTE_LOG}")
        return 1
    print(f"  recompute done ({time.perf_counter() - t0:.1f}s; log {RECOMPUTE_LOG})")

    audit = Audit()
    print("[recomputation vs committed artifacts]")
    stored_dir = M25_DIR
    rec_dir = RECOMPUTE_DIR

    max_point = _compare_rows(
        audit,
        "point_metrics.csv reproduces (4 levels x 3 seeds x 2 experts)",
        _read_csv(rec_dir / "point_metrics.csv"),
        _read_csv(stored_dir / "point_metrics.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"]), r["model"]),
        float_fields=POINT_FIELDS,
        int_fields=("n", "n_images", "K"),
    )
    max_boot = _compare_rows(
        audit,
        "paired_bootstrap.csv reproduces (4 levels x 3 seeds x 4 metrics)",
        _read_csv(rec_dir / "paired_bootstrap.csv"),
        _read_csv(stored_dir / "paired_bootstrap.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"]), r["metric"]),
        float_fields=BOOT_FLOAT_FIELDS,
        int_fields=BOOT_INT_FIELDS + ("K",),
    )
    max_disagree = _compare_rows(
        audit,
        "expert_disagreement.csv reproduces (4 levels x 3 seeds)",
        _read_csv(rec_dir / "expert_disagreement.csv"),
        _read_csv(stored_dir / "expert_disagreement.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"])),
        float_fields=DISAGREE_FIELDS,
        int_fields=("n",),
    )
    max_calib = _compare_rows(
        audit,
        "calibration.csv reproduces (4 levels x 3 seeds x 2 experts)",
        _read_csv(rec_dir / "calibration.csv"),
        _read_csv(stored_dir / "calibration.csv"),
        key_fn=lambda r: (r["level"], int(r["seed"]), r["model"]),
        float_fields=CALIB_FIELDS,
    )

    # ---- verdict tree (severity deltas + CIs + pattern) ----------------------
    rec_verdict = json.loads((rec_dir / "verdict.json").read_text(encoding="utf-8"))
    sto_verdict = json.loads((stored_dir / "verdict.json").read_text(encoding="utf-8"))
    tree_out: List[str] = []
    _tree_diffs("", rec_verdict, sto_verdict, tree_out, limit=40)
    audit.check(
        "verdict.json tree reproduces (severity + pattern + conditions + envelope)",
        not tree_out,
        f"mismatches={len(tree_out)}" + (f" first={tree_out[0]}" if tree_out else ""),
    )

    print("[specialist headline]")
    for level in ("m0", "m2", "m4", "m8"):
        auroc = rec_verdict["severity"][level]["auroc_correct"]
        print(
            f"  {level}: delta={float(auroc['delta']):+.4f} "
            f"CI[{float(auroc['ci_low']):+.4f}, {float(auroc['ci_high']):+.4f}] "
            f"pos/neg={auroc['n_positive']}/{auroc['n_negative']}"
        )
    audit.check(
        f"pattern: stored={sto_verdict['pattern']} recomputed={rec_verdict['pattern']}",
        sto_verdict["pattern"] == rec_verdict["pattern"],
    )

    passed = all(entry["ok"] for entry in audit.checks)
    payload = {
        "artifact": "v2m_m25_audit",
        "passed": bool(passed),
        "recompute_dir": str(RECOMPUTE_DIR),
        "bootstrap": {
            "replicates": 5000, "seed": 0, "ci": 0.95,
            "metrics": ["auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"],
        },
        "specialist_pattern": rec_verdict["pattern"],
        "severity_headline": {
            level: rec_verdict["severity"][level]["auroc_correct"]
            for level in ("m0", "m2", "m4", "m8")
        },
        "max_deltas": {
            "point_metrics": max_point,
            "paired_bootstrap": max_boot,
            "expert_disagreement": max_disagree,
            "calibration": max_calib,
        },
        "verdict_tree_mismatches": tree_out,
        "checks": audit.checks,
        "runtime_sec": round(time.perf_counter() - started, 1),
    }
    out_path = M25_DIR / "m25_audit.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"V2M_M25_AUDIT_{'PASS' if passed else 'FAIL'} "
        f"checks={len(audit.checks)} failed={sum(1 for c in audit.checks if not c['ok'])} "
        f"max_point={max_point:.3e} max_boot={max_boot:.3e} "
        f"runtime={time.perf_counter() - started:.1f}s"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
