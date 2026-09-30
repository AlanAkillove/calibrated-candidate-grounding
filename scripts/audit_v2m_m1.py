#!/usr/bin/env python
"""V2-M M1 consistency audit: recompute the frozen M1 record from raw confidences.

Recompute chain (no model is trained, no score is regenerated):

    frozen B3 scores -> correct events -> per-seed ``confidences.npz``
    -> point metrics -> paired image-cluster bootstrap (5000 reps, seed 0,
    shared draws) -> M1 gate verdicts

and compare every recomputed value against the committed M1 artifacts
(``m1_random_only/point_metrics.csv`` / ``bootstrap/pairs.csv`` / ``gate.json``).

The checked comparisons (section-0 audit list of the M2 round):

* SameCategory-K5: LCR-E1b, LCR-Aggregate-MLP, LCR-LCR-noGate
* K20-random / K50-random: LCR-E1b
* Random-K5: LCR-E1b (the M1 random guard input)
* the rebuilt M1 gate verdicts

Writes ``results/v2_local_competition/m1_audit.json``; exit code 0 = PASS.

Usage
-----
    python scripts/audit_v2m_m1.py
"""

from __future__ import annotations

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

from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402

__all__ = ["main"]

TOLERANCE = 1e-10

AUDIT_PAIRS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "Random-K5": (("LCR", "E1b"),),
    "SameCategory-K5": (("LCR", "E1b"), ("LCR", "Aggregate-MLP"), ("LCR", "LCR-noGate")),
    "K20-random": (("LCR", "E1b"),),
    "K50-random": (("LCR", "E1b"),),
}


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_v2m_m1", _REPO / "scripts" / "run_v2m_m1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _abs(path: Any) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO / p


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

    def close(self, name: str, recomputed: float, stored: float) -> None:
        delta = abs(float(recomputed) - float(stored))
        self.check(name, delta <= TOLERANCE, f"max|delta|={delta:.3e}")


def main() -> int:
    started = time.perf_counter()
    runner = _load_runner()
    root = _abs(runner.OUT_ROOT)
    m1_dir = root / "m1_random_only"
    boot_csv = root / "bootstrap" / "pairs.csv"
    gate_path = root / "gate.json"
    point_csv = m1_dir / "point_metrics.csv"

    print("V2-M M1 consistency audit")
    print(f"  protocol sha256: {runner._sha256_file(_abs(runner.PROTOCOL_PATH))}")

    # ---- recomputed universes (clusters + correct events) -----------------
    cohort_hc = shard.load_hard_cohort(
        features_dir=_abs(runner.PHASE05_DIR), manifests_root=_abs(runner.MANIFESTS)
    )
    rows_same4 = np.flatnonzero(np.asarray(cohort_hc.masks["same4"], dtype=bool))
    hard_ids = np.asarray(cohort_hc.sentence_id[rows_same4], dtype=np.int64)

    universe: Dict[int, Dict[str, np.ndarray]] = {}
    for k in (5, 20, 50):
        store = sdata.load_embedding_store(k, out_root=_abs(runner.EMB_ROOT))
        universe[k] = {
            "sentence_id": np.asarray(store.sentence_id, dtype=np.int64),
            "image_id": np.asarray(store.image_id, dtype=np.int64),
            "eval_split": np.asarray(store.eval_split),
        }
        del store

    def _test_mask(k: int) -> np.ndarray:
        return np.isin(universe[k]["eval_split"], ("testA", "testB"))

    audit = Audit()
    print("[derivation]")
    audit.check(
        "hard5 row universe (9487 rows / 1424 images)",
        hard_ids.size == 9487 and np.unique(cohort_hc.image_id[rows_same4]).size == 1424,
        f"rows={hard_ids.size} images={np.unique(cohort_hc.image_id[rows_same4]).size}",
    )
    for k in (5, 20, 50):
        audit.check(
            f"K{k} test universe (10286 rows)",
            int(_test_mask(k).sum()) == 10286,
            f"rows={int(_test_mask(k).sum())}",
        )

    clusters: Dict[str, np.ndarray] = {
        "Random-K5": universe[5]["image_id"][_test_mask(5)],
        "SameCategory-K5": np.asarray(cohort_hc.image_id[rows_same4], dtype=np.int64),
        "K20-random": universe[20]["image_id"][_test_mask(20)],
        "K50-random": universe[50]["image_id"][_test_mask(50)],
    }
    correct_derived: Dict[Tuple[str, int], np.ndarray] = {}
    for seed in runner.SEEDS:
        ref5 = sdata.load_scorer_canonical(f"b3_seed{seed}", 5, b3_root=_abs(runner.B3_ROOT))
        pos5 = np.searchsorted(ref5.sentence_id, universe[5]["sentence_id"])
        audit.check(
            f"seed{seed} K5 store rows are a scorer subset",
            pos5.size == universe[5]["sentence_id"].size
            and np.array_equal(ref5.sentence_id[pos5], universe[5]["sentence_id"]),
        )
        correct5 = np.argmax(np.asarray(ref5.scores[pos5]), axis=1) == 0
        correct_derived[("Random-K5", seed)] = correct5[_test_mask(5)]
        store_sid = universe[5]["sentence_id"]
        matched_rows = np.searchsorted(store_sid, hard_ids)
        audit.check(
            f"seed{seed} same4 rows are a K5-store subset",
            np.array_equal(store_sid[matched_rows], hard_ids),
        )
        correct_derived[("Random-K5-matched", seed)] = correct5[matched_rows]
        del ref5
        for k, cohort in ((20, "K20-random"), (50, "K50-random")):
            refk = sdata.load_scorer_canonical(f"b3_seed{seed}", k, b3_root=_abs(runner.B3_ROOT))
            posk = np.searchsorted(refk.sentence_id, universe[k]["sentence_id"])
            audit.check(
                f"seed{seed} K{k} store rows are a scorer subset",
                posk.size == universe[k]["sentence_id"].size
                and np.array_equal(refk.sentence_id[posk], universe[k]["sentence_id"]),
            )
            correctk = np.argmax(np.asarray(refk.scores[posk]), axis=1) == 0
            correct_derived[(cohort, seed)] = correctk[_test_mask(k)]
            del refk
        pred_path = _abs(runner.HARD5_PREDICTIONS) / f"hard5__b3_seed{seed}.npz"
        with np.load(pred_path) as pred:
            pred_ids = np.asarray(pred["sentence_id"], dtype=np.int64)
            pred_scores = np.asarray(pred["scores"])
        audit.check(
            f"seed{seed} hard5 frozen predictions align with the same4 rows",
            np.array_equal(pred_ids, hard_ids),
        )
        correct_derived[("SameCategory-K5", seed)] = np.argmax(pred_scores, axis=1) == 0

    # ---- recompute point metrics + bootstrap rows -------------------------
    stored_point = {
        (r["cohort"], int(r["seed"]), r["model"]): r for r in _read_csv(point_csv)
    }
    stored_boot = {
        (r["cohort"], int(r["seed"]), f"{r['model_a']}-{r['model_b']}", r["metric"]): r
        for r in _read_csv(boot_csv)
    }
    point_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    max_point_delta = 0.0
    max_boot_delta = 0.0
    correct_mismatch = 0
    for seed in runner.SEEDS:
        z = np.load(m1_dir / f"seed_{seed}" / "confidences.npz")
        for cohort in runner.COHORT_ORDER:
            correct = np.asarray(z[f"{cohort}__correct"], dtype=np.uint8).astype(bool)
            derived = correct_derived[(cohort, seed)]
            if not (correct.shape == derived.shape and np.array_equal(correct, derived)):
                correct_mismatch += 1
            for model in runner.MODELS:
                conf = np.asarray(z[f"{cohort}__{model}"], dtype=np.float64)
                row = reval.point_metric_row(conf, correct, probability=conf)
                row.update(
                    {"cohort": cohort, "K": runner.COHORT_K[cohort], "seed": seed, "model": model}
                )
                point_rows.append(row)
                stored = stored_point[(cohort, seed, model)]
                for key in ("auroc_correct", "e_aurc", "rer_at_50"):
                    max_point_delta = max(
                        max_point_delta, abs(float(row[key]) - float(stored[key]))
                    )
            for model_a, model_b in AUDIT_PAIRS.get(cohort, ()):
                conf_a = np.asarray(z[f"{cohort}__{model_a}"], dtype=np.float64)
                conf_b = np.asarray(z[f"{cohort}__{model_b}"], dtype=np.float64)
                rows = reval.model_vs_model_bootstrap_row(
                    conf_a, correct, conf_b, correct, clusters[cohort],
                    eval_split=shard.POOLED, K=runner.COHORT_K[cohort],
                    model_a=model_a, model_b=model_b, metrics=runner.BOOT_METRICS,
                    replicates=runner.BOOT_REPLICATES, seed=runner.BOOT_SEED, ci=runner.BOOT_CI,
                )
                for row in rows:
                    key = (cohort, seed, f"{model_a}-{model_b}", row["metric"])
                    stored = stored_boot[key]
                    for field in ("diff", "ci_low", "ci_high", "mean_a", "mean_b"):
                        max_boot_delta = max(
                            max_boot_delta, abs(float(row[field]) - float(stored[field]))
                        )
                    if (int(row["n"]), int(row["n_clusters"])) != (
                        int(stored["n"]), int(stored["n_clusters"])
                    ):
                        max_boot_delta = float("inf")
                    row.update({"cohort": cohort, "seed": seed})
                    boot_rows.append(row)
        z.close()

    print("[recomputation vs stored artifacts]")
    audit.check(
        "confidences.npz correctness == frozen B3 events (all cohorts/seeds)",
        correct_mismatch == 0,
        f"mismatches={correct_mismatch}",
    )
    audit.check(
        "point metrics reproduce (auroc / e_aurc / rer50, 75 rows)",
        max_point_delta <= TOLERANCE,
        f"max|delta|={max_point_delta:.3e}",
    )
    audit.check(
        "paired bootstrap reproduces (8 pair-cohorts x 3 seeds x 3 metrics)",
        max_boot_delta <= TOLERANCE,
        f"max|delta|={max_boot_delta:.3e}",
    )

    # ---- rebuild the M1 gate ----------------------------------------------
    stored_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    rec_gate = runner._build_gate(point_rows, boot_rows, list(runner.SEEDS))
    print("[gate rebuild]")
    pairs = [
        ("M1_GO.verdict", stored_gate["gates"]["M1_GO"]["verdict"], rec_gate["gates"]["M1_GO"]["verdict"]),
        ("M1_STRONG.verdict", stored_gate["gates"]["M1_STRONG"]["verdict"], rec_gate["gates"]["M1_STRONG"]["verdict"]),
        (
            "structure_contribution.verdict",
            stored_gate["gates"]["structure_contribution"]["verdict"],
            rec_gate["gates"]["structure_contribution"]["verdict"],
        ),
        (
            "gate_contribution.significantly_better",
            stored_gate["gates"]["gate_contribution"]["significantly_better"],
            rec_gate["gates"]["gate_contribution"]["significantly_better"],
        ),
        (
            "k_generalization.no_systematic_degradation",
            stored_gate["gates"]["k_generalization"]["no_systematic_degradation"],
            rec_gate["gates"]["k_generalization"]["no_systematic_degradation"],
        ),
    ]
    for name, stored, recomputed in pairs:
        audit.check(f"{name}: stored={stored} recomputed={recomputed}", stored == recomputed)
    for key in ("delta_auroc", "delta_auroc_ci_low", "e_aurc_reduction", "rer50_gain_pp"):
        audit.close(
            f"M1_GO.{key}",
            rec_gate["gates"]["M1_GO"][key],
            stored_gate["gates"]["M1_GO"][key],
        )
    for key in ("delta_auroc_LCR_vs_AggregateMLP", "delta_auroc_LCR_vs_noGate"):
        section = (
            "structure_contribution" if "AggregateMLP" in key else "gate_contribution"
        )
        audit.close(
            f"{section}.{key}",
            rec_gate["gates"][section][key],
            stored_gate["gates"][section][key],
        )

    passed = all(entry["ok"] for entry in audit.checks)
    payload = {
        "artifact": "v2m_m1_audit",
        "passed": bool(passed),
        "protocol_sha256": runner._sha256_file(_abs(runner.PROTOCOL_PATH)),
        "bootstrap": {
            "replicates": runner.BOOT_REPLICATES,
            "seed": runner.BOOT_SEED,
            "ci": runner.BOOT_CI,
            "metrics": list(runner.BOOT_METRICS),
        },
        "max_point_delta": max_point_delta,
        "max_bootstrap_delta": max_boot_delta,
        "correctness_mismatches": correct_mismatch,
        "checks": audit.checks,
        "runtime_sec": round(time.perf_counter() - started, 1),
    }
    out_path = root / "m1_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        f"V2M_M1_AUDIT_{'PASS' if passed else 'FAIL'} "
        f"checks={len(audit.checks)} failed={sum(1 for c in audit.checks if not c['ok'])} "
        f"max_point_delta={max_point_delta:.3e} max_boot_delta={max_boot_delta:.3e} "
        f"runtime={time.perf_counter() - started:.1f}s"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
