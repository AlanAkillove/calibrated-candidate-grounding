"""Rebuild Phase 1F reliability features through its original construction path.

This read-only audit uses saved Phase 1F score matrices and candidate manifests,
materialises their frozen B3 cache embeddings with the original helper, calls
``run_phase1f._build_cell`` for feature construction, and predicts only through
the directly loaded A11 coefficient bundle. It writes its report under the B
repair output directory and never refits a model.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccg.models.b3_data import B3Corpus
from ccg.reliability import features as rfeat
from ccg.semantic import frozen as sfrozen
from ccg.semantic import frozen_load
from ccg.semantic import hard_scores as hscores
from ccg.semantic import features as sfeat

OUT = ROOT / "results/research_repair_v1/information"
PHASE1F = ROOT / "results/phase1f_hard_semantic"
FROZEN = ROOT / "results/phase1e_refcocog_external/frozen_models"
SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
CELLS = ("rand5", "hard5")
RAW_SCORE_STOP_ATOL = 1e-4


def load_phase1f():
    path = ROOT / "scripts/run_phase1f.py"
    spec = importlib.util.spec_from_file_location("repair_audit_phase1f", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def max_feature_delta(left: np.ndarray, right: np.ndarray, names: tuple[str, ...]) -> dict:
    delta = np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)), axis=0)
    return {
        "max_abs_by_feature": {name: float(value) for name, value in zip(names, delta, strict=True)},
        "max_abs_overall": float(delta.max(initial=0.0)),
        "n_features_nonzero_delta": int(np.count_nonzero(delta)),
    }


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    p1f = load_phase1f()
    bundle = frozen_load.load_models(FROZEN)
    # This audit is CPU-only and bounds BLAS to the authorized two threads.
    with threadpool_limits(limits=2):
        cohort = p1f.shard.load_hard_cohort(
            features_dir=ROOT / "results/phase05_score_sufficiency/features",
            manifests_root=ROOT / "cache/manifests",
        )
        corpus = B3Corpus(
            ROOT / "cache/features", ROOT / "cache/manifests",
            ROOT / "data/raw/refcoco+/refcoco+/refs(unc).p", ROOT / "cache/proposals.h5",
            image_sizes_path=ROOT / "cache/image_sizes.npz", ks=(5, 10), regime="random", preload=True,
        )
        cached_query_dtype = str(np.asarray(corpus._text(int(cohort.sentence_id[0]))).dtype)
        cached_region_dtype = str(np.asarray(corpus._region(int(cohort.image_id[0]))).dtype)
        variant_by_name = {variant.variant: variant for variant in p1f.VARIANTS}
        report = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "RUNNING",
            "scope": "Phase 1F rand5 and hard5 × three B3 seeds; exact original feature-construction path; direct loaded frozen bundle; no fit",
            "source_pipeline": {
                "runner": "scripts/run_phase1f.py::_variant_rows/_variant_samples/_cell_local/_build_cell",
                "feature_inputs": "frozen Phase 1F candidate manifests + saved Phase 1F float32 raw-score arrays + B3Corpus cached embeddings via hard_scores.materialise_examples",
                "score_dtype": "stored float32, promoted in _build_cell to float64 before Stats17 and semantic_stats",
                "embedding_dtype": {
                    "cached_query_embedding": cached_query_dtype,
                    "cached_candidate_embedding": cached_region_dtype,
                    "materialise_examples_query_output": "float32",
                    "materialise_examples_candidate_output": "float32",
                },
                "reliability": "FrozenExternalModels.predict loaded from phase1e_refcocog_external/frozen_models; no refitting",
                "source_bundle_evidence": "results/research_repair_v1/information/reference_frozen_bundle_verification.json",
            },
            "tolerance": float(sfrozen._TOL),
            "phase1f_raw_score_stop_atol": RAW_SCORE_STOP_ATOL,
            "diagnostic_attempts": [{
                "attempt": 1,
                "observed_failure": "AssertionError: rand5/b3_seed1 had sentence_id/ref_id/image_id/correct all identical, but exact raw_scores array_equal was false",
                "root_cause": "the first audit incorrectly required canonical Phase0B raw scores to be bitwise identical to Phase1F's saved scorer-forward float32 inputs",
                "correction": "rebuild Phase1F features from the exact saved Phase1F score rows and report canonical-vs-saved raw-score difference separately against A8.4 stop atol and ranking identity",
                "trace_persistence": "failure traceback was observed in the execution result; no dedicated log file existed for that first debug invocation",
            }],
            "thread_limit": 2,
            "cells": [],
        }
        original_apply = sfrozen.apply_frozen
        try:
            for cell_name in CELLS:
                cell = p1f.CELL_BY_NAME[cell_name]
                variant = variant_by_name[cell.source]
                variant_rows = p1f._variant_rows(cohort, variant)
                samples = p1f._variant_samples(cohort, variant, variant_rows)
                batch = hscores.materialise_examples(corpus, samples, cell.k, text_cache={})
                local = p1f._cell_local(cohort, variant_rows, cell)
                cell_report = {"cell": cell_name, "K": int(cell.k), "seeds": []}

                for seed in SEEDS:
                    ref_path = PHASE1F / "predictions" / f"{cell_name}__{seed}.npz"
                    with np.load(ref_path, allow_pickle=False) as archive:
                        ref = {key: np.asarray(archive[key]) for key in archive.files}
                    if variant.regime == "random":
                        canonical_scores = p1f.sdata.load_scorer_canonical(seed, cell.k)
                        positions = np.searchsorted(canonical_scores.sentence_id, cohort.sentence_id[variant_rows])
                        if not np.array_equal(canonical_scores.sentence_id[positions], cohort.sentence_id[variant_rows]):
                            raise AssertionError(f"{cell_name}/{seed}: random raw-score canonical rows do not cover variant")
                        score_full = np.asarray(canonical_scores.scores[positions], dtype=np.float32)
                    else:
                        # This hard variant's full row universe is exactly the saved cell.
                        score_full = np.asarray(ref["scores"], dtype=np.float32)
                    if score_full.shape != (variant_rows.size, cell.k):
                        raise AssertionError(f"{cell_name}/{seed}: source scores {score_full.shape} vs variant rows {variant_rows.size}")
                    source_selected_scores = np.asarray(score_full[local], dtype=np.float32)
                    saved_selected_scores = np.asarray(ref["scores"], dtype=np.float32)
                    score_source_delta = float(np.max(np.abs(
                        source_selected_scores.astype(np.float64) - saved_selected_scores.astype(np.float64)
                    )))
                    score_source_ranking_identical = bool(np.array_equal(
                        np.argsort(-source_selected_scores, axis=1, kind="stable"),
                        np.argsort(-saved_selected_scores, axis=1, kind="stable"),
                    ))
                    # Rebuild features from the exact saved Phase 1F float32
                    # scores; retain the Phase0B comparison above separately.
                    score_full = score_full.copy()
                    score_full[local] = saved_selected_scores
                    artifact = bundle.seed(seed)
                    fake_seed = SimpleNamespace(temperature=artifact.temperature)
                    # Reuse Phase 1F's original _build_cell implementation while
                    # replacing only its refit-model call with the loaded bundle.
                    sfrozen.apply_frozen = lambda _model, stats, sem, _seed=seed: bundle.predict(_seed, stats, sem)
                    data = p1f._build_cell(
                        cell, batch, score_full, local, cohort, variant_rows, fake_seed
                    )
                    sfrozen.apply_frozen = original_apply

                    identities = {}
                    for key in ("sentence_id", "ref_id", "image_id"):
                        identities[key] = bool(np.array_equal(getattr(data, key), ref[key]))
                    identities["correct"] = bool(np.array_equal(data.correct, np.asarray(ref["correct"], dtype=bool)))
                    if not all(identities.values()):
                        raise AssertionError(f"{cell_name}/{seed}: Phase 1F row identity/correctness mismatch: {identities}")

                    with np.load(OUT / "derived" / f"phase1f_{cell_name}_{seed}.npz", allow_pickle=False) as derived:
                        stats_delta = max_feature_delta(data.stats17, derived["stats17"], tuple(rfeat.stat_feature_names()))
                        sem_delta = max_feature_delta(data.sem16, derived["sem16"], tuple(sfeat.SEMANTIC_STAT_NAMES))
                    expected_stats = np.asarray(ref["conf_stats"], dtype=np.float64)
                    expected_e1b = np.asarray(ref["conf_e1b"], dtype=np.float64)
                    pred_stats, pred_e1b = bundle.predict(seed, data.stats17, data.sem16)
                    stats_pred_error = float(np.max(np.abs(pred_stats - expected_stats)))
                    e1b_pred_error = float(np.max(np.abs(pred_e1b - expected_e1b)))
                    cell_report["seeds"].append({
                        "seed": seed,
                        "n_rows": int(data.scores.shape[0]),
                        "row_identity": identities,
                        "stored_score_dtype": str(np.asarray(ref["scores"]).dtype),
                        "promoted_score_dtype": str(np.asarray(data.scores, dtype=np.float64).dtype),
                        "canonical_vs_saved_raw_score_max_abs": score_source_delta,
                        "canonical_vs_saved_raw_score_ranking_identical": score_source_ranking_identical,
                        "canonical_vs_saved_raw_score_check": (
                            "NOT_APPLICABLE_HARD_CANDIDATES" if variant.regime != "random" else
                            "PASS_WITHIN_STOP_ATOL" if score_source_delta <= RAW_SCORE_STOP_ATOL and score_source_ranking_identical else
                            "FAIL_STOP_ATOL_OR_RANKING"
                        ),
                        "feature_rebuild_score_source": "saved Phase1F scores; canonical Phase0B comparison is separate diagnostic",
                        "materialised_query_dtype": str(batch.z_q.dtype),
                        "materialised_candidate_dtype": str(batch.z_i.dtype),
                        "stats17_vs_B_derived": stats_delta,
                        "sem16_vs_B_derived": sem_delta,
                        "conf_stats_max_abs_vs_phase1f": stats_pred_error,
                        "conf_e1b_max_abs_vs_phase1f": e1b_pred_error,
                        "tolerance": float(sfrozen._TOL),
                        "prediction_check": "PASS" if stats_pred_error <= sfrozen._TOL and e1b_pred_error <= sfrozen._TOL else "FAIL",
                    })
                    del data
                cell_report["max_feature_delta_over_seeds"] = {
                    "stats17": max((s["stats17_vs_B_derived"]["max_abs_overall"] for s in cell_report["seeds"]), default=0.0),
                    "sem16": max((s["sem16_vs_B_derived"]["max_abs_overall"] for s in cell_report["seeds"]), default=0.0),
                }
                report["cells"].append(cell_report)
                del batch
            report["status"] = "PASS" if all(
                seed["prediction_check"] == "PASS"
                and all(seed["row_identity"].values())
                for cell in report["cells"] for seed in cell["seeds"]
            ) else "FAIL"
        finally:
            sfrozen.apply_frozen = original_apply
            del corpus

    report["seconds"] = round(time.time() - started, 2)
    target = OUT / "reference_phase1f_feature_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "seconds": report["seconds"], "path": str(target), "cells": [
        {"cell": cell["cell"], "max_stats17_feature_delta": cell["max_feature_delta_over_seeds"]["stats17"],
         "max_sem16_feature_delta": cell["max_feature_delta_over_seeds"]["sem16"],
         "conf_stats_max_abs": max(s["conf_stats_max_abs_vs_phase1f"] for s in cell["seeds"]),
         "conf_e1b_max_abs": max(s["conf_e1b_max_abs_vs_phase1f"] for s in cell["seeds"])}
        for cell in report["cells"]
    ]}, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
