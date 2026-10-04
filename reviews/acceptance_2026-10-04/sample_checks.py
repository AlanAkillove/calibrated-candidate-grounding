"""Recompute scientific checks from predictions without repair estimators."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/research_repair_v1"
sys.path.insert(0, str(ROOT / "src"))
failures, checks = [], []


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def equal(actual, expected, tag, tolerance=1e-12):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    checks.append({"check": tag, "max_abs_error": error, "atol": tolerance})
    if not np.isfinite(error) or error > tolerance:
        failures.append(tag)


frame = pd.read_csv(OUT / "information/semantic_ablation_predictions.csv.gz")
table = pd.read_csv(OUT / "information/information_bootstrap.csv")
seeds = ("b3_seed1", "b3_seed2", "b3_seed3")
selected_replicates = (0, 17, 1024, 4999)
calls = load(OUT / "information/information_bootstrap.json")["calls"]
with np.load(OUT / "information/information_bootstrap_raw.npz", allow_pickle=False) as archive:
    for call in calls:
        scope, split = call["scope"], call["eval_split"]
        cells = {"random": [f"randomK{k}" for k in (5, 10, 20, 50)],
                 "matched_same4": ["rand5", "hard5"],
                 "dose_same8": [f"expb_m{m}" for m in (0, 2, 4, 8)]}[scope]
        subset = frame[frame.cell.isin(cells) & frame.eval_split.isin(["testA", "testB"])]
        if split != "__pooled_test__":
            subset = subset[subset.eval_split == split]
        data = {(seed, cell, group): subset[(subset.seed == seed) & (subset.cell == cell) & (subset.model == group)].reset_index(drop=True)
                for seed in seeds for cell in cells for group in ("S", "S+Q", "S+V", "Full")}
        ref = data[(seeds[0], cells[0], "S")]
        if len(ref) != call["n_rows"]:
            failures.append("call cohort size " + call["call_id"])
        ids = ref.image_id.to_numpy()
        # Independent implementation of first-seen image clusters. Every
        # expression, including multiple sentences for a ref, remains in its image.
        images = dict.fromkeys(ids.tolist())
        groups = [np.flatnonzero(ids == image) for image in images]
        rng = np.random.default_rng(0)
        rows_by_draw = {}
        for draw in range(5000):
            cluster_indices = rng.integers(0, len(groups), size=len(groups))
            if draw in selected_replicates:
                rows_by_draw[draw] = np.concatenate([groups[i] for i in cluster_indices])
        for cell in cells:
            estimates = table[(table.scope == scope) & (table.cell == cell) & (table.eval_split == split)
                              & (table.contrast == "Full_minus_S_plus_Q") & (table.metric == "auroc_correct")]
            if len(estimates) != 1:
                failures.append("primary estimate missing " + scope + cell + split)
                continue
            record = estimates.iloc[0]
            values = {}
            for draw, indices in [("point", np.arange(len(ref))), *rows_by_draw.items()]:
                per_seed = []
                for seed in seeds:
                    full, query = data[(seed, cell, "Full")], data[(seed, cell, "S+Q")]
                    for d in (full, query):
                        if not np.array_equal(d[["sentence_id", "ref_id", "image_id"]], ref[["sentence_id", "ref_id", "image_id"]]):
                            failures.append("identity alignment " + seed + cell + split)
                    if not np.array_equal(full.grounding_correct, query.grounding_correct):
                        failures.append("grounding changed " + seed + cell)
                    y = full.grounding_correct.to_numpy()[indices]
                    per_seed.append(roc_auc_score(y, full.probability.to_numpy()[indices]) - roc_auc_score(y, query.probability.to_numpy()[indices]))
                values[draw] = float(np.mean(per_seed))
            equal(values["point"], record.point, f"primary point {scope}/{cell}/{split}")
            equal([values[i] for i in selected_replicates], archive[record.raw_replicates_key][list(selected_replicates)],
                  f"fresh shared draws {scope}/{cell}/{split}")

protocol = load(OUT / "information/protocol.json")
split = load(ROOT / "results/phase05_score_sufficiency/split_manifest.json")
train_images, tune_images = set(split["train_images"]), set(split["tune_images"])
if train_images & tune_images:
    failures.append("train/tune image leakage")
full_names = protocol["feature_groups"]["Full"]
normalization_records = []
for seed in seeds:
    raw = {}
    identity = {}
    for k in (5, 10, 20, 50):
        with np.load(OUT / f"information/derived/random_{seed}_K{k}.npz", allow_pickle=False) as saved:
            raw[k] = np.column_stack([saved["stats17"], saved["sem16"]])
        with np.load(ROOT / f"results/phase0b_independent/seed_{seed[-1]}/raw_scores/K{k}.npz", allow_pickle=False) as saved:
            identity[k] = {name: saved[name] for name in ("sentence_id", "image_id", "eval_split")}
    test_images = set(identity[5]["image_id"][np.isin(identity[5]["eval_split"], ["testA", "testB"])].tolist())
    if test_images & (train_images | tune_images):
        failures.append("test image leakage " + seed)
    for group in ("S", "S+Q", "S+V", "Full", "largeS"):
        file = ("stats_logistic_largeK20_50.json" if group == "largeS" else "ablation_" + group.replace("+", "_plus_") + ".json")
        model = load(OUT / "information/models" / seed / file)
        names = protocol["feature_groups"]["S" if group == "largeS" else group]
        columns = [full_names.index(name) for name in names]
        ks = (20, 50) if group == "largeS" else (5, 10)
        train = np.vstack([raw[k][np.isin(identity[k]["image_id"], list(train_images))][:, columns] for k in ks])
        equal(train.mean(axis=0), model["normalization"]["mean"], f"train-only mean {seed}/{group}")
        equal(train.std(axis=0), model["normalization"]["std"], f"train-only std {seed}/{group}")
        if train.shape[0] != 7372 or train.shape[0] != model["n_train_rows"]:
            failures.append("unequal training budget " + seed + group)
        if group != "largeS":
            for k in (5, 10, 20, 50):
                pred = frame[(frame.seed == seed) & (frame.cell == f"randomK{k}") & (frame.model == group)]
                ids = identity[k]["sentence_id"]
                positions = pd.Index(ids).get_indexer(pred.sentence_id)
                if np.any(positions < 0):
                    failures.append("prediction not in frozen rows " + seed + group)
                x = raw[k][positions][:, columns]
                z = (x - np.asarray(model["normalization"]["mean"])) / (np.asarray(model["normalization"]["std"]) + 1e-8)
                computed = expit(z @ np.asarray(model["coefficients"]) + model["intercept"])
                equal(computed, pred.probability, f"saved linear-model predictions {seed}/{group}/K{k}")
        normalization_records.append({"seed": seed, "group": group, "ks": list(ks), "n_train": len(train)})

# Explicitly record uncovered edge cases without hiding them in suite PASS.
from ccg.repairs.statistics import metric_bundle
from ccg.metrics.selective import aurc, e_aurc, risk_coverage_curve
coverage, risk = risk_coverage_curve([0.8, 0.3], [1, 0])
finite_eaurc = e_aurc(aurc(coverage, risk), 0.5)
try:
    isolated_risk = {"result": metric_bundle([0.8, 0.3], [1, 0], metrics=("risk_at_50",))}
except Exception as error:
    isolated_risk = {"error": type(error).__name__, "detail": str(error)}
report = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks,
          "independent_sample_scope": "30 primary points plus 120 shared draw means recomputed from sample predictions using sklearn AUROC; no repair estimator called",
          "training_normalization": normalization_records, "isolated_risk_edge_case": isolated_risk,
          "finite_perfect_ranking_eaurc": finite_eaurc,
          "edge_cases_are_findings_not_test_failures": True}
(HERE / "sample_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(report["status"], len(checks), "checks;", len(failures), "failures; edge cases", isolated_risk, finite_eaurc)
raise SystemExit(0 if not failures else 1)
