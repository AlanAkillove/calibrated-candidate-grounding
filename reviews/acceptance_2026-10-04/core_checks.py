"""Independent main-result, accounting, and candidate-identity spot checks."""
import csv
import json
import os
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/research_repair_v1"
HERE = Path(__file__).resolve().parent
failures, checks = [], []


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare(actual, expected, name, atol=1e-12):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    checks.append({"check": name, "max_abs_error": error, "atol": atol})
    if not np.isfinite(error) or error > atol:
        failures.append(name)


summary = load(OUT / "statistics/summary.json")
job = next(j for j in summary["jobs"] if j["job_id"] == "phase0b___pooled_test__")
estimate = next(e for e in job["estimates"] if e["name"] == "crossK_K5_minus_K50__global_T::auroc_correct")
samples = {}
for seed in (1, 2, 3):
    with np.load(ROOT / f"results/phase0b_independent/seed_{seed}/per_sentence_predictions.npz", allow_pickle=False) as z:
        selected = np.isin(z["eval_split"], ["testA", "testB"])
        indices = np.flatnonzero(selected)
        indices = indices[np.argsort(z["sentence_ids"][indices], kind="stable")]
        samples[seed] = {name: z[name][indices] for name in ("sentence_ids", "image_ids", "correct_K5", "correct_K50", "confidence_global_T_corrected_K5", "confidence_global_T_corrected_K50")}
reference = samples[1]
for seed in (2, 3):
    compare(samples[seed]["sentence_ids"], reference["sentence_ids"], f"main seed alignment {seed}", atol=0)
images = reference["image_ids"]
clusters = [np.flatnonzero(images == im) for im in dict.fromkeys(images.tolist())]
rng = np.random.default_rng(0)
chosen = (0, 17, 1024, 4999)
draws = {"point": np.arange(images.size)}
for i in range(5000):
    indices = rng.integers(0, len(clusters), size=len(clusters))
    if i in chosen:
        draws[i] = np.concatenate([clusters[j] for j in indices])
with np.load(ROOT / estimate["raw_replicates"], allow_pickle=False) as raw:
    for i, indices in draws.items():
        per_seed = []
        for seed, data in samples.items():
            difference = roc_auc_score(data["correct_K5"][indices], data["confidence_global_T_corrected_K5"][indices]) - roc_auc_score(data["correct_K50"][indices], data["confidence_global_T_corrected_K50"][indices])
            per_seed.append(difference)
            expected = estimate["seed_estimates"][f"b3_seed{seed}"] if i == "point" else raw[f"seed_b3_seed{seed}__{estimate['name']}"][i]
            compare(difference, expected, f"main AUROC gap seed{seed} draw{i}")
        expected = estimate["point"] if i == "point" else raw["aggregate__" + estimate["name"]][i]
        compare(np.mean(per_seed), expected, f"main AUROC gap mean draw{i}")

with np.load(OUT / "mechanism/bootstrap_raw_replicates.npz", allow_pickle=False) as raw:
    for family in ("RPN", "DETR", "GDINO"):
        for suffix in ("mean3", "seed1", "seed2", "seed3"):
            get = lambda name: raw[f"{family}__{name}__{suffix}"]
            compare(get("interaction_I"), get("A11") - get("A10") - get("A01") + get("A00"), f"interaction all5000 {family}/{suffix}")
            compare(get("D_total"), get("D_label") + get("D_confidence"), f"Shapley all5000 {family}/{suffix}")
            compare(get("L_at_p5") + get("C_at_r50"), get("A11") - get("A00"), f"first path all5000 {family}/{suffix}")
            compare(get("C_at_r5") + get("L_at_p50"), get("A11") - get("A00"), f"second path all5000 {family}/{suffix}")

audit_dir = OUT / "candidates/audit_run_20261003_Cslot_audit4"
with (audit_dir / "mismatch_by_expression.csv").open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
annotations = {str(a["id"]): a for a in load(ROOT / "data/raw/refcoco+/refcoco+/instances.json")["annotations"]}
for row in rows:
    annotation = annotations[str(row["ann_id"])]
    if int(annotation["category_id"]) != int(row["true_target_category_id"]) or int(annotation["image_id"]) != int(row["image_id"]):
        failures.append("true target annotation mismatch " + row["sentence_id"])
    expected = int(row["true_target_category_id"]) != int(row["target_proposal_assigned_category_id"])
    if expected != bool(int(row["category_mismatch"])):
        failures.append("category mismatch flag " + row["sentence_id"])
audit_units = {"family_expression_rows": len(rows), "unique_sentence_ids": len({r["sentence_id"] for r in rows}),
               "mismatch_rows": sum(int(r["category_mismatch"]) for r in rows),
               "mismatch_family_object_pairs": len({(r["family"], r["ann_id"]) for r in rows if r["category_mismatch"] == "1"})}
with (audit_dir / "source_candidate_availability.csv").open(encoding="utf-8-sig", newline="") as handle:
    for r in csv.DictReader(handle):
        expected = int(r["historical_candidate_available"]) and int(r["corrected_candidate_available"])
        if expected != int(r["in_old_new_common_intersection"]):
            failures.append("availability intersection flag " + str(r))
report = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks,
          "category_annotation_rows_checked": len(rows), "audit_units": audit_units,
          "scope": "main B3 AUROC gap point/four fresh draws; mechanism identities across every saved draw; target category IDs and availability flags"}
(HERE / "core_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(report["status"], len(checks), "numeric checks; units", audit_units)
raise SystemExit(0 if not failures else 1)
