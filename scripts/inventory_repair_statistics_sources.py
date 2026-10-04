"""Create a frozen-input inventory for the remaining Research Repair statistics scopes.

This is an input/identity audit only. It never runs a bootstrap, inference, or model
fit, and writes only under results/research_repair_v1/statistics/.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.repair_statistics as rs  # noqa: E402

OUT = ROOT / "results" / "research_repair_v1" / "statistics"
SOURCE_INDEX = OUT / "v2_source_inventory.json"
REPORT = OUT / "remaining_scope_inventory.json"


def _load_npz(rel: str) -> dict[str, np.ndarray]:
    return rs.load_npz(ROOT / rel)


def _same_identity(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], *,
                   ids_key: str = "sentence_id") -> bool:
    return all(np.array_equal(reference[k], candidate[k]) for k in (ids_key, "ref_id", "image_id"))


def _record_by_path(index: dict, rel: str) -> dict:
    for group in ("npz_artifacts", "summary_artifacts"):
        for item in index.get(group, []):
            if item.get("path") == rel:
                return item
    raise KeyError(f"source is absent from the prepared inventory: {rel}")


def _seed_paths(pattern: str, seeds=(1, 2, 3)) -> list[str]:
    return [pattern.format(seed=seed) for seed in seeds]


def audit() -> dict:
    index = json.loads(SOURCE_INDEX.read_text(encoding="utf-8"))
    frozen_files = []
    for group in ("npz_artifacts", "summary_artifacts"):
        for entry in index.get(group, []):
            rec = rs.frozen_source(ROOT / entry["path"])
            if rec["size_bytes"] != entry["size_bytes"] or rec["sha256"] != entry["sha256"]:
                raise ValueError(f"prepared inventory disagrees with frozen manifest: {entry['path']}")
            frozen_files.append(rec)

    v2g: dict[str, dict] = {}
    for bb in ("b1_phase0b", "b2_phase0b"):
        arrays = [_load_npz(f"results/v2_backbone_generalization/{bb}/seed_{seed}/per_sentence_predictions.npz")
                  for seed in (1, 2, 3)]
        ref = arrays[0]
        identity_keys = ("sentence_ids", "ref_ids", "image_ids", "eval_split")
        if any(not all(np.array_equal(ref[k], item[k]) for k in identity_keys) for item in arrays[1:]):
            raise ValueError(f"{bb}: seed prediction identity columns are not aligned")
        split = np.asarray(ref["eval_split"]).astype(str)
        split_counts = {str(key): int(value) for key, value in Counter(split.tolist()).items()}
        test_mask = np.isin(split, ("testA", "testB"))
        v2g[bb] = {
            "n_all_common_rows": int(split.size),
            "n_all_common_images": int(np.unique(ref["image_ids"]).size),
            "n_testA_testB_rows": int(test_mask.sum()),
            "n_testA_testB_images": int(np.unique(ref["image_ids"][test_mask]).size),
            "split_rows": split_counts,
            "seeds": [1, 2, 3],
            "identity_columns": ["sentence_ids", "ref_ids", "image_ids", "eval_split"],
            "seed_identity_status": "PASS",
        }
    v2g_g3_preflight = []
    for job in rs.build_v2g_g3_jobs():
        anchors = rs._verify_observed_point_anchors(job)
        v2g_g3_preflight.append({
            "job_id": job["job_id"], "n_rows": int(np.asarray(job["image_ids"]).size),
            "n_images": int(np.unique(job["image_ids"]).size), "n_seeds": len(job["seed_cells"]),
            "n_conditions": len(next(iter(job["seed_cells"].values()))),
            "n_estimates": len(job["estimates"]), "observed_anchor_count": len(anchors),
            "anchor_status": "PASS" if anchors and not job.get("legacy_anchor_unavailable") else
                "NEW_COHORT_NO_HISTORICAL_ANCHOR" if not anchors and job["cohort"] == "__pooled_test__" else "PARTIAL_OR_UNAVAILABLE",
            "legacy_cohort_classification": job["legacy_cohort_classification"],
        })

    d2: dict[str, dict] = {}
    d2_root = "results/v2_d2_refcoco_lang/predictions"
    d2_keys = ["sentence_id", "ref_id", "image_id"]
    for regime in ("random_k5", "random_k10", "random_k20", "random_k50", "hard_k5"):
        per_seed = [_load_npz(f"{d2_root}/d2__{regime}__b3_seed{seed}.npz") for seed in (1, 2, 3)]
        ref = per_seed[0]
        for seed, item in enumerate(per_seed[1:], start=2):
            if not all(np.array_equal(ref[k], item[k]) for k in d2_keys):
                raise ValueError(f"D2 {regime}: seed{seed} identity rows differ")
        sid = np.asarray(ref["sentence_id"], dtype=np.int64)
        if np.unique(sid).size != sid.size:
            raise ValueError(f"D2 {regime}: duplicate sentence_id")
        d2[regime] = {
            "n_rows": int(sid.size), "n_images": int(np.unique(ref["image_id"]).size),
            "unique_sentence_ids": True, "seed_identity_status": "PASS",
        }
    d2_random_sets = []
    for k in (5, 10, 20, 50):
        d2_random_sets.append(set(_load_npz(f"{d2_root}/d2__random_k{k}__b3_seed1.npz")["sentence_id"].tolist()))
    d2_common = set.intersection(*d2_random_sets)
    d2["random_common_k50"] = {"n_rows": len(d2_common), "status": "DERIVED_FROM_NATIVE_SENTENCE_IDS"}
    d2hard = set(_load_npz(f"{d2_root}/d2__hard_k5__b3_seed1.npz")["sentence_id"].tolist())
    d2rand = set(_load_npz(f"{d2_root}/d2__random_k5__b3_seed1.npz")["sentence_id"].tolist())
    d2["hard_random5_intersection"] = {"n_rows": len(d2hard & d2rand), "status": "NATIVE_ID_INTERSECTION_ONLY"}
    d2_c1_job = rs.build_d2_c1_jobs()[0]
    d2_c1_anchors = rs._verify_observed_point_anchors(d2_c1_job)
    d2["c1_observed_point_preflight"] = {
        "job_id": d2_c1_job["job_id"], "n_rows": int(np.asarray(d2_c1_job["image_ids"]).size),
        "n_images": int(np.unique(d2_c1_job["image_ids"]).size), "n_seeds": len(d2_c1_job["seed_cells"]),
        "n_conditions": len(next(iter(d2_c1_job["seed_cells"].values()))),
        "n_estimates": len(d2_c1_job["estimates"]), "observed_anchor_count": len(d2_c1_anchors),
        "legacy_anchor_unavailable": d2_c1_job.get("legacy_anchor_unavailable", []),
        "anchor_status": "CORE_POINT_ANCHORS_PASS_AUX_METRICS_NOT_IN_OLD_TABLE" if d2_c1_anchors else "FAIL",
    }

    clean_rel = "results/v2_data_robustness/d1_reviewed_annotations/clean_manifest.csv"
    clean = rs.read_csv(ROOT / clean_rel)
    clean_cohort = clean[clean["in_phase0b_cohort"].astype(str).str.lower() == "true"]
    if clean["sent_id"].duplicated().any():
        raise ValueError("D1 clean_manifest.csv has duplicate sent_id values")
    d1_counts = {str(key): int(value) for key, value in clean_cohort.groupby("split").size().to_dict().items()}
    d1 = {"clean_manifest_rows": int(len(clean)), "clean_in_phase0b_rows": int(len(clean_cohort)),
          "clean_rows_by_split": d1_counts, "unique_sentence_ids": True,
          "prediction_route": "Filter frozen Phase0A/Phase0B/Phase1F native-ID predictions by clean_manifest.sent_id; no model recovery."}

    proposal: dict[str, dict] = {}
    for family in ("RPN", "DETR", "GDINO"):
        path_base = "results/v2_proposal_robustness/predictions"
        by_k = {}
        sets = []
        for k in (5, 10, 20, 50):
            rel = f"{path_base}/p1__{family}__random_k{k}__b3_seed1.npz"
            arr = _load_npz(rel)
            sid = np.asarray(arr["sentence_id"], dtype=np.int64)
            if np.unique(sid).size != sid.size:
                raise ValueError(f"proposal {family} K{k}: duplicate sentence IDs")
            by_k[str(k)] = {"n_rows": int(sid.size), "n_images": int(np.unique(arr["image_id"]).size)}
            sets.append(set(sid.tolist()))
        intersection = set.intersection(*sets)
        hard_path = f"{path_base}/p1__{family}__hard_k5__b3_seed1.npz"
        has_hard = (ROOT / hard_path).is_file()
        hard_rows = None
        if has_hard:
            hard = _load_npz(hard_path)
            hard_rows = int(len(hard["sentence_id"]))
        proposal[family] = {
            "random_rows_by_k": by_k, "common_k50_rows": len(intersection),
            "has_hard_k5_archive": has_hard, "hard_k5_rows": hard_rows,
        }

    local: dict[str, dict] = {}
    m1_files = _seed_paths("results/v2_local_competition/m1_random_only/seed_{seed}/confidences.npz")
    m1_shapes = []
    for rel in m1_files:
        arrays = _load_npz(rel)
        m1_shapes.append({k: int(v.size) for k, v in arrays.items() if k.endswith("__correct")})
    local["M1_LCR"] = {
        "seed_archives": m1_files, "cohort_lengths_by_seed": m1_shapes,
        "native_sentence_id_in_archive": False,
        "prediction_route": "Bridge existing confidence arrays to exact sorted K5/K20/K50 scorer IDs and Phase1F hard5 IDs; verify per-row image_id/correct where archived and historical points before bootstrap.",
        "recovery_status": "EXISTING_CONFIDENCES_IDENTITY_BRIDGE_REQUIRED",
    }
    m1_preflight = []
    for job in rs.build_v2m_m1_jobs():
        anchors = rs._verify_observed_point_anchors(job)
        m1_preflight.append({
            "job_id": job["job_id"], "n_rows": int(np.asarray(job["image_ids"]).size),
            "n_images": int(np.unique(job["image_ids"]).size),
            "n_seeds": len(job["seed_cells"]), "n_conditions": len(next(iter(job["seed_cells"].values()))),
            "n_estimates": len(job["estimates"]), "observed_anchor_count": len(anchors),
            "anchor_status": "PASS" if anchors and not job.get("legacy_anchor_unavailable") else
                "CORE_FINITE_ANCHORS_PASS_AUX_METRICS_UNREPORTED" if anchors else "PARTIAL_OR_UNAVAILABLE",
            "sentence_id_status": job["sentence_id_status"], "evidence": job["evidence"],
        })
    local["M1_LCR"]["observed_point_preflight"] = m1_preflight
    local["M1_LCR"]["recovery_status"] = "VERIFIED_RUNNER_ORDER_AND_POINT_ANCHORS"
    m2_files = _seed_paths("results/v2_local_competition/m2_curriculum/seed_{seed}/confidences.npz")
    m2 = []
    for rel in m2_files:
        arrays = _load_npz(rel)
        row_lengths = {f"m{level}": int(arrays[f"m{level}__correct"].size) for level in (0, 2, 4, 8)}
        identity_status = True
        for level in (0, 2, 4, 8):
            if not np.array_equal(arrays[f"m{level}__image_id"], arrays["m0__image_id"]):
                identity_status = False
        m2.append({"path": rel, "rows_by_level": row_lengths,
                   "image_ids_match_across_levels": identity_status,
                   "native_sentence_id_in_archive": False})
    local["M2"] = {
        "seed_archives": m2,
        "prediction_route": "Use original runner's expb_m{0,2,4,8} frozen Phase1F source row order; compare image_id/correct arrays elementwise and recover sentence/ref IDs positionally only after those checks.",
        "recovery_status": "EXISTING_CONFIDENCES_SOURCE_ROW_BRIDGE_REQUIRED",
    }

    scopes = [
        {
            "scope_id": "V2G_G3_cardinality",
            "main_endpoint_or_gate": "Per-backbone native/global_T_corrected K5-to-K10/20/50 reliability; frozen G3 gate on K5-vs-K50.",
            "expected_jobs": [f"v2g_g3_{bb}_{cohort}" for bb in ("b1", "b2") for cohort in
                              ("val_select", "val_calib", "testA", "testB", "__all_common__", "__pooled_test__")],
            "expected_job_count": 12,
            "prediction_recovery": "Direct per_sentence_predictions.npz for both backbones × 3 seeds; native sentence/ref/image/split IDs; no model recovery.",
            "legacy_pool_identity": "Historical __pooled__ in seed bootstrap.csv has n=20,799 over val_select+val_calib+testA+testB (no train). New testA+testB pool n=10,286 is supplemental and is not anchored to old __pooled__.",
            "preflight": {"by_backbone": v2g, "observed_point_jobs": v2g_g3_preflight},
            "status": "READY_OBSERVED_POINT_ANCHORS_PASS_HISTORICAL_COHORTS_NEW_TEST_POOL_UNANCHORED",
            "original_method_note": "Old G3 gate reads frozen per-seed bootstrap.csv; rebootstrap from image clusters, keep transformed E-AURC worsening denominator K5 explicit.",
        },
        {
            "scope_id": "V2G_G4_hard_semantic",
            "main_endpoint_or_gate": "G4 hard5 semantic delta and hard-minus-random amplification, with frozen manipulation check and gate.",
            "expected_jobs": ["v2g_g4_b1", "v2g_g4_b2"], "expected_job_count": 2,
            "prediction_recovery": "No row-wise hard/random prediction archive is present; reconstruct with frozen B3 scorers, frozen PhaseA R1/R2 artifacts, and original hard/random candidate source; do not fit/retrain models.",
            "status": "FROZEN_FORWARD_REQUIRED_NOT_STARTED",
            "source_artifacts": ["results/v2_backbone_generalization/g4_phaseA", "results/v2_backbone_generalization/g4_phaseB", "scripts/run_v2g_hard.py"],
            "recovery_note": "Need separate frozen forward lane and repair-owned prediction NPZ before observed anchor preflight; aggregate CSVs alone cannot establish rowwise image-cluster bootstrap.",
        },
        {
            "scope_id": "D1_clean_robustness",
            "main_endpoint_or_gate": "C1 cardinality and C4 hard-semantic amplification on the frozen D1_clean subset.",
            "expected_jobs": ["d1_c1_clean", "d1_c4_clean"], "expected_job_count": 2,
            "prediction_recovery": d1["prediction_route"], "preflight": d1,
            "status": "DIRECT_NATIVE_ID_INPUTS_ADAPTER_NOT_REGISTERED",
            "source_artifacts": [clean_rel, "results/phase0a_corrected/per_sentence_predictions.npz",
                                 "results/phase0b_independent/seed_{1,2,3}/per_sentence_predictions.npz",
                                 "results/phase1f_hard_semantic/predictions/*.npz"],
            "original_method_note": "C1 includes frozen cosine B0 plus three frozen B3 seeds; C4 retains the Phase1F same4 mask filtered by clean sent_id.",
        },
        {
            "scope_id": "D2_language_robustness",
            "main_endpoint_or_gate": "C1 cardinality G3 replication and C4 hard-semantic G4 replication.",
            "expected_jobs": ["d2_c1_common_k50", "d2_c4_random_hard5"], "expected_job_count": 2,
            "prediction_recovery": "Direct native-ID D2 NPZs; random K intersection is derived from all K IDs; C4 is matched only by shared sentence IDs.",
            "preflight": d2,
            "status": "PARTIAL_C1_READY_C4_ADAPTER_PENDING",
            "endpoint_status": {
                "C1": "READY_CORE_POINT_AND_PAIRED_CONTRAST_ANCHORS_PASS",
                "C4": "NATIVE_ID_SOURCES_PRESENT_CONTINUOUS_MANIPULATION_CLUSTER_BOOTSTRAP_ADAPTER_PENDING",
            },
            "source_artifacts": [f"{d2_root}/d2__{regime}__b3_seed{{1,2,3}}.npz" for regime in
                                 ("random_k5", "random_k10", "random_k20", "random_k50", "hard_k5")],
            "original_method_note": "Preserve per-seed model effects first, then seed mean; apply D2 C1/C4 legacy gates after CI adequacy checks.",
        },
        {
            "scope_id": "Proposal_P1_P2",
            "main_endpoint_or_gate": "P1 F4 C1 cardinality for RPN/DETR, P2 C1 for GDINO, P1 F5 C4 hard-semantic for available RPN/DETR.",
            "expected_jobs": ["p1_c1_RPN", "p1_c1_DETR", "p2_c1_GDINO", "p1_c4_RPN", "p1_c4_DETR"],
            "expected_job_count": 5,
            "prediction_recovery": "Direct row-level per-family NPZs with sentence/ref/image IDs; derive each family common-K50 via exact sentence_id intersection and C4 only for families with hard_k5 archives.",
            "preflight": proposal,
            "status": "SOURCE_IDENTITY_PASS_ADAPTER_NOT_REGISTERED",
            "original_method_note": "RPN/DETR/GDINO are separate proposal-family cohorts. Do not transfer a missing hard5 cell across families; keep raw_top1/raw_margin AUROC secondary to the frozen gate.",
        },
        {
            "scope_id": "V2M_M1_LCR",
            "main_endpoint_or_gate": "M1 LCR vs E1b/Aggregate-MLP main GO/STRONG comparisons, matched hard cohort, and LCR-vs-noGate contribution.",
            "expected_jobs": ["m1_random_k5", "m1_random_k5_matched", "m1_samecategory_k5",
                              "m1_k20_random", "m1_k50_random"],
            "expected_job_count": 5,
            "prediction_recovery": local["M1_LCR"]["prediction_route"], "preflight": local["M1_LCR"],
            "status": "READY_CORE_POINT_ANCHORS_PASS_AUX_RER80_90_UNANCHORED",
            "source_artifacts": m1_files + ["results/phase0b_independent/seed_{1,2,3}/per_sentence_predictions.npz",
                                           "results/phase1f_hard_semantic/predictions/rand5__b3_seed*.npz",
                                           "results/phase1f_hard_semantic/predictions/hard5__b3_seed*.npz"],
            "original_method_note": "LCR is an M1 condition; retain original cohort-specific pair plans and separately report LCR-noGate contrasts.",
        },
        {
            "scope_id": "V2M_M2_curriculum",
            "main_endpoint_or_gate": "M2 m0/m2/m4/m8 severity-level comparisons and original m8 GO gate.",
            "expected_jobs": ["m2_curriculum_m0_m2_m4_m8"], "expected_job_count": 1,
            "prediction_recovery": local["M2"]["prediction_route"], "preflight": local["M2"],
            "status": "IDENTITY_BRIDGE_REQUIRED_NO_MODEL_RECOVERY_EXPECTED",
            "source_artifacts": m2_files + ["results/phase1f_hard_semantic/predictions/expb_m{0,2,4,8}__b3_seed*.npz"],
            "original_method_note": "Severity macro metrics must be computed inside each shared image-cluster replicate; original m8 gate rule and threshold remain unchanged.",
        },
        {
            "scope_id": "V2M_M25_specialist",
            "main_endpoint_or_gate": "M25 specialist-vs-generalist severity contrasts and frozen diagnostic verdict.",
            "expected_jobs": ["m25_specialist_m0_m2_m4_m8"], "expected_job_count": 1,
            "prediction_recovery": "No rowwise test predictions archived. Re-run only frozen M1/M2 model scoring on the original severity feature rows; no fit/retraining.",
            "status": "FROZEN_MODEL_FORWARD_REQUIRED_NOT_STARTED",
            "source_artifacts": ["results/v2_local_competition/m25_specialist_audit", "scripts/run_v2m_m25.py",
                                 "results/v2_local_competition/m1_random_only", "results/v2_local_competition/m2_curriculum"],
            "recovery_note": "Do not use published point/paired_bootstrap aggregates as a substitute for shared-draw raw predictions.",
        },
        {
            "scope_id": "V2M_M3_mixture",
            "main_endpoint_or_gate": "B0 exploratory; B1/B2 confirmatory severity-macro mixture effects and unchanged gate.",
            "expected_jobs": ["m3_b0_severity_macro", "m3_b1_severity_macro", "m3_b2_severity_macro"],
            "expected_job_count": 3,
            "prediction_recovery": "No rowwise mixture prediction archives; reproduce the deterministic frozen expert/mixture fitting and test scoring under the original parameters, then retain rowwise image IDs.",
            "status": "ORIGINAL_DETERMINISTIC_FIT_AND_FORWARD_REQUIRED_NOT_STARTED",
            "source_artifacts": ["results/v2_local_competition/m3_mixture", "scripts/run_v2m_m3_b0.py",
                                 "scripts/run_v2m_m3_conf.py", "scripts/audit_v2m_m3_b0.py", "scripts/audit_v2m_m3_conf.py"],
            "recovery_note": "Use original frozen data/model identities and deterministic training configuration; resource slot required before executing any fit/forward.",
            "original_method_note": "Compute severity macro within each image-cluster replicate. B0 remains exploratory and has no success gate; B1/B2 use the frozen gate only.",
        },
    ]
    return {
        "schema": "ccg.research_repair_v1.statistics.remaining_scope_inventory.v1",
        "audit_kind": "read_only_source_and_identity_inventory",
        "created_utc": rs.utc_now(),
        "frozen_manifest_verification": {
            "input_index": "results/research_repair_v1/statistics/v2_source_inventory.json",
            "npz_artifacts_verified": len(index.get("npz_artifacts", [])),
            "summary_artifacts_verified": len(index.get("summary_artifacts", [])),
            "unique_frozen_files_verified": len({x["path"] for x in frozen_files}),
            "status": "PASS",
        },
        "scopes": scopes,
    }


def main() -> int:
    payload = audit()
    rs.atomic_json(REPORT, payload)
    print(f"Wrote {REPORT.relative_to(ROOT)}; verified {payload['frozen_manifest_verification']['unique_frozen_files_verified']} frozen inputs.")
    for scope in payload["scopes"]:
        print(f"{scope['scope_id']}: {scope['status']} ({scope['expected_job_count']} anticipated jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
