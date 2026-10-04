"""Four-corner, two-path AUROC mechanism accounting for Research Repair v1."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from ccg.repairs.atomic import replace_with_retry
from ccg.metrics.discrimination import _average_ranks, auroc_correct
from ccg.repairs.bootstrap import assert_row_alignment, shared_image_cluster_bootstrap
from ccg.rq4 import decomposition as decomposition

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "research_repair_v1" / "mechanism"
PRED_DIR = ROOT / "results" / "v2_proposal_robustness" / "predictions"
SOURCE_DIR = ROOT / "results" / "v2_rq4_mechanism" / "m1_transition_confidence"
SOURCE_INPUT_MANIFEST = SOURCE_DIR / "input_artifact_manifest.csv"
C1_POINT_REFERENCE = ROOT / "results" / "v2_proposal_robustness" / "p2_c1_gdino" / "c1_point.csv"
FAMILIES = ("RPN", "DETR", "GDINO")
SEEDS = (1, 2, 3)
CORNER_NAMES = ("A00", "A10", "A01", "A11")
ESTIMATE_NAMES = (
    "A00", "A10", "A01", "A11",
    "L_at_p5", "L_at_p50", "C_at_r5", "C_at_r50",
    "interaction_I", "shapley_label_delta", "shapley_confidence_delta",
    "D_total", "D_label", "D_confidence", "shapley_identity_residual",
)
FORMULAS = {
    "A00": "AUROC(r5, p5)",
    "A10": "AUROC(r50, p5)",
    "A01": "AUROC(r5, p50)",
    "A11": "AUROC(r50, p50)",
    "L_at_p5": "A10 - A00",
    "L_at_p50": "A11 - A01",
    "C_at_r5": "A01 - A00",
    "C_at_r50": "A11 - A10",
    "interaction_I": "A11 - A10 - A01 + A00 = L_at_p50 - L_at_p5 = C_at_r50 - C_at_r5",
    "shapley_label_delta": "0.5 * (L_at_p5 + L_at_p50)",
    "shapley_confidence_delta": "0.5 * (C_at_r5 + C_at_r50)",
    "D_total": "A00 - A11",
    "D_label": "-shapley_label_delta",
    "D_confidence": "-shapley_confidence_delta",
    "shapley_identity_residual": "D_total - D_label - D_confidence",
}


class MechanismAuditError(RuntimeError):
    """Raised when frozen prediction identity or reproduction checks fail."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [repair-mechanism] {message}", flush=True)


def _validate_frozen_manifest() -> list[dict[str, Any]]:
    if not SOURCE_INPUT_MANIFEST.is_file():
        raise MechanismAuditError(f"frozen mechanism input manifest missing: {SOURCE_INPUT_MANIFEST}")
    entries: list[dict[str, Any]] = []
    with SOURCE_INPUT_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["path"]
            if not path.is_file():
                raise MechanismAuditError(f"frozen mechanism input is missing: {row['path']}")
            actual_size = int(path.stat().st_size)
            actual_hash = _sha256(path)
            if actual_size != int(row["file_size"]) or actual_hash != row["sha256"]:
                raise MechanismAuditError(
                    f"frozen mechanism input identity changed: {row['path']} "
                    f"size={actual_size} sha256={actual_hash}"
                )
            entries.append({
                "path": row["path"], "file_size": actual_size, "sha256": actual_hash,
                "role": row["role"], "family": row["family"], "K": row["K"], "seed": row["seed"],
            })
    prediction_paths = {row["path"] for row in entries if row["role"] == "prediction_artifact"}
    expected = {
        f"results/v2_proposal_robustness/predictions/p1__{family}__random_k{k}__b3_seed{seed}.npz"
        for family in FAMILIES for k in (5, 50) for seed in SEEDS
    }
    if prediction_paths != expected:
        raise MechanismAuditError(
            f"frozen manifest prediction scope differs: missing={sorted(expected - prediction_paths)}, "
            f"unexpected={sorted(prediction_paths - expected)}"
        )
    if C1_POINT_REFERENCE.relative_to(ROOT).as_posix() not in {row["path"] for row in entries}:
        raise MechanismAuditError("published C1 point reference is not covered by the frozen manifest")
    return entries


def derive_paths_from_corners(
    a00: float, a10: float, a01: float, a11: float
) -> Dict[str, float]:
    """Return the two signed paths, interaction and Shapley identity from four corners."""
    corners = {"A00": float(a00), "A10": float(a10), "A01": float(a01), "A11": float(a11)}
    nan = float("nan")

    def difference(minuend: str, subtrahend: str) -> float:
        left, right = corners[minuend], corners[subtrahend]
        return left - right if math.isfinite(left) and math.isfinite(right) else nan

    l_p5 = difference("A10", "A00")
    l_p50 = difference("A11", "A01")
    c_r5 = difference("A01", "A00")
    c_r50 = difference("A11", "A10")
    shapley_label = 0.5 * (l_p5 + l_p50) if math.isfinite(l_p5) and math.isfinite(l_p50) else nan
    shapley_confidence = 0.5 * (c_r5 + c_r50) if math.isfinite(c_r5) and math.isfinite(c_r50) else nan
    d_total = difference("A00", "A11")
    d_label = -shapley_label if math.isfinite(shapley_label) else nan
    d_confidence = -shapley_confidence if math.isfinite(shapley_confidence) else nan
    residual = (
        d_total - d_label - d_confidence
        if all(math.isfinite(value) for value in (d_total, d_label, d_confidence)) else nan
    )
    interaction = (
        corners["A11"] - corners["A10"] - corners["A01"] + corners["A00"]
        if all(math.isfinite(value) for value in corners.values()) else nan
    )
    return {
        **corners,
        "L_at_p5": l_p5,
        "L_at_p50": l_p50,
        "C_at_r5": c_r5,
        "C_at_r50": c_r50,
        "interaction_I": interaction,
        "shapley_label_delta": shapley_label,
        "shapley_confidence_delta": shapley_confidence,
        "D_total": d_total,
        "D_label": d_label,
        "D_confidence": d_confidence,
        "shapley_identity_residual": residual,
    }


def _rank_auc(ranks: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    positive = labels == 1.0
    n_pos = int(positive.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def four_corner_paths(
    p5: Any, r5: Any, p50: Any, r50: Any
) -> Dict[str, float]:
    """Compute all corners by ranking each frozen confidence vector once per draw."""
    confidence5 = np.asarray(p5, dtype=np.float64).reshape(-1)
    labels5 = np.asarray(r5, dtype=np.float64).reshape(-1)
    confidence50 = np.asarray(p50, dtype=np.float64).reshape(-1)
    labels50 = np.asarray(r50, dtype=np.float64).reshape(-1)
    if not (confidence5.size == labels5.size == confidence50.size == labels50.size):
        raise ValueError("four-corner inputs must have identical row counts")
    if confidence5.size == 0 or not all(
        np.all(np.isfinite(values)) for values in (confidence5, confidence50, labels5, labels50)
    ):
        raise ValueError("four-corner inputs must be non-empty and finite")
    if any(not np.all((labels == 0.0) | (labels == 1.0)) for labels in (labels5, labels50)):
        raise ValueError("correctness labels must be binary")
    ranks5 = _average_ranks(confidence5)
    ranks50 = _average_ranks(confidence50)
    return derive_paths_from_corners(
        _rank_auc(ranks5, labels5), _rank_auc(ranks5, labels50),
        _rank_auc(ranks50, labels5), _rank_auc(ranks50, labels50),
    )


def _read_prediction(path: Path) -> Dict[str, np.ndarray]:
    required = ("sentence_id", "ref_id", "image_id", "correct", "conf_msp")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise MechanismAuditError(f"{path}: missing columns {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    n = arrays["sentence_id"].size
    if any(value.ndim != 1 or value.size != n for value in arrays.values()):
        raise MechanismAuditError(f"{path}: identity/prediction arrays are not aligned vectors")
    order = np.argsort(arrays["sentence_id"].astype(np.int64), kind="stable")
    arrays = {name: value[order] for name, value in arrays.items()}
    ids = arrays["sentence_id"].astype(np.int64)
    if np.unique(ids).size != ids.size:
        raise MechanismAuditError(f"{path}: sentence_id is not unique")
    arrays["sentence_id"] = ids
    arrays["ref_id"] = arrays["ref_id"].astype(np.int64)
    arrays["image_id"] = arrays["image_id"].astype(np.int64)
    arrays["correct"] = arrays["correct"].astype(np.float64)
    arrays["conf_msp"] = arrays["conf_msp"].astype(np.float64)
    if not np.all(np.isfinite(arrays["conf_msp"])):
        raise MechanismAuditError(f"{path}: confidence contains non-finite values")
    if not np.all((arrays["correct"] == 0.0) | (arrays["correct"] == 1.0)):
        raise MechanismAuditError(f"{path}: correctness is not binary")
    return arrays


def _read_c1_point_reference() -> Dict[tuple[str, str, int], Dict[str, float]]:
    rows: Dict[tuple[str, str, int], Dict[str, float]] = {}
    with C1_POINT_REFERENCE.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") != "global_T_corrected" or not str(row.get("seed", "")).startswith("seed"):
                continue
            key = (row["family"], row["seed"], int(row["K"]))
            rows[key] = {"n": float(row["n"]), "auroc_correct": float(row["auroc_correct"])}
    return rows


def load_family_cohort(
    family: str,
    reference: Mapping[tuple[str, str, int], Mapping[str, float]],
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, Dict[str, Any]]:
    """Load the frozen common-K50 cohort and verify every seed/reference anchor."""
    per_seed: list[dict[str, np.ndarray]] = []
    canonical: Dict[str, Dict[str, np.ndarray]] = {}
    common_ids: np.ndarray | None = None
    image_ids: np.ndarray | None = None
    for seed in SEEDS:
        tag = f"seed{seed}"
        k5 = _read_prediction(PRED_DIR / f"p1__{family}__random_k5__b3_seed{seed}.npz")
        k50 = _read_prediction(PRED_DIR / f"p1__{family}__random_k50__b3_seed{seed}.npz")
        if seed == SEEDS[0]:
            canonical = {"K5": k5, "K50": k50}
            ids50 = k50["sentence_id"]
            k5_positions = np.searchsorted(k5["sentence_id"], ids50)
            if np.any(k5_positions >= k5["sentence_id"].size) or not np.array_equal(
                k5["sentence_id"][k5_positions], ids50
            ):
                raise MechanismAuditError(f"{family}: K50 sentence IDs are not a subset of K5")
            k5_sub = {name: values[k5_positions] for name, values in k5.items()}
            assert_row_alignment(
                k50, k5_sub, required_fields=("ref_id", "image_id"),
                reference_name=f"{family}/K50/seed1", candidate_name=f"{family}/K5-on-K50/seed1",
            )
            common_ids = ids50.copy()
            image_ids = k50["image_id"].copy()
        else:
            assert_row_alignment(
                canonical["K5"], k5, required_fields=("ref_id", "image_id"),
                reference_name=f"{family}/K5/seed1", candidate_name=f"{family}/K5/{tag}",
            )
            assert_row_alignment(
                canonical["K50"], k50, required_fields=("ref_id", "image_id"),
                reference_name=f"{family}/K50/seed1", candidate_name=f"{family}/K50/{tag}",
            )
        assert common_ids is not None and image_ids is not None
        if not np.array_equal(k50["sentence_id"], common_ids):
            raise MechanismAuditError(f"{family}/{tag}: K50 sentence_id differs from seed1")
        k5_pos = np.searchsorted(k5["sentence_id"], common_ids)
        if np.any(k5_pos >= k5["sentence_id"].size) or not np.array_equal(
            k5["sentence_id"][k5_pos], common_ids
        ):
            raise MechanismAuditError(f"{family}/{tag}: K50 rows are not nested inside K5")
        k5_common = {name: values[k5_pos] for name, values in k5.items()}
        assert_row_alignment(
            k50, k5_common, required_fields=("ref_id", "image_id"),
            reference_name=f"{family}/K50/{tag}", candidate_name=f"{family}/K5-common/{tag}",
        )
        if not np.array_equal(k50["image_id"], image_ids):
            raise MechanismAuditError(f"{family}/{tag}: image IDs differ from seed1")
        labels = decomposition.transition_groups(k5_common["correct"], k50["correct"])
        decomposition.check_structural_invariant(labels, family=family, seed=tag)
        for k, confidence, correct in (
            (5, k5_common["conf_msp"], k5_common["correct"]),
            (50, k50["conf_msp"], k50["correct"]),
        ):
            observed = auroc_correct(confidence, correct)
            anchor = reference[(family, tag, k)]
            if int(anchor["n"]) != common_ids.size:
                raise MechanismAuditError(
                    f"{family}/{tag}/K{k}: C1 reference n={anchor['n']} != cohort {common_ids.size}"
                )
            if not np.isfinite(observed) or abs(observed - float(anchor["auroc_correct"])) > 1e-12:
                raise MechanismAuditError(
                    f"{family}/{tag}/K{k}: AUROC {observed:.16g} does not reproduce "
                    f"the C1 anchor {anchor['auroc_correct']:.16g}"
                )
        per_seed.append({
            "sentence_id": common_ids.copy(), "image_id": image_ids.copy(),
            "r5": k5_common["correct"], "p5": k5_common["conf_msp"],
            "r50": k50["correct"], "p50": k50["conf_msp"],
        })
    assert common_ids is not None and image_ids is not None
    groups = [decomposition.transition_groups(row["r5"], row["r50"]).counts for row in per_seed]
    metadata = {
        "family": family, "n_rows": int(common_ids.size),
        "n_images": int(np.unique(image_ids).size), "transition_counts_by_seed": groups,
        "k5_prediction_rows_seed1": int(canonical["K5"]["sentence_id"].size),
        "k50_prediction_rows_seed1": int(canonical["K50"]["sentence_id"].size),
        "cohort_definition": "sentence_id intersection of each frozen random K5/K50 family pair; K50 must be a subset of K5; rows sorted by sentence_id",
    }
    return per_seed, image_ids, metadata


def _family_statistic(arrays: Sequence[Mapping[str, np.ndarray]]):
    def statistic(row_indices: np.ndarray) -> Dict[str, Dict[int, float]]:
        by_seed = [
            four_corner_paths(
                row["p5"][row_indices], row["r5"][row_indices],
                row["p50"][row_indices], row["r50"][row_indices],
            )
            for row in arrays
        ]
        return {
            name: {seed: values[name] for seed, values in zip(SEEDS, by_seed, strict=True)}
            for name in ESTIMATE_NAMES
        }
    return statistic


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    replace_with_retry(temp, path)


def _json_safe(value: Any) -> Any:
    """Convert non-finite estimates to JSON null without changing saved raw replicates."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    replace_with_retry(temp, path)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    replace_with_retry(temp, path)


def _analysis_status(*, formal_protocol: bool, all_families_complete: bool) -> str:
    if not all_families_complete:
        return "PARTIAL"
    return "COMPLETE" if formal_protocol else "PILOT"


def _write_family_artifacts(
    out_dir: Path,
    family: str,
    *,
    summary: Mapping[str, Any],
    corner_rows: Sequence[Mapping[str, Any]],
    ci_rows: Sequence[Mapping[str, Any]],
    raw_replicates: Mapping[str, np.ndarray],
) -> Dict[str, str]:
    """Commit one family before moving on, so interruption leaves traceable partial work."""
    family_dir = Path(out_dir) / "families" / family
    family_dir.mkdir(parents=True, exist_ok=True)
    corner_columns = ["family", "seed", "n_rows", "n_images", *ESTIMATE_NAMES, "S", "F", "E", "G"]
    ci_columns = list(ci_rows[0]) if ci_rows else []
    _write_csv(family_dir / "four_corners_paths_interaction.csv", corner_rows, corner_columns)
    _write_csv(family_dir / "bootstrap_ci.csv", ci_rows, ci_columns)
    _write_npz_atomic(family_dir / "bootstrap_raw_replicates.npz", raw_replicates)
    _write_json(family_dir / "summary.json", summary)
    return {
        "directory": (Path("families") / family).as_posix(),
        "summary": (Path("families") / family / "summary.json").as_posix(),
        "bootstrap_ci": (Path("families") / family / "bootstrap_ci.csv").as_posix(),
        "bootstrap_raw_replicates": (Path("families") / family / "bootstrap_raw_replicates.npz").as_posix(),
    }


def run_mechanism_analysis(
    *, out_dir: Path = OUT_DIR, n_replicates: int = 5000, seed: int = 0, ci: float = 0.95,
    formal_run: bool | None = None,
) -> Dict[str, Any]:
    """Run the exact descriptive mechanism accounting on frozen per-row predictions."""
    out_dir = Path(out_dir)
    protocol_parameters_match = int(n_replicates) == 5000 and int(seed) == 0 and float(ci) == 0.95
    if formal_run is True and not protocol_parameters_match:
        raise ValueError("formal mechanism analysis is fixed at 5000 replicates, seed 0, and 95% CI")
    formal_protocol = protocol_parameters_match and formal_run is not False
    analysis_kind = "FORMAL" if formal_protocol else "PILOT"
    try:
        out_dir.resolve().relative_to(OUT_DIR.resolve())
    except ValueError as exc:
        raise ValueError("mechanism output directory must remain inside results/research_repair_v1/mechanism") from exc
    if formal_protocol and out_dir.resolve() != OUT_DIR.resolve():
        raise ValueError("formal mechanism output must use the mechanism root directory")
    if not formal_protocol and (out_dir.resolve() == OUT_DIR.resolve() or not out_dir.name.startswith("pilot_")):
        raise ValueError("pilot mechanism output must use a separate pilot_<ID> directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_inputs = _validate_frozen_manifest()
    reference = _read_c1_point_reference()
    _log("frozen input manifest verified; loading aligned K5/K50 identities and C1 anchors")
    input_hash = _sha256(SOURCE_INPUT_MANIFEST)
    family_outputs: Dict[str, Any] = {}
    corner_path_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    raw_replicates: Dict[str, np.ndarray] = {}
    family_artifacts: Dict[str, Dict[str, str]] = {}
    completed_families: list[str] = []
    _write_json(out_dir / "progress_summary.json", {
        "schema": "research-repair-v1-mechanism-progress-v1",
        "status": "RUNNING" if formal_protocol else "PILOT_IN_PROGRESS",
        "analysis_kind": analysis_kind,
        "completed_families": completed_families,
        "currently_processing_family": None,
        "n_replicates": int(n_replicates), "seed": int(seed), "ci_level": float(ci),
    })
    for family in FAMILIES:
        _write_json(out_dir / "progress_summary.json", {
            "schema": "research-repair-v1-mechanism-progress-v1",
            "status": "RUNNING" if formal_protocol else "PILOT_IN_PROGRESS",
            "analysis_kind": analysis_kind,
            "completed_families": completed_families,
            "currently_processing_family": family,
            "n_replicates": int(n_replicates), "seed": int(seed), "ci_level": float(ci),
        })
        arrays, image_ids, cohort_metadata = load_family_cohort(family, reference)
        _log(f"{family}: verified n={cohort_metadata['n_rows']} rows, images={cohort_metadata['n_images']}; bootstrapping")
        bootstrap = shared_image_cluster_bootstrap(
            image_ids, _family_statistic(arrays), n_replicates=n_replicates, seed=seed, ci=ci
        )
        estimate_summary = bootstrap["estimates"]
        seed_points: Dict[int, Dict[str, float]] = {}
        for name in ESTIMATE_NAMES:
            est = estimate_summary[name]
            raw_replicates[f"{family}__{name}__mean3"] = np.asarray(est["replicates"], dtype=np.float64)
            for seed_id in SEEDS:
                per_seed = est["per_seed"][seed_id]
                raw_replicates[f"{family}__{name}__seed{seed_id}"] = np.asarray(
                    est["per_seed_replicates"][seed_id], dtype=np.float64
                )
                seed_points.setdefault(seed_id, {})[name] = float(per_seed["point"])
            ci_row: Dict[str, Any] = {
                "family": family, "estimate": name, "formula": FORMULAS[name],
                "point_mean3": float(est["point"]), "ci_low_mean3": est["ci_low"],
                "ci_high_mean3": est["ci_high"], "bootstrap_std_mean3": est["bootstrap_std"],
                "seed_standard_deviation": est["seed_standard_deviation"],
                "valid_replicates": est["valid_replicates"],
                "invalid_replicates": est["invalid_replicates"],
                "n_replicates": int(n_replicates), "ci_level": float(ci),
                "resample_unit": "image_cluster", "seed": int(seed),
                "n_rows": int(cohort_metadata["n_rows"]), "n_images": int(cohort_metadata["n_images"]),
            }
            for seed_id in SEEDS:
                per_seed = est["per_seed"][seed_id]
                ci_row.update({
                    f"seed{seed_id}_point": float(per_seed["point"]),
                    f"seed{seed_id}_ci_low": per_seed["ci_low"],
                    f"seed{seed_id}_ci_high": per_seed["ci_high"],
                    f"seed{seed_id}_valid_replicates": per_seed["valid_replicates"],
                    f"seed{seed_id}_invalid_replicates": per_seed["invalid_replicates"],
                })
            ci_rows.append(ci_row)
        for seed_id in SEEDS:
            row = {
                "family": family, "seed": f"seed{seed_id}",
                "n_rows": int(cohort_metadata["n_rows"]), "n_images": int(cohort_metadata["n_images"]),
                **seed_points[seed_id],
                **decomposition.transition_groups(arrays[seed_id - 1]["r5"], arrays[seed_id - 1]["r50"]).counts,
            }
            if abs(row["shapley_identity_residual"]) > 1e-12:
                raise MechanismAuditError(f"{family}/seed{seed_id}: Shapley identity residual exceeds 1e-12")
            corner_path_rows.append(row)
        mean_row: Dict[str, Any] = {
            "family": family, "seed": "mean3",
            "n_rows": int(cohort_metadata["n_rows"]), "n_images": int(cohort_metadata["n_images"]),
        }
        mean_row.update({name: float(estimate_summary[name]["point"]) for name in ESTIMATE_NAMES})
        if abs(mean_row["shapley_identity_residual"]) > 1e-12:
            raise MechanismAuditError(f"{family}/mean3: Shapley identity residual exceeds 1e-12")
        corner_path_rows.append(mean_row)
        family_outputs[family] = {
            **cohort_metadata,
            "estimates": {
                name: {
                    "point_mean3": float(estimate_summary[name]["point"]),
                    "ci_low": estimate_summary[name]["ci_low"],
                    "ci_high": estimate_summary[name]["ci_high"],
                    "valid_replicates": int(estimate_summary[name]["valid_replicates"]),
                    "invalid_replicates": int(estimate_summary[name]["invalid_replicates"]),
                    "seed_standard_deviation": float(estimate_summary[name]["seed_standard_deviation"]),
                    "per_seed_points": {
                        f"seed{seed_id}": float(estimate_summary[name]["per_seed"][seed_id]["point"])
                        for seed_id in SEEDS
                    },
                }
                for name in ESTIMATE_NAMES
            },
            "interaction_interpretation": "descriptive path interaction; not a causal mechanism test",
        }
        family_raw = {
            key: value for key, value in raw_replicates.items()
            if key.startswith(f"{family}__")
        }
        family_ci_rows = [row for row in ci_rows if row["family"] == family]
        family_corner_rows = [row for row in corner_path_rows if row["family"] == family]
        family_summary = {
            "schema": "research-repair-v1-mechanism-family-summary-v1",
            "status": "FAMILY_COMPLETE",
            "analysis_kind": analysis_kind,
            "family": family,
            "n_replicates": int(n_replicates), "seed": int(seed), "ci_level": float(ci),
            "resample_unit": "image_cluster",
            "cohort": cohort_metadata,
            "estimates": family_outputs[family]["estimates"],
            "interaction_interpretation": "descriptive path interaction; not a causal mechanism test",
        }
        family_artifacts[family] = _write_family_artifacts(
            out_dir, family, summary=family_summary, corner_rows=family_corner_rows,
            ci_rows=family_ci_rows, raw_replicates=family_raw,
        )
        completed_families.append(family)
        _write_json(out_dir / "progress_summary.json", {
            "schema": "research-repair-v1-mechanism-progress-v1",
            "status": "PARTIAL" if formal_protocol else "PILOT_IN_PROGRESS",
            "analysis_kind": analysis_kind,
            "completed_families": completed_families,
            "currently_processing_family": None,
            "family_artifacts": family_artifacts,
            "n_replicates": int(n_replicates), "seed": int(seed), "ci_level": float(ci),
        })
    corner_columns = [
        "family", "seed", "n_rows", "n_images", *ESTIMATE_NAMES, "S", "F", "E", "G",
    ]
    _write_csv(out_dir / "four_corners_paths_interaction.csv", corner_path_rows, corner_columns)
    ci_columns = list(ci_rows[0]) if ci_rows else []
    _write_csv(out_dir / "bootstrap_ci.csv", ci_rows, ci_columns)
    _write_npz_atomic(out_dir / "bootstrap_raw_replicates.npz", raw_replicates)
    summary = {
        "schema": "research-repair-v1-mechanism-summary-v1",
        "artifact": "four_corner_two_path_auroc_decomposition",
        "status": _analysis_status(formal_protocol=formal_protocol,
                                   all_families_complete=len(completed_families) == len(FAMILIES)),
        "analysis_kind": analysis_kind,
        "classification": "DESCRIPTIVE_MECHANISM_DECOMPOSITION",
        "interpretation_boundary": "exact statistical accounting of frozen C1 predictions; not causal identification; confidence paths measure AUROC ranking change, not temperature or calibration-scale effects",
        "scope": {
            "families": list(FAMILIES), "seeds": [f"seed{i}" for i in SEEDS],
            "comparison": "frozen random K5 -> random K50 common-K50 cohort",
            "new_features": 0, "new_training": 0, "new_model_forward": 0, "gpu_used": False,
        },
        "formulas": FORMULAS,
        "bootstrap": {
            "method": "shared_image_cluster_bootstrap", "resample_unit": "image_cluster",
            "n_replicates": int(n_replicates),
            "seed": int(seed), "ci_level": float(ci), "interval": "percentile",
            "draw_sharing": "within each family, one image-cluster draw is shared by all four corners, both paths, interaction, Shapley components, and all three fixed seeds",
        },
        "frozen_inputs": {
            "manifest_path": SOURCE_INPUT_MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": input_hash, "verified_artifact_count": len(frozen_inputs),
            "artifact_hashes": frozen_inputs,
            "c1_reference_path": C1_POINT_REFERENCE.relative_to(ROOT).as_posix(),
            "reproduction_tolerance": 1e-12,
        },
        "families": family_outputs,
        "output_files": {
            "four_corners_paths_interaction": "four_corners_paths_interaction.csv",
            "bootstrap_ci": "bootstrap_ci.csv",
            "bootstrap_raw_replicates": "bootstrap_raw_replicates.npz",
            "progress_summary": "progress_summary.json",
            "families": family_artifacts,
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "progress_summary.json", {
        "schema": "research-repair-v1-mechanism-progress-v1",
        "status": summary["status"],
        "analysis_kind": analysis_kind,
        "completed_families": completed_families,
        "currently_processing_family": None,
        "family_artifacts": family_artifacts,
        "summary": "summary.json",
        "n_replicates": int(n_replicates), "seed": int(seed), "ci_level": float(ci),
    })
    _log("completed all families and wrote the shared-bootstrap mechanism summary")
    return summary


__all__ = [
    "ESTIMATE_NAMES", "FORMULAS", "MechanismAuditError", "derive_paths_from_corners",
    "four_corner_paths", "load_family_cohort", "run_mechanism_analysis",
    "_analysis_status", "_json_safe", "_write_family_artifacts", "_write_json", "_write_npz_atomic",
]
