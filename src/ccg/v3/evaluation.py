"""V3 FineCops grounding and correctness-reliability summaries.

The V3 selective-risk estimate uses expected fractional acceptance at a tied
boundary. It is invariant to row ordering and follows the predeclared V3
protocol; legacy selective metrics retain their existing stable-order rule.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from ..metrics.discrimination import auroc_correct


def _vectors(confidence: Any, correctness: Any) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = _binary_correctness(correctness)
    if scores.size != correct.size:
        raise ValueError("confidence and correctness lengths differ")
    if not np.isfinite(scores).all():
        raise ValueError("confidence must be finite")
    return scores, correct


def _binary_correctness(values: Any) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError("correctness must contain only finite binary 0/1 values")
    if np.any((raw != 0.0) & (raw != 1.0)):
        raise ValueError("correctness must contain only binary 0/1 values")
    return raw.astype(bool)


def fractional_boundary_risk(confidence: Any, correctness: Any, coverage: float) -> float:
    """Expected error risk at coverage, randomizing uniformly within cutoff ties.

    The acceptance budget is exactly ``coverage * N``. All scores above the
    boundary are accepted; the required fraction of the tied boundary group is
    accepted uniformly. Expected errors are divided by that exact budget.
    """
    scores, correct = _vectors(confidence, correctness)
    level = float(coverage)
    if not np.isfinite(level) or level <= 0.0 or level > 1.0:
        raise ValueError("coverage must be finite and in (0, 1]")
    n = int(scores.size)
    if n == 0:
        return float("nan")
    budget = level * n
    ordered = np.sort(scores)[::-1]
    cutoff_position = min(n - 1, max(0, int(np.ceil(budget)) - 1))
    cutoff = float(ordered[cutoff_position])
    above = scores > cutoff
    tied = scores == cutoff
    need_from_tie = budget - float(above.sum())
    fraction = min(1.0, max(0.0, need_from_tie / float(tied.sum())))
    accepted_errors = float(np.sum(~correct[above])) + fraction * float(np.sum(~correct[tied]))
    return float(accepted_errors / budget)


def correctness_reliability(confidence: Any, correctness: Any) -> dict[str, float | int | None]:
    """Correctness AUROC and fractional-boundary risk at 50% and 80% coverage."""
    scores, correct = _vectors(confidence, correctness)
    if scores.size == 0:
        return {"n": 0, "auroc_correct": None, "risk50": None, "risk80": None}
    auc = float(auroc_correct(scores, correct))
    return {
        "n": int(scores.size),
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "auroc_correct": auc if np.isfinite(auc) else None,
        "risk50": _finite_or_none(fractional_boundary_risk(scores, correct, 0.5)),
        "risk80": _finite_or_none(fractional_boundary_risk(scores, correct, 0.8)),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize all rows by requested K while preserving unavailable flow rows.

    Rows passed to this function must describe one mode/backbone/seed. Empty
    candidate rows count as incorrect only for flow and all-row accuracy;
    their correctness remains null in the saved prediction table. Reliability
    estimates use rows with ``k_eff >= 5`` and report any missing-confidence
    denominator explicitly.
    """
    if not rows:
        return {}
    for key in ("mode", "backbone", "seed"):
        values = {str(row[key]) for row in rows if key in row}
        if len(values) > 1:
            raise ValueError(f"prediction summary cannot pool multiple {key} values: {sorted(values)}")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["requested_k"])].append(row)
    output: dict[str, Any] = {}
    for k, group in sorted(grouped.items()):
        n_all = len(group)
        present_values = [row["target_present"] for row in group]
        if any(not isinstance(value, (bool, np.bool_)) for value in present_values):
            raise ValueError("target_present must be explicitly boolean")
        present = np.asarray(present_values, dtype=bool)
        k_eff = np.asarray([int(row["k_eff"]) for row in group], dtype=np.int64)
        if np.any(k_eff < 0) or np.any(k_eff > k):
            raise ValueError("effective K must be between zero and requested K")
        nonempty = np.asarray([value > 0 for value in k_eff], dtype=bool)
        correct_values: list[bool] = []
        for row, has_candidates in zip(group, nonempty, strict=True):
            value = row.get("correct")
            if value is None:
                if has_candidates:
                    raise ValueError("correctness may be null only for an empty candidate set")
                correct_values.append(False)  # explicit failed-grounding flow convention
            elif isinstance(value, (bool, np.bool_)):
                correct_values.append(bool(value))
            elif isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) and value in (0, 1):
                correct_values.append(bool(value))
            else:
                raise ValueError("correctness must be binary or null for an empty candidate set")
        correct_all = np.asarray(correct_values, dtype=bool)
        eligible = k_eff >= 5
        present_acc_mask = present & nonempty
        models = sorted({name for row in group for name in (row.get("confidence") or {})})
        model_metrics: dict[str, Any] = {}
        finite_by_model: dict[str, np.ndarray] = {}
        for model in models:
            values: list[float] = []
            for row in group:
                raw_value = (row.get("confidence") or {}).get(model)
                if raw_value is None:
                    values.append(np.nan)
                else:
                    number = float(raw_value)
                    if not np.isfinite(number):
                        raise ValueError(f"{model} confidence must be finite or explicitly null")
                    values.append(number)
            raw_confidence = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(raw_confidence)
            finite_by_model[model] = finite
            overall_indices = np.flatnonzero(eligible)
            conditional_indices = np.flatnonzero(eligible & present)
            overall_indices = overall_indices[finite[overall_indices]]
            conditional_indices = conditional_indices[finite[conditional_indices]]
            model_metrics[model] = {
                "overall": correctness_reliability(raw_confidence[overall_indices], correct_all[overall_indices]),
                "target_present": correctness_reliability(
                    raw_confidence[conditional_indices], correct_all[conditional_indices]
                ),
                "n_missing_confidence_overall_eligible": int(eligible.sum() - overall_indices.size),
                "n_missing_confidence_target_present_eligible": int((eligible & present).sum() - conditional_indices.size),
                "missing_reason_counts": _missing_reasons(group, model, eligible),
            }

        # Pairwise comparisons are explicitly recomputed on the intersection of
        # available-confidence rows; per-model estimates above may use different
        # denominators when a feature family is unavailable for a row.
        paired: dict[str, Any] = {}
        pair_names = (
            ("Full_vs_S+Q", "Full", "S+Q"),
            ("SDS_large_vs_small", "SDS_large", "SDS_small"),
        )
        for label, left, right in pair_names:
            if left not in finite_by_model or right not in finite_by_model:
                continue
            common = eligible & finite_by_model[left] & finite_by_model[right]
            left_present = common & present
            right_conf = np.asarray([
                (row.get("confidence") or {}).get(right, np.nan) for row in group
            ], dtype=np.float64)
            left_conf = np.asarray([
                (row.get("confidence") or {}).get(left, np.nan) for row in group
            ], dtype=np.float64)
            paired[label] = {
                "n_common_rows": int(common.sum()),
                "n_common_target_present_rows": int(left_present.sum()),
                "left_missing_confidence_rows": int((eligible & ~finite_by_model[left]).sum()),
                "right_missing_confidence_rows": int((eligible & ~finite_by_model[right]).sum()),
                "left": correctness_reliability(left_conf[common], correct_all[common]),
                "right": correctness_reliability(right_conf[common], correct_all[common]),
                "left_target_present": correctness_reliability(left_conf[left_present], correct_all[left_present]),
                "right_target_present": correctness_reliability(right_conf[left_present], correct_all[left_present]),
            }
        output[str(k)] = {
            "requested_k": int(k),
            "n_rows": n_all,
            "n_empty_candidates": int((~nonempty).sum()),
            "n_basic_grounding_only": int(((k_eff > 0) & (k_eff < 5)).sum()),
            "n_reliability_eligible_k_ge_5": int(eligible.sum()),
            "target_coverage": float(present.mean()) if n_all else None,
            "n_target_present": int(present.sum()),
            "bank_target_coverage": (
                float(np.asarray([bool(row.get("bank_target_coverage", False)) for row in group]).mean())
                if n_all else None
            ),
            "valid_candidate_coverage": float(nonempty.mean()) if n_all else None,
            "conditional_grounding_accuracy": (
                float(correct_all[present_acc_mask].mean()) if np.any(present_acc_mask) else None
            ),
            "n_target_present_nonempty": int(present_acc_mask.sum()),
            "all_rows_grounding_accuracy": float(correct_all.mean()) if n_all else None,
            "failed_grounding_rows": int((~nonempty).sum()),
            "reliability": model_metrics,
            "paired_model_common_rows": paired,
        }
    return output


def _missing_reasons(
    rows: Sequence[Mapping[str, Any]], model: str, eligible: np.ndarray
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for index in np.flatnonzero(eligible):
        row = rows[int(index)]
        confidence = (row.get("confidence") or {}).get(model)
        if confidence is not None and np.isfinite(float(confidence)):
            continue
        reasons = row.get("confidence_missing_reason") or {}
        reason = str(reasons.get(model, "missing_confidence"))
        counts[reason] += 1
    return dict(sorted(counts.items()))


def controlled_common_k50_mask(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """One common control cohort across every scorer and all four comparisons."""
    return np.asarray([
        int(row["requested_k"]) == 50
        and int(row["k_eff"]) == 50
        and bool(row.get("target_present"))
        and row.get("missing_reason") in (None, "")
        for row in rows
    ], dtype=bool)


def primary_point_estimates(
    *,
    image_ids: Sequence[Any],
    common_mask: Sequence[bool],
    confidence_by_seed: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    correctness_by_seed: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
) -> dict[str, Any]:
    """Compute the four predeclared controlled point contrasts (no bootstrap)."""
    image = np.asarray(image_ids)
    mask = np.asarray(common_mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError("common K50 mask and image IDs must align")
    n = int(mask.sum())
    if n == 0:
        raise ValueError("controlled common K50 cohort is empty")
    required_seeds = ("b3_seed1", "b3_seed2", "b3_seed3")
    seed_values: dict[str, dict[str, float | None]] = {}

    # These names deliberately encode the backbone because the winner
    # correctness label is backbone-specific even when candidate IDs match.
    def auc(seed: str, model: str, backbone: str, k: int) -> float | None:
        confidence = np.asarray(confidence_by_seed[seed][model][k], dtype=np.float64)
        correct = _binary_correctness(correctness_by_seed[seed][backbone][k])
        if confidence.shape != mask.shape or correct.shape != mask.shape:
            raise ValueError(f"{seed}/{backbone}/{model}/K{k}: prediction rows do not align to the K50 cohort")
        if not np.isfinite(confidence).all():
            raise ValueError(f"{seed}/{backbone}/{model}/K{k}: missing confidence on common controlled cohort")
        value = float(auroc_correct(confidence[mask], correct[mask]))
        if not np.isfinite(value):
            return None
        return value

    for seed in required_seeds:
        values = {
            "C1_B0_MSP_AUROC_K5_minus_K50": (
                auc(seed, "B0_MSP", "b0", 5), auc(seed, "B0_MSP", "b0", 50)
            ),
            "C2_B0_Full_minus_SQ_AUROC_K50": (
                auc(seed, "B0_Full", "b0", 50), auc(seed, "B0_SQ", "b0", 50)
            ),
            # B16's independent grounding winner defines C3 correctness.
            "C3_B16_Full_minus_SQ_AUROC_K50": (
                auc(seed, "B16_Full", "b16", 50), auc(seed, "B16_SQ", "b16", 50)
            ),
            "C4_B0_SDS_large_minus_small_AUROC_K50": (
                auc(seed, "B0_SDS_large", "b0", 50), auc(seed, "B0_SDS_small", "b0", 50)
            ),
        }
        def difference(left: float | None, right: float | None) -> float | None:
            return float(left - right) if left is not None and right is not None else None

        seed_values[seed] = {name: difference(*pair) for name, pair in values.items()}
    names = tuple(next(iter(seed_values.values())).keys())
    return {
        "cohort": "controlled_common_K50",
        "n_rows": n,
        "n_images": int(np.unique(image[mask]).size),
        "estimator": "mean of three per-seed point contrasts; predictions are never ensembled",
        "seed_contrasts": seed_values,
        "point_contrasts": {
            name: (
                float(np.mean([seed_values[seed][name] for seed in required_seeds]))
                if all(seed_values[seed][name] is not None for seed in required_seeds)
                else None
            )
            for name in names
        },
        "undefined_seed_counts": {
            name: sum(seed_values[seed][name] is None for seed in required_seeds)
            for name in names
        },
        "uncertainty": "not computed; formal 5000-draw bootstrap awaits resource grant",
    }

