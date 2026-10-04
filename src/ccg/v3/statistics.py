"""Shared-cluster inference for the frozen V3 JSONL prediction interface.

Tie groups are sorted once. Integer row multiplicities reproduce explicit
cluster resampling exactly, including fractional-boundary selective risk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ccg.repairs.bootstrap import shared_image_cluster_bootstrap
from ccg.v3.protocol import sha256

SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
KS = (5, 10, 20, 50)
METRICS = ("auroc_correct", "risk50", "risk80", "brier_binary")
CONTRAST_METRICS = ("auroc_correct", "risk50", "risk80")


class WeightedReliability:
    """Fixed rows and score ties, evaluated with bootstrap row multiplicities."""

    def __init__(self, confidence: Sequence[float], correctness: Sequence[float], mask=None):
        scores = np.asarray(confidence, dtype=float)
        correct = np.asarray(correctness, dtype=float)
        if scores.ndim != 1 or correct.shape != scores.shape:
            raise ValueError("confidence and correctness must be aligned vectors")
        selected = np.ones(scores.size, bool) if mask is None else np.asarray(mask, bool)
        if selected.shape != scores.shape:
            raise ValueError("mask must align with predictions")
        if not np.isfinite(scores[selected]).all():
            raise ValueError("selected confidence must be finite")
        if np.any((scores[selected] < 0) | (scores[selected] > 1)):
            raise ValueError("probability confidence must lie in [0,1]")
        if not np.isfinite(correct[selected]).all() or np.any(~np.isin(correct[selected], [0, 1])):
            raise ValueError("selected correctness must be finite binary")
        self.n = scores.size
        self.order = np.flatnonzero(selected)[np.argsort(-scores[selected], kind="stable")]
        self.correct = correct[self.order]
        self.squared_error = (scores[self.order] - self.correct) ** 2
        sorted_scores = scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1] if self.order.size else np.array([], int)

    def evaluate(self, weights: np.ndarray) -> dict[str, float]:
        values = np.asarray(weights, dtype=float)
        if values.shape != (self.n,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("row multiplicities must align and be nonnegative finite")
        w = values[self.order]
        total = float(w.sum())
        if total == 0:
            return dict.fromkeys(METRICS, float("nan"))
        positive = np.add.reduceat(w * self.correct, self.starts)
        negative = np.add.reduceat(w * (1 - self.correct), self.starts)
        counts = positive + negative
        npos, nneg = float(positive.sum()), float(negative.sum())
        auc = float(np.sum(positive * (nneg - np.cumsum(negative) + .5 * negative)) / (npos * nneg)) if npos and nneg else float("nan")
        result = dict.fromkeys(METRICS, float("nan"))
        result.update(auroc_correct=auc, brier_binary=float(np.dot(w, self.squared_error) / total))
        cumulative = np.cumsum(counts)
        error_before = np.r_[0., np.cumsum(negative)[:-1]]
        count_before = np.r_[0., cumulative[:-1]]
        for level, name in ((.5, "risk50"), (.8, "risk80")):
            budget = level * total
            boundary = int(np.searchsorted(cumulative, budget, side="left"))
            # Empty tie groups introduced by a bootstrap have zero weight and
            # searchsorted skips them at every strictly positive budget.
            fraction = (budget - count_before[boundary]) / counts[boundary]
            result[name] = float((error_before[boundary] + fraction * negative[boundary]) / budget)
        return result


def load_prediction_grid(path: str | Path) -> tuple[list[str], np.ndarray, dict]:
    """Keep expressions distinct; reject duplicates and missing grid cells."""
    grid, record_images, order = {}, {}, []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row = dict(row)
            if "k_eff" not in row:
                row["k_eff"] = row["effective_k"]
            if "confidence" not in row:
                row["confidence"] = row.get("confidences")
            record = str(row["record_id"])
            image = str(row["image_id"])
            if row.get("source_split") != "val":
                raise ValueError(f"line {line_number}: confirmation row must be official val")
            if record not in record_images:
                order.append(record)
                record_images[record] = image
            elif record_images[record] != image:
                raise ValueError(f"image identity changed within {record}")
            key = (str(row["mode"]), str(row["backbone"]), str(row["seed"]), int(row["requested_k"]))
            rows = grid.setdefault(key, {})
            if record in rows:
                raise ValueError(f"duplicate prediction: {record}/{key}")
            k_eff = int(row["k_eff"])
            if k_eff < 0 or k_eff > key[3] or (k_eff == 0 and row["target_present"]):
                raise ValueError(f"invalid candidate flow: {record}/{key}")
            correct = row.get("correct")
            if (k_eff == 0 and correct is not None) or (k_eff > 0 and (not isinstance(correct, (bool, int)) or correct not in (0, 1))):
                raise ValueError(f"invalid scored/empty correctness: {record}/{key}")
            if correct and not row["target_present"]:
                raise ValueError(f"correct prediction cannot lack a covered target: {record}/{key}")
            rows[record] = row
    if not order:
        raise ValueError("empty prediction file")
    for mode in ("controlled", "natural"):
        for backbone in ("b0", "b16"):
            for seed in SEEDS:
                for k in KS:
                    key = mode, backbone, seed, k
                    if key not in grid or set(grid[key]) != set(order):
                        raise ValueError(f"incomplete expression grid: {key}")
    return order, np.array([record_images[key] for key in order]), grid


def _arrays(rows: list[dict], model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray([(row.get("confidence") or {}).get(model, np.nan) for row in rows], float)
    correct = np.asarray([0 if row["correct"] is None else row["correct"] for row in rows], float)
    eligible = np.asarray([int(row["k_eff"]) >= 5 for row in rows], bool) & np.isfinite(scores)
    return scores, correct, eligible


def _bootstrap(image_ids, evaluate, *, n_replicates):
    if np.unique(image_ids).size < 2:
        return {"status": "UNAVAILABLE_FEWER_THAN_TWO_IMAGES", "n_images": int(np.unique(image_ids).size)}
    return shared_image_cluster_bootstrap(image_ids, evaluate, n_replicates=n_replicates, seed=0, ci=.95)


def controlled_inference(order, images, grid, *, n_replicates=5000):
    """Four contrasts and absolute endpoints on the identical K50 cohort."""
    common = np.ones(len(order), bool)
    for backbone in ("b0", "b16"):
        for seed in SEEDS:
            rows = [grid[("controlled", backbone, seed, 50)][record] for record in order]
            common &= np.asarray([row["k_eff"] == 50 and bool(row["target_present"]) and not row.get("missing_reason") for row in rows])
    selected = np.flatnonzero(common)
    models = (("b0", "MSP", 5), ("b0", "MSP", 50), ("b0", "S", 50), ("b0", "S+Q", 50),
              ("b0", "S+V", 50), ("b0", "Full", 50), ("b16", "S", 50), ("b16", "S+Q", 50),
              ("b16", "S+V", 50), ("b16", "Full", 50), ("b0", "SDS_small", 50), ("b0", "SDS_large", 50))
    profiles = {}
    for seed in SEEDS:
        for backbone, model, k in models:
            rows = [grid[("controlled", backbone, seed, k)][order[i]] for i in selected]
            scores, correct, eligible = _arrays(rows, model)
            if not eligible.all():
                raise ValueError(f"missing primary confidence on common cohort: {seed}/{backbone}/{model}/{k}")
            profiles[seed, backbone, model, k] = WeightedReliability(scores, correct)
    contrasts = {
        "C1": (("b0", "MSP", 5), ("b0", "MSP", 50)),
        "C2": (("b0", "Full", 50), ("b0", "S+Q", 50)),
        "C3": (("b16", "Full", 50), ("b16", "S+Q", 50)),
        "C4": (("b0", "SDS_large", 50), ("b0", "SDS_small", 50)),
    }
    def evaluate(indices):
        weights = np.bincount(indices, minlength=selected.size)
        cached = {key: profile.evaluate(weights) for key, profile in profiles.items()}
        estimates = {}
        for backbone, model, k in models:
            for metric in METRICS:
                estimates[f"absolute/{backbone}/{model}/K{k}/{metric}"] = {seed: cached[seed, backbone, model, k][metric] for seed in SEEDS}
        for name, (left, right) in contrasts.items():
            for metric in CONTRAST_METRICS:
                estimates[f"{name}/{metric}"] = {seed: cached[(seed, *left)][metric] - cached[(seed, *right)][metric] for seed in SEEDS}
        return estimates
    result = _bootstrap(images[selected], evaluate, n_replicates=n_replicates)
    result["flow"] = {"all_expressions": len(order), "common_expressions": int(common.sum()),
                      "excluded_expressions": int((~common).sum()), "common_images": int(np.unique(images[common]).size)}
    return result


def natural_inference(order, images, grid, *, n_replicates=5000):
    """Whole-flow, covered and paired reliability with shared full-image draws."""
    prepared = {}
    for backbone in ("b0", "b16"):
        for seed in SEEDS:
            for k in KS:
                rows = [grid[("natural", backbone, seed, k)][record] for record in order]
                present = np.array([bool(row["target_present"]) for row in rows])
                bank_present = np.array([bool(row["bank_target_coverage"]) for row in rows])
                if np.any(present & ~bank_present):
                    raise ValueError("Presented target coverage cannot exceed bank coverage")
                correct = np.array([0 if row["correct"] is None else row["correct"] for row in rows], float)
                keff = np.array([row["k_eff"] for row in rows], int)
                model_names = ("MSP", "S", "S+Q", "S+V", "Full")
                profiles, masks = {}, {}
                for model in model_names:
                    scores, corr, eligible = _arrays(rows, model)
                    masks[model] = eligible
                    for scope, mask in (("overall", eligible), ("covered", eligible & present)):
                        profiles[model, scope] = WeightedReliability(scores, corr, mask)
                pair_mask = masks["Full"] & masks["S+Q"]
                for model in ("Full", "S+Q"):
                    scores, corr, _ = _arrays(rows, model)
                    for scope, mask in (("paired_overall", pair_mask), ("paired_covered", pair_mask & present)):
                        profiles[model, scope] = WeightedReliability(scores, corr, mask)
                    # Compare the same correct predictions against uncovered
                    # errors. This is a ranking audit, not a causal mechanism.
                    profiles[model, "paired_cross_uncovered"] = WeightedReliability(scores, corr, pair_mask & ((corr == 1) | ~present))
                prepared[backbone, seed, k] = (profiles, present, correct, keff, masks, pair_mask, bank_present)
    flow = {}
    for (backbone, seed, k), (_, present, correct, keff, masks, pair_mask, bank_present) in prepared.items():
        flow[f"{backbone}/{seed}/K{k}"] = {
            "n_all": len(order), "n_covered": int(present.sum()), "n_empty": int((keff == 0).sum()),
            "n_basic_only": int(((keff > 0) & (keff < 5)).sum()),
            "n_bank_covered": int(bank_present.sum()),
            "actual_k_histogram": {str(value): int((keff == value).sum()) for value in np.unique(keff)},
            "model_denominators": {model: {"overall": int(mask.sum()), "covered": int((mask & present).sum()),
                                              "missing_in_eligible": int(((keff >= 5) & ~mask).sum())} for model, mask in masks.items()},
            "paired_denominators": {"overall": int(pair_mask.sum()), "covered": int((pair_mask & present).sum())},
        }
    def ratio(values, weights, mask=None):
        w = weights if mask is None else weights * mask
        return float(np.dot(w, values) / w.sum()) if w.sum() else float("nan")
    def evaluate(indices):
        weights = np.bincount(indices, minlength=len(order))
        estimates = {}
        for (backbone, seed, k), (profiles, present, correct, keff, masks, pair_mask, bank_present) in prepared.items():
            prefix = f"{backbone}/K{k}"
            for name, value in (("target_coverage", ratio(present, weights)),
                                ("bank_target_coverage", ratio(bank_present, weights)),
                                ("accuracy_all", ratio(correct, weights)),
                                ("accuracy_covered", ratio(correct, weights, present)),
                                ("output_rate", ratio(keff > 0, weights))):
                estimates.setdefault(f"{prefix}/flow/{name}", {})[seed] = value
            cache = {key: profile.evaluate(weights) for key, profile in profiles.items()}
            for (model, scope), metrics in cache.items():
                for metric, value in metrics.items():
                    if scope == "paired_cross_uncovered" and metric != "auroc_correct":
                        continue
                    estimates.setdefault(f"{prefix}/{scope}/{model}/{metric}", {})[seed] = value
            for scope in ("paired_overall", "paired_covered"):
                for metric in CONTRAST_METRICS:
                    value = cache["Full", scope][metric] - cache["S+Q", scope][metric]
                    estimates.setdefault(f"{prefix}/{scope}/Full_minus_SQ/{metric}", {})[seed] = value
            delta_all = cache["Full", "paired_overall"]["auroc_correct"] - cache["S+Q", "paired_overall"]["auroc_correct"]
            delta_covered = cache["Full", "paired_covered"]["auroc_correct"] - cache["S+Q", "paired_covered"]["auroc_correct"]
            delta_cross = cache["Full", "paired_cross_uncovered"]["auroc_correct"] - cache["S+Q", "paired_cross_uncovered"]["auroc_correct"]
            negative = pair_mask & (correct == 0)
            negative_count = float(weights[negative].sum())
            uncovered_weight = float(weights[negative & ~present].sum()) / negative_count if negative_count else float("nan")
            covered_contribution = 0. if uncovered_weight == 1 else (1 - uncovered_weight) * delta_covered
            uncovered_contribution = 0. if uncovered_weight == 0 else uncovered_weight * delta_cross
            for name, value in (("overall_minus_covered_gain", delta_all - delta_covered),
                                ("cross_uncovered_gain", delta_cross), ("uncovered_negative_weight", uncovered_weight),
                                ("covered_ranking_contribution", covered_contribution),
                                ("uncovered_ranking_contribution", uncovered_contribution)):
                estimates.setdefault(f"{prefix}/ranking_audit/{name}", {})[seed] = value
            if np.isfinite([delta_all, covered_contribution, uncovered_contribution]).all():
                if not np.isclose(delta_all, covered_contribution + uncovered_contribution, atol=1e-12, rtol=0):
                    raise AssertionError("Natural pairwise AUROC accounting identity failed")
        return estimates
    result = _bootstrap(images, evaluate, n_replicates=n_replicates)
    result["flow"] = flow
    return result


def _safe(value):
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def export_inference(result: dict, destination: Path, *, primary: bool) -> dict:
    """Save all raw seed/mean draws once; derive both CI levels from same draws."""
    destination.mkdir(parents=True, exist_ok=True)
    arrays, estimates, keys = {}, {}, {}
    for index, (name, record) in enumerate(result.get("estimates", {}).items()):
        prefix = f"e{index:04d}"
        arrays[f"{prefix}_mean"] = record["replicates"]
        for seed in SEEDS:
            arrays[f"{prefix}_{seed}"] = record["per_seed_replicates"][seed]
        compact = {key: value for key, value in record.items() if key not in ("replicates", "per_seed_replicates", "per_seed")}
        compact["per_seed"] = {seed: {key: value for key, value in item.items() if key != "replicates"} for seed, item in record["per_seed"].items()}
        if primary and name in {f"C{i}/auroc_correct" for i in range(1, 5)}:
            valid = record["replicates"][np.isfinite(record["replicates"])]
            compact["bonferroni_9875"] = {"ci_level": .9875, "ci_low": float(np.quantile(valid, .00625)) if valid.size >= 2 else None,
                                           "ci_high": float(np.quantile(valid, .99375)) if valid.size >= 2 else None,
                                           "conditioning": record["ci_conditioning"]}
        keys[name] = {"mean": f"{prefix}_mean", "per_seed": {seed: f"{prefix}_{seed}" for seed in SEEDS}}
        estimates[name] = compact
    raw = destination / "draws.npz"
    np.savez_compressed(raw, **arrays)
    summary = {key: value for key, value in result.items() if key != "estimates"}
    summary.update({"status": result.get("status", "COMPLETE"), "risk_ties": "fractional_boundary_tie",
                    "estimator": "mean of three fixed-seed effects, never prediction ensemble", "estimates": estimates,
                    "raw_draws": {"path": raw.name, "sha256": sha256(raw), "size_bytes": raw.stat().st_size, "keys": keys}})
    (destination / "summary.json").write_text(json.dumps(_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return _safe(summary)
