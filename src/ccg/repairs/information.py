"""Research Repair information-source and ID/OOD experiment helpers.

All train/validation normalization is fitted from the selected training images.
The module only consumes frozen grounding scores and cached embeddings; it does
not run an image encoder or retrain a grounding scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..metrics.discrimination import auroc_correct
from ..reliability import features as rfeat
from ..reliability.models import LogisticModel, ScoreDeepSets
from ..semantic import features as sfeat

__all__ = [
    "B3_SEEDS",
    "FEATURE_GROUPS",
    "LOGISTIC_CS",
    "Q_NAMES",
    "SDS_LRS",
    "TRAINING_REGIMES",
    "V_NAMES",
    "assert_nested_correctness",
    "choose_logistic_c",
    "compute_semantic_stats_chunked",
    "feature_block",
    "fit_group_normalization",
    "fit_score_deepsets_normalization",
    "fit_selected_logistic",
    "load_logistic_predictor",
    "nll_binary",
    "score_deepsets_inputs",
    "sigmoid",
    "train_score_deepsets_grid",
]

B3_SEEDS = ("b3_seed1", "b3_seed2", "b3_seed3")
TRAINING_REGIMES: Mapping[str, tuple[int, int]] = {
    "smallK5_10": (5, 10),
    "largeK20_50": (20, 50),
}
LOGISTIC_CS = (0.1, 1.0, 10.0)
SDS_LRS = (1e-4, 3e-4, 1e-3)
Q_NAMES = (
    "clip_top1",
    "clip_top2",
    "clip_margin12",
    "clip_entropy",
    "clip_normH",
    "clip_rank_top1",
    "q_top3",
    "q_margin13",
)
V_NAMES = (
    "cand_vmax",
    "cand_vmean",
    "cand_vstd",
    "cand_top12_sim",
    "cand_top15_mean",
    "density_070",
    "density_080",
    "cand_top13_sim",
)
FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "S": tuple(rfeat.stat_feature_names()),
    "S+Q": tuple(rfeat.stat_feature_names()) + Q_NAMES,
    "S+V": tuple(rfeat.stat_feature_names()) + V_NAMES,
    # Preserve the original E1b column order for a direct reference check.
    "Full": tuple(rfeat.stat_feature_names()) + tuple(sfeat.SEMANTIC_STAT_NAMES),
}
_SEMANTIC_INDEX = {name: index for index, name in enumerate(sfeat.SEMANTIC_STAT_NAMES)}


def assert_nested_correctness(correct_by_k: Mapping[int, np.ndarray]) -> None:
    """Reject impossible correct-after-incorrect flips under nested candidate sets.

    Candidate sets grow monotonically while preserving the target and every
    earlier candidate; therefore adding candidates may turn a correct top-1
    choice into an error, but cannot make an earlier error correct again.
    """
    ks = sorted(int(k) for k in correct_by_k)
    if not ks:
        raise ValueError("correct_by_k must contain at least one K")
    previous = np.asarray(correct_by_k[ks[0]], dtype=bool).reshape(-1)
    for k in ks[1:]:
        current = np.asarray(correct_by_k[k], dtype=bool).reshape(-1)
        if current.shape != previous.shape:
            raise ValueError(f"K={k}: correctness rows differ from K={ks[ks.index(k) - 1]}")
        flip = (~previous) & current
        if np.any(flip):
            row = int(np.flatnonzero(flip)[0])
            raise AssertionError(f"nested correctness flips 0->1 at K={k}, row={row}")
        previous = current


@dataclass(frozen=True)
class LogisticPredictor:
    """Small serializable view of a fitted binary logistic model."""

    c: float
    coefficients: np.ndarray
    intercept: float

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.coefficients.size:
            raise ValueError(
                f"logistic input must be [n, {self.coefficients.size}], got {values.shape}"
            )
        return sigmoid(values @ self.coefficients + float(self.intercept))


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(values, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def nll_binary(probability: np.ndarray, target: np.ndarray) -> float:
    """Mean Bernoulli negative log likelihood with a finite endpoint guard."""
    p = np.clip(np.asarray(probability, dtype=np.float64).reshape(-1), 1e-15, 1.0 - 1e-15)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if p.size != y.size or p.size == 0:
        raise ValueError("probability and target must be non-empty vectors of equal length")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def compute_semantic_stats_chunked(
    z_q: np.ndarray,
    z_i: np.ndarray,
    scores: np.ndarray,
    *,
    chunk_rows: int = 512,
) -> np.ndarray:
    """Compute frozen 16-D semantic summaries without expanding a full K block."""
    q = np.asarray(z_q)
    candidates = np.asarray(z_i)
    raw_scores = np.asarray(scores)
    if q.ndim != 2 or candidates.ndim != 3 or raw_scores.ndim != 2:
        raise ValueError("expected z_q [n,d], z_i [n,K,d], and scores [n,K]")
    if candidates.shape[0] != q.shape[0] or raw_scores.shape != candidates.shape[:2]:
        raise ValueError("embedding and score batch/K dimensions do not match")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    result = np.empty((q.shape[0], len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, q.shape[0], int(chunk_rows)):
        stop = min(start + int(chunk_rows), q.shape[0])
        result[start:stop] = sfeat.semantic_stats(
            q[start:stop], candidates[start:stop], raw_scores[start:stop]
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("semantic summary contains a non-finite value")
    return result


def feature_block(stats17: np.ndarray, sem16: np.ndarray, group: str) -> np.ndarray:
    """Select a predeclared S/Q/V feature group in its frozen column order."""
    if group not in FEATURE_GROUPS:
        raise KeyError(f"unknown feature group {group!r}; expected {tuple(FEATURE_GROUPS)}")
    stats = np.asarray(stats17, dtype=np.float64)
    semantic = np.asarray(sem16, dtype=np.float64)
    if stats.ndim != 2 or stats.shape[1] != len(rfeat.stat_feature_names()):
        raise ValueError(f"stats17 must be [n,17], got {stats.shape}")
    if semantic.shape != (stats.shape[0], len(sfeat.SEMANTIC_STAT_NAMES)):
        raise ValueError(f"sem16 must be [n,16], got {semantic.shape}")
    names = FEATURE_GROUPS[group]
    values: list[np.ndarray] = []
    stats_names = rfeat.stat_feature_names()
    for name in names:
        if name in stats_names:
            values.append(stats[:, stats_names.index(name)])
        else:
            values.append(semantic[:, _SEMANTIC_INDEX[name]])
    matrix = np.column_stack(values)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"feature group {group} contains non-finite values")
    return matrix


def fit_group_normalization(
    raw_by_k: Mapping[int, np.ndarray],
    train_masks: Mapping[int, np.ndarray],
    ks: Sequence[int],
    keys: Sequence[str],
) -> tuple[rfeat.NormalizationFit, dict[int, np.ndarray]]:
    """Fit one cross-K feature standardizer using only reliability-train rows."""
    train_parts = []
    for k in ks:
        matrix = np.asarray(raw_by_k[int(k)], dtype=np.float64)
        mask = np.asarray(train_masks[int(k)], dtype=bool)
        if matrix.ndim != 2 or matrix.shape[0] != mask.size:
            raise ValueError(f"K={k}: feature rows and training mask differ")
        if not mask.any():
            raise ValueError(f"K={k}: no reliability-train rows")
        train_parts.append(matrix[mask])
    train = np.vstack(train_parts)
    fit = rfeat.normalize_fit(train, fit_rows=np.arange(train.shape[0]), keys=keys)
    return fit, {int(k): rfeat.normalize_apply(np.asarray(v), fit) for k, v in raw_by_k.items()}


def fit_score_deepsets_normalization(
    scores_by_k: Mapping[int, np.ndarray],
    train_masks: Mapping[int, np.ndarray],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Fit ScoreDeepSets' original global score, logK and top1 scalers on train rows.

    The logK fit intentionally constructs the training rows first. The old
    Phase 0.5 implementation passed indices into concatenated full-K vectors,
    accidentally selecting a K5 prefix instead of the K5/K10 train subset.
    """
    score_train = []
    logk_train = []
    top1_train = []
    for k in ks:
        values = np.asarray(scores_by_k[int(k)], dtype=np.float64)
        mask = np.asarray(train_masks[int(k)], dtype=bool)
        if values.ndim != 2 or values.shape[0] != mask.size:
            raise ValueError(f"K={k}: score rows and training mask differ")
        if not mask.any():
            raise ValueError(f"K={k}: no reliability-train rows")
        score_train.append(values[mask])
        logk_train.append(np.full(int(mask.sum()), np.log(float(k)), dtype=np.float64))
        top1_train.append(np.max(values[mask], axis=1))
    score_mu, score_sigma = rfeat.entry_moments(score_train)
    if not np.isfinite(score_sigma) or score_sigma <= 0.0:
        raise ValueError("training score standard deviation must be positive and finite")
    logk_values = np.concatenate(logk_train)[:, None]
    logk_fit = rfeat.normalize_fit(
        logk_values, fit_rows=np.arange(logk_values.shape[0]), keys=("log_k_std",)
    )
    top1_values = np.concatenate(top1_train)[:, None]
    top1_fit = rfeat.normalize_fit(
        top1_values, fit_rows=np.arange(top1_values.shape[0]), keys=("z_top1_std",)
    )
    score_normalized = {
        int(k): ((np.asarray(values, dtype=np.float64) - score_mu) / (score_sigma + 1e-8)).astype(
            np.float32
        )
        for k, values in scores_by_k.items()
    }
    logk_normalized: dict[int, np.ndarray] = {}
    ztop1_normalized: dict[int, np.ndarray] = {}
    for k, values in scores_by_k.items():
        k = int(k)
        n = np.asarray(values).shape[0]
        logk_normalized[k] = rfeat.normalize_apply(
            np.full((n, 1), np.log(float(k)), dtype=np.float64), logk_fit
        )[:, 0].astype(np.float32)
        ztop1_normalized[k] = rfeat.normalize_apply(
            np.max(np.asarray(values, dtype=np.float64), axis=1)[:, None], top1_fit
        )[:, 0].astype(np.float32)
    return {
        "score_mu": float(score_mu),
        "score_sigma": float(score_sigma),
        "log_k_fit": logk_fit,
        "top1_fit": top1_fit,
        "scores": score_normalized,
        "log_k": logk_normalized,
        "z_top1": ztop1_normalized,
    }


def score_deepsets_inputs(
    normalized: Mapping[str, Any],
    correct_by_k: Mapping[int, np.ndarray],
    masks_by_k: Mapping[int, np.ndarray],
    ks: Sequence[int],
) -> dict[str, np.ndarray]:
    """Pack selected K rows with the original zero-padding and boolean mask."""
    score_parts = []
    logk_parts = []
    top1_parts = []
    label_parts = []
    for k in ks:
        k = int(k)
        mask = np.asarray(masks_by_k[k], dtype=bool)
        score_parts.append(np.asarray(normalized["scores"][k])[mask])
        logk_parts.append(np.asarray(normalized["log_k"][k])[mask])
        top1_parts.append(np.asarray(normalized["z_top1"][k])[mask])
        label_parts.append(np.asarray(correct_by_k[k], dtype=np.float32)[mask])
    padded, valid = rfeat.sds_pack(score_parts)
    return {
        "scores": padded,
        "mask": valid,
        "log_k": np.concatenate(logk_parts).astype(np.float32),
        "z_top1": np.concatenate(top1_parts).astype(np.float32),
        "y": np.concatenate(label_parts).astype(np.float32),
        "n_by_k": {int(k): int(np.asarray(masks_by_k[int(k)], dtype=bool).sum()) for k in ks},
    }


def choose_logistic_c(scores: Mapping[float, float], *, tolerance: float = 0.002) -> float:
    """Choose highest tune AUROC; within the fixed tolerance prefer smaller C."""
    if tuple(sorted(float(c) for c in scores)) != tuple(sorted(LOGISTIC_CS)):
        raise ValueError(f"expected the fixed C grid {LOGISTIC_CS}, got {tuple(scores)}")
    finite = {float(c): float(value) for c, value in scores.items() if np.isfinite(value)}
    if len(finite) != len(LOGISTIC_CS):
        raise ValueError("all fixed C candidates need finite tune AUROC")
    best_score = max(finite.values())
    eligible = [c for c, value in finite.items() if best_score - value < float(tolerance)]
    return min(eligible)


def fit_selected_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    tune_by_k: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    cs: Sequence[float] = LOGISTIC_CS,
    tie_tolerance: float = 0.002,
) -> tuple[LogisticPredictor, dict[str, Any], list[dict[str, Any]]]:
    """Fit the fixed C grid and select by equal-weight tune AUROC over two K cells."""
    if tuple(float(c) for c in cs) != LOGISTIC_CS:
        raise ValueError(f"the repair uses the frozen C grid {LOGISTIC_CS}")
    records: list[dict[str, Any]] = []
    models: dict[float, tuple[LogisticModel, list[tuple[int, np.ndarray, np.ndarray]]]] = {}
    tune_scores: dict[float, float] = {}
    for c in LOGISTIC_CS:
        model = LogisticModel(C=float(c)).fit(x_train, y_train)
        per_k: list[tuple[int, np.ndarray, np.ndarray]] = []
        aurocs: list[float] = []
        tune_probs: list[np.ndarray] = []
        tune_labels: list[np.ndarray] = []
        for k in sorted(tune_by_k):
            x_val, y_val = tune_by_k[k]
            pred = model.predict_proba(x_val)
            per_k.append((int(k), pred, np.asarray(y_val, dtype=bool)))
            aurocs.append(float(auroc_correct(pred, y_val)))
            tune_probs.append(pred)
            tune_labels.append(np.asarray(y_val, dtype=bool))
        mean_auc = float(np.mean(aurocs))
        tune_scores[float(c)] = mean_auc
        train_prob = model.predict_proba(x_train)
        train_auc = float(auroc_correct(train_prob, y_train))
        train_loss = nll_binary(train_prob, y_train)
        tune_loss = nll_binary(np.concatenate(tune_probs), np.concatenate(tune_labels))
        rec = {
            "model_kind": "stats_logistic",
            "hyperparameter": float(c),
            "epoch": 0,
            "train_loss": train_loss,
            "train_auroc": train_auc,
            "tune_loss": tune_loss,
            "tune_auroc": mean_auc,
            "selected": False,
        }
        for idx, k in enumerate(sorted(tune_by_k)):
            rec[f"tune_auroc_K{k}"] = aurocs[idx]
        records.append(rec)
        models[float(c)] = (model, per_k)
    chosen_c = choose_logistic_c(tune_scores, tolerance=tie_tolerance)
    records[LOGISTIC_CS.index(chosen_c)]["selected"] = True
    model = models[chosen_c][0]
    coef, intercept = model.coefficients()
    predictor = LogisticPredictor(
        c=float(chosen_c), coefficients=np.asarray(coef, dtype=np.float64), intercept=float(intercept)
    )
    chosen_record = next(row for row in records if row["selected"])
    selected = {
        "family": "stats_logistic",
        "chosen_C": float(chosen_c),
        "tie_tolerance": float(tie_tolerance),
        "tune_mean_auroc": float(chosen_record["tune_auroc"]),
        "train_loss": float(chosen_record["train_loss"]),
        "train_auroc": float(chosen_record["train_auroc"]),
        "tune_loss": float(chosen_record["tune_loss"]),
        "tune_auroc_by_K": {
            int(k): float(chosen_record[f"tune_auroc_K{k}"]) for k in sorted(tune_by_k)
        },
        "n_train_rows": int(np.asarray(y_train).size),
        "n_parameters": int(model.n_parameters()),
        "coefficients": coef.tolist(),
        "intercept": float(intercept),
    }
    return predictor, selected, records


def load_logistic_predictor(payload: Mapping[str, Any]) -> LogisticPredictor:
    """Restore a selected linear model from a JSON-ready selection record."""
    return LogisticPredictor(
        c=float(payload["chosen_C"]),
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        intercept=float(payload["intercept"]),
    )


def train_score_deepsets_grid(
    train_inputs: Mapping[str, np.ndarray],
    tune_inputs: Mapping[str, np.ndarray],
    tune_by_k: Mapping[int, Mapping[str, np.ndarray]],
    *,
    seed: int,
    learning_rates: Sequence[float] = SDS_LRS,
    epochs: int = 300,
    patience: int = 30,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
) -> tuple[ScoreDeepSets, dict[str, Any], list[dict[str, Any]]]:
    """Train the original ScoreDeepSets architecture/grid and select on tune AUROC."""
    if tuple(float(lr) for lr in learning_rates) != SDS_LRS:
        raise ValueError(f"the repair uses the frozen ScoreDeepSets LR grid {SDS_LRS}")
    records: list[dict[str, Any]] = []
    fitted: dict[float, tuple[ScoreDeepSets, dict[str, Any], float, float, dict[int, float]]] = {}
    for lr in SDS_LRS:
        model = ScoreDeepSets(seed=int(seed), include_logk=True, include_ztop1=True)
        history: list[dict[str, Any]] = []

        def log_epoch(raw: Mapping[str, Any]) -> None:
            # Record train and tune AUROC each epoch while restoring train mode
            # after predict_proba; the model has no dropout/batch-normalization.
            is_training = bool(model._net.training)
            train_prob = model.predict_proba(
                train_inputs["scores"], train_inputs["mask"],
                train_inputs["log_k"], train_inputs["z_top1"],
            )
            tune_prob = model.predict_proba(
                tune_inputs["scores"], tune_inputs["mask"],
                tune_inputs["log_k"], tune_inputs["z_top1"],
            )
            model._net.train(is_training)
            rec = dict(raw)
            rec["train_auroc"] = float(auroc_correct(train_prob, train_inputs["y"]))
            rec["tune_auroc"] = float(auroc_correct(tune_prob, tune_inputs["y"]))
            rec["model_kind"] = "score_deepsets"
            rec["hyperparameter"] = float(lr)
            rec["selected"] = False
            history.append(rec)

        fit_info = model.fit(
            train_inputs["scores"], train_inputs["mask"], train_inputs["log_k"],
            train_inputs["z_top1"], train_inputs["y"], lr=float(lr),
            scores_val=tune_inputs["scores"], mask_val=tune_inputs["mask"],
            log_k_val=tune_inputs["log_k"], z_top1_val=tune_inputs["z_top1"],
            y_val=tune_inputs["y"], epochs=int(epochs), batch_size=int(batch_size),
            weight_decay=float(weight_decay), patience=int(patience), log=log_epoch,
        )
        per_k_auc: dict[int, float] = {}
        per_k_loss: dict[int, float] = {}
        for k, payload in tune_by_k.items():
            pred = model.predict_proba(
                payload["scores"], payload["mask"], payload["log_k"], payload["z_top1"]
            )
            per_k_auc[int(k)] = float(auroc_correct(pred, payload["y"]))
            per_k_loss[int(k)] = nll_binary(pred, payload["y"])
        selection_auc = float(np.mean(list(per_k_auc.values())))
        final_train_prob = model.predict_proba(
            train_inputs["scores"], train_inputs["mask"],
            train_inputs["log_k"], train_inputs["z_top1"],
        )
        final_tune_prob = model.predict_proba(
            tune_inputs["scores"], tune_inputs["mask"],
            tune_inputs["log_k"], tune_inputs["z_top1"],
        )
        records.extend(history)
        fitted[float(lr)] = (
            model,
            fit_info,
            selection_auc,
            nll_binary(final_train_prob, train_inputs["y"]),
            per_k_auc,
        )
        for rec in history:
            rec["selected_epoch"] = False
    # Stable first-maximum keeps the predeclared grid order as the tie-break.
    best_lr = SDS_LRS[0]
    for lr in SDS_LRS[1:]:
        if fitted[float(lr)][2] > fitted[float(best_lr)][2]:
            best_lr = float(lr)
    model, fit_info, tune_mean_auc, train_loss, per_k_auc = fitted[float(best_lr)]
    for rec in records:
        selected_lr = float(rec["hyperparameter"]) == float(best_lr)
        rec["selected"] = selected_lr
        rec["selected_epoch"] = selected_lr and int(rec["epoch"]) == int(fit_info["best_epoch"])
        # Keep both the repository's original NLL names and explicit train/tune
        # aliases in the new curve table for stable downstream schemas.
        rec["train_loss"] = rec.get("train_nll")
        rec["tune_loss"] = rec.get("val_nll")
    train_prob = model.predict_proba(
        train_inputs["scores"], train_inputs["mask"],
        train_inputs["log_k"], train_inputs["z_top1"],
    )
    tune_prob = model.predict_proba(
        tune_inputs["scores"], tune_inputs["mask"],
        tune_inputs["log_k"], tune_inputs["z_top1"],
    )
    selected = {
        "family": "score_deepsets",
        "chosen_learning_rate": float(best_lr),
        "grid": list(SDS_LRS),
        "architecture": {
            "phi": [1, 16, 16],
            "pooling": ["mean", "max", "top1"],
            "head": [50, 32, 1],
            "include_z_top1": True,
            "include_log_k": True,
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "max_epochs": int(epochs),
            "patience": int(patience),
        },
        "tune_mean_auroc": float(tune_mean_auc),
        "tune_auroc_by_K": {int(k): float(v) for k, v in per_k_auc.items()},
        "train_loss": float(train_loss),
        "train_auroc": float(auroc_correct(train_prob, train_inputs["y"])),
        "tune_loss": float(nll_binary(tune_prob, tune_inputs["y"])),
        "tune_auroc": float(auroc_correct(tune_prob, tune_inputs["y"])),
        "best_epoch": int(fit_info["best_epoch"]),
        "epochs_run": int(fit_info["epochs_run"]),
        "stopped_early": bool(fit_info["stopped_early"]),
        "n_train_rows": int(np.asarray(train_inputs["y"]).size),
        "n_tune_rows": int(np.asarray(tune_inputs["y"]).size),
        "n_parameters": int(model.n_parameters()),
        "per_lr_selection_auroc": {str(lr): float(fitted[float(lr)][2]) for lr in SDS_LRS},
    }
    return model, selected, records

