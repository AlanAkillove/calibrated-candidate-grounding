"""Phase 0B - B3 independent (candidate-blind) MLP evaluation + cosine comparison.

This module scores the B3 baseline (:class:`ccg.models.independent.IndependentMLPScorer`)
on the **exact** Phase 0A evaluation cohort and compares it, row for row, with the
frozen cosine audit (``results/phase0a_corrected``).  It re-uses the Phase 0A
interfaces read-only and never re-samples a candidate set:

* the evaluation rows/order come from :func:`ccg.experiment.phase0a.load_cosine_inputs`
  (via :class:`ccg.models.b3_data.B3Corpus`), and the primary cohort is the
  ``common_cohort_rows`` intersection - the same pooled 20,799 sentences of the
  cosine audit;
* the candidate sets are ``C_K = [target] + distractor_order[:K-1]`` of the frozen
  manifests (never rebuilt here).

Raw-score invariance (atol = 0)
-------------------------------
For every sentence the model is run **once** on the *maximal prefix*
``C_sup = [target] + distractor_order[:K_sup-1]`` (``K_sup = min(max(Ks), order+1)``,
``C_50`` in the frozen four-K audit).  Every reported ``K`` is then a pure *gather*
of that one score table (:func:`gather_superset_for_k`): scoring each ``K``
separately would run differently shaped GEMMs and break the bit-identity the
``atol=0`` check insists on.  The gather is a module-level hook so a test can
inject a deliberate inconsistency.

Probability variants (B3 has no cosine logit scale)
---------------------------------------------------
Three variants are reported side by side, never collapsed:

``native``
    ``P = softmax(s)`` (the MLP's own logits, ``T = 1``).  **Primary variant**.
``global_T_corrected``
    ``P = softmax(s / T*)`` with **one** ``T*`` fitted on **val_calib** only,
    pooled over ``K in {5, 10}`` (protocol section 22).  The corrected fit uses
    :func:`ccg.calibration.temperature_opt.fit_temperature_log_space`
    (log-space, bound-collision aware).
``oracle_T_K``
    ``P = softmax(s / T_K)`` with a per-``K`` temperature fitted on the val_calib
    rows of that ``K`` - **ORACLE / DIAGNOSTIC**, never a legal model result.

Calibration isolation (hard requirement)
----------------------------------------
Only ``val_calib`` rows may ever enter a fit; the fitting helpers re-assert this
on their inputs and raise :class:`ccg.experiment.phase0a.CalibrationIsolationError`
on any other eval split.  testA / testB and ``K in {20, 50}`` never participate.

Hard sanity checks / STOP contract
----------------------------------
Before any result artifact is written three checks run per seed:

1. **score invariance** - shared candidates are bit-identical across K (``atol=0``);
2. **rank monotonicity** - ``rank(K') >= rank(K)`` on the nested sets;
3. **accuracy monotonicity** - ``Acc(K5) >= Acc(K10) >= Acc(K20) >= Acc(K50)``.

A violation writes ``seed_{s}/VALIDATION_FAILURE.json`` and raises
``SystemExit(2)`` - a broken driver never produces plottable numbers.

Artifacts (``results/phase0b_independent``)
-------------------------------------------
``seed_{s}/raw_scores/K{K}.npz`` (raw logits - never dropped),
``ranking_metrics.csv``, ``calibration_metrics.csv``,
``normalized_selective_metrics.csv``, ``reliability_bins.csv``,
``diagnostics.csv``, ``bootstrap.csv``, ``per_sentence_predictions.npz``,
``eval_metadata.json``; and the aggregate ``aggregate.csv``,
``paired_vs_cosine.csv`` and ``metadata.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccg.calibration.temperature_opt import (
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    TemperatureFit,
    fit_temperature_log_space,
)
from ccg.experiment import phase0a
from ccg.metrics import (
    aurc,
    brier_binary,
    confidence_accuracy_gap,
    mrr,
    multiclass_brier,
    multiclass_nll,
    nll_binary,
    reliability_table,
    risk_at_coverage,
    risk_coverage_curve,
    target_rank,
    top1_accuracy,
    top_label_ece,
    top_label_ece_adaptive,
    topk_accuracy,
)
from ccg.metrics.discrimination import auprc_correct, auroc_correct
from ccg.metrics.selective import e_aurc, oracle_aurc, rer_at_coverage
from ccg.models.b3_data import (
    B3Corpus,
    GEO_DIM,
    IMAGE_SIZES_FILENAME,
    build_image_sizes,
)
from ccg.models.independent import IndependentMLPScorer, build_candidate_inputs

__all__ = [
    "B3_VARIANTS",
    "B3_VARIANT_GLOBAL_CORRECTED",
    "B3_VARIANT_NATIVE",
    "B3_VARIANT_ORACLE_K",
    "CONFIDENCE_BINS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_KS",
    "DEFAULT_SEEDS",
    "EVAL_METADATA_NAME",
    "GEO_DIM",
    "MODEL_FILENAME",
    "PRIMARY_SPLITS",
    "RAW_SCORES_DIRNAME",
    "RER_LEVELS",
    "TRAINING_FILENAME",
    "fit_corrected_global_temperature",
    "fit_oracle_per_k_temperature",
    "gather_superset_for_k",
    "load_b3_model",
    "paired_cluster_bootstrap_ratio",
    "run_b3_eval",
    "score_sets_independent",
]

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_KS: Tuple[int, ...] = phase0a.DEFAULT_KS
PRIMARY_SPLITS: Tuple[str, ...] = phase0a.PRIMARY_SPLITS
REGIME_RANDOM: str = phase0a.REGIME_RANDOM
LEGAL_CALIBRATION_KS: Tuple[int, ...] = phase0a.LEGAL_CALIBRATION_KS
RELIABILITY_BINS: int = phase0a.RELIABILITY_BINS
FAILURE_FILENAME: str = phase0a.FAILURE_FILENAME
SPLIT_CODES: Mapping[str, int] = phase0a.SPLIT_CODES

DEFAULT_SEEDS: Tuple[int, ...] = (1, 2, 3)
DEFAULT_BATCH_SIZE = 128
DEFAULT_OUT = "results/phase0b_independent"
#: Relative location of the frozen cosine per-sentence predictions.
COSINE_PREDICTIONS = "results/phase0a_corrected/per_sentence_predictions.npz"

#: B3 probability variants (report column order).
B3_VARIANT_NATIVE = "native"
B3_VARIANT_GLOBAL_CORRECTED = "global_T_corrected"
B3_VARIANT_ORACLE_K = "oracle_T_K"
B3_VARIANTS: Tuple[str, ...] = (
    B3_VARIANT_NATIVE,
    B3_VARIANT_GLOBAL_CORRECTED,
    B3_VARIANT_ORACLE_K,
)
#: Coverage levels of the reported risk-erosion ratios.
RER_LEVELS: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
#: Equal-width confidence bins of the distribution diagnostic.
CONFIDENCE_BINS = 10

#: Artifact file names inside ``seed_{s}/``.
RAW_SCORES_DIRNAME = "raw_scores"
MODEL_FILENAME = "model.npz"
TRAINING_FILENAME = "training.json"
EVAL_METADATA_NAME = "eval_metadata.json"
PER_SENTENCE_NAME = "per_sentence_predictions.npz"

_POOLED = phase0a._pooled_name()


# ---------------------------------------------------------------------------
# superset scoring (raw-score invariance by construction)
# ---------------------------------------------------------------------------
def gather_superset_for_k(
    superset_scores: Any,
    superset_candidates: Any,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather the ``C_K`` raw scores of one sentence from its superset score table.

    The candidate sets are nested, so ``C_K = [target] + distractor_order[:K-1]``
    is exactly the length-``k`` prefix of the scored superset
    ``C_sup = [target] + distractor_order[:K_sup-1]``.  Returning a *slice* of
    the one score table (never a re-scored set) is what makes every shared
    candidate bit-identical across ``K`` (``atol=0``).

    Module-level hook: :func:`score_sets_independent` resolves it through the
    module namespace, so a test can monkeypatch ``phase0b.gather_superset_for_k``
    to inject a deliberate per-``K`` inconsistency.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(scores_k[float32, k], candidate_indices_k[int64, k])``.
    """
    kk = int(k)
    if kk < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    scores = np.asarray(superset_scores, dtype=np.float32).reshape(-1)
    candidates = np.asarray(superset_candidates, dtype=np.int64).reshape(-1)
    if scores.shape[0] < kk or candidates.shape[0] < kk:
        raise ValueError(
            f"superset has {scores.shape[0]} scores / {candidates.shape[0]} candidates "
            f"but K={kk} was requested"
        )
    return (
        np.array(scores[:kk], dtype=np.float32, copy=True),
        np.array(candidates[:kk], dtype=np.int64, copy=True),
    )


def _forward_superset(
    model: IndependentMLPScorer,
    queries: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    geometries: Sequence[np.ndarray],
) -> np.ndarray:
    """One batched MLP forward of ``B`` supersets -> ``[B, K_sup]`` float32 logits.

    Every superset in the batch shares one ``K_sup`` (the caller buckets by
    ``K_sup``), so the ``[B, K_sup, d_in]`` tensor stacks cleanly.  The network
    is row-wise, hence batching across sentences changes nothing but speed.
    """
    import torch  # local import keeps the numpy-only import path light

    rows = [
        build_candidate_inputs(query, cand, geo, model.geo_dim)
        for query, cand, geo in zip(queries, candidates, geometries)
    ]
    stacked = np.stack(rows, axis=0).astype(np.float32, copy=False)
    tensor = torch.as_tensor(stacked, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        out = model.net(tensor).reshape(len(rows), -1)
    logits = out.detach().cpu().numpy().astype(np.float32, copy=False)
    temperature = float(getattr(model, "temperature", 1.0))
    if temperature != 1.0:
        logits = (logits / temperature).astype(np.float32, copy=False)
    return logits


def score_sets_independent(
    corpus: B3Corpus,
    records: Sequence[Any],
    model: IndependentMLPScorer,
    ks: Sequence[int] = DEFAULT_KS,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gather_fn: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> phase0a.ScoreBundle:
    """Score every sentence's nested ``C_K`` from one superset forward per sentence.

    Membership, row order and skip counting mirror :func:`ccg.experiment.phase0a.score_sets`
    exactly, so the resulting :class:`ccg.experiment.phase0a.ScoreBundle` feeds the
    Phase 0A sanity checks, cohort helpers and metric code unchanged.
    """
    record_list = list(records)
    Ks = tuple(sorted({int(k) for k in ks}))
    if not Ks:
        raise ValueError("ks is empty")
    if any(k < 2 for k in Ks):
        raise ValueError(f"every K must be >= 2 (candidate set includes the target), got {Ks}")
    max_k = max(Ks)

    member_rows: Dict[int, List[int]] = {k: [] for k in Ks}
    row_of_record: Dict[int, np.ndarray] = {
        k: np.full(len(record_list), -1, dtype=np.int64) for k in Ks
    }
    skipped_target: Dict[int, int] = {k: 0 for k in Ks}
    skipped_insufficient: Dict[int, int] = {k: 0 for k in Ks}

    for pos, record in enumerate(record_list):
        order_size = int(np.asarray(record.distractor_order).size)
        for k in Ks:
            if record.target_index is None:
                skipped_target[k] += 1
                continue
            if order_size < k - 1:
                skipped_insufficient[k] += 1
                continue
            row_of_record[k][pos] = len(member_rows[k])
            member_rows[k].append(pos)

    raw_scores: Dict[int, np.ndarray] = {}
    candidate_indices: Dict[int, np.ndarray] = {}
    target_local: Dict[int, np.ndarray] = {}
    for k in Ks:
        n = len(member_rows[k])
        raw_scores[k] = np.empty((n, k), dtype=np.float32)
        candidate_indices[k] = np.empty((n, k), dtype=np.int32)
        target_local[k] = np.empty(n, dtype=np.int32)

    gather = gather_superset_for_k if gather_fn is None else gather_fn

    active = [
        pos for pos in range(len(record_list)) if any(row_of_record[k][pos] >= 0 for k in Ks)
    ]
    k_sup_of: Dict[int, int] = {}
    buckets: Dict[int, List[int]] = {}
    for pos in active:
        order_size = int(np.asarray(record_list[pos].distractor_order).size)
        k_sup = min(max_k, order_size + 1)
        k_sup_of[pos] = k_sup
        buckets.setdefault(k_sup, []).append(pos)

    batch_size = max(1, int(batch_size))
    planned = sum((len(v) + batch_size - 1) // batch_size for v in buckets.values())
    done = 0
    for k_sup in sorted(buckets):
        positions = buckets[k_sup]
        for start in range(0, len(positions), batch_size):
            chunk = positions[start : start + batch_size]
            queries: List[np.ndarray] = []
            candidates: List[np.ndarray] = []
            geometries: List[np.ndarray] = []
            sup_cands: List[np.ndarray] = []
            for pos in chunk:
                record = record_list[pos]
                example = corpus.example_for(record, k_sup)
                queries.append(example.query_feature)
                candidates.append(example.candidate_features)
                geometries.append(example.geometry)
                sup_cands.append(B3Corpus.candidate_bank_indices(record, k_sup))
            supersets = _forward_superset(model, queries, candidates, geometries)
            if supersets.shape != (len(chunk), k_sup):
                raise ValueError(
                    f"superset forward returned {supersets.shape}, expected "
                    f"({len(chunk)}, {k_sup})"
                )
            for local_i, pos in enumerate(chunk):
                record = record_list[pos]
                sup_scores = supersets[local_i]
                sup_cand = sup_cands[local_i]
                for k in Ks:
                    row = int(row_of_record[k][pos])
                    if row < 0:
                        continue
                    scores_k, cand_k = gather(sup_scores, sup_cand, k)
                    scores_k = np.asarray(scores_k, dtype=np.float32).reshape(-1)
                    if scores_k.shape[0] != k:
                        raise ValueError(
                            f"gather hook returned {scores_k.shape[0]} scores for K={k} of "
                            f"sentence {record.sentence_id}"
                        )
                    if not np.all(np.isfinite(scores_k)):
                        raise ValueError(
                            f"non-finite score for sentence {record.sentence_id}, K={k}"
                        )
                    local = phase0a.target_local_index(cand_k, int(record.target_index))
                    raw_scores[k][row] = scores_k
                    candidate_indices[k][row] = np.asarray(cand_k, dtype=np.int32)
                    target_local[k][row] = int(local)
            done += 1
            if on_progress is not None:
                on_progress(done, planned, f"K_sup={k_sup}")

    records_by_k = {k: [record_list[pos] for pos in member_rows[k]] for k in Ks}
    return phase0a.ScoreBundle(
        ks=Ks,
        records_by_k=records_by_k,
        raw_scores=raw_scores,
        candidate_indices=candidate_indices,
        target_local=target_local,
        n_skipped_target_missing=skipped_target,
        n_skipped_insufficient=skipped_insufficient,
        n_l2_renormalized=0,
    )


# ---------------------------------------------------------------------------
# temperature fitting (val_calib only; hard isolation)
# ---------------------------------------------------------------------------
def _assert_val_calib(bundle: phase0a.ScoreBundle, k: int, rows: np.ndarray) -> None:
    """Raise :class:`CalibrationIsolationError` unless every row is ``val_calib``."""
    splits = bundle.splits(int(k))[rows]
    offenders = sorted(
        {str(item) for item in np.unique(splits).tolist() if str(item) != "val_calib"}
    )
    if offenders:
        raise phase0a.CalibrationIsolationError(
            f"calibration fitting may only see val_calib rows, but K={int(k)} rows carry "
            f"eval splits {offenders}; testA/testB and K in {{20, 50}} never participate "
            "(protocol section 9). Restrict the input to val_calib first."
        )


def fit_corrected_global_temperature(
    bundle: phase0a.ScoreBundle,
    *,
    ks: Sequence[int] = LEGAL_CALIBRATION_KS,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    rows_by_k: Optional[Mapping[int, np.ndarray]] = None,
) -> TemperatureFit:
    """Fit **one** corrected global temperature on val_calib (``K in {5, 10}``).

    Default rows = the ``common_cohort_rows`` of ``ks`` restricted to ``val_calib``
    (a single shared sentence cohort, so the pooled fit is honest).  Passing
    ``rows_by_k`` replaces them but the calibration-isolation assertion still runs
    on exactly those rows - any non-``val_calib`` split raises
    :class:`ccg.experiment.phase0a.CalibrationIsolationError`.
    """
    ks_t = tuple(sorted({int(k) for k in ks}))
    if not ks_t:
        raise ValueError("ks is empty")
    if not set(ks_t) <= set(LEGAL_CALIBRATION_KS):
        raise ValueError(
            f"a legal corrected temperature is fitted on K in {LEGAL_CALIBRATION_KS} only; "
            f"got K={list(ks_t)} (per-K fitting belongs to the oracle diagnostic)"
        )
    for k in ks_t:
        if k not in bundle.raw_scores:
            raise KeyError(f"K={k} is not part of this bundle: {bundle.ks}")

    if rows_by_k is None:
        _, common_rows, _ = phase0a.common_cohort_rows(bundle, ks_t)
        rows: Dict[int, np.ndarray] = {
            k: common_rows[k][bundle.splits(k)[common_rows[k]] == "val_calib"] for k in ks_t
        }
    else:
        rows = {}
        for k in ks_t:
            if k not in rows_by_k:
                raise KeyError(f"rows_by_k misses K={k}")
            rows[k] = np.asarray(rows_by_k[k], dtype=np.int64).reshape(-1)

    for k in ks_t:
        _assert_val_calib(bundle, k, rows[k])

    sets: List[np.ndarray] = []
    labels: List[int] = []
    for k in ks_t:
        for row in bundle.raw_scores[k][rows[k]]:
            sets.append(row)
        labels.extend(int(value) for value in bundle.target_local[k][rows[k]].tolist())
    if not sets:
        raise ValueError("no val_calib candidate sets to fit the corrected temperature on")
    return fit_temperature_log_space(sets, labels, t_min=float(t_min), t_max=float(t_max))


def fit_oracle_per_k_temperature(
    bundle: phase0a.ScoreBundle,
    k: int,
    *,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    rows: Optional[np.ndarray] = None,
) -> TemperatureFit:
    """Fit the per-``K`` **oracle/diagnostic** temperature on the val_calib rows of ``k``.

    ORACLE: ``T_K`` is fitted on the same val_calib rows it is evaluated on and is
    never a legal model result.  The rows are asserted ``val_calib`` regardless.
    """
    kk = int(k)
    if kk not in bundle.raw_scores:
        raise KeyError(f"K={kk} is not part of this bundle: {bundle.ks}")
    if rows is None:
        rows_arr = bundle.rows_for_split(kk, "val_calib")
    else:
        rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
    _assert_val_calib(bundle, kk, rows_arr)
    sets = [row for row in bundle.raw_scores[kk][rows_arr]]
    labels = [int(value) for value in bundle.target_local[kk][rows_arr].tolist()]
    if not sets:
        raise ValueError(f"K={kk}: no val_calib rows to fit the oracle temperature on")
    return fit_temperature_log_space(sets, labels, t_min=float(t_min), t_max=float(t_max))


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------
def load_b3_model(
    seed_dir: Any,
    *,
    device: str = "cpu",
) -> Tuple[IndependentMLPScorer, Dict[str, Any]]:
    """Rebuild one B3 checkpoint from ``seed_{s}/model.npz`` + ``training.json``.

    ``training.json`` carries ``feature_dim`` / ``hidden_dim`` / ``geo_dim`` /
    ``temperature`` (written by ``scripts/train_b3.py``); the state dict alone
    cannot restore ``geo_dim``, so the config is authoritative.  A missing or
    incomplete checkpoint raises instead of being silently defaulted.
    """
    seed_dir = Path(seed_dir)
    model_path = seed_dir / MODEL_FILENAME
    train_path = seed_dir / TRAINING_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"B3 checkpoint {model_path} not found")
    if not train_path.exists():
        raise FileNotFoundError(f"B3 training metadata {train_path} not found")
    payload = json.loads(train_path.read_text(encoding="utf-8"))
    feature_dim = int(payload.get("feature_dim", 512))
    hidden_dim = int(payload.get("hidden_dim", 128))
    geo_dim = int(payload.get("geo_dim", GEO_DIM))
    temperature = float(payload.get("temperature", 1.0))
    seed = int(payload.get("seed", 0))
    model = IndependentMLPScorer(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        geo_dim=geo_dim,
        temperature=temperature,
        device=device,
        seed=seed,
    )
    model.load(model_path)
    return model, payload


# ---------------------------------------------------------------------------
# common-cohort views
# ---------------------------------------------------------------------------
def _common_view(
    bundle: phase0a.ScoreBundle,
    ks: Sequence[int],
) -> Tuple[Tuple[int, ...], Dict[int, np.ndarray], np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Aligned common-cohort view: ``(Ks, rows, sentence_ids, splits, image_ids)``.

    ``rows[k]`` index ``bundle.raw_scores[k]``; row ``i`` of every K is the *same*
    sentence (``sentence_ids[i]``), so cross-K comparisons share one denominator.
    """
    Ks, common_rows, common_ids = phase0a.common_cohort_rows(bundle, ks)
    reference = Ks[0]
    splits = np.asarray(bundle.splits(reference))[common_rows[reference]]
    sids = bundle.sentence_ids(reference)[common_rows[reference]]
    if not np.array_equal(sids, common_ids):
        raise RuntimeError("common-cohort sentence alignment failed")
    images = {k: bundle.image_ids(k)[common_rows[k]] for k in Ks}
    return Ks, common_rows, common_ids, splits, images


def _name_mask(name: str, splits: np.ndarray) -> np.ndarray:
    if name == _POOLED:
        return np.ones(splits.shape[0], dtype=bool)
    return splits == name


# ---------------------------------------------------------------------------
# metric cells
# ---------------------------------------------------------------------------
def _probabilities(
    bundle: phase0a.ScoreBundle,
    k: int,
    rows: np.ndarray,
    variant: str,
    temps: Mapping[str, Any],
) -> np.ndarray:
    """Probabilities of one (K, row-set) cell under a B3 variant."""
    raw = bundle.raw_scores[int(k)][rows]
    if variant == B3_VARIANT_NATIVE:
        return phase0a.probabilities_for_variant(raw, phase0a.VARIANT_T1)
    if variant == B3_VARIANT_GLOBAL_CORRECTED:
        return phase0a.probabilities_for_variant(
            raw, phase0a.VARIANT_GLOBAL, temperature=float(temps["global"])
        )
    if variant == B3_VARIANT_ORACLE_K:
        oracle = temps["oracle"]
        if int(k) not in oracle:
            raise KeyError(f"oracle_T_K variant needs a fitted T for K={int(k)}")
        return phase0a.probabilities_for_variant(
            raw, phase0a.VARIANT_GLOBAL, temperature=float(oracle[int(k)])
        )
    raise ValueError(f"unknown B3 variant {variant!r}; expected one of {B3_VARIANTS}")


def _variant_temperature(variant: str, k: int, temps: Mapping[str, Any]) -> float:
    if variant == B3_VARIANT_NATIVE:
        return 1.0
    if variant == B3_VARIANT_GLOBAL_CORRECTED:
        return float(temps["global"])
    if variant == B3_VARIANT_ORACLE_K:
        return float(temps["oracle"][int(k)])
    raise ValueError(f"unknown B3 variant {variant!r}")


def _mean_margin(prob: np.ndarray) -> float:
    """Mean ``top1 - top2`` probability margin (nan for a single candidate)."""
    if prob.shape[1] < 2:
        return float("nan")
    part = np.partition(prob.astype(np.float64), -2, axis=1)
    return float(np.mean(part[:, -1] - part[:, -2]))


def _entropy_mean(prob: np.ndarray, eps: float = 1e-12) -> float:
    clipped = np.clip(prob.astype(np.float64), eps, 1.0)
    return float(np.mean(-np.sum(clipped * np.log(clipped), axis=1)))


def _names_of(bundle: phase0a.ScoreBundle, splits: Optional[Sequence[str]]) -> List[str]:
    return list(phase0a._split_names_of(bundle, splits)) + [_POOLED]


def ranking_rows(
    bundle: phase0a.ScoreBundle,
    names: Sequence[str],
    splits: np.ndarray,
    common_rows: Mapping[int, np.ndarray],
    ks: Sequence[int],
) -> List[Dict[str, Any]]:
    """``ranking_metrics.csv`` rows (variant-invariant: softmax is monotone)."""
    rows: List[Dict[str, Any]] = []
    for name in names:
        mask = _name_mask(name, splits)
        for k in ks:
            row_set = common_rows[int(k)][mask]
            if row_set.size == 0:
                continue
            cell = phase0a._ranking_cell(bundle, int(k), row_set)
            rows.append({"eval_split": name, "K": int(k), "denominator": "common", **cell})
    return rows


def calibration_rows(
    bundle: phase0a.ScoreBundle,
    names: Sequence[str],
    splits: np.ndarray,
    common_rows: Mapping[int, np.ndarray],
    ks: Sequence[int],
    temps: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """``calibration_metrics.csv`` rows: split x K x variant (common cohort)."""
    rows: List[Dict[str, Any]] = []
    for name in names:
        mask = _name_mask(name, splits)
        for k in ks:
            row_set = common_rows[int(k)][mask]
            if row_set.size == 0:
                continue
            raw = bundle.raw_scores[int(k)][row_set]
            targets = bundle.target_local[int(k)][row_set]
            correct = phase0a._correctness(bundle, int(k), row_set)
            for variant in B3_VARIANTS:
                prob = _probabilities(bundle, int(k), row_set, variant, temps)
                confidence = prob.max(axis=1)
                temp = _variant_temperature(variant, int(k), temps)
                rows.append(
                    {
                        "eval_split": name,
                        "K": int(k),
                        "variant": variant,
                        "denominator": "common",
                        "n": int(row_set.size),
                        "accuracy": float(np.mean(correct)),
                        "ece_adaptive": float(
                            top_label_ece_adaptive(confidence, correct, n_bins=RELIABILITY_BINS)
                        ),
                        "ece_equal_width": float(
                            top_label_ece(confidence, correct, n_bins=RELIABILITY_BINS)
                        ),
                        "brier_binary": float(brier_binary(confidence, correct)),
                        "nll_binary": float(nll_binary(confidence, correct)),
                        "conf_acc_gap": float(confidence_accuracy_gap(confidence, correct)),
                        "multiclass_nll": float(
                            multiclass_nll(raw, targets, temperature=temp)
                        ),
                        "multiclass_brier": float(
                            multiclass_brier(raw, targets, temperature=temp)
                        ),
                        "mean_entropy": _entropy_mean(prob),
                    }
                )
    return rows


def selective_rows(
    bundle: phase0a.ScoreBundle,
    names: Sequence[str],
    splits: np.ndarray,
    common_rows: Mapping[int, np.ndarray],
    ks: Sequence[int],
    temps: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """``normalized_selective_metrics.csv`` rows: split x K x variant."""
    rows: List[Dict[str, Any]] = []
    for name in names:
        mask = _name_mask(name, splits)
        for k in ks:
            row_set = common_rows[int(k)][mask]
            if row_set.size < 2:
                continue
            correct = phase0a._correctness(bundle, int(k), row_set)
            accuracy = float(np.mean(correct))
            for variant in B3_VARIANTS:
                prob = _probabilities(bundle, int(k), row_set, variant, temps)
                confidence = prob.max(axis=1)
                coverage, risk = risk_coverage_curve(confidence, correct)
                aurc_value = float(aurc(coverage, risk))
                entry: Dict[str, Any] = {
                    "eval_split": name,
                    "K": int(k),
                    "variant": variant,
                    "denominator": "common",
                    "n": int(row_set.size),
                    "accuracy": accuracy,
                    "aurc": aurc_value,
                    "aurc_oracle": float(oracle_aurc(1.0 - accuracy)),
                    "e_aurc": float(e_aurc(aurc_value, accuracy)),
                    "auroc_correct": float(auroc_correct(confidence, correct)),
                    "auprc_correct": float(auprc_correct(confidence, correct)),
                }
                for level in RER_LEVELS:
                    entry[f"rer_at_{int(round(level * 100))}"] = float(
                        rer_at_coverage(confidence, correct, level)
                    )
                rows.append(entry)
    return rows


def reliability_rows(
    bundle: phase0a.ScoreBundle,
    names: Sequence[str],
    splits: np.ndarray,
    common_rows: Mapping[int, np.ndarray],
    ks: Sequence[int],
    temps: Mapping[str, Any],
    *,
    n_bins: int = RELIABILITY_BINS,
) -> List[Dict[str, Any]]:
    """``reliability_bins.csv`` rows: split x K x variant x binning (fixed + equal-mass)."""
    rows: List[Dict[str, Any]] = []
    for name in names:
        mask = _name_mask(name, splits)
        for k in ks:
            row_set = common_rows[int(k)][mask]
            if row_set.size == 0:
                continue
            correct = phase0a._correctness(bundle, int(k), row_set)
            for variant in B3_VARIANTS:
                prob = _probabilities(bundle, int(k), row_set, variant, temps)
                confidence = prob.max(axis=1)
                for adaptive, label in ((False, "equal_width"), (True, "equal_mass")):
                    table = reliability_table(
                        confidence, correct, n_bins=int(n_bins), adaptive=adaptive
                    )
                    edges = np.asarray(table["bin_edges"], dtype=np.float64)
                    for index in range(int(table["n_bins"])):
                        rows.append(
                            {
                                "eval_split": name,
                                "K": int(k),
                                "variant": variant,
                                "binning": label,
                                "bin_index": int(index),
                                "bin_low": float(edges[index]),
                                "bin_high": float(edges[index + 1]),
                                "bin_confidence": float(table["avg_confidence"][index]),
                                "bin_accuracy": float(table["accuracy"][index]),
                                "bin_count": int(table["counts"][index]),
                            }
                        )
    return rows


def diagnostics_rows(
    bundle: phase0a.ScoreBundle,
    names: Sequence[str],
    splits: np.ndarray,
    common_rows: Mapping[int, np.ndarray],
    ks: Sequence[int],
    temps: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """``diagnostics.csv`` rows: confidence distribution (10 bins) + entropy + margin."""
    rows: List[Dict[str, Any]] = []
    for name in names:
        mask = _name_mask(name, splits)
        for k in ks:
            row_set = common_rows[int(k)][mask]
            if row_set.size == 0:
                continue
            for variant in B3_VARIANTS:
                prob = _probabilities(bundle, int(k), row_set, variant, temps)
                confidence = prob.max(axis=1)
                bins = np.clip(
                    np.floor(confidence * CONFIDENCE_BINS).astype(np.int64),
                    0,
                    CONFIDENCE_BINS - 1,
                )
                counts = np.bincount(bins, minlength=CONFIDENCE_BINS).astype(np.int64)
                entry: Dict[str, Any] = {
                    "eval_split": name,
                    "K": int(k),
                    "variant": variant,
                    "n": int(row_set.size),
                    "mean_entropy": _entropy_mean(prob),
                    "mean_margin": _mean_margin(prob),
                }
                for index in range(CONFIDENCE_BINS):
                    entry[f"conf_bin_{index}_count"] = int(counts[index])
                rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# bootstrap metric registry (SampleStats -> float)
# ---------------------------------------------------------------------------
def _e_aurc_metric(stats: phase0a.SampleStats) -> float:
    """Excess AURC of a (resampled) stats block."""
    accuracy = float(np.mean(stats.correct))
    coverage, risk = risk_coverage_curve(stats.confidence, stats.correct)
    return float(e_aurc(float(aurc(coverage, risk)), accuracy))


def _auroc_metric(stats: phase0a.SampleStats) -> float:
    return float(auroc_correct(stats.confidence, stats.correct))


def _make_rer_metric(level: float) -> Callable[[phase0a.SampleStats], float]:
    def metric(stats: phase0a.SampleStats) -> float:
        return float(rer_at_coverage(stats.confidence, stats.correct, float(level)))

    metric.__name__ = f"rer_at_{int(round(level * 100))}"
    return metric


def _stats_for(
    bundle: phase0a.ScoreBundle,
    k: int,
    rows: np.ndarray,
    variant: str,
    temps: Mapping[str, Any],
) -> phase0a.SampleStats:
    prob = _probabilities(bundle, int(k), rows, variant, temps)
    return phase0a.SampleStats.from_conf_correct(
        prob.max(axis=1), phase0a._correctness(bundle, int(k), rows)
    )


# ---------------------------------------------------------------------------
# paired cluster bootstrap of a *relative* difference
# ---------------------------------------------------------------------------
def paired_cluster_bootstrap_ratio(
    metric_fn: Callable[[phase0a.SampleStats], float],
    pred_a: Any,
    pred_b: Any,
    cluster_ids: Any = None,
    *,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    metric_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Image-clustered paired bootstrap of the *relative* gap ``(A - B) / B``.

    Same paired design as :func:`ccg.experiment.phase0a.paired_cluster_bootstrap`
    (A and B share every draw), but the replicate statistic is the ratio so the
    E-AURC comparison is scale-free.  Replicates where ``B == 0`` (degenerate
    denominator) are dropped from the quantile; ``mean_a`` / ``mean_b`` stay the
    raw point estimates of the two metrics.
    """
    a = pred_a if isinstance(pred_a, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*pred_a)
    b = pred_b if isinstance(pred_b, phase0a.SampleStats) else phase0a.SampleStats.from_conf_correct(*pred_b)
    if len(a) != len(b):
        raise ValueError(f"pred_a has {len(a)} rows but pred_b has {len(b)}")
    n = len(a)
    if n < 2:
        raise ValueError("paired bootstrap needs at least 2 rows")
    reps = int(n_replicates)
    if reps < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    level = float(ci)
    if not 0.0 < level < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    point_a = float(metric_fn(a))
    point_b = float(metric_fn(b))
    rng = np.random.default_rng(int(seed))
    if cluster_ids is None:
        resample_unit = "sample"
        sampler = None
        n_clusters = n
    else:
        clusters = np.asarray(cluster_ids).reshape(-1)
        if clusters.shape[0] != n:
            raise ValueError(
                f"cluster_ids has {clusters.shape[0]} entries but predictions have {n} rows"
            )
        sampler = phase0a._ClusterSampler(clusters)
        resample_unit = "image"
        n_clusters = sampler.n_clusters

    replicates = np.full(reps, np.nan, dtype=np.float64)
    for index in range(reps):
        draw = rng.integers(0, n, size=n) if sampler is None else sampler.draw(rng)
        value_a = float(metric_fn(a.take(draw)))
        value_b = float(metric_fn(b.take(draw)))
        if value_b != 0.0:
            replicates[index] = (value_a - value_b) / value_b
    valid = replicates[np.isfinite(replicates)]
    if valid.size:
        alpha = (1.0 - level) / 2.0
        ci_low = float(np.quantile(valid, alpha))
        ci_high = float(np.quantile(valid, 1.0 - alpha))
    else:
        ci_low = ci_high = float("nan")
    return {
        "metric": metric_name or getattr(metric_fn, "__name__", "metric"),
        "diff": float((point_a - point_b) / point_b) if point_b != 0.0 else float("nan"),
        "mean_a": point_a,
        "mean_b": point_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": level,
        "std_diff": float(np.nanstd(valid)) if valid.size else float("nan"),
        "n": int(n),
        "n_clusters": int(n_clusters),
        "resample_unit": resample_unit,
        "n_replicates": reps,
        "n_valid_replicates": int(valid.size),
    }


#: Fitting-free bootstrap metric registry: ``(name, fn, kind)``.
def _bootstrap_metric_specs() -> List[Tuple[str, Callable[[phase0a.SampleStats], float], str]]:
    return [
        ("e_aurc", _e_aurc_metric, "absolute"),
        ("e_aurc", _e_aurc_metric, "relative"),
        ("auroc_correct", _auroc_metric, "absolute"),
        ("rer_at_50", _make_rer_metric(0.50), "absolute"),
        ("rer_at_80", _make_rer_metric(0.80), "absolute"),
        ("ece_adaptive", phase0a.ece_metric, "absolute"),
    ]


def bootstrap_rows(
    bundle: phase0a.ScoreBundle,
    names: Sequence[str],
    splits: np.ndarray,
    common_rows: Mapping[int, np.ndarray],
    images: Mapping[int, np.ndarray],
    ks: Sequence[int],
    temps: Mapping[str, Any],
    *,
    replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """``bootstrap.csv`` rows: ``K=5`` vs ``K in {10, 20, 50}`` per split x variant.

    Metrics: ``E-AURC`` (absolute **and** relative), ``AUROC_correct``,
    ``RER@50`` / ``RER@80`` and adaptive ``ECE`` - each with an image-clustered
    paired CI.  The E-AURC ratio is the headline cross-K normalised contrast.
    """
    ks_t = tuple(sorted({int(k) for k in ks}))
    if 5 not in ks_t:
        raise ValueError(f"the K=5 baseline is required for the pairwise bootstrap, got {ks_t}")
    others = [k for k in ks_t if k != 5]
    specs = _bootstrap_metric_specs()

    planned = 0
    for name in names:
        mask = _name_mask(name, splits)
        for k_b in others:
            if common_rows[5][mask].size >= 2 and common_rows[k_b][mask].size >= 2:
                planned += len(B3_VARIANTS) * len(specs)

    stats_cache: Dict[Tuple[str, int, str], phase0a.SampleStats] = {}

    def stats_for(name: str, k: int, variant: str) -> phase0a.SampleStats:
        key = (name, int(k), variant)
        if key not in stats_cache:
            mask = _name_mask(name, splits)
            stats_cache[key] = _stats_for(bundle, int(k), common_rows[int(k)][mask], variant, temps)
        return stats_cache[key]

    rows: List[Dict[str, Any]] = []
    for name in names:
        mask = _name_mask(name, splits)
        if common_rows[5][mask].size < 2:
            continue
        clusters = images[5][mask]
        for k_b in others:
            if common_rows[k_b][mask].size < 2:
                continue
            for variant in B3_VARIANTS:
                stats_a = stats_for(name, 5, variant)
                stats_b = stats_for(name, k_b, variant)
                for metric_key, metric_fn, kind in specs:
                    if kind == "relative":
                        result = paired_cluster_bootstrap_ratio(
                            metric_fn, stats_a, stats_b, clusters,
                            n_replicates=int(replicates), seed=int(seed), ci=float(ci),
                            metric_name=metric_key,
                        )
                    else:
                        result = phase0a.paired_cluster_bootstrap(
                            metric_fn, stats_a, stats_b, clusters,
                            n_replicates=int(replicates), seed=int(seed), ci=float(ci),
                            metric_name=metric_key,
                        )
                    rows.append(
                        {
                            "eval_split": name,
                            "K_a": 5,
                            "K_b": int(k_b),
                            "variant": variant,
                            "metric": metric_key,
                            "diff_kind": kind,
                            "n": int(result["n"]),
                            "n_clusters": int(result["n_clusters"]),
                            "resample_unit": result["resample_unit"],
                            "mean_a": float(result["mean_a"]),
                            "mean_b": float(result["mean_b"]),
                            "diff": float(result["diff"]),
                            "ci_low": float(result["ci_low"]),
                            "ci_high": float(result["ci_high"]),
                            "ci_level": float(result["ci_level"]),
                            "n_replicates": int(result["n_replicates"]),
                            "std_diff": float(result["std_diff"]),
                        }
                    )
                    if on_progress is not None:
                        on_progress(
                            len(rows), planned, f"{name} {metric_key}/{variant} K5-vs-K{k_b}"
                        )
    return rows


# ---------------------------------------------------------------------------
# artifact field schemas
# ---------------------------------------------------------------------------
RANKING_FIELDS = [
    "eval_split", "K", "denominator", "n", "n_correct", "top1", "top5",
    "mean_target_rank", "mrr",
]
CALIBRATION_FIELDS = [
    "eval_split", "K", "variant", "denominator", "n", "accuracy", "ece_adaptive",
    "ece_equal_width", "brier_binary", "nll_binary", "conf_acc_gap",
    "multiclass_nll", "multiclass_brier", "mean_entropy",
]
SELECTIVE_FIELDS = [
    "eval_split", "K", "variant", "denominator", "n", "accuracy", "aurc",
    "aurc_oracle", "e_aurc", "auroc_correct", "auprc_correct",
    "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95",
]
RELIABILITY_FIELDS = [
    "eval_split", "K", "variant", "binning", "bin_index", "bin_low", "bin_high",
    "bin_confidence", "bin_accuracy", "bin_count",
]
DIAGNOSTICS_FIELDS = [
    "eval_split", "K", "variant", "n", "mean_entropy", "mean_margin",
] + [f"conf_bin_{index}_count" for index in range(CONFIDENCE_BINS)]
BOOTSTRAP_FIELDS = [
    "eval_split", "K_a", "K_b", "variant", "metric", "diff_kind", "n", "n_clusters",
    "resample_unit", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "ci_level",
    "n_replicates", "std_diff",
]
AGGREGATE_FIELDS = ["eval_split", "K", "variant", "metric", "mean", "std", "n_seeds"]
PAIRED_FIELDS = [
    "eval_split", "K", "metric", "cosine", "b3_mean", "b3_std", "delta_mean",
    "delta_ci_low", "delta_ci_high", "ci_level", "n", "n_clusters", "n_replicates",
    "n_seeds", "status",
]

_AGGREGATE_METRICS: Dict[str, Tuple[List[str], Tuple[str, ...]]] = {
    "ranking": (
        ["top1", "top5", "mean_target_rank", "mrr"],
        ("(invariant)",),
    ),
    "calibration": (
        [
            "accuracy", "ece_adaptive", "ece_equal_width", "brier_binary", "nll_binary",
            "conf_acc_gap", "multiclass_nll", "multiclass_brier", "mean_entropy",
        ],
        B3_VARIANTS,
    ),
    "selective": (
        [
            "accuracy", "aurc", "aurc_oracle", "e_aurc", "auroc_correct", "auprc_correct",
            "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95",
        ],
        B3_VARIANTS,
    ),
}


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read a metric CSV back, coercing numeric cells (used by ``--resume``)."""
    import csv

    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            row: Dict[str, Any] = {}
            for key, value in record.items():
                if value in ("", None):
                    row[key] = None
                    continue
                try:
                    number = float(value)
                except ValueError:
                    row[key] = value
                    continue
                row[key] = int(number) if float(number).is_integer() else number
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# artifact writers (per seed)
# ---------------------------------------------------------------------------
def _write_raw_scores(
    seed_dir: Path,
    bundle: phase0a.ScoreBundle,
    common_rows: Mapping[int, np.ndarray],
) -> List[str]:
    """``seed_{s}/raw_scores/K{K}.npz`` - the frozen raw-logit format.

    Keys: ``scores`` ``[n_K, K]`` float32 (common-cohort row order),
    ``sentence_id`` / ``ref_id`` / ``image_id`` / ``eval_split`` ``[n_K]``,
    ``target_local`` ``[n_K]``.
    """
    dest = Path(seed_dir) / RAW_SCORES_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for k in bundle.ks:
        rows = common_rows[int(k)]
        records = [bundle.records_by_k[int(k)][int(row)] for row in rows.tolist()]
        payload = {
            "scores": bundle.raw_scores[int(k)][rows].astype(np.float32, copy=False),
            "sentence_id": np.asarray([r.sentence_id for r in records], dtype=np.int64),
            "ref_id": np.asarray([r.ref_id for r in records], dtype=np.int64),
            "image_id": np.asarray([r.image_id for r in records], dtype=np.int64),
            "eval_split": np.asarray([r.eval_split for r in records], dtype="<U10"),
            "target_local": bundle.target_local[int(k)][rows].astype(np.int32, copy=False),
        }
        path = dest / f"K{int(k)}.npz"
        np.savez_compressed(str(path), **payload)
        written.append(str(path))
    return written


def _write_per_sentence(
    seed_dir: Path,
    bundle: phase0a.ScoreBundle,
    common_rows: Mapping[int, np.ndarray],
    ks: Sequence[int],
    temps: Mapping[str, Any],
) -> str:
    """``seed_{s}/per_sentence_predictions.npz`` (id columns in the plural spelling;
    the paired reader accepts the cosine audit's singular spellings too)."""
    reference = int(tuple(sorted(ks))[0])
    ref_rows = common_rows[reference]
    ref_records = [bundle.records_by_k[reference][int(row)] for row in ref_rows.tolist()]
    payload: Dict[str, np.ndarray] = {
        "sentence_ids": np.asarray([r.sentence_id for r in ref_records], dtype=np.int64),
        "ref_ids": np.asarray([r.ref_id for r in ref_records], dtype=np.int64),
        "image_ids": np.asarray([r.image_id for r in ref_records], dtype=np.int64),
        "eval_split": np.asarray([r.eval_split for r in ref_records], dtype="<U10"),
    }
    for k in ks:
        rows = common_rows[int(k)]
        raw = bundle.raw_scores[int(k)][rows]
        payload[f"correct_K{int(k)}"] = phase0a._correctness(bundle, int(k), rows)
        payload[f"target_rank_K{int(k)}"] = target_rank(
            raw, bundle.target_local[int(k)][rows]
        ).astype(np.int64)
        payload[f"predicted_index_K{int(k)}"] = np.argmax(raw, axis=1).astype(np.int64)
        for variant in B3_VARIANTS:
            prob = _probabilities(bundle, int(k), rows, variant, temps)
            payload[f"confidence_{variant}_K{int(k)}"] = prob.max(axis=1).astype(np.float32)
    path = Path(seed_dir) / PER_SENTENCE_NAME
    np.savez_compressed(str(path), **payload)
    return str(path)


def _write_seed_tables(
    seed_dir: Path,
    tables: Mapping[str, List[Dict[str, Any]]],
) -> List[str]:
    """Write the five per-seed metric CSVs with their frozen column order."""
    written: List[str] = []
    for name, fields in (
        ("ranking_metrics.csv", RANKING_FIELDS),
        ("calibration_metrics.csv", CALIBRATION_FIELDS),
        ("normalized_selective_metrics.csv", SELECTIVE_FIELDS),
        ("reliability_bins.csv", RELIABILITY_FIELDS),
        ("diagnostics.csv", DIAGNOSTICS_FIELDS),
        ("bootstrap.csv", BOOTSTRAP_FIELDS),
    ):
        key = name.replace(".csv", "")
        rows = tables.get(key, [])
        if name == "bootstrap.csv":
            rows = tables.get("bootstrap", [])
        path = phase0a._write_csv(Path(seed_dir) / name, fields, rows)
        written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# per-seed evaluation
# ---------------------------------------------------------------------------
def _metrics_progress_total(names: Sequence[str], ks: Sequence[int]) -> int:
    return max(1, len(names) * len(tuple(ks)))


def evaluate_seed(
    *,
    seed: int,
    seed_dir: Path,
    corpus: B3Corpus,
    records: Sequence[Any],
    ks: Sequence[int],
    splits: Optional[Sequence[str]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_ci: float,
    device: str,
    t_min: float,
    t_max: float,
    on_scoring: Optional[Callable[[int, int, str], None]] = None,
    on_metrics: Optional[Callable[[int, int, str], None]] = None,
    on_bootstrap: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Score, sanity-check, fit, measure and write every artifact of one seed."""
    def say(message: str) -> None:
        if log is not None:
            log(message)

    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    ks_t = tuple(sorted({int(k) for k in ks}))

    model, train_payload = load_b3_model(seed_dir, device=device)
    say(f"[phase0b] seed {seed}: model rebuilt ({model.num_parameters()} params, device={device})")

    bundle = score_sets_independent(
        corpus, records, model, ks_t, on_progress=on_scoring
    )
    say(f"[phase0b] seed {seed}: scored {bundle.n_rows(ks_t[0])} K={ks_t[0]} rows")

    try:
        checks = phase0a.run_hard_checks(bundle)
    except phase0a.ValidationFailure as exc:
        phase0a._write_json(
            seed_dir / FAILURE_FILENAME,
            {
                "status": "STOP",
                "seed": int(seed),
                "check": exc.check,
                "message": exc.message,
                "violations": exc.violations,
                "created_utc": phase0a._utc_now(),
                "note": (
                    "a hard sanity check of protocol sections 17-19 failed for this B3 "
                    "seed; the evaluation aborted before any result artifact was written"
                ),
            },
        )
        say(f"[phase0b] seed {seed}: HARD CHECK FAILED [{exc.check}] -> {FAILURE_FILENAME}; exit 2")
        raise SystemExit(2) from exc

    legal_ks = tuple(k for k in ks_t if k in LEGAL_CALIBRATION_KS)
    global_fit = fit_corrected_global_temperature(
        bundle, ks=legal_ks, t_min=float(t_min), t_max=float(t_max)
    )
    oracle_fits = {
        int(k): fit_oracle_per_k_temperature(
            bundle, int(k), t_min=float(t_min), t_max=float(t_max)
        )
        for k in ks_t
    }
    temps: Dict[str, Any] = {
        "global": float(global_fit.temperature),
        "oracle": {int(k): float(fit.temperature) for k, fit in oracle_fits.items()},
    }
    say(
        f"[phase0b] seed {seed}: corrected global T* = {temps['global']:.6f} "
        f"(interior={global_fit.interior}); oracle T = "
        + ", ".join(f"K{k}={v:.4f}" for k, v in sorted(temps["oracle"].items()))
    )

    Ks_c, common_rows, common_ids, split_labels, images = _common_view(bundle, ks_t)
    names = _names_of(bundle, splits)

    ranking = ranking_rows(bundle, names, split_labels, common_rows, Ks_c)
    calibration = calibration_rows(bundle, names, split_labels, common_rows, Ks_c, temps)
    selective = selective_rows(bundle, names, split_labels, common_rows, Ks_c, temps)
    if on_metrics is not None:
        total = _metrics_progress_total(names, Ks_c)
        done = 0
        for name in names:
            for k in Ks_c:
                done += 1
                on_metrics(done, total, f"{name} K{k}")
    reliability = reliability_rows(bundle, names, split_labels, common_rows, Ks_c, temps)
    diagnostics = diagnostics_rows(bundle, names, split_labels, common_rows, Ks_c, temps)

    bootstrap = bootstrap_rows(
        bundle, names, split_labels, common_rows, images, Ks_c, temps,
        replicates=int(bootstrap_replicates), seed=int(bootstrap_seed), ci=float(bootstrap_ci),
        on_progress=on_bootstrap,
    )

    _write_raw_scores(seed_dir, bundle, common_rows)
    _write_per_sentence(seed_dir, bundle, common_rows, Ks_c, temps)
    _write_seed_tables(
        seed_dir,
        {
            "ranking_metrics": ranking,
            "calibration_metrics": calibration,
            "normalized_selective_metrics": selective,
            "reliability_bins": reliability,
            "diagnostics": diagnostics,
            "bootstrap": bootstrap,
        },
    )

    pooled = split_labels == _POOLED if False else np.ones(split_labels.shape[0], dtype=bool)
    metadata = {
        "artifact": "phase0b_independent_seed",
        "seed": int(seed),
        "status": "complete",
        "created_utc": phase0a._utc_now(),
        "git_commit": phase0a._git_commit(),
        "device": str(device),
        "ks": [int(k) for k in Ks_c],
        "n_rows_common": {str(int(k)): int(common_rows[int(k)].size) for k in Ks_c},
        "n_rows_common_pooled": int(pooled.sum()),
        "temperature_corrected": float(global_fit.temperature),
        "corrected_fit": global_fit.to_dict(),
        "oracle_fits": {str(int(k)): fit.to_dict() for k, fit in oracle_fits.items()},
        "training": train_payload,
        "hard_checks": checks,
    }
    phase0a._write_json(seed_dir / EVAL_METADATA_NAME, metadata)
    say(f"[phase0b] seed {seed}: {len(bootstrap)} bootstrap comparisons, artifacts written")

    return {
        "seed": int(seed),
        "status": "complete",
        "temperature_corrected": float(global_fit.temperature),
        "corrected_interior": bool(global_fit.interior),
        "corrected_warning": global_fit.warning,
        "oracle_temperatures": {str(int(k)): float(v) for k, v in temps["oracle"].items()},
        "training": train_payload,
        "ranking_metrics": ranking,
        "calibration_metrics": calibration,
        "selective_metrics": selective,
        "bootstrap": bootstrap,
    }


def _read_seed_rows(seed_dir: Path) -> Dict[str, Any]:
    """Rebuild one seed's summary from its CSVs (``--resume`` fast path)."""
    seed_dir = Path(seed_dir)
    metadata_path = seed_dir / EVAL_METADATA_NAME
    if not metadata_path.exists():
        return {"status": "missing"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("status")) != "complete":
        return {"status": "incomplete"}
    return {
        "seed": int(metadata.get("seed", 0)),
        "status": "complete",
        "temperature_corrected": float(metadata.get("temperature_corrected", float("nan"))),
        "corrected_interior": bool(metadata.get("corrected_fit", {}).get("interior", False)),
        "corrected_warning": metadata.get("corrected_fit", {}).get("warning"),
        "oracle_temperatures": {
            str(k): float(v.get("temperature", float("nan")))
            for k, v in (metadata.get("oracle_fits") or {}).items()
        },
        "training": metadata.get("training", {}),
        "ranking_metrics": _read_csv_rows(seed_dir / "ranking_metrics.csv"),
        "calibration_metrics": _read_csv_rows(seed_dir / "calibration_metrics.csv"),
        "selective_metrics": _read_csv_rows(seed_dir / "normalized_selective_metrics.csv"),
        "bootstrap": _read_csv_rows(seed_dir / "bootstrap.csv"),
    }


# ---------------------------------------------------------------------------
# aggregation across seeds
# ---------------------------------------------------------------------------
def aggregate_rows(seed_summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """``aggregate.csv`` rows: mean/std across seeds per split x K x variant x metric.

    Ranking metrics are variant-invariant (labelled ``(invariant)``); calibration
    and selective metrics carry the three B3 variants.
    """
    # bucket by (split, K, variant, metric) -> {seed -> value}. A metric that is
    # reported by more than one table (``accuracy`` lives in both the calibration
    # and the selective table) must still contribute exactly once per seed, so the
    # inner key is the seed id and ``n_seeds`` counts distinct seeds - never rows.
    buckets: Dict[Tuple[str, int, str, str], Dict[int, float]] = {}
    for position, summary in enumerate(seed_summaries):
        if str(summary.get("status")) != "complete":
            continue
        seed_id = int(summary.get("seed", position))
        for table, (metric_names, variants) in _AGGREGATE_METRICS.items():
            for row in summary.get(f"{table}_metrics", []) or []:
                if row.get("eval_split") is None:
                    continue
                split = str(row["eval_split"])
                k = int(row["K"])
                row_variant = row.get("variant")
                for metric in metric_names:
                    if table == "ranking":
                        variant = "(invariant)"
                    else:
                        variant = str(row_variant)
                        if variant not in variants:
                            continue
                    value = row.get(metric)
                    if value in (None, ""):
                        continue
                    buckets.setdefault((split, k, variant, metric), {})[seed_id] = float(value)

    rows: List[Dict[str, Any]] = []
    for (split, k, variant, metric), per_seed in sorted(buckets.items()):
        array = np.asarray([per_seed[s] for s in sorted(per_seed)], dtype=np.float64)
        std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        rows.append(
            {
                "eval_split": split,
                "K": int(k),
                "variant": variant,
                "metric": metric,
                "mean": float(np.mean(array)),
                "std": std,
                "n_seeds": int(array.size),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# paired comparison against the frozen cosine audit
# ---------------------------------------------------------------------------
_PAIRED_METRICS: Tuple[Tuple[str, Callable[[phase0a.SampleStats], float]], ...] = (
    ("accuracy", phase0a.accuracy_metric),
    ("ece_adaptive", phase0a.ece_metric),
    ("e_aurc", _e_aurc_metric),
    ("auroc_correct", _auroc_metric),
    ("rer_at_50", _make_rer_metric(0.50)),
    ("rer_at_80", _make_rer_metric(0.80)),
)

#: Candidate key stems of the cosine "global_T" confidence column.
_COSINE_GLOBAL_STEMS = (
    "confidence_global_T",
    "confidence_global_T_corrected",
    "confidence_globalT",
    "confidence_global_T_phase0a",
)


def _cosine_global_key(store: Any, k: int) -> Optional[str]:
    for stem in _COSINE_GLOBAL_STEMS:
        key = f"{stem}_K{int(k)}"
        if key in store.files:
            return key
    return None


def _pick_store_key(store: Any, candidates: Sequence[str]) -> str:
    """First candidate id column present in an ``np.load`` archive or payload dict.

    Real-world spelling drift: the corrected cosine audit writes singular id
    columns (``sentence_id``) while the B3 per-sentence writer emits plurals
    (``sentence_ids``) - the paired reader must accept both spellings.
    """
    files = getattr(store, "files", None)
    available = set(files) if files is not None else set(store)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(
        f"none of {list(candidates)} found in the store "
        f"(available: {sorted(available)[:12]})"
    )


def paired_vs_cosine_rows(
    out_dir: Path,
    seed_summaries: Sequence[Mapping[str, Any]],
    *,
    cosine_predictions: Optional[Any],
    ks: Sequence[int] = (5, 20, 50),
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 0,
    bootstrap_ci: float = 0.95,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """``paired_vs_cosine.csv``: B3 (global-T corrected) vs the frozen cosine audit.

    For ``K in {5, 20, 50}`` and six headline metrics the B3 mean/std across seeds
    is compared to the cosine ``global_T`` value on the *same* sentences, with an
    image-clustered paired bootstrap CI of ``B3 - cosine`` (rows pooled over seeds,
    cluster = image_id).  Id columns of both stores are accepted in the singular
    (corrected-audit) and plural (B3) spellings.  A missing cosine file yields
    explicit ``status="cosine_missing"`` rows instead of a silent omission.
    """
    out_dir = Path(out_dir)
    ks_t = tuple(int(k) for k in ks)
    complete = [s for s in seed_summaries if str(s.get("status")) == "complete"]

    def _missing(status: str) -> List[Dict[str, Any]]:
        return [
            {
                "eval_split": _POOLED, "K": int(k), "metric": name,
                "cosine": None, "b3_mean": None, "b3_std": None,
                "delta_mean": None, "delta_ci_low": None, "delta_ci_high": None,
                "ci_level": float(bootstrap_ci), "n": None, "n_clusters": None,
                "n_replicates": None, "n_seeds": len(complete), "status": status,
            }
            for k in ks_t
            for name, _ in _PAIRED_METRICS
        ]

    if cosine_predictions is None:
        return _missing("cosine_missing")
    cosine_path = Path(cosine_predictions)
    if not cosine_path.exists():
        if log is not None:
            log(f"[phase0b] cosine predictions {cosine_path} not found -> paired table stubbed")
        return _missing("cosine_missing")
    if not complete:
        return _missing("no_seeds")

    seed_payloads: List[Dict[str, Any]] = []
    for summary in complete:
        seed_dir = out_dir / f"seed_{int(summary['seed'])}"
        path = seed_dir / PER_SENTENCE_NAME
        if not path.exists():
            return _missing("b3_missing")
        with np.load(path, allow_pickle=False) as store:
            payload = {key: np.asarray(store[key]) for key in store.files}
        seed_payloads.append(payload)

    n_planned = max(1, len(ks_t) * len(_PAIRED_METRICS))
    done = 0
    rows: List[Dict[str, Any]] = []
    with np.load(cosine_path, allow_pickle=False) as cos_store:
        cos_ids_raw = np.asarray(
            cos_store[_pick_store_key(cos_store, ("sentence_ids", "sentence_id"))]
        ).reshape(-1)
        cos_order = np.argsort(cos_ids_raw, kind="stable")
        cos_ids = cos_ids_raw[cos_order]
        for k in ks_t:
            key = _cosine_global_key(cos_store, k)
            correct_key = f"correct_K{int(k)}"
            if key is None or correct_key not in cos_store.files:
                if log is not None:
                    log(f"[phase0b] cosine file has no global-T columns for K={k}")
                for name, _ in _PAIRED_METRICS:
                    rows.append(
                        {
                            "eval_split": _POOLED, "K": int(k), "metric": name,
                            "cosine": None, "b3_mean": None, "b3_std": None,
                            "delta_mean": None, "delta_ci_low": None, "delta_ci_high": None,
                            "ci_level": float(bootstrap_ci), "n": None, "n_clusters": None,
                            "n_replicates": None, "n_seeds": len(complete),
                            "status": f"cosine_missing_k{k}",
                        }
                    )
                    done += 1
                    if on_progress is not None:
                        on_progress(done, n_planned, f"K{k} {name}")
                continue
            cos_conf = np.asarray(cos_store[key]).reshape(-1).astype(np.float64)
            cos_correct = np.asarray(cos_store[correct_key]).reshape(-1).astype(np.float64)

            b3_conf_list: List[np.ndarray] = []
            b3_correct_list: List[np.ndarray] = []
            cos_conf_list: List[np.ndarray] = []
            cos_correct_list: List[np.ndarray] = []
            clusters_list: List[np.ndarray] = []
            for payload in seed_payloads:
                b3_key = f"confidence_{B3_VARIANT_GLOBAL_CORRECTED}_K{int(k)}"
                if b3_key not in payload or f"correct_K{int(k)}" not in payload:
                    return _missing(f"b3_missing_k{k}")
                b3_ids_raw = np.asarray(
                    payload[_pick_store_key(payload, ("sentence_ids", "sentence_id"))]
                ).reshape(-1)
                b3_order = np.argsort(b3_ids_raw, kind="stable")
                b3_ids = b3_ids_raw[b3_order]
                common = np.intersect1d(b3_ids, cos_ids, assume_unique=False)
                if common.size == 0:
                    return _missing(f"no_overlap_k{k}")
                # robust to any row order in either file: sort -> search -> map back
                b3_pos = b3_order[np.searchsorted(b3_ids, common)]
                cos_pos = cos_order[np.searchsorted(cos_ids, common)]
                b3_conf_list.append(np.asarray(payload[b3_key]).reshape(-1)[b3_pos].astype(np.float64))
                b3_correct_list.append(
                    np.asarray(payload[f"correct_K{int(k)}"]).reshape(-1)[b3_pos].astype(np.float64)
                )
                cos_conf_list.append(cos_conf[cos_pos])
                cos_correct_list.append(cos_correct[cos_pos])
                clusters_list.append(
                    np.asarray(payload[_pick_store_key(payload, ("image_ids", "image_id"))])[
                        b3_pos
                    ]
                )

            b3_conf_all = np.concatenate(b3_conf_list)
            b3_correct_all = np.concatenate(b3_correct_list)
            cos_conf_all = np.concatenate(cos_conf_list)
            cos_correct_all = np.concatenate(cos_correct_list)
            clusters_all = np.concatenate(clusters_list)
            b3_stats = phase0a.SampleStats.from_conf_correct(b3_conf_all, b3_correct_all)
            cos_stats = phase0a.SampleStats.from_conf_correct(cos_conf_all, cos_correct_all)

            for name, metric_fn in _PAIRED_METRICS:
                per_seed_b3 = [
                    float(metric_fn(phase0a.SampleStats.from_conf_correct(conf, corr)))
                    for conf, corr in zip(b3_conf_list, b3_correct_list)
                ]
                cosine_value = float(metric_fn(cos_stats))
                b3_mean = float(np.mean(per_seed_b3)) if per_seed_b3 else float("nan")
                b3_std = float(np.std(per_seed_b3, ddof=1)) if len(per_seed_b3) > 1 else 0.0
                result = phase0a.paired_cluster_bootstrap(
                    metric_fn, b3_stats, cos_stats, clusters_all,
                    n_replicates=int(bootstrap_replicates), seed=int(bootstrap_seed),
                    ci=float(bootstrap_ci), metric_name=name,
                )
                rows.append(
                    {
                        "eval_split": _POOLED,
                        "K": int(k),
                        "metric": name,
                        "cosine": cosine_value,
                        "b3_mean": b3_mean,
                        "b3_std": b3_std,
                        "delta_mean": float(b3_mean - cosine_value),
                        "delta_ci_low": float(result["ci_low"]),
                        "delta_ci_high": float(result["ci_high"]),
                        "ci_level": float(result["ci_level"]),
                        "n": int(result["n"]),
                        "n_clusters": int(result["n_clusters"]),
                        "n_replicates": int(result["n_replicates"]),
                        "n_seeds": len(per_seed_b3),
                        "status": "ok",
                    }
                )
                done += 1
                if on_progress is not None:
                    on_progress(done, n_planned, f"K{k} {name}")
    return rows


# ---------------------------------------------------------------------------
# the evaluation driver
# ---------------------------------------------------------------------------
def run_b3_eval(
    *,
    features_root: Any,
    manifests_root: Any,
    bank_path: Any,
    refs: Any,
    out_dir: Any = DEFAULT_OUT,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    ks: Sequence[int] = DEFAULT_KS,
    splits: Sequence[str] = PRIMARY_SPLITS,
    regime: str = REGIME_RANDOM,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 0,
    bootstrap_ci: float = 0.95,
    device: str = "cuda",
    image_sizes_path: Any = f"cache/{IMAGE_SIZES_FILENAME}",
    images_root: Any = "data/raw/mscoco",
    image_manifest_csv: Any = "data/full_image_manifest.csv",
    cosine_predictions: Any = COSINE_PREDICTIONS,
    resume: bool = False,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    on_scoring: Optional[Callable[[int, int, str], None]] = None,
    on_metrics: Optional[Callable[[int, int, str], None]] = None,
    on_bootstrap: Optional[Callable[[int, int, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Evaluate every B3 seed on the frozen Phase 0A cohort and write the artifact set.

    A failed hard sanity check of any seed writes ``seed_{s}/VALIDATION_FAILURE.json``
    and raises ``SystemExit(2)`` before that seed's result artifacts exist - a
    broken seed never contributes plottable numbers.
    """
    started = time.perf_counter()

    def say(message: str) -> None:
        if log is not None:
            log(message)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_t = tuple(int(s) for s in seeds)
    ks_t = tuple(sorted({int(k) for k in ks}))
    splits_t = tuple(str(name) for name in splits)

    image_sizes_path = Path(image_sizes_path)
    if not image_sizes_path.exists():
        say(f"[phase0b] building image-size cache at {image_sizes_path}")
        build_image_sizes(images_root, image_manifest_csv, image_sizes_path)

    corpus = B3Corpus(
        features_root,
        manifests_root,
        refs,
        bank_path,
        image_sizes_path=image_sizes_path,
        ks=ks_t,
        regime=str(regime),
    )
    records = corpus.eval_records(splits_t, ks_t)
    say(f"[phase0b] loaded {len(records)} evaluation sentences for splits={list(splits_t)}")

    summaries: List[Dict[str, Any]] = []
    try:
        for seed in seeds_t:
            seed_dir = out_dir / f"seed_{seed}"
            if resume:
                cached = _read_seed_rows(seed_dir)
                if cached.get("status") == "complete":
                    say(f"[phase0b] seed {seed}: resumed from {seed_dir}")
                    summaries.append(cached)
                    continue
            summaries.append(
                evaluate_seed(
                    seed=seed,
                    seed_dir=seed_dir,
                    corpus=corpus,
                    records=records,
                    ks=ks_t,
                    splits=splits_t,
                    bootstrap_replicates=int(bootstrap_replicates),
                    bootstrap_seed=int(bootstrap_seed),
                    bootstrap_ci=float(bootstrap_ci),
                    device=str(device),
                    t_min=float(t_min),
                    t_max=float(t_max),
                    on_scoring=on_scoring,
                    on_metrics=on_metrics,
                    on_bootstrap=on_bootstrap,
                    log=log,
                )
            )
    finally:
        corpus.close()

    aggregate = aggregate_rows(summaries)
    phase0a._write_csv(out_dir / "aggregate.csv", AGGREGATE_FIELDS, aggregate)

    paired = paired_vs_cosine_rows(
        out_dir,
        summaries,
        cosine_predictions=cosine_predictions,
        ks=tuple(k for k in (5, 20, 50) if k in ks_t) or (5,),
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_ci=float(bootstrap_ci),
        log=log,
    )
    phase0a._write_csv(out_dir / "paired_vs_cosine.csv", PAIRED_FIELDS, paired)

    total_seconds = time.perf_counter() - started
    metadata = {
        "artifact": "phase0b_independent",
        "created_utc": phase0a._utc_now(),
        "git_commit": phase0a._git_commit(),
        "python": __import__("sys").version.split()[0],
        "numpy": np.__version__,
        "config": {
            "features_root": str(features_root),
            "manifests_root": str(manifests_root),
            "bank_path": str(bank_path),
            "refs": str(refs) if isinstance(refs, (str, Path)) else "<records>",
            "out_dir": str(out_dir),
            "seeds": [int(s) for s in seeds_t],
            "ks": [int(k) for k in ks_t],
            "splits": [str(name) for name in splits_t],
            "regime": str(regime),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_ci": float(bootstrap_ci),
            "device": str(device),
            "temperature_bounds": [float(t_min), float(t_max)],
            "cosine_predictions": str(cosine_predictions),
            "resume": bool(resume),
        },
        "n_records": int(len(records)),
        "variants": list(B3_VARIANTS),
        "per_seed": {
            str(int(summary["seed"])): {
                "status": summary.get("status"),
                "temperature_corrected": summary.get("temperature_corrected"),
                "corrected_interior": summary.get("corrected_interior"),
                "corrected_warning": summary.get("corrected_warning"),
                "oracle_temperatures": summary.get("oracle_temperatures"),
                "training": summary.get("training"),
            }
            for summary in summaries
        },
        "paired_vs_cosine": {
            "n_rows": int(len(paired)),
            "status": sorted({str(row["status"]) for row in paired}),
        },
        "total_seconds": float(total_seconds),
        "artifacts": [
            "metadata.json",
            "aggregate.csv",
            "paired_vs_cosine.csv",
            "seed_{s}/" + RAW_SCORES_DIRNAME + "/K{K}.npz",
            "seed_{s}/ranking_metrics.csv",
            "seed_{s}/calibration_metrics.csv",
            "seed_{s}/normalized_selective_metrics.csv",
            "seed_{s}/reliability_bins.csv",
            "seed_{s}/diagnostics.csv",
            "seed_{s}/bootstrap.csv",
            "seed_{s}/" + PER_SENTENCE_NAME,
            "seed_{s}/" + EVAL_METADATA_NAME,
            "seed_{s}/" + FAILURE_FILENAME + " (written only on a failed hard check)",
        ],
    }
    phase0a._write_json(out_dir / "metadata.json", metadata)
    say(f"[phase0b] total: {total_seconds:.2f}s -> {out_dir}")

    return {
        "out_dir": str(out_dir),
        "seeds": [int(s) for s in seeds_t],
        "n_records": int(len(records)),
        "per_seed": metadata["per_seed"],
        "n_aggregate_rows": int(len(aggregate)),
        "n_paired_rows": int(len(paired)),
        "paired_status": metadata["paired_vs_cosine"]["status"],
        "total_seconds": float(total_seconds),
    }
