"""Phase 1F memory model recovery (protocol Amendment A8.4).

Phase 1F is a *confirmatory* stage: it may only re-use the two frozen
reliability models that were trained on the random regime
(``reliability_train`` x ``K in {5, 10}``):

* **Stats Logistic** -- L2 logistic regression on the 17-d score statistics
  (``stat_logK`` variant); and
* **E1b** -- L2 logistic regression on the 17-d score statistics concatenated
  with the frozen 16-d semantic statistics.

The frozen artifacts only store the coefficient tables and the hyper-parameters
(``C``), never the fitted estimator objects.  This module therefore **re-fits
the two logistic models deterministically on the frozen training rows** (same
hyper-parameters, same standardisation, no RNG) to recover them, and then proves
the recovery is bit-comparable to the frozen artifacts through four checks
(the A8.4 STOP condition):

1. :func:`recover_frozen_models` stats-logistic predictions vs the Phase 0.5
   ``predictions/{scorer}/stats_logistic.csv.gz`` files (multiset equality per
   ``(K, split)``);
2. E1b coefficients vs Phase 1 ``e1_logistic/coefficients.csv``;
3. Stats / E1b ``tune_mean_auroc`` vs the frozen selection tables; and
4. the stats-17 train-only normalisation vs ``features/{scorer}_normalisation.json``.

Any deviation above ``1e-9`` raises :class:`AssertionError`.  Nothing here
trains any *new* model, draws any random number or mutates the frozen inputs.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..metrics.discrimination import auroc_correct
from ..reliability import data as rdata
from ..reliability import features as rfeat
from ..reliability import models as rmodels
from ..semantic import data as sdata
from ..semantic import features as sfeat

__all__ = [
    "DEFAULT_B3_ROOT",
    "DEFAULT_EMB_ROOT",
    "DEFAULT_PHASE1",
    "DEFAULT_PHASE05",
    "TRAIN_KS",
    "FrozenReliability",
    "FrozenSeedModels",
    "apply_frozen",
    "recover_frozen_models",
]

#: Phase 0.5 (score sufficiency) frozen artifact root.
DEFAULT_PHASE05 = Path("results/phase05_score_sufficiency")
#: Phase 1 (semantic sufficiency) frozen artifact root.
DEFAULT_PHASE1 = Path("results/phase1_semantic_sufficiency")
#: Frozen-cohort embedding archive root (``cache/semantic_phase1``).
DEFAULT_EMB_ROOT = Path("cache/semantic_phase1")
#: Phase 0B independent-scorer artifact root (loads the corrected temperatures).
DEFAULT_B3_ROOT = rdata.B3_ROOT

#: The only candidate-set sizes the frozen models were trained / tuned on (A8.4).
TRAIN_KS: Tuple[int, ...] = (5, 10)
#: The splits compared against the frozen Phase 0.5 prediction files.
_COMPARE_SPLITS: Tuple[str, ...] = ("val_select", "testA", "testB")
#: Frozen model identifiers as stored in the selection / coefficient tables.
_STATS_MODEL = "stats_logistic"
_E1B_NAME = "e1b_stats_semantic"
#: Normalisation variant key of the 17-d score statistics.
_STATS_VARIANT = "stats_logK"
#: Hard consistency tolerance of every A8.4 check.
_TOL = 1e-9
#: Ordered verification-check names (per-seed detail keys + ``all`` summary).
_CHECK_KEYS: Tuple[str, ...] = (
    "stats_logistic_pred_max_abs",
    "e1b_coefficients_max_abs",
    "stats_tune_auroc_max_abs",
    "e1b_tune_auroc_max_abs",
    "stats17_normalisation_max_abs",
)


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrozenSeedModels:
    """One B3 seed's recovered frozen Stats-Logistic / E1b model pair.

    Attributes
    ----------
    scorer:
        The B3 scorer id (``"b3_seed{i}"``).
    temperature:
        The frozen corrected global temperature read from Phase 0B metadata.
    stats_fit / sem_fit:
        Train-only cross-row normalisations of the 17-d score statistics and the
        16-d semantic statistics (protocol A6.3 / A7.3).
    stats_clf / e1b_clf:
        The deterministically re-fitted 17-d and 33-d L2 logistic models.
    verification:
        Per-check ``max|delta|`` values (keys :data:`_CHECK_KEYS`).
    """

    scorer: str
    temperature: float
    stats_fit: rfeat.NormalizationFit
    sem_fit: rfeat.NormalizationFit
    stats_clf: rmodels.LogisticModel
    e1b_clf: rmodels.LogisticModel
    verification: Dict[str, float]


@dataclass(frozen=True)
class FrozenReliability:
    """All recovered seeds plus the per-seed and ``all`` verification tables.

    ``verification[scorer]`` holds the per-seed ``max|delta|`` detail;
    ``verification["all"]`` holds the element-wise maximum over the seeds.
    """

    seeds: Dict[str, FrozenSeedModels]
    verification: Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# small loaders
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Dict:
    """Load a UTF-8 JSON document into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_temperature(b3_root: Path, scorer: str) -> float:
    """Frozen per-seed corrected global temperature from Phase 0B metadata."""
    seed_tag = scorer.split("b3_seed")[1]
    path = Path(b3_root) / f"seed_{seed_tag}" / "eval_metadata.json"
    payload = _read_json(path)
    return float(payload["temperature_corrected"])


def _load_stats_selection(phase05_root: Path) -> Dict[str, Dict[str, float]]:
    """Per-scorer ``chosen_hp`` / ``tune_mean_auroc`` of the Phase 0.5 Stats Logistic."""
    payload = _read_json(Path(phase05_root) / _STATS_MODEL / "selection.json")
    out: Dict[str, Dict[str, float]] = {}
    for scorer, entry in payload["per_scorer"].items():
        out[str(scorer)] = {
            "chosen_hp": float(entry["chosen_hp"]),
            "tune_mean_auroc": float(entry["tune_mean_auroc"]),
        }
    return out


def _load_e1b_selection(phase1_root: Path) -> Dict[str, Dict[str, float]]:
    """Per-scorer ``selected_hp`` / ``tune_mean_auroc`` of the Phase 1 E1b model."""
    payload = _read_json(Path(phase1_root) / "e1_logistic" / "selection.json")
    out: Dict[str, Dict[str, float]] = {}
    for scorer, entry in payload["per_scorer"].items():
        models = entry.get("models", {})
        if _E1B_NAME not in models:
            continue
        model = models[_E1B_NAME]
        out[str(scorer)] = {
            "selected_hp": float(model["selected_hp"]),
            "tune_mean_auroc": float(model["tune_mean_auroc"]),
        }
    return out


def _load_e1b_coefficients(
    phase1_root: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[float]]]:
    """E1b coefficient vectors keyed by scorer, aligned to the frozen feature order.

    The CSV stores one row per ``(scorer, feature)`` with a ``block`` column; the
    returned vectors follow ``stat_feature_names() + SEMANTIC_STAT_NAMES`` exactly.
    An optional intercept row (``feature`` in ``{"intercept", "bias", "const"}``)
    is returned separately per scorer, or ``None`` when the table has none.
    """
    path = Path(phase1_root) / "e1_logistic" / "coefficients.csv"
    expected = list(rfeat.stat_feature_names()) + list(sfeat.SEMANTIC_STAT_NAMES)
    blocks: Dict[str, Dict[str, float]] = {}
    intercepts: Dict[str, Optional[float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scorer = str(row["scorer"])
            feature = str(row["feature"])
            if feature.lower() in ("intercept", "bias", "const"):
                intercepts[scorer] = float(row["coefficient"])
                continue
            blocks.setdefault(scorer, {})[feature] = float(row["coefficient"])
    coefs: Dict[str, np.ndarray] = {}
    for scorer, mapping in blocks.items():
        missing = [name for name in expected if name not in mapping]
        if missing:
            raise ValueError(f"{path}: scorer {scorer!r} is missing coefficients for {missing}")
        coefs[scorer] = np.asarray([mapping[name] for name in expected], dtype=np.float64)
    return coefs, intercepts


def _load_phase05_predictions(phase05_root: Path, scorer: str) -> Dict[Tuple[int, str], np.ndarray]:
    """Frozen Phase 0.5 Stats-Logistic predictions grouped by ``(K, eval_split)``."""
    path = Path(phase05_root) / "predictions" / scorer / f"{_STATS_MODEL}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"phase 0.5 predictions not found: {path}")
    grouped: Dict[Tuple[int, str], List[float]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["K"]), str(row["eval_split"]))
            grouped.setdefault(key, []).append(float(row["reliability_score"]))
    return {key: np.asarray(values, dtype=np.float64) for key, values in grouped.items()}


def _load_stats17_normalisation(features_dir: Path, scorer: str) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen ``stats_logK`` mean / std (17-d) from ``{scorer}_normalisation.json``."""
    payload = _read_json(Path(features_dir) / f"{scorer}_normalisation.json")
    variant = payload["variants"][_STATS_VARIANT]
    return (
        np.asarray(variant["mean"], dtype=np.float64),
        np.asarray(variant["std"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# feature blocks / normalisation
# ---------------------------------------------------------------------------
def _chunked_semantic_stats(store: sdata.EmbeddingStore, scores: np.ndarray, *, chunk: int = 2048) -> np.ndarray:
    """Chunked :func:`ccg.semantic.features.semantic_stats` over one ``K`` store.

    Mirrors ``scripts/run_phase1.py::_e1_stats`` (float32 embedding views,
    float64 scores) so the 16-d block is bit-identical to the Phase 1 pipeline.
    """
    n = len(store)
    out = np.empty((n, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        z_q = np.asarray(store.z_q[start:stop], dtype=np.float32)
        z_i = np.asarray(store.z_i[start:stop], dtype=np.float32)
        out[start:stop] = sfeat.semantic_stats(z_q, z_i, np.asarray(scores[start:stop], dtype=np.float64))
    return out


def _fit_standardise(
    raw_by_k: Dict[int, np.ndarray],
    train_flags: Dict[int, np.ndarray],
    keys: Sequence[str],
) -> Tuple[rfeat.NormalizationFit, Dict[int, np.ndarray]]:
    """Train-only cross-row standardisation of one K-block family (A6.3 / A7.3).

    The ``mean``/``std`` are fitted on ``vstack([train rows of K=5, train rows of
    K=10])`` and then applied unchanged to every row of every ``K``.
    """
    combined = np.vstack([np.asarray(raw_by_k[k])[train_flags[k]] for k in TRAIN_KS])
    fit = rfeat.normalize_fit(combined, fit_rows=np.arange(combined.shape[0]), keys=tuple(keys))
    standardised = {
        k: np.asarray(rfeat.normalize_apply(np.asarray(raw_by_k[k]), fit), dtype=np.float64) for k in TRAIN_KS
    }
    return fit, standardised


def _tune_mean_auroc(
    conf_by_k: Dict[int, np.ndarray],
    correct_by_k: Dict[int, np.ndarray],
    tune_flags: Dict[int, np.ndarray],
) -> float:
    """Primary selection metric: mean ``AUROC_correct`` over the tune K5/K10 rows."""
    values = [
        float(auroc_correct(np.asarray(conf_by_k[k])[tune_flags[k]], np.asarray(correct_by_k[k])[tune_flags[k]]))
        for k in TRAIN_KS
    ]
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------
def recover_frozen_models(
    *,
    scorers: Sequence[str] = ("b3_seed1", "b3_seed2", "b3_seed3"),
    phase05_root: Path = DEFAULT_PHASE05,
    phase1_root: Path = DEFAULT_PHASE1,
    emb_root: Path = DEFAULT_EMB_ROOT,
    features_dir: Path = DEFAULT_PHASE05 / "features",
    split_manifest: Path = DEFAULT_PHASE05 / "split_manifest.json",
    b3_root: Path = DEFAULT_B3_ROOT,
    log: Optional[Callable[[str], None]] = None,
) -> FrozenReliability:
    """Re-fit and verify the frozen Stats-Logistic / E1b models (A8.4).

    Loads the frozen cohort embeddings (K5 / K10), the frozen A6.2 split and the
    frozen hyper-parameter tables, then for every ``scorer`` re-fits the two
    logistic models on the frozen ``reliability_train`` rows and runs the four
    consistency checks.  Raises :class:`AssertionError` (with the offending
    scorer / check / value) as soon as any ``max|delta|`` exceeds ``1e-9``.
    """
    emit = log if log is not None else (lambda _message: None)
    phase05_root = Path(phase05_root)
    phase1_root = Path(phase1_root)
    features_dir = Path(features_dir)
    split_manifest = Path(split_manifest)
    b3_root = Path(b3_root)

    split = sdata.load_split_from_manifest(split_manifest)
    emit(f"[frozen] A6.2 split: {int(split.train_images.size)} train / {int(split.tune_images.size)} tune images")

    stores = {k: sdata.load_embedding_store(k, out_root=Path(emb_root)) for k in TRAIN_KS}
    emit(f"[frozen] embedding stores loaded for K={list(TRAIN_KS)}")

    stats_selection = _load_stats_selection(phase05_root)
    e1b_selection = _load_e1b_selection(phase1_root)
    e1b_coefs, e1b_intercepts = _load_e1b_coefficients(phase1_root)
    if not any(value is not None for value in e1b_intercepts.values()):
        emit(
            "[frozen] e1_logistic/coefficients.csv has no intercept row "
            "(columns: scorer,feature,block,coefficient,coefficient_std); "
            "only the 33-d coefficient vector is compared"
        )

    seeds: Dict[str, FrozenSeedModels] = {}
    for scorer in scorers:
        scorer = str(scorer)
        temperature = _load_temperature(b3_root, scorer)
        cores = {k: sdata.load_scorer_canonical(scorer, k, b3_root=b3_root) for k in TRAIN_KS}
        anchor = cores[TRAIN_KS[0]]
        eval_split = np.asarray(anchor.eval_split)
        image_id = np.asarray(anchor.image_id, dtype=np.int64)
        train_flags = {k: split.row_mask(eval_split, image_id, kind="reliability_train") for k in TRAIN_KS}
        tune_flags = {k: split.row_mask(eval_split, image_id, kind="reliability_tune") for k in TRAIN_KS}
        correct = {k: np.asarray(cores[k].correct, dtype=bool) for k in TRAIN_KS}
        emit(f"[frozen] {scorer}: T={temperature:.6f}, train={int(train_flags[TRAIN_KS[0]].sum())} "
             f"(K5) / {int(train_flags[TRAIN_KS[1]].sum())} (K10)")

        # -- feature blocks (score stats + semantic stats) ------------------
        stats17_raw = {
            k: rfeat.stat_features(np.asarray(cores[k].scores, dtype=np.float64), temperature=temperature)
            for k in TRAIN_KS
        }
        sem16_raw = {k: _chunked_semantic_stats(stores[k], np.asarray(cores[k].scores)) for k in TRAIN_KS}

        stats_fit, stats17_std = _fit_standardise(stats17_raw, train_flags, rfeat.stat_feature_names())
        sem_fit, sem16_std = _fit_standardise(sem16_raw, train_flags, sfeat.SEMANTIC_STAT_NAMES)
        e1b_std = {k: np.hstack([stats17_std[k], sem16_std[k]]) for k in TRAIN_KS}

        y_train = np.concatenate([correct[k][train_flags[k]].astype(np.float64) for k in TRAIN_KS])

        # -- re-fit the two frozen logistic models --------------------------
        stats_clf = rmodels.LogisticModel(C=float(stats_selection[scorer]["chosen_hp"]))
        stats_clf.fit(np.vstack([stats17_std[k][train_flags[k]] for k in TRAIN_KS]), y_train)
        e1b_clf = rmodels.LogisticModel(C=float(e1b_selection[scorer]["selected_hp"]))
        e1b_clf.fit(np.vstack([e1b_std[k][train_flags[k]] for k in TRAIN_KS]), y_train)

        stats_conf = {k: stats_clf.predict_proba(stats17_std[k]) for k in TRAIN_KS}
        e1b_conf = {k: e1b_clf.predict_proba(e1b_std[k]) for k in TRAIN_KS}

        # -- check 1: stats-logistic predictions vs Phase 0.5 files ---------
        frozen_pred = _load_phase05_predictions(phase05_root, scorer)
        pred_delta = 0.0
        for k in TRAIN_KS:
            for split_name in _COMPARE_SPLITS:
                ours = np.sort(np.asarray(stats_conf[k][eval_split == split_name], dtype=np.float64))
                key = (k, split_name)
                ref = frozen_pred.get(key)
                ref_size = 0 if ref is None else int(ref.size)
                if ours.size != ref_size:
                    raise AssertionError(
                        f"{scorer}: stats_logistic K={k} split={split_name}: row count {ours.size} "
                        f"!= frozen {ref_size}"
                    )
                if ours.size:
                    pred_delta = max(pred_delta, float(np.abs(ours - np.sort(ref)).max()))

        # -- check 2: E1b coefficient vector vs Phase 1 coefficients.csv ----
        coef, intercept = e1b_clf.coefficients()
        ref_coef = e1b_coefs.get(scorer)
        if ref_coef is None:
            raise AssertionError(f"{scorer}: no E1b coefficients found in e1_logistic/coefficients.csv")
        coef_delta = float(np.abs(np.asarray(coef, dtype=np.float64) - ref_coef).max())
        ref_intercept = e1b_intercepts.get(scorer)
        if ref_intercept is not None:
            emit(f"[frozen] {scorer}: intercept present in CSV -> max|delta| = {abs(intercept - ref_intercept):.3e}")
            coef_delta = max(coef_delta, float(abs(intercept - ref_intercept)))

        # -- check 3: tune mean AUROC (stats + e1b) vs frozen selections ----
        stats_tune_delta = abs(_tune_mean_auroc(stats_conf, correct, tune_flags)
                               - float(stats_selection[scorer]["tune_mean_auroc"]))
        e1b_tune_delta = abs(_tune_mean_auroc(e1b_conf, correct, tune_flags)
                             - float(e1b_selection[scorer]["tune_mean_auroc"]))

        # -- check 4: stats-17 train-only normalisation vs frozen features --
        ref_mean, ref_std = _load_stats17_normalisation(features_dir, scorer)
        norm_delta = max(
            float(np.abs(np.asarray(stats_fit.mean) - ref_mean).max()),
            float(np.abs(np.asarray(stats_fit.std) - ref_std).max()),
        )

        checks = {
            "stats_logistic_pred_max_abs": pred_delta,
            "e1b_coefficients_max_abs": coef_delta,
            "stats_tune_auroc_max_abs": float(stats_tune_delta),
            "e1b_tune_auroc_max_abs": float(e1b_tune_delta),
            "stats17_normalisation_max_abs": norm_delta,
        }
        for name, value in checks.items():
            if not (value <= _TOL):
                raise AssertionError(
                    f"{scorer}: A8.4 check {name} = {value:.3e} exceeds tolerance {_TOL:.0e}"
                )
        emit(
            f"[frozen] {scorer}: recovered OK "
            + ", ".join(f"{name}={checks[name]:.2e}" for name in _CHECK_KEYS)
        )

        seeds[scorer] = FrozenSeedModels(
            scorer=scorer,
            temperature=temperature,
            stats_fit=stats_fit,
            sem_fit=sem_fit,
            stats_clf=stats_clf,
            e1b_clf=e1b_clf,
            verification=checks,
        )

    verification: Dict[str, Dict[str, float]] = {scorer: dict(seeds[scorer].verification) for scorer in seeds}
    verification["all"] = {
        name: float(max(seeds[scorer].verification[name] for scorer in seeds)) for name in _CHECK_KEYS
    }
    emit("[frozen] all-seed max: " + ", ".join(f"{name}={verification['all'][name]:.2e}" for name in _CHECK_KEYS))
    return FrozenReliability(seeds=seeds, verification=verification)


# ---------------------------------------------------------------------------
# frozen inference
# ---------------------------------------------------------------------------
def apply_frozen(
    seed_models: FrozenSeedModels,
    stats17_raw: np.ndarray,
    sem16_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the recovered frozen models without ever re-fitting.

    ``stats17_raw`` / ``sem16_raw`` are the *un-standardised* ``[n, 17]`` and
    ``[n, 16]`` feature blocks; the stored train-only normalisations and the two
    frozen logistic models are applied as-is.  Returns ``(stats_conf, e1b_conf)``,
    both ``[n]`` ``float64`` (``P(correct)``).
    """
    stats_std = rfeat.normalize_apply(np.asarray(stats17_raw, dtype=np.float64), seed_models.stats_fit)
    sem_std = rfeat.normalize_apply(np.asarray(sem16_raw, dtype=np.float64), seed_models.sem_fit)
    e1b_std = np.hstack([stats_std, sem_std])
    stats_conf = seed_models.stats_clf.predict_proba(stats_std)
    e1b_conf = seed_models.e1b_clf.predict_proba(e1b_std)
    return np.asarray(stats_conf, dtype=np.float64), np.asarray(e1b_conf, dtype=np.float64)
