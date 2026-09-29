"""Phase 1F driver - Hard-Competition Semantic Confirmation (protocol Amendment A8).

Frozen-model confirmatory stress test: the B3 checkpoints, the Stats-Logistic /
E1b coefficients and the train-only normalisations are inherited verbatim from
Phase 0.5 / Phase 1 (``K in {5, 10}``, random regime); the only thing that
changes is the *candidate composition* (matched random -> GT same-category).

    conda activate deepminer
    python scripts/run_phase1f.py --out results/phase1f_hard_semantic \\
        --bootstrap-replicates 5000 --log-file logs_phase1f_driver.txt
    python scripts/run_phase1f.py --smoke       # fast end-to-end check (1 seed, 100 reps)

Stages: cohort -> protocol.json -> raw-score STOP check -> frozen scoring of the
construction cells -> features + frozen models -> point metrics -> paired
bootstraps (pair / ratio / diff-of-diffs) -> manipulation check / grounding
difficulty / subgroups -> A8.6 gate -> figures -> artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.experiment.phase0a import SampleStats  # noqa: E402
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import evaluate as seval  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import frozen as sfrozen  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402
from ccg.semantic import hard_eval as heval  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402

__all__ = ["build_parser", "main"]

# ---------------------------------------------------------------------------
# frozen configuration (A8)
# ---------------------------------------------------------------------------
B3_SEEDS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
POOLED = shard.POOLED
E1B_NAME = "e1b_stats_semantic"
STATS_NAME = "stats_logistic"
MSP_NAME = "msp"
CELL_MODELS: Tuple[str, ...] = (MSP_NAME, STATS_NAME, E1B_NAME)
PAIR_METRICS: Tuple[str, ...] = ("auroc_correct", "e_aurc", "rer_at_50", "rer_at_80")
EXPB_METRICS: Tuple[str, ...] = ("auroc_correct", "rer_at_50")
SEM_CHUNK = 2048
DEFAULT_OUT = Path("results/phase1f_hard_semantic")
MANIFEST_PROVENANCE = "cache/manifests (manifests-v1, seed 20260927; frozen)"

#: metric columns averaged into the ``b3_mean`` rows of reliability_metrics.csv
AGG_METRIC_FIELDS: Tuple[str, ...] = (
    "b3_accuracy", "auroc_correct", "aurc", "aurc_oracle", "e_aurc",
    "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95",
    "ece_adaptive", "brier_binary", "nll_binary",
)
#: numeric columns averaged into the ``b3_mean`` rows of semantic_increment.csv
INCREMENT_NUMERIC_FIELDS: Tuple[str, ...] = (
    "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high",
    "e_aurc_reduction", "e_aurc_reduction_ci_low", "e_aurc_reduction_ci_high",
    "delta_e_aurc", "delta_e_aurc_ci_low", "delta_e_aurc_ci_high",
    "rer50_gain_pp", "rer50_gain_pp_ci_low", "rer50_gain_pp_ci_high",
    "rer80_gain_pp", "rer80_gain_pp_ci_low", "rer80_gain_pp_ci_high",
)


@dataclass(frozen=True)
class CellSpec:
    """One evaluated (construction, cohort) cell of the A8 layout."""

    cell: str
    regime: str            # "random" | "same_category" | "level"
    k: int
    cohort: str            # "same4" | "same8" | "same9"
    source: str            # variant supplying the rows
    rows: str              # "all" (variant rows == cell rows) or a cohort mask
    level: Optional[int] = None
    hard_fraction: float = 0.0


@dataclass(frozen=True)
class VariantSpec:
    """One scored candidate construction (before cell slicing)."""

    variant: str
    regime: str            # "random" | "same_category" | "level"
    k: int
    cohort: str            # row mask of the variant ("base" = full pooled cohort)
    level: Optional[int] = None
    stop_k: Optional[int] = None   # STOP-verify against frozen raw_scores


CELL_SPECS: Tuple[CellSpec, ...] = (
    CellSpec("rand5", "random", 5, "same4", "base_r5", "same4"),
    CellSpec("hard5", "same_category", 5, "same4", "hard5", "all", hard_fraction=1.0),
    CellSpec("rand10", "random", 10, "same9", "base_r10", "same9"),
    CellSpec("hard10", "same_category", 10, "same9", "hard10", "all", hard_fraction=1.0),
    CellSpec("expb_m0", "level", 10, "same8", "expb_m0", "all", level=0, hard_fraction=0.0),
    CellSpec("expb_m2", "level", 10, "same8", "expb_m2", "all", level=2, hard_fraction=2.0 / 9.0),
    CellSpec("expb_m4", "level", 10, "same8", "expb_m4", "all", level=4, hard_fraction=4.0 / 9.0),
    CellSpec("expb_m8", "level", 10, "same8", "expb_m8", "all", level=8, hard_fraction=8.0 / 9.0),
)
CELL_BY_NAME: Dict[str, CellSpec] = {spec.cell: spec for spec in CELL_SPECS}
PRIMARY_CELLS: Tuple[str, ...] = ("rand5", "hard5")
SECONDARY_CELLS: Tuple[str, ...] = ("rand10", "hard10")
EXPB_CELLS: Tuple[str, ...] = ("expb_m0", "expb_m2", "expb_m4", "expb_m8")
PAIRS: Tuple[Tuple[str, str], ...] = (("rand5", "hard5"), ("rand10", "hard10"))

VARIANTS: Tuple[VariantSpec, ...] = (
    VariantSpec("base_r5", "random", 5, "base", stop_k=5),
    VariantSpec("base_r10", "random", 10, "base", stop_k=10),
    VariantSpec("hard5", "same_category", 5, "same4"),
    VariantSpec("hard10", "same_category", 10, "same9"),
    VariantSpec("expb_m0", "level", 10, "same8", level=0),
    VariantSpec("expb_m2", "level", 10, "same8", level=2),
    VariantSpec("expb_m4", "level", 10, "same8", level=4),
    VariantSpec("expb_m8", "level", 10, "same8", level=8),
)
CELLS_BY_SOURCE: Dict[str, List[CellSpec]] = {}
for _spec in CELL_SPECS:
    CELLS_BY_SOURCE.setdefault(_spec.source, []).append(_spec)
CONSTRUCTION_NOTES: Mapping[str, str] = {
    "rand5": "random regime C_5 = [target] + order[:4] (frozen random manifest)",
    "hard5": "same_category regime C_5 = [target] + same-cat prefix (frozen same_category manifest)",
    "rand10": "random regime C_10 = [target] + order[:9] (frozen random manifest)",
    "hard10": "same_category regime C_10 = [target] + same-cat prefix (frozen same_category manifest)",
    "expb_m0": "level m=0: [target] + 0 same-cat + first 9 of the manifest rest shuffle",
    "expb_m2": "level m=2: [target] + first 2 same-cat + first 7 of the manifest rest shuffle",
    "expb_m4": "level m=4: [target] + first 4 same-cat + first 5 of the manifest rest shuffle",
    "expb_m8": "level m=8: [target] + first 8 same-cat + first 1 of the manifest rest shuffle",
}


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass
class CellData:
    """Per-(cell, seed) arrays after frozen scoring + frozen feature application."""

    sentence_id: np.ndarray
    ref_id: np.ndarray
    image_id: np.ndarray
    eval_split: np.ndarray
    scores: np.ndarray            # float32 [n, k]
    correct: np.ndarray           # bool [n]
    conf: Dict[str, np.ndarray]   # model -> float64 [n]
    sem16: np.ndarray             # float64 [n, 16] raw (un-standardised)
    stats17: np.ndarray           # float64 [n, 17] raw
    margin: np.ndarray            # top1 - top2 raw score
    msp_score: np.ndarray         # scalar MSP confidence (R0)
    entropy: np.ndarray           # H(softmax(scores / T))


@dataclass
class PairBundle:
    """Bootstrap products of one pair cell (per seed)."""

    abs_rows: Dict[str, Dict[str, Any]]
    ratio: Dict[str, Any]
    dod: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# small helpers (mirroring the phase-1 driver conventions)
# ---------------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames) if fieldnames is not None else _union_fields(rows)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in names})
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_logger(log_file: Optional[Path]) -> Callable[[str], None]:
    handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_file, "a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return log


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strip_arrays(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not isinstance(value, np.ndarray)}


# ---------------------------------------------------------------------------
# stage 1: load (cohort / frozen models / corpus)
# ---------------------------------------------------------------------------
def _load_stage(args: argparse.Namespace, log: Callable[[str], None]) -> Dict[str, Any]:
    cohort = shard.load_hard_cohort(
        features_dir=Path(args.phase05) / "features",
        manifests_root=Path(args.manifests),
        log=log,
    )
    log(f"[phase1f] cohort: {len(cohort)} pooled rows / "
        f"{int(np.unique(cohort.image_id).size)} images / {int(np.unique(cohort.ref_id).size)} refs")

    frozen_rel = sfrozen.recover_frozen_models(
        scorers=tuple(args.seeds),
        phase05_root=Path(args.phase05),
        phase1_root=Path(args.phase1),
        emb_root=Path(args.embeddings),
        features_dir=Path(args.phase05) / "features",
        split_manifest=Path(args.phase05) / "split_manifest.json",
        b3_root=Path(args.b3_root),
        log=log,
    )

    corpus = B3Corpus(
        Path(args.features),
        Path(args.manifests),
        args.refs,
        Path(args.bank),
        image_sizes_path=Path(args.image_sizes),
        ks=(5, 10),
        regime="random",
        preload=True,
    )
    log("[phase1f] B3Corpus preloaded (region/text/bank)")

    device = str(args.device)
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = hscores.load_frozen_scorers(
        b3_root=Path(args.b3_root),
        seeds=tuple(int(name.split("b3_seed")[1]) for name in args.seeds),
        device=device,
    )
    log(f"[phase1f] frozen B3 scorers loaded on {device}: {sorted(models)}")
    return {"cohort": cohort, "frozen": frozen_rel, "corpus": corpus, "models": models, "device": device}


# ---------------------------------------------------------------------------
# stage 2: STOP check (A8.4) - random C_K must reproduce the frozen raw_scores
# ---------------------------------------------------------------------------
def _stop_check(
    cohort: shard.HardCohort,
    corpus: B3Corpus,
    models: Mapping[str, Any],
    seeds: Sequence[str],
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    all_rows = np.arange(len(cohort))
    for spec in (VARIANTS[0], VARIANTS[1]):  # base_r5, base_r10
        samples = [cohort.sample(int(row), "random") for row in all_rows]
        batch = hscores.materialise_examples(corpus, samples, spec.k, text_cache={})
        for scorer in seeds:
            scores = hscores.score_examples(models[scorer], batch, batch_size=args.batch_size)
            reference = sdata.load_scorer_canonical(scorer, spec.k, b3_root=Path(args.b3_root))
            positions = np.searchsorted(reference.sentence_id, cohort.sentence_id)
            if not np.array_equal(reference.sentence_id[positions], cohort.sentence_id):
                raise AssertionError(
                    f"STOP: {scorer} K={spec.k}: cohort sentence ids are not a subset of the "
                    "frozen canonical rows"
                )
            frozen_scores = np.asarray(reference.scores[positions], dtype=np.float32)
            per_scorer = hscores.verify_against_raw_scores(
                {scorer: scores}, {scorer: frozen_scores}, k=spec.k, atol=float(args.stop_atol)
            )
            key = f"{spec.variant}/{scorer}"
            deltas[key] = float(per_scorer[scorer])
            log(f"[phase1f] STOP check {key}: max|delta|={deltas[key]:.3e} (K={spec.k})")
        del batch
    return deltas


# ---------------------------------------------------------------------------
# stage 3: frozen scoring of every variant + cell feature blocks
# ---------------------------------------------------------------------------
def _variant_rows(cohort: shard.HardCohort, spec: VariantSpec) -> np.ndarray:
    if spec.cohort == "base":
        return np.arange(len(cohort))
    return np.flatnonzero(cohort.masks[spec.cohort])


def _variant_samples(cohort: shard.HardCohort, spec: VariantSpec, rows: np.ndarray) -> List[Any]:
    if spec.regime == "level":
        return [cohort.level_sample(int(row), int(spec.level), k=spec.k) for row in rows]
    return [cohort.sample(int(row), spec.regime) for row in rows]


def _cell_local(cohort: shard.HardCohort, variant_rows: np.ndarray, cell: CellSpec) -> np.ndarray:
    if cell.rows == "all":
        return np.arange(variant_rows.size)
    return np.flatnonzero(cohort.masks[cell.rows][variant_rows])


def _semantic_stats_chunked(z_q: np.ndarray, z_i: np.ndarray, scores: np.ndarray) -> np.ndarray:
    n = int(z_q.shape[0])
    out = np.empty((n, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, n, SEM_CHUNK):
        stop = min(start + SEM_CHUNK, n)
        out[start:stop] = sfeat.semantic_stats(z_q[start:stop], z_i[start:stop], scores[start:stop])
    return out


def _build_cell(
    cell: CellSpec,
    batch: hscores.B3ExampleBatch,
    scores_full: np.ndarray,
    local: np.ndarray,
    cohort: shard.HardCohort,
    variant_rows: np.ndarray,
    seed_models: sfrozen.FrozenSeedModels,
) -> CellData:
    scores = np.asarray(scores_full[local], dtype=np.float32)
    z_q = np.asarray(batch.z_q[local], dtype=np.float32)   # float16 values widened to f32
    z_i = np.asarray(batch.z_i[local], dtype=np.float32)
    scores64 = scores.astype(np.float64)
    temperature = float(seed_models.temperature)

    stats17 = np.asarray(rfeat.stat_features(scores64, temperature=temperature), dtype=np.float64)
    sem16 = _semantic_stats_chunked(z_q, z_i, scores64)
    stats_conf, e1b_conf = sfrozen.apply_frozen(seed_models, stats17, sem16)
    scalars = rfeat.scalar_confidence(scores64, temperature=temperature)
    ordered = np.sort(scores64, axis=1)[:, ::-1]

    rows = variant_rows[local]
    return CellData(
        sentence_id=np.asarray(cohort.sentence_id[rows], dtype=np.int64),
        ref_id=np.asarray(cohort.ref_id[rows], dtype=np.int64),
        image_id=np.asarray(cohort.image_id[rows], dtype=np.int64),
        eval_split=np.asarray(cohort.eval_split[rows]),
        scores=scores,
        correct=np.argmax(scores, axis=1) == 0,
        conf={MSP_NAME: np.asarray(scalars["msp"], dtype=np.float64),
              STATS_NAME: np.asarray(stats_conf, dtype=np.float64),
              E1B_NAME: np.asarray(e1b_conf, dtype=np.float64)},
        sem16=sem16,
        stats17=stats17,
        margin=np.asarray(ordered[:, 0] - ordered[:, 1], dtype=np.float64),
        msp_score=np.asarray(scalars["msp"], dtype=np.float64),
        entropy=np.asarray(-scalars["neg_entropy"], dtype=np.float64),
    )


def _score_stage(
    args: argparse.Namespace,
    loaded: Mapping[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Dict[str, CellData]]:
    cohort: shard.HardCohort = loaded["cohort"]
    corpus: B3Corpus = loaded["corpus"]
    models: Mapping[str, Any] = loaded["models"]
    frozen_rel: sfrozen.FrozenReliability = loaded["frozen"]
    seeds: Sequence[str] = tuple(args.seeds)
    text_cache: Dict[int, np.ndarray] = {}
    cell_data: Dict[str, Dict[str, CellData]] = {scorer: {} for scorer in seeds}
    for spec in VARIANTS:
        rows = _variant_rows(cohort, spec)
        samples = _variant_samples(cohort, spec, rows)
        batch = hscores.materialise_examples(corpus, samples, spec.k, text_cache=text_cache)
        log(f"[phase1f] variant {spec.variant}: {rows.size} rows x K={spec.k} materialised")
        for scorer in seeds:
            scores_full = hscores.score_examples(models[scorer], batch, batch_size=args.batch_size)
            for cell in CELLS_BY_SOURCE[spec.variant]:
                local = _cell_local(cohort, rows, cell)
                cell_data[scorer][cell.cell] = _build_cell(
                    cell, batch, scores_full, local, cohort, rows, frozen_rel.seeds[scorer]
                )
                marker = " (STOP-verified variant)" if spec.stop_k is not None else ""
                log(f"[phase1f]   {scorer}/{cell.cell}: n={local.size}{marker}")
        del batch
    return cell_data


# ---------------------------------------------------------------------------
# point metrics + b3_mean aggregation
# ---------------------------------------------------------------------------
def _point_rows(cell_data: Mapping[str, Mapping[str, CellData]], seeds: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scorer in seeds:
        for cell in CELL_SPECS:
            data = cell_data[scorer][cell.cell]
            for model in CELL_MODELS:
                metrics = reval.point_metric_row(
                    data.conf[model], data.correct, probability=data.conf[model]
                )
                row: Dict[str, Any] = {
                    "scorer": scorer, "cell": cell.cell, "regime": cell.regime, "K": cell.k,
                    "cohort": cell.cohort, "level": "" if cell.level is None else cell.level,
                    "hard_fraction": cell.hard_fraction, "n": int(data.correct.size),
                    "model": model, "b3_accuracy": float(np.mean(data.correct)),
                }
                row.update({key: metrics[key] for key in metrics})
                rows.append(row)
    return rows


def _b3_mean(rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str],
             mean_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        groups.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key, members in groups.items():
        mean_row: Dict[str, Any] = dict(members[0])
        mean_row["scorer"] = "b3_mean"
        for field in mean_fields:
            values = [
                _finite(member.get(field)) for member in members
            ]
            if values and all(value is not None for value in values):
                mean_row[field] = float(np.mean([float(v) for v in values]))
        out.append(mean_row)
    return out


# ---------------------------------------------------------------------------
# stage 4: bootstraps (pair / ratio / diff-of-diffs)
# ---------------------------------------------------------------------------
def _pair_extras(scorer: str, cell: CellSpec, ref_cell: Optional[CellSpec]) -> Dict[str, Any]:
    return {
        "scorer": scorer,
        "cell": cell.cell,
        "ref_cell": "" if ref_cell is None else ref_cell.cell,
        "cohort": cell.cohort,
        "regime": cell.regime,
        "K": cell.k,
        "level": "" if cell.level is None else cell.level,
        "hard_fraction": cell.hard_fraction,
    }


def _bootstrap_cell(
    data: CellData,
    scorer: str,
    cell: CellSpec,
    *,
    metrics: Sequence[str],
    replicates: int,
    bootstrap_seed: int,
    ci: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    clusters = np.asarray(data.image_id, dtype=np.int64)
    abs_rows = reval.model_vs_model_bootstrap_row(
        data.conf[E1B_NAME], data.correct, data.conf[STATS_NAME], data.correct, clusters,
        eval_split=POOLED, K=cell.k, model_a=E1B_NAME, model_b=STATS_NAME,
        metrics=tuple(metrics), replicates=int(replicates), seed=int(bootstrap_seed), ci=float(ci),
    )
    ratio = seval.model_vs_model_e_aurc_ratio_row(
        data.conf[E1B_NAME], data.correct, data.conf[STATS_NAME], data.correct, clusters,
        eval_split=POOLED, K=cell.k, model_sem=E1B_NAME, model_score=STATS_NAME,
        replicates=int(replicates), seed=int(bootstrap_seed), ci=float(ci),
    )
    abs_map = {str(row["metric"]): row for row in abs_rows}
    return abs_map, ratio


def _dod_rows(
    hard: CellData,
    rand: CellData,
    scorer: str,
    hard_cell: CellSpec,
    rand_cell: CellSpec,
    *,
    replicates: int,
    bootstrap_seed: int,
    ci: float,
) -> Dict[str, Dict[str, Any]]:
    if not np.array_equal(hard.sentence_id, rand.sentence_id):
        raise AssertionError(f"{scorer}/{hard_cell.cell}: hard/rand rows are not matched")
    # correctness is regime-dependent (different candidate sets), so each regime's
    # labels travel with its own SampleStats pair through the shared cluster draw.
    clusters = np.asarray(hard.image_id, dtype=np.int64)
    out: Dict[str, Dict[str, Any]] = {}
    specs = (
        ("auroc_correct", reval._metric_fn("auroc_correct"), "diff"),
        ("e_aurc_reduction", reval._metric_fn("e_aurc"), "ratio"),
        ("rer_at_50_gain_pp", reval._metric_fn("rer_at_50"), "diff"),
    )
    for name, metric_fn, kind in specs:
        args_hard = (SampleStats.from_conf_correct(hard.conf[E1B_NAME], hard.correct),
                     SampleStats.from_conf_correct(hard.conf[STATS_NAME], hard.correct))
        args_rand = (SampleStats.from_conf_correct(rand.conf[E1B_NAME], rand.correct),
                     SampleStats.from_conf_correct(rand.conf[STATS_NAME], rand.correct))
        if kind == "ratio":
            result = heval.paired_diff_of_diffs_ratio_bootstrap(
                metric_fn, args_hard[0], args_hard[1], args_rand[0], args_rand[1], clusters,
                n_replicates=int(replicates), seed=int(bootstrap_seed), ci=float(ci), metric_name=name,
            )
        else:
            result = heval.paired_diff_of_diffs_bootstrap(
                metric_fn, args_hard[0], args_hard[1], args_rand[0], args_rand[1], clusters,
                n_replicates=int(replicates), seed=int(bootstrap_seed), ci=float(ci), metric_name=name,
            )
        if name == "rer_at_50_gain_pp":
            result = dict(result)
            result["diff"] = 100.0 * float(result["diff"])
            result["hard_diff"] = 100.0 * float(result["hard_diff"])
            result["rand_diff"] = 100.0 * float(result["rand_diff"])
            result["ci_low"] = 100.0 * float(result["ci_low"])
            result["ci_high"] = 100.0 * float(result["ci_high"])
        row = _strip_arrays(result)
        row.update(_pair_extras(scorer, hard_cell, rand_cell))
        row.update({
            "row_kind": "diff_of_diffs",
            "hard_cell": hard_cell.cell,
            "rand_cell": rand_cell.cell,
            "pair": f"{hard_cell.k}",
            "bootstrap_seed": int(bootstrap_seed),
            "ci_level": float(ci),
        })
        out[name] = row
    return out


def _bootstrap_stage(
    args: argparse.Namespace,
    cell_data: Mapping[str, Mapping[str, CellData]],
    log: Callable[[str], None],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, PairBundle]]]:
    seeds: Sequence[str] = tuple(args.seeds)
    replicates = int(args.bootstrap_replicates)
    bootstrap_seed = int(args.bootstrap_seed)
    ci = float(args.ci)
    boot_rows: List[Dict[str, Any]] = []
    pair_bundles: Dict[str, Dict[str, PairBundle]] = {scorer: {} for scorer in seeds}

    for scorer in seeds:
        for rand_name, hard_name in PAIRS:
            rand_cell = CELL_BY_NAME[rand_name]
            hard_cell = CELL_BY_NAME[hard_name]
            for cell in (rand_cell, hard_cell):
                abs_map, ratio = _bootstrap_cell(
                    cell_data[scorer][cell.cell], scorer, cell,
                    metrics=PAIR_METRICS, replicates=replicates,
                    bootstrap_seed=bootstrap_seed, ci=ci,
                )
                for row in abs_map.values():
                    shaped = dict(row)
                    shaped.update(_pair_extras(scorer, cell, None))
                    shaped["row_kind"] = "pairwise"
                    shaped["bootstrap_seed"] = bootstrap_seed
                    boot_rows.append(shaped)
                ratio_row = dict(ratio)
                ratio_row.update(_pair_extras(scorer, cell, None))
                ratio_row["row_kind"] = "pairwise_ratio"
                ratio_row["bootstrap_seed"] = bootstrap_seed
                boot_rows.append(ratio_row)
                pair_bundles[scorer][cell.cell] = PairBundle(abs_rows=abs_map, ratio=ratio, dod={})
            dod = _dod_rows(
                cell_data[scorer][hard_name], cell_data[scorer][rand_name], scorer, hard_cell, rand_cell,
                replicates=replicates, bootstrap_seed=bootstrap_seed, ci=ci,
            )
            pair_bundles[scorer][hard_name] = PairBundle(
                abs_rows=pair_bundles[scorer][hard_name].abs_rows,
                ratio=pair_bundles[scorer][hard_name].ratio,
                dod=dod,
            )
            for row in dod.values():
                boot_rows.append(row)
            log(f"[phase1f] bootstrap {scorer}/K{hard_cell.k}: pairwise+ratio+diff-of-diffs done")

        for cell_name in EXPB_CELLS:
            cell = CELL_BY_NAME[cell_name]
            abs_map, ratio = _bootstrap_cell(
                cell_data[scorer][cell_name], scorer, cell,
                metrics=EXPB_METRICS, replicates=replicates, bootstrap_seed=bootstrap_seed, ci=ci,
            )
            for row in abs_map.values():
                shaped = dict(row)
                shaped.update(_pair_extras(scorer, cell, None))
                shaped["row_kind"] = "expb_pairwise"
                shaped["bootstrap_seed"] = bootstrap_seed
                boot_rows.append(shaped)
            ratio_row = dict(ratio)
            ratio_row.update(_pair_extras(scorer, cell, None))
            ratio_row["row_kind"] = "expb_ratio"
            ratio_row["bootstrap_seed"] = bootstrap_seed
            boot_rows.append(ratio_row)
            pair_bundles[scorer][cell_name] = PairBundle(abs_rows=abs_map, ratio=ratio, dod={})
        log(f"[phase1f] bootstrap {scorer}: expb levels done")
    return boot_rows, pair_bundles


# ---------------------------------------------------------------------------
# derived tables: semantic_increment / paired_random_vs_hard
# ---------------------------------------------------------------------------
def _increment_rows(
    pair_bundles: Mapping[str, Mapping[str, PairBundle]], seeds: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cells_in_order = [CELL_BY_NAME[name] for name in ("rand5", "hard5", "rand10", "hard10")] + \
        [CELL_BY_NAME[name] for name in EXPB_CELLS]
    for scorer in seeds:
        for cell in cells_in_order:
            bundle = pair_bundles[scorer].get(cell.cell)
            if bundle is None:
                continue
            auroc = bundle.abs_rows["auroc_correct"]
            rer50 = bundle.abs_rows["rer_at_50"]
            # expb cells are bootstrapped on EXPB_METRICS only: no e_aurc / rer_at_80
            rer80 = bundle.abs_rows.get("rer_at_80")
            e_abs = bundle.abs_rows.get("e_aurc")
            ratio = bundle.ratio
            rows.append({
                "scorer": scorer, "cell": cell.cell, "regime": cell.regime, "K": cell.k,
                "cohort": cell.cohort, "level": "" if cell.level is None else cell.level,
                "hard_fraction": cell.hard_fraction, "n": int(auroc["n"]),
                "delta_auroc": float(auroc["diff"]),
                "delta_auroc_ci_low": float(auroc["ci_low"]),
                "delta_auroc_ci_high": float(auroc["ci_high"]),
                "e_aurc_reduction": None if ratio is None else float(ratio["reduction"]),
                "e_aurc_reduction_ci_low": None if ratio is None else float(ratio["reduction_ci_low"]),
                "e_aurc_reduction_ci_high": None if ratio is None else float(ratio["reduction_ci_high"]),
                "delta_e_aurc": None if e_abs is None else float(e_abs["diff"]),
                "delta_e_aurc_ci_low": None if e_abs is None else float(e_abs["ci_low"]),
                "delta_e_aurc_ci_high": None if e_abs is None else float(e_abs["ci_high"]),
                "rer50_gain_pp": 100.0 * float(rer50["diff"]),
                "rer50_gain_pp_ci_low": 100.0 * float(rer50["ci_low"]),
                "rer50_gain_pp_ci_high": 100.0 * float(rer50["ci_high"]),
                "rer80_gain_pp": 100.0 * float(rer80["diff"]) if rer80 is not None else None,
                "rer80_gain_pp_ci_low": 100.0 * float(rer80["ci_low"]) if rer80 is not None else None,
                "rer80_gain_pp_ci_high": 100.0 * float(rer80["ci_high"]) if rer80 is not None else None,
            })
    return rows


def _paired_rows(
    pair_bundles: Mapping[str, Mapping[str, PairBundle]], seeds: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scorer in seeds:
        for cell_name in ("hard5", "hard10"):
            bundle = pair_bundles[scorer].get(cell_name)
            if bundle is None:
                continue
            for row in bundle.dod.values():
                rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# stage 5: manipulation check / grounding difficulty / subgroups
# ---------------------------------------------------------------------------
CRITERION_FEATURES: Tuple[str, ...] = ("cand_vmax", "cand_top12_sim", "clip_margin12")
SCORE_SCALARS: Tuple[str, ...] = ("margin", "msp", "entropy")
#: A8.7 / instruction section 21 secondary error-concentration groupings.
SUBGROUP_FEATURES: Tuple[str, ...] = ("cand_vmax", "cand_top12_sim")


def _manipulation_check(
    args: argparse.Namespace,
    cell_data: Mapping[str, Mapping[str, CellData]],
    log: Callable[[str], None],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seeds: Sequence[str] = tuple(args.seeds)
    replicates = int(args.bootstrap_replicates)
    bootstrap_seed = int(args.bootstrap_seed)
    ci = float(args.ci)
    index = {name: pos for pos, name in enumerate(sfeat.SEMANTIC_STAT_NAMES)}
    rows: List[Dict[str, Any]] = []
    per_seed_pass: Dict[str, bool] = {}

    for scorer in seeds:
        hard = cell_data[scorer]["hard5"]
        rand = cell_data[scorer]["rand5"]
        clusters = np.asarray(hard.image_id, dtype=np.int64)
        # criterion features: paired cluster bootstrap CI (A8.7)
        for feature in CRITERION_FEATURES:
            hard_values = np.asarray(hard.sem16[:, index[feature]], dtype=np.float64)
            rand_values = np.asarray(rand.sem16[:, index[feature]], dtype=np.float64)
            result = heval.paired_shift_bootstrap(
                hard_values, rand_values, clusters,
                n_replicates=replicates, seed=bootstrap_seed, ci=ci, name=feature,
            )
            row = _strip_arrays(result)
            row.update({
                "scorer": scorer, "feature": feature, "group": "semantic16",
                "cohort": "same4", "K": 5, "n": int(result["n"]),
                "n_clusters": int(result["n_clusters"]), "n_replicates": int(result["n_replicates"]),
                "ci_level": float(ci),
            })
            rows.append(row)
        # descriptive shifts: every semantic / score-stat feature + score scalars
        for feature in sfeat.SEMANTIC_STAT_NAMES:
            hard_values = np.asarray(hard.sem16[:, index[feature]], dtype=np.float64)
            rand_values = np.asarray(rand.sem16[:, index[feature]], dtype=np.float64)
            rows.append({
                "scorer": scorer, "feature": feature, "group": "semantic16",
                "cohort": "same4", "K": 5, "n": int(hard_values.size),
                "mean_hard": float(hard_values.mean()), "mean_rand": float(rand_values.mean()),
                "diff": float(hard_values.mean() - rand_values.mean()),
                "descriptive_only": True,
            })
        for pos, feature in enumerate(rfeat.stat_feature_names()):
            hard_values = np.asarray(hard.stats17[:, pos], dtype=np.float64)
            rand_values = np.asarray(rand.stats17[:, pos], dtype=np.float64)
            rows.append({
                "scorer": scorer, "feature": feature, "group": "stats17",
                "cohort": "same4", "K": 5, "n": int(hard_values.size),
                "mean_hard": float(hard_values.mean()), "mean_rand": float(rand_values.mean()),
                "diff": float(hard_values.mean() - rand_values.mean()),
                "descriptive_only": True,
            })
        for feature in SCORE_SCALARS:
            attribute = {"margin": "margin", "msp": "msp_score", "entropy": "entropy"}[feature]
            hard_values = np.asarray(getattr(hard, attribute), dtype=np.float64)
            rand_values = np.asarray(getattr(rand, attribute), dtype=np.float64)
            rows.append({
                "scorer": scorer, "feature": feature, "group": "score_scalars",
                "cohort": "same4", "K": 5, "n": int(hard_values.size),
                "mean_hard": float(hard_values.mean()), "mean_rand": float(rand_values.mean()),
                "diff": float(hard_values.mean() - rand_values.mean()),
                "descriptive_only": True,
            })
        vmax_row = next(row for row in rows if row["scorer"] == scorer and row["feature"] == "cand_vmax"
                        and row["group"] == "semantic16" and "ci_low" in row)
        top12_row = next(row for row in rows if row["scorer"] == scorer and row["feature"] == "cand_top12_sim"
                         and row["group"] == "semantic16" and "ci_low" in row)
        per_seed_pass[scorer] = bool(
            _finite(vmax_row.get("ci_low")) is not None and float(vmax_row["ci_low"]) > 0.0
            and _finite(top12_row.get("ci_low")) is not None and float(top12_row["ci_low"]) > 0.0
            and float(vmax_row["diff"]) > 0.0 and float(top12_row["diff"]) > 0.0
        )

    n_pass = int(sum(1 for value in per_seed_pass.values() if value))
    verdict = {
        "criterion": "cand_vmax and cand_top12_sim shifts positive with CI_low > 0",
        "per_seed_pass": per_seed_pass,
        "n_pass": n_pass,
        "n_seeds": len(seeds),
        "manipulation_ok": bool(n_pass >= math.ceil(2 * len(seeds) / 3)),
    }
    log(f"[phase1f] manipulation check: {n_pass}/{len(seeds)} seeds pass -> ok={verdict['manipulation_ok']}")
    return rows, verdict


def _grounding_rows(
    args: argparse.Namespace,
    cell_data: Mapping[str, Mapping[str, CellData]],
) -> List[Dict[str, Any]]:
    seeds: Sequence[str] = tuple(args.seeds)
    replicates = int(args.bootstrap_replicates)
    rows: List[Dict[str, Any]] = []
    for scorer in seeds:
        base_rows: Dict[str, Dict[str, Any]] = {}
        for cell in CELL_SPECS:
            if cell.cell not in ("rand5", "hard5", "rand10", "hard10"):
                continue
            data = cell_data[scorer][cell.cell]
            row = {
                "scorer": scorer, "cell": cell.cell, "regime": cell.regime, "K": cell.k,
                "cohort": cell.cohort, "n": int(data.correct.size),
                "b3_accuracy": float(np.mean(data.correct)),
                "margin_mean": float(data.margin.mean()),
                "msp_mean": float(data.msp_score.mean()),
                "entropy_mean": float(data.entropy.mean()),
            }
            base_rows[cell.cell] = row
            rows.append(row)
        for k, rand_name, hard_name in ((5, "rand5", "hard5"), (10, "rand10", "hard10")):
            hard = cell_data[scorer][hard_name]
            rand = cell_data[scorer][rand_name]
            clusters = np.asarray(hard.image_id, dtype=np.int64)
            acc = heval.paired_shift_bootstrap(
                hard.correct.astype(np.float64), rand.correct.astype(np.float64), clusters,
                n_replicates=replicates, seed=int(args.bootstrap_seed), ci=float(args.ci), name="accuracy",
            )
            rows.append({
                "scorer": scorer, "cell": f"shift_K{k}", "regime": "hard-minus-random", "K": k,
                "cohort": hard_name.replace("hard", "same"),
                "metric": "b3_accuracy", "diff": float(acc["diff"]),
                "ci_low": float(acc["ci_low"]), "ci_high": float(acc["ci_high"]),
                "n": int(acc["n"]), "n_clusters": int(acc["n_clusters"]),
                "n_replicates": int(acc["n_replicates"]), "ci_level": float(acc["ci_level"]),
            })
            for metric, hard_value, rand_value in (
                ("margin_mean", hard.margin.mean(), rand.margin.mean()),
                ("msp_mean", hard.msp_score.mean(), rand.msp_score.mean()),
                ("entropy_mean", hard.entropy.mean(), rand.entropy.mean()),
            ):
                rows.append({
                    "scorer": scorer, "cell": f"shift_K{k}", "regime": "hard-minus-random", "K": k,
                    "cohort": hard_name.replace("hard", "same"), "metric": metric,
                    "diff": float(hard_value - rand_value), "descriptive_only": True,
                })
    return rows


def _subgroup_rows(
    args: argparse.Namespace,
    cell_data: Mapping[str, Mapping[str, CellData]],
) -> List[Dict[str, Any]]:
    seeds: Sequence[str] = tuple(args.seeds)
    index = {name: pos for pos, name in enumerate(sfeat.SEMANTIC_STAT_NAMES)}
    rows: List[Dict[str, Any]] = []
    for scorer in seeds:
        data = cell_data[scorer]["hard5"]
        for group_by in SUBGROUP_FEATURES:
            values = np.asarray(data.sem16[:, index[group_by]], dtype=np.float64)
            labels = heval.quartile_labels(values)
            for group in ("all", "Q1", "Q2", "Q3", "Q4"):
                sel = np.ones(values.size, dtype=bool) if group == "all" else labels == group
                if not np.any(sel):
                    continue
                conf_e1b = np.asarray(data.conf[E1B_NAME], dtype=np.float64)[sel]
                conf_stats = np.asarray(data.conf[STATS_NAME], dtype=np.float64)[sel]
                corr = np.asarray(data.correct, dtype=bool)[sel]
                row: Dict[str, Any] = {
                    "scorer": scorer, "cell": "hard5", "group_by": group_by, "group": group,
                    "n": int(sel.sum()), "b3_accuracy": float(np.mean(corr)),
                }
                for model, conf in ((E1B_NAME, conf_e1b), (STATS_NAME, conf_stats)):
                    if np.unique(corr).size < 2:
                        row[f"auroc_{model}"] = None
                        row[f"e_aurc_{model}"] = None
                        continue
                    metrics = reval.point_metric_row(conf, corr)
                    row[f"auroc_{model}"] = metrics["auroc_correct"]
                    row[f"e_aurc_{model}"] = metrics["e_aurc"]
                row["delta_auroc"] = (
                    None if row[f"auroc_{E1B_NAME}"] is None or row[f"auroc_{STATS_NAME}"] is None
                    else float(row[f"auroc_{E1B_NAME}"] - row[f"auroc_{STATS_NAME}"])
                )
                row["e_aurc_reduction"] = (
                    None if not row[f"e_aurc_{STATS_NAME}"] or not row[f"e_aurc_{E1B_NAME}"]
                    else float((row[f"e_aurc_{STATS_NAME}"] - row[f"e_aurc_{E1B_NAME}"]) / row[f"e_aurc_{STATS_NAME}"])
                )
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# stage 6: A8.6 gate
# ---------------------------------------------------------------------------
def _gate_payload(
    args: argparse.Namespace,
    pair_bundles: Mapping[str, Mapping[str, PairBundle]],
    manipulation: Mapping[str, Any],
    cell_data: Mapping[str, Mapping[str, CellData]],
) -> Dict[str, Any]:
    seeds: Sequence[str] = tuple(args.seeds)
    cell: Dict[str, Any] = {}
    cell_std: Dict[str, Any] = {}
    for field, source, scale in (
        ("delta_auroc", "auroc", 1.0),
        ("delta_auroc_ci_low", "auroc_low", 1.0),
        ("delta_auroc_ci_high", "auroc_high", 1.0),
        ("e_aurc_reduction", "ratio", 1.0),
        ("e_aurc_reduction_ci_low", "ratio_low", 1.0),
        ("e_aurc_reduction_ci_high", "ratio_high", 1.0),
        ("rer50_gain_pp", "rer50", 100.0),
        ("rer50_gain_pp_ci_low", "rer50_low", 100.0),
        ("rer50_gain_pp_ci_high", "rer50_high", 100.0),
    ):
        values: List[float] = []
        for scorer in seeds:
            bundle = pair_bundles[scorer]["hard5"]
            if source == "auroc":
                values.append(float(bundle.abs_rows["auroc_correct"]["diff"]))
            elif source == "auroc_low":
                values.append(float(bundle.abs_rows["auroc_correct"]["ci_low"]))
            elif source == "auroc_high":
                values.append(float(bundle.abs_rows["auroc_correct"]["ci_high"]))
            elif source == "ratio":
                values.append(float(bundle.ratio["reduction"]))
            elif source == "ratio_low":
                values.append(float(bundle.ratio["reduction_ci_low"]))
            elif source == "ratio_high":
                values.append(float(bundle.ratio["reduction_ci_high"]))
            elif source == "rer50":
                values.append(100.0 * float(bundle.abs_rows["rer_at_50"]["diff"]))
            elif source == "rer50_low":
                values.append(100.0 * float(bundle.abs_rows["rer_at_50"]["ci_low"]))
            else:
                values.append(100.0 * float(bundle.abs_rows["rer_at_50"]["ci_high"]))
        cell[field] = float(np.mean(values))
        cell_std[field] = float(np.std(values))

    dod_auroc = [float(pair_bundles[scorer]["hard5"].dod["auroc_correct"]["diff"]) for scorer in seeds]
    dod_auroc_low = [float(pair_bundles[scorer]["hard5"].dod["auroc_correct"]["ci_low"]) for scorer in seeds]
    dod_auroc_high = [float(pair_bundles[scorer]["hard5"].dod["auroc_correct"]["ci_high"]) for scorer in seeds]
    dod = {
        "diff": float(np.mean(dod_auroc)),
        "ci_low": float(np.mean(dod_auroc_low)),
        "ci_high": float(np.mean(dod_auroc_high)),
        "per_seed": {scorer: float(value) for scorer, value in zip(seeds, dod_auroc)},
    }
    seed_lows = [float(pair_bundles[scorer]["hard5"].abs_rows["auroc_correct"]["ci_low"]) for scorer in seeds]
    seed_mean_delta = float(np.mean([float(pair_bundles[scorer]["hard5"].abs_rows["auroc_correct"]["diff"]) for scorer in seeds]))
    verdict = heval.hard_gate(
        cell, dod, seed_delta_ci_lows=seed_lows, seed_mean_delta=seed_mean_delta, thresholds=None
    )

    k10_cell = {
        "delta_auroc": float(np.mean([float(pair_bundles[scorer]["hard10"].abs_rows["auroc_correct"]["diff"]) for scorer in seeds])),
        "e_aurc_reduction": float(np.mean([float(pair_bundles[scorer]["hard10"].ratio["reduction"]) for scorer in seeds])),
        "rer50_gain_pp": float(np.mean([100.0 * float(pair_bundles[scorer]["hard10"].abs_rows["rer_at_50"]["diff"]) for scorer in seeds])),
    }
    return {
        "protocol": "A8.6 (frozen before any hard-regime result)",
        "comparison": f"{STATS_NAME} -> {E1B_NAME} on SameCat-K5 (matched random control)",
        "cell_samecat_k5": cell,
        "cell_samecat_k5_seed_std": cell_std,
        "diff_of_diffs_auroc": dod,
        "per_seed_delta_ci_low_k5": seed_lows,
        "seed_mean_delta_k5": seed_mean_delta,
        "verdict": verdict,
        "k10_cell_informational": k10_cell,
        "manipulation": dict(manipulation),
        "notes": [
            "cell values are 3-seed means of per-seed paired-bootstrap rows (values and CI endpoints).",
            "cell_samecat_k5_seed_std is the across-seed standard deviation of those per-seed values (A8.5 mean +/- std).",
            "diff_of_diffs = dAUROC^hard - dAUROC^rand, same image-cluster draw (A8.5).",
            "per-seed significance = per-seed SameCat-K5 dAUROC bootstrap CI_low > 0.",
            "never read without the manipulation-check block: a failed check invalidates the stress test (A8.7).",
        ],
    }


# ---------------------------------------------------------------------------
# stage 7: figures (A8 section 28)
# ---------------------------------------------------------------------------
def _figures(
    out_dir: Path,
    args: argparse.Namespace,
    pair_bundles: Mapping[str, Mapping[str, PairBundle]],
    cell_data: Mapping[str, Mapping[str, CellData]],
    log: Callable[[str], None],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    seeds: Sequence[str] = tuple(args.seeds)
    written: List[str] = []

    def _save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    # Figure 1: B3 accuracy random vs hard (K5)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(seeds))
    rand_acc = [float(np.mean(cell_data[scorer]["rand5"].correct)) for scorer in seeds]
    hard_acc = [float(np.mean(cell_data[scorer]["hard5"].correct)) for scorer in seeds]
    ax.bar(xs - 0.18, rand_acc, width=0.36, label="random K5", color="#7f9fc4")
    ax.bar(xs + 0.18, hard_acc, width=0.36, label="same-category K5", color="#b4635f")
    ax.set_xticks(xs, list(seeds))
    ax.set_ylabel("B3 grounding accuracy")
    ax.set_title("Fig 1 - stress-test strength: random vs GT same-category (K=5)")
    ax.legend()
    _save(fig, "fig1_grounding_accuracy_random_vs_hard.png")

    # Figure 2: dAUROC(E1b - Stats) random vs hard (K5), 3 seeds + mean
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rand_delta = [float(pair_bundles[scorer]["rand5"].abs_rows["auroc_correct"]["diff"]) for scorer in seeds]
    hard_delta = [float(pair_bundles[scorer]["hard5"].abs_rows["auroc_correct"]["diff"]) for scorer in seeds]
    rx = xs - 0.18
    hx = xs + 0.18
    ax.scatter(rx, rand_delta, color="#7f9fc4", label="random K5", zorder=3)
    ax.scatter(hx, hard_delta, color="#b4635f", label="same-category K5", zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(heval.A8_THRESHOLDS["auroc_gain"], color="grey", linestyle="--", linewidth=0.8,
               label="A8 threshold 0.02")
    ax.set_xticks(xs, list(seeds))
    ax.set_ylabel("dAUROC (E1b - Stats)")
    ax.set_title("Fig 2 - semantic increment by regime (K=5)")
    ax.legend()
    _save(fig, "fig2_delta_auroc_random_vs_hard.png")

    # Figure 3: E-AURC reduction and RER@50 gain, random vs hard (3-seed mean + dots)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    red_rand = [100.0 * float(pair_bundles[scorer]["rand5"].ratio["reduction"]) for scorer in seeds]
    red_hard = [100.0 * float(pair_bundles[scorer]["hard5"].ratio["reduction"]) for scorer in seeds]
    rer_rand = [100.0 * float(pair_bundles[scorer]["rand5"].abs_rows["rer_at_50"]["diff"]) for scorer in seeds]
    rer_hard = [100.0 * float(pair_bundles[scorer]["hard5"].abs_rows["rer_at_50"]["diff"]) for scorer in seeds]
    for ax, rand_values, hard_values, ylabel, title in (
        (axes[0], red_rand, red_hard, "E-AURC reduction (%)", "E-AURC reduction"),
        (axes[1], rer_rand, rer_hard, "RER@50 gain (pp)", "RER@50 gain"),
    ):
        ax.bar([-0.18], [np.mean(rand_values)], width=0.36, color="#7f9fc4", label="random K5")
        ax.bar([0.18], [np.mean(hard_values)], width=0.36, color="#b4635f", label="same-category K5")
        ax.scatter(np.full(len(seeds), -0.18), rand_values, color="#2f4b6e", zorder=3, s=22)
        ax.scatter(np.full(len(seeds), 0.18), hard_values, color="#7a2f2b", zorder=3, s=22)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks([0.0], [""])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Fig 3 - frozen semantic gain by regime (K=5, 3 seeds + mean)")
    _save(fig, "fig3_reduction_and_rer50.png")

    # Figure 4: cand_vmax distribution random vs same-category (seed 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    index = {name: pos for pos, name in enumerate(sfeat.SEMANTIC_STAT_NAMES)}
    scorer = seeds[0]
    hard_values = np.asarray(cell_data[scorer]["hard5"].sem16[:, index["cand_vmax"]], dtype=np.float64)
    rand_values = np.asarray(cell_data[scorer]["rand5"].sem16[:, index["cand_vmax"]], dtype=np.float64)
    ax.hist(rand_values, bins=40, alpha=0.6, label="random K5", color="#7f9fc4", density=True)
    ax.hist(hard_values, bins=40, alpha=0.6, label="same-category K5", color="#b4635f", density=True)
    ax.set_xlabel("cand_vmax")
    ax.set_ylabel("density")
    ax.set_title(f"Fig 4 - candidate competition shift ({scorer})")
    ax.legend()
    _save(fig, "fig4_cand_vmax_distribution.png")
    log(f"[phase1f] figures written: {len(written)}")
    return written


# ---------------------------------------------------------------------------
# stage 8: artifacts (dumps)
# ---------------------------------------------------------------------------
def _cell_manifest_indices(cohort: shard.HardCohort, cell: CellSpec) -> Tuple[np.ndarray, np.ndarray]:
    spec = next(spec for spec in VARIANTS if spec.variant == cell.source)
    rows = _variant_rows(cohort, spec)
    local = _cell_local(cohort, rows, cell)
    keep = rows[local]
    indices = np.empty((keep.size, cell.k), dtype=np.int64)
    for pos, row in enumerate(keep.tolist()):
        if cell.level is None:
            indices[pos] = cohort.candidate_indices(row, spec.regime, cell.k)
        else:
            indices[pos] = cohort.level_indices(row, cell.level, k=cell.k)
    return indices, keep


def _dump_manifests(out_dir: Path, cohort: shard.HardCohort, log: Callable[[str], None]) -> List[str]:
    man_dir = Path(out_dir) / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {}
    written: List[str] = []
    for cell in CELL_SPECS:
        indices, keep = _cell_manifest_indices(cohort, cell)
        path = man_dir / f"{cell.cell}_candidates.npz"
        np.savez_compressed(
            path,
            sentence_id=np.asarray(cohort.sentence_id[keep], dtype=np.int64),
            ref_id=np.asarray(cohort.ref_id[keep], dtype=np.int64),
            image_id=np.asarray(cohort.image_id[keep], dtype=np.int64),
            candidate_indices=indices,
            target_index=indices[:, 0].copy(),
        )
        written.append(str(path))
        index[cell.cell] = {
            "cell": cell.cell, "regime": cell.regime, "K": cell.k, "cohort": cell.cohort,
            "level": cell.level, "hard_fraction": cell.hard_fraction,
            "construction": CONSTRUCTION_NOTES[cell.cell],
            "n_rows": int(keep.size),
            "n_images": int(np.unique(cohort.image_id[keep]).size),
            "file": path.name,
            "source": MANIFEST_PROVENANCE,
        }
    _write_json(man_dir / "index.json", index)
    log(f"[phase1f] manifest dump: {len(written)} cell tables + index.json")
    return written


def _dump_predictions(
    out_dir: Path,
    cell_data: Mapping[str, Mapping[str, CellData]],
    seeds: Sequence[str],
) -> List[str]:
    pred_dir = Path(out_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for scorer in seeds:
        for cell in CELL_SPECS:
            data = cell_data[scorer][cell.cell]
            path = pred_dir / f"{cell.cell}__{scorer}.npz"
            np.savez_compressed(
                path,
                sentence_id=data.sentence_id,
                ref_id=data.ref_id,
                image_id=data.image_id,
                correct=np.asarray(data.correct, dtype=np.int8),
                scores=np.asarray(data.scores, dtype=np.float32),
                conf_msp=np.asarray(data.conf[MSP_NAME], dtype=np.float64),
                conf_stats=np.asarray(data.conf[STATS_NAME], dtype=np.float64),
                conf_e1b=np.asarray(data.conf[E1B_NAME], dtype=np.float64),
                margin=np.asarray(data.margin, dtype=np.float64),
                entropy=np.asarray(data.entropy, dtype=np.float64),
            )
            written.append(str(path))
    return written


def _protocol_payload(
    args: argparse.Namespace,
    loaded: Mapping[str, Any],
    stop_checks: Mapping[str, float],
    reference_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    frozen_rel: sfrozen.FrozenReliability = loaded["frozen"]
    cohort: shard.HardCohort = loaded["cohort"]
    return {
        "phase": "1F",
        "amendment": "A8 - Hard-Competition Semantic Confirmation",
        "stage_declaration": (
            "staged confirmatory follow-up created after the Phase 1 A7 verdict was known "
            "(INCONCLUSIVE); not part of the original preregistration; A8 gate frozen before "
            "any hard-regime result existed"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_models": {
            "scorers": list(args.seeds),
            "temperature_source": "results/phase0b_independent/seed_{i}/eval_metadata.json",
            "verification": {key: dict(value) for key, value in frozen_rel.verification.items()},
        },
        "cohort": shard.cohort_summary(cohort),
        "stop_checks": dict(stop_checks),
        "cells": [
            {"cell": cell.cell, "regime": cell.regime, "K": cell.k, "cohort": cell.cohort,
             "level": cell.level, "hard_fraction": cell.hard_fraction,
             "construction": CONSTRUCTION_NOTES[cell.cell]}
            for cell in CELL_SPECS
        ],
        "gate_thresholds": dict(heval.A8_THRESHOLDS),
        "bootstrap": {
            "replicates": int(args.bootstrap_replicates), "seed": int(args.bootstrap_seed),
            "ci": float(args.ci), "unit": "image cluster",
        },
        "manifests": MANIFEST_PROVENANCE,
        "reference_hashes": dict(reference_hashes),
        "prohibited": [
            "training new reliability models", "retraining E1b", "feature selection",
            "E2/E3 tuning", "candidate-aware grounding", "reranking", "Transformer",
            "attention", "FineCops", "CLIP-hard primary analysis", "backbone changes",
            "target omission",
        ],
        "artifacts": [
            "protocol.json", "cohort_summary.json", "manifests/", "manipulation_check.csv",
            "grounding_difficulty.csv", "reliability_metrics.csv", "semantic_increment.csv",
            "paired_random_vs_hard.csv", "bootstrap.csv", "subgroup_analysis.csv", "gate.json",
            "figures/", "predictions/", "metadata.json",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": loaded["device"],
        },
    }


def _reference_hashes(args: argparse.Namespace, seeds: Sequence[str]) -> Dict[str, str]:
    paths: List[Path] = [
        Path(args.manifests) / "random_testA.jsonl",
        Path(args.manifests) / "random_testB.jsonl",
        Path(args.manifests) / "same_category_testA.jsonl",
        Path(args.manifests) / "same_category_testB.jsonl",
        Path(args.phase05) / "split_manifest.json",
        Path(args.phase05) / "stats_logistic" / "selection.json",
        Path(args.phase1) / "e1_logistic" / "coefficients.csv",
        Path(args.phase1) / "e1_logistic" / "selection.json",
    ]
    for scorer in seeds:
        tag = scorer.split("b3_seed")[1]
        for k in (5, 10):
            paths.append(Path(args.b3_root) / f"seed_{tag}" / "raw_scores" / f"K{k}.npz")
    out: Dict[str, str] = {}
    for path in paths:
        if path.exists():
            out[str(path)] = _sha256_file(path)
    return out


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1F - hard-competition semantic confirmation (Amendment A8)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--features", type=Path, default=Path("cache/features"))
    parser.add_argument("--manifests", type=Path, default=Path("cache/manifests"))
    parser.add_argument("--bank", type=Path, default=Path("cache/proposals.h5"))
    parser.add_argument("--refs", type=Path, default=Path("data/raw/refcoco+/refcoco+/refs(unc).p"))
    parser.add_argument("--image-sizes", type=Path, default=Path("cache/image_sizes.npz"))
    parser.add_argument("--b3-root", type=Path, default=Path("results/phase0b_independent"))
    parser.add_argument("--phase05", type=Path, default=Path("results/phase05_score_sufficiency"))
    parser.add_argument("--phase1", type=Path, default=Path("results/phase1_semantic_sufficiency"))
    parser.add_argument("--embeddings", type=Path, default=Path("cache/semantic_phase1"))
    parser.add_argument("--seeds", nargs="+", default=list(B3_SEEDS), choices=list(B3_SEEDS))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--stop-atol", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-file", type=Path, default=Path("logs_phase1f_driver.txt"))
    parser.add_argument("--smoke", action="store_true",
                        help="fast end-to-end check: 1 seed, 100 bootstrap replicates, *_smoke out dir")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.seeds = list(args.seeds[:1])
        args.bootstrap_replicates = min(int(args.bootstrap_replicates), 100)
        if args.out == DEFAULT_OUT:
            args.out = DEFAULT_OUT.with_name(DEFAULT_OUT.name + "_smoke")
    log = _make_logger(Path(args.log_file))
    started = time.perf_counter()
    stage_time: Dict[str, float] = {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def stage(name: str, t0: float) -> None:
        stage_time[name] = time.perf_counter() - t0
        log(f"[phase1f] stage {name}: {stage_time[name]:.1f}s")

    log(f"[phase1f] start (seeds={list(args.seeds)}, replicates={args.bootstrap_replicates}, "
        f"out={out_dir}, smoke={bool(args.smoke)})")

    # -- load ----------------------------------------------------------------
    t0 = time.perf_counter()
    loaded = _load_stage(args, log)
    stage("load", t0)

    # -- STOP check ----------------------------------------------------------
    t0 = time.perf_counter()
    stop_checks = _stop_check(loaded["cohort"], loaded["corpus"], loaded["models"], list(args.seeds), args, log)
    stage("stop_check", t0)

    # -- protocol.json (before any modelling output is consumed) -------------
    t0 = time.perf_counter()
    _write_json(out_dir / "protocol.json", _protocol_payload(
        args, loaded, stop_checks, _reference_hashes(args, list(args.seeds))))
    _write_json(out_dir / "cohort_summary.json", {
        **shard.cohort_summary(loaded["cohort"]),
        "stop_checks_max_abs_delta": dict(stop_checks),
        "frozen_verification": {key: dict(value) for key, value in loaded["frozen"].verification.items()},
        "matched_control": "per-ref target / sorted-order / n_same equality asserted in load_hard_cohort",
    })
    stage("protocol", t0)

    # -- frozen scoring ------------------------------------------------------
    t0 = time.perf_counter()
    cell_data = _score_stage(args, loaded, log)
    stage("score", t0)

    # -- point metrics -------------------------------------------------------
    t0 = time.perf_counter()
    point_rows = _point_rows(cell_data, list(args.seeds))
    agg_rows = point_rows + _b3_mean(point_rows, ("cell", "model"), AGG_METRIC_FIELDS)
    _write_csv(out_dir / "reliability_metrics.csv", agg_rows)
    stage("metrics", t0)

    # -- bootstraps ----------------------------------------------------------
    t0 = time.perf_counter()
    boot_rows, pair_bundles = _bootstrap_stage(args, cell_data, log)
    _write_csv(out_dir / "bootstrap.csv", boot_rows)
    increment_rows = _increment_rows(pair_bundles, list(args.seeds))
    increment_rows = increment_rows + _b3_mean(increment_rows, ("cell",), INCREMENT_NUMERIC_FIELDS)
    _write_csv(out_dir / "semantic_increment.csv", increment_rows)
    paired_rows = _paired_rows(pair_bundles, list(args.seeds))
    mean_key_fields = ("pair", "metric")
    mean_fields = ("hard_diff", "rand_diff", "diff", "ci_low", "ci_high", "hard_reduction", "rand_reduction")
    paired_rows = paired_rows + _b3_mean(paired_rows, mean_key_fields, mean_fields)
    _write_csv(out_dir / "paired_random_vs_hard.csv", paired_rows)
    stage("bootstrap", t0)

    # -- analyses ------------------------------------------------------------
    t0 = time.perf_counter()
    manipulation_rows, manipulation = _manipulation_check(args, cell_data, log)
    _write_csv(out_dir / "manipulation_check.csv", manipulation_rows)
    grounding_rows = _grounding_rows(args, cell_data)
    grounding_base = [row for row in grounding_rows if "metric" not in row]
    grounding_shift = [row for row in grounding_rows if "metric" in row]
    grounding_base = grounding_base + _b3_mean(
        grounding_base, ("cell",), ("n", "b3_accuracy", "margin_mean", "msp_mean", "entropy_mean")
    )
    grounding_shift = grounding_shift + _b3_mean(
        grounding_shift, ("cell", "metric"), ("diff", "ci_low", "ci_high")
    )
    _write_csv(out_dir / "grounding_difficulty.csv", grounding_base + grounding_shift)
    subgroup_rows = _subgroup_rows(args, cell_data)
    subgroup_rows = subgroup_rows + _b3_mean(
        subgroup_rows, ("group_by", "group"),
        ("n", "b3_accuracy", f"auroc_{E1B_NAME}", f"auroc_{STATS_NAME}", "delta_auroc",
         f"e_aurc_{E1B_NAME}", f"e_aurc_{STATS_NAME}", "e_aurc_reduction"),
    )
    _write_csv(out_dir / "subgroup_analysis.csv", subgroup_rows)
    stage("analyses", t0)

    # -- gate ----------------------------------------------------------------
    t0 = time.perf_counter()
    gate = _gate_payload(args, pair_bundles, manipulation, cell_data)
    _write_json(out_dir / "gate.json", gate)
    log(f"[phase1f] A8 gate verdict = {gate['verdict']['verdict']} ({gate['verdict']['verdict_label']})")
    stage("gate", t0)

    # -- figures / dumps -----------------------------------------------------
    t0 = time.perf_counter()
    figures = _figures(out_dir, args, pair_bundles, cell_data, log)
    manifest_files = _dump_manifests(out_dir, loaded["cohort"], log)
    prediction_files = _dump_predictions(out_dir, cell_data, list(args.seeds))
    stage("figures_dumps", t0)

    metadata = {
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.perf_counter() - started,
        "stage_seconds": stage_time,
        "smoke": bool(args.smoke),
        "seeds": list(args.seeds),
        "bootstrap": {"replicates": int(args.bootstrap_replicates), "seed": int(args.bootstrap_seed),
                      "ci": float(args.ci)},
        "gate_verdict": gate["verdict"]["verdict"],
        "gate_label": gate["verdict"]["verdict_label"],
        "manipulation_ok": bool(manipulation["manipulation_ok"]),
        "figures": figures,
        "n_manifest_files": len(manifest_files),
        "n_prediction_files": len(prediction_files),
        "files": ["protocol.json", "cohort_summary.json", "reliability_metrics.csv",
                  "semantic_increment.csv", "paired_random_vs_hard.csv", "bootstrap.csv",
                  "manipulation_check.csv", "grounding_difficulty.csv", "subgroup_analysis.csv",
                  "gate.json"],
    }
    _write_json(out_dir / "metadata.json", metadata)
    log(f"[phase1f] done in {metadata['runtime_seconds']:.1f}s")
    print(f"PHASE1F_COMPLETE verdict={metadata['gate_verdict']} out={out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
