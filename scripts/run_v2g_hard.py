#!/usr/bin/env python
"""V2-G G4 Phase B: hard-competition amplification + manipulation + gate.

Backbone-general replication of the V1 hard-semantic mechanism (protocol
``results/v2_backbone_generalization/g4_protocol_freeze.json``, items 6-10).
The frozen V2 reliability models fitted in Phase A (``g4_phaseA/{bb}/seed_{s}/
models.pkl``: train-only normalisation + R1/R2 Logistic + selected C + corrected
T) are applied *verbatim* to matched Random / Same-Category candidate cells built
from the frozen Phase 0.5 canonical hard cohort; only the candidate composition
changes between the two regimes.

Pipeline (mirrors scripts/run_phase1f.py, adapted to V2):

    load cohort + V2 corpus + frozen V2 scorers + Phase A pickles
    -> STOP check (random C_K reproduces the frozen raw_scores)
    -> materialise + score every variant, build the matched cells
    -> apply the frozen R1/R2 to each cell -> confidences
    -> amplification A = dAUROC^hard - dAUROC^rand (shared image-cluster draw)
    -> V2 manipulation check (winner_competitor_max_cos up, winner_top2_cos up,
       q_margin12 down)
    -> V2 hard gate (freeze item 9, NOT the V1 A8 gate)
    -> dose response (expb_m0/2/4/8, Spearman rho) + G5 cross-backbone table

Usage
-----
    conda activate deepminer
    python -u scripts/run_v2g_hard.py --device cuda \
        --bootstrap-replicates 5000 --log-file logs_v2g_phaseB.txt
    python -u scripts/run_v2g_hard.py --smoke       # 1 seed, 100 reps

Nothing here trains or re-tunes: every frozen choice is read from the freeze
file / Phase A artifacts.  A suspect alignment (matched-cohort or STOP drift)
raises immediately rather than writing a suspect artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402
from ccg.semantic import hard_eval as heval  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

__all__ = ["build_parser", "main"]

# ---------------------------------------------------------------------------
# frozen configuration (cross-checked against g4_protocol_freeze.json)
# ---------------------------------------------------------------------------
MANIFESTS = Path("cache/manifests")
BANK = Path("cache/proposals.h5")
REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
IMAGE_SIZES = Path("cache/image_sizes.npz")
PHASE05_DIR = Path("results/phase05_score_sufficiency/features")

V2_ROOT = Path("results/v2_backbone_generalization")
FREEZE_PATH = V2_ROOT / "g4_protocol_freeze.json"
A_OUT = V2_ROOT / "g4_phaseA"
B_OUT = V2_ROOT / "g4_phaseB"
G3_PATH = V2_ROOT / "g3_cardinality_gate.json"

SEEDS: Tuple[int, ...] = (1, 2, 3)
BACKBONES: Tuple[Tuple[str, str], ...] = (
    ("b1", "openclip_b16"),
    ("b2", "siglip_b16"),
)
POOLED = shard.POOLED
R1_NAME = "R1_stats"
R2_NAME = "R2_stats_sem"
STATS17_NAMES: Tuple[str, ...] = tuple(rfeat.stat_feature_names())
SEM14_NAMES: Tuple[str, ...] = tuple(v2feat.V2_PRIMARY_SEMANTIC_NAMES)

# Gate thresholds (freeze item 9) -- V2-G, NOT the V1 A8 gate.
DELTA_HARD_MIN = 0.015
AMPLIFICATION_MIN = 0.01
DOSE_RHO_MIN = 0.8
#: V2 manipulation metrics (freeze item 7).
MANIPULATION: Tuple[Tuple[str, str], ...] = tuple(v2feat.MANIPULATION_METRICS)


@dataclass(frozen=True)
class CellSpec:
    cell: str
    regime: str        # "random" | "same_category" | "level"
    k: int
    cohort: str        # row mask of the variant ("base" = full pooled cohort)
    source: str        # variant supplying the rows
    rows: str          # "all" or a cohort mask name applied on top
    level: Optional[int] = None
    hard_fraction: float = 0.0


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
CELL_BY_NAME: Dict[str, CellSpec] = {s.cell: s for s in CELL_SPECS}
EXPB_CELLS: Tuple[str, ...] = ("expb_m0", "expb_m2", "expb_m4", "expb_m8")
#: variants to materialise (each may feed more than one cell).
VARIANTS: Tuple[Tuple[str, str, int], ...] = (
    ("base_r5", "random", 5),
    ("base_r10", "random", 10),
    ("hard5", "same_category", 5),
    ("hard10", "same_category", 10),
    ("expb_m0", "level", 10),
    ("expb_m2", "level", 10),
    ("expb_m4", "level", 10),
    ("expb_m8", "level", 10),
)
LEVEL_OF_VARIANT: Dict[str, int] = {"expb_m0": 0, "expb_m2": 2, "expb_m4": 4, "expb_m8": 8}
CELLS_BY_SOURCE: Dict[str, List[CellSpec]] = {}
for _s in CELL_SPECS:
    CELLS_BY_SOURCE.setdefault(_s.source, []).append(_s)


@dataclass
class CellBundle:
    """Per-(cell, backbone, seed) arrays after frozen scoring + Phase A models."""

    sentence_id: np.ndarray
    image_id: np.ndarray
    correct: np.ndarray
    conf: Dict[str, np.ndarray]        # R1 / R2 -> [n] float64
    sem14: np.ndarray                  # [n, 14] raw (for manipulation)


# ---------------------------------------------------------------------------
# freeze + Phase A model loading
# ---------------------------------------------------------------------------
def _load_freeze() -> Dict[str, Any]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def _backbone_paths(freeze: Mapping[str, Any], tag: str) -> Dict[str, Any]:
    for key, bb in freeze["backbones"].items():
        if bb["tag"] == tag:
            return {"key": key, "features_root": Path(bb["features_root"]),
                    "b3_root": Path(bb["b3_root"]), "feature_dim": int(bb["feature_dim"])}
    raise KeyError(f"backbone tag {tag!r} absent from freeze file")


def _load_phase_a(tag: str, seed: int) -> Dict[str, Any]:
    path = A_OUT / tag / f"seed_{seed}" / "models.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Phase A model missing: {path} (run --phase A first)")
    with open(path, "rb") as fh:
        pk = pickle.load(fh)
    for key in ("stats_fit", "sem_fit", "R1_model", "R2_model", "T"):
        if key not in pk:
            raise KeyError(f"{path}: pickle missing key {key!r}")
    return pk


def _cardinality_replicated(g3_key: str) -> Optional[bool]:
    if not G3_PATH.exists():
        return None
    g3 = json.loads(G3_PATH.read_text(encoding="utf-8"))
    bb = g3.get("backbones", {}).get(g3_key)
    return None if bb is None else bool(bb.get("CARDINALITY_REPLICATED"))


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _make_logger(log_file: Optional[Path]) -> Callable[[str], None]:
    stream = open(log_file, "a", encoding="utf-8") if log_file else None

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        if stream is not None:
            stream.write(line + "\n")
            stream.flush()

    return log


def _auroc(conf: np.ndarray, correct: np.ndarray) -> Optional[float]:
    c = np.asarray(conf, dtype=np.float64).reshape(-1)
    y = np.asarray(correct, dtype=np.float64).reshape(-1)
    if c.size == 0 or np.unique(y).size < 2:
        return None
    return float(auroc_correct(c, y))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    tmp.replace(path)


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Average-rank Pearson correlation (returns None on a degenerate ranking)."""
    if len(x) < 3 or len(x) != len(y):
        return None

    def _rank(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.size, dtype=np.float64)
        i = 0
        while i < a.size:
            j = i
            while j + 1 < a.size and a[order[j + 1]] == a[order[i]]:
                j += 1
            ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# cohort / variant assembly (adapted from run_phase1f)
# ---------------------------------------------------------------------------
def _variant_rows(cohort: shard.HardCohort, cohort_mask: str) -> np.ndarray:
    if cohort_mask == "base":
        return np.arange(len(cohort))
    return np.flatnonzero(cohort.masks[cohort_mask])


def _variant_samples(cohort: shard.HardCohort, regime: str, level: Optional[int],
                     k: int, rows: np.ndarray) -> List[Any]:
    if regime == "level":
        return [cohort.level_sample(int(r), int(level), k=k) for r in rows]
    return [cohort.sample(int(r), regime) for r in rows]


def _cell_local(cohort: shard.HardCohort, variant_rows: np.ndarray, cell: CellSpec) -> np.ndarray:
    if cell.rows == "all":
        return np.arange(variant_rows.size)
    return np.flatnonzero(cohort.masks[cell.rows][variant_rows])


# ---------------------------------------------------------------------------
# STOP check (random C_K reproduces the frozen V2 raw_scores)
# ---------------------------------------------------------------------------
def _stop_check(cohort: shard.HardCohort, corpus: B3Corpus, models: Mapping[str, Any],
                paths: Mapping[str, Any], b3_seeds: Sequence[str], args: argparse.Namespace,
                log: Callable[[str], None]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    all_rows = np.arange(len(cohort))
    for variant, regime, k in VARIANTS[:2]:  # base_r5, base_r10 (random regime)
        samples = _variant_samples(cohort, regime, None, k, all_rows)
        batch = hscores.materialise_examples(corpus, samples, k, text_cache={})
        for scorer in tqdm(b3_seeds, desc=f"[{paths['key']}] STOP", unit="seed", disable=None):
            scores = hscores.score_examples(models[scorer], batch, batch_size=args.batch_size)
            ref = sdata.load_scorer_canonical(scorer, k, b3_root=paths["b3_root"])
            pos = np.searchsorted(ref.sentence_id, cohort.sentence_id)
            if not np.array_equal(ref.sentence_id[pos], cohort.sentence_id):
                raise AssertionError(
                    f"STOP {paths['key']}/{scorer} K={k}: cohort ids not a subset of frozen rows")
            frozen = np.asarray(ref.scores[pos], dtype=np.float32)
            per = hscores.verify_against_raw_scores(
                {scorer: scores}, {scorer: frozen}, k=k, atol=float(args.stop_atol))
            deltas[f"{variant}/{scorer}"] = float(per[scorer])
            log(f"[B] STOP {paths['key']}/{scorer} K={k}: max|delta|={per[scorer]:.3e}")
        del batch
    return deltas


# ---------------------------------------------------------------------------
# build one matched cell (score -> features -> Phase A models)
# ---------------------------------------------------------------------------
def _build_cell(cell: CellSpec, batch: hscores.B3ExampleBatch, scores_full: np.ndarray,
                local: np.ndarray, cohort: shard.HardCohort, variant_rows: np.ndarray,
                pk: Mapping[str, Any]) -> CellBundle:
    scores = np.asarray(scores_full[local], dtype=np.float64)
    z_q = np.asarray(batch.z_q[local], dtype=np.float64)
    z_i = np.asarray(batch.z_i[local], dtype=np.float64)
    temperature = float(pk["T"])

    stats_raw = np.asarray(rfeat.stat_features(scores, temperature=temperature), dtype=np.float64)
    sem_raw = np.asarray(v2feat.v2_primary_semantic_stats(z_q, z_i, scores), dtype=np.float64)
    if stats_raw.shape[1] != len(STATS17_NAMES):
        raise AssertionError(f"{cell.cell}: stats width {stats_raw.shape[1]} != {len(STATS17_NAMES)}")
    if sem_raw.shape[1] != len(SEM14_NAMES):
        raise AssertionError(f"{cell.cell}: semantic width {sem_raw.shape[1]} != {len(SEM14_NAMES)}")

    stats_std = np.asarray(rfeat.normalize_apply(stats_raw, pk["stats_fit"]), dtype=np.float64)
    sem_std = np.asarray(rfeat.normalize_apply(sem_raw, pk["sem_fit"]), dtype=np.float64)
    conf_r1 = np.asarray(pk["R1_model"].predict_proba(stats_std), dtype=np.float64)
    conf_r2 = np.asarray(pk["R2_model"].predict_proba(np.hstack([stats_std, sem_std])), dtype=np.float64)

    rows = variant_rows[local]
    return CellBundle(
        sentence_id=np.asarray(cohort.sentence_id[rows], dtype=np.int64),
        image_id=np.asarray(cohort.image_id[rows], dtype=np.int64),
        correct=np.argmax(scores, axis=1) == 0,
        conf={R1_NAME: conf_r1, R2_NAME: conf_r2},
        sem14=sem_raw,
    )


def _score_stage(cohort: shard.HardCohort, corpus: B3Corpus, models: Mapping[str, Any],
                 phase_a: Mapping[str, Mapping[str, Any]], b3_seeds: Sequence[str],
                 args: argparse.Namespace, log: Callable[[str], None],
                 tag: str = "") -> Dict[str, Dict[str, CellBundle]]:
    text_cache: Dict[int, np.ndarray] = {}
    cell_data: Dict[str, Dict[str, CellBundle]] = {sc: {} for sc in b3_seeds}
    for variant, regime, k in tqdm(VARIANTS, desc=f"[{tag}] score variants", unit="variant", disable=None):
        cohort_mask = next(s.cohort for s in CELL_SPECS if s.source == variant)
        rows = _variant_rows(cohort, cohort_mask)
        level = LEVEL_OF_VARIANT.get(variant)
        samples = _variant_samples(cohort, regime, level, k, rows)
        batch = hscores.materialise_examples(corpus, samples, k, text_cache=text_cache)
        log(f"[B] variant {variant}: {rows.size} rows x K={k} materialised")
        for scorer in b3_seeds:
            scores_full = hscores.score_examples(models[scorer], batch, batch_size=args.batch_size)
            pk = phase_a[scorer]
            for cell in CELLS_BY_SOURCE[variant]:
                local = _cell_local(cohort, rows, cell)
                cell_data[scorer][cell.cell] = _build_cell(
                    cell, batch, scores_full, local, cohort, rows, pk)
        del batch
    return cell_data


# ---------------------------------------------------------------------------
# amplification (dAUROC hard, diff-of-diffs A)
# ---------------------------------------------------------------------------
def _amplification_rows(cell_data: Mapping[str, Mapping[str, CellBundle]], b3_seeds: Sequence[str],
                        tag: str, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    per_seed: Dict[str, Dict[str, Any]] = {}
    auroc_fn = reval._metric_fn("auroc_correct")
    for rand_name, hard_name, k in (("rand5", "hard5", 5), ("rand10", "hard10", 10)):
        for scorer in tqdm(b3_seeds, desc=f"[{tag}] amp K{k}", unit="seed", disable=None):
            hard = cell_data[scorer][hard_name]
            rand = cell_data[scorer][rand_name]
            if not np.array_equal(hard.sentence_id, rand.sentence_id):
                raise AssertionError(f"{tag}/{scorer}: {hard_name}/{rand_name} rows not matched")
            clusters = np.asarray(hard.image_id, dtype=np.int64)
            # dAUROC^hard = AUROC(R2) - AUROC(R1) on the hard cell, paired CI.
            dh = reval.model_vs_model_bootstrap_row(
                hard.conf[R2_NAME], hard.correct, hard.conf[R1_NAME], hard.correct, clusters,
                eval_split=POOLED, K=k, model_a=R2_NAME, model_b=R1_NAME,
                metrics=("auroc_correct",), replicates=int(args.bootstrap_replicates),
                seed=int(args.bootstrap_seed), ci=float(args.ci))[0]
            dr = reval.model_vs_model_bootstrap_row(
                rand.conf[R2_NAME], rand.correct, rand.conf[R1_NAME], rand.correct, clusters,
                eval_split=POOLED, K=k, model_a=R2_NAME, model_b=R1_NAME,
                metrics=("auroc_correct",), replicates=int(args.bootstrap_replicates),
                seed=int(args.bootstrap_seed), ci=float(args.ci))[0]
            # A = dAUROC^hard - dAUROC^rand (shared cluster draw across the four blocks).
            dod = heval.paired_diff_of_diffs_bootstrap(
                auroc_fn, (hard.conf[R2_NAME], hard.correct), (hard.conf[R1_NAME], hard.correct),
                (rand.conf[R2_NAME], rand.correct), (rand.conf[R1_NAME], rand.correct), clusters,
                n_replicates=int(args.bootstrap_replicates), seed=int(args.bootstrap_seed),
                ci=float(args.ci), metric_name="auroc_correct")
            row = {
                "backbone": tag, "scorer": scorer, "K": k, "hard_cell": hard_name, "rand_cell": rand_name,
                "n": int(dh["n"]),
                "delta_hard": float(dh["diff"]), "delta_hard_ci_low": float(dh["ci_low"]),
                "delta_hard_ci_high": float(dh["ci_high"]),
                "delta_rand": float(dr["diff"]), "delta_rand_ci_low": float(dr["ci_low"]),
                "delta_rand_ci_high": float(dr["ci_high"]),
                "amplification": float(dod["diff"]), "amplification_ci_low": float(dod["ci_low"]),
                "amplification_ci_high": float(dod["ci_high"]),
                "auroc_R1_hard": _auroc(hard.conf[R1_NAME], hard.correct),
                "auroc_R2_hard": _auroc(hard.conf[R2_NAME], hard.correct),
                "bootstrap_seed": int(args.bootstrap_seed), "ci_level": float(args.ci),
            }
            rows.append(row)
            per_seed.setdefault(f"K{k}", {})[scorer] = row
    return rows, per_seed


# ---------------------------------------------------------------------------
# V2 manipulation check
# ---------------------------------------------------------------------------
def _manipulation_rows(cell_data: Mapping[str, Mapping[str, CellBundle]], b3_seeds: Sequence[str],
                       tag: str, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    index = {name: pos for pos, name in enumerate(SEM14_NAMES)}
    rows: List[Dict[str, Any]] = []
    per_seed: Dict[str, Dict[str, Any]] = {}
    for scorer in tqdm(b3_seeds, desc=f"[{tag}] manipulation", unit="seed", disable=None):
        hard = cell_data[scorer]["hard5"]
        rand = cell_data[scorer]["rand5"]
        clusters = np.asarray(hard.image_id, dtype=np.int64)
        dir_ok = 0
        ci_excl = 0
        for feature, direction in MANIPULATION:
            h = np.asarray(hard.sem14[:, index[feature]], dtype=np.float64)
            r = np.asarray(rand.sem14[:, index[feature]], dtype=np.float64)
            res = heval.paired_shift_bootstrap(
                h, r, clusters, n_replicates=int(args.bootstrap_replicates),
                seed=int(args.bootstrap_seed), ci=float(args.ci), name=feature)
            diff = float(res["diff"])
            ci_low = float(res["ci_low"])
            ci_high = float(res["ci_high"])
            correct_dir = diff > 0.0 if direction == "up" else diff < 0.0
            excludes = ci_low > 0.0 if direction == "up" else ci_high < 0.0
            dir_ok += int(correct_dir)
            ci_excl += int(excludes)
            rows.append({
                "backbone": tag, "scorer": scorer, "feature": feature, "direction": direction,
                "mean_hard": float(res["mean_hard"]), "mean_rand": float(res["mean_rand"]),
                "diff": diff, "ci_low": ci_low, "ci_high": ci_high,
                "correct_direction": bool(correct_dir), "ci_excludes_0": bool(excludes),
                "n": int(res["n"]), "n_clusters": int(res["n_clusters"]),
            })
        seed_valid = bool(dir_ok >= 2 and ci_excl >= 1)
        per_seed[scorer] = {"metrics_dir_ok": int(dir_ok), "metrics_ci_excl": int(ci_excl),
                            "seed_valid": seed_valid}
    n_valid = int(sum(1 for v in per_seed.values() if v["seed_valid"]))
    manipulation_ok = bool(n_valid >= math.ceil(2 * len(b3_seeds) / 3))
    return rows, {"metrics": [[f, d] for f, d in MANIPULATION], "per_seed": per_seed,
                  "n_seed_valid": n_valid, "n_seeds": len(b3_seeds),
                  "rule": "per-seed: >=2/3 metrics correct dir AND >=1 CI excludes 0; "
                          "backbone ok = >=2/3 seeds valid",
                  "manipulation_ok": manipulation_ok}


# ---------------------------------------------------------------------------
# dose response
# ---------------------------------------------------------------------------
def _dose_rows(cell_data: Mapping[str, Mapping[str, CellBundle]], b3_seeds: Sequence[str],
               tag: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    per_level: Dict[int, List[float]] = {LEVEL_OF_VARIANT[v]: [] for v in LEVEL_OF_VARIANT}
    rows: List[Dict[str, Any]] = []
    for scorer in b3_seeds:
        for cell_name in EXPB_CELLS:
            cell = CELL_BY_NAME[cell_name]
            data = cell_data[scorer][cell_name]
            a1 = _auroc(data.conf[R1_NAME], data.correct)
            a2 = _auroc(data.conf[R2_NAME], data.correct)
            delta = None if (a1 is None or a2 is None) else a2 - a1
            rows.append({"backbone": tag, "scorer": scorer, "cell": cell_name, "level": cell.level,
                         "hard_fraction": cell.hard_fraction, "n": int(data.correct.size),
                         "auroc_R1": a1, "auroc_R2": a2, "delta_auroc": delta})
            if delta is not None:
                per_level[int(cell.level)].append(delta)
    means = {m: float(np.mean(v)) for m, v in per_level.items() if v}
    levels = sorted(means)
    rho = _spearman(levels, [means[m] for m in levels]) if len(levels) >= 3 else None
    return rows, {"levels": levels, "delta_mean_by_level": means, "spearman_rho": rho,
                  "replicated": bool(rho is not None and rho > DOSE_RHO_MIN),
                  "role": "secondary (not in the overall gate)"}


# ---------------------------------------------------------------------------
# point metric rows (all cells x R1/R2)
# ---------------------------------------------------------------------------
def _point_rows(cell_data: Mapping[str, Mapping[str, CellBundle]], b3_seeds: Sequence[str],
                tag: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scorer in b3_seeds:
        for cell in CELL_SPECS:
            data = cell_data[scorer][cell.cell]
            for model in (R1_NAME, R2_NAME):
                metrics = reval.point_metric_row(data.conf[model], data.correct,
                                                 probability=data.conf[model])
                rows.append({
                    "backbone": tag, "scorer": scorer, "cell": cell.cell, "regime": cell.regime,
                    "K": cell.k, "cohort": cell.cohort, "level": "" if cell.level is None else cell.level,
                    "hard_fraction": cell.hard_fraction, "model": model, "n": int(data.correct.size),
                    "b3_accuracy": float(np.mean(data.correct)),
                    "auroc_correct": metrics.get("auroc_correct"),
                    "e_aurc": metrics.get("e_aurc"),
                    "rer_at_50": metrics.get("rer_at_50"),
                })
    return rows


# ---------------------------------------------------------------------------
# V2 hard gate (freeze item 9) -- NOT the V1 A8 gate
# ---------------------------------------------------------------------------
def _hard_gate(per_seed_k5: Mapping[str, Mapping[str, Any]], manipulation_ok: bool,
               seed_lows: Sequence[float], seed_deltas: Sequence[float]) -> Dict[str, Any]:
    dh = [float(v["delta_hard"]) for v in per_seed_k5.values()]
    dh_lo = [float(v["delta_hard_ci_low"]) for v in per_seed_k5.values()]
    amp = [float(v["amplification"]) for v in per_seed_k5.values()]
    amp_lo = [float(v["amplification_ci_low"]) for v in per_seed_k5.values()]
    delta_hard_mean = float(np.mean(dh))
    delta_hard_ci_low_mean = float(np.mean(dh_lo))
    amplification_mean = float(np.mean(amp))
    amplification_ci_low_mean = float(np.mean(amp_lo))
    n_same_dir = int(sum(1 for d in seed_deltas if d > 0.0))

    pass_delta = bool(delta_hard_mean >= DELTA_HARD_MIN and delta_hard_ci_low_mean > 0.0)
    pass_amp = bool(amplification_mean >= AMPLIFICATION_MIN and amplification_ci_low_mean > 0.0)
    instability = bool(pass_delta and pass_amp and n_same_dir <= 1)

    if not manipulation_ok:
        verdict = "NOT_ASSESSABLE"
    elif pass_delta and pass_amp:
        verdict = "HARD_SEMANTIC_REPLICATED"
    else:
        verdict = "NOT_REPLICATED"
    return {
        "protocol": "V2-G G4 item 9 (frozen before Phase A)",
        "comparison": f"{R1_NAME} -> {R2_NAME} on SameCat-K5 (matched random control)",
        "delta_hard_mean": delta_hard_mean,
        "delta_hard_ci_low_mean": delta_hard_ci_low_mean,
        "amplification_mean": amplification_mean,
        "amplification_ci_low_mean": amplification_ci_low_mean,
        "per_seed_delta_hard": {s: float(v["delta_hard"]) for s, v in per_seed_k5.items()},
        "per_seed_delta_hard_ci_low": {s: float(v["delta_hard_ci_low"]) for s, v in per_seed_k5.items()},
        "per_seed_amplification": {s: float(v["amplification"]) for s, v in per_seed_k5.items()},
        "per_seed_amplification_ci_low": {s: float(v["amplification_ci_low"]) for s, v in per_seed_k5.items()},
        "pass_delta_hard": pass_delta,
        "pass_amplification": pass_amp,
        "n_seed_same_direction": n_same_dir,
        "n_seeds": len(per_seed_k5),
        "instability": instability,
        "manipulation_ok": bool(manipulation_ok),
        "verdict": verdict,
        "thresholds": {"delta_hard_min": DELTA_HARD_MIN, "delta_hard_ci_low_gt": 0.0,
                       "amplification_min": AMPLIFICATION_MIN, "amplification_ci_low_gt": 0.0,
                       "requires_manipulation_valid": True},
        "seed_mean_delta_reference": float(np.mean(seed_deltas)) if seed_deltas else None,
        "seed_ci_low_reference": [float(x) for x in seed_lows],
    }


# ---------------------------------------------------------------------------
# per-backbone driver
# ---------------------------------------------------------------------------
def _run_backbone(tag: str, bb_dir: str, paths: Mapping[str, Any], cohort: shard.HardCohort,
                  args: argparse.Namespace, log: Callable[[str], None]) -> Dict[str, Any]:
    b3_seeds = [f"b3_seed{s}" for s in SEEDS]
    phase_a = {f"b3_seed{s}": _load_phase_a(tag, s) for s in SEEDS}

    device = str(args.device)
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    models = hscores.load_frozen_scorers(b3_root=paths["b3_root"], seeds=SEEDS, device=device)
    log(f"[B] {tag}: frozen V2 scorers loaded on {device}: {sorted(models)}")

    corpus = B3Corpus(
        paths["features_root"], MANIFESTS, REFS, BANK,
        image_sizes_path=IMAGE_SIZES, ks=(5, 10), regime="random", preload=True,
    )
    log(f"[B] {tag}: B3Corpus preloaded feature_dim={int(corpus.feature_dim)}")

    stop = _stop_check(cohort, corpus, models, paths, b3_seeds, args, log)
    cell_data = _score_stage(cohort, corpus, models, phase_a, b3_seeds, args, log, tag=tag)
    corpus.close()

    amp_rows, per_seed = _amplification_rows(cell_data, b3_seeds, tag, args)
    k5 = per_seed["K5"]
    manip_rows, manip = _manipulation_rows(cell_data, b3_seeds, tag, args)
    dose_rows, dose = _dose_rows(cell_data, b3_seeds, tag)
    point = _point_rows(cell_data, b3_seeds, tag)

    seed_lows = [float(v["delta_hard_ci_low"]) for v in k5.values()]
    seed_deltas = [float(v["delta_hard"]) for v in k5.values()]
    gate = _hard_gate(k5, manip["manipulation_ok"], seed_lows, seed_deltas)

    bb_out = B_OUT / tag
    bb_out.mkdir(parents=True, exist_ok=True)
    _write_csv(bb_out / "amplification.csv", amp_rows)
    _write_csv(bb_out / "manipulation_check.csv", manip_rows)
    _write_csv(bb_out / "dose_response.csv", dose_rows)
    _write_csv(bb_out / "reliability_metrics.csv", point)
    _write_json(bb_out / "gate.json", {**gate, "manipulation": manip, "dose": dose,
                                       "stop_checks_max_abs_delta": stop})
    log(f"[B] {tag}: K5 delta_hard={gate['delta_hard_mean']:+.4f} "
        f"amplification={gate['amplification_mean']:+.4f} "
        f"manip_ok={gate['manipulation_ok']} -> {gate['verdict']}")
    return {"tag": tag, "key": paths["key"], "gate": gate, "manipulation": manip,
            "dose": dose, "stop": stop, "amp_rows": amp_rows}


# ---------------------------------------------------------------------------
# G5 cross-backbone overall gate
# ---------------------------------------------------------------------------
#: tag -> backbone directory name (for the g3 key lookup).
BACKBONE_TAG: Dict[str, str] = {"b1": "openclip_b16", "b2": "siglip_b16"}


def _g5_overall(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    # B0 (V1 B/32) is frozen-replicated for both cardinality and hard-semantic.
    rows = [{"backbone": "B0_v1_b32", "cardinality_replicated": True,
             "hard_semantic_replicated": True, "source": "V1 (frozen)"}]
    for r in results:
        card = _cardinality_replicated(f"{r['key']}_{BACKBONE_TAG[r['tag']]}")
        hard = r["gate"]["verdict"] == "HARD_SEMANTIC_REPLICATED"
        rows.append({"backbone": f"{r['key']}_{BACKBONE_TAG[r['tag']]}",
                     "cardinality_replicated": card, "hard_semantic_replicated": bool(hard),
                     "hard_verdict": r["gate"]["verdict"], "manipulation_ok": r["gate"]["manipulation_ok"]})
    both = int(sum(1 for x in rows if x["cardinality_replicated"] and x["hard_semantic_replicated"]))
    total = len(rows)
    authorized = bool(both >= math.ceil(2 * total / 3))
    all_three_hard = all(x["hard_semantic_replicated"] for x in rows)
    return {
        "artifact": "v2g_g5_overall_gate",
        "cross_backbone": rows,
        "n_backbones": total,
        "n_both_replicated": both,
        "authorized_rule": ">=2/3 backbones with BOTH cardinality AND hard-semantic replicated",
        "V2_METHOD_DEVELOPMENT_AUTHORIZED": authorized,
        "STRONG_CROSS_BACKBONE_GENERALITY": bool(both == total and all_three_hard),
        "note": "B0 counts as the V1 B/32 frozen replication; B1/B2 come from this G4 Phase B.",
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), choices=list(SEEDS))
    p.add_argument("--bootstrap-replicates", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--stop-atol", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--log-file", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="fast end-to-end check: 1 seed, 100 bootstrap replicates, *_smoke out dir")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    global SEEDS, B_OUT
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.seeds = list(args.seeds[:1])
        args.bootstrap_replicates = min(int(args.bootstrap_replicates), 100)
        if B_OUT == V2_ROOT / "g4_phaseB":
            B_OUT = V2_ROOT / "g4_phaseB_smoke"
    SEEDS = tuple(int(s) for s in args.seeds)
    log = _make_logger(Path(args.log_file) if args.log_file else None)
    started = time.perf_counter()
    B_OUT.mkdir(parents=True, exist_ok=True)

    freeze = _load_freeze()
    log(f"[B] start (backbones={[t for t, _ in BACKBONES]}, seeds={list(SEEDS)}, "
        f"replicates={args.bootstrap_replicates}, out={B_OUT}, smoke={bool(args.smoke)})")

    cohort = shard.load_hard_cohort(features_dir=PHASE05_DIR, manifests_root=MANIFESTS, log=log)
    log(f"[B] cohort: {len(cohort)} pooled-test rows / "
        f"{int(np.unique(cohort.image_id).size)} images / {int(np.unique(cohort.ref_id).size)} refs")

    results = []
    for tag, bb_dir in tqdm(BACKBONES, desc="backbone", unit="bb", disable=None):
        paths = _backbone_paths(freeze, bb_dir)
        results.append(_run_backbone(tag, bb_dir, paths, cohort, args, log))

    g5 = _g5_overall(results)
    _write_json(B_OUT / "g5_overall_gate.json", g5)

    metadata = {
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.perf_counter() - started,
        "smoke": bool(args.smoke), "seeds": list(SEEDS),
        "bootstrap": {"replicates": int(args.bootstrap_replicates), "seed": int(args.bootstrap_seed),
                      "ci": float(args.ci)},
        "per_backbone": {r["tag"]: {"verdict": r["gate"]["verdict"],
                                    "delta_hard_mean": r["gate"]["delta_hard_mean"],
                                    "amplification_mean": r["gate"]["amplification_mean"],
                                    "manipulation_ok": r["gate"]["manipulation_ok"],
                                    "dose_replicated": r["dose"]["replicated"]}
                         for r in results},
        "authorized": g5["V2_METHOD_DEVELOPMENT_AUTHORIZED"],
    }
    _write_json(B_OUT / "metadata.json", metadata)
    log(f"[B] done in {metadata['runtime_seconds']:.1f}s")
    print(f"V2G_PHASEB_COMPLETE authorized={g5['V2_METHOD_DEVELOPMENT_AUTHORIZED']} "
          f"verdicts={[r['gate']['verdict'] for r in results]} out={B_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())