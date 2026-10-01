"""V2-P1 replication core - shared frozen-stack scoring + gate primitives for P1-F4 (C1)
and P1-F5 (C4).

Route F (P1-F3) proved the *entire* frozen V1 B0 reliability stack runs on DETR
proposals without adaptation.  P1-F4 / P1-F5 now replay the two frozen V1 primary
analyses - C1 candidate-cardinality robustness and C4 hard-semantic amplification -
on the DETR proposal family, with **zero** new training parameters.  This module owns
everything the two stage-drivers share so neither re-implements the frozen pipeline:

* corpus construction for a ``family`` / ``regime`` / nested ``ks`` (the DETR swap is
  only the proposal *bank*; the candidate-construction regime is the frozen V1
  seeded-random / same-category one - see the P1-A0 clarification);
* nested-K cohort eligibility + common-K50 selection + attrition (monotone ``C5`` ⊂
  ``C10`` ⊂ ``C20`` ⊂ ``C50`` on *identical* rows);
* chunked materialisation + scoring of the frozen stack (B3 seed1/2/3 + per-seed
  temperature -> conf_msp; frozen R1 stats_logistic / E1b e1b_stats_semantic ->
  conf_stats / conf_e1b; V2 14-d backbone-neutral semantic stats -> manipulation);
* per-``(family, job, scorer)`` prediction npz write/read aligned by ``sentence_id``
  (mirrors ``run_d2_c1_c4.load_pred``);
* the frozen V2-G G3 / G4 gate logic and the image-cluster paired-bootstrap wrappers;
* read-only frozen RPN (V1) reference loaders for the proposal-family table.

Nothing here fits a model.  Every entry point is used inside
``ccg.semantic.frozen_load.fit_is_forbidden()`` by the drivers.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from ccg.experiment import phase0a, phase0b
from ccg.experiment.phase0a import SampleStats
from ccg.features import cache as fcache
from ccg.external import frozen_identity as fi
from ccg.models.b3_data import B3Corpus
from ccg.data.manifests import ManifestFile
from ccg.semantic.hard import FrozenSample
from ccg.reliability import evaluate as reval
from ccg.reliability import features as rfeat
from ccg.semantic import features as sfeat
from ccg.semantic import hard_scores as hscores
from ccg.semantic import frozen_load as fl
from ccg.v2 import semantic_features as v2feat

# ---------------------------------------------------------------------------
# frozen inputs (V1 B0) - reused verbatim from scripts/p1_f3_route_f_compatibility.py
# ---------------------------------------------------------------------------
B3_ROOT = Path("results/phase0b_independent")
FROZEN_MODELS = Path("results/phase1e_refcocog_external/frozen_models")
FROZEN_ARTIFACT_MANIFEST = FROZEN_MODELS / "frozen_artifact_manifest.json"
REFS_PATH = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
IMAGE_SIZES_PATH = Path("cache/image_sizes.npz")
MANIFEST_SEED = 20260927

V1_PHASE0B_BOOTSTRAP = str(B3_ROOT) + "/seed_{seed}/bootstrap.csv"
V1_PHASE0B_AGGREGATE = B3_ROOT / "aggregate.csv"
V1_PHASE1F_GATE_JSON = Path("results/phase1f_hard_semantic/gate.json")

FAMILIES: Dict[str, Dict[str, Any]] = {
    "RPN": {
        "features_root": Path("cache/features"),
        "manifests_root": Path("cache/manifests"),
        "bank": Path("cache/proposals.h5"),
    },
    "DETR": {
        "features_root": Path("cache/features_detr_r50"),
        "manifests_root": Path("cache/manifests_detr"),
        "bank": Path("cache/proposals_detr_r50.h5"),
    },
}

B3_SEEDS: Tuple[int, ...] = (1, 2, 3)
SCORERS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
KS: Tuple[int, ...] = (5, 10, 20, 50)
K_BASELINE = 5
PRIMARY_KB = 50
POOLED_TEST: Tuple[str, ...] = ("testA", "testB")
SEM_CHUNK = 4096

STATS_NAME = "stats_logistic"       # R1
E1B_NAME = "e1b_stats_semantic"     # E1b / R2
POOLED = "__pooled__"

# ---------------------------------------------------------------------------
# shared bootstrap config (identical to phase0b / phase1f / V2-G / D2)
# ---------------------------------------------------------------------------
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI = 0.95

# V2-G G3 C1 cardinality gate thresholds (analyze_v2g_cardinality, frozen).
AUC_THRESHOLD = 0.03
EAURC_WORSEN_THRESHOLD = 0.20
RER50_DROP_THRESHOLD = 0.10
# V2-G G4 C4 hard gate thresholds (run_v2g_hard, freeze item 9).
DELTA_HARD_MIN = 0.015
AMPLIFICATION_MIN = 0.01
MANIPULATION: Tuple[Tuple[str, str], ...] = tuple(v2feat.MANIPULATION_METRICS)
V2_SEM_INDEX = {name: pos for pos, name in enumerate(v2feat.V2_PRIMARY_SEMANTIC_NAMES)}

#: stat columns used for the calibration-independent secondary diagnostic (section 7).
RAW_TOP1_COL = 1
RAW_MARGIN12_COL = 4

#: prediction npz columns shared by every (family, job, scorer).
PRED_COLUMNS: Tuple[str, ...] = (
    "sentence_id", "ref_id", "image_id", "correct",
    "conf_msp", "conf_stats", "conf_e1b", "sem14",
    "raw_top1", "raw_margin12",
)


# ===========================================================================
# small pure helpers
# ===========================================================================
def _mean(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values], dtype=np.float64).reshape(-1)
    return float(arr.mean()) if arr.size else float("nan")


def _std(values: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in values], dtype=np.float64).reshape(-1)
    return float(arr.std(ddof=0)) if arr.size else float("nan")


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    return _mean(values), _std(values)


def _sign_excludes_zero(ci_low: float, ci_high: float) -> Tuple[bool, str]:
    if ci_low > 0:
        return True, "positive"
    if ci_high < 0:
        return True, "negative"
    return False, ("positive" if ci_low + ci_high > 0 else "negative")


# ===========================================================================
# corpus + nested-K cohort construction
# ===========================================================================
def build_corpus(family: str, regime: str, ks: Sequence[int] = KS) -> B3Corpus:
    cfg = FAMILIES[family]
    return B3Corpus(
        cfg["features_root"],
        cfg["manifests_root"],
        REFS_PATH,
        cfg["bank"],
        image_sizes_path=IMAGE_SIZES_PATH,
        ks=tuple(int(k) for k in ks),
        regime=regime,
        preload=True,
    )


def is_eligible(record: Any, k: int) -> bool:
    """Nested-K eligibility for a Phase-0A ``SentenceRecord``.

    Uses the frozen ``eligible`` map when present; falls back to the equivalent
    ``target present AND distractor_order >= k - 1`` rule (``C_K`` prefix).
    """
    k = int(k)
    el = getattr(record, "eligible", None)
    if el is not None and k in el:
        return bool(el[k])
    if getattr(record, "target_index", None) is None:
        return False
    return int(np.asarray(record.distractor_order).size) >= k - 1


def eligible_ids(records: Sequence[Any], k: int) -> Set[int]:
    return {int(r.sentence_id) for r in records if is_eligible(r, k)}


def cohort_records(corpus: B3Corpus, k: int) -> List[Any]:
    """Nested-K cohort: POOLED-TEST records eligible at ``k`` (target + >= k-1)."""
    records = corpus.eval_records(POOLED_TEST, ks=(5, 10, 20, 50))
    return [r for r in records if is_eligible(r, k)]


def nested_cohort_ids(records: Sequence[Any], k: int) -> Set[int]:
    return eligible_ids(records, k)


def attrition_report(corpus: B3Corpus, regime: str) -> Dict[str, Any]:
    """Nested-random cohort attrition over K (protocol section 3).

    Reports total target-present rows, per-K eligibility, the primary common-K50
    cohort size, and the distinct images / refs of that cohort.  No resampling.
    Random regime only; the same-category cohort uses :func:`build_hard_samples`.
    """
    if regime != "random":
        raise ValueError("attrition_report is random-regime only; use build_hard_samples")
    records = corpus.eval_records(POOLED_TEST, ks=(5, 10, 20, 50))
    target_present = [r for r in records if getattr(r, "target_index", None) is not None]
    per_k = {int(k): sum(1 for r in records if is_eligible(r, k)) for k in KS}
    common = [r for r in records if is_eligible(r, PRIMARY_KB)]
    return {
        "regime": regime,
        "total_target_present_rows": int(len(target_present)),
        "eligible": {str(k): int(per_k[k]) for k in KS},
        "primary_common_K50_rows": int(len(common)),
        "images": int(len({int(r.image_id) for r in common})),
        "refs": int(len({int(r.ref_id) for r in common})),
        "k50_availability": round(
            per_k[PRIMARY_KB] / max(1, len(target_present)), 4
        ),
    }


# ===========================================================================
# same-category (hard) cohort built family-agnostically from the manifests
# ===========================================================================
HARD_K = 5
HARD_SAME_CUTOFF = 4  # A8.2 ``same4``: target + >= 4 same-category distractors


def _resolve_manifest_path(manifests_root: Path, regime: str, split: str) -> Path:
    """Resolve ``{regime}_{split}.jsonl`` under either manifest layout (flat or ``manifests/``)."""
    root = Path(manifests_root)
    for base in (root, root / "manifests"):
        p = base / f"{regime}_{split}.jsonl"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{regime}_{split}.jsonl not found under {root} or {root / 'manifests'}"
    )


def load_ref_entries(family: str, regime: str) -> Dict[int, Any]:
    """``{ref_id: ManifestEntry}`` for a family's pooled-test regime manifests."""
    root = FAMILIES[family]["manifests_root"]
    table: Dict[int, Any] = {}
    for split in POOLED_TEST:
        manifest = ManifestFile.load(_resolve_manifest_path(root, regime, split))
        for entry in manifest.entries:
            table[int(entry.ref_id)] = entry
    return table


def build_hard_samples(family: str) -> Dict[str, Any]:
    """Construct the matched SameCategory-K5 / Random-K5 cohort for one family.

    The expression universe (``sentence_id`` / ``ref_id`` / ``image_id``) comes from
    the random-regime records; the hard distractor ordering + ``n_same_category_available``
    come from the same-category manifest joined on ``ref_id``.  A row is hard-K5
    eligible when the target is present, the same-category ordering carries >= 4
    distractors, at least 4 same-category proposals are supplied, and the random /
    hard ``target_index`` agree (the matched-control property).  The paired random
    control is the *same* expression's random-K5 (identical target / image / K) -
    distractor composition is the only difference.
    """
    corpus = build_corpus(family, "random", ks=(5, 10, 20, 50))
    try:
        rand_records = [
            r for r in corpus.eval_records(POOLED_TEST, ks=(5, 10, 20, 50))
            if getattr(r, "target_index", None) is not None
            and int(np.asarray(r.distractor_order).size) >= HARD_K - 1
        ]
    finally:
        corpus.close()
    hard_entries = load_ref_entries(family, "same_category")

    hard_samples: List[FrozenSample] = []
    matched_random: List[Any] = []
    excluded_no_entry = excluded_not_eligible = target_mismatch = 0
    for r in rand_records:
        he = hard_entries.get(int(r.ref_id))
        if he is None:
            excluded_no_entry += 1
            continue
        if he.target_index is None or int(np.asarray(he.distractor_order).size) < HARD_K - 1:
            excluded_not_eligible += 1
            continue
        if int(he.target_index) != int(r.target_index):
            target_mismatch += 1
            continue
        if int(he.n_same_category_available) < HARD_SAME_CUTOFF:
            excluded_not_eligible += 1
            continue
        if not bool(he.eligible.get(HARD_K, True)):
            excluded_not_eligible += 1
            continue
        hard_samples.append(
            FrozenSample(
                sentence_id=int(r.sentence_id),
                ref_id=int(r.ref_id),
                image_id=int(r.image_id),
                target_index=int(he.target_index),
                distractor_order=np.asarray(he.distractor_order, dtype=np.int64),
            )
        )
        matched_random.append(r)
    attrition = {
        "regime": "same_category",
        "eligible_rows": int(len(hard_samples)),
        "images": int(len({int(s.image_id) for s in hard_samples})),
        "refs": int(len({int(s.ref_id) for s in hard_samples})),
        "excluded_no_entry": int(excluded_no_entry),
        "excluded_not_eligible": int(excluded_not_eligible),
        "excluded_target_mismatch": int(target_mismatch),
        "same_cat_ge_4_availability": round(
            len(hard_samples) / max(1, len(rand_records)), 4
        ),
    }
    return {
        "hard_samples": hard_samples,
        "matched_random_records": matched_random,
        "matched_ids": {int(s.sentence_id) for s in hard_samples},
        "attrition": attrition,
    }


# ===========================================================================
# frozen-stack scoring (chunked materialisation -> per-seed aligned arrays)
# ===========================================================================
def _semantic_stats_chunked(z_q: np.ndarray, z_i: np.ndarray, scores: np.ndarray) -> np.ndarray:
    n = int(z_q.shape[0])
    out = np.empty((n, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, n, SEM_CHUNK):
        stop = min(start + SEM_CHUNK, n)
        out[start:stop] = sfeat.semantic_stats(z_q[start:stop], z_i[start:stop], scores[start:stop])
    return out


def _v2_sem_chunked(z_q: np.ndarray, z_i: np.ndarray, scores: np.ndarray) -> np.ndarray:
    n = int(z_q.shape[0])
    out = np.empty((n, len(v2feat.V2_PRIMARY_SEMANTIC_NAMES)), dtype=np.float64)
    for start in range(0, n, SEM_CHUNK):
        stop = min(start + SEM_CHUNK, n)
        out[start:stop] = v2feat.v2_primary_semantic_stats(
            z_q[start:stop], z_i[start:stop], scores[start:stop]
        )
    return out


def score_cohort(
    family: str,
    regime: str,
    k: int,
    records: Sequence[Any],
    *,
    models: Mapping[str, Any],
    bundle: fl.FrozenExternalModels,
    batch_size: int = 64,
    row_chunk: int = 1024,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Score ``records`` at nested-K ``k`` through the frozen stack for every seed.

    Returns ``{scorer: column_dict}`` where each column dict is aligned to
    ``records`` order and carries the shared :data:`PRED_COLUMNS`.  Materialisation
    is chunked to bound peak memory for ``K = 50`` (the region block is the big one).
    """
    n = len(records)
    if n == 0:
        raise RuntimeError(f"{family}/{regime}/K{k}: empty cohort")
    per_seed: Dict[str, Dict[str, np.ndarray]] = {}
    for scorer in SCORERS:
        per_seed[scorer] = {
            "sentence_id": np.empty(n, dtype=np.int64),
            "ref_id": np.empty(n, dtype=np.int64),
            "image_id": np.empty(n, dtype=np.int64),
            "correct": np.empty(n, dtype=np.float64),
            "conf_msp": np.empty(n, dtype=np.float64),
            "conf_stats": np.empty(n, dtype=np.float64),
            "conf_e1b": np.empty(n, dtype=np.float64),
            "raw_top1": np.empty(n, dtype=np.float64),
            "raw_margin12": np.empty(n, dtype=np.float64),
            "sem14": np.empty((n, len(v2feat.V2_PRIMARY_SEMANTIC_NAMES)), dtype=np.float64),
        }

    corpus = build_corpus(family, regime, ks=(5, 10, 20, 50))
    text_cache: Dict[int, np.ndarray] = {}
    try:
        for start in range(0, n, row_chunk):
            stop = min(start + row_chunk, n)
            block = records[start:stop]
            batch = hscores.materialise_examples(corpus, block, k, text_cache=text_cache)
            for i, rec in enumerate(block):
                gi = start + i
                for scorer in SCORERS:
                    per_seed[scorer]["sentence_id"][gi] = int(rec.sentence_id)
                    per_seed[scorer]["ref_id"][gi] = int(rec.ref_id)
                    per_seed[scorer]["image_id"][gi] = int(rec.image_id)
            for scorer in SCORERS:
                temperature = float(bundle.seed(scorer).temperature)
                scores = hscores.score_examples(models[scorer], batch, batch_size=batch_size)
                scores64 = np.asarray(scores, dtype=np.float64)
                correct = (np.argmax(scores, axis=1) == 0).astype(np.float64)
                scalars = rfeat.scalar_confidence(scores64, temperature=temperature)
                stats17 = np.asarray(
                    rfeat.stat_features(scores64, temperature=temperature), dtype=np.float64
                )
                sem16 = _semantic_stats_chunked(batch.z_q, batch.z_i, scores64)
                r1_conf, e1b_conf = bundle.predict(scorer, stats17, sem16)
                sem14 = _v2_sem_chunked(batch.z_q, batch.z_i, scores64)
                d = per_seed[scorer]
                sl = slice(start, stop)
                d["correct"][sl] = correct
                d["conf_msp"][sl] = np.asarray(scalars["msp"], dtype=np.float64)
                d["conf_stats"][sl] = np.asarray(r1_conf, dtype=np.float64)
                d["conf_e1b"][sl] = np.asarray(e1b_conf, dtype=np.float64)
                d["raw_top1"][sl] = stats17[:, RAW_TOP1_COL]
                d["raw_margin12"][sl] = stats17[:, RAW_MARGIN12_COL]
                d["sem14"][sl] = sem14
    finally:
        corpus.close()
    return per_seed


def job_name(regime: str, k: int) -> str:
    return "hard_k5" if regime == "same_category" else f"random_k{k}"


def pred_path(pred_dir: Path, family: str, job: str, scorer: str) -> Path:
    return Path(pred_dir) / f"p1__{family}__{job}__{scorer}.npz"


def write_predictions(
    pred_dir: Path, family: str, job: str, per_seed: Mapping[str, Mapping[str, np.ndarray]]
) -> List[Path]:
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for scorer, cols in per_seed.items():
        path = pred_path(pred_dir, family, job, scorer)
        payload = {c: np.asarray(cols[c]) for c in PRED_COLUMNS}
        np.savez(path, **payload)
        written.append(path)
    return written


def load_pred(
    pred_dir: Path, family: str, job: str, scorer: str, keep_ids: Set[int]
) -> Dict[str, np.ndarray]:
    """Load one prediction npz, keep ``keep_ids`` rows, sort by ``sentence_id``.

    Sorting by ``sentence_id`` makes any two jobs restricted to the same id set
    align row-for-row, so the drivers can assert matchedness (the D2 contract).
    """
    raw = np.load(pred_path(pred_dir, family, job, scorer), allow_pickle=False)
    sid = np.asarray(raw["sentence_id"], dtype=np.int64)
    keep = np.fromiter(sorted(int(i) for i in keep_ids), dtype=np.int64)
    mask = np.isin(sid, keep)
    order = np.argsort(sid[mask], kind="stable")
    idx = np.nonzero(mask)[0][order]
    out: Dict[str, np.ndarray] = {}
    for key in PRED_COLUMNS:
        out[key] = np.asarray(raw[key])[idx]
    out["correct"] = out["correct"].astype(np.float64)
    for key in ("conf_msp", "conf_stats", "conf_e1b", "sem14", "raw_top1", "raw_margin12"):
        out[key] = np.asarray(out[key], dtype=np.float64)
    return out


# ===========================================================================
# frozen RPN scoring reproduction check (A8.4 STOP guarantee)
# ===========================================================================
def load_frozen_raw_scores(seed: int, k: int, want_splits: Sequence[str]) -> Dict[int, np.ndarray]:
    """``sentence_id -> raw [K] logits`` from the frozen RPN Phase-0B scores."""
    path = B3_ROOT / f"seed_{seed}" / "raw_scores" / f"K{k}.npz"
    with np.load(path) as z:
        scores = np.asarray(z["scores"], dtype=np.float32)
        sid = np.asarray(z["sentence_id"], dtype=np.int64).reshape(-1)
        split = np.asarray([str(s) for s in z["eval_split"]])
    want = set(want_splits)
    mapping: Dict[int, np.ndarray] = {}
    for i in range(sid.shape[0]):
        if split[i] in want:
            mapping[int(sid[i])] = scores[i]
    return mapping


def verify_rpn_reproduction(
    family: str,
    regime: str,
    k: int,
    records: Sequence[Any],
    models: Mapping[str, Any],
    bundle: fl.FrozenExternalModels,
    *,
    atol: float = 1e-4,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Re-score RPN at K=k and confirm the frozen checkpoints reproduce raw_scores.

    Runs on the intersection with the frozen Phase-0B ``raw_scores`` (the shared
    nested-K rows) - the same guarantee P1-F3 asserts, now per nested K.
    """
    per_seed = score_cohort(
        family, regime, k, records, models=models, bundle=bundle, batch_size=batch_size
    )
    deltas: Dict[str, float] = {}
    n_common_total = 0
    sids = per_seed[SCORERS[0]]["sentence_id"].astype(int)
    sid_to_pos = {int(s): i for i, s in enumerate(sids)}
    verify_corpus = build_corpus(family, regime)
    try:
        for seed in B3_SEEDS:
            scorer = f"b3_seed{seed}"
            frozen = load_frozen_raw_scores(seed, k, POOLED_TEST)
            common_ids = [s for s in sids if s in frozen]
            if not common_ids:
                raise KeyError(f"RPN verify K{k} seed_{seed}: empty intersection with frozen raw_scores")
            positions = np.array([sid_to_pos[int(s)] for s in common_ids], dtype=int)
            sub_records = [records[int(p)] for p in positions]
            batch = hscores.materialise_examples(verify_corpus, sub_records, k)
            recomputed = hscores.score_examples(models[scorer], batch, batch_size=batch_size)
            reference = np.stack([frozen[int(s)] for s in common_ids], axis=0).astype(np.float32)
            delta = hscores.verify_against_raw_scores(
                {scorer: recomputed}, {scorer: reference}, k=k, atol=atol
            )
            deltas[scorer] = delta[scorer]
            n_common_total = int(len(common_ids))
    finally:
        verify_corpus.close()
    return {
        "performed": True,
        "K": int(k),
        "max_abs_delta": deltas,
        "atol": atol,
        "n_cohort": int(len(records)),
        "n_intersection": n_common_total,
        "coverage": round(n_common_total / max(1, len(records)), 4),
    }


# ===========================================================================
# C1 - cardinality robustness gate (V2-G G3, applied verbatim)
# ===========================================================================
C1_METRIC_SPECS: Tuple[Tuple[str, str], ...] = (
    ("auroc_correct", "absolute"),
    ("e_aurc", "absolute"),
    ("e_aurc", "relative"),
    ("rer_at_50", "absolute"),
    ("rer_at_80", "absolute"),
)


def c1_point_row(
    conf: np.ndarray, correct: np.ndarray, image_id: np.ndarray, k: int, seed: str
) -> Dict[str, Any]:
    point = reval.point_metric_row(conf, correct, probability=conf)
    return {
        "model": "global_T_corrected", "seed": seed, "K": int(k),
        "n": int(conf.size), "n_images": int(np.unique(image_id).size),
        "accuracy": float(np.mean(correct)),
        "auroc_correct": float(point["auroc_correct"]),
        "e_aurc": float(point["e_aurc"]),
        "rer_at_50": float(point["rer_at_50"]),
        "rer_at_80": float(point["rer_at_80"]),
    }


def _c1_bootstrap(fn, stats_a, stats_b, clusters, kind, metric):
    if kind == "relative":
        return phase0b.paired_cluster_bootstrap_ratio(
            fn, stats_a, stats_b, clusters,
            n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            ci=BOOTSTRAP_CI, metric_name=metric)
    return phase0a.paired_cluster_bootstrap(
        fn, stats_a, stats_b, clusters,
        n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
        ci=BOOTSTRAP_CI, metric_name=metric)


def _worsening_from_rel(mean_a: float, mean_b: float, ci_low: float, ci_high: float) -> Tuple[float, float, float]:
    """Convert the (a-b)/b ratio (baseline K_b) to K5-baselined worsening + CI.

    Identical to the frozen V2-G / D2 mapping (``w = -r / (1 + r)``, monotone, CI
    endpoints mapped through the same transform then re-sorted).
    """
    r = (mean_a - mean_b) / mean_b
    t = lambda x: -x / (1.0 + x)  # noqa: E731
    w = t(r)
    lo, hi = t(ci_low), t(ci_high)
    return w, min(lo, hi), max(lo, hi)


def c1_gate(stage: Mapping[Tuple[str, int], Mapping[str, Mapping[str, float]]]) -> Dict[str, Any]:
    """Re-apply the V2-G G3 cardinality gate to per-seed K5->K50 bootstrap rows."""
    per_seed_auc = [stage[("auroc_correct|absolute", PRIMARY_KB)][f"seed{s}"] for s in B3_SEEDS]
    per_seed_eaurc = [stage[("e_aurc|relative", PRIMARY_KB)][f"seed{s}"] for s in B3_SEEDS]
    per_seed_rer = [stage[("rer_at_50|absolute", PRIMARY_KB)][f"seed{s}"] for s in B3_SEEDS]
    per_seed_rer80_key = ("rer_at_80|absolute", PRIMARY_KB)
    per_seed_rer80 = (
        [stage[per_seed_rer80_key][f"seed{s}"] for s in B3_SEEDS]
        if per_seed_rer80_key in stage else None
    )

    auc_effects = [v["diff"] for v in per_seed_auc]
    auc_all_excl = all(
        _sign_excludes_zero(v["ci_low"], v["ci_high"])[0] and v["ci_low"] > 0
        for v in per_seed_auc
    )
    route_a = bool(_mean(auc_effects) >= AUC_THRESHOLD and auc_all_excl)

    eaurc_wors = [
        _worsening_from_rel(v["mean_a"], v["mean_b"], v["ci_low"], v["ci_high"])[0]
        for v in per_seed_eaurc
    ]
    eaurc_all_excl = all(_sign_excludes_zero(v["ci_low"], v["ci_high"])[0] for v in per_seed_eaurc)
    rer_effects = [v["diff"] for v in per_seed_rer]
    rer_all_excl = all(
        _sign_excludes_zero(v["ci_low"], v["ci_high"])[0] and v["ci_low"] > 0
        for v in per_seed_rer
    )
    route_b = bool(
        _mean(eaurc_wors) >= EAURC_WORSEN_THRESHOLD
        and _mean(rer_effects) >= RER50_DROP_THRESHOLD
        and eaurc_all_excl and rer_all_excl
    )
    return {
        "primary_comparison": f"K{K_BASELINE}_vs_K{PRIMARY_KB}",
        "variant": "global_T_corrected",
        "auc_drop_mean": _mean(auc_effects),
        "auc_drop_ci_low_mean": _mean([v["ci_low"] for v in per_seed_auc]),
        "auc_all_seeds_ci_exclude_0": bool(auc_all_excl),
        "eaurc_worsening_mean": _mean(eaurc_wors),
        "eaurc_all_seeds_ci_exclude_0": bool(eaurc_all_excl),
        "rer50_drop_mean": _mean(rer_effects),
        "rer80_drop_mean": (_mean([v["diff"] for v in per_seed_rer80])
                            if per_seed_rer80 is not None else float("nan")),
        "rer50_all_seeds_ci_exclude_0": bool(rer_all_excl),
        "route_a_passed": route_a,
        "route_b_passed": route_b,
        "replicated": bool(route_a or route_b),
        "thresholds": {
            "route_a_auc_drop_min": AUC_THRESHOLD,
            "route_b_eaurc_worsen_min": EAURC_WORSEN_THRESHOLD,
            "route_b_rer50_drop_min": RER50_DROP_THRESHOLD,
        },
    }


# ===========================================================================
# C4 - hard semantic amplification gate (V2-G G4 item 9)
# ===========================================================================
def manipulation_valid_seed(dir_ok: int, ci_excl: int) -> bool:
    return bool(dir_ok >= 2 and ci_excl >= 1)


def manipulation_aggregate_ok(n_valid: int, n_seeds: int) -> bool:
    return bool(n_valid >= math.ceil(2 * n_seeds / 3))


def c4_hard_gate(per_seed: Mapping[str, Mapping[str, float]], manipulation_ok: bool) -> Dict[str, Any]:
    """V2-G G4 (run_v2g_hard._hard_gate) replayed on a proposal-family cohort."""
    dh = [float(v["delta_hard"]) for v in per_seed.values()]
    dh_lo = [float(v["delta_hard_ci_low"]) for v in per_seed.values()]
    amp = [float(v["amplification"]) for v in per_seed.values()]
    amp_lo = [float(v["amplification_ci_low"]) for v in per_seed.values()]
    dr = [float(v["delta_rand"]) for v in per_seed.values()]
    delta_hard_mean = _mean(dh)
    delta_hard_ci_low_mean = _mean(dh_lo)
    amplification_mean = _mean(amp)
    amplification_ci_low_mean = _mean(amp_lo)
    n_same_dir = int(sum(1 for d in dh if d > 0.0))

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
        "protocol": "V2-G G4 item 9 (frozen), replayed on the proposal family",
        "comparison": f"{STATS_NAME} -> {E1B_NAME} on SameCat-K5 (matched random control)",
        "delta_hard_mean": delta_hard_mean,
        "delta_hard_ci_low_mean": delta_hard_ci_low_mean,
        "delta_rand_mean": _mean(dr),
        "amplification_mean": amplification_mean,
        "amplification_ci_low_mean": amplification_ci_low_mean,
        "pass_delta_hard": pass_delta,
        "pass_amplification": pass_amp,
        "n_seed_same_direction": n_same_dir,
        "n_seeds": len(per_seed),
        "instability": instability,
        "manipulation_ok": bool(manipulation_ok),
        "verdict": verdict,
        "thresholds": {
            "delta_hard_min": DELTA_HARD_MIN,
            "amplification_min": AMPLIFICATION_MIN,
            "requires_manipulation_valid": True,
        },
    }


def c4_verdict_label(gate: Mapping[str, Any]) -> str:
    return {
        "HARD_SEMANTIC_REPLICATED": "YES",
        "NOT_REPLICATED": "NO",
        "NOT_ASSESSABLE": "NOT_ASSESSABLE",
    }[gate["verdict"]]


# ===========================================================================
# overall P1 verdict (protocol section 22)
# ===========================================================================
def overall_p1_verdict(c1_replicated: bool, c4_verdict: str) -> Dict[str, Any]:
    """C4 verdict is ``YES`` / ``NO`` / ``NOT_ASSESSABLE`` (P1-F5 gate label)."""
    c4_yes = c4_verdict == "YES"
    if c1_replicated and c4_yes:
        return {
            "label": "CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST",
            "value": "YES",
            "detail": "candidate-cardinality degradation and hard-semantic amplification "
                      "both replicate under DETR proposals.",
        }
    if c1_replicated and c4_verdict == "NOT_ASSESSABLE":
        return {
            "label": "CARDINALITY_PROPOSAL_ROBUST__HARD_AMPLIFICATION_NOT_ASSESSABLE",
            "value": "PARTIAL",
            "detail": "C1 replicates; C4 is not assessable because the manipulation "
                      "check failed (never reported as FAIL).",
        }
    if c1_replicated or c4_yes:
        return {
            "label": "PARTIAL_PROPOSAL_ROBUSTNESS",
            "value": "PARTIAL",
            "detail": "exactly one of C1 / C4 replicates under DETR proposals.",
        }
    if c4_verdict == "NOT_ASSESSABLE":
        return {
            "label": "PARTIAL_PROPOSAL_ROBUSTNESS",
            "value": "PARTIAL",
            "detail": "C1 not replicated and C4 not assessable.",
        }
    return {
        "label": "PROPOSAL_FAMILY_SENSITIVITY_WARNING",
        "value": "NO",
        "detail": "both C1 and C4 fail under a valid manipulation - the core findings "
                  "are proposal-family sensitive.",
    }


# ===========================================================================
# matched proposal-family intersection (protocol sections 8 / 18)
# ===========================================================================
def matched_expression_ids(
    family_a_ids: Set[int], family_b_ids: Set[int]
) -> Set[int]:
    """Deterministic intersection of two families' eligible expression id sets."""
    return {int(i) for i in family_a_ids if int(i) in family_b_ids}


# ===========================================================================
# frozen RPN (V1) reference loaders - read-only, for the proposal-family table
# ===========================================================================
def load_rpn_c1_reference() -> Dict[str, Any]:
    """The frozen phase0b K5->K50 ``global_T_corrected`` deltas (3-seed means)."""
    keys = {"auroc_correct|absolute", "e_aurc|relative", "rer_at_50|absolute", "rer_at_80|absolute"}
    per_key: Dict[str, List[float]] = {k: [] for k in keys}
    per_key_ci: Dict[str, List[float]] = {k: [] for k in keys}
    point_by_k: Dict[str, Dict[str, List[float]]] = {}
    for seed in B3_SEEDS:
        path = _REPO_PATH(V1_PHASE0B_BOOTSTRAP.format(seed=seed))
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row["eval_split"] == POOLED and row["variant"] == "global_T_corrected"
                        and int(row["K_a"]) == K_BASELINE and int(row["K_b"]) == PRIMARY_KB):
                    key = f"{row['metric']}|{row['diff_kind']}"
                    if key in per_key:
                        per_key[key].append(float(row["diff"]))
                        per_key_ci[key].append(float(row["ci_low"]))
    out: Dict[str, Any] = {
        key: {"diff_mean": _mean(vals), "ci_low_mean": _mean(per_key_ci[key])}
        for key, vals in per_key.items() if vals
    }
    # point metrics per K from aggregate.csv (pooled global_T_corrected).
    agg = _load_phase0b_aggregate_pooled()
    out["point_by_k"] = agg
    return out


def _REPO_PATH(rel: Any) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(__file__).resolve().parents[1] / p


def _load_phase0b_aggregate_pooled() -> Dict[str, Dict[str, float]]:
    """``{K: {metric: mean}}`` for pooled ``global_T_corrected`` rows (read-only)."""
    out: Dict[str, Dict[str, float]] = {}
    path = V1_PHASE0B_AGGREGATE
    if not path.exists():
        return out
    want = {"accuracy", "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["eval_split"] == POOLED and row["variant"] == "global_T_corrected"
                    and row["metric"] in want):
                k = str(int(row["K"]))
                out.setdefault(k, {})[row["metric"]] = float(row["mean"])
    return out


def load_rpn_c4_reference() -> Dict[str, Any]:
    """The frozen phase1f V1 gate: delta-hard, diff-of-diffs, verdict, cohort."""
    if not V1_PHASE1F_GATE_JSON.exists():
        return {}
    gate = json.loads(V1_PHASE1F_GATE_JSON.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {"verdict": gate.get("verdict", {}).get("verdict")}
    dod = gate.get("diff_of_diffs_auroc", {})
    out["amplification"] = {"diff": dod.get("diff"), "ci_low": dod.get("ci_low"),
                            "ci_high": dod.get("ci_high")}
    cell = gate.get("cell_samecat_k5", {})
    out["delta_hard"] = cell.get("delta_auroc")
    out["e_aurc_reduction"] = cell.get("e_aurc_reduction")
    out["rer50_gain_pp"] = cell.get("rer50_gain_pp")
    # delta_rand = delta_hard - amplification (both from the frozen gate).
    try:
        out["delta_rand"] = float(out["delta_hard"]) - float(out["amplification"]["diff"])
    except (TypeError, ValueError):
        out["delta_rand"] = None
    return out
