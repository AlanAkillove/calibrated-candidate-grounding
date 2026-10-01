#!/usr/bin/env python
"""V2-M3.1 cross-backbone confirmatory evaluation (B1 OpenCLIP B/16, B2 SigLIP B/16).

Frozen before the run (amendment V2-M3.1,
``results/v2_local_competition/m3_mixture/amendment_v2m31.json``): the B0
developmental result is already known, the B1/B2 mixture results are unseen, and
every formula (competition index, adaptive alpha, static control, expert
definitions, loss), every metric and every gate threshold is frozen and
identical to the B0 implementation -- no B0-driven redesign.

Per backbone (own ``reliability_train`` CDF, own ``reliability_tune`` mixer):

    E_random      = frozen V2-G Phase A R2 (random-only Stats+Semantic logistic)
    E_curriculum  = fresh E1b trained with the identical M2 curriculum protocol
                    (m={0,2,4} balanced, C grid 0.1/1/10, tie 0.002, same split)
    EqualMix / StaticMix / AdaptiveMix = ``ccg.mixture`` frozen formulas

    frozen K=10 severity cells m in {0,2,4,8} (same8 cohort, identical rows for
    all five models) -> point metrics -> paired image-cluster bootstrap
    (5000 reps, seed 0) -> macro bootstrap with shared draws -> frozen
    confirmatory gates:

        primary   : delta_macro(Adaptive - Static) >= 0.003 and CI_low > 0
        safety    : m0 (vs Random) >= -0.003 and m8 (vs Curriculum) >= -0.003
        selective : AUROC-ONLY label unless >= 3% macro E-AURC or >= 1pp RER@50

Usage
-----
    python -u scripts/run_v2m_m3_conf.py --log-file logs_v2m_m3_conf.txt
    python -u scripts/run_v2m_m3_conf.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.mixture import (  # noqa: E402
    AdaptiveMix,
    CompetitionIndex,
    EqualMix,
    StaticMix,
    competition_features_from_sem14,
    macro_paired_cluster_bootstrap,
)
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

__all__ = ["build_parser", "main"]

OUT_ROOT = Path("results/v2_local_competition")
M3_DIR = OUT_ROOT / "m3_mixture"
CONF_DIR = M3_DIR / "conf"
V2G_ROOT = Path("results/v2_backbone_generalization")
FREEZE_PATH = V2G_ROOT / "g4_protocol_freeze.json"
PHASE_A_ROOT = V2G_ROOT / "g4_phaseA"
PHASE_B_ROOT = V2G_ROOT / "g4_phaseB"
EXPB_B0_DIR = Path("results/phase1f_hard_semantic/predictions")

MANIFESTS = Path("cache/manifests")
BANK = Path("cache/proposals.h5")
REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
IMAGE_SIZES = Path("cache/image_sizes.npz")
#: V2-G per-backbone embedding stores (B1/B2; NOT the B0 ``cache/semantic_phase1``).
EMB_ROOT = Path("cache/v2_semantic")
SPLIT_MANIFEST = Path("results/phase05_score_sufficiency/split_manifest.json")

BACKBONES: Tuple[Tuple[str, str], ...] = (("b1", "openclip_b16"), ("b2", "siglip_b16"))
LEVELS: Tuple[int, ...] = (0, 2, 4, 8)
MIXER_FIT_LEVELS: Tuple[int, ...] = (0, 2, 4)
MODELS: Tuple[str, ...] = ("E_random", "E_curriculum", "EqualMix", "StaticMix", "AdaptiveMix")
PER_LEVEL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("AdaptiveMix", "StaticMix"),
    ("AdaptiveMix", "E_random"),
    ("AdaptiveMix", "E_curriculum"),
)
MACRO_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("AdaptiveMix", "StaticMix"),
    ("AdaptiveMix", "E_random"),
    ("AdaptiveMix", "E_curriculum"),
)
BOOT_REPLICATES: int = 5000
BOOT_SEED: int = 0
BOOT_CI: float = 0.95
#: Anchor tolerance for checks whose inputs load verbatim from frozen
#: artifacts (V2-G Phase A K5 store: the stored scores + stored embeddings
#: are the only inputs), i.e. bitwise-reproducible by construction.
ANCHOR_STORE_TOL: float = 1e-9
#: Anchor tolerance for re-scoring checks: V2-G Phase B never persisted its
#: per-example dose-response scores, so the m-level R2 AUROC anchor compares
#: against a re-score.  Float32 re-scoring is not bitwise reproducible
#: (observed drift ~2.8e-07 AUROC; same class as the frozen 1e-4 score-level
#: STOP tolerance), while any wrong row set / model / T / feature pipeline
#: shifts AUROC by >= 1e-3 and is still caught.
ANCHOR_RESCORE_TOL: float = 1e-4

#: Frozen confirmatory gate constants (amendment V2-M3.1; never adjusted).
GATE_DELTA_MACRO_MIN: float = 0.003
GATE_EXTREME_SAFETY_MIN: float = -0.003
GATE_EAURC_REL_MIN: float = 0.03
GATE_RER50_GAIN_PP_MIN: float = 1.0

COLOURS = {
    "E_random": "#1f77b4", "E_curriculum": "#d62728", "EqualMix": "#7f7f7f",
    "StaticMix": "#ff7f0e", "AdaptiveMix": "#2ca02c",
}
POINT_FIELDS: Tuple[str, ...] = (
    "auroc_correct", "e_aurc", "rer_at_50", "rer_at_80",
    "ece_adaptive", "brier_binary", "nll_binary",
)


# ---------------------------------------------------------------------------
# small helpers (house style, mirrors run_v2m_m3_b0.py)
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


def _abs(path: Any) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO / p


def _load_module(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git(args: Sequence[str]) -> Optional[str]:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(_REPO), capture_output=True, text=True, timeout=30
        )
    except Exception:  # bookkeeping must never break a run
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _corrected_T(b3_root: Path, seed: int) -> float:
    """Per-(backbone, seed) corrected global T (read-only, interior-checked)."""
    em = _read_json(b3_root / f"seed_{int(seed)}" / "eval_metadata.json")
    value = float(em["temperature_corrected"])
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{b3_root} seed_{seed}: invalid corrected T {value}")
    if not bool(em.get("corrected_fit", {}).get("interior", em.get("corrected_interior", True))):
        raise ValueError(f"{b3_root} seed_{seed}: corrected T fit is not interior")
    return value


def _p_expert(
    model: Any,
    stats17: np.ndarray,
    sem14: np.ndarray,
    stats_fit: Any,
    sem_fit: Any,
) -> np.ndarray:
    """Frozen expert inference through the model object itself.

    Uses the fitted model's own ``predict_proba`` (the exact V2-G / M2 code
    path) rather than a hand-rolled sigmoid, so that the E_random anchors are
    bit-reproducible against the V2-G artifacts.
    """
    x31 = np.hstack(
        [
            rfeat.normalize_apply(np.asarray(stats17), stats_fit),
            rfeat.normalize_apply(np.asarray(sem14), sem_fit),
        ]
    )
    return np.asarray(model.predict_proba(x31), dtype=np.float64)


def _aggregate(
    boot_rows: Sequence[Mapping[str, Any]],
    level: str,
    model_a: str,
    model_b: str,
    metric: str,
) -> Optional[Dict[str, Any]]:
    sel = [
        r for r in boot_rows
        if r["level"] == level and r["model_a"] == model_a and r["model_b"] == model_b
        and r["metric"] == metric
    ]
    if not sel:
        return None
    sel.sort(key=lambda r: int(r["seed"]))
    diffs = [float(r["diff"]) for r in sel]
    lows = [float(r["ci_low"]) for r in sel]
    highs = [float(r["ci_high"]) for r in sel]
    return {
        "n_seeds": len(sel),
        "seeds": [int(r["seed"]) for r in sel],
        "delta": float(np.mean(diffs)),
        "ci_low": float(np.mean(lows)),
        "ci_high": float(np.mean(highs)),
        "per_seed": {int(r["seed"]): float(r["diff"]) for r in sel},
        "per_seed_ci": {int(r["seed"]): [float(r["ci_low"]), float(r["ci_high"])] for r in sel},
        "n_positive": int(sum(d > 0.0 for d in diffs)),
        "n_negative": int(sum(d < 0.0 for d in diffs)),
    }


def _stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# frozen artifact loading
# ---------------------------------------------------------------------------
def _backbone_paths(freeze: Mapping[str, Any], tag: str) -> Dict[str, Any]:
    for key, bb in freeze["backbones"].items():
        if bb["tag"] == tag:
            return {
                "key": key,
                "features_root": _abs(Path(bb["features_root"])),
                "b3_root": _abs(Path(bb["b3_root"])),
                "feature_dim": int(bb["feature_dim"]),
            }
    raise KeyError(f"backbone tag {tag!r} absent from the freeze file")


def _load_phase_a(tag: str, seed: int) -> Dict[str, Any]:
    path = PHASE_A_ROOT / tag / f"seed_{seed}" / "models.pkl"
    if not path.exists():
        raise FileNotFoundError(f"V2-G Phase A model missing: {path}")
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    for key in ("stats_fit", "sem_fit", "R1_model", "R2_model", "T"):
        if key not in payload:
            raise KeyError(f"{path}: pickle missing key {key!r}")
    return payload


def _load_store_bundle(
    bb_dir: str, paths: Mapping[str, Any], seed: int, k: int, temperature: float
) -> Dict[str, Any]:
    """Store rows of one K aligned 1:1 with the frozen scorer of this seed."""
    store = sdata.load_embedding_store(k, out_root=EMB_ROOT / bb_dir)
    ref = sdata.load_scorer_canonical(f"b3_seed{seed}", k, b3_root=paths["b3_root"])
    pos = np.searchsorted(ref.sentence_id, store.sentence_id)
    if pos.size != len(store) or not np.array_equal(ref.sentence_id[pos], store.sentence_id):
        raise AssertionError(f"{bb_dir}/K={k}/seed{seed}: store ids are not a subset of scorer rows")
    target_local = np.asarray(ref.target_local[pos], dtype=np.int32)
    if not np.all(target_local == 0):
        raise AssertionError(f"{bb_dir}/K={k}/seed{seed}: frozen target_local is not column 0")
    scores = np.asarray(ref.scores[pos], dtype=np.float64)
    z_q = np.asarray(store.z_q, dtype=np.float64)
    z_i = np.asarray(store.z_i, dtype=np.float64)
    stats17 = np.asarray(rfeat.stat_features(scores, temperature=temperature), dtype=np.float64)
    sem14 = np.asarray(v2feat.v2_primary_semantic_stats(z_q, z_i, scores), dtype=np.float64)
    correct = np.argmax(scores, axis=1) == 0
    if not np.array_equal(correct, np.asarray(ref.correct[pos], dtype=bool)):
        raise AssertionError(f"{bb_dir}/K={k}/seed{seed}: correctness differs from frozen scorer")
    bundle = {
        "k": int(k), "scores": scores, "correct": correct,
        "image_id": np.asarray(store.image_id, dtype=np.int64),
        "eval_split": np.asarray(store.eval_split),
        "sentence_id": np.asarray(store.sentence_id, dtype=np.int64),
        "ref_id": np.asarray(store.ref_id, dtype=np.int64),
        "stats17": stats17, "sem14": sem14,
    }
    del store, ref
    return bundle


# ---------------------------------------------------------------------------
# frozen-cell scoring (identical construction to B0; backbone checkpoint only)
# ---------------------------------------------------------------------------
def _test_level_blocks(
    corpus: B3Corpus,
    cohort_hc: shard.HardCohort,
    rows_same8: np.ndarray,
    scorers: Mapping[str, Any],
    seed: int,
    temperature: float,
    text_cache: Dict[int, np.ndarray],
    *,
    tag: str,
) -> Dict[int, Dict[str, Any]]:
    """same8 K=10 severity cells m in {0,2,4,8}, scored with this backbone.

    The row construction is backbone-independent: every cell is cross-checked
    against the frozen B0 ``expb`` arrays (sentence / image / ref rows) and the
    candidate indices are hashed into the artifacts.
    """
    model = scorers[f"b3_seed{seed}"]
    ids = np.asarray(cohort_hc.sentence_id[rows_same8], dtype=np.int64)
    imgs = np.asarray(cohort_hc.image_id[rows_same8], dtype=np.int64)
    refs = np.asarray(cohort_hc.ref_id[rows_same8], dtype=np.int64)
    blocks: Dict[int, Dict[str, Any]] = {}
    for m in LEVELS:
        batch, cand = _materialise(corpus, cohort_hc, rows_same8, int(m), text_cache)
        scores = np.asarray(hscores.score_examples(model, batch, batch_size=256), dtype=np.float64)
        feats = _features_from_arrays(scores, batch.z_q, batch.z_i, temperature, tag=f"{tag} test m{m}/seed{seed}")
        frozen_path = EXPB_B0_DIR / f"expb_m{int(m)}__b3_seed{seed}.npz"
        with np.load(frozen_path) as frozen:
            f_ids = np.asarray(frozen["sentence_id"], dtype=np.int64)
            f_img = np.asarray(frozen["image_id"], dtype=np.int64)
            f_ref = np.asarray(frozen["ref_id"], dtype=np.int64)
        if not (
            np.array_equal(f_ids, ids) and np.array_equal(f_img, imgs) and np.array_equal(f_ref, refs)
        ):
            raise AssertionError(
                f"{tag} expb_m{m}/seed{seed}: constructed rows differ from the frozen B0 cell"
            )
        correct = np.argmax(scores, axis=1) == 0
        blocks[int(m)] = {
            "rows": rows_same8,
            "sentence_id": ids, "image_id": imgs, "ref_id": refs,
            "candidate_sha256": _sha256_array(np.asarray(cand, dtype=np.int64)),
            "scores": scores, "correct": correct,
            "stats17": feats["stats17"], "sem14": feats["sem14"],
        }
        del batch
    return blocks


def _materialise(
    corpus: B3Corpus,
    cohort: Any,
    rows: np.ndarray,
    m: int,
    text_cache: Dict[int, np.ndarray],
) -> Tuple[hscores.B3ExampleBatch, np.ndarray]:
    """Level-``m`` K=10 cell from the frozen construction (no resampling)."""
    k = 10
    row_list = np.asarray(rows, dtype=np.int64).tolist()
    cand = np.empty((len(row_list), k), dtype=np.int64)
    samples: List[Any] = []
    for i, row in enumerate(row_list):
        sample = cohort.level_sample(int(row), int(m), k)
        cand[i] = B3Corpus.candidate_bank_indices(sample, k)
        samples.append(sample)
    batch = hscores.materialise_examples(corpus, samples, k, text_cache=text_cache)
    return batch, cand


def _features_from_arrays(
    scores: np.ndarray, z_q: np.ndarray, z_i: np.ndarray, temperature: float, *, tag: str
) -> Dict[str, np.ndarray]:
    """stats17 + sem14 of one row block (chunked; identical calls to the runners)."""
    sc_all = np.asarray(scores, dtype=np.float64)
    n = int(sc_all.shape[0])
    stats17 = np.empty((n, len(rfeat.stat_feature_names())), dtype=np.float64)
    sem14 = np.empty((n, len(v2feat.V2_PRIMARY_SEMANTIC_NAMES)), dtype=np.float64)
    for start in range(0, n, 2048):
        stop = min(start + 2048, n)
        sc = sc_all[start:stop]
        zq = np.asarray(z_q[start:stop], dtype=np.float64)
        zi = np.asarray(z_i[start:stop], dtype=np.float64)
        stats17[start:stop] = rfeat.stat_features(sc, temperature=temperature)
        sem14[start:stop] = v2feat.v2_primary_semantic_stats(zq, zi, sc)
    for name, block in (("stats17", stats17), ("sem14", sem14)):
        if not np.all(np.isfinite(block)):
            raise AssertionError(f"{tag}: non-finite {name}")
    return {"stats17": stats17, "sem14": sem14}


# ---------------------------------------------------------------------------
# per-seed confirmatory run of one backbone
# ---------------------------------------------------------------------------
def _run_seed(
    seed: int,
    *,
    tag: str,
    bb_dir: str,
    paths: Mapping[str, Any],
    phase_a: Mapping[str, Any],
    runner: Any,
    corpus: B3Corpus,
    cohort_hc: shard.HardCohort,
    rows_same8: np.ndarray,
    val_cohort: Any,
    scorers: Mapping[str, Any],
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    t_seed = time.perf_counter()
    temperature = _corrected_T(paths["b3_root"], int(seed))
    pk = phase_a[f"b3_seed{seed}"]
    log(f"[M3-CONF {tag} seed{seed}] start (T_corrected={temperature:.6f})")

    # -- 1. E_random anchor: reproduce V2-G Phase A / dose-response -------------
    bundle10 = _load_store_bundle(bb_dir, paths, seed, 10, temperature)
    bundle5 = _load_store_bundle(bb_dir, paths, seed, 5, temperature)
    test5 = np.isin(np.asarray(bundle5["eval_split"]), ("testA", "testB"))
    p_r_k5 = _p_expert(
        pk["R2_model"], np.asarray(bundle5["stats17"])[test5], np.asarray(bundle5["sem14"])[test5],
        pk["stats_fit"], pk["sem_fit"],
    )
    auroc_k5 = float(
        reval.point_metric_row(
            p_r_k5, np.asarray(bundle5["correct"])[test5], probability=p_r_k5
        )["auroc_correct"]
    )
    stored_k5 = _phase_a_k5_auroc(tag, int(seed))
    anchor_phase_a = abs(auroc_k5 - stored_k5)
    if not (anchor_phase_a <= ANCHOR_STORE_TOL):
        raise AssertionError(
            f"{tag} seed{seed}: E_random does not reproduce V2-G Phase A K5 pooled-test "
            f"AUROC ({auroc_k5:.6f} vs {stored_k5:.6f})"
        )
    log(
        f"[M3-CONF {tag} seed{seed}] E_random reproduces V2-G Phase A K5 AUROC "
        f"({auroc_k5:.6f}, max|delta|={anchor_phase_a:.2e})"
    )

    # -- 2. val level cells + curriculum standardisation (M2 protocol) ----------
    r1_coef, r1_intercept = pk["R1_model"].coefficients()
    b_of = runner._b_function(pk["stats_fit"], np.asarray(r1_coef, dtype=np.float64), float(r1_intercept))
    text_cache: Dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    val_blocks = runner._val_level_blocks(
        corpus, val_cohort, scorers, int(seed), temperature, text_cache, b_of, log
    )
    log(f"[M3-CONF {tag} seed{seed}] val cells materialised ({time.perf_counter() - t0:.1f}s)")
    stop_val = runner._val_stop_check(
        corpus, val_cohort, scorers, int(seed), bundle10, val_blocks, text_cache, log
    )
    train_stats = np.vstack(
        [np.asarray(val_blocks["train"][m]["stats17"]) for m in MIXER_FIT_LEVELS]
    )
    train_sem = np.vstack(
        [np.asarray(val_blocks["train"][m]["sem14"]) for m in MIXER_FIT_LEVELS]
    )
    stats_fit_curr = rfeat.normalize_fit(
        train_stats, fit_rows=np.arange(train_stats.shape[0]), keys=runner.STATS17_NAMES
    )
    sem_fit_curr = rfeat.normalize_fit(
        train_sem, fit_rows=np.arange(train_sem.shape[0]), keys=runner.SEM14_NAMES
    )
    correct_train = {m: val_blocks["train"][m]["correct"] for m in MIXER_FIT_LEVELS}
    correct_tune = {m: val_blocks["tune"][m]["correct"] for m in MIXER_FIT_LEVELS}
    x31_std = {
        side: {
            m: np.hstack(
                [
                    rfeat.normalize_apply(
                        np.asarray(val_blocks[side][m]["stats17"]), stats_fit_curr
                    ),
                    rfeat.normalize_apply(
                        np.asarray(val_blocks[side][m]["sem14"]), sem_fit_curr
                    ),
                ]
            )
            for m in MIXER_FIT_LEVELS
        }
        for side in ("train", "tune")
    }

    # -- 3. E_curriculum: identical M2 curriculum protocol ----------------------
    e1b = runner._fit_logistic_curriculum(
        {m: x31_std["train"][m] for m in MIXER_FIT_LEVELS}, correct_train,
        {m: x31_std["tune"][m] for m in MIXER_FIT_LEVELS}, correct_tune,
        name=f"{tag}:seed{seed}:E_curriculum", log=log,
    )
    log(
        f"[M3-CONF {tag} seed{seed}] E_curriculum trained (C={e1b['chosen_C']:g}, "
        f"tune balanced AUROC={e1b['tune_mean_auroc']:.4f})"
    )

    # -- 4. competition index: train-only CDF (frozen formula) ------------------
    train_features = {
        name: np.concatenate(
            [
                competition_features_from_sem14(
                    np.asarray(val_blocks["train"][m]["sem14"]), runner.SEM14_NAMES
                )[name]
                for m in MIXER_FIT_LEVELS
            ]
        )
        for name in ("winner_competitor_max_cos", "winner_top2_cos", "q_margin12")
    }
    index = CompetitionIndex().fit(train_features)
    log(f"[M3-CONF {tag} seed{seed}] competition index fitted (train-only CDF)")

    # -- 5. mixers fitted on reliability_tune m={0,2,4} only --------------------
    groups: Dict[int, Dict[str, np.ndarray]] = {}
    for m in MIXER_FIT_LEVELS:
        block = val_blocks["tune"][m]
        sem14 = np.asarray(block["sem14"])
        a_tune = index.transform(competition_features_from_sem14(sem14, runner.SEM14_NAMES))
        groups[int(m)] = {
            "p_r": _p_expert(
                pk["R2_model"], block["stats17"], sem14, pk["stats_fit"], pk["sem_fit"]
            ),
            "p_c": np.asarray(e1b["model"].predict_proba(x31_std["tune"][m]), dtype=np.float64),
            "a": a_tune,
            "y": np.asarray(block["correct"], dtype=np.float64),
        }
    static_mix = StaticMix().fit(groups)
    adaptive_mix = AdaptiveMix().fit(groups)
    fit_record = {
        "static_c": float(static_mix.c),
        "static_nll": float(static_mix.nll),
        "static_success": bool(static_mix.success),
        "adaptive_theta": float(adaptive_mix.theta),
        "adaptive_tau": float(adaptive_mix.tau),
        "adaptive_beta": float(adaptive_mix.beta()),
        "adaptive_nll": float(adaptive_mix.nll),
        "adaptive_success": bool(adaptive_mix.success),
        "tune_rows": {f"m{m}": int(groups[m]["y"].size) for m in MIXER_FIT_LEVELS},
        "curriculum_C": float(e1b["chosen_C"]),
        "curriculum_tune_balanced_auroc": float(e1b["tune_mean_auroc"]),
    }
    log(
        f"[M3-CONF {tag} seed{seed}] mixers fitted on tune m={{0,2,4}}: "
        f"c={static_mix.c:.4f} theta={adaptive_mix.theta:.4f} tau={adaptive_mix.tau:.4f} "
        f"beta={adaptive_mix.beta():.4f}"
    )

    # -- 6. frozen test cells (same8, K=10) + dose anchor -----------------------
    t0 = time.perf_counter()
    test_blocks = _test_level_blocks(
        corpus, cohort_hc, rows_same8, scorers, int(seed), temperature, text_cache, tag=tag
    )
    log(f"[M3-CONF {tag} seed{seed}] frozen test cells scored ({time.perf_counter() - t0:.1f}s)")

    # -- 7. inference + metrics + bootstraps ------------------------------------
    point_rows: List[Dict[str, Any]] = []
    calib_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []
    regret_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    anchor_rows: List[Dict[str, Any]] = []
    per_level: Dict[int, Dict[str, np.ndarray]] = {}
    reference_clusters: Optional[np.ndarray] = None
    anchor_dose_max = 0.0
    for m in LEVELS:
        level = f"m{m}"
        block = test_blocks[int(m)]
        correct = np.asarray(block["correct"], dtype=bool)
        clusters = np.asarray(block["image_id"], dtype=np.int64)
        if reference_clusters is None:
            reference_clusters = clusters.copy()
        elif not (
            clusters.shape == reference_clusters.shape
            and np.array_equal(clusters, reference_clusters)
        ):
            raise AssertionError(f"{tag} seed{seed}/{level}: severity cells disagree on the shared rows")
        sem14 = np.asarray(block["sem14"])
        p_r = _p_expert(
            pk["R2_model"], block["stats17"], sem14, pk["stats_fit"], pk["sem_fit"]
        )
        p_c = _p_expert(
            e1b["model"], block["stats17"], sem14, stats_fit_curr, sem_fit_curr
        )
        stored_dose = _dose_r2_auroc(tag, int(seed), int(m))
        auroc_pr = float(reval.point_metric_row(p_r, correct, probability=p_r)["auroc_correct"])
        anchor_dose = abs(auroc_pr - stored_dose)
        anchor_dose_max = max(anchor_dose_max, anchor_dose)
        if not (anchor_dose <= ANCHOR_RESCORE_TOL):
            raise AssertionError(
                f"{tag} seed{seed}/{level}: E_random does not reproduce the V2-G dose-response "
                f"R2 AUROC ({auroc_pr:.6f} vs {stored_dose:.6f})"
            )
        anchor_rows.append(
            {
                "backbone": tag, "seed": int(seed), "level": level,
                "auroc_e_random": auroc_pr, "auroc_r2_v2g": stored_dose, "abs_delta": anchor_dose,
                "candidate_sha256": block["candidate_sha256"],
            }
        )

        a_test = index.transform(competition_features_from_sem14(sem14, runner.SEM14_NAMES))
        predictions = {
            "E_random": p_r,
            "E_curriculum": p_c,
            "EqualMix": EqualMix().predict_proba(p_r, p_c, a_test),
            "StaticMix": static_mix.predict_proba(p_r, p_c, a_test),
            "AdaptiveMix": adaptive_mix.predict_proba(p_r, p_c, a_test),
        }
        for model in MODELS:
            if predictions[model].shape != correct.shape:
                raise AssertionError(f"{tag} seed{seed}/{level}/{model}: row universe mismatch")
        per_level[int(m)] = {**predictions, "correct": correct, "clusters": clusters, "a": a_test}

        for model in MODELS:
            row = reval.point_metric_row(predictions[model], correct, probability=predictions[model])
            row.update(
                {
                    "level": level, "K": 10, "seed": int(seed), "model": model,
                    "n": int(correct.size), "n_images": int(np.unique(clusters).size),
                    "b3_accuracy": float(correct.mean()),
                }
            )
            point_rows.append(row)
            calib_rows.append(
                {
                    "level": level, "seed": int(seed), "model": model,
                    "mean_pred": float(np.mean(predictions[model])),
                    "accuracy": float(correct.mean()),
                    "ece_adaptive": float(row["ece_adaptive"]),
                    "brier_binary": float(row["brier_binary"]),
                    "nll_binary": float(row["nll_binary"]),
                }
            )

        alpha_values = adaptive_mix.alpha(a_test)
        alpha_rows.append(
            {
                "level": level, "seed": int(seed), "n": int(a_test.size),
                "a_mean": float(np.mean(a_test)), "a_median": float(np.median(a_test)),
                "a_p10": float(np.percentile(a_test, 10.0)),
                "a_p90": float(np.percentile(a_test, 90.0)),
                "alpha_mean": float(np.mean(alpha_values)),
                "alpha_median": float(np.median(alpha_values)),
            }
        )

        auroc_r = auroc_pr
        auroc_c = float(reval.point_metric_row(p_c, correct, probability=p_c)["auroc_correct"])
        oracle = max(auroc_r, auroc_c)
        for model in MODELS:
            model_auroc = float(
                reval.point_metric_row(
                    predictions[model], correct, probability=predictions[model]
                )["auroc_correct"]
            )
            regret_rows.append(
                {
                    "level": level, "seed": int(seed), "model": model,
                    "oracle_auroc": oracle, "regret": oracle - model_auroc,
                }
            )

        for model_a, model_b in PER_LEVEL_PAIRS:
            rows = reval.model_vs_model_bootstrap_row(
                predictions[model_a], correct, predictions[model_b], correct, clusters,
                eval_split=shard.POOLED, K=10,
                model_a=model_a, model_b=model_b, metrics=runner.BOOT_METRICS,
                replicates=int(args.bootstrap_replicates), seed=int(args.bootstrap_seed),
                ci=float(args.ci),
            )
            for row in rows:
                row.update({"level": level, "seed": int(seed)})
                boot_rows.append(row)
        log(
            f"  [M3-CONF {tag} seed{seed}] {level}: dAUROC(A-S)="
            f"{float(np.mean([r['diff'] for r in boot_rows if r['level'] == level and r['model_a'] == 'AdaptiveMix' and r['model_b'] == 'StaticMix' and r['metric'] == 'auroc_correct'])):+.4f}"
        )

    # -- 8. macro bootstrap (shared draws across severity) ----------------------
    metric = reval._metric_fn("auroc_correct")
    for model_a, model_b in MACRO_PAIRS:
        pairs = [
            (
                (per_level[int(m)][model_a], per_level[int(m)]["correct"]),
                (per_level[int(m)][model_b], per_level[int(m)]["correct"]),
            )
            for m in LEVELS
        ]
        result = macro_paired_cluster_bootstrap(
            metric, pairs, reference_clusters,
            n_replicates=int(args.bootstrap_replicates), seed=int(args.bootstrap_seed),
            ci=float(args.ci), metric_name="auroc_correct",
        )
        boot_rows.append(
            {
                "level": "macro", "K": 10, "seed": int(seed),
                "model_a": model_a, "model_b": model_b, "metric": "auroc_correct",
                "diff": float(result["diff"]), "ci_low": float(result["ci_low"]),
                "ci_high": float(result["ci_high"]), "mean_a": float(result["mean_a"]),
                "mean_b": float(result["mean_b"]), "n": int(result["n"]),
                "n_clusters": int(result["n_clusters"]),
                "n_replicates": int(result["n_replicates"]),
                "ci_level": float(result["ci_level"]),
            }
        )

    # -- 9. macro metrics -------------------------------------------------------
    macro_rows: List[Dict[str, Any]] = []
    for model in MODELS:
        macro_auroc = float(
            np.mean(
                [float(r["auroc_correct"]) for r in point_rows if r["model"] == model]
            )
        )
        macro_rows.append(
            {
                "seed": int(seed), "model": model, "macro_auroc": macro_auroc,
                "macro_e_aurc": float(
                    np.mean([float(r["e_aurc"]) for r in point_rows if r["model"] == model])
                ),
                "macro_rer50": float(
                    np.mean([float(r["rer_at_50"]) for r in point_rows if r["model"] == model])
                ),
                "macro_rer80": float(
                    np.mean([float(r["rer_at_80"]) for r in point_rows if r["model"] == model])
                ),
            }
        )

    log(
        f"[M3-CONF {tag} seed{seed}] done in {time.perf_counter() - t_seed:.1f}s "
        f"(anchor phaseA {anchor_phase_a:.2e}; dose max {anchor_dose_max:.2e}; "
        f"val stop {stop_val['max_delta']:.2e})"
    )
    return {
        "seed": int(seed),
        "temperature": temperature,
        "fit": fit_record,
        "index_summary": index.summary(),
        "anchor_phase_a_k5_auroc": float(auroc_k5),
        "anchor_phase_a_stored": float(stored_k5),
        "anchor_phase_a_max_delta": float(anchor_phase_a),
        "anchor_dose_max_delta": float(anchor_dose_max),
        "val_stop_max_delta": float(stop_val["max_delta"]),
        "point_rows": point_rows,
        "macro_rows": macro_rows,
        "calib_rows": calib_rows,
        "alpha_rows": alpha_rows,
        "regret_rows": regret_rows,
        "boot_rows": boot_rows,
        "anchor_rows": anchor_rows,
    }


# ---------------------------------------------------------------------------
# frozen V2-G reference values (anchors)
# ---------------------------------------------------------------------------
def _phase_a_k5_auroc(tag: str, seed: int) -> float:
    path = PHASE_A_ROOT / "phaseA_summary.csv"
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["backbone"] == tag and int(row["seed"]) == seed
                and int(row["K"]) == 5 and row["model"] == "R2_stats_sem"
            ):
                return float(row["auroc_correct"])
    raise KeyError(f"phaseA_summary missing {tag}/seed{seed}/K5/R2_stats_sem")


def _dose_r2_auroc(tag: str, seed: int, level: int) -> float:
    """V2-G Phase B dose-response R2 AUROC of one (backbone, seed, level) cell."""
    path = PHASE_B_ROOT / tag / "dose_response.csv"
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row["scorer"]) == f"b3_seed{seed}" and int(row["level"]) == int(level):
                return float(row["auroc_R2"])
    raise KeyError(f"dose_response missing {tag}/seed{seed}/m{level}")


# ---------------------------------------------------------------------------
# gate assembly (frozen Amendment V2-M3.1)
# ---------------------------------------------------------------------------
def _backbone_gate(
    per_seed: Sequence[Mapping[str, Any]],
    boot_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    runner: Any,
) -> Dict[str, Any]:
    metrics = tuple(runner.BOOT_METRICS)
    delta_macro = {
        "vs_static": _aggregate(boot_rows, "macro", "AdaptiveMix", "StaticMix", "auroc_correct"),
        "vs_random": _aggregate(boot_rows, "macro", "AdaptiveMix", "E_random", "auroc_correct"),
        "vs_curriculum": _aggregate(boot_rows, "macro", "AdaptiveMix", "E_curriculum", "auroc_correct"),
    }
    extreme = {
        "m0_adaptive_minus_random": _aggregate(boot_rows, "m0", "AdaptiveMix", "E_random", "auroc_correct"),
        "m8_adaptive_minus_curriculum": _aggregate(boot_rows, "m8", "AdaptiveMix", "E_curriculum", "auroc_correct"),
    }
    primary = delta_macro["vs_static"]
    primary_pass = bool(
        float(primary["delta"]) >= GATE_DELTA_MACRO_MIN and float(primary["ci_low"]) > 0.0
    )
    safety = {
        "m0_pass": bool(float(extreme["m0_adaptive_minus_random"]["delta"]) >= GATE_EXTREME_SAFETY_MIN),
        "m8_pass": bool(float(extreme["m8_adaptive_minus_curriculum"]["delta"]) >= GATE_EXTREME_SAFETY_MIN),
        "threshold": GATE_EXTREME_SAFETY_MIN,
    }
    safety_pass = bool(safety["m0_pass"] and safety["m8_pass"])

    macro_metrics: Dict[str, Any] = {}
    for model in MODELS:
        rows = sorted([r for r in macro_rows if r["model"] == model], key=lambda r: int(r["seed"]))
        macro_metrics[model] = {
            "macro_auroc": _stats([r["macro_auroc"] for r in rows]),
            "macro_e_aurc": _stats([r["macro_e_aurc"] for r in rows]),
            "macro_rer50": _stats([r["macro_rer50"] for r in rows]),
            "macro_rer80": _stats([r["macro_rer80"] for r in rows]),
            "per_seed": [
                {k: r[k] for k in ("seed", "macro_auroc", "macro_e_aurc", "macro_rer50", "macro_rer80")}
                for r in rows
            ],
        }
    static_e = float(macro_metrics["StaticMix"]["macro_e_aurc"]["mean"])
    adaptive_e = float(macro_metrics["AdaptiveMix"]["macro_e_aurc"]["mean"])
    e_aurc_rel = (
        float((static_e - adaptive_e) / static_e)
        if np.isfinite(static_e) and abs(static_e) > 0.0
        else None
    )
    static_r50 = float(macro_metrics["StaticMix"]["macro_rer50"]["mean"])
    adaptive_r50 = float(macro_metrics["AdaptiveMix"]["macro_rer50"]["mean"])
    rer50_gain_pp = float(100.0 * (adaptive_r50 - static_r50))
    selective_support = bool(
        (e_aurc_rel is not None and e_aurc_rel >= GATE_EAURC_REL_MIN)
        or rer50_gain_pp >= GATE_RER50_GAIN_PP_MIN
    )

    alpha_by_level: Dict[str, Any] = {}
    for m in LEVELS:
        sel = [r for r in alpha_rows if r["level"] == f"m{m}"]
        alpha_by_level[f"m{m}"] = {
            "a_mean": _stats([r["a_mean"] for r in sel]),
            "a_median": _stats([r["a_median"] for r in sel]),
            "a_p10": _stats([r["a_p10"] for r in sel]),
            "a_p90": _stats([r["a_p90"] for r in sel]),
            "alpha_mean": _stats([r["alpha_mean"] for r in sel]),
            "alpha_median": _stats([r["alpha_median"] for r in sel]),
        }
    alpha_seq = [float(alpha_by_level[f"m{m}"]["alpha_mean"]["mean"]) for m in LEVELS]
    a_seq = [float(alpha_by_level[f"m{m}"]["a_mean"]["mean"]) for m in LEVELS]
    alpha_gap = float(alpha_seq[-1] - alpha_seq[0])

    fit_keys = ("static_c", "adaptive_theta", "adaptive_tau", "adaptive_beta")
    fits = {key: _stats([float(e["fit"][key]) for e in per_seed]) for key in fit_keys}
    fits["curriculum_C"] = [float(e["fit"]["curriculum_C"]) for e in per_seed]
    fits["success"] = {
        "static": [bool(e["fit"]["static_success"]) for e in per_seed],
        "adaptive": [bool(e["fit"]["adaptive_success"]) for e in per_seed],
    }

    anchor = {
        "phase_a_k5_max_delta": max(float(e["anchor_phase_a_max_delta"]) for e in per_seed),
        "dose_r2_max_delta": max(float(e["anchor_dose_max_delta"]) for e in per_seed),
        "val_stop_max_delta": max(float(e["val_stop_max_delta"]) for e in per_seed),
        "tolerances": {
            "phase_a_k5_store": ANCHOR_STORE_TOL,
            "dose_rescore": ANCHOR_RESCORE_TOL,
        },
    }

    return {
        "primary_comparison": "AdaptiveMix vs StaticMix (macro AUROC, shared-draws paired bootstrap)",
        "delta_macro": delta_macro,
        "primary_gate": {
            "rule": "delta_macro >= 0.003 AND CI_low > 0",
            "threshold": GATE_DELTA_MACRO_MIN,
            "pass": primary_pass,
        },
        "extreme_safety": {**extreme, **safety, "pass": safety_pass},
        "selective_support": {
            "macro_e_aurc": {
                "adaptive": adaptive_e, "static": static_e, "relative_improvement": e_aurc_rel,
                "threshold": GATE_EAURC_REL_MIN,
            },
            "macro_rer50": {
                "adaptive": adaptive_r50, "static": static_r50,
                "gain_pp": rer50_gain_pp, "threshold_pp": GATE_RER50_GAIN_PP_MIN,
            },
            "draft_rule_satisfied": selective_support,
            "label_if_pass_without_support": "AUROC-ONLY METHOD SIGNAL",
        },
        "pass": bool(primary_pass and safety_pass),
        "macro_metrics": macro_metrics,
        "alpha_by_level": alpha_by_level,
        "alpha_trend": {
            "alpha_mean_by_m": {f"m{m}": float(alpha_seq[i]) for i, m in enumerate(LEVELS)},
            "a_mean_by_m": {f"m{m}": float(a_seq[i]) for i, m in enumerate(LEVELS)},
            "alpha_m8_minus_m0": alpha_gap,
            "alpha_non_decreasing_in_m": bool(
                all(alpha_seq[i + 1] >= alpha_seq[i] - 1e-12 for i in range(len(alpha_seq) - 1))
            ),
        },
        "fits": fits,
        "anchors": anchor,
        "boot_metrics": list(metrics),
    }


def _cross_backbone_verdict(gates: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    tags = sorted(gates)
    n_pass = int(sum(1 for tag in tags if gates[tag]["pass"]))
    if n_pass == len(tags) and len(tags) >= 2:
        verdict = "STRONG ADAPTIVE-MIXTURE GENERALITY"
    elif n_pass == len(tags) - 1:
        other = [tag for tag in tags if not gates[tag]["pass"]][0]
        delta = gates[other]["delta_macro"]["vs_static"]
        harm = bool(float(delta["delta"]) < 0.0 and float(delta["ci_high"]) < 0.0)
        verdict = "ADAPTIVE MIXTURE NOT SUPPORTED" if harm else "PARTIAL METHOD SUPPORT"
    else:
        verdict = "ADAPTIVE MIXTURE NOT SUPPORTED"
    return {
        "artifact": "v2m_m3_conf_verdict",
        "amendment": "V2-M3.1",
        "backbones": {
            tag: {
                "pass": bool(gates[tag]["pass"]),
                "primary_pass": bool(gates[tag]["primary_gate"]["pass"]),
                "safety_pass": bool(gates[tag]["extreme_safety"]["pass"]),
                "selective_support": bool(gates[tag]["selective_support"]["draft_rule_satisfied"]),
                "delta_macro": float(gates[tag]["delta_macro"]["vs_static"]["delta"]),
                "delta_macro_ci": [
                    float(gates[tag]["delta_macro"]["vs_static"]["ci_low"]),
                    float(gates[tag]["delta_macro"]["vs_static"]["ci_high"]),
                ],
            }
            for tag in tags
        },
        "n_pass": n_pass,
        "n_backbones": len(tags),
        "verdict": verdict,
        "rule": (
            "2/2 pass -> STRONG ADAPTIVE-MIXTURE GENERALITY; 1 pass with the other "
            "non-significant and not significantly negative -> PARTIAL METHOD SUPPORT; "
            "otherwise -> ADAPTIVE MIXTURE NOT SUPPORTED (stop the M3 line)"
        ),
        "stop": bool(verdict == "ADAPTIVE MIXTURE NOT SUPPORTED"),
    }


# ---------------------------------------------------------------------------
# figures (per backbone)
# ---------------------------------------------------------------------------
def _write_figures(
    tag: str,
    point_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None],
) -> List[Path]:
    x = np.arange(len(LEVELS), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    ax1, ax2 = axes
    for model in MODELS:
        ys: List[float] = []
        es: List[float] = []
        for m in LEVELS:
            vals = [
                float(r["auroc_correct"]) for r in point_rows
                if r["model"] == model and r["level"] == f"m{m}"
            ]
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals)))
        ax1.errorbar(x, ys, yerr=es, marker="o", capsize=3.0, linewidth=1.7,
                     color=COLOURS[model], label=model)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"m={m}" for m in LEVELS])
    ax1.set_xlabel("competition severity m (same-category distractors), K=10")
    ax1.set_ylabel("AUROC(correct), frozen test cells")
    ax1.set_title(f"V2-M3.1 {tag}: 5-model family vs severity (seed mean +/- std)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    a_means, a_stds, alpha_means, alpha_stds = [], [], [], []
    for m in LEVELS:
        sel = [r for r in alpha_rows if r["level"] == f"m{m}"]
        a_means.append(float(np.mean([r["a_mean"] for r in sel])))
        a_stds.append(float(np.std([r["a_mean"] for r in sel])))
        alpha_means.append(float(np.mean([r["alpha_mean"] for r in sel])))
        alpha_stds.append(float(np.std([r["alpha_mean"] for r in sel])))
    ax2.errorbar(x, alpha_means, yerr=alpha_stds, marker="s", capsize=3.0, linewidth=1.7,
                 color=COLOURS["AdaptiveMix"], label="alpha(A) mean (AdaptiveMix)")
    ax2.errorbar(x, a_means, yerr=a_stds, marker="^", capsize=3.0, linewidth=1.5,
                 linestyle="--", color="#7f7f7f", label="A mean (competition index)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"m={m}" for m in LEVELS])
    ax2.set_xlabel("competition severity m (same-category distractors), K=10")
    ax2.set_ylabel("mean in [0, 1]")
    ax2.set_title(f"V2-M3.1 {tag}: A and alpha(A) by severity "
                  f"(alpha gap m8-m0 = {gate['alpha_trend']['alpha_m8_minus_m0']:+.4f})")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path1 = fig_dir / f"m3_conf_{tag}_curves.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)
    log(f"[M3-CONF {tag}] figure written: {path1}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    ax1, ax2 = axes
    positions = np.arange(len(MODELS), dtype=float)
    means, stds = [], []
    for model in MODELS:
        vals = [float(r["macro_auroc"]) for r in macro_rows if r["model"] == model]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    ax1.bar(positions, means, yerr=stds, capsize=4.0, color=[COLOURS[m] for m in MODELS])
    for index, model in enumerate(MODELS):
        vals = [float(r["macro_auroc"]) for r in macro_rows if r["model"] == model]
        ax1.scatter(np.full(len(vals), positions[index]), vals, color="black", s=14, zorder=3)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(list(MODELS), rotation=12)
    ax1.set_ylabel("MacroAUROC over m in {0,2,4,8}")
    ax1.set_title(f"V2-M3.1 {tag}: MacroAUROC (seed mean +/- std, dots = seeds)")
    ax1.grid(alpha=0.3, axis="y")

    dmacro = gate["delta_macro"]["vs_static"]
    seeds_sorted = sorted(int(s) for s in dmacro["per_seed"])
    deltas = np.array([float(dmacro["per_seed"][s]) for s in seeds_sorted])
    lows = np.array([float(dmacro["per_seed_ci"][s][0]) for s in seeds_sorted])
    highs = np.array([float(dmacro["per_seed_ci"][s][1]) for s in seeds_sorted])
    ax2.errorbar(
        np.arange(len(seeds_sorted), dtype=float), deltas,
        yerr=np.vstack([deltas - lows, highs - deltas]), marker="o", linestyle="none",
        capsize=4.0, color=COLOURS["AdaptiveMix"],
        label="per-seed delta_macro (95% paired CI)",
    )
    ax2.axhline(float(dmacro["delta"]), color=COLOURS["AdaptiveMix"], linewidth=1.2,
                linestyle="--", label=f"seed mean = {float(dmacro['delta']):+.4f}")
    ax2.axhline(0.0, color="black", linewidth=0.9)
    ax2.axhline(GATE_DELTA_MACRO_MIN, color="#9467bd", linewidth=1.0, linestyle=":",
                label="frozen gate 0.003")
    ax2.set_xticks(np.arange(len(seeds_sorted), dtype=float))
    ax2.set_xticklabels([f"seed {s}" for s in seeds_sorted])
    ax2.set_ylabel("MacroAUROC(AdaptiveMix) - MacroAUROC(StaticMix)")
    ax2.set_title(f"V2-M3.1 {tag}: primary comparison "
                  f"(pass={gate['primary_gate']['pass']})")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path2 = fig_dir / f"m3_conf_{tag}_macro_delta.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    log(f"[M3-CONF {tag}] figure written: {path2}")
    return [path1, path2]


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backbones", default="b1,b2", help="comma-separated tags (b1,b2)")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--bootstrap-replicates", type=int, default=BOOT_REPLICATES)
    p.add_argument("--bootstrap-seed", type=int, default=BOOT_SEED)
    p.add_argument("--ci", type=float, default=BOOT_CI)
    p.add_argument(
        "--smoke", action="store_true",
        help="1 seed / 100 reps / results/v2_local_competition/m3_mixture/conf/_smoke",
    )
    p.add_argument("--out-dir", default=None)
    p.add_argument("--log-file", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds: Tuple[int, ...] = (
        (1,) if args.smoke else tuple(int(s) for s in str(args.seeds).split(","))
    )
    reps = 100 if args.smoke else int(args.bootstrap_replicates)
    tags = tuple(str(t).strip() for t in str(args.backbones).split(","))
    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else ((CONF_DIR / "_smoke") if args.smoke else CONF_DIR)
    )
    run_args = argparse.Namespace(
        bootstrap_replicates=int(reps), bootstrap_seed=int(args.bootstrap_seed),
        ci=float(args.ci),
    )
    log = _make_logger(Path(args.log_file) if args.log_file else None)
    started = time.perf_counter()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    runner = _load_module("run_v2m_m2", "scripts/run_v2m_m2.py")
    freeze = _read_json(FREEZE_PATH)
    cohort_hc = shard.load_hard_cohort(
        features_dir=sdata.PHASE05_FEATURES_DIR, manifests_root=_abs(MANIFESTS), log=log
    )
    rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    build_module = _load_module("build_v2m_m2_manifests", "scripts/build_v2m_m2_manifests.py")
    val_cohort = build_module.load_val_cohort()

    log(
        "[M3-CONF] V2-M3.1 cross-backbone confirmatory run start "
        f"(backbones={list(tags)}, seeds={list(seeds)}, reps={reps})"
    )
    gates: Dict[str, Dict[str, Any]] = {}
    for tag in tags:
        bb_dir = dict((t, d) for t, d in BACKBONES)[tag]
        paths = _backbone_paths(freeze, bb_dir)
        phase_a = {f"b3_seed{s}": _load_phase_a(tag, s) for s in seeds}
        scorers = hscores.load_frozen_scorers(b3_root=paths["b3_root"], seeds=seeds, device="cpu")
        corpus = B3Corpus(
            paths["features_root"], _abs(MANIFESTS), _abs(REFS), _abs(BANK),
            image_sizes_path=_abs(IMAGE_SIZES), ks=(5,), regime="random", preload=True,
        )
        per_seed: List[Dict[str, Any]] = []
        try:
            for seed in seeds:
                per_seed.append(
                    _run_seed(
                        int(seed), tag=tag, bb_dir=bb_dir, paths=paths, phase_a=phase_a,
                        runner=runner, corpus=corpus, cohort_hc=cohort_hc,
                        rows_same8=rows_same8, val_cohort=val_cohort,
                        scorers=scorers, args=run_args, log=log,
                    )
                )
        finally:
            corpus.close()

        point_rows = [r for e in per_seed for r in e["point_rows"]]
        macro_rows = [r for e in per_seed for r in e["macro_rows"]]
        boot_rows = [r for e in per_seed for r in e["boot_rows"]]
        alpha_rows = [r for e in per_seed for r in e["alpha_rows"]]
        regret_rows = [r for e in per_seed for r in e["regret_rows"]]
        calib_rows = [r for e in per_seed for r in e["calib_rows"]]
        anchor_rows = [r for e in per_seed for r in e["anchor_rows"]]

        gate = _backbone_gate(per_seed, boot_rows, macro_rows, alpha_rows, runner)
        gates[tag] = gate

        bb_out = out_root / tag
        _write_csv(
            bb_out / "point_metrics.csv", point_rows,
            ("level", "K", "seed", "model", "n", "n_images", "b3_accuracy") + POINT_FIELDS,
        )
        _write_csv(
            bb_out / "macro_metrics.csv", macro_rows,
            ("seed", "model", "macro_auroc", "macro_e_aurc", "macro_rer50", "macro_rer80"),
        )
        _write_csv(
            bb_out / "bootstrap_pairs.csv", boot_rows,
            ("level", "K", "seed", "model_a", "model_b", "metric", "diff", "ci_low", "ci_high",
             "mean_a", "mean_b", "n", "n_clusters", "n_replicates", "ci_level"),
        )
        _write_csv(
            bb_out / "alpha_diagnostics.csv", alpha_rows,
            ("level", "seed", "n", "a_mean", "a_median", "a_p10", "a_p90",
             "alpha_mean", "alpha_median"),
        )
        _write_csv(
            bb_out / "regret.csv", regret_rows,
            ("level", "seed", "model", "oracle_auroc", "regret"),
        )
        _write_csv(
            bb_out / "calibration.csv", calib_rows,
            ("level", "seed", "model", "mean_pred", "accuracy", "ece_adaptive",
             "brier_binary", "nll_binary"),
        )
        _write_csv(
            bb_out / "dose_anchor.csv", anchor_rows,
            ("backbone", "seed", "level", "auroc_e_random", "auroc_r2_v2g", "abs_delta",
             "candidate_sha256"),
        )
        figures = _write_figures(tag, point_rows, macro_rows, alpha_rows, gate, bb_out, log)
        gate_payload = {
            **gate,
            "artifact": f"v2m_m3_conf_gate_{tag}",
            "backbone": tag,
            "backbone_dir": bb_dir,
            "label": "CONFIRMATORY (Amendment V2-M3.1; frozen before the run)",
            "figures": [str(p) for p in figures],
            "repro_checks": {
                "e_random_vs_phase_a_k5_max_delta": max(
                    float(e["anchor_phase_a_max_delta"]) for e in per_seed
                ),
                "e_random_vs_dose_r2_max_delta": max(
                    float(e["anchor_dose_max_delta"]) for e in per_seed
                ),
                "val_stop_random_k10_max_delta": max(
                    float(e["val_stop_max_delta"]) for e in per_seed
                ),
                "tolerances": {
                    "phase_a_k5_store": ANCHOR_STORE_TOL,
                    "dose_rescore": ANCHOR_RESCORE_TOL,
                },
            },
            "runtime_sec": round(time.perf_counter() - started, 1),
        }
        _write_json(bb_out / "gate.json", gate_payload)
        log(
            f"[M3-CONF {tag}] delta_macro={float(gate['delta_macro']['vs_static']['delta']):+.4f} "
            f"CI=[{float(gate['delta_macro']['vs_static']['ci_low']):+.4f},"
            f"{float(gate['delta_macro']['vs_static']['ci_high']):+.4f}] "
            f"primary={gate['primary_gate']['pass']} safety={gate['extreme_safety']['pass']} "
            f"alpha_gap={gate['alpha_trend']['alpha_m8_minus_m0']:+.4f} "
            f"-> pass={gate['pass']}"
        )

    verdict = _cross_backbone_verdict(gates)
    _write_json(out_root / "cross_backbone_verdict.json", verdict)

    metadata = {
        "artifact": "v2m_m3_conf_metadata",
        "amendment": "V2-M3.1",
        "label": "CONFIRMATORY",
        "branch": _git(["branch", "--show-current"]),
        "head": _git(["rev-parse", "HEAD"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "started_utc": started_utc,
        "runtime_sec": round(time.perf_counter() - started, 1),
        "backbones": list(tags),
        "seeds": list(seeds),
        "levels": list(LEVELS),
        "mixer_fit_levels": list(MIXER_FIT_LEVELS),
        "k_level": 10,
        "bootstrap": {
            "replicates": int(reps), "seed": int(args.bootstrap_seed), "ci": float(args.ci),
            "metrics": list(runner.BOOT_METRICS), "shared_draws_across_severity": True,
        },
        "gates": {
            "delta_macro_min": GATE_DELTA_MACRO_MIN,
            "extreme_safety_min": GATE_EXTREME_SAFETY_MIN,
            "e_aurc_rel_min": GATE_EAURC_REL_MIN,
            "rer50_gain_pp_min": GATE_RER50_GAIN_PP_MIN,
        },
        "sources": {
            "amendment_sha": _sha256_file(M3_DIR / "amendment_v2m31.json"),
            "mixer_module_sha": _sha256_file(_REPO / "src/ccg/mixture/mixer.py"),
            "bootstrap_module_sha": _sha256_file(_REPO / "src/ccg/mixture/bootstrap.py"),
            "g4_freeze_sha": _sha256_file(FREEZE_PATH),
            "phase_a_models_sha": {
                f"{t}/seed{s}": _sha256_file(PHASE_A_ROOT / t / f"seed_{s}" / "models.pkl")
                for t in tags for s in seeds
            },
        },
        "verdict": verdict["verdict"],
        "read_only_experts": True,
        "no_training_except_frozen_curriculum_protocol": True,
    }
    _write_json(out_root / "metadata.json", metadata)

    log("=" * 78)
    log("V2-M3.1 cross-backbone confirmatory report")
    for tag in tags:
        gate = gates[tag]
        log(
            f"  {tag}: delta_macro={float(gate['delta_macro']['vs_static']['delta']):+.4f} "
            f"CI=[{float(gate['delta_macro']['vs_static']['ci_low']):+.4f},"
            f"{float(gate['delta_macro']['vs_static']['ci_high']):+.4f}] "
            f"| primary={gate['primary_gate']['pass']} safety={gate['extreme_safety']['pass']} "
            f"| c={gate['fits']['static_c']['mean']:.4f} "
            f"beta={gate['fits']['adaptive_beta']['mean']:.4f} "
            f"tau={gate['fits']['adaptive_tau']['mean']:.4f} "
            f"alpha(m8-m0)={gate['alpha_trend']['alpha_m8_minus_m0']:+.4f} "
            f"| pass={gate['pass']}"
        )
    log(f"  verdict: {verdict['verdict']} ({verdict['n_pass']}/{verdict['n_backbones']})")
    log("=" * 78)
    log(
        "V2M_M3_CONF_COMPLETE "
        f"verdict={verdict['verdict'].replace(' ', '_')} "
        + " ".join(
            f"{tag}:dmacro={float(gates[tag]['delta_macro']['vs_static']['delta']):+.4f}"
            for tag in tags
        )
        + f" runtime={time.perf_counter() - started:.1f}s out={out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
