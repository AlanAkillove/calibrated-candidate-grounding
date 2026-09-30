#!/usr/bin/env python
"""V2-G G4 reliability pipeline (R1 Stats / R2 Stats+V2-Semantic Logistic).

Backbone-general replication of the V1 hard-semantic mechanism.  This driver
implements the frozen G4 protocol (see
``results/v2_backbone_generalization/g4_protocol_freeze.json``).

Sub-phases (run separately, so the long embedding-store build can be a
background job):

    build-stores : materialise the per-backbone full-cohort (z_q, z_i) archives
                   for K in {5, 10} from the frozen V2 feature cache (reuses
                   ``ccg.semantic.data.build_embedding_store``; no CLIP forward).
    A            : fit R1 / R2 fresh per (backbone, scorer seed) on Random
                   K5/K10 ``reliability_train`` (select C on ``reliability_tune``)
                   and report the Random-cohort delta (engineering check).

Every frozen choice (feature list / order, train-only normalisation, C grid,
selection split, per-backbone/seed corrected T, gate thresholds) is taken from
the freeze file -- this module never re-tunes any of them.

Usage
-----
    python -u scripts/run_v2g_reliability.py --phase build-stores \
        --device cuda --log-file logs_v2g_phaseA.txt
    python -u scripts/run_v2g_reliability.py --phase A --log-file logs_v2g_phaseA.txt

Exit 0 on success; a non-fatal per-cell guard (grounding failure, alignment)
raises immediately rather than writing a suspect artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import data as rdata  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.reliability import models as rmodels  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.v2 import semantic_features as v2feat  # noqa: E402

__all__ = ["build_parser", "main"]

# ---------------------------------------------------------------------------
# frozen configuration (all values cross-checked against g4_protocol_freeze.json)
# ---------------------------------------------------------------------------
KS: Tuple[int, ...] = (5, 10)
TRAIN_KS: Tuple[int, ...] = (5, 10)
C_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0)
SELECTION_TIE: float = 0.002
SEEDS: Tuple[int, ...] = (1, 2, 3)
SPLIT_SEED: int = rdata.DEFAULT_SPLIT_SEED      # 20260928 (frozen A6.2)
SPLIT_FRAC: float = rdata.DEFAULT_TRAIN_FRAC    # 0.70

MANIFESTS = Path("cache/manifests")
BANK = Path("cache/proposals.h5")
REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
IMAGE_SIZES = Path("cache/image_sizes.npz")
PHASE05_DIR = Path("results/phase05_score_sufficiency/features")

V2_ROOT = Path("results/v2_backbone_generalization")
FREEZE_PATH = V2_ROOT / "g4_protocol_freeze.json"
EMB_ROOT = Path("cache/v2_semantic")
A_OUT = V2_ROOT / "g4_phaseA"

STATS17_NAMES: Tuple[str, ...] = tuple(rfeat.stat_feature_names())
SEM14_NAMES: Tuple[str, ...] = tuple(v2feat.V2_PRIMARY_SEMANTIC_NAMES)
BACKBONES: Tuple[Tuple[str, str], ...] = (
    ("b1", "openclip_b16"),
    ("b2", "siglip_b16"),
)


# ---------------------------------------------------------------------------
# freeze helpers
# ---------------------------------------------------------------------------
def _load_freeze() -> Dict[str, Any]:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def _backbone_paths(freeze: Mapping[str, Any], tag: str) -> Dict[str, Any]:
    for key, bb in freeze["backbones"].items():
        if bb["tag"] == tag:
            return {"key": key, "features_root": Path(bb["features_root"]),
                    "b3_root": Path(bb["b3_root"]), "feature_dim": int(bb["feature_dim"])}
    raise KeyError(f"backbone tag {tag!r} absent from freeze file")


def _corrected_T(b3_root: Path, seed: int) -> float:
    """Read the per-(backbone,seed) corrected global T from its eval metadata."""
    em = json.loads((b3_root / f"seed_{seed}" / "eval_metadata.json").read_text(encoding="utf-8"))
    t = float(em["temperature_corrected"])
    if not (np.isfinite(t) and t > 0):
        raise ValueError(f"{b3_root} seed_{seed}: invalid corrected T {t}")
    if not bool(em.get("corrected_fit", {}).get("interior", em.get("corrected_interior", True))):
        # interior already verified in G2; keep a loud guard against regressions
        raise ValueError(f"{b3_root} seed_{seed}: corrected T fit is not interior")
    return t


# ---------------------------------------------------------------------------
# small numeric helpers (mirror the frozen A7 selection primitives)
# ---------------------------------------------------------------------------
def _auroc(confidence: np.ndarray, correct: np.ndarray) -> Optional[float]:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.size == 0 or np.unique(corr).size < 2:
        return None
    return float(auroc_correct(conf, corr))


def _tune_mean_auroc(conf_by_k: Mapping[int, np.ndarray], correct_by_k: Mapping[int, np.ndarray],
                     masks: Mapping[int, np.ndarray]) -> Optional[float]:
    values: List[float] = []
    for k in TRAIN_KS:
        mask = masks[k]
        value = _auroc(np.asarray(conf_by_k[k])[mask], np.asarray(correct_by_k[k])[mask])
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def _select_by_tie(values: Sequence[float], configs: Sequence[Any]) -> int:
    best = int(np.argmax(np.asarray(values, dtype=np.float64)))
    for idx in range(best):
        if values[best] - values[idx] < SELECTION_TIE:
            return idx
    return best


# ---------------------------------------------------------------------------
# store building (phase build-stores)
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


def phase_build_stores(freeze: Mapping[str, Any], *, device: str, overwrite: bool,
                       log: Callable[[str], None]) -> None:
    for tag, bb_dir in BACKBONES:
        paths = _backbone_paths(freeze, bb_dir)
        out_root = EMB_ROOT / bb_dir
        out_root.mkdir(parents=True, exist_ok=True)
        corpus = B3Corpus(
            paths["features_root"], MANIFESTS, REFS, BANK,
            image_sizes_path=IMAGE_SIZES, ks=KS, regime="random", preload=True,
        )
        log(f"[store] {tag}: B3Corpus preloaded feature_dim={int(corpus.feature_dim)}")
        for k in KS:
            sdata.build_embedding_store(
                k, out_root=out_root, features_root=paths["features_root"],
                manifests_root=MANIFESTS, refs=REFS, bank_path=BANK,
                image_sizes_path=IMAGE_SIZES, phase05_features_dir=PHASE05_DIR,
                b3_root=paths["b3_root"], corpus=corpus, overwrite=overwrite, log=log,
            )
        corpus.close()
    log("[store] all embedding stores present")


# ---------------------------------------------------------------------------
# aligned per-(backbone,seed,K) feature assembly
# ---------------------------------------------------------------------------
def _aligned_scoring(bb_dir: str, paths: Mapping[str, Any], seed: int, k: int,
                     log: Callable[[str], None]) -> Dict[str, np.ndarray]:
    """Load the store + scorer aligned 1:1 on canonical ``sentence_id`` rows."""
    store = sdata.load_embedding_store(k, out_root=EMB_ROOT / bb_dir)
    ref = sdata.load_scorer_canonical(f"b3_seed{seed}", k, b3_root=paths["b3_root"])
    pos = np.searchsorted(ref.sentence_id, store.sentence_id)
    if pos.size != len(store) or not np.array_equal(ref.sentence_id[pos], store.sentence_id):
        raise AssertionError(f"{bb_dir}/seed_{seed}/K{k}: store ids are not a subset of scorer rows")
    target_local = np.asarray(ref.target_local[pos], dtype=np.int32)
    if not np.all(target_local == 0):
        raise AssertionError(
            f"{bb_dir}/seed_{seed}/K{k}: scorer target_local is not column 0 -- "
            "candidate layout differs from the embedding store (alignment unsafe)")
    scores = np.asarray(ref.scores[pos], dtype=np.float64)
    z_q = np.asarray(store.z_q, dtype=np.float64)
    z_i = np.asarray(store.z_i, dtype=np.float64)
    if z_i.shape != (len(store), k, z_q.shape[1]):
        raise AssertionError(f"{bb_dir}/seed_{seed}/K{k}: z_i shape {z_i.shape} unexpected")
    return {
        "scores": scores, "z_q": z_q, "z_i": z_i,
        "image_id": np.asarray(store.image_id, dtype=np.int64),
        "eval_split": np.asarray(store.eval_split),
        "correct": np.asarray(ref.correct[pos], dtype=bool),
    }


def _model_vs_stats(seed_rows: Mapping[int, Dict[str, np.ndarray]], T: float,
                    log: Callable[[str], None]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """stats17_raw [n,17] and sem14_raw [n,14] per K, using the seed's own corrected T."""
    stats_raw = {k: np.asarray(rfeat.stat_features(rows["scores"], temperature=T), dtype=np.float64)
                 for k, rows in seed_rows.items()}
    sem_raw = {k: v2feat.v2_primary_semantic_stats(rows["z_q"], rows["z_i"], rows["scores"])
               for k, rows in seed_rows.items()}
    for k in seed_rows:
        if stats_raw[k].shape[1] != len(STATS17_NAMES):
            raise AssertionError(f"stats width {stats_raw[k].shape[1]} != {len(STATS17_NAMES)}")
        if sem_raw[k].shape[1] != len(SEM14_NAMES):
            raise AssertionError(f"semantic width {sem_raw[k].shape[1]} != {len(SEM14_NAMES)}")
    return stats_raw, sem_raw


def _standardised(raw_by_k: Mapping[int, np.ndarray], train_masks: Mapping[int, np.ndarray],
                  keys: Sequence[str]) -> Tuple[Dict[int, np.ndarray], Any]:
    """Train-only standardisation; returns the block family + the (persisted) fit."""
    combined = np.vstack([np.asarray(raw_by_k[k])[train_masks[k]] for k in TRAIN_KS])
    fit = rfeat.normalize_fit(combined, fit_rows=np.arange(combined.shape[0]), keys=tuple(keys))
    std = {k: np.asarray(rfeat.normalize_apply(np.asarray(raw_by_k[k]), fit), dtype=np.float64)
           for k in raw_by_k}
    return std, fit


def _fit_logistic(std_by_k: Mapping[int, np.ndarray], correct_by_k: Mapping[int, np.ndarray],
                  train_masks: Mapping[int, np.ndarray], tune_masks: Mapping[int, np.ndarray],
                  *, name: str) -> Dict[str, Any]:
    """C-grid logistic fit with A7 selection (fit R1/R2 identically)."""
    train_x = np.vstack([np.asarray(std_by_k[k])[train_masks[k]] for k in TRAIN_KS])
    train_y = np.concatenate([np.asarray(correct_by_k[k], dtype=np.float64)[train_masks[k]] for k in TRAIN_KS])
    scored: List[Tuple[float, float]] = []
    fitted: Dict[float, Any] = {}
    for c_value in C_GRID:
        clf = rmodels.LogisticModel(C=float(c_value))
        clf.fit(train_x, train_y)
        conf_k = {k: clf.predict_proba(std_by_k[k]) for k in std_by_k}
        value = _tune_mean_auroc(conf_k, correct_by_k, tune_masks)
        if value is None:
            raise RuntimeError(f"{name}: tune AUROC undefined for C={c_value}")
        scored.append((float(c_value), float(value)))
        fitted[float(c_value)] = clf
    best_idx = _select_by_tie([v for _, v in scored], [c for c, _ in scored])
    best_c, best_value = scored[best_idx]
    model = fitted[best_c]
    conf_all = {k: model.predict_proba(std_by_k[k]) for k in std_by_k}
    coef, intercept = model.coefficients()
    return {
        "model": model, "chosen_C": best_c, "tune_mean_auroc": best_value,
        "grid": [c for c, _ in scored], "values": [v for _, v in scored],
        "conf": conf_all, "n_params": int(model.n_parameters()),
        "coef": np.asarray(coef, dtype=np.float64), "intercept": float(intercept),
    }


def _pooled_test_row(conf_by_k: Mapping[int, np.ndarray], correct_by_k: Mapping[int, np.ndarray],
                     test_mask: Mapping[int, np.ndarray], k: int, model: str) -> Dict[str, Any]:
    conf = np.asarray(conf_by_k[k])[test_mask[k]]
    corr = np.asarray(correct_by_k[k])[test_mask[k]]
    return {"K": k, "model": model, "n": int(corr.size),
            "b3_accuracy": float(np.mean(corr)) if corr.size else None,
            "auroc_correct": _auroc(conf, corr)}


def phase_a(freeze: Mapping[str, Any], *, overwrite: bool, log: Callable[[str], None]) -> None:
    A_OUT.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    for tag, bb_dir in BACKBONES:
        paths = _backbone_paths(freeze, bb_dir)
        for seed in SEEDS:
            T = _corrected_T(paths["b3_root"], seed)
            seed_rows = {k: _aligned_scoring(bb_dir, paths, seed, k, log) for k in KS}
            eval_split = seed_rows[KS[0]]["eval_split"]
            image_id = seed_rows[KS[0]]["image_id"]
            split = rdata.build_reliability_split(eval_split, image_id, seed=SPLIT_SEED,
                                                  train_frac=SPLIT_FRAC)
            train_masks = {k: split.row_mask(seed_rows[k]["eval_split"], seed_rows[k]["image_id"],
                                             kind="reliability_train") for k in KS}
            tune_masks = {k: split.row_mask(seed_rows[k]["eval_split"], seed_rows[k]["image_id"],
                                            kind="reliability_tune") for k in KS}
            test_masks = {k: np.isin(seed_rows[k]["eval_split"], ("testA", "testB")) for k in KS}
            correct_by_k = {k: seed_rows[k]["correct"] for k in KS}

            stats_raw, sem_raw = _model_vs_stats(seed_rows, T, log)
            stats_std, stats_fit = _standardised(stats_raw, train_masks, STATS17_NAMES)
            sem_std, sem_fit = _standardised(sem_raw, train_masks, SEM14_NAMES)
            r2_std = {k: np.hstack([stats_std[k], sem_std[k]]) for k in KS}

            r1 = _fit_logistic(stats_std, correct_by_k, train_masks, tune_masks, name=f"{tag}s{seed}:R1")
            r2 = _fit_logistic(r2_std, correct_by_k, train_masks, tune_masks, name=f"{tag}s{seed}:R2")

            out_dir = A_OUT / tag / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "models.pkl", "wb") as fh:
                pickle.dump({"stats_fit": stats_fit, "sem_fit": sem_fit,
                             "R1_model": r1["model"], "R2_model": r2["model"],
                             "R1_C": r1["chosen_C"], "R2_C": r2["chosen_C"], "T": T}, fh)

            rows: List[Dict[str, Any]] = []
            for k in KS:
                rows.append({**_pooled_test_row(r1["conf"], correct_by_k, test_masks, k, "R1_stats"),
                             "backbone": tag, "seed": seed, "T": T})
                rows.append({**_pooled_test_row(r2["conf"], correct_by_k, test_masks, k, "R2_stats_sem"),
                             "backbone": tag, "seed": seed, "T": T})
            a5 = next(r["auroc_correct"] for r in rows if r["K"] == 5 and r["model"] == "R1_stats")
            a5b = next(r["auroc_correct"] for r in rows if r["K"] == 5 and r["model"] == "R2_stats_sem")
            delta_rand = None if (a5 is None or a5b is None) else a5b - a5
            for r in rows:
                if r["K"] == 5:
                    r["delta_auroc_rand_K5"] = delta_rand
            summary_rows.extend(rows)

            manifest = {
                "backbone": tag, "seed": seed, "feature_dim": paths["feature_dim"],
                "T_corrected": T,
                "R1": {"chosen_C": r1["chosen_C"], "tune_mean_auroc": r1["tune_mean_auroc"],
                       "n_params": r1["n_params"], "grid": r1["grid"], "values": r1["values"]},
                "R2": {"chosen_C": r2["chosen_C"], "tune_mean_auroc": r2["tune_mean_auroc"],
                       "n_params": r2["n_params"], "grid": r2["grid"], "values": r2["values"]},
                "stats_fit_rows": int(sum(int(train_masks[k].sum()) for k in TRAIN_KS)),
                "feature_names": {"stats17": list(STATS17_NAMES), "sem14": list(SEM14_NAMES)},
                "grounding_failure_K5": bool(a5 is not None and _acc5(correct_by_k, test_masks) < 0.30),
            }
            (out_dir / "model_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"[A] {tag} seed_{seed}: R1 C={r1['chosen_C']} R2 C={r2['chosen_C']} "
                f"Delta_rand(K5 pooled-test)={delta_rand if delta_rand is None else round(delta_rand, 4)}")

    _write_summary(summary_rows)
    log(f"[A] wrote {A_OUT / 'phaseA_summary.csv'} ({len(summary_rows)} rows)")


def _acc5(correct_by_k: Mapping[int, np.ndarray], test_masks: Mapping[int, np.ndarray]) -> float:
    m = test_masks[5]
    return float(np.mean(np.asarray(correct_by_k[5])[m])) if m.any() else 1.0


def _write_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ("backbone", "seed", "K", "model", "n", "b3_accuracy", "auroc_correct",
              "delta_auroc_rand_K5", "T")
    path = A_OUT / "phaseA_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({f: r.get(f) for f in fields})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("build-stores", "A", "all"), required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true", help="rebuild embedding stores even if present")
    p.add_argument("--log-file", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = _make_logger(Path(args.log_file) if args.log_file else None)
    freeze = _load_freeze()
    if args.phase in ("build-stores", "all"):
        phase_build_stores(freeze, device=args.device, overwrite=args.overwrite, log=log)
    if args.phase in ("A", "all"):
        phase_a(freeze, overwrite=args.overwrite, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
