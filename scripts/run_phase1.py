#!/usr/bin/env python
"""Run the Phase 1 **candidate semantic information sufficiency audit** (protocol A7).

Usage
-----
::

    python scripts/run_phase1.py \
        --out results/phase1_semantic_sufficiency \
        --embeddings cache/semantic_phase1 \
        --phase05 results/phase05_score_sufficiency \
        --features cache/features --manifests cache/manifests \
        --bank cache/proposals.h5 \
        --refs "data/raw/refcoco+/refcoco+/refs(unc).p" \
        --bootstrap-replicates 5000 --log-file logs_phase1_driver.txt

Everything is frozen by Amendment A7 (docs/research_protocol.md): the B3 ranking
never changes, reliability models only *read* the frozen scores/rankings, the
A6.2 split is reused verbatim, K in {5,10} is the only training/tuning region and
the A7.6 gate semantics are evaluated on the pooled OOD cells.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.metrics.discrimination import auroc_correct  # noqa: E402
from ccg.reliability import data as rdata  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.reliability import models as rmodels  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import evaluate as seval  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import models as smodels  # noqa: E402

__all__ = ["build_parser", "main"]

# ---------------------------------------------------------------------------
# frozen configuration (protocol A7)
# ---------------------------------------------------------------------------
B3_SEEDS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
KS: Tuple[int, ...] = (5, 10, 20, 50)
TRAIN_KS: Tuple[int, ...] = (5, 10)
OOD_KS: Tuple[int, ...] = (20, 50)
#: splits whose metrics are reported (val_calib is the train/tune region).
REPORT_SPLITS: Tuple[str, ...] = ("val_select", "testA", "testB")
POOLED = "__pooled_test__"
E1_NAME = "e1_semantic_stats"
E1B_NAME = "e1b_stats_semantic"
E2_NAMES: Tuple[str, ...] = ("e2_score", "e2_semantic", "e2_combined")
E3_FULL, E3_TOP5, E3_LOGK = "e3_full", "e3_top5", "e3_logk"
E3_NAMES: Tuple[str, ...] = (E3_FULL, E3_TOP5, E3_LOGK)
E1_NAMES: Tuple[str, ...] = (E1_NAME, E1B_NAME)
SCORE_ONLY_NAMES: Tuple[str, ...] = ("msp", "stats_logistic")
SEMANTIC_NAMES: Tuple[str, ...] = E1_NAMES + E2_NAMES + E3_NAMES
#: Models eligible as the A7.6 "best semantic model": only those that actually
#: consume candidate/query semantic embeddings (E2-score is a score-stats ablation
#: and stays out of the gate, per instruction section 28).
GATE_SEMANTIC_NAMES: Tuple[str, ...] = (E1_NAME, E1B_NAME, "e2_semantic", "e2_combined",
                                        E3_FULL, E3_TOP5)
#: cross-K bootstrap headline models (protocol A7.5 runtime scope).
HEADLINE_MODELS: Tuple[str, ...] = ("msp", "stats_logistic", E1B_NAME, "e2_combined", E3_FULL, E3_TOP5)
#: frozen model-vs-model pairs ``(semantic, score_only)`` (P1-P4 + the MSP extras of A7.5);
#: a positive absolute ``diff`` means the *semantic* side is better.
PAIRS: Tuple[Tuple[str, str], ...] = (
    (E1B_NAME, "stats_logistic"),          # P1
    ("e2_combined", "stats_logistic"),     # P2
    (E3_FULL, "stats_logistic"),           # P3
    (E3_FULL, "e2_combined"),              # P4 (top-competitor vs full set)
    (E1B_NAME, "msp"),
    ("e2_combined", "msp"),
    (E3_FULL, "msp"),
)
#: hyper-parameter grids (protocol A7.3; TieBreak: |dAUROC|<0.002 -> simpler).
C_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0)
LR_GRID: Tuple[float, ...] = (1e-4, 3e-4, 1e-3)
TINY_C_GRID: Tuple[float, ...] = (1.0,)
TINY_LR_GRID: Tuple[float, ...] = (3e-4,)
SELECTION_TIE: float = 0.002
#: metric columns of one aggregate row (mirrors phase 0.5's reliability_metrics.csv).
AGG_FIELDS: Tuple[str, ...] = (
    "scorer", "model", "information", "params", "eval_split", "K", "n",
    "tune_mean_auroc", "selected_hp", "auroc_correct", "aurc", "aurc_oracle",
    "e_aurc", "rer_at_50", "rer_at_80", "rer_at_90", "rer_at_95",
    "ece_adaptive", "brier_binary", "nll_binary",
)
#: definitions of the 16 E1 statistics (protocol A7.3, human-readable).
E1_DEFINITIONS: Mapping[str, str] = {
    "clip_top1": "cos(z_q, z_(1)) -- B3 rank 1",
    "clip_top2": "cos(z_q, z_(2))",
    "clip_margin12": "clip_top1 - clip_top2",
    "clip_entropy": "H(softmax(a / 0.01)), a_i = cos(z_q, z_i), nats (CLIP ViT-B/32 logit scale)",
    "clip_normH": "1 - clip_entropy / log K",
    "clip_rank_top1": "average rank of the B3 winner under CLIP-cosine order / K",
    "cand_vmax": "max_{j != t} cos(z_t, z_j)",
    "cand_vmean": "mean_{j != t} cos(z_t, z_j)",
    "cand_vstd": "std_{j != t} cos(z_t, z_j) (ddof=0)",
    "cand_top12_sim": "cos(z_(1), z_(2))",
    "cand_top15_mean": "mean_{r=2..5} cos(z_(1), z_(r))",
    "density_070": "share of j != t with cos(z_t, z_j) > 0.70",
    "density_080": "share of j != t with cos(z_t, z_j) > 0.80",
    "q_top3": "cos(z_q, z_(3))",
    "q_margin13": "clip_top1 - cos(z_q, z_(3))",
    "cand_top13_sim": "cos(z_(1), z_(3))",
}


# ---------------------------------------------------------------------------
# small helpers
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
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_logger(log_file: Optional[Path]) -> Callable[[str], None]:
    handle = None
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handle = log_file.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = str(message)
        print(line, flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return log


def _bar(desc: str, total: int):  # pragma: no cover - cosmetic
    from tqdm import tqdm

    return tqdm(total=total, desc=desc, unit="cell", disable=None)


def _auroc(confidence: np.ndarray, correct: np.ndarray) -> Optional[float]:
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    corr = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.size == 0 or np.unique(corr).size < 2:
        return None
    return float(auroc_correct(conf, corr))


def _tune_mean_auroc(conf_by_k: Mapping[int, np.ndarray], correct_by_k: Mapping[int, np.ndarray],
                     masks: Mapping[int, np.ndarray]) -> Optional[float]:
    """Primary selection metric (A7.4): mean AUROC_correct over tune K5/K10."""
    values = []
    for k in TRAIN_KS:
        mask = masks[k]
        value = _auroc(np.asarray(conf_by_k[k])[mask], np.asarray(correct_by_k[k])[mask])
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def _select_by_tie(values: Sequence[float], configs: Sequence[Any]) -> int:
    """Index of the best config; within ``SELECTION_TIE`` prefer the earlier (simpler) one."""
    best = int(np.argmax(np.asarray(values, dtype=np.float64)))
    for idx in range(best):
        if values[best] - values[idx] < SELECTION_TIE:
            return idx
    return best


def _concat_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)])


def _gather_rows(block: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """``block[n, K, d] -> [n, m, d]`` by taking ``indices[n, m]`` per row."""
    n, _, d = block.shape
    rows = np.arange(n)[:, None]
    return np.asarray(block[rows, indices], dtype=np.float32)


def _pack_e3(per_k: Mapping[int, Mapping[str, np.ndarray]], ks: Sequence[int]) -> Dict[str, np.ndarray]:
    """Pack per-K native arrays into padded tensors for the DeepSets model.

    ``per_k[k]`` holds ``z_q [n,512], z_i [n,c,512], score [n,c], prob [n,c]``
    (all rows of that K; ``c`` is the block width - ``k`` for the full variant,
    the fixed 5 for the top-5 variant).  Rows are stacked in ``ks`` order and
    padded to ``width = max_c`` with zero blocks + mask 0; ``log_k`` carries the
    *native* ``log K`` of each row (not the packed width).
    """
    k_list = [int(k) for k in ks]
    widths = {k: int(np.asarray(per_k[k]["z_i"]).shape[1]) for k in k_list}
    width = max(widths.values())
    z_q_parts, z_i_parts, mask_parts, score_parts, prob_parts, log_k_parts = [], [], [], [], [], []
    for k in k_list:
        data = per_k[k]
        n = int(data["z_q"].shape[0])
        cols = widths[k]
        for key in ("z_q", "z_i", "score", "prob"):
            block = np.asarray(data[key])
            if key == "z_i":
                expected = (n, cols, sdata.FEATURE_DIM)
            elif key == "z_q":
                expected = (n, sdata.FEATURE_DIM)
            else:
                expected = (n, cols)
            if block.shape != expected:
                raise ValueError(f"_pack_e3: K={k} key {key!r} has shape {block.shape}, expected {expected}")
        z_q_parts.append(np.asarray(data["z_q"], dtype=np.float32))
        z_i = np.zeros((n, width, sdata.FEATURE_DIM), dtype=np.float32)
        z_i[:, :cols, :] = np.asarray(data["z_i"], dtype=np.float32)
        z_i_parts.append(z_i)
        mask = np.zeros((n, width), dtype=bool)
        mask[:, :cols] = True
        mask_parts.append(mask)
        score = np.zeros((n, width), dtype=np.float32)
        score[:, :cols] = np.asarray(data["score"], dtype=np.float32)
        score_parts.append(score)
        prob = np.zeros((n, width), dtype=np.float32)
        prob[:, :cols] = np.asarray(data["prob"], dtype=np.float32)
        prob_parts.append(prob)
        log_k_parts.append(np.full((n,), float(np.log(k)), dtype=np.float32))
    return {
        "z_q": np.concatenate(z_q_parts, axis=0),
        "z_i": np.concatenate(z_i_parts, axis=0),
        "mask": np.concatenate(mask_parts, axis=0),
        "score": np.concatenate(score_parts, axis=0),
        "prob": np.concatenate(prob_parts, axis=0),
        "log_k": np.concatenate(log_k_parts, axis=0),
    }


def _e2_inputs_for_k(k: int, store: sdata.EmbeddingStore, scores: np.ndarray,
                     temperature: float) -> Dict[str, np.ndarray]:
    """E2 input mapping of one K (native rows, no packing, float16 views).

    The embedding blocks stay float16 (the model casts per batch), so the four
    K-block families together cost a few hundred MiB instead of ~1.5 GiB.
    """
    order = sfeat.b3_order(scores)
    z_i = store.z_i  # float16 view
    rows = np.arange(order.shape[0])[:, None]
    return {
        "z_q": store.z_q,
        "z_top1": np.ascontiguousarray(z_i[rows, order[:, :1]])[:, 0, :],
        "z_top2": np.ascontiguousarray(z_i[rows, order[:, 1:2]])[:, 0, :],
        "score_stats": np.asarray(sdata.score_stat_block(scores, temperature=temperature), dtype=np.float32),
    }


def _predict_e3_chunked(model: Any, per_k: Mapping[int, Mapping[str, np.ndarray]], k: int,
                        *, chunk: int = 4096) -> np.ndarray:
    """Predict one K in row chunks so the float32 packed block stays ~0.4 GiB."""
    n = int(np.asarray(per_k[k]["z_q"]).shape[0])
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, int(chunk)):
        stop = min(start + int(chunk), n)
        packed = _pack_e3({k: {key: np.asarray(value)[start:stop] for key, value in per_k[k].items()}}, (k,))
        out[start:stop] = model.predict_proba(packed)
        del packed
    return out


def _e3_native_for_k(k: int, store: sdata.EmbeddingStore, scores: np.ndarray,
                     temperature: float, *, top5: bool) -> Dict[str, np.ndarray]:
    """Native (unpadded) E3 arrays of one K; ``top5`` keeps the five best scores.

    The candidate block stays a **float16 view** into the store (no float32
    copy); the per-call float32 cast happens inside :func:`_pack_e3`, whose
    callers must free the packed tensors right after use (K50 float32 is ~2 GiB).
    """
    z_i = store.z_i  # float16 view, no copy
    prob = np.asarray(sdata.corrected_probs(scores, temperature=temperature), dtype=np.float32)
    if top5:
        idx = sfeat.top5_indices(scores)
        rows = np.arange(scores.shape[0])[:, None]
        z_i = np.ascontiguousarray(z_i[rows, idx])  # float16 [n, 5, 512]
        score = np.asarray(scores, dtype=np.float32)[rows, idx]
        prob = prob[rows, idx]
    else:
        score = np.asarray(scores, dtype=np.float32)
    return {
        "z_q": store.z_q,
        "z_i": z_i,
        "score": score,
        "prob": prob,
    }


# ---------------------------------------------------------------------------
# stage 1: load frozen scores / embeddings / split
# ---------------------------------------------------------------------------
def _load_scores(b3_root: Path, log: Callable[[str], None]) -> Dict[str, Dict[int, rdata.ScorerScores]]:
    per_seed: Dict[str, Dict[int, rdata.ScorerScores]] = {}
    for scorer in B3_SEEDS:
        per_k: Dict[int, rdata.ScorerScores] = {}
        for k in KS:
            per_k[k] = sdata.load_scorer_canonical(scorer, k, b3_root=b3_root)
        ref = per_k[KS[0]].sentence_id
        for k in KS[1:]:
            if not np.array_equal(per_k[k].sentence_id, ref):
                raise AssertionError(f"{scorer}: K={k} row universe differs from K={KS[0]}")
            for field in ("image_id", "eval_split", "target_local"):
                if not np.array_equal(getattr(per_k[k], field), getattr(per_k[KS[0]], field)):
                    raise AssertionError(f"{scorer}: K={k} field {field} differs across K")
        per_seed[scorer] = per_k
        log(f"[phase1] {scorer}: loaded {ref.size} canonical rows x K={list(KS)}")
    ref_seed = per_seed[B3_SEEDS[0]][KS[0]].sentence_id
    for scorer in B3_SEEDS[1:]:
        if not np.array_equal(per_seed[scorer][KS[0]].sentence_id, ref_seed):
            raise AssertionError(f"{scorer}: row universe differs from {B3_SEEDS[0]}")
    return per_seed


def _load_temperatures(b3_root: Path, log: Callable[[str], None]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for scorer in B3_SEEDS:
        seed = scorer.split("b3_seed")[1]
        payload = json.loads((Path(b3_root) / f"seed_{seed}" / "eval_metadata.json").read_text(encoding="utf-8"))
        out[scorer] = float(payload["temperature_corrected"])
        log(f"[phase1] {scorer}: corrected global T = {out[scorer]:.6f}")
    return out


def _load_embeddings(emb_root: Path, scores: Dict[str, Dict[int, rdata.ScorerScores]],
                     log: Callable[[str], None]) -> Dict[int, sdata.EmbeddingStore]:
    stores: Dict[int, sdata.EmbeddingStore] = {}
    anchor = scores[B3_SEEDS[0]]
    for k in KS:
        store = sdata.load_embedding_store(k, out_root=emb_root)
        ref = anchor[k]
        if not np.array_equal(store.sentence_id, ref.sentence_id):
            raise AssertionError(f"embeddings K={k}: sentence_id differs from the frozen cohort")
        if not np.array_equal(store.image_id, ref.image_id):
            raise AssertionError(f"embeddings K={k}: image_id differs from the frozen cohort")
        if not np.array_equal(store.eval_split, ref.eval_split):
            raise AssertionError(f"embeddings K={k}: eval_split differs from the frozen cohort")
        if not np.array_equal(store.target_local, ref.target_local):
            raise AssertionError(f"embeddings K={k}: target_local differs from the frozen cohort")
        if not np.all(store.target_local == 0):
            raise AssertionError(f"embeddings K={k}: target_local is not all-zero (C_K layout broken)")
        stores[k] = store
        log(f"[phase1] embeddings K={k}: {len(store)} rows aligned with the frozen cohort")
    return stores


def _load_phase05_reference(phase05_root: Path, scorer: str, model: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Load one phase-0.5 prediction file, verifying the (split, K) row layout."""
    path = Path(phase05_root) / "predictions" / scorer / f"{model}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"phase 0.5 predictions not found: {path}")
    per_k: Dict[int, List[Tuple[int, int, int, float]]] = {k: [] for k in KS}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            per_k[int(row["K"])].append(
                (int(row["ref_id"]), int(row["image_id"]), int(row["grounding_correct"]), float(row["reliability_score"]))
            )
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for k in KS:
        arr = np.asarray(per_k[k])
        out[k] = {
            "ref_id": arr[:, 0].astype(np.int64),
            "image_id": arr[:, 1].astype(np.int64),
            "correct": arr[:, 2].astype(bool),
            "score": arr[:, 3].astype(np.float64),
        }
    return out


def _verify_against_phase05(local: Mapping[int, np.ndarray], reference: Mapping[int, Mapping[str, np.ndarray]],
                            scores: Mapping[int, rdata.ScorerScores], model: str,
                            log: Callable[[str], None]) -> float:
    """Assert the locally reproduced confidence equals the frozen 0.5 predictions.

    The phase 0.5 files store their rows **split-grouped** (val_select, then
    testA, then testB; canonical order inside each block), so the cohort rows
    are re-ordered to that layout before the comparison.
    """
    split_rank = {"val_select": 0, "testA": 1, "testB": 2}
    worst = 0.0
    for k in KS:
        ref = reference[k]
        sample = scores[k]
        keep = np.isin(sample.eval_split, np.asarray(list(split_rank)))
        idx = np.flatnonzero(keep)
        order = np.argsort(np.asarray([split_rank[str(s)] for s in sample.eval_split[idx]]), kind="stable")
        idx = idx[order]
        if idx.size != ref["ref_id"].size:
            raise AssertionError(
                f"{model} K={k}: phase 0.5 prediction rows {ref['ref_id'].size} != cohort subset {idx.size}"
            )
        if not np.array_equal(sample.ref_id[idx], ref["ref_id"]) or not np.array_equal(
            sample.image_id[idx], ref["image_id"]
        ):
            raise AssertionError(f"{model} K={k}: phase 0.5 prediction rows do not match the cohort order")
        delta = np.abs(np.asarray(local[k], dtype=np.float64)[idx] - ref["score"])
        worst = max(worst, float(delta.max()))
    if worst > 1e-6:
        raise RuntimeError(f"{model}: local recompute deviates from phase 0.5 by {worst:.3e} (> 1e-6)")
    log(f"[phase1] {model}: phase 0.5 reproduction check passed (max |delta| = {worst:.2e})")
    return worst


# ---------------------------------------------------------------------------
# stage 2: E1 semantic statistics
# ---------------------------------------------------------------------------
def _e1_stats(store: sdata.EmbeddingStore, scores: np.ndarray, *, chunk: int = 2048,
              log: Optional[Callable[[str], None]] = None) -> np.ndarray:
    """Chunked :func:`ccg.semantic.features.semantic_stats` over one K store."""
    n = len(store)
    out = np.empty((n, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        z_q = np.asarray(store.z_q[start:stop], dtype=np.float32)
        z_i = np.asarray(store.z_i[start:stop], dtype=np.float32)
        out[start:stop] = sfeat.semantic_stats(z_q, z_i, np.asarray(scores[start:stop], dtype=np.float64))
        if log is not None and (start // chunk) % 5 == 0:
            log(f"[phase1] E1 stats: rows {stop}/{n}")
    return out


def _write_semantic_stats_csv(path: Path, stats: Mapping[str, Mapping[int, np.ndarray]]) -> None:
    rows: List[Dict[str, Any]] = []
    for name in sfeat.SEMANTIC_STAT_NAMES:
        idx = sfeat.SEMANTIC_STAT_NAMES.index(name)
        for scorer, per_k in stats.items():
            for k, matrix in per_k.items():
                column = np.asarray(matrix[:, idx], dtype=np.float64)
                rows.append({
                    "stat": name,
                    "definition": E1_DEFINITIONS[name],
                    "scorer": scorer,
                    "K": int(k),
                    "mean": float(column.mean()),
                    "std": float(column.std(ddof=0)),
                    "min": float(column.min()),
                    "max": float(column.max()),
                    "n": int(column.size),
                })
    _write_csv(path, ("stat", "definition", "scorer", "K", "mean", "std", "min", "max", "n"), rows)


# ---------------------------------------------------------------------------
# stage 3: one scorer seed's full model pipeline
# ---------------------------------------------------------------------------
def _subset_inputs(inputs: Mapping[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value)[idx] for key, value in inputs.items()}


def _row_masks(eval_split: np.ndarray, image_id: np.ndarray, split: rdata.ReliabilitySplit) -> Dict[str, np.ndarray]:
    return {
        "train": split.row_mask(eval_split, image_id, kind="reliability_train"),
        "tune": split.row_mask(eval_split, image_id, kind="reliability_tune"),
        "val_select": eval_split == "val_select",
        "testA": eval_split == "testA",
        "testB": eval_split == "testB",
        POOLED: np.isin(eval_split, ("testA", "testB")),
    }


def _standardised_blocks(raw_by_k: Mapping[int, np.ndarray], train_masks: Mapping[int, np.ndarray],
                         keys: Sequence[str]) -> Dict[int, np.ndarray]:
    """Train-only standardisation of one K-block family (protocol A6.3 / A7.3)."""
    combined = np.vstack([np.asarray(raw_by_k[k])[train_masks[k]] for k in TRAIN_KS])
    fit = rfeat.normalize_fit(combined, fit_rows=np.arange(combined.shape[0]), keys=tuple(keys))
    return {k: np.asarray(rfeat.normalize_apply(np.asarray(raw_by_k[k]), fit), dtype=np.float64) for k in KS}


def _run_seed(
    scorer: str,
    cores: Mapping[int, rdata.ScorerScores],
    temperature: float,
    stores: Mapping[int, sdata.EmbeddingStore],
    split: rdata.ReliabilitySplit,
    sem_stats: Mapping[int, np.ndarray],
    stats17_raw: Mapping[int, np.ndarray],
    *,
    phase05_root: Path,
    cfg: Mapping[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Train / select / predict every A7 model for one B3 scorer seed."""
    anchor = cores[KS[0]]
    eval_split = np.asarray(anchor.eval_split)
    image_id = np.asarray(anchor.image_id, dtype=np.int64)
    masks = _row_masks(eval_split, image_id, split)
    correct_by_k = {k: np.asarray(cores[k].correct, dtype=bool) for k in KS}
    y_by_k = {k: correct_by_k[k].astype(np.float64) for k in KS}
    reliability_seed = int(scorer.split("b3_seed")[1])

    tune_flags = {k: masks["tune"] for k in KS}
    train_flags = {k: masks["train"] for k in KS}

    conf: Dict[str, Dict[int, np.ndarray]] = {}
    prob: Dict[str, Dict[int, np.ndarray]] = {}
    params: Dict[str, int] = {}
    information: Dict[str, str] = {}
    selected_hp: Dict[str, Any] = {}
    tune_auroc: Dict[str, float] = {}
    selections: List[Dict[str, Any]] = []
    e0_checks: Dict[str, float] = {}
    coefficients: Dict[str, Any] = {}
    timings: Dict[str, float] = {}

    # -- E0a: MSP (frozen scalar) -----------------------------------------
    conf["msp"] = {k: np.asarray(rfeat.scalar_confidence(np.asarray(cores[k].scores, dtype=np.float64),
                                                         temperature=temperature)["msp"], dtype=np.float64)
                   for k in KS}
    prob["msp"] = conf["msp"]
    params["msp"] = 0
    information["msp"] = "score_scalar"
    selected_hp["msp"] = None
    tune_auroc["msp"] = _tune_mean_auroc(conf["msp"], y_by_k, tune_flags)

    # -- E1a / E1b inputs --------------------------------------------------
    sem_std = _standardised_blocks(sem_stats, train_flags, sfeat.SEMANTIC_STAT_NAMES)
    stats17_std = _standardised_blocks(stats17_raw, train_flags, tuple(rfeat.stat_feature_names()))
    e1b_std = {k: np.hstack([stats17_std[k], sem_std[k]]) for k in KS}

    # -- E0b: Stats Logistic (reproduced with the phase 0.5 chosen C) ------
    sel_path = Path(phase05_root) / "stats_logistic" / "selection.json"
    sel_payload = json.loads(sel_path.read_text(encoding="utf-8"))
    c_chosen = float(sel_payload["per_scorer"][scorer]["chosen_hp"])
    stats_clf = rmodels.LogisticModel(C=c_chosen)
    x_train = np.vstack([stats17_std[k][train_flags[k]] for k in TRAIN_KS])
    y_train = np.concatenate([y_by_k[k][train_flags[k]] for k in TRAIN_KS])
    stats_clf.fit(x_train, y_train)
    conf["stats_logistic"] = {k: stats_clf.predict_proba(stats17_std[k]) for k in KS}
    prob["stats_logistic"] = conf["stats_logistic"]
    params["stats_logistic"] = int(stats_clf.n_parameters())
    information["stats_logistic"] = "score_summary_statistics"
    selected_hp["stats_logistic"] = c_chosen
    tune_auroc["stats_logistic"] = _tune_mean_auroc(conf["stats_logistic"], y_by_k, tune_flags)

    # -- E1 logistic family -------------------------------------------------
    for name, blocks in ((E1_NAME, sem_std), (E1B_NAME, e1b_std)):
        train_x = np.vstack([blocks[k][train_flags[k]] for k in TRAIN_KS])
        train_y = np.concatenate([y_by_k[k][train_flags[k]] for k in TRAIN_KS])
        grid = cfg["c_grid"]
        scored: List[Tuple[float, float]] = []
        fitted: Dict[float, Any] = {}
        for c_value in grid:
            clf = rmodels.LogisticModel(C=float(c_value))
            clf.fit(train_x, train_y)
            conf_k = {k: clf.predict_proba(blocks[k]) for k in KS}
            value = _tune_mean_auroc(conf_k, y_by_k, tune_flags)
            if value is None:
                raise RuntimeError(f"{name}: tune AUROC undefined for C={c_value}")
            scored.append((float(c_value), float(value)))
            fitted[float(c_value)] = clf
        best_idx = _select_by_tie([v for _, v in scored], [c for c, _ in scored])
        best_c, best_value = scored[best_idx]
        model = fitted[best_c]
        conf[name] = {k: model.predict_proba(blocks[k]) for k in KS}
        prob[name] = conf[name]
        params[name] = int(model.n_parameters())
        information[name] = "semantic_statistics" if name == E1_NAME else "score_statistics+semantic_statistics"
        selected_hp[name] = best_c
        tune_auroc[name] = best_value
        selections.append({"model": name, "grid": [c for c, _ in scored], "values": [v for _, v in scored],
                           "chosen": best_c})
        if name == E1B_NAME:
            coef = model.coefficients()
            coefficients[name] = {
                "feature_names": list(rfeat.stat_feature_names()) + list(sfeat.SEMANTIC_STAT_NAMES),
                "coef": np.asarray(coef[0], dtype=np.float64).tolist(),
                "intercept": float(np.asarray(coef[1]).reshape(-1)[0]),
            }

    # -- E2 / E3 inputs ------------------------------------------------------
    scores_by_k = {k: np.asarray(cores[k].scores, dtype=np.float64) for k in KS}
    e2_native = {k: _e2_inputs_for_k(k, stores[k], scores_by_k[k], temperature) for k in KS}
    e3_native_full = {k: _e3_native_for_k(k, stores[k], scores_by_k[k], temperature, top5=False) for k in KS}
    e3_native_top5 = {k: _e3_native_for_k(k, stores[k], scores_by_k[k], temperature, top5=True) for k in KS}

    def _pack_rows(per_k: Mapping[int, Mapping[str, np.ndarray]], flags: Mapping[str, np.ndarray],
                   key: str) -> np.ndarray:
        """Row-concat one key of the K5+K10 inputs selected by ``flags``."""
        return np.concatenate([np.asarray(per_k[k][key])[flags[k]] for k in TRAIN_KS], axis=0)

    def _fit_torch_family(name: str, per_k: Mapping[int, Mapping[str, np.ndarray]], builder: Callable[[float], Any],
                          *, log_messages: bool = False) -> None:
        keys = tuple(per_k[TRAIN_KS[0]].keys())
        train_in = {key: _pack_rows(per_k, train_flags, key) for key in keys}
        val_in = {key: _pack_rows(per_k, tune_flags, key) for key in keys}
        train_y = np.concatenate([y_by_k[k][train_flags[k]] for k in TRAIN_KS])
        val_y = np.concatenate([y_by_k[k][tune_flags[k]] for k in TRAIN_KS])
        scored: List[Tuple[float, float]] = []
        fitted: Dict[float, Any] = {}
        for lr in cfg["lr_grid"]:
            model = builder(float(lr))
            fit_started = time.perf_counter()
            model.fit(train_in, train_y, lr=float(lr), inputs_val=val_in, y_val=val_y,
                      epochs=cfg["max_epochs"], batch_size=256, patience=cfg["patience"],
                      log=None)
            timings[f"{name}:{lr}"] = time.perf_counter() - fit_started
            conf_k = {k: model.predict_proba(per_k[k]) for k in KS}
            value = _tune_mean_auroc(conf_k, y_by_k, tune_flags)
            if value is None:
                raise RuntimeError(f"{name}: tune AUROC undefined for lr={lr}")
            scored.append((float(lr), float(value)))
            fitted[float(lr)] = model
            if log_messages:
                log(f"[phase1]   {name} lr={lr:g}: tune mean AUROC = {value:.4f}")
        best_idx = _select_by_tie([v for _, v in scored], [c for c, _ in scored])
        best_lr, best_value = scored[best_idx]
        model = fitted[best_lr]
        conf[name] = {k: model.predict_proba(per_k[k]) for k in KS}
        prob[name] = conf[name]
        params[name] = int(model.n_parameters())
        selected_hp[name] = best_lr
        tune_auroc[name] = best_value
        selections.append({"model": name, "grid": [c for c, _ in scored], "values": [v for _, v in scored],
                           "chosen": best_lr})

    # -- E2: score / semantic / combined ------------------------------------
    for name, (use_score, use_sem) in zip(E2_NAMES, ((True, False), (False, True), (True, True))):
        def _builder(lr: float, use_score: bool = use_score, use_sem: bool = use_sem) -> Any:
            return smodels.TopCompetitorModel(seed=reliability_seed, include_score=use_score,
                                              include_semantic=use_sem)
        _fit_torch_family(name, e2_native, _builder, log_messages=True)
        information[name] = "score_stats" if not use_sem else ("top_competitor_semantics" if not use_score
                                                                else "score_stats+top_competitor_semantics")

    # -- E3: full / top5 / +logK ---------------------------------------------
    def _e3_packed(per_k: Mapping[int, Mapping[str, np.ndarray]], ks: Sequence[int]) -> Dict[str, np.ndarray]:
        return _pack_e3({k: per_k[k] for k in ks}, ks)

    for name, per_k, include_logk in ((E3_FULL, e3_native_full, False), (E3_TOP5, e3_native_top5, False),
                                      (E3_LOGK, e3_native_full, True)):
        train_in = _pack_e3({k: {key: np.asarray(value)[train_flags[k]] for key, value in per_k[k].items()}
                             for k in TRAIN_KS}, TRAIN_KS)
        val_in = _pack_e3({k: {key: np.asarray(value)[tune_flags[k]] for key, value in per_k[k].items()}
                           for k in TRAIN_KS}, TRAIN_KS)
        train_y = np.concatenate([y_by_k[k][train_flags[k]] for k in TRAIN_KS])
        val_y = np.concatenate([y_by_k[k][tune_flags[k]] for k in TRAIN_KS])
        scored_e3: List[Tuple[float, float]] = []
        fitted_e3: Dict[float, Any] = {}
        conf_by_lr: Dict[float, Dict[int, np.ndarray]] = {}
        for lr in cfg["lr_grid"]:
            fit_started = time.perf_counter()
            model = smodels.SemanticDeepSets(seed=reliability_seed, include_logk=include_logk)
            model.fit(train_in, train_y, lr=float(lr), inputs_val=val_in, y_val=val_y,
                      epochs=cfg["max_epochs"], batch_size=256, patience=cfg["patience"], log=None)
            timings[f"{name}:{lr}"] = time.perf_counter() - fit_started
            # pack one K at a time in row chunks: the K50 float32 block (~2 GiB
            # unpacked) must never be held whole
            conf_k: Dict[int, np.ndarray] = {}
            for k in KS:
                conf_k[k] = _predict_e3_chunked(model, per_k, k)
            value = _tune_mean_auroc(conf_k, y_by_k, tune_flags)
            if value is None:
                raise RuntimeError(f"{name}: tune AUROC undefined for lr={lr}")
            scored_e3.append((float(lr), float(value)))
            fitted_e3[float(lr)] = model
            conf_by_lr[float(lr)] = conf_k
            log(f"[phase1]   {name} lr={lr:g}: tune mean AUROC = {value:.4f}")
        best_idx = _select_by_tie([v for _, v in scored_e3], [c for c, _ in scored_e3])
        best_lr, best_value = scored_e3[best_idx]
        conf[name] = conf_by_lr[best_lr]
        prob[name] = conf[name]
        params[name] = int(fitted_e3[best_lr].n_parameters())
        information[name] = "semantic_deepsets_top5" if name == E3_TOP5 else (
            "semantic_deepsets+logK" if include_logk else "semantic_deepsets")
        selected_hp[name] = best_lr
        tune_auroc[name] = best_value
        selections.append({"model": name, "grid": [c for c, _ in scored_e3], "values": [v for _, v in scored_e3],
                           "chosen": best_lr})

    # -- E0 reproduction checks --------------------------------------------
    ref_msp = _load_phase05_reference(Path(phase05_root), scorer, "msp")
    e0_checks["msp"] = _verify_against_phase05(conf["msp"], ref_msp, cores, "msp", log)
    ref_stats = _load_phase05_reference(Path(phase05_root), scorer, "stats_logistic")
    e0_checks["stats_logistic"] = _verify_against_phase05(conf["stats_logistic"], ref_stats, cores,
                                                          "stats_logistic", log)

    # -- family selections ---------------------------------------------------
    best_score_only = max(SCORE_ONLY_NAMES, key=lambda name: (tune_auroc[name], -params[name]))
    best_semantic = max(SEMANTIC_NAMES, key=lambda name: (tune_auroc[name], -params[name]))
    log(f"[phase1] {scorer}: best score-only = {best_score_only} "
        f"(tune {tune_auroc[best_score_only]:.4f}); best semantic = {best_semantic} "
        f"(tune {tune_auroc[best_semantic]:.4f})")

    return {
        "scorer": scorer,
        "temperature": float(temperature),
        "conf": conf,
        "prob": prob,
        "params": params,
        "information": information,
        "selected_hp": selected_hp,
        "tune_auroc": tune_auroc,
        "selections": selections,
        "e0_checks": e0_checks,
        "coefficients": coefficients,
        "timings": timings,
        "best_score_only": best_score_only,
        "best_semantic": best_semantic,
        "eval_split": eval_split,
        "image_id": image_id,
        "correct": correct_by_k,
        "masks": masks,
        "ref_id": np.asarray(anchor.ref_id, dtype=np.int64),
        "top1_score": {k: np.asarray(scores_by_k[k], dtype=np.float64).max(axis=1) for k in KS},
        "scalars": {k: rfeat.scalar_confidence(np.asarray(cores[k].scores, dtype=np.float64),
                                                temperature=temperature) for k in KS},
    }


ALL_MODELS: Tuple[str, ...] = SCORE_ONLY_NAMES + SEMANTIC_NAMES


# ---------------------------------------------------------------------------
# stage 4: aggregate metrics / bootstrap / A7.7 analyses
# ---------------------------------------------------------------------------
def _aggregate_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for res in results:
        for model in ALL_MODELS:
            for split in REPORT_SPLITS + (POOLED,):
                mask = np.asarray(res["masks"][split], dtype=bool)
                for k in KS:
                    conf = np.asarray(res["conf"][model][k], dtype=np.float64)[mask]
                    corr = np.asarray(res["correct"][k], dtype=bool)[mask]
                    proba = np.asarray(res["prob"][model][k], dtype=np.float64)[mask]
                    row: Dict[str, Any] = {
                        "scorer": res["scorer"],
                        "model": model,
                        "information": res["information"][model],
                        "params": int(res["params"][model]),
                        "eval_split": split,
                        "K": int(k),
                        "n": int(np.sum(mask)),
                        "tune_mean_auroc": res["tune_auroc"][model],
                        "selected_hp": res["selected_hp"][model],
                    }
                    row.update(reval.point_metric_row(conf, corr, probability=proba))
                    rows.append(row)
    return rows


def _bootstrap_stage(
    results: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
    ci: float,
    log: Callable[[str], None],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str, str, int], Dict[str, Dict[str, Any]]]]:
    """Cross-K degradation rows + frozen pairwise rows (absolute + E-AURC ratio).

    Returns the CSV rows and a cache keyed ``(scorer, model_sem, model_score, K)``
    holding the absolute + ratio dicts (used by the A7.6 gate).
    """
    cross_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, str, int], Dict[str, Dict[str, Any]]] = {}
    n_calls = len(results) * len(HEADLINE_MODELS) * 2 * 3  # cross-K
    n_calls += len(results) * len(PAIRS) * 2 * 2  # pairwise at K=5,50 (absolute + ratio)
    bar = _bar("bootstrap", n_calls)
    for res in results:
        scorer = res["scorer"]
        image_id = np.asarray(res["image_id"], dtype=np.int64)
        for model in HEADLINE_MODELS:
            for k_lo, k_hi in ((5, 20), (5, 50)):
                for split in ("testA", "testB", POOLED):
                    mask = np.asarray(res["masks"][split], dtype=bool)
                    rows = reval.cross_k_bootstrap_row(
                        np.asarray(res["conf"][model][k_lo])[mask],
                        np.asarray(res["correct"][k_lo])[mask],
                        np.asarray(res["conf"][model][k_hi])[mask],
                        np.asarray(res["correct"][k_hi])[mask],
                        image_id[mask],
                        eval_split=split,
                        k_lo=int(k_lo),
                        k_hi=int(k_hi),
                        replicates=int(replicates),
                        seed=int(seed),
                        ci=float(ci),
                    )
                    for row in rows:
                        row["scorer"] = scorer
                        row["model"] = model
                        row["row_kind"] = "cross_k"
                        cross_rows.append(row)
                    bar.update(1)
        for model_sem, model_score in PAIRS:
            for k in (5, 50):
                mask = np.asarray(res["masks"][POOLED], dtype=bool)
                conf_sem = np.asarray(res["conf"][model_sem][k], dtype=np.float64)[mask]
                conf_score = np.asarray(res["conf"][model_score][k], dtype=np.float64)[mask]
                corr = np.asarray(res["correct"][k], dtype=bool)[mask]
                clusters = image_id[mask]
                abs_rows = reval.model_vs_model_bootstrap_row(
                    conf_sem, corr, conf_score, corr, clusters,
                    eval_split=POOLED, K=int(k), model_a=model_sem, model_b=model_score,
                    replicates=int(replicates), seed=int(seed), ci=float(ci),
                )
                ratio = seval.model_vs_model_e_aurc_ratio_row(
                    conf_sem, corr, conf_score, corr, clusters,
                    eval_split=POOLED, K=int(k), model_sem=model_sem, model_score=model_score,
                    replicates=int(replicates), seed=int(seed), ci=float(ci),
                )
                for row in abs_rows:
                    row["scorer"] = scorer
                    row["row_kind"] = "pairwise"
                    pair_rows.append(row)
                ratio["scorer"] = scorer
                ratio["row_kind"] = "pairwise_ratio"
                pair_rows.append(ratio)
                entry: Dict[str, Dict[str, Any]] = {
                    "ratio": ratio,
                    "abs": {str(row["metric"]): row for row in abs_rows},
                }
                cache[(scorer, model_sem, model_score, int(k))] = entry
                bar.update(2)
    bar.close()
    log(f"[phase1] bootstrap: {len(cross_rows)} cross-K rows + {len(pair_rows)} pairwise rows")
    return cross_rows + pair_rows, cache


def _ood_stability_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """A7.7 (??32): does the model reduce K-degradation or improve everything?"""
    rows: List[Dict[str, Any]] = []
    for res in results:
        for model in ALL_MODELS:
            out: Dict[str, Any] = {"scorer": res["scorer"], "model": model}
            for k, split in ((5, POOLED), (50, POOLED)):
                mask = np.asarray(res["masks"][split], dtype=bool)
                conf = np.asarray(res["conf"][model][k], dtype=np.float64)[mask]
                corr = np.asarray(res["correct"][k], dtype=bool)[mask]
                proba = np.asarray(res["prob"][model][k], dtype=np.float64)[mask]
                row = reval.point_metric_row(conf, corr, probability=proba)
                out[f"auroc_{k}"] = row["auroc_correct"]
                out[f"e_aurc_{k}"] = row["e_aurc"]
                out[f"rer50_{k}"] = row["rer_at_50"]
            out["auroc_gap_5_50"] = out["auroc_5"] - out["auroc_50"]
            out["e_aurc_ratio_50_5"] = out["e_aurc_50"] / out["e_aurc_5"]
            out["rer50_gap_5_50"] = out["rer50_5"] - out["rer50_50"]
            rows.append(out)
    mean_rows: List[Dict[str, Any]] = []
    for model in ALL_MODELS:
        sel = [row for row in rows if row["model"] == model]
        mean_row: Dict[str, Any] = {"scorer": "b3_mean", "model": model}
        for field in sel[0]:
            if field in ("scorer", "model"):
                continue
            mean_row[field] = float(np.mean([row[field] for row in sel]))
        mean_rows.append(mean_row)
    return rows + mean_rows


def _quartile_labels(values: np.ndarray) -> np.ndarray:
    """Deterministic Q1..Q4 labels from row quantiles (equal-count as far as ties allow)."""
    edges = np.quantile(np.asarray(values, dtype=np.float64), [0.25, 0.50, 0.75])
    labels = np.full(values.shape[0], "Q1", dtype=object)
    labels[np.asarray(values) > edges[0]] = "Q2"
    labels[np.asarray(values) > edges[1]] = "Q3"
    labels[np.asarray(values) > edges[2]] = "Q4"
    return labels


def _error_subset_rows(
    results: Sequence[Mapping[str, Any]],
    sem_stats: Mapping[str, Mapping[int, np.ndarray]],
) -> List[Dict[str, Any]]:
    """A7.7 (??33): K50 groups by candidate/CLIP ambiguity (q1..q4) + the B3 A/B split."""
    idx = {name: pos for pos, name in enumerate(sfeat.SEMANTIC_STAT_NAMES)}
    models = ("stats_logistic", E1B_NAME, "e2_combined", E3_FULL)
    rows: List[Dict[str, Any]] = []
    for res in results:
        scorer = res["scorer"]
        mask = np.asarray(res["masks"][POOLED], dtype=bool)
        for group_by, stat_name in (("cand_top12_sim", "cand_top12_sim"), ("clip_margin12", "clip_margin12")):
            values = np.asarray(sem_stats[scorer][50][:, idx[stat_name]], dtype=np.float64)[mask]
            labels = _quartile_labels(values)
            for group in ("all", "Q1", "Q2", "Q3", "Q4"):
                if group == "all":
                    sel = np.ones(mask.sum(), dtype=bool)
                else:
                    sel = labels == group
                if not np.any(sel):
                    continue
                out: Dict[str, Any] = {"scorer": scorer, "group_by": group_by, "group": group,
                                       "n": int(sel.sum())}
                corr = np.asarray(res["correct"][50], dtype=bool)[mask][sel]
                out["b3_error_rate"] = float(1.0 - corr.mean())
                for model in models:
                    conf = np.asarray(res["conf"][model][50], dtype=np.float64)[mask][sel]
                    out[f"auroc_{model}"] = _auroc(conf, corr)
                out[f"delta_{E1B_NAME}"] = (
                    None if out[f"auroc_{E1B_NAME}"] is None or out["auroc_stats_logistic"] is None
                    else out[f"auroc_{E1B_NAME}"] - out["auroc_stats_logistic"])
                out["delta_e2_combined"] = (
                    None if out["auroc_e2_combined"] is None or out["auroc_stats_logistic"] is None
                    else out["auroc_e2_combined"] - out["auroc_stats_logistic"])
                out[f"delta_{E3_FULL}"] = (
                    None if out[f"auroc_{E3_FULL}"] is None or out["auroc_stats_logistic"] is None
                    else out[f"auroc_{E3_FULL}"] - out["auroc_stats_logistic"])
                rows.append(out)
    return rows


def _matched_score_rows(
    results: Sequence[Mapping[str, Any]],
    sem_stats: Mapping[str, Mapping[int, np.ndarray]],
    *,
    replicates: int,
    seed: int,
    ci: float,
) -> List[Dict[str, Any]]:
    """A7.7 (??34): nearest-neighbour matching on score statistics (never on correctness)."""
    idx = {name: pos for pos, name in enumerate(sfeat.SEMANTIC_STAT_NAMES)}
    rows: List[Dict[str, Any]] = []
    for res in results:
        scorer = res["scorer"]
        mask = np.asarray(res["masks"][POOLED], dtype=bool)
        correct = np.asarray(res["correct"][50], dtype=bool)[mask]
        image_id = np.asarray(res["image_id"], dtype=np.int64)[mask]
        sim = np.asarray(sem_stats[scorer][50][:, idx["cand_top12_sim"]], dtype=np.float64)[mask]
        scalars = res["scalars"][50]
        features = np.column_stack([
            np.asarray(scalars["msp"], dtype=np.float64)[mask],
            np.asarray(scalars["margin"], dtype=np.float64)[mask],
            -np.asarray(scalars["neg_entropy"], dtype=np.float64)[mask],  # H(P)
        ])
        mu = features.mean(axis=0)
        sigma = features.std(axis=0)
        sigma[sigma == 0.0] = 1.0
        z = (features - mu) / sigma
        threshold = float(np.quantile(sim, 0.75))
        treated = np.flatnonzero(sim > threshold)
        control = np.flatnonzero(sim <= threshold)
        used = np.zeros(control.size, dtype=bool)
        pairs: List[Tuple[int, int]] = []
        for row_idx in treated:
            dist = np.linalg.norm(z[control] - z[row_idx], axis=1)
            dist[used] = np.inf
            j = int(np.argmin(dist))
            if not np.isfinite(dist[j]):
                continue
            used[j] = True
            pairs.append((int(row_idx), int(control[j])))
        if not pairs:
            continue
        pair_a = np.asarray([p[0] for p in pairs], dtype=np.int64)
        pair_b = np.asarray([p[1] for p in pairs], dtype=np.int64)
        delta = float(correct[pair_a].mean() - correct[pair_b].mean())
        clusters = np.unique(image_id)
        rng = np.random.default_rng(int(seed))
        draws = rng.integers(0, clusters.size, size=(int(replicates), clusters.size))
        reps = np.empty(int(replicates), dtype=np.float64)
        pair_cluster = image_id[pair_a]
        sorted_clusters = np.sort(clusters)
        for rep in range(int(replicates)):
            sel_images = sorted_clusters[draws[rep]]
            take = np.isin(pair_cluster, sel_images)
            if not np.any(take):
                reps[rep] = np.nan
                continue
            reps[rep] = correct[pair_a[take]].mean() - correct[pair_b[take]].mean()
        alpha = (1.0 - float(ci)) / 2.0
        valid = reps[np.isfinite(reps)]
        rows.append({
            "scorer": scorer,
            "K": 50,
            "eval_split": POOLED,
            "n_treated": int(treated.size),
            "n_control": int(control.size),
            "n_matched": int(pair_a.size),
            "treat_threshold_top12_sim": threshold,
            "mean_correct_treated": float(correct[pair_a].mean()),
            "mean_correct_control": float(correct[pair_b].mean()),
            "delta_correct": delta,
            "ci_low": float(np.quantile(valid, alpha)) if valid.size else None,
            "ci_high": float(np.quantile(valid, 1.0 - alpha)) if valid.size else None,
            "n_valid_replicates": int(valid.size),
            "n_clusters": int(clusters.size),
            "mean_match_distance": float(np.mean([np.linalg.norm(z[a] - z[b]) for a, b in pairs])),
            "replicates": int(replicates),
            "ci_level": float(ci),
            "seed": int(seed),
        })
    return rows


def _coefficient_rows(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """A7.7 (??35): standardised E1b coefficients, per seed plus mean/std rows."""
    rows: List[Dict[str, Any]] = []
    by_feature: Dict[str, List[float]] = {}
    for res in results:
        payload = res["coefficients"].get(E1B_NAME)
        if payload is None:
            continue
        for name, value in zip(payload["feature_names"], payload["coef"]):
            rows.append({"scorer": res["scorer"], "feature": name, "block": "stats" if not name.startswith(
                ("clip_", "cand_", "density_", "q_")) else "semantic", "coefficient": float(value)})
            by_feature.setdefault(name, []).append(float(value))
    for name, values in by_feature.items():
        block = "stats" if not name.startswith(("clip_", "cand_", "density_", "q_")) else "semantic"
        rows.append({"scorer": "b3_mean", "feature": name, "block": block,
                     "coefficient": float(np.mean(values)), "coefficient_std": float(np.std(values, ddof=0))})
    return rows


# ---------------------------------------------------------------------------
# stage 5: outputs (predictions / figures / gate / protocol)
# ---------------------------------------------------------------------------
def _union_fields(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _dump_predictions(out_dir: Path, results: Sequence[Mapping[str, Any]]) -> List[str]:
    """Save the raw reliability predictions (A7.9 / ??39 columns) per scorer seed."""
    written: List[str] = []
    out_root = Path(out_dir) / "predictions"
    out_root.mkdir(parents=True, exist_ok=True)
    for res in results:
        scorer = res["scorer"]
        best_so = res["best_score_only"]
        best_sem = res["best_semantic"]
        path = out_root / f"{scorer}.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ref_id", "image_id", "K", "eval_split", "grounding_correct", "B3_score",
                             "score_only_reliability", "semantic_reliability", "scorer_seed"])
            keep = np.isin(np.asarray(res["eval_split"]), np.asarray(["val_select", "testA", "testB"]))
            seed_number = int(scorer.split("b3_seed")[1])
            for k in KS:
                ref_id = np.asarray(res["ref_id"])[keep]
                image_id = np.asarray(res["image_id"])[keep]
                splits = np.asarray(res["eval_split"])[keep]
                correct = np.asarray(res["correct"][k], dtype=bool)[keep]
                b3_score = np.asarray(res["top1_score"][k])[keep]
                conf_so = np.asarray(res["conf"][best_so][k], dtype=np.float64)[keep]
                conf_sem = np.asarray(res["conf"][best_sem][k], dtype=np.float64)[keep]
                for i in range(ref_id.size):
                    writer.writerow([int(ref_id[i]), int(image_id[i]), int(k), str(splits[i]),
                                     int(correct[i]), f"{float(b3_score[i]):.10g}",
                                     f"{float(conf_so[i]):.10g}", f"{float(conf_sem[i]):.10g}",
                                     seed_number])
        written.append(str(path.relative_to(out_dir)))
    return written


def _headline_table(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Seed-mean/std per model per K on __pooled_test__ (reporting table)."""
    table: Dict[str, Any] = {}
    for model in ALL_MODELS:
        per_k: Dict[str, Any] = {}
        for k in KS:
            vals = {"auroc_correct": [], "e_aurc": [], "rer_at_50": [], "rer_at_80": []}
            for res in results:
                mask = np.asarray(res["masks"][POOLED], dtype=bool)
                row = reval.point_metric_row(np.asarray(res["conf"][model][k], dtype=np.float64)[mask],
                                             np.asarray(res["correct"][k], dtype=bool)[mask])
                for metric in vals:
                    vals[metric].append(float(row[metric]))
            per_k[str(k)] = {metric: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=0)),
                                      "per_seed": [float(x) for x in v]}
                             for metric, v in vals.items()}
        table[model] = per_k
    return table


def _figure_k_curves(out_dir: Path, results: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Figure A (AUROC) + Figure B (E-AURC / RER@50) vs K, pooled 3-seed mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = ("msp", "stats_logistic", E1B_NAME, "e2_combined", E3_FULL)
    labels = {"msp": "MSP", "stats_logistic": "Stats Logistic", E1B_NAME: "Stats + Semantic",
              "e2_combined": "Top-Competitor", E3_FULL: "Semantic DeepSets"}
    series: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}
    for model in curves:
        per_metric: Dict[str, Tuple[List[float], List[float]]] = {}
        for metric in ("auroc_correct", "e_aurc", "rer_at_50"):
            means, stds = [], []
            for k in KS:
                vals = []
                for res in results:
                    mask = np.asarray(res["masks"][POOLED], dtype=bool)
                    row = reval.point_metric_row(np.asarray(res["conf"][model][k], dtype=np.float64)[mask],
                                                 np.asarray(res["correct"][k], dtype=bool)[mask])
                    vals.append(float(row[metric]))
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=0)))
            per_metric[metric] = (means, stds)
        series[model] = per_metric

    out: Dict[str, str] = {}
    panels = (("auroc_correct", "AUROC_correct"), ("e_aurc", "E-AURC"), ("rer_at_50", "RER@50"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (metric, title) in zip(axes, panels):
        for model in curves:
            means, stds = series[model][metric]
            ax.errorbar(KS, means, yerr=stds, marker="o", capsize=3, label=labels[model])
        ax.set_xlabel("K (candidate-set size)")
        ax.set_title(f"{title} (pooled test, 3-seed mean +/- std)")
        ax.set_xticks(list(KS))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path_a = Path(out_dir) / "figures" / "semantic_reliability_vs_K.png"
    path_a.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_a, dpi=160)
    plt.close(fig)
    out["figure_A"] = str(path_a.relative_to(out_dir))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], "e_aurc", "E-AURC"), (axes[1], "rer_at_50", "RER@50")):
        for model in curves:
            means, stds = series[model][metric]
            ax.errorbar(KS, means, yerr=stds, marker="o", capsize=3, label=labels[model])
        ax.set_xlabel("K (candidate-set size)")
        ax.set_title(f"{title} (pooled test, 3-seed mean +/- std)")
        ax.set_xticks(list(KS))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path_b = Path(out_dir) / "figures" / "selective_risk_vs_K.png"
    fig.savefig(path_b, dpi=160)
    plt.close(fig)
    out["figure_B"] = str(path_b.relative_to(out_dir))
    return out


def _figure_quartiles(out_dir: Path, subset_rows: Sequence[Mapping[str, Any]]) -> str:
    """Figure C: K50 AUROC / B3 error rate by semantic-ambiguity quartile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ("stats_logistic", E1B_NAME, "e2_combined", E3_FULL)
    labels = {"stats_logistic": "Stats Logistic", E1B_NAME: "Stats + Semantic",
              "e2_combined": "Top-Competitor", E3_FULL: "Semantic DeepSets"}
    groups = ("Q1", "Q2", "Q3", "Q4")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row_i, group_by in enumerate(("cand_top12_sim", "clip_margin12")):
        ax_auroc, ax_err = axes[row_i][0], axes[row_i][1]
        for model in models:
            xs, ys = [], []
            for group in groups:
                vals = [float(row[f"auroc_{model}"]) for row in subset_rows
                        if row["group_by"] == group_by and row["group"] == group
                        and row.get(f"auroc_{model}") is not None]
                xs.append(group)
                ys.append(float(np.mean(vals)) if vals else np.nan)
            ax_auroc.plot(xs, ys, marker="o", label=labels[model])
        err = []
        for group in groups:
            vals = [float(row["b3_error_rate"]) for row in subset_rows
                    if row["group_by"] == group_by and row["group"] == group]
            err.append(float(np.mean(vals)) if vals else np.nan)
        ax_err.bar(list(groups), err, color="0.4")
        ax_auroc.set_title(f"AUROC by {group_by} quartile")
        ax_err.set_title(f"B3 error rate by {group_by} quartile")
        ax_auroc.grid(alpha=0.3)
        ax_err.grid(alpha=0.3)
        ax_auroc.set_ylim(0.5, 1.0)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path = Path(out_dir) / "figures" / "ambiguity_stratified_k50.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path.relative_to(out_dir))


# ---------------------------------------------------------------------------
# stage 6: A7.6 gate / protocol / metadata
# ---------------------------------------------------------------------------
def _pair_entry(
    res: Mapping[str, Any],
    pair_cache: Dict[Tuple[str, str, str, int], Dict[str, Dict[str, Any]]],
    model_sem: str,
    model_score: str,
    k: int,
    *,
    replicates: int,
    seed: int,
    ci: float,
) -> Dict[str, Dict[str, Any]]:
    """Absolute + E-AURC-ratio bootstrap rows of one pair cell (cached)."""
    key = (res["scorer"], model_sem, model_score, int(k))
    entry = pair_cache.get(key)
    if entry is not None:
        return entry
    mask = np.asarray(res["masks"][POOLED], dtype=bool)
    conf_sem = np.asarray(res["conf"][model_sem][k], dtype=np.float64)[mask]
    conf_score = np.asarray(res["conf"][model_score][k], dtype=np.float64)[mask]
    corr = np.asarray(res["correct"][k], dtype=bool)[mask]
    clusters = np.asarray(res["image_id"], dtype=np.int64)[mask]
    abs_rows = reval.model_vs_model_bootstrap_row(
        conf_sem, corr, conf_score, corr, clusters,
        eval_split=POOLED, K=int(k), model_a=model_sem, model_b=model_score,
        replicates=int(replicates), seed=int(seed), ci=float(ci),
    )
    ratio = seval.model_vs_model_e_aurc_ratio_row(
        conf_sem, corr, conf_score, corr, clusters,
        eval_split=POOLED, K=int(k), model_sem=model_sem, model_score=model_score,
        replicates=int(replicates), seed=int(seed), ci=float(ci),
    )
    entry = {"ratio": ratio, "abs": {str(row["metric"]): row for row in abs_rows}}
    pair_cache[key] = entry
    return entry


def _gate_stage(
    results: Sequence[Mapping[str, Any]],
    pair_cache: Dict[Tuple[str, str, str, int], Dict[str, Dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
    ci: float,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """A7.6 candidate-semantic GO gate on the pooled OOD cells."""
    mean_tune = {name: float(np.mean([res["tune_auroc"][name] for res in results])) for name in ALL_MODELS}
    best_score_only = max(SCORE_ONLY_NAMES, key=lambda name: mean_tune[name])
    best_semantic = max(GATE_SEMANTIC_NAMES, key=lambda name: mean_tune[name])
    cells: Dict[int, Dict[str, float]] = {}
    for k in (20, 50):
        entries = [
            _pair_entry(res, pair_cache, best_semantic, best_score_only, k,
                        replicates=replicates, seed=seed, ci=ci)
            for res in results
        ]
        cells[int(k)] = {
            "delta_auroc": float(np.mean([e["abs"]["auroc_correct"]["diff"] for e in entries])),
            "delta_auroc_ci_low": float(np.mean([e["abs"]["auroc_correct"]["ci_low"] for e in entries])),
            "delta_auroc_ci_high": float(np.mean([e["abs"]["auroc_correct"]["ci_high"] for e in entries])),
            "e_aurc_reduction": float(np.mean([e["ratio"]["reduction"] for e in entries])),
            "e_aurc_reduction_ci_low": float(np.mean([e["ratio"]["reduction_ci_low"] for e in entries])),
            "e_aurc_reduction_ci_high": float(np.mean([e["ratio"]["reduction_ci_high"] for e in entries])),
            "rer50_gain_pp": float(100.0 * np.mean([e["abs"]["rer_at_50"]["diff"] for e in entries])),
            "rer50_gain_pp_ci_low": float(100.0 * np.mean([e["abs"]["rer_at_50"]["ci_low"] for e in entries])),
            "rer50_gain_pp_ci_high": float(100.0 * np.mean([e["abs"]["rer_at_50"]["ci_high"] for e in entries])),
        }
    k50_entries = [
        _pair_entry(res, pair_cache, best_semantic, best_score_only, 50,
                    replicates=replicates, seed=seed, ci=ci)
        for res in results
    ]
    seed_lows = [float(e["abs"]["auroc_correct"]["ci_low"]) for e in k50_entries]
    seed_mean_delta = float(np.mean([e["abs"]["auroc_correct"]["diff"] for e in k50_entries]))
    verdict = seval.semantic_gate(cells, seed_lows, seed_mean_delta=seed_mean_delta)
    log(f"[phase1] gate: {best_score_only} -> {best_semantic} | verdict={verdict['verdict']}")
    return {
        "comparison": {
            "best_score_only": best_score_only,
            "best_semantic": best_semantic,
            "selected_by": "3-seed mean tune AUROC_correct (K5/K10), A7.6",
            "best_score_only_ci_level": "see bootstrap.csv for the underlying rows",
        },
        "tune_mean_auroc": mean_tune,
        "cells": {str(k): v for k, v in cells.items()},
        "per_seed_k50_delta_ci_low": seed_lows,
        "seed_mean_delta_k50": seed_mean_delta,
        "verdict": verdict,
        "thresholds": dict(seval.SEMANTIC_THRESHOLDS),
        "notes": [
            "cell values are 3-seed MEANS of the per-seed paired-bootstrap rows (values and CI endpoints), "
            "matching the Phase 0.5 b3_mean convention.",
            "cells are the pooled test splits (testA + testB); A7.6 requires K20 AND K50.",
            "per-seed significance = per-seed K50 delta-AUROC bootstrap CI_low > 0.",
        ],
    }


def _protocol_payload(
    args: argparse.Namespace,
    cohort: Mapping[int, int],
    split: rdata.ReliabilitySplit,
    embedding_index_sha: str,
    reference_hashes: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    import numpy as _np

    try:
        import torch

        torch_version = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        torch_version = None
        device = None
    return {
        "protocol": "Amendment A7 — Phase 1 candidate semantic information sufficiency audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository": str(_REPO),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": _np.__version__,
            "torch": torch_version,
            "device": device,
            "platform": platform.platform(),
        },
        "config": dict(cfg),
        "cohort": {"rows_per_k": {str(k): int(v) for k, v in cohort.items()}, "ks": list(KS)},
        "split": {
            "manifest": str(Path(args.phase05) / "split_manifest.json"),
            "seed": int(split.seed),
            "train_frac": float(split.train_frac),
            "n_train_images": int(split.train_images.size),
            "n_tune_images": int(split.tune_images.size),
            "reused_from": "results/phase05_score_sufficiency/split_manifest.json (A6.2)",
        },
        "embeddings": {
            "root": str(args.embeddings),
            "index_sha256": embedding_index_sha,
            "source": "cache/features (frozen OpenCLIP ViT-B/32, L2-normalised, float16)",
            "layout": "C_K = [target] + distractor_order[:K-1]; target_local == 0",
        },
        "phase05_reference_hashes": dict(reference_hashes),
        "model_zoo": {
            "score_only": list(SCORE_ONLY_NAMES),
            "semantic": list(SEMANTIC_NAMES),
            "pairs": [{"semantic": a, "score_only": b} for a, b in PAIRS],
            "headline_models": list(HEADLINE_MODELS),
        },
        "selection": {
            "metric": "mean AUROC_correct over reliability_tune K5/K10",
            "tie_break": f"earlier (simpler) grid value when |delta| < {SELECTION_TIE}",
            "grids": {"logistic_C": list(cfg["c_grid"]), "torch_lr": list(cfg["lr_grid"])},
            "reliability_seed": "int(scorer suffix), derived deterministically (A7 sec.25)",
        },
        "gate_thresholds": dict(seval.SEMANTIC_THRESHOLDS),
        "bootstrap": {"replicates": int(cfg["bootstrap_replicates"]), "seed": int(cfg["bootstrap_seed"]),
                      "ci": float(cfg["ci"]), "unit": "image cluster"},
        "prohibited": [
            "changing grounding scores / reranking", "candidate-aware grounding training",
            "Set Transformer / cross-attention", "CLIP finetuning", "global image feature",
            "geometry / objectness inputs", "GT metadata", "FineCops / RefCOCOg", "RL / VLM",
        ],
        "artifacts": [
            "protocol.json", "semantic_features/", "semantic_stats.csv", "e1_logistic/",
            "e2_top_competitor/", "e3_semantic_deepsets/", "aggregate.csv",
            "pairwise_vs_score_only.csv", "bootstrap.csv", "error_subsets.csv",
            "matched_score_analysis.csv", "ood_stability.csv", "sufficiency_gate.json",
            "figures/", "predictions/",
        ],
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 candidate semantic information sufficiency audit (A7)")
    parser.add_argument("--out", type=Path, default=Path("results/phase1_semantic_sufficiency"),
                        help="artifact directory")
    parser.add_argument("--embeddings", type=Path, default=Path("cache/semantic_phase1"),
                        help="embedding archive root (built on demand)")
    parser.add_argument("--phase05", type=Path, default=Path("results/phase05_score_sufficiency"),
                        help="frozen phase 0.5 artifacts (split manifest / predictions / selection)")
    parser.add_argument("--features", type=Path, default=Path("cache/features"))
    parser.add_argument("--manifests", type=Path, default=Path("cache/manifests"))
    parser.add_argument("--bank", type=Path, default=Path("cache/proposals.h5"))
    parser.add_argument("--refs", type=Path, default=Path("data/raw/refcoco+/refcoco+/refs(unc).p"))
    parser.add_argument("--image-sizes", type=Path, default=Path("cache/image_sizes.npz"))
    parser.add_argument("--b3-root", type=Path, default=rdata.B3_ROOT)
    parser.add_argument("--seeds", nargs="+", default=list(B3_SEEDS), choices=list(B3_SEEDS))
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--tiny", action="store_true",
                        help="smoke mode: single-value grids (C=1, lr=3e-4) for a fast end-to-end run")
    parser.add_argument("--skip-embeddings-build", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    return parser


def _family_payloads(results: Sequence[Mapping[str, Any]], agg_rows: Sequence[Mapping[str, Any]],
                     out_dir: Path) -> List[str]:
    families = (("e1_logistic", E1_NAMES), ("e2_top_competitor", E2_NAMES),
                ("e3_semantic_deepsets", E3_NAMES))
    written: List[str] = []
    for family, models in families:
        family_dir = Path(out_dir) / family
        family_dir.mkdir(parents=True, exist_ok=True)
        selection = {
            "family": family,
            "selection_metric": "mean AUROC_correct over reliability_tune K5/K10",
            "per_scorer": {
                res["scorer"]: {
                    "models": {model: {"selected_hp": res["selected_hp"][model],
                                        "tune_mean_auroc": res["tune_auroc"][model],
                                        "params": int(res["params"][model]),
                                        "information": res["information"][model]}
                               for model in models},
                }
                for res in results
            },
            "grid_search": {res["scorer"]: [entry for entry in res["selections"] if entry["model"] in models]
                            for res in results},
        }
        _write_json(family_dir / "selection.json", selection)
        rows = [row for row in agg_rows if row["model"] in models]
        _write_csv(family_dir / "metrics.csv", AGG_FIELDS, rows)
        written.extend([f"{family}/selection.json", f"{family}/metrics.csv"])
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = _make_logger(args.log_file)
    started = time.perf_counter()
    stage_time: Dict[str, float] = {}
    cfg: Dict[str, Any] = {
        "c_grid": list(TINY_C_GRID if args.tiny else C_GRID),
        "lr_grid": list(TINY_LR_GRID if args.tiny else LR_GRID),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "ci": float(args.ci),
        "tiny": bool(args.tiny),
    }
    log(f"[phase1] out       : {out_dir}")
    log(f"[phase1] seeds     : {list(args.seeds)}")
    log(f"[phase1] bootstrap : {cfg['bootstrap_replicates']} reps, seed={cfg['bootstrap_seed']}, "
        f"ci={cfg['ci']}")

    def stage(name: str, t0: float) -> None:
        stage_time[name] = time.perf_counter() - t0
        log(f"[phase1] stage {name}: {stage_time[name]:.1f}s")

    # -- stage 1: load ------------------------------------------------------
    t0 = time.perf_counter()
    scores = _load_scores(Path(args.b3_root), log)
    temperatures = _load_temperatures(Path(args.b3_root), log)
    for k in KS:
        archive = sdata.embedding_file(k, out_root=Path(args.embeddings))
        if not archive.exists() and not args.skip_embeddings_build:
            sdata.build_embedding_store(
                k, out_root=Path(args.embeddings), features_root=Path(args.features),
                manifests_root=Path(args.manifests), refs=Path(args.refs), bank_path=Path(args.bank),
                image_sizes_path=Path(args.image_sizes),
                phase05_features_dir=Path(args.phase05) / "features", b3_root=Path(args.b3_root), log=log)
    stores = _load_embeddings(Path(args.embeddings), scores, log)
    split = sdata.load_split_from_manifest(Path(args.phase05) / "split_manifest.json")
    anchor = scores[B3_SEEDS[0]][KS[0]]
    sdata.assert_split_matches_fresh(split, anchor.eval_split, anchor.image_id)
    log(f"[phase1] split: train={split.train_images.size} tune={split.tune_images.size} "
        "(verified against a fresh A6.2 rebuild)")
    stage("load", t0)

    # -- protocol.json before any modelling ---------------------------------
    reference_hashes: Dict[str, str] = {}
    manifest_path = Path(args.phase05) / "split_manifest.json"
    reference_hashes[str(manifest_path)] = _sha256_file(manifest_path)
    sel_path = Path(args.phase05) / "stats_logistic" / "selection.json"
    reference_hashes[str(sel_path)] = _sha256_file(sel_path)
    for scorer in args.seeds:
        for model in ("msp", "stats_logistic"):
            path = Path(args.phase05) / "predictions" / scorer / f"{model}.csv.gz"
            reference_hashes[str(path)] = _sha256_file(path)
    for scorer in args.seeds:
        seed_dir = Path(args.b3_root) / f"seed_{scorer.split('b3_seed')[1]}" / "raw_scores"
        for k in (5, 50):
            path = seed_dir / f"K{k}.npz"
            reference_hashes[str(path)] = _sha256_file(path)
    embedding_index = Path(args.embeddings) / sdata.EMBEDDING_INDEX_NAME
    _write_json(out_dir / "protocol.json", _protocol_payload(
        args, {k: len(store) for k, store in stores.items()}, split,
        _sha256_file(embedding_index) if embedding_index.exists() else "", reference_hashes, cfg))
    log("[phase1] protocol.json frozen (A7 design + provenance hashes)")

    # -- stage 2: E1 statistics ---------------------------------------------
    t0 = time.perf_counter()
    (out_dir / "semantic_features").mkdir(parents=True, exist_ok=True)
    sem_stats: Dict[str, Dict[int, np.ndarray]] = {}
    stats17: Dict[str, Dict[int, np.ndarray]] = {}
    for scorer in args.seeds:
        per_k: Dict[int, np.ndarray] = {}
        per_k17: Dict[int, np.ndarray] = {}
        for k in KS:
            per_k[k] = _e1_stats(stores[k], np.asarray(scores[scorer][k].scores, dtype=np.float64))
            per_k17[k] = np.asarray(rfeat.stat_features(np.asarray(scores[scorer][k].scores, dtype=np.float64),
                                                        temperature=temperatures[scorer]), dtype=np.float64)
        sem_stats[scorer] = per_k
        stats17[scorer] = per_k17
        np.savez(out_dir / "semantic_features" / f"e1_stats_{scorer}.npz",
                 names=np.asarray(sfeat.SEMANTIC_STAT_NAMES),
                 **{f"stats_K{k}": per_k[k] for k in KS})
        log(f"[phase1] E1 stats: {scorer} done")
    _write_semantic_stats_csv(out_dir / "semantic_stats.csv", sem_stats)
    stage("features", t0)

    # -- stage 3: per-seed modelling ----------------------------------------
    t0 = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for scorer in args.seeds:
        log(f"[phase1] running scorer {scorer} ...")
        res = _run_seed(scorer, scores[scorer], temperatures[scorer], stores, split,
                        sem_stats[scorer], stats17[scorer], phase05_root=Path(args.phase05), cfg=cfg, log=log)
        results.append(res)
        log(f"[phase1] {scorer}: pipeline done")
    stage("models", t0)

    # -- stage 4: aggregate / family dirs / predictions ----------------------
    t0 = time.perf_counter()
    agg_rows = _aggregate_rows(results)
    _write_csv(out_dir / "aggregate.csv", AGG_FIELDS, agg_rows)
    written_files = _family_payloads(results, agg_rows, out_dir)
    written_files += _dump_predictions(out_dir, results)
    stage("aggregate", t0)

    # -- stage 5: bootstrap + A7.7 analyses ---------------------------------
    t0 = time.perf_counter()
    boot_rows, pair_cache = _bootstrap_stage(results, replicates=cfg["bootstrap_replicates"],
                                             seed=cfg["bootstrap_seed"], ci=cfg["ci"], log=log)
    boot_fields = _union_fields(boot_rows)
    _write_csv(out_dir / "bootstrap.csv", boot_fields, boot_rows)
    pair_rows = [row for row in boot_rows if row.get("row_kind") in ("pairwise", "pairwise_ratio")]
    _write_csv(out_dir / "pairwise_vs_score_only.csv", _union_fields(pair_rows), pair_rows)
    stage("bootstrap", t0)

    t0 = time.perf_counter()
    ood_rows = _ood_stability_rows(results)
    _write_csv(out_dir / "ood_stability.csv", _union_fields(ood_rows), ood_rows)
    subset_rows = _error_subset_rows(results, sem_stats)
    _write_csv(out_dir / "error_subsets.csv", _union_fields(subset_rows), subset_rows)
    matched_rows = _matched_score_rows(results, sem_stats, replicates=int(cfg["bootstrap_replicates"]),
                                       seed=int(cfg["bootstrap_seed"]), ci=float(cfg["ci"]))
    _write_csv(out_dir / "matched_score_analysis.csv", _union_fields(matched_rows), matched_rows)
    coef_rows = _coefficient_rows(results)
    _write_csv(out_dir / "e1_logistic" / "coefficients.csv", _union_fields(coef_rows), coef_rows)
    stage("analyses", t0)

    # -- stage 6: figures / gate / metadata ---------------------------------
    t0 = time.perf_counter()
    figures = _figure_k_curves(out_dir, results)
    figures["figure_C"] = _figure_quartiles(out_dir, subset_rows)
    gate = _gate_stage(results, pair_cache, replicates=int(cfg["bootstrap_replicates"]),
                       seed=int(cfg["bootstrap_seed"]), ci=float(cfg["ci"]), log=log)
    _write_json(out_dir / "sufficiency_gate.json", gate)
    stage("figures+gate", t0)

    headline = _headline_table(results)
    metadata = {
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.perf_counter() - started,
        "stage_seconds": stage_time,
        "config": cfg,
        "seeds": list(args.seeds),
        "e0_reproduction_checks": {res["scorer"]: res["e0_checks"] for res in results},
        "selections": {res["scorer"]: {"best_score_only": res["best_score_only"],
                                        "best_semantic": res["best_semantic"],
                                        "tune_mean_auroc": res["tune_auroc"],
                                        "selected_hp": res["selected_hp"],
                                        "params": res["params"]} for res in results},
        "headline_pooled": headline,
        "gate_verdict": gate["verdict"]["verdict"],
        "gate_comparison": gate["comparison"],
        "figures": figures,
        "files": written_files + ["aggregate.csv", "bootstrap.csv", "pairwise_vs_score_only.csv",
                                  "error_subsets.csv", "matched_score_analysis.csv", "ood_stability.csv",
                                  "semantic_stats.csv", "sufficiency_gate.json", "protocol.json"],
    }
    _write_json(out_dir / "metadata.json", metadata)
    log(f"[phase1] done in {metadata['runtime_seconds']:.1f}s; gate verdict = {metadata['gate_verdict']}")
    print(f"PHASE1_COMPLETE verdict={metadata['gate_verdict']} out={out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
