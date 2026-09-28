"""Frozen-cohort semantic embeddings for Phase 1 (protocol A7).

This module owns the *data* half of the Phase 1 audit:

* it materialises the cached L2-normalised OpenCLIP query/candidate embeddings
  of every row of the Phase 0.5 canonical cohort through
  :class:`ccg.models.b3_data.B3Corpus` (no re-extraction, no CLIP forward);
* it stores them as one ``.npz`` per ``K`` under ``cache/semantic_phase1/``
  (float16, ``[n, 512]`` query + ``[n, K, 512]`` candidate blocks whose local
  order is exactly the frozen ``C_K`` layout - target at local index 0);
* it loads the canonical frozen score matrices (:mod:`ccg.reliability.data`)
  and the frozen reliability split manifest, with alignment assertions.

Nothing here trains, scores or reranks anything: the B3 raw scores and their
argmax are consumed as frozen inputs only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..experiment import phase0a
from ..models.b3_data import B3Corpus
from ..reliability import data as rdata
from ..reliability import features as rfeat

__all__ = [
    "DEFAULT_B3_ROOT",
    "DEFAULT_OUT_ROOT",
    "EMBEDDING_INDEX_NAME",
    "FEATURE_DIM",
    "PHASE05_FEATURES_DIR",
    "EmbeddingStore",
    "build_embedding_store",
    "embedding_file",
    "embedding_provenance",
    "load_embedding_store",
    "load_phase05_cohort",
    "load_scorer_canonical",
    "load_split_from_manifest",
    "score_stat_block",
    "corrected_probs",
    "assert_split_matches_fresh",
]

#: OpenCLIP ViT-B/32 joint embedding width (frozen).
FEATURE_DIM = 512
#: Where the per-K embedding archives live (gitignored; buildable at any time).
DEFAULT_OUT_ROOT = Path("cache/semantic_phase1")
#: Provenance index next to the archives.
EMBEDDING_INDEX_NAME = "embedding_index.json"
#: Frozen Phase 0.5 feature directory holding the canonical cohort rows.
PHASE05_FEATURES_DIR = Path("results/phase05_score_sufficiency/features")
#: Frozen Phase 0B artifact root (raw scores / target_local).
DEFAULT_B3_ROOT = rdata.B3_ROOT


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingStore:
    """One ``K``'s frozen-cohort query/candidate embeddings (canonical rows).

    Attributes
    ----------
    k:
        Candidate-set size of the stored blocks.
    sentence_id / ref_id / image_id / eval_split / target_local:
        Row identities, aligned 1:1 with the Phase 0.5 canonical cohort
        (``sentence_id`` ascending, unique).
    z_q:
        ``[n, 512]`` float16 query embeddings (L2-normalised, cached values).
    z_i:
        ``[n, K, 512]`` float16 candidate embeddings; local index 0 is the
        target (frozen ``C_K`` layout).
    """

    k: int
    sentence_id: np.ndarray
    ref_id: np.ndarray
    image_id: np.ndarray
    eval_split: np.ndarray
    target_local: np.ndarray
    z_q: np.ndarray
    z_i: np.ndarray

    def __len__(self) -> int:
        return int(self.sentence_id.shape[0])


# ---------------------------------------------------------------------------
# cohort / score loaders
# ---------------------------------------------------------------------------
def load_phase05_cohort(
    k: int,
    *,
    features_dir: Path = PHASE05_FEATURES_DIR,
    anchor_scorer: str = "b3_seed1",
) -> Dict[str, np.ndarray]:
    """Canonical cohort rows of one ``K`` from the frozen Phase 0.5 features.

    Returns ``sentence_id`` (ascending, unique) plus the matching ``ref_id`` /
    ``image_id`` / ``eval_split`` arrays and the row-wise B3 correctness.  This
    is the row universe every Phase 1 artifact must align with.
    """
    path = Path(features_dir) / f"{anchor_scorer}.npz"
    if not path.exists():
        raise FileNotFoundError(f"phase 0.5 features not found: {path}")
    with np.load(path) as data:
        out = {
            "sentence_id": np.asarray(data[f"sentence_id_K{k}"], dtype=np.int64),
            "ref_id": np.asarray(data[f"ref_id_K{k}"], dtype=np.int64),
            "image_id": np.asarray(data[f"image_id_K{k}"], dtype=np.int64),
            "eval_split": np.asarray(data[f"eval_split_K{k}"]),
            "correct": np.asarray(data[f"correct_K{k}"], dtype=bool),
        }
    sid = out["sentence_id"]
    if sid.size == 0:
        raise ValueError(f"empty cohort for K={k} in {path}")
    if not np.all(np.diff(sid) > 0):
        raise ValueError(f"{path}: sentence_id_K{k} is not strictly ascending")
    return out


def load_scorer_canonical(
    scorer_id: str,
    k: int,
    *,
    b3_root: Path = DEFAULT_B3_ROOT,
    cosine_root: Path = rdata.COSINE_ROOT,
) -> rdata.ScorerScores:
    """One frozen scorer's rows in canonical order (``sentence_id`` ascending).

    Mirrors the Phase 0.5 loader verbatim: load, stable-sort by ``sentence_id``,
    reject duplicates - so every downstream array is aligned with the embedding
    store rows.
    """
    sample = rdata.load_scorer_scores(scorer_id, k, b3_root=Path(b3_root), cosine_root=Path(cosine_root))
    order = np.argsort(sample.sentence_id, kind="stable")
    sid = sample.sentence_id[order]
    if not np.all(np.diff(sid) > 0):
        raise ValueError(f"{scorer_id} K={k}: duplicate sentence ids after canonical sort")
    return rdata.ScorerScores(
        scorer_id=sample.scorer_id,
        k=int(sample.k),
        scores=np.asarray(sample.scores, dtype=np.float32)[order],
        sentence_id=sid,
        ref_id=sample.ref_id[order],
        image_id=sample.image_id[order],
        eval_split=sample.eval_split[order],
        target_local=sample.target_local[order],
    )


def _ids_sha256(ids: np.ndarray) -> str:
    """Same provenance hash as :func:`ccg.reliability.data.write_split_manifest`."""
    ordered = np.sort(np.asarray(ids, dtype=np.int64))
    joined = ",".join(str(int(x)) for x in ordered)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_split_from_manifest(path: Path) -> rdata.ReliabilitySplit:
    """Rebuild the frozen A6.2 split from its manifest, verifying both hashes.

    The manifest is the single source of truth for Phase 1: the split is never
    re-drawn, so this loader only reads and verifies.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = np.asarray(sorted(int(x) for x in payload["train_images"]), dtype=np.int64)
    tune = np.asarray(sorted(int(x) for x in payload["tune_images"]), dtype=np.int64)
    if int(payload.get("n_train", train.size)) != train.size:
        raise ValueError(f"{path}: n_train disagrees with the stored list")
    if int(payload.get("n_tune", tune.size)) != tune.size:
        raise ValueError(f"{path}: n_tune disagrees with the stored list")
    for side, ids in (("train", train), ("tune", tune)):
        stored = str(payload.get(f"sha256_{side}", ""))
        if stored and stored != _ids_sha256(ids):
            raise ValueError(f"{path}: sha256_{side} does not match the stored ids")
    overlap = np.intersect1d(train, tune)
    if overlap.size:
        raise ValueError(f"{path}: train/tune overlap on {overlap.size} image ids")
    return rdata.ReliabilitySplit(
        seed=int(payload["seed"]),
        train_frac=float(payload["train_frac"]),
        train_images=train,
        tune_images=tune,
    )


def assert_split_matches_fresh(
    split: rdata.ReliabilitySplit,
    eval_split: np.ndarray,
    image_id: np.ndarray,
) -> None:
    """Check the manifest split equals a fresh deterministic rebuild (A6.2).

    Raises :class:`AssertionError` on any mismatch: Phase 1 must reuse the
    Phase 0.5 partition exactly, never a re-draw.
    """
    fresh = rdata.build_reliability_split(eval_split, image_id, seed=split.seed, train_frac=split.train_frac)
    if not np.array_equal(fresh.train_images, split.train_images):
        raise AssertionError("reliability_train images differ from the fresh A6.2 rebuild")
    if not np.array_equal(fresh.tune_images, split.tune_images):
        raise AssertionError("reliability_tune images differ from the fresh A6.2 rebuild")


# ---------------------------------------------------------------------------
# score-derived inputs (frozen scalars for E2 / E3)
# ---------------------------------------------------------------------------
def score_stat_block(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    """``[n, 4]`` frozen score statistics ``[top1, top2, margin12, msp]``.

    ``top1/top2/margin12`` are raw score units (``s_ (1)``, ``s_ (2)``, their
    gap) and ``msp`` is ``max_i softmax(scores / T)[i]`` - the exact Phase 0.5
    definitions (:func:`ccg.reliability.features.scalar_confidence`).
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError(f"scores must be [n, K>=2], got {matrix.shape}")
    scales = rfeat.scalar_confidence(matrix, temperature=float(temperature))
    ordered = np.sort(matrix, axis=1)[:, ::-1]
    return np.column_stack(
        [ordered[:, 0], ordered[:, 1], scales["margin"], scales["msp"]]
    ).astype(np.float64)


def corrected_probs(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    """Row-wise ``softmax(scores / T)`` as ``[n, K]`` float64 (stable)."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be [n, K], got {matrix.shape}")
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    shifted = matrix / temp
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# embedding store build / load
# ---------------------------------------------------------------------------
def embedding_file(k: int, *, out_root: Path = DEFAULT_OUT_ROOT) -> Path:
    """Path of one ``K``'s embedding archive under ``out_root``."""
    return Path(out_root) / f"embeddings_K{int(k)}.npz"


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _update_index(out_root: Path, k: int, entry: Mapping[str, Any]) -> None:
    """Read-modify-write the provenance index (atomic)."""
    path = Path(out_root) / EMBEDDING_INDEX_NAME
    payload: Dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload[f"K{int(k)}"] = dict(entry)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def build_embedding_store(
    k: int,
    *,
    out_root: Path = DEFAULT_OUT_ROOT,
    features_root: Path = Path("cache/features"),
    manifests_root: Path = Path("cache/manifests"),
    refs: Any = Path("data/raw/refcoco+/refcoco+/refs(unc).p"),
    bank_path: Path = Path("cache/proposals.h5"),
    image_sizes_path: Path = Path("cache/image_sizes.npz"),
    phase05_features_dir: Path = PHASE05_FEATURES_DIR,
    b3_root: Path = DEFAULT_B3_ROOT,
    corpus: Optional[B3Corpus] = None,
    overwrite: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Path:
    """Materialise the ``K`` embedding archive from the frozen feature cache.

    For every row of the Phase 0.5 canonical cohort (ascending ``sentence_id``)
    the corresponding Phase 0A record is materialised through
    :meth:`ccg.models.b3_data.B3Corpus.example_for`, which gathers the stored
    region/text blocks in the frozen ``C_K`` local order (target at index 0).
    The result is verified row-by-row against the cohort identities and stored
    with provenance hashes.

    The internal corpus is opened with ``preload=True`` (region/text/bank read
    once sequentially, ~1.4 GiB RAM): the on-demand LRU path costs ~15 ms per
    image on the chunked h5, which would make the full build ~1 h per ``K``.

    Idempotent: an existing archive is returned untouched unless ``overwrite``.
    """
    k = int(k)
    out_path = embedding_file(k, out_root=Path(out_root))
    if out_path.exists() and not overwrite:
        if log is not None:
            log(f"[semantic] embeddings K={k} already present at {out_path}")
        return out_path

    cohort = load_phase05_cohort(k, features_dir=Path(phase05_features_dir))
    n = int(cohort["sentence_id"].size)
    if log is not None:
        log(f"[semantic] building embeddings K={k}: {n} rows")

    if corpus is None:
        corpus = B3Corpus(
            features_root,
            manifests_root,
            refs,
            bank_path,
            image_sizes_path=image_sizes_path,
            ks=(k,),
            regime="random",
            preload=True,
        )
    splits = tuple(phase0a.PRIMARY_SPLITS)
    records = corpus.eval_records(splits, (k,))
    by_sid: Dict[int, Any] = {int(rec.sentence_id): rec for rec in records}
    missing = [int(s) for s in cohort["sentence_id"] if int(s) not in by_sid]
    if missing:
        raise ValueError(
            f"K={k}: {len(missing)} cohort sentence(s) absent from the phase-0A record set "
            f"(first few: {missing[:5]})"
        )

    z_q = np.empty((n, FEATURE_DIM), dtype=np.float16)
    z_i = np.empty((n, k, FEATURE_DIM), dtype=np.float16)
    target_local = np.zeros(n, dtype=np.int32)

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - tqdm is a hard dep of the repo CLI path
        tqdm = None  # type: ignore[assignment]
    iterator = range(n)
    if tqdm is not None:
        iterator = tqdm(range(n), total=n, desc=f"embeddings K={k}", unit="row", disable=None)

    for i in iterator:
        sid = int(cohort["sentence_id"][i])
        record = by_sid[sid]
        if int(record.image_id) != int(cohort["image_id"][i]):
            raise AssertionError(f"K={k} row {i}: image_id mismatch vs cohort ({sid})")
        if str(record.eval_split) != str(cohort["eval_split"][i]):
            raise AssertionError(f"K={k} row {i}: eval_split mismatch vs cohort ({sid})")
        example = corpus.example_for(record, k)
        candidates = np.asarray(example.candidate_features, dtype=np.float16)
        if candidates.shape != (k, FEATURE_DIM):
            raise AssertionError(
                f"K={k} row {i}: candidate block {candidates.shape} != {(k, FEATURE_DIM)}"
            )
        z_q[i] = np.asarray(example.query_feature, dtype=np.float16)
        z_i[i] = candidates

    # -- sanity: cached embeddings are L2-normalised; target sits at local 0
    for name, block in (("z_q", z_q.astype(np.float64)), ("z_i", z_i.astype(np.float64))):
        norms = np.linalg.norm(block, axis=-1)
        worst = float(np.abs(norms - 1.0).max())
        if worst > 2e-3:
            raise AssertionError(f"K={k}: {name} not L2-normalised (max |1-norm| = {worst})")

    payload = {
        "sentence_id": cohort["sentence_id"],
        "ref_id": cohort["ref_id"],
        "image_id": cohort["image_id"],
        "eval_split": cohort["eval_split"],
        "target_local": target_local,
        "z_q": z_q,
        "z_i": z_i,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp.npz")
    with open(tmp, "wb") as handle:
        np.savez(handle, **payload)
    os.replace(str(tmp), str(out_path))

    entry = {
        "k": k,
        "file": out_path.name,
        "n": n,
        "z_q_sha256": _sha256_array(z_q),
        "z_i_sha256": _sha256_array(z_i),
        "sentence_id_sha256": _sha256_array(cohort["sentence_id"]),
        "dtype": "float16",
        "feature_dim": FEATURE_DIM,
        "source": {
            "features_root": str(features_root),
            "manifests_root": str(manifests_root),
            "refs": str(refs),
            "bank_path": str(bank_path),
            "phase05_features_dir": str(phase05_features_dir),
            "b3_root": str(b3_root),
        },
        "local_layout": "C_K = [target] + distractor_order[:K-1]; target_local == 0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _update_index(Path(out_root), k, entry)
    if log is not None:
        log(f"[semantic] wrote {out_path} ({n} rows x K={k})")
    return out_path


def load_embedding_store(k: int, *, out_root: Path = DEFAULT_OUT_ROOT) -> EmbeddingStore:
    """Load one ``K`` embedding archive (float16 blocks stay as stored)."""
    path = embedding_file(k, out_root=Path(out_root))
    if not path.exists():
        raise FileNotFoundError(
            f"embedding archive {path} not found; build it with build_embedding_store()"
        )
    with np.load(path) as data:
        store = EmbeddingStore(
            k=int(k),
            sentence_id=np.asarray(data["sentence_id"], dtype=np.int64),
            ref_id=np.asarray(data["ref_id"], dtype=np.int64),
            image_id=np.asarray(data["image_id"], dtype=np.int64),
            eval_split=np.asarray(data["eval_split"]),
            target_local=np.asarray(data["target_local"], dtype=np.int32),
            z_q=np.asarray(data["z_q"], dtype=np.float16),
            z_i=np.asarray(data["z_i"], dtype=np.float16),
        )
    if store.z_q.shape != (len(store), FEATURE_DIM):
        raise ValueError(f"{path}: z_q shape {store.z_q.shape} inconsistent with n={len(store)}")
    if store.z_i.shape != (len(store), int(k), FEATURE_DIM):
        raise ValueError(f"{path}: z_i shape {store.z_i.shape} inconsistent with K={k}")
    if not np.all(np.diff(store.sentence_id) > 0):
        raise ValueError(f"{path}: sentence_id not strictly ascending")
    return store


def embedding_provenance(k: int, *, out_root: Path = DEFAULT_OUT_ROOT) -> Dict[str, Any]:
    """Provenance entry of one ``K`` archive from the build index (or ``{}``)."""
    path = Path(out_root) / EMBEDDING_INDEX_NAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get(f"K{int(k)}", {}))
