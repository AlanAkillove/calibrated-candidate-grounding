"""Frozen score-matrix loading and the Phase 0.5 reliability split (Amendment A6).

This module is the *data* half of the Phase 0.5 score-information sufficiency
audit.  It reads the frozen grounding-scorer score matrices that Phase 0 already
produced and never re-trains anything:

* **B3 independent MLP** -- ``results/phase0b_independent/seed_{1,2,3}/raw_scores/K{k}.npz``
  (keys ``scores [n,k] float32``, ``sentence_id``/``ref_id``/``image_id`` ``int64``,
  ``eval_split <U10``, ``target_local int32``; the ``n`` common-cohort rows are in
  the same order as ``per_sentence_predictions.npz``).
* **B1 frozen cosine** -- ``results/phase0a_cosine/raw_predictions/K{k}.npz``
  (keys ``raw_scores``, ``sentence_ids``/``ref_ids``/``image_ids`` ``int64``,
  ``split_codes int8``, ``target_local int32``).  Its rows are filtered down to
  the B3 common cohort -- the ``sentence_id`` set of ``seed_1/raw_scores/K5.npz``.

The split is frozen by Amendment A6.2: the ``val_calib`` images are partitioned
at the *image* level into ``reliability_train`` / ``reliability_tune`` with a
deterministic permutation (``seed=20260928``, ``train_frac=0.70``) written to a
manifest *before* any model runs.

Nothing here reads candidate embeddings, geometry, objectness or GT metadata:
only the scalar score rows and their identities (Amendment A6.7).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from ccg.experiment.phase0a import SPLIT_CODES

__all__ = [
    "B3_ROOT",
    "COSINE_ROOT",
    "DEFAULT_SPLIT_SEED",
    "DEFAULT_TRAIN_FRAC",
    "DEFAULT_VAL_SPLIT_NAME",
    "SCORER_IDS",
    "ReliabilitySplit",
    "ScorerScores",
    "build_reliability_split",
    "common_sentence_ids",
    "load_scorer_scores",
    "write_split_manifest",
]

#: Frozen scorer identifiers (Amendment A6.1): 3 B3 MLP seeds + frozen cosine.
SCORER_IDS: tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3", "cosine")
#: Frozen split seed / train fraction (Amendment A6.2).
DEFAULT_SPLIT_SEED = 20260928
DEFAULT_TRAIN_FRAC = 0.70
#: The reliability split is carved out of this eval split only.
DEFAULT_VAL_SPLIT_NAME = "val_calib"
#: Default on-disk roots (relative to the repository working directory).
B3_ROOT = Path("results/phase0b_independent")
COSINE_ROOT = Path("results/phase0a_cosine")

#: scorer_id -> B3 seed directory name.
_B3_SEED_DIR: Mapping[str, str] = {
    "b3_seed1": "seed_1",
    "b3_seed2": "seed_2",
    "b3_seed3": "seed_3",
}
#: split code (int8 stored in the cosine npz) -> eval split name.
_CODE_TO_SPLIT: Mapping[int, str] = {int(code): name for name, code in SPLIT_CODES.items()}
#: sub-directory names of the two artifact families.
_B3_RAW_SCORES_DIRNAME = "raw_scores"
_COSINE_RAW_PREDICTIONS_DIRNAME = "raw_predictions"
#: the two split kinds a :class:`ReliabilitySplit` can select rows for.
_MASK_KINDS: tuple[str, ...] = ("reliability_train", "reliability_tune")


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScorerScores:
    """One frozen scorer's score matrix at one ``K`` over the common cohort.

    Attributes
    ----------
    scorer_id / k:
        The scorer (one of :data:`SCORER_IDS`) and the candidate-set size.
    scores:
        ``[n, k]`` ``float32`` candidate scores; higher = more likely the target.
    sentence_id / ref_id / image_id:
        ``int64`` identities of each row (the resampling cluster is ``image_id``).
    eval_split:
        ``<U10`` eval-split name per row (``val_select``/``val_calib``/``testA``/
        ``testB``); on the cosine side it is decoded from the stored code.
    target_local:
        ``int32`` local index of the target inside each row's candidate set.
    """

    scorer_id: str
    k: int
    scores: NDArray[np.float32]
    sentence_id: NDArray[np.int64]
    ref_id: NDArray[np.int64]
    image_id: NDArray[np.int64]
    eval_split: NDArray[np.str_]
    target_local: NDArray[np.int32]

    @property
    def correct(self) -> NDArray[np.bool_]:
        """``argmax(scores, axis=1) == target_local``: the top-1 correctness event."""
        return np.argmax(self.scores, axis=1) == self.target_local

    def __len__(self) -> int:
        return int(self.scores.shape[0])


@dataclass(frozen=True)
class ReliabilitySplit:
    """Image-level ``reliability_train`` / ``reliability_tune`` partition of ``val_calib``.

    The partition is deterministic and frozen (Amendment A6.2): identical for a
    given ``seed`` and never re-drawn between runs.
    """

    seed: int
    train_frac: float
    train_images: NDArray[np.int64]
    tune_images: NDArray[np.int64]

    def row_mask(
        self,
        eval_split: NDArray[np.str_],
        image_id: NDArray[np.int64],
        *,
        kind: str,
    ) -> NDArray[np.bool_]:
        """Boolean row selector for ``reliability_train`` / ``reliability_tune``.

        ``mask = (eval_split == "val_calib") & (image_id in <that side's images>)``.
        Only those two ``kind`` values are accepted; anything else raises.
        """
        if kind not in _MASK_KINDS:
            raise ValueError(
                f"unknown split kind {kind!r}; expected one of {_MASK_KINDS}"
            )
        split = np.asarray(eval_split)
        ids = np.asarray(image_id, dtype=np.int64)
        if split.shape[0] != ids.shape[0]:
            raise ValueError(
                f"eval_split ({split.shape[0]}) and image_id ({ids.shape[0]}) differ in length"
            )
        images = self.train_images if kind == "reliability_train" else self.tune_images
        return (split == DEFAULT_VAL_SPLIT_NAME) & np.isin(ids, images)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary (image id lists included)."""
        return {
            "seed": int(self.seed),
            "train_frac": float(self.train_frac),
            "n_train": int(self.train_images.size),
            "n_tune": int(self.tune_images.size),
            "train_images": [int(x) for x in self.train_images],
            "tune_images": [int(x) for x in self.tune_images],
        }


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def common_sentence_ids(*, b3_root: Path = B3_ROOT) -> NDArray[np.int64]:
    """Sorted unique common-cohort ``sentence_id`` set (Amendment A6.1).

    Read from the B3 ``seed_1/raw_scores/K5.npz`` rows; the cosine rows are
    filtered to exactly this set.
    """
    path = (
        Path(b3_root) / _B3_SEED_DIR["b3_seed1"] / _B3_RAW_SCORES_DIRNAME / "K5.npz"
    )
    with np.load(path) as data:
        ids = np.asarray(data["sentence_id"], dtype=np.int64).reshape(-1)
    return np.unique(ids)


def _decode_split_codes(split_codes: NDArray) -> NDArray[np.str_]:
    """Map stored ``int8`` split codes to ``<U10`` eval-split names."""
    codes = np.asarray(split_codes).reshape(-1)
    out = np.empty(codes.shape[0], dtype="<U10")
    for code in np.unique(codes):
        name = _CODE_TO_SPLIT.get(int(code))
        if name is None:
            raise ValueError(f"unknown split code {int(code)!r}; expected keys {sorted(_CODE_TO_SPLIT)}")
        out[codes == code] = name
    return out


def _load_b3(scorer_id: str, k: int, b3_root: Path) -> ScorerScores:
    path = b3_root / _B3_SEED_DIR[scorer_id] / _B3_RAW_SCORES_DIRNAME / f"K{k}.npz"
    with np.load(path) as data:
        scores = np.asarray(data["scores"], dtype=np.float32)
        sentence_id = np.asarray(data["sentence_id"], dtype=np.int64)
        ref_id = np.asarray(data["ref_id"], dtype=np.int64)
        image_id = np.asarray(data["image_id"], dtype=np.int64)
        eval_split = np.asarray(data["eval_split"])
        target_local = np.asarray(data["target_local"], dtype=np.int32)
    return ScorerScores(
        scorer_id=scorer_id,
        k=int(k),
        scores=scores,
        sentence_id=sentence_id,
        ref_id=ref_id,
        image_id=image_id,
        eval_split=eval_split,
        target_local=target_local,
    )


def _load_cosine(
    scorer_id: str,
    k: int,
    cosine_root: Path,
    common_ids: Optional[NDArray[np.int64]],
) -> ScorerScores:
    path = cosine_root / _COSINE_RAW_PREDICTIONS_DIRNAME / f"K{k}.npz"
    with np.load(path) as data:
        scores = np.asarray(data["raw_scores"], dtype=np.float32)
        sentence_id = np.asarray(data["sentence_ids"], dtype=np.int64)
        ref_id = np.asarray(data["ref_ids"], dtype=np.int64)
        image_id = np.asarray(data["image_ids"], dtype=np.int64)
        split_codes = np.asarray(data["split_codes"])
        target_local = np.asarray(data["target_local"], dtype=np.int32)

    common = (
        np.unique(np.asarray(common_ids, dtype=np.int64))
        if common_ids is not None
        else common_sentence_ids()
    )
    keep = np.isin(sentence_id, common)
    scores = scores[keep]
    sentence_id = sentence_id[keep]
    ref_id = ref_id[keep]
    image_id = image_id[keep]
    split_codes = split_codes[keep]
    target_local = target_local[keep]

    missing = np.setdiff1d(common, sentence_id)
    if missing.size:
        raise ValueError(
            f"cosine scorer {scorer_id!r} K={k}: {int(missing.size)} common-cohort "
            f"sentence_id(s) missing after filtering (first few: {missing[:10].tolist()})"
        )
    eval_split = _decode_split_codes(split_codes)
    return ScorerScores(
        scorer_id=scorer_id,
        k=int(k),
        scores=scores,
        sentence_id=sentence_id,
        ref_id=ref_id,
        image_id=image_id,
        eval_split=eval_split,
        target_local=target_local,
    )


def load_scorer_scores(
    scorer_id: str,
    k: int,
    *,
    b3_root: Path = B3_ROOT,
    cosine_root: Path = COSINE_ROOT,
    common_ids: Optional[NDArray[np.int64]] = None,
) -> ScorerScores:
    """Load one frozen scorer's ``[n, k]`` score matrix over the common cohort.

    ``b3_seed{i}`` reads ``{b3_root}/seed_i/raw_scores/K{k}.npz`` verbatim.
    ``cosine`` reads ``{cosine_root}/raw_predictions/K{k}.npz`` and filters its
    rows to ``common_ids`` (defaulting to :func:`common_sentence_ids`); the filter
    must be exact - a single missing common-cohort row raises :class:`ValueError`.
    """
    if scorer_id not in SCORER_IDS:
        raise ValueError(f"unknown scorer_id {scorer_id!r}; expected one of {SCORER_IDS}")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if scorer_id.startswith("b3_"):
        return _load_b3(scorer_id, k, Path(b3_root))
    return _load_cosine(scorer_id, k, Path(cosine_root), common_ids)


# ---------------------------------------------------------------------------
# split construction
# ---------------------------------------------------------------------------
def build_reliability_split(
    eval_split: NDArray[np.str_],
    image_id: NDArray[np.int64],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    train_frac: float = DEFAULT_TRAIN_FRAC,
) -> ReliabilitySplit:
    """Build the frozen image-level split of the ``val_calib`` images.

    ``images = sorted(unique(image_id[eval_split == "val_calib"]))``; a
    deterministic permutation (``np.random.default_rng(seed)``) assigns the first
    ``round(train_frac * n_images)`` images to ``reliability_train`` and the rest
    to ``reliability_tune``; both sides are stored sorted ``int64``.  Disjoint and
    exhaustive by construction, and identical across runs for a fixed ``seed``.
    """
    split = np.asarray(eval_split)
    ids = np.asarray(image_id, dtype=np.int64)
    if split.shape[0] != ids.shape[0]:
        raise ValueError(
            f"eval_split ({split.shape[0]}) and image_id ({ids.shape[0]}) differ in length"
        )
    images = np.unique(ids[split == DEFAULT_VAL_SPLIT_NAME])
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(images.shape[0])
    n_train = int(round(float(train_frac) * int(images.shape[0])))
    train = np.sort(images[perm[:n_train]]).astype(np.int64, copy=False)
    tune = np.sort(images[perm[n_train:]]).astype(np.int64, copy=False)
    return ReliabilitySplit(
        seed=int(seed),
        train_frac=float(train_frac),
        train_images=train,
        tune_images=tune,
    )


def _ids_sha256(ids: NDArray[np.int64]) -> str:
    """``sha256`` of the comma-joined sorted image ids (stable provenance hash)."""
    ordered = np.sort(np.asarray(ids, dtype=np.int64))
    joined = ",".join(str(int(x)) for x in ordered)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def write_split_manifest(
    path: Path,
    split: ReliabilitySplit,
    *,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Write the split manifest JSON and return the exact payload written.

    The payload carries ``seed`` / ``train_frac`` / ``n_images`` / ``n_train`` /
    ``n_tune``, the two image-id lists, a ``sha256`` of the comma-joined sorted
    ids for each side and ``created_utc``.  ``extra`` (if given) is merged in.
    """
    payload: dict[str, Any] = {
        "seed": int(split.seed),
        "train_frac": float(split.train_frac),
        "n_images": int(split.train_images.size + split.tune_images.size),
        "n_train": int(split.train_images.size),
        "n_tune": int(split.tune_images.size),
        "train_images": [int(x) for x in split.train_images],
        "tune_images": [int(x) for x in split.tune_images],
        "sha256_train": _ids_sha256(split.train_images),
        "sha256_tune": _ids_sha256(split.tune_images),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(dict(extra))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return payload
