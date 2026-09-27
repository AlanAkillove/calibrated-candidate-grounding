"""Candidate-set construction (protocol sections 5 and 6).

The cardinality ``K`` and the hardness of a set must vary as independently as
possible, which is why the primary constructor produces **nested** sets::

    C_5 subset C_10 subset C_20 subset C_50      (same target in every set)

so an increase of ``K`` can only ever come from *additional distractors*, never
from a different target proposal.

Everything in this module is pure numpy: no torch, no CLIP, no file IO, no GPU.
Hard-negative constructors consume *already cached* embeddings and the IoU
metadata stored in a :class:`~ccg.data.types.ProposalBank`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .proposals import iou_matrix
from .types import CandidateSet, ProposalBank

__all__ = [
    "HARDNESS_RANDOM",
    "HARDNESS_SAME_CATEGORY",
    "HARDNESS_CLIP",
    "DEFAULT_KS",
    "CandidateShortage",
    "CandidateAvailability",
    "NestedCandidateSets",
    "build_nested_random_sets",
    "build_ranked_nested_sets",
    "select_same_category_hard_negatives",
    "build_same_category_hard_sets",
    "select_clip_hard_negatives",
    "build_clip_hard_sets",
    "synthetic_omit",
    "assert_nested",
]

HARDNESS_RANDOM = "random"
HARDNESS_SAME_CATEGORY = "same_category_hard"
HARDNESS_CLIP = "clip_hard"

#: Phase 0 test grid (section 10): train K in {5,10}, test K in {5,10,20,50}.
DEFAULT_KS: tuple[int, ...] = (5, 10, 20, 50)

_EPS = 1e-8


# ---------------------------------------------------------------------------
# availability bookkeeping (section 10: never silently filter hard samples)
# ---------------------------------------------------------------------------
@dataclass
class CandidateShortage:
    """One request that could not be satisfied because the pool was too small."""

    ref_id: int
    requested_K: int
    pool_size: int
    max_possible_K: int

    def to_dict(self) -> dict:
        return {
            "ref_id": self.ref_id,
            "requested_K": self.requested_K,
            "pool_size": self.pool_size,
            "max_possible_K": self.max_possible_K,
        }


@dataclass
class CandidateAvailability:
    """Aggregate "could we build C_K?" statistics over many requests.

    ``missing[K]`` counts requests that asked for ``K`` candidates but whose
    proposal pool could not supply them; ``counts[K]`` the successful ones.  A
    report is *always* attached to a build result so that the Phase 0 write-up
    can state the candidate availability distribution before picking a maximum
    ``K`` (protocol: do not silently drop the difficult examples).
    """

    requested_Ks: tuple[int, ...] = DEFAULT_KS
    counts: Dict[int, int] = field(default_factory=dict)
    missing: Dict[int, int] = field(default_factory=dict)
    shortages: List[CandidateShortage] = field(default_factory=list)
    num_requests: int = 0
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.requested_Ks = tuple(int(k) for k in self.requested_Ks)
        for k in self.requested_Ks:
            self.counts.setdefault(k, 0)
            self.missing.setdefault(k, 0)

    def record(self, shortage: CandidateShortage) -> None:
        self.missing[shortage.requested_K] = self.missing.get(shortage.requested_K, 0) + 1
        self.shortages.append(shortage)

    def succeeded(self, k: int) -> None:
        self.counts[k] = self.counts.get(k, 0) + 1

    def note(self, message: str) -> None:
        """Record a non-fatal observation (e.g. "not enough same-category negatives")."""
        self.notes.append(message)

    @property
    def availability(self) -> Dict[int, float]:
        """``K`` -> fraction of requests that produced a usable set."""
        total = max(self.num_requests, 1)
        return {int(k): self.counts.get(int(k), 0) / total for k in self.requested_Ks}

    def merge(self, other: "CandidateAvailability") -> "CandidateAvailability":
        """Fold another report into this one (same ``requested_Ks`` expected)."""
        if tuple(other.requested_Ks) != tuple(self.requested_Ks):
            raise ValueError("cannot merge availability reports with different requested_Ks")
        for k in self.requested_Ks:
            self.counts[k] = self.counts.get(k, 0) + other.counts.get(k, 0)
            self.missing[k] = self.missing.get(k, 0) + other.missing.get(k, 0)
        self.shortages.extend(other.shortages)
        self.num_requests += other.num_requests
        return self

    def to_dict(self) -> dict:
        return {
            "requested_Ks": list(self.requested_Ks),
            "num_requests": self.num_requests,
            "counts": {str(k): int(v) for k, v in sorted(self.counts.items())},
            "missing": {str(k): int(v) for k, v in sorted(self.missing.items())},
            "availability": {str(k): float(v) for k, v in sorted(self.availability.items())},
            "example_shortages": [s.to_dict() for s in self.shortages[:20]],
            "num_shortages": len(self.shortages),
            "notes": list(self.notes),
        }


class NestedCandidateSets(dict):
    """``dict`` of ``K -> CandidateSet | None`` plus its availability report.

    It *is* a plain dict (so ``result[10]`` and ``result.items()`` work), but it
    also carries ``.availability`` so a shortage can never be lost on the way to
    the reporting layer.  ``None`` values mark sizes whose pool was too small -
    callers must report them, not filter them out.
    """

    availability: CandidateAvailability

    def __init__(self, *args, availability: Optional[CandidateAvailability] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.availability = availability if availability is not None else CandidateAvailability()

    @property
    def max_built_K(self) -> Optional[int]:
        built = [k for k, value in self.items() if value is not None]
        return max(built) if built else None

    def built(self) -> Dict[int, CandidateSet]:
        return {int(k): v for k, v in self.items() if v is not None}


# ---------------------------------------------------------------------------
# ranking helpers
# ---------------------------------------------------------------------------
def _rng(rng: np.random.Generator | np.random.RandomState | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(rng.randint(0, 2**31 - 1))
    if rng is None:
        return np.random.default_rng(0)
    return np.random.default_rng(int(rng))


def _dedup_pool(pool: np.ndarray, target_idx: int) -> np.ndarray:
    """Drop the target (and repeats) from a distractor pool, keeping order."""
    keep: List[int] = []
    seen = set()
    for raw in np.asarray(pool, dtype=np.int64).reshape(-1).tolist():
        index = int(raw)
        if index == int(target_idx) or index in seen:
            continue
        if index < 0:
            raise ValueError(f"pool contains a negative proposal index: {index}")
        seen.add(index)
        keep.append(index)
    return np.asarray(keep, dtype=np.int64)


def rank_random_distractors(pool: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """Random ordering of the distractor pool (nested random sets baseline)."""
    order = rng.permutation(np.asarray(pool, dtype=np.int64))
    return order.astype(np.int64, copy=False)


def build_ranked_nested_sets(
    target_idx: int,
    ranked_distractors: Sequence[int],
    Ks: Sequence[int] = DEFAULT_KS,
    *,
    ref_id: int = 0,
    hardness: str = HARDNESS_RANDOM,
    regime: str = HARDNESS_RANDOM,
    image_id: Optional[int] = None,
    shuffle_positions: bool = True,
    rng: np.random.Generator | None = None,
    availability: Optional[CandidateAvailability] = None,
) -> NestedCandidateSets:
    """Nested sets from a *pre-ranked* distractor list (prefix = hardest/first).

    ``C_K = {target} + ranked_distractors[: K-1]``, therefore nesting holds by
    construction for any ranking strategy (random permutation, same-category IoU
    order, CLIP similarity order).  ``availability`` records every ``K`` that
    cannot be filled and the corresponding entry becomes ``None``.
    """
    Ks = tuple(int(k) for k in Ks)
    if any(k < 2 for k in Ks):
        raise ValueError(f"Ks must all be >= 2 (target + at least one distractor), got {Ks}")
    ranked = np.asarray(ranked_distractors, dtype=np.int64).reshape(-1)
    target_idx = int(target_idx)
    if target_idx < 0:
        raise ValueError(f"target_idx must be non-negative, got {target_idx}")
    if np.any(ranked == target_idx):
        raise ValueError("ranked_distractors must exclude the target proposal")

    rng = _rng(rng)
    report = availability if availability is not None else CandidateAvailability(requested_Ks=Ks)
    if tuple(report.requested_Ks) != Ks:
        report = CandidateAvailability(requested_Ks=Ks)
    report.num_requests += 1

    out = NestedCandidateSets(availability=report)
    pool_size = int(ranked.size)
    max_possible_K = pool_size + 1
    blocked = False
    for k in sorted(Ks):
        if blocked or pool_size < k - 1:
            blocked = True  # nesting: a size that does not fit invalidates larger ones
            report.record(
                CandidateShortage(
                    ref_id=ref_id, requested_K=k, pool_size=pool_size, max_possible_K=max_possible_K
                )
            )
            out[k] = None
            continue
        distractors = ranked[: k - 1]
        indices = np.concatenate(([target_idx], distractors)).astype(np.int64)
        target_position = 0
        if shuffle_positions:
            target_position = int(rng.integers(0, k))
            indices = _move_to(indices, source=0, destination=target_position)
        out[k] = CandidateSet(
            ref_id=int(ref_id),
            candidate_indices=indices,
            target_index=target_position if shuffle_positions else 0,
            K=k,
            hardness=hardness,
            regime=regime,
            target_present=True,
            image_id=image_id,
        )
        report.succeeded(k)
    return out


def _move_to(array: np.ndarray, source: int, destination: int) -> np.ndarray:
    """Move one element of a 1-D array to a new position, keeping other order."""
    values = array.tolist()
    value = values.pop(source)
    values.insert(destination, value)
    return np.asarray(values, dtype=array.dtype)


def build_nested_random_sets(
    target_idx: int,
    pool_indices: Sequence[int],
    Ks: Sequence[int] = DEFAULT_KS,
    rng: np.random.Generator | int | None = None,
    *,
    ref_id: int = 0,
    regime: str = HARDNESS_RANDOM,
    image_id: Optional[int] = None,
    shuffle_positions: bool = True,
    availability: Optional[CandidateAvailability] = None,
) -> NestedCandidateSets:
    """Nested ``C_5 subset C_10 subset C_20 subset C_50`` random candidate sets.

    Parameters
    ----------
    target_idx:
        Row index (in the image's :class:`ProposalBank`) of the *unique* target
        proposal.  It stays identical across all produced ``K``.
    pool_indices:
        Distractor pool: bank row indices available as negatives.  The target is
        removed defensively if it appears in the pool.
    Ks:
        Requested set sizes (default ``(5, 10, 20, 50)``).
    rng:
        ``numpy.random.Generator`` (preferred), ``RandomState`` or an integer
        seed.  Fixed seeds are required by the protocol: candidate sets are
        pre-generated and frozen, never re-sampled at evaluation time.

    Returns
    -------
    NestedCandidateSets
        ``dict`` ``K -> CandidateSet`` (``None`` where the pool was too small),
        with an :class:`CandidateAvailability` report attached.
    """
    rng_ = _rng(rng)
    pool = _dedup_pool(np.asarray(pool_indices, dtype=np.int64), target_idx)
    ranked = rank_random_distractors(pool, rng_)
    return build_ranked_nested_sets(
        target_idx,
        ranked,
        Ks,
        ref_id=ref_id,
        hardness=HARDNESS_RANDOM,
        regime=regime,
        image_id=image_id,
        shuffle_positions=shuffle_positions,
        rng=rng_,
        availability=availability,
    )


# ---------------------------------------------------------------------------
# same-category hard negatives
# ---------------------------------------------------------------------------
def _category_of(object_categories, object_id: int) -> Optional[int]:
    """Read the category of ``object_id`` from a mapping or an indexed array."""
    if object_categories is None:
        return None
    if isinstance(object_categories, Mapping):
        return object_categories.get(int(object_id), object_categories.get(str(object_id)))
    arr = np.asarray(object_categories)
    if arr.ndim == 1:
        if 0 <= int(object_id) < arr.size:
            return int(arr[int(object_id)])
        raise ValueError(f"object id {object_id} out of range for category array of size {arr.size}")
    raise ValueError(f"object_categories must be a mapping or a 1-D array, got shape {arr.shape}")


def select_same_category_hard_negatives(
    bank: ProposalBank,
    target_object_id: int,
    object_categories,
    *,
    target_proposal_idx: Optional[int] = None,
    k: Optional[int] = None,
    min_iou: float = 0.0,
) -> np.ndarray:
    """Rank distractor proposals by their IoU with *same-category* GT objects.

    Protocol section 5: "if a proposal overlaps another COCO GT object of the
    same category as the target, prefer it as a negative".  The ranking consumes
    the bank metadata (``gt_assignment`` = nearest GT object id, ``gt_ious`` =
    IoU with that object), so it needs no image access:

    * a proposal is eligible when its assigned GT object exists
      (``gt_assignment >= 0``), differs from the target object, and has the same
      category as ``target_object_id``;
    * its hardness score is ``gt_ious[i]`` (IoU with that same-category object),
      filtered by ``min_iou``;
    * output is proposal row indices sorted by descending score, ties broken by
      ascending row index (fully deterministic), excluding
      ``target_proposal_idx``.

    ``object_categories`` is either a ``{object_id: category_id}`` mapping or a
    1-D array indexed by object id.  Raises ``ValueError`` when the bank carries
    no GT metadata at all - guessing silently would corrupt the hardness regime.
    """
    if bank.gt_ious is None or bank.gt_assignment is None:
        raise ValueError(
            f"bank for image {bank.image_id} has no gt_ious/gt_assignment metadata; "
            "same-category hard negatives require it (see ccg.data.proposals.write_bank)"
        )
    target_object_id = int(target_object_id)
    target_category = _category_of(object_categories, target_object_id)
    if target_category is None:
        raise ValueError(
            f"category of target object {target_object_id} is unknown; cannot select "
            "same-category negatives"
        )

    scores = np.full(bank.N, -np.inf, dtype=np.float32)
    assigned = bank.gt_assignment >= 0
    same_category = np.asarray(
        [
            (_category_of(object_categories, int(obj_id)) == target_category)
            if flag
            else False
            for obj_id, flag in zip(bank.gt_assignment, assigned)
        ],
        dtype=bool,
    )
    eligible = same_category & (bank.gt_assignment != target_object_id) & (bank.gt_ious >= min_iou)
    scores[eligible] = bank.gt_ious[eligible]
    if target_proposal_idx is not None:
        scores[int(target_proposal_idx)] = -np.inf

    order = np.lexsort((np.arange(bank.N), -scores))
    ranked = order[np.isfinite(scores[order])]
    ranked = ranked.astype(np.int64, copy=False)
    return ranked if k is None else ranked[: int(k)]


def build_same_category_hard_sets(
    bank: ProposalBank,
    target_proposal_idx: int,
    target_object_id: int,
    object_categories,
    Ks: Sequence[int] = DEFAULT_KS,
    rng: np.random.Generator | int | None = None,
    *,
    ref_id: int = 0,
    image_id: Optional[int] = None,
    min_iou: float = 0.0,
    availability: Optional[CandidateAvailability] = None,
) -> NestedCandidateSets:
    """Nested hard sets: same-category ranking, random order for the remainder.

    Hard negatives are always prefixes of the ranked list; once they are
    exhausted the remaining slots fall back to random distractors, so ``C_5
    subset C_10`` still holds.  Fewer same-category negatives than ``K-1`` is a
    property of the image and is recorded in the availability report notes.
    """
    hard = select_same_category_hard_negatives(
        bank,
        target_object_id,
        object_categories,
        target_proposal_idx=target_proposal_idx,
        min_iou=min_iou,
    )
    rng_ = _rng(rng)
    remaining = np.setdiff1d(
        np.arange(bank.N, dtype=np.int64),
        np.concatenate(([int(target_proposal_idx)], hard)),
        assume_unique=False,
    )
    ranked = np.concatenate([hard, rng_.permutation(remaining)]).astype(np.int64)
    report = availability if availability is not None else CandidateAvailability(requested_Ks=Ks)
    if int(hard.size) < max(int(k) for k in Ks) - 1:
        report.note(
            f"ref {ref_id}: only {int(hard.size)} same-category negatives available; "
            "larger K slots fall back to random distractors"
        )
    return build_ranked_nested_sets(
        target_proposal_idx,
        ranked,
        Ks,
        ref_id=ref_id,
        hardness=HARDNESS_SAME_CATEGORY,
        regime=HARDNESS_SAME_CATEGORY,
        image_id=image_id,
        rng=rng_,
        availability=report,
    )


# ---------------------------------------------------------------------------
# CLIP-hard negatives
# ---------------------------------------------------------------------------
def cosine_similarity(query_feature: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against ``[M, D]`` features (pure numpy)."""
    query = np.asarray(query_feature, dtype=np.float32).reshape(-1)
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"features must be 2-D [M,D], got shape {matrix.shape}")
    if matrix.shape[1] != query.size:
        raise ValueError(f"feature dim mismatch: query {query.size} vs features {matrix.shape}")
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    if query_norm <= _EPS:
        raise ValueError("query_feature has (near-)zero norm; cannot compute cosine similarity")
    sims = matrix @ query / (query_norm * np.maximum(row_norms, _EPS))
    sims[row_norms <= _EPS] = 0.0  # a zero crop embedding is "uninformative", not maximally similar
    return sims.astype(np.float32, copy=False)


def select_clip_hard_negatives(
    query_feature: np.ndarray,
    proposal_features: np.ndarray,
    exclude_idx: Sequence[int] = (),
    *,
    k: Optional[int] = None,
    min_similarity: float = -np.inf,
) -> np.ndarray:
    """Rank wrong proposals by CLIP query-crop cosine similarity (descending).

    Diagnostic/adversarial regime (protocol section 5): these sets are *not* a
    realistic detector distribution - they deliberately put the crops CLIP likes
    most next to the query as negatives.  Inputs are cached embeddings only.

    Parameters
    ----------
    query_feature:
        ``[D]`` CLIP text embedding of the expression.
    proposal_features:
        ``[N, D]`` cached crop embeddings, row-aligned with the proposal bank.
    exclude_idx:
        Proposal row ids that must not be selected (the target, and typically
        every proposal with IoU >= threshold with the target GT box - those are
        "equivalent correct answers" and would poison the negative pool).
    """
    matrix = np.asarray(proposal_features, dtype=np.float32)
    sims = cosine_similarity(query_feature, matrix)
    excluded = set(int(i) for i in np.asarray(exclude_idx, dtype=np.int64).reshape(-1).tolist())
    allowed = np.ones(sims.size, dtype=bool)
    for index in excluded:
        if not 0 <= index < sims.size:
            raise ValueError(f"exclude_idx contains out-of-range proposal id {index}")
        allowed[index] = False
    sims = np.where(allowed, sims, -np.inf)
    candidates = np.nonzero((sims > -np.inf) & (sims >= min_similarity))[0]
    order = candidates[np.argsort(-sims[candidates], kind="stable")]
    order = order.astype(np.int64, copy=False)
    return order if k is None else order[: int(k)]


def build_clip_hard_sets(
    ref_id: int,
    target_proposal_idx: int,
    query_feature: np.ndarray,
    proposal_features: np.ndarray,
    Ks: Sequence[int] = DEFAULT_KS,
    rng: np.random.Generator | int | None = None,
    *,
    exclude_idx: Sequence[int] = (),
    image_id: Optional[int] = None,
    availability: Optional[CandidateAvailability] = None,
) -> NestedCandidateSets:
    """Nested CLIP-hard sets (hardest-by-similarity prefix, random tail)."""
    matrix = np.asarray(proposal_features, dtype=np.float32)
    n = matrix.shape[0]
    excluded = np.unique(
        np.concatenate(
            [
                np.asarray([int(target_proposal_idx)], dtype=np.int64),
                np.asarray(exclude_idx, dtype=np.int64).reshape(-1),
            ]
        )
    )
    hard = select_clip_hard_negatives(query_feature, matrix, exclude_idx=excluded, k=None)
    rng_ = _rng(rng)
    remaining = np.setdiff1d(np.arange(n, dtype=np.int64), np.concatenate([excluded, hard]))
    ranked = np.concatenate([hard, rng_.permutation(remaining)]).astype(np.int64)
    report = availability if availability is not None else CandidateAvailability(requested_Ks=Ks)
    if int(hard.size) < max(int(k) for k in Ks) - 1:
        report.note(
            f"ref {ref_id}: only {int(hard.size)} non-target CLIP-ranked negatives for image "
            f"{image_id}; larger K slots fall back to random distractors"
        )
    return build_ranked_nested_sets(
        target_proposal_idx,
        ranked,
        Ks,
        ref_id=ref_id,
        hardness=HARDNESS_CLIP,
        regime=HARDNESS_CLIP,
        image_id=image_id,
        rng=rng_,
        availability=report,
    )


# ---------------------------------------------------------------------------
# target omission (section 6)
# ---------------------------------------------------------------------------
def synthetic_omit(candidate_set: CandidateSet, *, regime: Optional[str] = None) -> CandidateSet:
    """Drop the target proposal from a set -> ``y = NONE`` (synthetic omission).

    ``C^- = C \\ {c*}`` with ``target_index=None`` and ``target_present=False``.
    The hardness label is preserved (only the target disappears, the distractor
    difficulty distribution stays comparable to the parent set) while the regime
    is tagged with ``"_omit"`` so the two can never be conflated in reporting.
    Raises when the set already lacks a target - a double omission would be a
    silent data bug.
    """
    if not candidate_set.target_present:
        raise ValueError(
            f"candidate set {candidate_set.ref_id} already has no target; "
            "synthetic_omit() expects a target-present set"
        )
    target = candidate_set.target_proposal_index
    indices = candidate_set.candidate_indices[candidate_set.candidate_indices != target]
    if indices.size == 0:
        raise ValueError("removing the target would leave an empty candidate set")
    return CandidateSet(
        ref_id=candidate_set.ref_id,
        candidate_indices=indices,
        target_index=None,
        K=int(indices.size),
        hardness=candidate_set.hardness,
        regime=regime or f"{candidate_set.regime}_omit",
        target_present=False,
        image_id=candidate_set.image_id,
    )


# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------
def assert_nested(sets: Mapping[int, CandidateSet], target_proposal_idx: Optional[int] = None) -> None:
    """Assert ``K1 < K2 => C_{K1} subset C_{K2}`` and a constant target.

    Raises ``AssertionError`` with the offending pair - this is the invariant
    that keeps the cardinality effect separable from the proposal effect.
    """
    keys = sorted(int(k) for k in sets)
    previous = None
    for key in keys:
        current = sets[key]
        if current is None:
            continue
        if target_proposal_idx is not None and current.target_proposal_index != int(target_proposal_idx):
            raise AssertionError(
                f"target changed between K values: {current.target_proposal_index} "
                f"!= {target_proposal_idx} at K={key}"
            )
        if previous is not None:
            missing = np.setdiff1d(previous.candidate_indices, current.candidate_indices)
            if missing.size:
                raise AssertionError(f"C_{previous.K} is not a subset of C_{key}: missing {missing.tolist()}")
        previous = current
