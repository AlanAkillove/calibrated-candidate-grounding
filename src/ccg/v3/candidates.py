"""V3 FineCops rows and natural/controlled candidate construction.

Natural candidates are the stable objectness prefix of every crop-valid row in
the frozen N=64 bank. Controlled candidates use one deterministic IoU target
and one seeded permutation of non-target, non-correct proposals for every K.
The two constructions deliberately have different eligibility rules.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

IOU_THRESHOLD = 0.5
REQUESTED_KS = (5, 10, 20, 50)
CONTROL_SEED = 20260927


@dataclass(frozen=True)
class FineCopsRow:
    """One official positive expression, retaining its source identity."""

    record_id: str
    expr_id: str
    sentence_id: int
    image_id: int
    gqa_image_id: str
    source_split: str
    expression: str
    gt_boxxyxy: np.ndarray
    actual_image_path: Path
    level: str = ""


@dataclass(frozen=True)
class CandidateCell:
    """One variable-width candidate row; proposal indices are original bank IDs."""

    requested_k: int
    candidate_indices: np.ndarray
    k_eff: int
    target_present: bool
    target_coverage: bool
    presented_valid_target_count: int
    bank_valid_target_count: int
    # Backward-readable alias: this has always meant valid targets anywhere
    # in the frozen N=64 bank, not target proposals in the selected prefix.
    valid_target_count: int
    target_proposal_index: int | None
    missing_reason: str | None


def _as_box(value: Any, *, where: str) -> np.ndarray:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{where}: empty target box")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = [part.strip() for part in raw.split(",")]
    if isinstance(value, Mapping):
        value = [value[key] for key in ("x1", "y1", "x2", "y2")]
    box = np.asarray(value, dtype=np.float32).reshape(-1)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"{where}: target box must contain four finite xyxy coordinates")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{where}: target box is degenerate: {box.tolist()}")
    return box


def _field(row: Mapping[str, Any], names: Sequence[str], *, required: bool = True, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    if required:
        raise ValueError(f"cohort row is missing one of {tuple(names)}")
    return default


def load_cohort(
    path: str | Path,
    *,
    expected_split: str | None = None,
    root: str | Path | None = None,
) -> list[FineCopsRow]:
    """Load canonical V3 CSV/JSONL cohort rows without deduplicating expressions."""
    source = Path(path).resolve()
    workspace = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
    rows: list[FineCopsRow] = []
    seen: dict[tuple[str, str, str], int] = {}
    if source.suffix.lower() in (".jsonl", ".ndjson"):
        with source.open("r", encoding="utf-8-sig") as handle:
            source_rows = []
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_no}: expected one JSON object per line")
                source_rows.append((line_no, value))
    else:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"{source}: missing CSV header")
            source_rows = [(line, raw) for line, raw in enumerate(reader, start=2)]
    for line, raw in source_rows:
        split = str(_field(raw, ("source_split", "split"))).strip()
        if expected_split is not None and split != expected_split:
            raise ValueError(f"{source}:{line}: expected source_split={expected_split!r}, got {split!r}")
        expr_raw = _field(raw, ("expr_id", "sentence_id"))
        expr_id = str(expr_raw).strip()
        try:
            sentence_id = int(_field(raw, ("sentence_id", "expr_id")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}:{line}: expression key must be integer for feature cache") from exc
        gqa_image_id = str(_field(raw, ("gqa_image_id", "image_id"))).strip()
        try:
            image_id = int(gqa_image_id)
        except ValueError as exc:
            raise ValueError(f"{source}:{line}: positive cohort requires numeric GQA image ID") from exc
        expression = str(_field(raw, ("expression", "raw_expression", "text")))
        box_value = _field(raw, ("gt_boxxyxy", "gt_box_xyxy", "target_bbox_xyxy", "gt_box"))
        box = _as_box(box_value, where=f"{source}:{line}")
        image_path = Path(str(_field(raw, ("actual_image_path", "image_path"))))
        if not image_path.is_absolute():
            image_path = (workspace / image_path).resolve()
        namespace = str(_field(raw, ("source_namespace", "dataset"), required=False, default="finecops"))
        key = (namespace, split, expr_id)
        if key in seen:
            raise ValueError(f"{source}:{line}: duplicate source-qualified expression {key}; no rows were dropped")
        seen[key] = line
        rows.append(FineCopsRow(
            record_id=f"{namespace}:{split}:{expr_id}",
            expr_id=expr_id,
            sentence_id=sentence_id,
            image_id=image_id,
            gqa_image_id=gqa_image_id,
            source_split=split,
            expression=expression,
            gt_boxxyxy=box,
            actual_image_path=image_path,
            level=str(_field(raw, ("level",), required=False, default="")),
        ))
    if not rows:
        raise ValueError(f"{source}: no cohort rows")
    return rows


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU from one finite xyxy target to a [N,4] proposal matrix."""
    target = np.asarray(box, dtype=np.float64).reshape(4)
    candidates = np.asarray(boxes, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N,4], got {candidates.shape}")
    left = np.maximum(target[0], candidates[:, 0])
    top = np.maximum(target[1], candidates[:, 1])
    right = np.minimum(target[2], candidates[:, 2])
    bottom = np.minimum(target[3], candidates[:, 3])
    intersection = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
    target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    candidate_area = np.maximum(candidates[:, 2] - candidates[:, 0], 0.0) * np.maximum(
        candidates[:, 3] - candidates[:, 1], 0.0
    )
    union = target_area + candidate_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def stable_objectness_order(objectness: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Original proposal identities in descending objectness, stable on ties."""
    scores = np.asarray(objectness, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if scores.shape != valid.shape or not np.isfinite(scores).all():
        raise ValueError("objectness and proposal-validity vectors must align and be finite")
    return np.flatnonzero(valid)[np.argsort(-scores[valid], kind="stable")].astype(np.int64)


def natural_candidate_cells(
    boxes: np.ndarray,
    objectness: np.ndarray,
    valid_mask: np.ndarray,
    gt_box: np.ndarray,
    *,
    ks: Sequence[int] = REQUESTED_KS,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[int, CandidateCell]:
    """Top-K natural proposal prefixes, retaining every valid correct proposal."""
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if valid.size != box_array.shape[0] or np.asarray(objectness).reshape(-1).size != box_array.shape[0]:
        raise ValueError("proposal boxes, objectness, and validity mask are not aligned")
    if not np.isfinite(box_array).all():
        raise ValueError("proposal boxes must be finite")
    order = stable_objectness_order(objectness, valid)
    ious = box_iou_one_to_many(gt_box, box_array)
    valid_targets = order[ious[order] >= float(iou_threshold)]
    out: dict[int, CandidateCell] = {}
    for raw_k in ks:
        k = int(raw_k)
        if k < 1:
            raise ValueError(f"requested K must be positive, got {k}")
        candidates = order[: min(k, order.size)].copy()
        target_present = bool(np.any(ious[candidates] >= float(iou_threshold)))
        presented_target_count = int(np.sum(ious[candidates] >= float(iou_threshold)))
        if order.size == 0:
            missing = "no_valid_proposals"
        elif not target_present:
            missing = "no_valid_target_in_top_k" if valid_targets.size else "no_valid_target_proposal"
        else:
            missing = None
        out[k] = CandidateCell(
            requested_k=k,
            candidate_indices=candidates,
            k_eff=int(candidates.size),
            target_present=target_present,
            target_coverage=target_present,
            presented_valid_target_count=presented_target_count,
            bank_valid_target_count=int(valid_targets.size),
            valid_target_count=int(valid_targets.size),
            target_proposal_index=(
                int(valid_targets[np.lexsort((valid_targets, -ious[valid_targets]))[0]])
                if valid_targets.size else None
            ),
            missing_reason=missing,
        )
    return out


def _stable_integer_identity(row: FineCopsRow) -> int:
    try:
        return int(row.expr_id)
    except ValueError:
        digest = hashlib.sha256(row.record_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=False)


def controlled_candidate_cells(
    row: FineCopsRow,
    boxes: np.ndarray,
    objectness: np.ndarray,
    valid_mask: np.ndarray,
    gt_box: np.ndarray,
    *,
    ks: Sequence[int] = REQUESTED_KS,
    seed: int = CONTROL_SEED,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[int, CandidateCell]:
    """Historical nested random control: target forced, equivalent targets removed."""
    box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(objectness, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if scores.size != box_array.shape[0] or valid.size != box_array.shape[0]:
        raise ValueError("bank arrays and validity mask must have equal lengths")
    if not np.isfinite(box_array).all() or not np.isfinite(scores).all():
        raise ValueError("proposal boxes and objectness must be finite")
    valid_indices = np.flatnonzero(valid)
    ious = box_iou_one_to_many(gt_box, box_array)
    targets = valid_indices[ious[valid_indices] >= float(iou_threshold)]
    target: int | None = None
    if targets.size:
        # IoU desc; stable bank identity (the original row index) breaks ties.
        target = int(targets[np.lexsort((targets, -ious[targets]))[0]])
        distractors = valid_indices[ious[valid_indices] < float(iou_threshold)]
        identity = _stable_integer_identity(row)
        rng = np.random.default_rng([int(seed), identity])
        ranked = distractors[rng.permutation(distractors.size)].astype(np.int64, copy=False)
    else:
        ranked = np.empty(0, dtype=np.int64)

    out: dict[int, CandidateCell] = {}
    for raw_k in ks:
        k = int(raw_k)
        if k < 1:
            raise ValueError(f"requested K must be positive, got {k}")
        if target is None:
            candidates = np.empty(0, dtype=np.int64)
            missing = "no_valid_target_proposal"
        else:
            take = min(k - 1, ranked.size)
            candidates = np.concatenate(([target], ranked[:take])).astype(np.int64, copy=False)
            missing = None if candidates.size == k else "insufficient_control_distractors"
        out[k] = CandidateCell(
            requested_k=k,
            candidate_indices=candidates,
            k_eff=int(candidates.size),
            target_present=target is not None,
            target_coverage=target is not None,
            presented_valid_target_count=int(target is not None),
            bank_valid_target_count=int(targets.size),
            valid_target_count=int(targets.size),
            target_proposal_index=target,
            missing_reason=missing,
        )
    return out


def prediction_correctness(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    gt_box: np.ndarray,
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[int | None, float | None, bool | None]:
    """Return original-bank winner, winner IoU, and IoU-threshold correctness."""
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if indices.size != values.size:
        raise ValueError("candidate IDs and candidate scores must align")
    if not indices.size:
        return None, None, None
    if not np.isfinite(values).all():
        raise ValueError("candidate scores must be finite")
    proposal_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if np.any(indices < 0) or np.any(indices >= proposal_boxes.shape[0]):
        raise ValueError("candidate proposal identity is outside the source bank")
    local_order = np.argsort(-values, kind="stable")
    winner = int(indices[local_order[0]])
    iou = float(box_iou_one_to_many(gt_box, proposal_boxes[[winner]])[0])
    return winner, iou, bool(iou >= float(iou_threshold))


def controlled_common_k50_mask(cells_by_record: Mapping[str, Mapping[int, CandidateCell]]) -> np.ndarray:
    """Rows eligible for the common K=50 controlled comparisons."""
    ordered = list(cells_by_record.values())
    if not ordered:
        return np.zeros(0, dtype=bool)
    return np.asarray([
        bool(row[50].target_present and row[50].k_eff == 50 and row[50].missing_reason is None)
        for row in ordered
    ], dtype=bool)

