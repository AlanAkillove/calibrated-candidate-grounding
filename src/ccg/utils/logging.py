"""Experiment logging records (protocol section 28).

An experiment is only reproducible if its *raw* outputs are kept: predictions,
logits, candidate ids, confidence, correctness.  Storing the final accuracy
alone would make recomputing calibration / selective metrics impossible, so
:data:`PredictionRecord` exists next to :data:`ExperimentRecord` and both
serialise to JSONL (append-friendly, diff-friendly, no lock files).
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "ExperimentRecord",
    "PredictionRecord",
    "new_experiment_id",
    "utc_now_iso",
    "resolve_git_commit",
    "save_jsonl",
    "load_jsonl",
    "save_records",
    "save_predictions_npz",
    "load_predictions_npz",
    "jsonable",
]

#: Field order of the experiment record, matching section 28 exactly.
EXPERIMENT_RECORD_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "git_commit",
    "timestamp",
    "dataset",
    "split",
    "candidate_protocol",
    "K",
    "hardness",
    "target_presence",
    "backbone",
    "model",
    "seed",
    "training_config",
    "calibration_config",
    "metrics",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_experiment_id(prefix: str = "phase0") -> str:
    """Sortable, collision-free id: ``phase0-<utc>-<8 hex>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def resolve_git_commit(cwd: Optional[str | Path] = None) -> Optional[str]:
    """Short HEAD sha, or ``None`` outside a git checkout (never raises)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover - git missing / not a repo
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and paths into JSON-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, np.bool_)):
        return bool(value) if isinstance(value, np.bool_) else float(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass
class ExperimentRecord:
    """One row of the experiment log - the section 28 schema, field for field."""

    experiment_id: str = field(default_factory=new_experiment_id)
    git_commit: Optional[str] = None
    timestamp: str = field(default_factory=utc_now_iso)
    dataset: str = "refcoco+"
    split: str = "unc"
    candidate_protocol: str = "nested_random"
    K: Any = None
    hardness: str = "random"
    target_presence: str = "present"
    backbone: str = "open_clip ViT-B/32 laion2b_s34b_b79k"
    model: str = "b1_clip_cosine"
    seed: Optional[int] = None
    training_config: Dict[str, Any] = field(default_factory=dict)
    calibration_config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artefacts: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.git_commit is None:
            self.git_commit = resolve_git_commit()
        if self.target_presence not in ("present", "absent", "mixed"):
            raise ValueError(
                f"target_presence must be present/absent/mixed, got {self.target_presence!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        payload = {key: jsonable(getattr(self, key)) for key in EXPERIMENT_RECORD_FIELDS}
        payload["artefacts"] = jsonable(self.artefacts)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRecord":
        known = {f for f in EXPERIMENT_RECORD_FIELDS} | {"artefacts"}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class PredictionRecord:
    """Per-example raw output, so every metric can be recomputed later."""

    experiment_id: str = ""
    ref_id: int = -1
    image_id: Optional[int] = None
    split: str = ""
    K: int = 0
    hardness: str = "random"
    regime: str = "random"
    target_present: bool = True
    candidate_indices: Sequence[int] = ()
    logits: Sequence[float] = ()
    probabilities: Sequence[float] = ()
    predicted_index: Optional[int] = None
    confidence: Optional[float] = None
    top1_top2_margin: Optional[float] = None
    target_index: Optional[int] = None
    correct: Optional[bool] = None
    model: str = ""
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.candidate_indices = np.asarray(self.candidate_indices, dtype=np.int64).reshape(-1)
        self.logits = np.asarray(self.logits, dtype=np.float32).reshape(-1)
        self.probabilities = np.asarray(self.probabilities, dtype=np.float32).reshape(-1)
        self.K = int(self.K)
        if self.K and self.candidate_indices.size not in (0, self.K):
            raise ValueError(
                f"ref {self.ref_id}: {self.candidate_indices.size} candidate ids for K={self.K}"
            )
        if self.logits.size and self.K and self.logits.size != self.K:
            raise ValueError(f"ref {self.ref_id}: {self.logits.size} logits for K={self.K}")
        if self.probabilities.size and self.K and self.probabilities.size != self.K:
            raise ValueError(f"ref {self.ref_id}: {self.probabilities.size} probabilities for K={self.K}")
        if self.target_present and self.target_index is None:
            raise ValueError(f"ref {self.ref_id}: target_present=True requires a target_index")
        if not self.target_present and self.target_index is not None:
            raise ValueError(f"ref {self.ref_id}: target-absent sets must carry target_index=None")
        if self.correct is None and self.target_index is not None and self.predicted_index is not None:
            self.correct = bool(int(self.predicted_index) == int(self.target_index))

    @property
    def abstained(self) -> bool:
        return self.predicted_index is None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in asdict(self).items():
            out[key] = jsonable(value)
        return out


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def save_jsonl(records: Iterable[Any], path: str | Path, *, mode: str = "a") -> Path:
    """Append (or write) records as JSON lines; ``mode='w'`` truncates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            handle.write(json.dumps(jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_jsonl(path: str | Path, record_type: Optional[type] = None) -> List[Any]:
    """Read a JSONL file back into dicts or into ``record_type.from_dict`` objects."""
    path = Path(path)
    records: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(record_type.from_dict(payload) if record_type is not None else payload)
    return records


def save_records(records: Iterable[Any], path: str | Path) -> Path:
    """Convenience wrapper for a one-shot ``mode='w'`` write."""
    return save_jsonl(records, path, mode="w")


def save_predictions_npz(
    path: str | Path,
    *,
    ref_ids: Sequence[int],
    candidate_indices: Sequence[Sequence[int]],
    logits: Sequence[Sequence[float]],
    probabilities: Sequence[Sequence[float]],
    correct: Sequence[bool],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Ragged-safe compact store for a full evaluation pass.

    Candidate sets have different ``K`` across regimes, so the arrays are stored
    as object arrays *inside one npz* (a single file per run, never thousands of
    small files - protocol section 24).  JSONL remains the human-readable
    summary; this is the machine-readable dump that lets calibration and
    selective metrics be recomputed without re-running any model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ref_ids": np.asarray(list(ref_ids), dtype=np.int64),
        "candidate_indices": np.asarray(list(candidate_indices), dtype=object),
        "logits": np.asarray(list(logits), dtype=object),
        "probabilities": np.asarray(list(probabilities), dtype=object),
        "correct": np.asarray(list(correct), dtype=bool),
        "metadata": json.dumps(jsonable(dict(metadata or {}))),
    }
    if int(payload["ref_ids"].size) != len(candidate_indices) or int(payload["ref_ids"].size) != len(logits):
        raise ValueError("ref_ids, candidate_indices and logits must have the same length")
    np.savez_compressed(path, **payload)
    return path


def load_predictions_npz(path: str | Path) -> Dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as store:
        data = {key: store[key] for key in store.files}
    data["metadata"] = json.loads(str(data["metadata"]))
    for key in ("candidate_indices", "logits", "probabilities"):
        data[key] = [np.asarray(item) for item in data[key].tolist()]
    return data
