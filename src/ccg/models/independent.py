"""B3 - independent (candidate-blind) MLP scorer, protocol section 8.

Per candidate the model sees *only* that candidate:

``x_i = [ z_q , z_i , z_q * z_i , cos(z_q, z_i) , g_i ]``  ->  ``s_i``

with ``g_i`` the normalised box ``x, y, w, h`` plus relative ``area`` (and
optionally the detector objectness).  Architecture
``d_in -> hidden -> hidden -> 1`` with ``d_in = 3 * feature_dim + 1 + geo_dim``;
for the default ``feature_dim=512, hidden_dim=128, geo_dim=5`` this is
``1542 -> 128 -> 128 -> 1`` = **214,145 parameters** (``1542*128+128 + 128*128+128 +
128+1``), comfortably inside the ``< 0.3M`` Phase 0 budget (asserted by the test
suite, see :meth:`IndependentMLPScorer.num_parameters`).

Candidate independence
----------------------
During the forward pass a whole candidate set is laid out as a ``[K, d_in]``
batch and pushed through ``Linear -> ReLU -> Linear -> ReLU -> Linear``.  Every
one of those layers maps a *row* to a *row*, i.e. the operation is exactly
``f(x_1), ..., f(x_K)`` computed independently - batching is a pure speed
detail.  Consequently:

* no cross-candidate reduction (mean/max/softmax) exists inside the network;
* no ``BatchNorm`` is used (its statistics would couple the rows of a batch);
* the softmax over the candidate set happens *outside* the scorer, in the
  calibration / metric code.

Equivalent statement: for any permutation ``pi`` of the candidates,
``score(x)[pi] == score(x[pi])`` element-wise, which the test suite checks.
Reading another candidate - or the set size - would silently turn B3 into a
set-aware model and destroy the ablation the whole study rests on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import BaseScorer, as_candidate_matrix, softmax_np

__all__ = [
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_HIDDEN_DIM",
    "PARAM_BUDGET",
    "GEOMETRY_FIELDS",
    "ScoringExample",
    "IndependentMLPScorer",
    "build_candidate_inputs",
    "candidate_input_dim",
    "geometry_features",
]

#: OpenCLIP ViT-B/32 joint embedding size (section 3).
DEFAULT_FEATURE_DIM = 512
DEFAULT_HIDDEN_DIM = 128
#: Hard Phase 0 budget for a trainable decision module.
PARAM_BUDGET = 300_000
#: Default geometry layout, mirrored in :func:`geometry_features`.
GEOMETRY_FIELDS = ("x", "y", "w", "h", "area")

#: Fixed normalisation side used when the true image size is unknown.  It must
#: never be derived from the candidates of the current set, or the independence
#: property above would be broken by a data-dependent rescaling.
DEFAULT_NORMALISATION_SIDE = 640.0


def _require_torch():
    """Local import so the numpy-only parts of the package stay torch-free."""
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "IndependentMLPScorer needs torch (declared in pyproject.toml). The torch-free "
            "baselines (ccg.models.cosine, ccg.models.stats_calibrator.extract_score_stats) "
            "work without it."
        ) from exc


def candidate_input_dim(feature_dim: int = DEFAULT_FEATURE_DIM, geo_dim: int = len(GEOMETRY_FIELDS)) -> int:
    """``d_in = 3 * feature_dim + 1 + geo_dim`` (concatenation layout of B3)."""
    return 3 * int(feature_dim) + 1 + int(geo_dim)


def geometry_features(
    boxes: np.ndarray,
    image_size: Optional[Sequence[float]] = None,
    objectness: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Turn ``xyxy`` boxes into the ``g_i`` geometry block of B3.

    Layout: ``[x/W, y/H, w/W, h/H, area/(W*H)]`` and, when ``objectness`` is
    given, one extra column with the raw detector score -> ``geo_dim = 5`` or
    ``6``.  Coordinates are top-left corners of the *unpadded* box in image
    pixels; the normaliser is either the true ``image_size=(W, H)`` or the fixed
    :data:`DEFAULT_NORMALISATION_SIDE` (see the module docstring for why it is
    never computed from the candidate set itself).
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"boxes must be [K,4] xyxy, got {arr.shape}")
    if image_size is None:
        width = height = float(DEFAULT_NORMALISATION_SIDE)
    else:
        width, height = (float(image_size[0]), float(image_size[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size must be positive (W,H), got {tuple(image_size)}")
    features = np.empty((arr.shape[0], 5), dtype=np.float32)
    features[:, 0] = arr[:, 0] / width
    features[:, 1] = arr[:, 1] / height
    features[:, 2] = (arr[:, 2] - arr[:, 0]) / width
    features[:, 3] = (arr[:, 3] - arr[:, 1]) / height
    features[:, 4] = features[:, 2] * features[:, 3]
    if objectness is not None:
        scores = np.asarray(objectness, dtype=np.float32).reshape(-1)
        if scores.size != arr.shape[0]:
            raise ValueError(f"objectness has {scores.size} entries, expected {arr.shape[0]}")
        features = np.concatenate([features, scores.reshape(-1, 1)], axis=1)
    return features


def build_candidate_inputs(
    query_feature: np.ndarray,
    candidate_features: np.ndarray,
    geometry: Optional[np.ndarray] = None,
    geo_dim: int = len(GEOMETRY_FIELDS),
) -> np.ndarray:
    """Assemble the ``[K, d_in]`` MLP input matrix (``[z_q, z_i, z_q*z_i, cos, g]``).

    The query row is broadcast, never mixed: row ``i`` of the output depends on
    candidate ``i`` only.  Missing geometry is zero-filled to keep the width
    stable (a candidate without a box is not a candidate without a feature).
    """
    query = np.asarray(query_feature, dtype=np.float32).reshape(-1)
    candidates = as_candidate_matrix(candidate_features, query)
    k, d = candidates.shape
    product = candidates * query[None, :]
    norms = np.maximum(np.linalg.norm(candidates, axis=1), 1e-8)
    cos = (candidates @ query) / (float(np.linalg.norm(query)) * norms)
    cos = cos.astype(np.float32).reshape(-1, 1)
    if geometry is None:
        geometry = np.zeros((k, geo_dim), dtype=np.float32)
    else:
        geometry = np.asarray(geometry, dtype=np.float32)
        if geometry.ndim == 1:
            geometry = geometry.reshape(1, -1)
        if geometry.shape != (k, geo_dim):
            raise ValueError(
                f"geometry must have shape [K={k},{geo_dim}], got {geometry.shape}"
            )
    query_block = np.tile(query[None, :], (k, 1))
    return np.concatenate([query_block, candidates, product, cos, geometry], axis=1).astype(
        np.float32, copy=False
    )


@dataclass
class ScoringExample:
    """One training/evaluation instance of the independent scorer."""

    ref_id: int
    query_feature: np.ndarray
    candidate_features: np.ndarray
    geometry: Optional[np.ndarray]
    target_index: Optional[int]

    @property
    def K(self) -> int:
        return int(np.asarray(self.candidate_features).shape[0])

    @property
    def target_present(self) -> bool:
        return self.target_index is not None


class IndependentMLPScorer(BaseScorer):
    """B3 baseline: candidate-blind MLP over query/candidate interaction terms."""

    name = "b3_independent_mlp"

    def __init__(
        self,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        geo_dim: int = len(GEOMETRY_FIELDS),
        temperature: float = 1.0,
        device: str = "cpu",
        seed: int = 0,
        param_budget: int = PARAM_BUDGET,
    ) -> None:
        torch = _require_torch()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.geo_dim = int(geo_dim)
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.input_dim = candidate_input_dim(self.feature_dim, self.geo_dim)

        torch.manual_seed(self.seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        ).to(self.device)
        self._param_budget = int(param_budget)
        self.check_param_budget()

    # -- interface -----------------------------------------------------------
    def score(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """``[K]`` logits (float32 numpy); ``candidate_features`` is ``[K, D]``."""
        return self.score_torch(query_feature, candidate_features, geometry).detach().cpu().numpy()

    def score_torch(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
    ):
        """Same as :meth:`score` but keeps the gradient-carrying torch tensor."""
        torch = _require_torch()
        inputs = build_candidate_inputs(query_feature, candidate_features, geometry, self.geo_dim)
        with torch.no_grad():
            logits = self.net(torch.as_tensor(inputs, device=self.device)).reshape(-1)
        return logits / self.temperature

    def probabilities(
        self,
        query_feature: np.ndarray,
        candidate_features: np.ndarray,
        geometry: Optional[np.ndarray] = None,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """Softmax over the candidate set (applied *after* the independent pass)."""
        logits = self.score(query_feature, candidate_features, geometry)
        return softmax_np(logits, temperature=1.0 if temperature is None else temperature)

    # -- capacity ------------------------------------------------------------
    def num_parameters(self) -> int:
        """Total trainable scalar parameters (weights + biases)."""
        return int(sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def param_budget_usage(self) -> Dict[str, float]:
        count = self.num_parameters()
        return {
            "num_parameters": float(count),
            "budget": float(self._param_budget),
            "usage": float(count) / float(self._param_budget),
            "input_dim": float(self.input_dim),
            "analytic": float(self._analytic_parameter_count()),
        }

    def _analytic_parameter_count(self) -> int:
        """``d_in*h + h + h*h + h + h + 1`` - a closed form cross-check."""
        h = self.hidden_dim
        return (self.input_dim * h + h) + (h * h + h) + (h + 1)

    def check_param_budget(self) -> int:
        count = self.num_parameters()
        if count != self._analytic_parameter_count():
            raise AssertionError(
                f"parameter bookkeeping mismatch: torch={count} analytic={self._analytic_parameter_count()}"
            )
        if count >= self._param_budget:
            raise ValueError(
                f"independent MLP has {count} parameters, exceeding the {self._param_budget} "
                "Phase 0 budget; shrink hidden_dim or geo_dim instead of growing the module"
            )
        return count

    # -- training skeleton ---------------------------------------------------
    def train(
        self,
        examples: Sequence[ScoringExample],
        epochs: int = 10,
        lr: float = 1e-3,
        seed: Optional[int] = None,
        batch_size: int = 32,
        weight_decay: float = 0.0,
        log_every: int = 0,
    ) -> Dict[str, object]:
        """Cross-entropy training over candidate sets (Phase 0 skeleton).

        Each example contributes one ``K``-way softmax term: the loss is
        ``-log p_target`` with the softmax taken over *that example's*
        candidates.  Examples whose target is absent are **not** silently
        dropped - they are counted and returned in ``num_skipped_absent``
        (target-absent learning belongs to Phase 2, not to B3).

        Sets are bucketed by ``K`` so a batch shares one tensor shape; that is
        the only reason ``K`` appears in the plumbing.  Not covered by the unit
        tests (which assert forward shape, independence and capacity), so
        consider it validated only once the real feature cache exists.
        """
        torch = _require_torch()
        if seed is not None:
            torch.manual_seed(int(seed))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        buckets: Dict[int, List[ScoringExample]] = {}
        skipped = 0
        for example in examples:
            if not example.target_present:
                skipped += 1
                continue
            buckets.setdefault(example.K, []).append(example)

        history: List[float] = []
        self.net.train()
        for _ in range(int(epochs)):
            epoch_losses: List[float] = []
            for k in sorted(buckets):
                items = buckets[k]
                order = np.random.permutation(len(items)) if len(items) > 1 else np.arange(len(items))
                for start in range(0, len(order), int(batch_size)):
                    batch = [items[int(i)] for i in order[start : start + int(batch_size)]]
                    inputs, targets = self._stack_batch(batch)
                    logits = self.net(inputs).reshape(len(batch), -1)
                    if logits.ndim != 2 or logits.shape[0] != len(batch):
                        raise AssertionError(
                            f"expected [B,K] logits, got {tuple(logits.shape)} - "
                            "the independent MLP must not mix candidates"
                        )
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu().item()))
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            history.append(mean_loss)
            if log_every and len(history) % int(log_every) == 0:
                print(f"epoch {len(history)}/{epochs} loss={mean_loss:.4f}")
        self.net.eval()
        return {
            "loss_per_epoch": history,
            "num_skipped_absent": skipped,
            "num_examples": len(examples),
            "num_parameters": self.num_parameters(),
        }

    def _stack_batch(self, batch: Sequence[ScoringExample]):
        """Lay a same-``K`` bucket out as ``[B, K, d_in]`` (rows stay candidate-independent)."""
        torch = _require_torch()
        rows = [
            build_candidate_inputs(
                example.query_feature, example.candidate_features, example.geometry, self.geo_dim
            )
            for example in batch
        ]
        k = int(batch[0].K)
        d_in = rows[0].shape[1]
        for example, row in zip(batch, rows):
            if example.K != k:
                raise ValueError(f"batch mixes K={example.K} with K={k}; bucket by K first")
            if row.shape[1] != d_in:
                raise ValueError("inconsistent input width inside one batch")
        stacked = np.concatenate(rows, axis=0)
        tensor = torch.as_tensor(
            stacked.reshape(len(batch), k, d_in), dtype=torch.float32, device=self.device
        )
        targets = torch.as_tensor(
            [int(example.target_index) for example in batch], dtype=torch.long, device=self.device
        )
        return tensor, targets

    # -- persistence ---------------------------------------------------------
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy() for key, value in self.net.state_dict().items()}

    def save(self, path) -> None:
        np.savez(str(path), **self.state_dict())

    def load(self, path) -> None:
        torch = _require_torch()
        with np.load(str(path), allow_pickle=False) as store:
            state = {key: torch.as_tensor(store[key]) for key in store.files}
        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()

    @property
    def config(self) -> Dict[str, object]:
        """Everything needed to rebuild this module (experiment logging, section 28)."""
        return {
            "model": self.name,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "geo_dim": self.geo_dim,
            "input_dim": self.input_dim,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_parameters": self.num_parameters(),
        }
