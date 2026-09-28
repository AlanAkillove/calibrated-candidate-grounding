"""Phase 0.5 score-only reliability model zoo (protocol Amendment A6, frozen).

The Phase 0.5 reliability audit asks *how much information the candidate score
set already carries* about whether the top-1 grounding prediction is correct.
Amendment A6 (``docs/research_protocol.md``) freezes exactly three model
families, and all of them are **score-only**: their inputs are the scalar scores
of the frozen grounding scorers (plus set bookkeeping such as ``K``), never
candidate embeddings, query/crop embeddings, box geometry, objectness or GT
metadata.

Models
------
* :class:`LogisticModel` -- L2-penalised logistic regression (scikit-learn) on
  pre-standardised ``[n, d]`` features; the small L1 stats baseline.
* :class:`TinyMLP` -- ``d_in -> 32 -> 16 -> 1`` MLP (torch, CPU, float32); the
  small neural L1 baseline.
* :class:`ScoreDeepSets` -- permutation-invariant score-set model
  (``phi: 1 -> 16 -> 16`` GELU, mean/max/top-1 pooling plus ``z_top1`` /
  ``log_k``; ``< 5k`` parameters); the L2 baseline that only ever sees the
  score set.

Determinism
-----------
Both torch models call ``torch.manual_seed(seed)`` at the *start* of every
``fit`` (covering parameter initialisation) and use a dedicated
``torch.Generator(seed)`` for the mini-batch shuffle, so a re-fit with the same
seed and hyper-parameters reproduces bit-identical predictions.

Early-stopping contract
-----------------------
An epoch's validation record is ``{"epoch", "train_nll", "val_nll",
"val_auroc"}``.  When a validation set is supplied the model monitors
``val_nll`` (minimisation, ``patience`` epochs without improvement), restores
the best ``state_dict`` before returning, and reports ``best_epoch`` /
``best_val_nll``.  ``val_nll`` / ``val_auroc`` are ``None`` when no validation
set is given (no early stopping happens in that case).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from ..metrics.discrimination import auroc_correct

__all__ = ["LogisticModel", "TinyMLP", "ScoreDeepSets"]

#: Inference batch size for both torch models (memory-friendly on CPU).
_PREDICT_BATCH = 512
#: Default hidden widths of :class:`TinyMLP`.
_DEFAULT_MLP_HIDDEN: Tuple[int, ...] = (32, 16)
#: Hard parameter ceiling of the L2 ScoreDeepSets model (protocol A6.4).
_SCOREDEEPSETS_PARAM_LIMIT = 5000


class LogisticModel:
    """L2-penalised logistic regression (sklearn) on pre-standardised features.

    The caller is responsible for standardising ``X`` with the reliability_train
    ``mu``/``sigma`` (protocol A6.3); this class never re-scales its input.
    """

    def __init__(self, *, C: float = 1.0) -> None:
        if not np.isfinite(C) or C <= 0.0:
            raise ValueError(f"C must be a positive finite float, got {C!r}")
        self.C = float(C)
        self._clf: Optional[LogisticRegression] = None

    # -- fitting -------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        """Fit on ``X`` ``[n, d]`` float64 (already standardised) and ``y`` {0,1}."""
        X = _as_2d_float64(X, "X")
        y = _as_binary_labels(y, X.shape[0])
        clf = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=20000,
            tol=1e-10,
        )
        clf.fit(X, y)
        self._clf = clf
        return self

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``[n]`` float64 ``P(y=1)``."""
        clf = self._require_fitted()
        X = _as_2d_float64(X, "X")
        if X.shape[1] != clf.coef_.reshape(-1).size:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted on "
                f"{clf.coef_.reshape(-1).size}"
            )
        return np.asarray(clf.predict_proba(X)[:, 1], dtype=np.float64)

    # -- introspection -------------------------------------------------------
    def n_parameters(self) -> int:
        """Number of trainable parameters (``len(coef) + 1`` for the intercept)."""
        clf = self._require_fitted()
        return int(clf.coef_.reshape(-1).size) + 1

    def coefficients(self) -> Tuple[np.ndarray, float]:
        """Return ``(coef, intercept)`` with ``coef`` a ``[d]`` float64 array."""
        clf = self._require_fitted()
        coef = np.asarray(clf.coef_, dtype=np.float64).reshape(-1).copy()
        intercept = float(np.asarray(clf.intercept_, dtype=np.float64).reshape(-1)[0])
        return coef, intercept

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable summary: ``{"C", "coef", "intercept"}``."""
        coef, intercept = self.coefficients()
        return {"C": self.C, "coef": coef.tolist(), "intercept": intercept}

    def _require_fitted(self) -> LogisticRegression:
        if self._clf is None:
            raise RuntimeError("LogisticModel is not fitted; call fit() first")
        return self._clf

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        state = "fitted" if self._clf is not None else "unfitted"
        return f"LogisticModel(C={self.C!r}, {state})"


class TinyMLP:
    """``d_in -> 32 -> 16 -> 1`` sigmoid MLP (torch, CPU, float32), params < 2k."""

    def __init__(self, d_in: int, *, seed: int, hidden: Tuple[int, ...] = _DEFAULT_MLP_HIDDEN) -> None:
        self.d_in = int(d_in)
        if self.d_in < 1:
            raise ValueError(f"d_in must be positive, got {d_in!r}")
        self.seed = int(seed)
        self.hidden = tuple(int(h) for h in hidden)
        if len(self.hidden) == 0:
            raise ValueError("hidden must contain at least one layer width")
        self._net = self._build_net()

    # -- construction --------------------------------------------------------
    def _build_net(self) -> nn.Sequential:
        torch.manual_seed(self.seed)
        layers: List[nn.Module] = []
        prev = self.d_in
        for width in self.hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        net = nn.Sequential(*layers)
        net.eval()
        return net

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters()))

    # -- training ------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 300,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 30,
        log: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train with Adam + BCEWithLogitsLoss; monitor ``val_nll`` for early stop."""
        X = _as_2d_float32(X, "X", self.d_in)
        y = _as_binary_float(y, X.shape[0])
        has_val = _resolve_has_val(X_val, y_val, "X_val", "y_val")
        if has_val:
            X_val = _as_2d_float32(X_val, "X_val", self.d_in)
            y_val = _as_binary_float(y_val, X_val.shape[0])

        self._net = self._build_net()
        net = self._net
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = nn.BCEWithLogitsLoss()
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        if has_val:
            Xv_t = torch.from_numpy(X_val)
            yv_t = torch.from_numpy(y_val)
        n = int(X_t.shape[0])
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        history: List[Dict[str, Any]] = []
        best_state = copy.deepcopy(net.state_dict())
        best_val_nll = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        stopped_early = False
        epochs_run = 0

        for epoch in range(1, int(epochs) + 1):
            epochs_run = epoch
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for start in range(0, n, int(batch_size)):
                idx = perm[start : start + int(batch_size)]
                opt.zero_grad()
                logits = net(X_t[idx]).squeeze(-1)
                loss = loss_fn(logits, y_t[idx])
                loss.backward()
                opt.step()
                total += float(loss.item()) * int(idx.shape[0])
            train_nll = total / n

            val_nll: Optional[float] = None
            val_auroc: Optional[float] = None
            if has_val:
                with torch.no_grad():
                    v_logits = net(Xv_t).squeeze(-1)
                    val_nll = float(loss_fn(v_logits, yv_t).item())
                    v_prob = torch.sigmoid(v_logits).numpy().astype(np.float64)
                val_auroc = float(auroc_correct(v_prob, y_val))
                if val_nll < best_val_nll - 1e-12:
                    best_val_nll = val_nll
                    best_epoch = epoch
                    best_state = copy.deepcopy(net.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            record = {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "val_auroc": val_auroc,
            }
            history.append(record)
            if log is not None:
                log(record)

            if has_val and patience is not None and epochs_no_improve >= int(patience):
                stopped_early = True
                break

        if has_val and best_epoch != -1:
            net.load_state_dict(best_state)
        net.eval()

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_nll": (None if best_epoch == -1 else float(best_val_nll)),
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
        }

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``[n]`` float64 ``P(y=1)`` via batched ``sigmoid``."""
        X = _as_2d_float32(X, "X", self.d_in)
        net = self._net
        net.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, X.shape[0], _PREDICT_BATCH):
                xb = torch.from_numpy(X[start : start + _PREDICT_BATCH])
                chunks.append(torch.sigmoid(net(xb).squeeze(-1)).numpy())
        if not chunks:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TinyMLP(d_in={self.d_in}, hidden={self.hidden}, seed={self.seed})"


class _ScoreDeepSetsNet(nn.Module):
    """``phi`` (per-score MLP) + ``head`` (set-summary MLP); pooling is external."""

    def __init__(self, hidden: int, head_hidden: int, head_in: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )


class ScoreDeepSets:
    """Permutation-invariant score-set reliability model (score-only, params < 5k).

    Forward pass (per candidate set of variable size ``K``)::

        h_i       = phi(scores_i)                 # phi: 1 -> hidden -> hidden (GELU)
        u_mean    = mean_{i in valid} h_i
        u_max     = max_{i in valid} h_i
        h_top1    = h_{argmax_{i in valid} scores_i}
        head_in   = [u_mean, u_max, h_top1, (z_top1), (log_k)]
        p         = sigmoid(head(head_in))

    ``mask`` (``True`` = valid candidate) excludes padding from *every* pooling
    operation, so padding values can never change the output.  Rows that contain
    no valid candidate at all (all-padding) are handled *safely* rather than
    raising: the three pooled blocks are set to zero and the head still produces
    a finite probability.  Real candidate sets always have ``K >= 5`` (a ``K=1``
    singleton row is also accepted and yields a finite output).
    """

    def __init__(
        self,
        *,
        seed: int,
        hidden: int = 16,
        head_hidden: int = 32,
        include_logk: bool = True,
        include_ztop1: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.hidden = int(hidden)
        self.head_hidden = int(head_hidden)
        self.include_logk = bool(include_logk)
        self.include_ztop1 = bool(include_ztop1)
        if self.hidden < 1 or self.head_hidden < 1:
            raise ValueError("hidden and head_hidden must be positive")
        self.head_in = 3 * self.hidden + int(self.include_ztop1) + int(self.include_logk)
        self._net = self._build_net()
        n_params = self.n_parameters()
        if not n_params < _SCOREDEEPSETS_PARAM_LIMIT:
            raise ValueError(
                f"ScoreDeepSets must have < {_SCOREDEEPSETS_PARAM_LIMIT} parameters, "
                f"got {n_params}"
            )

    # -- construction --------------------------------------------------------
    def _build_net(self) -> _ScoreDeepSetsNet:
        torch.manual_seed(self.seed)
        return _ScoreDeepSetsNet(self.hidden, self.head_hidden, self.head_in)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters()))

    # -- forward -------------------------------------------------------------
    def _logits(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        log_k: torch.Tensor,
        z_top1: torch.Tensor,
    ) -> torch.Tensor:
        net = self._net
        hidden = net.phi(scores.unsqueeze(-1))  # [n, K, hidden]
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)  # [n, K, 1]
        raw_count = mask_f.sum(dim=1)  # [n, 1]
        has_valid = raw_count.squeeze(-1) > 0  # [n]
        denom = raw_count.clamp(min=1.0)

        u_mean = (hidden * mask_f).sum(dim=1) / denom

        neg = torch.finfo(hidden.dtype).min
        u_max = hidden.masked_fill(mask_f == 0, neg).max(dim=1).values

        s_masked = scores.squeeze(-1).masked_fill(~mask, neg)
        top_idx = s_masked.argmax(dim=1)
        h_top1 = hidden[torch.arange(hidden.shape[0]), top_idx]

        zero = torch.zeros_like(u_mean)
        u_mean = torch.where(has_valid.unsqueeze(-1), u_mean, zero)
        u_max = torch.where(has_valid.unsqueeze(-1), u_max, zero)
        h_top1 = torch.where(has_valid.unsqueeze(-1), h_top1, zero)

        parts = [u_mean, u_max, h_top1]
        if self.include_ztop1:
            parts.append(z_top1.unsqueeze(-1))
        if self.include_logk:
            parts.append(log_k.unsqueeze(-1))
        feats = torch.cat(parts, dim=1)
        return net.head(feats).squeeze(-1)

    # -- training ------------------------------------------------------------
    def fit(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        log_k: np.ndarray,
        z_top1: np.ndarray,
        y: np.ndarray,
        *,
        lr: float,
        scores_val: Optional[np.ndarray] = None,
        mask_val: Optional[np.ndarray] = None,
        log_k_val: Optional[np.ndarray] = None,
        z_top1_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 300,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 30,
        log: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train with Adam + BCEWithLogitsLoss; monitor ``val_nll`` for early stop."""
        scores, mask, log_k, z_top1, y = _prepare_sets(scores, mask, log_k, z_top1, y, "train")
        val_inputs = (scores_val, mask_val, log_k_val, z_top1_val, y_val)
        has_val = _resolve_has_val_group(val_inputs)
        if has_val:
            scores_val, mask_val, log_k_val, z_top1_val, y_val = _prepare_sets(
                scores_val, mask_val, log_k_val, z_top1_val, y_val, "val"
            )
            if scores_val.shape[1] != scores.shape[1]:
                raise ValueError(
                    f"val K ({scores_val.shape[1]}) must equal train K ({scores.shape[1]})"
                )

        self._net = self._build_net()
        net = self._net
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = nn.BCEWithLogitsLoss()

        s_t = torch.from_numpy(scores)
        m_t = torch.from_numpy(mask)
        k_t = torch.from_numpy(log_k)
        z_t = torch.from_numpy(z_top1)
        y_t = torch.from_numpy(y)
        if has_val:
            sv_t = torch.from_numpy(scores_val)
            mv_t = torch.from_numpy(mask_val)
            kv_t = torch.from_numpy(log_k_val)
            zv_t = torch.from_numpy(z_top1_val)
            yv_t = torch.from_numpy(y_val)

        n = int(s_t.shape[0])
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        history: List[Dict[str, Any]] = []
        best_state = copy.deepcopy(net.state_dict())
        best_val_nll = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        stopped_early = False
        epochs_run = 0

        for epoch in range(1, int(epochs) + 1):
            epochs_run = epoch
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for start in range(0, n, int(batch_size)):
                idx = perm[start : start + int(batch_size)]
                opt.zero_grad()
                logits = self._logits(s_t[idx], m_t[idx], k_t[idx], z_t[idx])
                loss = loss_fn(logits, y_t[idx])
                loss.backward()
                opt.step()
                total += float(loss.item()) * int(idx.shape[0])
            train_nll = total / n

            val_nll: Optional[float] = None
            val_auroc: Optional[float] = None
            if has_val:
                with torch.no_grad():
                    v_logits = self._logits(sv_t, mv_t, kv_t, zv_t)
                    val_nll = float(loss_fn(v_logits, yv_t).item())
                    v_prob = torch.sigmoid(v_logits).numpy().astype(np.float64)
                val_auroc = float(auroc_correct(v_prob, y_val))
                if val_nll < best_val_nll - 1e-12:
                    best_val_nll = val_nll
                    best_epoch = epoch
                    best_state = copy.deepcopy(net.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            record = {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                "val_auroc": val_auroc,
            }
            history.append(record)
            if log is not None:
                log(record)

            if has_val and patience is not None and epochs_no_improve >= int(patience):
                stopped_early = True
                break

        if has_val and best_epoch != -1:
            net.load_state_dict(best_state)
        net.eval()

        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_nll": (None if best_epoch == -1 else float(best_val_nll)),
            "epochs_run": epochs_run,
            "stopped_early": stopped_early,
        }

    # -- inference -----------------------------------------------------------
    def predict_proba(
        self,
        scores: np.ndarray,
        mask: np.ndarray,
        log_k: np.ndarray,
        z_top1: np.ndarray,
    ) -> np.ndarray:
        """Return ``[n]`` float64 ``P(y=1)`` via batched ``sigmoid``."""
        scores, mask, log_k, z_top1, _ = _prepare_sets(
            scores, mask, log_k, z_top1, None, "predict"
        )
        s_t = torch.from_numpy(scores)
        m_t = torch.from_numpy(mask)
        k_t = torch.from_numpy(log_k)
        z_t = torch.from_numpy(z_top1)
        net = self._net
        net.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, s_t.shape[0], _PREDICT_BATCH):
                sl = slice(start, start + _PREDICT_BATCH)
                logits = self._logits(s_t[sl], m_t[sl], k_t[sl], z_t[sl])
                chunks.append(torch.sigmoid(logits).numpy())
        if not chunks:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ScoreDeepSets(hidden={self.hidden}, head_hidden={self.head_hidden}, "
            f"include_logk={self.include_logk}, include_ztop1={self.include_ztop1}, "
            f"seed={self.seed})"
        )


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _as_2d_float64(X: Any, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [n, d], got shape {arr.shape}")
    return arr


def _as_2d_float32(X: Any, name: str, d_in: int) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D [n, d], got shape {arr.shape}")
    if arr.shape[1] != int(d_in):
        raise ValueError(f"{name} has {arr.shape[1]} features but d_in={d_in}")
    return arr


def _as_binary_labels(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"y has {arr.shape[0]} rows but X has {n}")
    return arr.astype(np.int64)


def _as_binary_float(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"y has {arr.shape[0]} rows but X has {n}")
    return arr


def _resolve_has_val(X_val: Any, y_val: Any, x_name: str, y_name: str) -> bool:
    if (X_val is None) != (y_val is None):
        raise ValueError(f"{x_name} and {y_name} must be provided together")
    return X_val is not None


def _resolve_has_val_group(val_inputs: Sequence[Any]) -> bool:
    n_given = sum(v is not None for v in val_inputs)
    if n_given == 0:
        return False
    if n_given != len(val_inputs):
        raise ValueError(
            "validation inputs must be provided together "
            "(scores_val, mask_val, log_k_val, z_top1_val, y_val)"
        )
    return True


def _prepare_sets(
    scores: Any,
    mask: Any,
    log_k: Any,
    z_top1: Any,
    y: Any,
    name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Coerce a score-set batch to ``[n,K] float32`` / ``[n,K] bool`` / ``[n]``."""
    scores_arr = np.asarray(scores, dtype=np.float32)
    if scores_arr.ndim != 2:
        raise ValueError(f"{name} scores must be 2-D [n, K], got shape {scores_arr.shape}")
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.shape != scores_arr.shape:
        raise ValueError(
            f"{name} mask shape {mask_arr.shape} must match scores {scores_arr.shape}"
        )
    n = int(scores_arr.shape[0])
    log_k_arr = _as_vector_float32(log_k, n, f"{name} log_k")
    z_top1_arr = _as_vector_float32(z_top1, n, f"{name} z_top1")
    y_arr: Optional[np.ndarray] = None
    if y is not None:
        y_arr = _as_binary_float(y, n)
    return scores_arr, mask_arr, log_k_arr, z_top1_arr, y_arr


def _as_vector_float32(values: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows but scores has {n}")
    return arr
