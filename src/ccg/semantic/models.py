"""Phase 1 candidate-semantic reliability models (protocol A7.3, frozen).

Amendment A7 (``docs/research_protocol.md``) freezes the Phase 1 model zoo that
asks whether *candidate semantic information* improves reliability prediction
beyond the score-only evidence audited in Phase 0.5.  Both models implemented
here are pure *reliability* estimators: the frozen B3 ranking is never touched,
the grounding scores are never changed or reranked, and no GT category / IoU /
identity, geometry or objectness feature is ever consumed.

Models
------
* :class:`TopCompetitorModel` (E2) -- a tiny interaction model over the query
  embedding, the top-1 / top-2 candidate embeddings and (optionally) four
  scalar score statistics.  ``P_q`` / ``P_v`` project the 512-d OpenCLIP
  embeddings into ``proj_dim`` and the interaction block is
  ``[q, h1, h2, q*h1, q*h2, h1*h2, |h1-h2|]`` (``7 * proj_dim`` dims).
* :class:`SemanticDeepSets` (E3) -- a permutation-invariant semantic DeepSets
  estimator over the whole candidate set (variable ``K`` via padding + mask).
  Each candidate contributes ``[r_i, q⊙r_i, s_i^, p_i, 1/rank_i, top1_ind_i]``
  and the set is pooled with mean / max / top-1.

Both models are self-contained (``numpy`` + ``torch`` only), select the compute
device with ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``
when ``device=None``, and are bit-reproducible for a fixed ``seed``.

Design notes
------------
* Only the raw *scalar* per-candidate quantities (standardised score, softmax
  probability, ``1/rank``, top-1 indicator) are computed in NumPy; the train
  split's ``mu`` / ``sigma`` are estimated once inside ``fit`` and reused at
  inference.  Every learned projection runs in ``torch`` so gradients flow
  through ``P_q`` / ``P_v`` / ``phi`` / ``head`` only.
* The mean pooling accumulates in ``float64`` so that a simultaneous
  permutation of the candidate axis is invariant down to well below ``1e-6``.
* Early stopping mirrors the Phase 0.5 contract: an epoch records the train and
  validation BCE, ``patience`` non-improving epochs trigger a stop, and the best
  ``state_dict`` is restored before returning.  With no validation set the model
  runs all ``epochs`` and the final epoch counts as "best" (``best_val_bce`` is
  ``None``).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn

__all__ = ["TopCompetitorModel", "SemanticDeepSets"]

#: Width of the frozen OpenCLIP embeddings consumed by both models.
_EMB_DIM = 512
#: Number of scalar score statistics consumed by E2.
_E2_SCORE_DIM = 4
#: Number of scalar per-candidate quantities appended to the E3 semantic block.
_E3_SCALAR_DIM = 4
#: Inference batch size (memory-friendly, works on CPU and CUDA).
_PREDICT_BATCH = 512


def _resolve_device(device: Optional[str]) -> torch.device:
    """Return the torch device for a model.

    Parameters
    ----------
    device : str or None
        An explicit device string (e.g. ``"cpu"`` / ``"cuda"``) or ``None`` to
        auto-select ``cuda`` when available else ``cpu``.

    Returns
    -------
    torch.device
        The resolved device.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# --------------------------------------------------------------------------- #
# small numpy validation / coercion helpers
# --------------------------------------------------------------------------- #
def _as_2d_float32(arr: Any, name: str, dim: Optional[int] = None) -> np.ndarray:
    """Coerce ``arr`` to a ``[n, d]`` float32 array (optionally checking ``d``)."""
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{name} must be 2-D [n, d], got shape {out.shape}")
    if dim is not None and out.shape[1] != int(dim):
        raise ValueError(f"{name} must have {int(dim)} columns, got {out.shape[1]}")
    return out


def _as_3d_float32(arr: Any, name: str, dim: Optional[int] = None) -> np.ndarray:
    """Coerce ``arr`` to an ``[n, K, d]`` float32 array (optionally checking ``d``)."""
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 3:
        raise ValueError(f"{name} must be 3-D [n, K, d], got shape {out.shape}")
    if dim is not None and out.shape[2] != int(dim):
        raise ValueError(f"{name} must have {int(dim)} features, got {out.shape[2]}")
    return out


def _as_vector_float32(arr: Any, n: int, name: str) -> np.ndarray:
    """Coerce ``arr`` to a ``[n]`` float32 vector."""
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if out.shape[0] != int(n):
        raise ValueError(f"{name} must have {int(n)} values, got {out.shape[0]}")
    return out


def _as_binary_float(y: Any, n: int) -> np.ndarray:
    """Coerce ``y`` to a ``[n]`` float32 vector of BCE targets."""
    out = np.asarray(y, dtype=np.float32).reshape(-1)
    if out.shape[0] != int(n):
        raise ValueError(f"y must have {int(n)} values, got {out.shape[0]}")
    return out


def _as_bool_mask(arr: Any, name: str, n: int, k: int) -> np.ndarray:
    """Coerce a candidate mask to a ``[n, K]`` bool array (accepts bool or 0/1)."""
    out = np.asarray(arr)
    if out.shape != (int(n), int(k)):
        raise ValueError(f"{name} must have shape {(int(n), int(k))}, got {out.shape}")
    if out.dtype == np.bool_:
        return out
    as_float = out.astype(np.float64)
    if not np.all((as_float == 0.0) | (as_float == 1.0)):
        raise ValueError(f"{name} must be boolean or 0/1")
    return as_float.astype(bool)


def _require_key(
    inputs: Mapping[str, Any], key: str, name: str, *, exc: type = KeyError
) -> Any:
    """Return ``inputs[key]`` or raise ``exc`` with a descriptive message."""
    try:
        return inputs[key]
    except (KeyError, TypeError):
        raise exc(f"{name} inputs missing required key {key!r}") from None


def _resolve_has_val(inputs_val: Any, y_val: Any) -> bool:
    """Return whether a validation set is provided (both parts given together)."""
    if (inputs_val is None) != (y_val is None):
        raise ValueError("inputs_val and y_val must be provided together")
    return inputs_val is not None


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a contiguous float32 numpy array into a tensor on ``device``."""
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device)


def _rank_and_top1(
    score: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(rank_feat, top1_ind)`` for a ``[n, K]`` score batch.

    Parameters
    ----------
    score : numpy.ndarray
        ``[n, K]`` float32 raw B3 scores (padding values are ignored via mask).
    mask : numpy.ndarray
        ``[n, K]`` bool valid-candidate mask.

    Returns
    -------
    rank_feat : numpy.ndarray
        ``[n, K]`` float32 ``1 / rank`` of each valid candidate (descending,
        stable rank among the row's valid candidates); ``0`` where masked.
    top1_ind : numpy.ndarray
        ``[n, K]`` float32 one-hot indicator of the first argmax of the valid
        scores (all-zero rows have no valid candidate); ``0`` where masked.
    """
    n, k = score.shape
    neg = np.where(mask, -score.astype(np.float64), np.inf)
    order = np.argsort(neg, axis=1, kind="stable")
    ranks = np.empty((n, k), dtype=np.float64)
    ranks[np.arange(n)[:, None], order] = np.arange(1, k + 1, dtype=np.float64)[None, :]
    rank_feat = np.where(mask, 1.0 / ranks, 0.0).astype(np.float32)

    top1_ind = np.zeros((n, k), dtype=np.float32)
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size:
        top1_ind[rows, order[rows, 0]] = 1.0
    return rank_feat, top1_ind


def _run_fit(
    net: nn.Module,
    forward_train: Callable[[torch.Tensor], torch.Tensor],
    n: int,
    y_t: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    patience: int,
    gen: torch.Generator,
    forward_val: Optional[Callable[[], torch.Tensor]] = None,
    yv_t: Optional[torch.Tensor] = None,
    log: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Shared Adam + ``BCEWithLogitsLoss`` training loop with early stopping.

    Parameters
    ----------
    net : torch.nn.Module
        Model to train in-place.
    forward_train : callable
        Maps a batch index tensor to logits ``[batch]``.
    n : int
        Number of training rows.
    y_t : torch.Tensor
        ``[n]`` float32 BCE targets.
    lr, weight_decay : float
        Adam hyper-parameters.
    epochs, batch_size : int
        Training schedule.
    patience : int
        Non-improving epochs tolerated when a validation set is supplied.
    gen : torch.Generator
        Deterministic mini-batch shuffle generator.
    forward_val : callable, optional
        Maps nothing to validation logits ``[n_val]``.
    yv_t : torch.Tensor, optional
        ``[n_val]`` float32 validation targets.
    log : callable, optional
        Callback receiving ``{"epoch", "train_bce", "val_bce"}`` per epoch.

    Returns
    -------
    dict
        ``{"epochs_run", "best_epoch", "best_val_bce", "stopped_early"}``.
    """
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.BCEWithLogitsLoss()
    has_val = forward_val is not None
    net.train()

    best_state = copy.deepcopy(net.state_dict())
    best_val_bce = float("inf")
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
            logits = forward_train(idx)
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(idx.shape[0])
        train_bce = total / n if n > 0 else 0.0

        val_bce: Optional[float] = None
        if has_val:
            with torch.no_grad():
                val_bce = float(loss_fn(forward_val(), yv_t).item())
            if val_bce < best_val_bce - 1e-12:
                best_val_bce = val_bce
                best_epoch = epoch
                best_state = copy.deepcopy(net.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if log is not None:
            log({"epoch": epoch, "train_bce": train_bce, "val_bce": val_bce})

        if has_val and patience is not None and epochs_no_improve >= int(patience):
            stopped_early = True
            break

    if has_val and best_epoch != -1:
        net.load_state_dict(best_state)
    net.eval()

    if not has_val:
        best_epoch = epochs_run
    return {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_bce": None if not has_val else float(best_val_bce),
        "stopped_early": stopped_early,
    }


# --------------------------------------------------------------------------- #
# E2 -- TopCompetitorModel
# --------------------------------------------------------------------------- #
class _TopCompetitorNet(nn.Module):
    """Interaction MLP over ``(z_q, z_top1, z_top2, score_stats)``."""

    def __init__(
        self,
        *,
        proj_dim: int,
        hidden: int,
        include_score: bool,
        include_semantic: bool,
    ) -> None:
        super().__init__()
        self.include_score = bool(include_score)
        self.include_semantic = bool(include_semantic)
        if self.include_semantic:
            self.p_q = nn.Linear(_EMB_DIM, int(proj_dim))
            self.p_v = nn.Linear(_EMB_DIM, int(proj_dim))
        in_dim = (7 * int(proj_dim) if self.include_semantic else 0) + (
            _E2_SCORE_DIM if self.include_score else 0
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(
        self,
        z_q: Optional[torch.Tensor],
        z_top1: Optional[torch.Tensor],
        z_top2: Optional[torch.Tensor],
        score_std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``[n]`` logits for a batch."""
        parts = []
        if self.include_semantic:
            q = self.p_q(z_q)
            h1 = self.p_v(z_top1)
            h2 = self.p_v(z_top2)
            parts.extend([q, h1, h2, q * h1, q * h2, h1 * h2, (h1 - h2).abs()])
        if self.include_score:
            parts.append(score_std)
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


class TopCompetitorModel:
    """E2 -- tiny ``(query, top1, top2)`` semantic interaction reliability model.

    The model consumes the query embedding, the top-1 / top-2 candidate
    embeddings (the two embeddings share a single projection ``P_v``) and,
    optionally, four scalar score statistics ``[top1_score, top2_score,
    margin12, msp]``.  It predicts ``P(top-1 grounding is correct)`` and never
    reranks the frozen B3 scores.

    Parameters
    ----------
    seed : int
        Seed for parameter initialisation and the mini-batch shuffle.
    proj_dim : int, optional
        Width of the shared embedding projection (default 64).
    hidden : int, optional
        Width of the single hidden layer of the head (default 128).
    include_score : bool, optional
        Whether the four scalar score statistics are consumed (default True).
    include_semantic : bool, optional
        Whether the semantic interaction block is consumed (default True).
    device : str or None, optional
        Compute device; ``None`` auto-selects CUDA when available.

    Raises
    ------
    ValueError
        If both ``include_score`` and ``include_semantic`` are ``False`` or a
        width is non-positive.
    """

    def __init__(
        self,
        *,
        seed: int,
        proj_dim: int = 64,
        hidden: int = 128,
        include_score: bool = True,
        include_semantic: bool = True,
        device: Optional[str] = None,
    ) -> None:
        if not include_score and not include_semantic:
            raise ValueError(
                "TopCompetitorModel needs at least one of include_score / include_semantic"
            )
        if int(proj_dim) < 1 or int(hidden) < 1:
            raise ValueError("proj_dim and hidden must be positive")
        self.seed = int(seed)
        self.proj_dim = int(proj_dim)
        self.hidden = int(hidden)
        self.include_score = bool(include_score)
        self.include_semantic = bool(include_semantic)
        self.device = _resolve_device(device)
        self._score_mu: Optional[np.ndarray] = None
        self._score_sigma: Optional[np.ndarray] = None
        self._fitted = False
        self._net = self._build_net()

    # -- construction -------------------------------------------------------- #
    def _build_net(self) -> _TopCompetitorNet:
        torch.manual_seed(self.seed)
        net = _TopCompetitorNet(
            proj_dim=self.proj_dim,
            hidden=self.hidden,
            include_score=self.include_score,
            include_semantic=self.include_semantic,
        )
        net.to(self.device)
        net.eval()
        return net

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return int(sum(p.numel() for p in self._net.parameters()))

    # -- input handling ------------------------------------------------------ #
    def _parse(self, inputs: Mapping[str, Any], name: str) -> Dict[str, Any]:
        """Validate and coerce an E2 input mapping into numpy arrays."""
        parsed: Dict[str, Any] = {}
        n: Optional[int] = None
        if self.include_semantic:
            z_q = _as_2d_float32(_require_key(inputs, "z_q", name), f"{name}.z_q", _EMB_DIM)
            z_top1 = _as_2d_float32(
                _require_key(inputs, "z_top1", name), f"{name}.z_top1", _EMB_DIM
            )
            z_top2 = _as_2d_float32(
                _require_key(inputs, "z_top2", name), f"{name}.z_top2", _EMB_DIM
            )
            n = int(z_q.shape[0])
            if z_top1.shape[0] != n or z_top2.shape[0] != n:
                raise ValueError(f"{name}: z_q / z_top1 / z_top2 must share n rows")
            parsed["z_q"] = z_q
            parsed["z_top1"] = z_top1
            parsed["z_top2"] = z_top2
        if self.include_score:
            score_stats = _as_2d_float32(
                _require_key(inputs, "score_stats", name), f"{name}.score_stats", _E2_SCORE_DIM
            )
            if n is None:
                n = int(score_stats.shape[0])
            elif score_stats.shape[0] != n:
                raise ValueError(f"{name}: score_stats must have {n} rows")
            parsed["score_stats"] = score_stats
        parsed["n"] = int(n)  # n is never None here (guarded in __init__)
        return parsed

    def _to_tensors(self, parsed: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert parsed numpy arrays to device tensors (score already standardised)."""
        tensors: Dict[str, torch.Tensor] = {}
        if self.include_semantic:
            tensors["z_q"] = _to_tensor(parsed["z_q"], self.device)
            tensors["z_top1"] = _to_tensor(parsed["z_top1"], self.device)
            tensors["z_top2"] = _to_tensor(parsed["z_top2"], self.device)
        if self.include_score:
            mu, sigma = self._score_mu, self._score_sigma
            if mu is None or sigma is None:
                raise RuntimeError("score standardisation is not fitted; call fit() first")
            std = ((parsed["score_stats"] - mu) / sigma).astype(np.float32)
            tensors["score_std"] = _to_tensor(std, self.device)
        return tensors

    @staticmethod
    def _forward(
        net: _TopCompetitorNet, tensors: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return net(
            tensors.get("z_q"),
            tensors.get("z_top1"),
            tensors.get("z_top2"),
            tensors.get("score_std"),
        )

    # -- training ------------------------------------------------------------ #
    def fit(
        self,
        inputs: Mapping[str, np.ndarray],
        y: np.ndarray,
        *,
        lr: float,
        inputs_val: Optional[Mapping[str, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 300,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 30,
        log: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train with Adam + ``BCEWithLogitsLoss`` and validation early stopping.

        Parameters
        ----------
        inputs : mapping of str to numpy.ndarray
            E2 input arrays for the training split (see the class docstring).
        y : numpy.ndarray
            ``[n]`` binary correctness targets.
        lr : float
            Adam learning rate.
        inputs_val, y_val : optional
            Validation split (both required together for early stopping).
        epochs, batch_size, weight_decay, patience : optional
            Training schedule / optimiser regularisation / early-stop patience.
        log : callable, optional
            Per-epoch callback receiving ``{"epoch", "train_bce", "val_bce"}``.

        Returns
        -------
        dict
            ``{"epochs_run", "best_epoch", "best_val_bce", "stopped_early",
            "n_parameters", "seed", "lr", "include_score", "include_semantic"}``.
        """
        torch.manual_seed(self.seed)
        train = self._parse(inputs, "train")
        n = int(train["n"])
        y_arr = _as_binary_float(y, n)

        if self.include_score:
            score_stats = train["score_stats"]
            mu = score_stats.mean(axis=0).astype(np.float32)
            sigma = score_stats.std(axis=0).astype(np.float32)
            sigma = np.where(sigma == 0.0, 1.0, sigma).astype(np.float32)
            self._score_mu = mu
            self._score_sigma = sigma

        tensors = self._to_tensors(train)
        y_t = _to_tensor(y_arr, self.device)

        has_val = _resolve_has_val(inputs_val, y_val)
        val_tensors: Optional[Dict[str, torch.Tensor]] = None
        yv_t: Optional[torch.Tensor] = None
        if has_val:
            val = self._parse(inputs_val, "val")
            n_val = int(val["n"])
            yv_t = _to_tensor(_as_binary_float(y_val, n_val), self.device)
            val_tensors = self._to_tensors(val)

        self._net = self._build_net()
        net = self._net
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        def forward_train(idx: torch.Tensor) -> torch.Tensor:
            return self._forward(
                net, {k: v[idx] for k, v in tensors.items()}
            )

        forward_val: Optional[Callable[[], torch.Tensor]] = None
        if val_tensors is not None:
            def forward_val() -> torch.Tensor:  # type: ignore[misc]
                return self._forward(net, val_tensors)

        result = _run_fit(
            net,
            forward_train,
            n,
            y_t,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            gen=gen,
            forward_val=forward_val,
            yv_t=yv_t,
            log=log,
        )
        self._fitted = True
        return {
            "epochs_run": result["epochs_run"],
            "best_epoch": result["best_epoch"],
            "best_val_bce": result["best_val_bce"],
            "stopped_early": result["stopped_early"],
            "n_parameters": self.n_parameters(),
            "seed": self.seed,
            "lr": float(lr),
            "include_score": self.include_score,
            "include_semantic": self.include_semantic,
        }

    # -- inference ----------------------------------------------------------- #
    def predict_proba(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return ``[n]`` float64 ``P(top-1 correct)`` in ``[0, 1]``."""
        self._require_fitted()
        parsed = self._parse(inputs, "predict")
        tensors = self._to_tensors(parsed)
        n = int(parsed["n"])
        net = self._net
        net.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, n, _PREDICT_BATCH):
                sl = slice(start, start + _PREDICT_BATCH)
                batch = {k: v[sl] for k, v in tensors.items()}
                logits = self._forward(net, batch)
                chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        if not chunks:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)

    # -- introspection ------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable summary of the model configuration."""
        return {
            "kind": "top_competitor",
            "proj_dim": self.proj_dim,
            "hidden": self.hidden,
            "include_score": self.include_score,
            "include_semantic": self.include_semantic,
            "n_parameters": self.n_parameters(),
            "seed": self.seed,
            "device": str(self.device),
        }

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TopCompetitorModel is not fitted; call fit() first")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TopCompetitorModel(proj_dim={self.proj_dim}, hidden={self.hidden}, "
            f"include_score={self.include_score}, include_semantic={self.include_semantic}, "
            f"seed={self.seed}, device={self.device})"
        )


# --------------------------------------------------------------------------- #
# E3 -- SemanticDeepSets
# --------------------------------------------------------------------------- #
class _SemanticDeepSetsNet(nn.Module):
    """Per-candidate ``phi`` plus set-summary ``head`` (pooling is external)."""

    def __init__(
        self, *, proj_dim: int, phi_hidden: int, head_hidden: int, include_logk: bool
    ) -> None:
        super().__init__()
        self.include_logk = bool(include_logk)
        self.proj_dim = int(proj_dim)
        self.cand_dim = 2 * int(proj_dim) + _E3_SCALAR_DIM
        self.p_v = nn.Linear(_EMB_DIM, int(proj_dim))
        self.p_q = nn.Linear(_EMB_DIM, int(proj_dim))
        self.phi = nn.Sequential(
            nn.Linear(self.cand_dim, int(phi_hidden)),
            nn.GELU(),
            nn.Linear(int(phi_hidden), int(phi_hidden)),
        )
        head_in = 2 * int(phi_hidden) + self.cand_dim + (1 if self.include_logk else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, int(head_hidden)),
            nn.GELU(),
            nn.Linear(int(head_hidden), 1),
        )

    def candidate_h(
        self,
        q: torch.Tensor,
        r: torch.Tensor,
        score_std: torch.Tensor,
        prob: torch.Tensor,
        rank_feat: torch.Tensor,
        top1_ind: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``[n, K, cand_dim]`` candidate features with masked rows zeroed."""
        qr = q.unsqueeze(1) * r
        h = torch.cat(
            [
                r,
                qr,
                score_std.unsqueeze(-1),
                prob.unsqueeze(-1),
                rank_feat.unsqueeze(-1),
                top1_ind.unsqueeze(-1),
            ],
            dim=-1,
        )
        return h * mask.unsqueeze(-1)

    def forward(
        self,
        z_q: torch.Tensor,
        z_i: torch.Tensor,
        mask: torch.Tensor,
        score_std: torch.Tensor,
        prob: torch.Tensor,
        rank_feat: torch.Tensor,
        top1_ind: torch.Tensor,
        log_k_std: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``[n]`` logits for a padded candidate batch."""
        q = self.p_q(z_q)
        r = self.p_v(z_i)
        mask3 = mask.unsqueeze(-1)
        h = self.candidate_h(q, r, score_std, prob, rank_feat, top1_ind, mask)
        phi = self.phi(h) * mask3

        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # float64 accumulation keeps the mean invariant under candidate permutation.
        u_mean = (phi.to(torch.float64).sum(dim=1) / count.to(torch.float64)).to(phi.dtype)

        neg = torch.finfo(phi.dtype).min
        u_max = phi.masked_fill(mask3 == 0, neg).max(dim=1).values

        has_valid = (mask.sum(dim=1, keepdim=True) > 0)
        zero = torch.zeros_like(u_mean)
        u_mean = torch.where(has_valid, u_mean, zero)
        u_max = torch.where(has_valid, u_max, zero)
        h_t = (h * top1_ind.unsqueeze(-1)).sum(dim=1)

        parts = [u_mean, u_max, h_t]
        if self.include_logk:
            parts.append(log_k_std.unsqueeze(-1))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)


class SemanticDeepSets:
    """E3 -- candidate-semantic reliability estimator (ranking stays frozen).

    A permutation-invariant DeepSets model over the whole candidate set with
    variable ``K`` handled through padding + mask.  Each valid candidate
    contributes ``h_i = [r_i, q⊙r_i, s_i^, p_i, 1/rank_i, top1_ind_i]`` and the
    set summary is ``u = [mean_i phi_i, max_i phi_i, h_top1]`` (optionally plus a
    standardised ``log K``).  The model maps candidate semantics and the frozen
    scores to ``P(top-1 correct)`` only; it never reranks.

    Parameters
    ----------
    seed : int
        Seed for parameter initialisation and the mini-batch shuffle.
    proj_dim : int, optional
        Width of the shared embedding projection (default 64).
    phi_hidden : int, optional
        Hidden width of the per-candidate ``phi`` MLP (default 128).
    head_hidden : int, optional
        Hidden width of the set-summary head (default 128).
    include_logk : bool, optional
        Whether the standardised ``log K`` is appended to the summary (default False).
    device : str or None, optional
        Compute device; ``None`` auto-selects CUDA when available.

    Raises
    ------
    ValueError
        If any width is non-positive.
    """

    def __init__(
        self,
        *,
        seed: int,
        proj_dim: int = 64,
        phi_hidden: int = 128,
        head_hidden: int = 128,
        include_logk: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if int(proj_dim) < 1 or int(phi_hidden) < 1 or int(head_hidden) < 1:
            raise ValueError("proj_dim, phi_hidden and head_hidden must be positive")
        self.seed = int(seed)
        self.proj_dim = int(proj_dim)
        self.phi_hidden = int(phi_hidden)
        self.head_hidden = int(head_hidden)
        self.include_logk = bool(include_logk)
        self.device = _resolve_device(device)
        self.cand_dim = 2 * self.proj_dim + _E3_SCALAR_DIM
        self._score_mu: Optional[float] = None
        self._score_sigma: Optional[float] = None
        self._logk_mu: Optional[float] = None
        self._logk_sigma: Optional[float] = None
        self._fitted = False
        self._net = self._build_net()

    # -- construction -------------------------------------------------------- #
    def _build_net(self) -> _SemanticDeepSetsNet:
        torch.manual_seed(self.seed)
        net = _SemanticDeepSetsNet(
            proj_dim=self.proj_dim,
            phi_hidden=self.phi_hidden,
            head_hidden=self.head_hidden,
            include_logk=self.include_logk,
        )
        net.to(self.device)
        net.eval()
        return net

    def n_parameters(self) -> int:
        """Total number of trainable parameters."""
        return int(sum(p.numel() for p in self._net.parameters()))

    # -- input handling ------------------------------------------------------ #
    def _parse(self, inputs: Mapping[str, Any], name: str) -> Dict[str, Any]:
        """Validate and coerce an E3 input mapping into numpy arrays."""
        z_q = _as_2d_float32(_require_key(inputs, "z_q", name), f"{name}.z_q", _EMB_DIM)
        z_i = _as_3d_float32(_require_key(inputs, "z_i", name), f"{name}.z_i", _EMB_DIM)
        n, k = int(z_i.shape[0]), int(z_i.shape[1])
        if z_q.shape[0] != n:
            raise ValueError(f"{name}: z_q must have {n} rows, got {z_q.shape[0]}")
        mask = _as_bool_mask(_require_key(inputs, "mask", name), f"{name}.mask", n, k)
        score = _as_2d_float32(_require_key(inputs, "score", name), f"{name}.score")
        prob = _as_2d_float32(_require_key(inputs, "prob", name), f"{name}.prob")
        if score.shape != (n, k):
            raise ValueError(f"{name}.score must have shape {(n, k)}, got {score.shape}")
        if prob.shape != (n, k):
            raise ValueError(f"{name}.prob must have shape {(n, k)}, got {prob.shape}")
        parsed: Dict[str, Any] = {
            "z_q": z_q,
            "z_i": z_i,
            "mask": mask,
            "score": score,
            "prob": prob,
            "n": n,
            "K": k,
        }
        if self.include_logk:
            log_k = _require_key(inputs, "log_k", name, exc=ValueError)
            parsed["log_k"] = _as_vector_float32(log_k, n, f"{name}.log_k")
        return parsed

    def _fit_standardizers(self, parsed: Mapping[str, Any]) -> None:
        """Estimate score / logK ``mu`` / ``sigma`` from the (train) valid entries."""
        mask = parsed["mask"]
        valid_scores = parsed["score"][mask]
        if valid_scores.size == 0:
            raise ValueError("training batch has no valid candidates to standardise scores")
        self._score_mu = float(valid_scores.mean())
        sigma = float(valid_scores.std())
        self._score_sigma = sigma if sigma > 0.0 else 1.0
        if self.include_logk:
            log_k = parsed["log_k"]
            self._logk_mu = float(log_k.mean())
            sigma_k = float(log_k.std())
            self._logk_sigma = sigma_k if sigma_k > 0.0 else 1.0

    def _score_standardizer(self, parsed: Mapping[str, Any]) -> Tuple[float, float]:
        """Return the ``(mu, sigma)`` used to standardise scores.

        Uses the fitted train statistics when available, otherwise estimates them
        from the valid entries of ``parsed`` (so the feature helper is usable
        before ``fit``).
        """
        if self._score_mu is not None and self._score_sigma is not None:
            return self._score_mu, self._score_sigma
        valid = parsed["score"][parsed["mask"]]
        if valid.size == 0:
            return 0.0, 1.0
        mu = float(valid.mean())
        sigma = float(valid.std())
        return mu, (sigma if sigma > 0.0 else 1.0)

    def _to_tensors(self, parsed: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert parsed numpy arrays to device tensors (derived features built here)."""
        mask = parsed["mask"]
        mu, sigma = self._score_standardizer(parsed)
        score_std = np.where(mask, (parsed["score"] - mu) / sigma, 0.0).astype(np.float32)
        rank_feat, top1_ind = _rank_and_top1(parsed["score"], mask)
        tensors = {
            "z_q": _to_tensor(parsed["z_q"], self.device),
            "z_i": _to_tensor(parsed["z_i"], self.device),
            "mask": _to_tensor(mask.astype(np.float32), self.device),
            "score_std": _to_tensor(score_std, self.device),
            "prob": _to_tensor(
                np.where(mask, parsed["prob"], 0.0).astype(np.float32), self.device
            ),
            "rank_feat": _to_tensor(rank_feat, self.device),
            "top1_ind": _to_tensor(top1_ind, self.device),
        }
        if self.include_logk:
            if self._logk_mu is None or self._logk_sigma is None:
                raise RuntimeError("logK standardisation is not fitted; call fit() first")
            log_k_std = ((parsed["log_k"] - self._logk_mu) / self._logk_sigma).astype(np.float32)
            tensors["log_k_std"] = _to_tensor(log_k_std, self.device)
        return tensors

    @staticmethod
    def _forward(
        net: _SemanticDeepSetsNet, tensors: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return net(
            tensors["z_q"],
            tensors["z_i"],
            tensors["mask"],
            tensors["score_std"],
            tensors["prob"],
            tensors["rank_feat"],
            tensors["top1_ind"],
            tensors.get("log_k_std"),
        )

    # -- feature helper ------------------------------------------------------ #
    def candidate_features(
        self,
        z_q: np.ndarray,
        z_i: np.ndarray,
        mask: np.ndarray,
        score: np.ndarray,
        prob: np.ndarray,
    ) -> np.ndarray:
        """Return the ``[n, K, 2 * proj_dim + 4]`` per-candidate features ``h_i``.

        Columns are ``[r_i, q⊙r_i, s_i^, p_i, rank_feat_i, top1_ind_i]``.  Masked
        candidates are returned as all-zero rows.

        Parameters
        ----------
        z_q : numpy.ndarray
            ``[n, 512]`` query embeddings.
        z_i : numpy.ndarray
            ``[n, K, 512]`` candidate embeddings (padding rows ignored via ``mask``).
        mask : numpy.ndarray
            ``[n, K]`` valid-candidate mask (bool or 0/1).
        score : numpy.ndarray
            ``[n, K]`` frozen B3 scores (padding values ignored via ``mask``).
        prob : numpy.ndarray
            ``[n, K]`` corrected softmax probabilities (padding values ignored).

        Returns
        -------
        numpy.ndarray
            ``[n, K, cand_dim]`` float32 candidate features.
        """
        parsed = self._parse(
            {"z_q": z_q, "z_i": z_i, "mask": mask, "score": score, "prob": prob},
            "candidate_features",
        )
        mask_arr = parsed["mask"]
        mu, sigma = self._score_standardizer(parsed)
        score_std = np.where(mask_arr, (parsed["score"] - mu) / sigma, 0.0).astype(np.float32)
        prob_arr = np.where(mask_arr, parsed["prob"], 0.0).astype(np.float32)
        rank_feat, top1_ind = _rank_and_top1(parsed["score"], mask_arr)

        net = self._net
        with torch.no_grad():
            q = net.p_q(_to_tensor(parsed["z_q"], self.device))
            r = net.p_v(_to_tensor(parsed["z_i"], self.device))
            h = net.candidate_h(
                q,
                r,
                _to_tensor(score_std, self.device),
                _to_tensor(prob_arr, self.device),
                _to_tensor(rank_feat, self.device),
                _to_tensor(top1_ind, self.device),
                _to_tensor(mask_arr.astype(np.float32), self.device),
            )
        return h.detach().cpu().numpy().astype(np.float32)

    # -- training ------------------------------------------------------------ #
    def fit(
        self,
        inputs: Mapping[str, np.ndarray],
        y: np.ndarray,
        *,
        lr: float,
        inputs_val: Optional[Mapping[str, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 300,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 30,
        log: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Train with Adam + ``BCEWithLogitsLoss`` and validation early stopping.

        Parameters
        ----------
        inputs : mapping of str to numpy.ndarray
            E3 input arrays (``z_q``, ``z_i``, ``mask``, ``score``, ``prob`` and
            ``log_k`` when ``include_logk``) for the training split.
        y : numpy.ndarray
            ``[n]`` binary correctness targets.
        lr : float
            Adam learning rate.
        inputs_val, y_val : optional
            Validation split (both required together for early stopping).
        epochs, batch_size, weight_decay, patience : optional
            Training schedule / optimiser regularisation / early-stop patience.
        log : callable, optional
            Per-epoch callback receiving ``{"epoch", "train_bce", "val_bce"}``.

        Returns
        -------
        dict
            ``{"epochs_run", "best_epoch", "best_val_bce", "stopped_early",
            "n_parameters", "seed", "lr", "include_logk"}``.
        """
        torch.manual_seed(self.seed)
        train = self._parse(inputs, "train")
        n = int(train["n"])
        y_arr = _as_binary_float(y, n)
        self._fit_standardizers(train)
        tensors = self._to_tensors(train)
        y_t = _to_tensor(y_arr, self.device)

        has_val = _resolve_has_val(inputs_val, y_val)
        val_tensors: Optional[Dict[str, torch.Tensor]] = None
        yv_t: Optional[torch.Tensor] = None
        if has_val:
            val = self._parse(inputs_val, "val")
            n_val = int(val["n"])
            yv_t = _to_tensor(_as_binary_float(y_val, n_val), self.device)
            val_tensors = self._to_tensors(val)

        self._net = self._build_net()
        net = self._net
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        def forward_train(idx: torch.Tensor) -> torch.Tensor:
            return self._forward(net, {k: v[idx] for k, v in tensors.items()})

        forward_val: Optional[Callable[[], torch.Tensor]] = None
        if val_tensors is not None:
            def forward_val() -> torch.Tensor:  # type: ignore[misc]
                return self._forward(net, val_tensors)

        result = _run_fit(
            net,
            forward_train,
            n,
            y_t,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            gen=gen,
            forward_val=forward_val,
            yv_t=yv_t,
            log=log,
        )
        self._fitted = True
        return {
            "epochs_run": result["epochs_run"],
            "best_epoch": result["best_epoch"],
            "best_val_bce": result["best_val_bce"],
            "stopped_early": result["stopped_early"],
            "n_parameters": self.n_parameters(),
            "seed": self.seed,
            "lr": float(lr),
            "include_logk": self.include_logk,
        }

    # -- inference ----------------------------------------------------------- #
    def predict_proba(self, inputs: Mapping[str, np.ndarray]) -> np.ndarray:
        """Return ``[n]`` float64 ``P(top-1 correct)`` in ``[0, 1]``."""
        self._require_fitted()
        parsed = self._parse(inputs, "predict")
        tensors = self._to_tensors(parsed)
        n = int(parsed["n"])
        net = self._net
        net.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, n, _PREDICT_BATCH):
                sl = slice(start, start + _PREDICT_BATCH)
                batch = {k: v[sl] for k, v in tensors.items()}
                logits = self._forward(net, batch)
                chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        if not chunks:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)

    # -- introspection ------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable summary of the model configuration."""
        return {
            "kind": "semantic_deepsets",
            "proj_dim": self.proj_dim,
            "phi_hidden": self.phi_hidden,
            "head_hidden": self.head_hidden,
            "include_logk": self.include_logk,
            "n_parameters": self.n_parameters(),
            "seed": self.seed,
            "device": str(self.device),
        }

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SemanticDeepSets is not fitted; call fit() first")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SemanticDeepSets(proj_dim={self.proj_dim}, phi_hidden={self.phi_hidden}, "
            f"head_hidden={self.head_hidden}, include_logk={self.include_logk}, "
            f"seed={self.seed}, device={self.device})"
        )
