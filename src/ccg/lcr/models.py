"""Frozen first-version LCR models and the Aggregate-MLP capacity control.

Three models share one training protocol (protocol
``results/v2_local_competition/protocol.json``, section "training_protocol"):

* :class:`LCR`        ``z = b + g * delta`` (``b = logit(R1)`` frozen input);
* :class:`LCRNoGate`  ``z = b + delta`` (gate ablation);
* :class:`AggregateMLP` the 31-d ``17 score stats + 14 semantic stats``
  capacity control (``31 -> 32 -> 16 -> 1``).

Common pieces:

* shared competitor encoder ``phi: 7 -> 32 -> 16`` (ReLU), ``h_c =
  [max_j phi(r_j), mean_j phi(r_j)]`` (32-d), correction ``delta: 32 -> 16 ->
  1`` (ReLU), gate ``g = sigmoid(w_g . [v_max, v_mean, ds_norm_top2] + b_g)``;
* exact TinyMLP house training loop: Adam + ``BCEWithLogitsLoss``,
  ``weight_decay = 1e-4`` frozen, epochs 300, batch 256, patience 30, monitor
  ``val_nll`` and restore the best state, ``torch.manual_seed(seed)`` for the
  build and a dedicated ``Generator(seed)`` for minibatch permutation;
* parameter budgets: LCR = 1333, LCR-noGate = 1329, Aggregate-MLP = 1569
  (all << 10k target of the protocol).

The models never touch grounding scores: ``b`` is an input constant and
nothing here can change a candidate ranking.  ``b`` is frozen at prediction
time by construction (it is not a learned parameter).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

__all__ = ["LCR", "LCRNoGate", "AggregateMLP", "M_COMPETITORS_USED", "REL_DIM"]

#: Inference batch size (memory-friendly on CPU), mirrors ``reliability.models``.
_PREDICT_BATCH = 512
#: Default hidden widths (frozen; never searched).
_ENC_HIDDEN: Tuple[int, ...] = (32, 16)
#: Competitor count consumed by the encoder (frozen M = 4).
M_COMPETITORS_USED = 4
#: Relation-vector width (frozen 7).
REL_DIM = 7
#: Aggregate-MLP input width (17 score stats + 14 semantic stats, frozen 31).
AGG_INPUT_DIM = 31


def _reject_unknown_keys(inputs: Mapping[str, Any], allowed: Tuple[str, ...]) -> None:
    """Enforce the frozen input contract: forbidden inputs must never pass.

    The protocol forbids raw embedding projections, GT category, objectness,
    target IoU and regime labels as model inputs; any key outside ``allowed``
    raises instead of being silently ignored.
    """
    unknown = sorted(set(inputs) - set(allowed))
    if unknown:
        raise ValueError(
            f"forbidden input key(s) {unknown}: this model accepts only {sorted(allowed)}"
        )


def _as_1d_float32(value: Any, name: str, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_2d_float32(value: Any, name: str, n: int, width: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    if width is not None and arr.shape[1] != width:
        raise ValueError(f"{name} has {arr.shape[1]} columns, expected {width}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_3d_float32(
    value: Any, name: str, n: int, m: Optional[int] = None, width: Optional[int] = None
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3-D, got shape {arr.shape}")
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    if m is not None and arr.shape[1] != m:
        raise ValueError(f"{name} has {arr.shape[1]} competitors, expected {m}")
    if width is not None and arr.shape[2] != width:
        raise ValueError(f"{name} has width {arr.shape[2]}, expected {width}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_binary_float(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"y has {arr.shape[0]} rows, expected {n}")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("y must be binary (0/1)")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_weight_tensor(weight: Any, n: int, name: str) -> Optional[torch.Tensor]:
    """Validate an optional non-negative weight vector and return it as float32."""
    if weight is None:
        return None
    arr = np.asarray(weight, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


class LCRNet(nn.Module):
    """The frozen first-version LCR core (see module docstring)."""

    def __init__(self, *, m: int = M_COMPETITORS_USED, rel_dim: int = REL_DIM) -> None:
        super().__init__()
        self.m = int(m)
        self.rel_dim = int(rel_dim)
        prev = self.rel_dim
        phi_layers: List[nn.Module] = []
        for width in _ENC_HIDDEN:
            phi_layers.append(nn.Linear(prev, width))
            phi_layers.append(nn.ReLU())
            prev = width
        self.phi = nn.Sequential(*phi_layers)
        self.gate = nn.Linear(3, 1)
        self.delta = nn.Sequential(
            nn.Linear(2 * _ENC_HIDDEN[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, b: torch.Tensor, r: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = self.phi(r)                       # [N, M, 16]
        h_max = h.max(dim=1).values           # [N, 16]
        h_mean = h.mean(dim=1)                # [N, 16]
        h_c = torch.cat([h_max, h_mean], dim=-1)  # [N, 32]
        g = torch.sigmoid(self.gate(gate))    # [N, 1]
        d = self.delta(h_c)                   # [N, 1]
        z = b.unsqueeze(-1) + g * d           # [N, 1]
        return z.squeeze(-1)


class LCRNoGateNet(nn.Module):
    """Gate ablation: ``z = b + delta(h_c)`` (identical elsewhere)."""

    def __init__(self, *, m: int = M_COMPETITORS_USED, rel_dim: int = REL_DIM) -> None:
        super().__init__()
        self.m = int(m)
        self.rel_dim = int(rel_dim)
        prev = self.rel_dim
        phi_layers: List[nn.Module] = []
        for width in _ENC_HIDDEN:
            phi_layers.append(nn.Linear(prev, width))
            phi_layers.append(nn.ReLU())
            prev = width
        self.phi = nn.Sequential(*phi_layers)
        self.delta = nn.Sequential(
            nn.Linear(2 * _ENC_HIDDEN[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        h = self.phi(r)
        h_max = h.max(dim=1).values
        h_mean = h.mean(dim=1)
        h_c = torch.cat([h_max, h_mean], dim=-1)
        d = self.delta(h_c)
        z = b.unsqueeze(-1) + d
        return z.squeeze(-1)


class AggregateMLPNet(nn.Module):
    """``31 -> 32 -> 16 -> 1`` capacity control on the frozen 31-d feature set."""

    def __init__(self, d_in: int = AGG_INPUT_DIM) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


class _TorchModelBase:
    """Shared fit/predict loop, mirroring ``ccg.reliability.models.TinyMLP``."""

    VARIANT: str = "base"

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self._net: nn.Module = self._build_net()

    # -- subclass hooks ------------------------------------------------------
    def _build_net(self) -> nn.Module:  # pragma: no cover - overridden
        raise NotImplementedError

    def _pack(self, inputs: Mapping[str, Any]) -> Tuple[np.ndarray, ...]:
        """Validate one input mapping and return the ordered float32 arrays."""
        raise NotImplementedError  # pragma: no cover

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self._net.parameters()))

    # -- training ------------------------------------------------------------
    def fit(
        self,
        inputs: Mapping[str, Any],
        y: Any,
        *,
        lr: float,
        inputs_val: Optional[Mapping[str, Any]] = None,
        y_val: Optional[Any] = None,
        epochs: int = 300,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        patience: int = 30,
        log: Optional[Any] = None,
        sample_weight: Optional[Any] = None,
        val_sample_weight: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train with Adam + BCEWithLogits; monitor ``val_nll`` for early stop.

        M2 regime-balanced curriculum (optional, loss-aggregation layer only):
        ``sample_weight`` holds ``s_i = N / (3 * n_regime)`` and each batch loss
        is ``mean(s_i * l_i)`` -- an unbiased estimator of ``(1/3) sum_r
        mean_r(l)`` at the same loss magnitude as the unweighted path, so the
        frozen optimiser / wd semantics are unchanged. ``val_sample_weight``
        holds ``v_i = 1 / (3 * n_regime)``; ``val_nll`` is then the weighted
        *sum* ``sum_i v_i * l_i = (1/3) sum_r mean_r(l)``. Without weights both
        paths reduce to the frozen M1 behaviour bit-for-bit.
        """
        train_arrays = self._pack(inputs)
        n = int(train_arrays[0].shape[0])
        y_t = torch.from_numpy(_as_binary_float(y, n))
        w_t = _as_weight_tensor(sample_weight, n, "sample_weight")
        has_val = inputs_val is not None and y_val is not None
        if (inputs_val is None) != (y_val is None):
            raise ValueError("inputs_val and y_val must be provided together")
        if val_sample_weight is not None and not has_val:
            raise ValueError("val_sample_weight requires inputs_val and y_val")
        val_tensors: Tuple[torch.Tensor, ...] = ()
        yv_t: Optional[torch.Tensor] = None
        vw_t: Optional[torch.Tensor] = None
        if has_val:
            val_arrays = self._pack(inputs_val)  # type: ignore[arg-type]
            yv_t = torch.from_numpy(_as_binary_float(y_val, val_arrays[0].shape[0]))
            vw_t = _as_weight_tensor(val_sample_weight, val_arrays[0].shape[0], "val_sample_weight")
            val_tensors = tuple(torch.from_numpy(a) for a in val_arrays)

        self._net = self._build_net()
        net = self._net
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        tensors = tuple(torch.from_numpy(a) for a in train_arrays)
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
                logits = net(*[t[idx] for t in tensors])
                element = loss_fn(logits, y_t[idx])
                loss = element.mean() if w_t is None else (element * w_t[idx]).mean()
                loss.backward()
                opt.step()
                total += float(loss.item()) * int(idx.shape[0])
            train_nll = total / n

            val_nll: Optional[float] = None
            if has_val:
                with torch.no_grad():
                    v_logits = net(*val_tensors)
                    element_v = loss_fn(v_logits, yv_t)
                    if vw_t is None:
                        val_nll = float(element_v.mean().item())
                    else:
                        # regime-balanced validation NLL: sum_i v_i * l_i
                        val_nll = float((element_v * vw_t).sum().item())
                if val_nll < best_val_nll - 1e-12:
                    best_val_nll = val_nll
                    best_epoch = epoch
                    best_state = copy.deepcopy(net.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            record = {"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll}
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
    def predict_proba(self, inputs: Mapping[str, Any]) -> np.ndarray:
        """Return ``[n]`` float64 ``P(correct)`` via batched ``sigmoid``."""
        arrays = self._pack(inputs)
        n = int(arrays[0].shape[0])
        tensors = tuple(torch.from_numpy(a) for a in arrays)
        net = self._net
        net.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, _PREDICT_BATCH):
                stop = min(start + _PREDICT_BATCH, n)
                logits = net(*[t[start:stop] for t in tensors])
                chunks.append(torch.sigmoid(logits).numpy())
        if not chunks:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate(chunks).astype(np.float64)

    # -- state serialisation (npz artifacts) ---------------------------------
    def state_arrays(self) -> Dict[str, np.ndarray]:
        """``state_dict`` as a flat ``name -> ndarray`` mapping (dots escaped)."""
        return {
            key.replace(".", "__"): value.detach().cpu().numpy()
            for key, value in self._net.state_dict().items()
        }

    def save_npz(self, path: Any) -> None:
        np.savez(str(path), **self.state_arrays())


class LCR(_TorchModelBase):
    """``z = b + g * delta(h_c)``; ``b`` frozen, ``g`` in (0, 1)."""

    VARIANT = "LCR"

    def __init__(self, *, seed: int, m: int = M_COMPETITORS_USED) -> None:
        self.m = int(m)
        super().__init__(seed=seed)

    def _build_net(self) -> nn.Module:
        torch.manual_seed(self.seed)
        net = LCRNet(m=self.m)
        net.eval()
        return net

    def _pack(self, inputs: Mapping[str, Any]) -> Tuple[np.ndarray, ...]:
        _reject_unknown_keys(inputs, ("b", "r", "gate"))
        b = np.asarray(inputs["b"], dtype=np.float64).reshape(-1)
        n = int(b.shape[0])
        r = _as_3d_float32(inputs["r"], "r", n, self.m, REL_DIM)
        gate = _as_2d_float32(inputs["gate"], "gate", n, 3)
        return (_as_1d_float32(b, "b", n), r, gate)

    def gate_values(self, inputs: Mapping[str, Any]) -> np.ndarray:
        """Diagnostic ``g`` of every row (sigmoid output, always in (0, 1))."""
        arrays = self._pack(inputs)
        tensors = tuple(torch.from_numpy(a) for a in arrays)
        with torch.no_grad():
            g = torch.sigmoid(self._net.gate(tensors[2])).squeeze(-1).numpy()
        return g.astype(np.float64)

    def correction_and_gate(self, inputs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """Diagnostic decomposition of ``z = b + g * delta`` (frozen M2 §13).

        Returns ``{"gate": g, "delta": d}`` as float64 arrays: ``g`` the sigmoid
        gate, ``d`` the raw correction logit of ``z - b``.  Read-only diagnostic;
        no parameter is updated and no score is touched.
        """
        arrays = self._pack(inputs)
        tensors = tuple(torch.from_numpy(a) for a in arrays)
        net = self._net
        net.eval()
        g_chunks: List[np.ndarray] = []
        d_chunks: List[np.ndarray] = []
        n = int(tensors[0].shape[0])
        with torch.no_grad():
            for start in range(0, n, _PREDICT_BATCH):
                stop = min(start + _PREDICT_BATCH, n)
                r = tensors[1][start:stop]
                h = net.phi(r)
                h_c = torch.cat([h.max(dim=1).values, h.mean(dim=1)], dim=-1)
                g_chunks.append(torch.sigmoid(net.gate(tensors[2][start:stop])).squeeze(-1).numpy())
                d_chunks.append(net.delta(h_c).squeeze(-1).numpy())
        if not g_chunks:
            empty = np.empty((0,), dtype=np.float64)
            return {"gate": empty, "delta": empty}
        return {
            "gate": np.concatenate(g_chunks).astype(np.float64),
            "delta": np.concatenate(d_chunks).astype(np.float64),
        }


class LCRNoGate(_TorchModelBase):
    """``z = b + delta(h_c)`` (gate ablation; identical elsewhere)."""

    VARIANT = "LCR-noGate"

    def __init__(self, *, seed: int, m: int = M_COMPETITORS_USED) -> None:
        self.m = int(m)
        super().__init__(seed=seed)

    def _build_net(self) -> nn.Module:
        torch.manual_seed(self.seed)
        net = LCRNoGateNet(m=self.m)
        net.eval()
        return net

    def _pack(self, inputs: Mapping[str, Any]) -> Tuple[np.ndarray, ...]:
        _reject_unknown_keys(inputs, ("b", "r"))
        b = np.asarray(inputs["b"], dtype=np.float64).reshape(-1)
        n = int(b.shape[0])
        r = _as_3d_float32(inputs["r"], "r", n, self.m, REL_DIM)
        return (_as_1d_float32(b, "b", n), r)


class AggregateMLP(_TorchModelBase):
    """``31 -> 32 -> 16 -> 1`` on the frozen 31-d aggregate features."""

    VARIANT = "Aggregate-MLP"

    def __init__(self, *, seed: int, d_in: int = AGG_INPUT_DIM) -> None:
        self.d_in = int(d_in)
        super().__init__(seed=seed)

    def _build_net(self) -> nn.Module:
        torch.manual_seed(self.seed)
        net = AggregateMLPNet(d_in=self.d_in)
        net.eval()
        return net

    def _pack(self, inputs: Mapping[str, Any]) -> Tuple[np.ndarray, ...]:
        _reject_unknown_keys(inputs, ("x",))
        x = np.asarray(inputs["x"], dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be 2-D, got shape {x.shape}")
        return (_as_2d_float32(x, "x", x.shape[0], self.d_in),)
