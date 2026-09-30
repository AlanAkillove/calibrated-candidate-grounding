"""Train the Phase 0B **B3 independent (candidate-blind) MLP** and select its LR.

Two stages, both driven by :class:`ccg.models.b3_data.B3Corpus` (frozen
candidate sets ``C_K = [target] + distractor_order[:K-1]``, K in {5,10} mixed
50:50 during training):

* **Stage 1 (grid)** - train one model per learning rate with ``seed=0`` and
  pick the LR by the **val_select mean candidate-set NLL over K in {5,10}**
  (``T=1``: softmax of the raw logits).  Only ``val_select`` is used for model
  selection / early stopping - ``val_calib`` / ``testA`` / ``testB`` /
  ``K=20`` / ``K=50`` are never consulted.  ``--probe-lr`` runs this stage only.
* **Stage 2 (final)** - retrain the best LR with ``seeds {1,2,3}`` on the same
  frozen manifests (only the parameter init and the minibatch order change) and
  write ``seed_{s}/model.npz`` (the ``IndependentMLPScorer`` state dict, exactly
  the keys ``IndependentMLPScorer.load`` expects) plus ``seed_{s}/training.json``.

Every epoch is deterministic given ``(seed, epoch)``: the training ``K`` of a
sentence is drawn from ``default_rng([seed, epoch, sentence_id])`` and the
in-bucket order from ``default_rng([seed, epoch])``.  ``--resume`` skips a seed
whose ``training.json`` is already ``complete`` (a fully re-run seed is
idempotent because of that determinism).

Usage::

    python -u scripts/train_b3.py --features cache/features \
        --manifests cache/manifests --bank cache/proposals.h5 \
        --refs "data/raw/refcoco+/refcoco+/refs(unc).p" \
        --out results/phase0b_independent --resume
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.models.b3_data import (  # noqa: E402
    EVAL_KS,
    GEO_DIM,
    TARGET_LOCAL_INDEX,
    TRAIN_KS,
    B3Corpus,
    B3Example,
    build_image_sizes,
)
from ccg.models.independent import IndependentMLPScorer, build_candidate_inputs  # noqa: E402

__all__ = ["build_parser", "main", "train_seed", "evaluate_nll", "run_grid"]

HIDDEN_DIM = 128
# V2-G: protocol.json (scorers.layerB) froze this round's trainable decision
# module at "params < 0.5M" with the first-layer width auto-adjusting to
# embed_dim.  The 768-d SigLIP backbone yields 312,577 params, which the stricter
# V1 Phase-0 code default (ccg.models.independent.PARAM_BUDGET = 300_000) would
# reject even though it satisfies this round's frozen cap.  The V2-G run therefore
# uses the round's own cap uniformly for every backbone, while hidden_dim stays
# frozen at 128 so the decision module is architecturally identical across
# B0/B1/B2.  The V1 constant itself is left untouched.
PARAM_BUDGET_V2 = 500_000
DEFAULT_LR_GRID: Tuple[float, ...] = (1e-4, 3e-4, 1e-3)
DEFAULT_SEEDS: Tuple[int, ...] = (1, 2, 3)
GRID_SEED = 0
SELECTION_METRIC = "val_select_mean_nll_K5_K10"
TRAIN_MANIFEST_META = "random_train.meta.json"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def _bank_fingerprint(manifests_root: Path) -> Optional[str]:
    for candidate in (manifests_root / TRAIN_MANIFEST_META, manifests_root / "manifests" / TRAIN_MANIFEST_META):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8")).get("bank_fingerprint")
            except Exception:  # noqa: BLE001
                return None
    return None


def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    import os

    os.replace(str(tmp), str(path))
    return path


def _resolve_device(requested: str, log) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        log(f"[train_b3] requested device={requested} but CUDA is unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _stack_inputs(examples: Sequence[B3Example], geo_dim: int, device: torch.device) -> torch.Tensor:
    """``[B, K, d_in]`` float32 tensor of a same-K batch (rows stay per-candidate)."""
    k = examples[0].K
    rows = [
        build_candidate_inputs(ex.query_feature, ex.candidate_features, ex.geometry, geo_dim)
        for ex in examples
    ]
    for ex, row in zip(examples, rows):
        if ex.K != k:
            raise ValueError(f"batch mixes K={ex.K} with K={k}; bucket by K first")
    arr = np.concatenate(rows, axis=0).reshape(len(examples), k, int(rows[0].shape[1]))
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# evaluation (val_select only for selection)
# ---------------------------------------------------------------------------
def evaluate_nll(
    corpus: B3Corpus,
    model: IndependentMLPScorer,
    split: Any,
    ks: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    """Mean candidate-set NLL (``T=1``) of ``split`` for every requested ``K``.

    ``NLL = -log softmax(s over C_K)[target]`` with the target at local index 0.
    Returns the per-``K`` means, the pooled mean and the row counts.
    """
    ks = tuple(int(k) for k in ks)
    sums = {k: 0.0 for k in ks}
    counts = {k: 0 for k in ks}
    pending: List[B3Example] = []
    cur_k: Optional[int] = None

    def run(batch: List[B3Example], k: int) -> float:
        inputs = _stack_inputs(batch, model.geo_dim, device)
        with torch.no_grad():
            logits = model.net(inputs).reshape(len(batch), k)
            logp = torch.log_softmax(logits, dim=1)[:, TARGET_LOCAL_INDEX]
        return float((-logp).sum().item())

    for example in corpus.iter_examples(split, ks, seed=0):
        k = example.K
        if cur_k is not None and k != cur_k:
            sums[cur_k] += run(pending, cur_k)
            counts[cur_k] += len(pending)
            pending = []
        cur_k = k
        pending.append(example)
        if len(pending) >= int(batch_size):
            sums[cur_k] += run(pending, cur_k)
            counts[cur_k] += len(pending)
            pending = []
    if pending:
        sums[cur_k] += run(pending, cur_k)
        counts[cur_k] += len(pending)

    nll_by_k = {k: (sums[k] / counts[k] if counts[k] else float("nan")) for k in ks}
    total = int(sum(counts.values()))
    pooled = float(sum(sums.values()) / total) if total else float("nan")
    return {"nll": pooled, "nll_by_k": nll_by_k, "counts": counts, "n_rows": total}


# ---------------------------------------------------------------------------
# one training run (seed x lr)
# ---------------------------------------------------------------------------
def train_seed(
    corpus: B3Corpus,
    *,
    seed: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    weight_decay: float,
    device: torch.device,
    eval_ks: Sequence[int] = TRAIN_KS,
    log=print,
    quiet_bars: bool = False,
) -> Tuple[IndependentMLPScorer, Dict[str, Any]]:
    """Train one B3 model; early-stop on val_select NLL and return the best state."""
    seed = int(seed)
    torch.manual_seed(seed)
    model = IndependentMLPScorer(
        feature_dim=corpus.feature_dim, hidden_dim=HIDDEN_DIM, geo_dim=GEO_DIM,
        device=str(device), seed=seed, param_budget=PARAM_BUDGET_V2
    )
    optimizer = torch.optim.Adam(
        model.net.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    eval_ks = tuple(int(k) for k in eval_ks)

    def flush(batch: List[B3Example], k: int) -> float:
        inputs = _stack_inputs(batch, model.geo_dim, device)
        targets = torch.zeros(len(batch), dtype=torch.long, device=device)
        logits = model.net(inputs).reshape(len(batch), k)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach().item())

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_nll = float("inf")
    best_epoch = -1
    bad_epochs = 0
    train_curve: List[float] = []
    val_curve: List[float] = []
    val_by_k: Dict[int, List[float]] = {k: [] for k in eval_ks}
    metrics: Dict[str, Any] = {"nll": float("nan"), "nll_by_k": {k: float("nan") for k in eval_ks}, "counts": {}}

    start = time.time()
    epochs_run = 0
    epoch_bar = tqdm(
        range(1, int(epochs) + 1),
        desc=f"seed{seed} lr{lr:g}",
        unit="epoch",
        disable=None if not quiet_bars else True,
        position=0,
        leave=True,
    )
    for epoch in epoch_bar:
        epochs_run = int(epoch)
        sizes = corpus.train_bucket_sizes(TRAIN_KS, seed=seed, epoch=epoch)
        n_batches = sum((n + int(batch_size) - 1) // int(batch_size) for n in sizes.values())
        step_bar = tqdm(
            total=int(n_batches),
            desc=f"  ep{epoch:02d}",
            unit="batch",
            disable=None if not quiet_bars else True,
            position=1,
            leave=False,
        )
        model.net.train()
        losses: List[float] = []
        pending: List[B3Example] = []
        cur_k: Optional[int] = None
        for example in corpus.iter_examples("train", TRAIN_KS, seed=seed, epoch=epoch):
            k = example.K
            if cur_k is not None and k != cur_k:
                losses.append(flush(pending, cur_k))
                step_bar.update(1)
                pending = []
            cur_k = k
            pending.append(example)
            if len(pending) >= int(batch_size):
                losses.append(flush(pending, cur_k))
                step_bar.update(1)
                pending = []
        if pending:
            losses.append(flush(pending, cur_k))
            step_bar.update(1)
        step_bar.close()

        model.net.eval()
        metrics = evaluate_nll(corpus, model, "val_select", eval_ks, device, batch_size)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_curve.append(train_loss)
        val_curve.append(float(metrics["nll"]))
        for k in eval_ks:
            val_by_k[k].append(float(metrics["nll_by_k"][k]))

        if float(metrics["nll"]) < best_nll - 1e-9:
            best_nll = float(metrics["nll"])
            best_epoch = int(epoch)
            bad_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.net.state_dict().items()}
        else:
            bad_epochs += 1
        epoch_bar.set_postfix_str(
            f"loss={train_loss:.4f} val_nll={metrics['nll']:.4f} best={best_nll:.4f} bad={bad_epochs}"
        )
        log(
            f"[train_b3] seed={seed} lr={lr:g} epoch={epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_select_nll={metrics['nll']:.4f} (K5={metrics['nll_by_k'].get(5, float('nan')):.4f} "
            f"K10={metrics['nll_by_k'].get(10, float('nan')):.4f}) best={best_nll:.4f}@{best_epoch}"
        )
        if bad_epochs >= int(patience):
            log(f"[train_b3] seed={seed} lr={lr:g} early stop at epoch {epoch} (best epoch {best_epoch})")
            break
    epoch_bar.close()

    if best_state is not None:
        model.net.load_state_dict(best_state)
    model.net.eval()
    wall = time.time() - start
    summary: Dict[str, Any] = {
        "seed": seed,
        "lr": float(lr),
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "best_val_select_nll": float(best_nll),
        "train_loss_curve": train_curve,
        "val_select_nll_curve": val_curve,
        "val_select_nll_by_k": {str(k): val_by_k[k] for k in eval_ks},
        "val_select_n_rows_by_k": {str(k): int(metrics["counts"].get(k, 0)) for k in eval_ks},
        "wall_seconds": float(wall),
        "num_parameters": int(model.num_parameters()),
    }
    return model, summary


# ---------------------------------------------------------------------------
# stage 1: LR grid
# ---------------------------------------------------------------------------
def run_grid(
    corpus: B3Corpus, args: argparse.Namespace, device: torch.device, log
) -> Dict[str, Any]:
    """Train seed=0 for every LR; select by val_select mean NLL (K in {5,10})."""
    eval_ks = tuple(int(k) for k in args.eval_ks)
    grid_seed = int(args.grid_seed)
    results: List[Dict[str, Any]] = []
    for lr in args.lr_grid:
        log(f"[train_b3][grid] lr={lr:g} seed={grid_seed} epochs={args.epochs} patience={args.patience}")
        _model, summary = train_seed(
            corpus,
            seed=grid_seed,
            lr=float(lr),
            epochs=int(args.epochs),
            patience=int(args.patience),
            batch_size=int(args.batch_size),
            weight_decay=float(args.weight_decay),
            device=device,
            eval_ks=eval_ks,
            log=log,
        )
        results.append(
            {
                "lr": float(lr),
                "best_val_select_nll": summary["best_val_select_nll"],
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "wall_seconds": summary["wall_seconds"],
                "val_select_nll_curve": summary["val_select_nll_curve"],
                "val_select_nll_by_k": summary["val_select_nll_by_k"],
                "train_loss_curve": summary["train_loss_curve"],
            }
        )
        log(
            f"[train_b3][grid] lr={lr:g} best_val_select_nll={summary['best_val_select_nll']:.4f} "
            f"best_epoch={summary['best_epoch']} wall={summary['wall_seconds']:.1f}s"
        )
    best = min(results, key=lambda item: item["best_val_select_nll"])
    return {
        "selection_metric": SELECTION_METRIC,
        "selection_split": "val_select",
        "selection_ks": list(eval_ks),
        "grid_seed": grid_seed,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "weight_decay": float(args.weight_decay),
        "train_ks": list(TRAIN_KS),
        "feature_dim": int(corpus.feature_dim),
        "hidden_dim": HIDDEN_DIM,
        "geo_dim": GEO_DIM,
        "param_budget": PARAM_BUDGET_V2,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "git_commit": _git_commit(),
        "created_utc": _now_iso(),
        "results": results,
        "best_lr": best["lr"],
        "best_val_select_nll": best["best_val_select_nll"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features", type=Path, default=Path("cache/features"))
    parser.add_argument("--manifests", type=Path, default=Path("cache/manifests"))
    parser.add_argument("--bank", type=Path, default=Path("cache/proposals.h5"))
    parser.add_argument("--refs", type=Path, default=Path("data/raw/refcoco+/refcoco+/refs(unc).p"))
    parser.add_argument("--out", type=Path, default=Path("results/phase0b_independent"))
    parser.add_argument("--lr-grid", type=float, nargs="+", default=list(DEFAULT_LR_GRID))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--eval-ks", type=int, nargs="+", default=list(TRAIN_KS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="skip finished seeds / reuse an existing grid")
    parser.add_argument("--probe-lr", action="store_true", help="run the LR grid only, no final seeds")
    parser.add_argument("--image-sizes", type=Path, default=Path("cache/image_sizes.npz"))
    parser.add_argument("--images-root", type=Path, default=Path("data/raw/mscoco"))
    parser.add_argument("--image-manifest", type=Path, default=Path("data/full_image_manifest.csv"))
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--region-cache", type=int, default=2048)
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="disable the bulk RAM preload of region/text/bank (slow random h5 path)",
    )
    parser.add_argument("--grid-seed", type=int, default=GRID_SEED)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = _resolve_device(args.device, log)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_sizes = Path(args.image_sizes)
    if not image_sizes.exists():
        log(f"[train_b3] image sizes missing -> building {image_sizes}")
        started = time.time()
        build_image_sizes(
            args.images_root, args.image_manifest, image_sizes, workers=int(args.image_workers), log=log
        )
        log(f"[train_b3] image sizes ready in {time.time() - started:.1f}s -> {image_sizes}")
    elif not image_sizes.is_absolute():
        log(f"[train_b3] image sizes: {image_sizes}")

    log(
        f"[train_b3] features={args.features} manifests={args.manifests} bank={args.bank} "
        f"refs={args.refs} device={device} epochs={args.epochs} patience={args.patience} "
        f"batch_size={args.batch_size} weight_decay={args.weight_decay}"
    )

    grid_path = out_dir / "grid_search.json"
    grid_payload: Optional[Dict[str, Any]] = None
    if args.resume and grid_path.exists():
        try:
            grid_payload = json.loads(grid_path.read_text(encoding="utf-8"))
            log(f"[train_b3] reusing grid_search.json (best_lr={grid_payload.get('best_lr')})")
        except Exception:  # noqa: BLE001
            grid_payload = None

    preload = not bool(args.no_preload)
    if preload:
        log("[train_b3] preloading region/text/bank stores into RAM (disable with --no-preload)")
    corpus = B3Corpus(
        args.features,
        args.manifests,
        args.refs,
        args.bank,
        image_sizes_path=image_sizes,
        region_cache_size=int(args.region_cache),
        ks=EVAL_KS,
        preload=preload,
    )
    try:
        if grid_payload is None:
            grid_payload = run_grid(corpus, args, device, log)
            _write_json(grid_path, grid_payload)
            log(f"[train_b3] wrote {grid_path} (best_lr={grid_payload['best_lr']})")
        best_lr = float(grid_payload["best_lr"])

        if args.probe_lr:
            log("[train_b3] --probe-lr: grid only, skipping the final seeds")
            return 0

        bank_fp = _bank_fingerprint(Path(args.manifests))
        git_commit = _git_commit()
        for seed in args.seeds:
            seed = int(seed)
            seed_dir = out_dir / f"seed_{seed}"
            training_path = seed_dir / "training.json"
            model_path = seed_dir / "model.npz"
            if args.resume and training_path.exists() and model_path.exists():
                status = None
                try:
                    status = json.loads(training_path.read_text(encoding="utf-8")).get("status")
                except Exception:  # noqa: BLE001
                    status = None
                if status == "complete":
                    log(f"[train_b3] skip seed {seed}: already complete ({model_path})")
                    continue
            seed_dir.mkdir(parents=True, exist_ok=True)
            log(f"[train_b3] training seed {seed} lr={best_lr:g}")
            model, summary = train_seed(
                corpus,
                seed=seed,
                lr=best_lr,
                epochs=int(args.epochs),
                patience=int(args.patience),
                batch_size=int(args.batch_size),
                weight_decay=float(args.weight_decay),
                device=device,
                eval_ks=tuple(int(k) for k in args.eval_ks),
                log=log,
            )
            model.save(model_path)
            record: Dict[str, Any] = {
                "status": "complete",
                "model": "b3_independent_mlp",
                "seed": seed,
                "lr": best_lr,
                "best_lr_from_grid": best_lr,
                "epochs_run": summary["epochs_run"],
                "best_epoch": summary["best_epoch"],
                "best_val_select_nll": summary["best_val_select_nll"],
                "val_select_nll_curve": summary["val_select_nll_curve"],
                "val_select_nll_by_k": summary["val_select_nll_by_k"],
                "val_select_n_rows_by_k": summary["val_select_n_rows_by_k"],
                "train_loss_curve": summary["train_loss_curve"],
                "wall_seconds": summary["wall_seconds"],
                "num_parameters": summary["num_parameters"],
                "feature_dim": int(model.feature_dim),
                "hidden_dim": HIDDEN_DIM,
                "geo_dim": GEO_DIM,
                "param_budget": PARAM_BUDGET_V2,
                "temperature": 1.0,
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "batch_size": int(args.batch_size),
                "weight_decay": float(args.weight_decay),
                "train_ks": list(TRAIN_KS),
                "eval_ks": list(int(k) for k in args.eval_ks),
                "selection_metric": SELECTION_METRIC,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "git_commit": git_commit,
                "bank_fingerprint": bank_fp,
                "manifest_meta": TRAIN_MANIFEST_META,
                "model_path": f"seed_{seed}/model.npz",
                "created_utc": _now_iso(),
            }
            _write_json(training_path, record)
            log(
                f"[train_b3] seed {seed} done: best_epoch={summary['best_epoch']} "
                f"val_select_nll={summary['best_val_select_nll']:.4f} "
                f"params={summary['num_parameters']} wall={summary['wall_seconds']:.1f}s -> {model_path}"
            )
    finally:
        corpus.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
