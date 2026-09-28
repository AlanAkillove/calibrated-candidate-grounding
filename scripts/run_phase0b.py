"""Run the Phase 0B **B3 independent (candidate-blind) MLP evaluation**.

This is a thin CLI over :func:`ccg.experiment.phase0b.run_b3_eval`.  It scores
every B3 seed on the **exact** Phase 0A evaluation cohort (the ``common_cohort_rows``
intersection - the same pooled 20,799 sentences of the cosine audit, never
re-sampled), fits the corrected global temperature on ``val_calib`` (K in {5, 10}
only), and writes the full artifact set (per-seed raw scores, ranking / calibration
/ selective / reliability / diagnostics / bootstrap tables, the cross-seed
``aggregate.csv`` and the ``paired_vs_cosine.csv`` comparison) under ``--out``.

Raw-score invariance (atol = 0) is guaranteed by construction: every sentence is
run *once* through the MLP on its maximal superset and every reported ``K`` is a
pure *gather* of that one score table.  A failure of any of the three hard sanity
checks (score invariance / rank monotonicity / accuracy monotonicity) writes
``seed_{s}/VALIDATION_FAILURE.json`` and exits 2 before any result artifact exists.

Usage
-----
    python -u scripts/run_phase0b.py \
        --features cache/features \
        --manifests cache/manifests \
        --bank cache/proposals.h5 \
        --refs "data/raw/refcoco+/refcoco+/refs(unc).p" \
        --seeds 1 2 3 \
        --out results/phase0b_independent \
        --bootstrap-replicates 5000 \
        --device cuda \
        --resume

Progress reporting: three live tqdm bars (stderr, ``disable=None``) - a
``scoring`` bar over the batched superset forwards, a ``metrics`` bar over the
per-split/K metric cells and a ``bootstrap`` bar over the paired comparisons.
A timestamped log is written to ``logs_phase0b_eval.txt`` (``--log-file``).

Exit codes: 0 on success; 2 when a hard sanity check STOPs a seed (the failure
JSON was written first); any other exception propagates with its traceback.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, TextIO

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.experiment.phase0b import (  # noqa: E402  (pure numpy / lazy torch)
    COSINE_PREDICTIONS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_KS,
    DEFAULT_OUT,
    DEFAULT_SEEDS,
    PRIMARY_SPLITS,
    REGIME_RANDOM,
    run_b3_eval,
)

__all__ = ["build_parser", "main"]

DEFAULT_FEATURES = Path("cache/features")
DEFAULT_MANIFESTS = Path("cache/manifests")
DEFAULT_BANK = Path("cache/proposals.h5")
DEFAULT_REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
DEFAULT_IMAGE_SIZES = Path("cache/image_sizes.npz")
DEFAULT_IMAGES_ROOT = Path("data/raw/mscoco")
DEFAULT_IMAGE_MANIFEST = Path("data/full_image_manifest.csv")
DEFAULT_LOG_FILE = Path("logs_phase0b_eval.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help=f"feature cache root (default: {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        default=DEFAULT_MANIFESTS,
        help=(
            "candidate-manifest root: either the manifests/ directory itself or "
            f"its parent (default: {DEFAULT_MANIFESTS})"
        ),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=DEFAULT_BANK,
        help=f"frozen proposal bank (bank-v1 hdf5) (default: {DEFAULT_BANK})",
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=DEFAULT_REFS,
        help=f"path to refs(unc).p (default: {DEFAULT_REFS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"artifact directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"B3 checkpoint seeds under <out>/seed_<s>/ (default: {list(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help=f"candidate-set sizes (default: {list(DEFAULT_KS)})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(PRIMARY_SPLITS),
        choices=list(PRIMARY_SPLITS),
        help=f"eval splits to report (default: {list(PRIMARY_SPLITS)})",
    )
    parser.add_argument(
        "--regime",
        default=REGIME_RANDOM,
        choices=[REGIME_RANDOM],
        help="construction regime; only 'random' is audited",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
        help="paired image-clustered bootstrap replicates (protocol minimum: 5000)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="bootstrap RNG seed")
    parser.add_argument("--bootstrap-ci", type=float, default=0.95, help="bootstrap CI level")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"superset-forward batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=1e-3,
        help="lower temperature bound of the log-space fit (default: 1e-3)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=10.0,
        help="upper temperature bound of the log-space fit (default: 10.0)",
    )
    parser.add_argument("--device", default="cuda", help="torch device for the MLP ('cuda'/'cpu')")
    parser.add_argument("--resume", action="store_true", help="skip seeds already evaluated")
    parser.add_argument(
        "--image-sizes",
        type=Path,
        default=DEFAULT_IMAGE_SIZES,
        help=f"derived image-size cache (built when absent) (default: {DEFAULT_IMAGE_SIZES})",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help=f"COCO images root used to build the size cache (default: {DEFAULT_IMAGES_ROOT})",
    )
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=DEFAULT_IMAGE_MANIFEST,
        help=f"image manifest csv used to build the size cache (default: {DEFAULT_IMAGE_MANIFEST})",
    )
    parser.add_argument(
        "--cosine-predictions",
        type=Path,
        default=Path(COSINE_PREDICTIONS),
        help=(
            "per-sentence predictions of the frozen cosine audit (global_T variant); "
            f"missing -> paired_vs_cosine.csv is stubbed (default: {COSINE_PREDICTIONS})"
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"timestamped run log (default: {DEFAULT_LOG_FILE}; '' disables)",
    )
    return parser


def _make_logger(log_file: Optional[Path]):
    """Return ``(log, handle)``: timestamped console + file logger.

    The file is opened lazily in append mode; an unopenable path (e.g. locked by
    a shell redirection) degrades to console-only logging instead of failing.
    """
    handle: Optional[TextIO] = None
    if log_file is not None and str(log_file) != "":
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
        except OSError:
            handle = None

    def log(message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        try:
            print(line, flush=True)
        except Exception:  # noqa: BLE001 - console encoding is best effort
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        if handle is not None:
            handle.write(line + "\n")
            handle.flush()

    return log, handle


def _stage_bar(desc: str, position: int):
    """A lazily mounted filling tqdm bar shared by one callback family."""
    holder = {"bar": None}

    def on_progress(done: int, total: int, label: str) -> None:
        bar = holder["bar"]
        if bar is None:
            bar = tqdm(
                total=int(total),
                desc=desc,
                unit="step",
                disable=None,
                position=position,
                leave=True,
            )
            holder["bar"] = bar
        bar.n = int(min(done, int(total)))
        bar.set_postfix_str(label)
        bar.refresh()

    def close() -> None:
        if holder["bar"] is not None:
            holder["bar"].close()

    return on_progress, close


def _require_file(path: Path, what: str, parser: argparse.ArgumentParser) -> Path:
    if not path.exists():
        parser.error(f"{what} not found: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log, handle = _make_logger(args.log_file)
    on_scoring, close_scoring = _stage_bar("scoring", 0)
    on_metrics, close_metrics = _stage_bar("metrics", 1)
    on_bootstrap, close_bootstrap = _stage_bar("bootstrap", 2)

    try:
        features = _require_file(args.features, "feature cache root", parser)
        manifests = _require_file(args.manifests, "manifest root", parser)
        bank = _require_file(args.bank, "proposal bank", parser)
        refs = _require_file(args.refs, "refs(unc).p", parser)

        log(f"[run_phase0b] features  : {features}")
        log(f"[run_phase0b] manifests : {manifests}")
        log(f"[run_phase0b] bank      : {bank}")
        log(f"[run_phase0b] refs      : {refs}")
        log(f"[run_phase0b] out       : {args.out}")
        log(f"[run_phase0b] seeds={list(args.seeds)} ks={list(args.ks)} splits={list(args.splits)}")
        log(f"[run_phase0b] device={args.device} batch_size={args.batch_size} resume={args.resume}")
        log(
            f"[run_phase0b] bootstrap: {args.bootstrap_replicates} replicates, "
            f"seed={args.bootstrap_seed}, ci={args.bootstrap_ci}"
        )

        result = run_b3_eval(
            features_root=features,
            manifests_root=manifests,
            bank_path=bank,
            refs=refs,
            out_dir=args.out,
            seeds=tuple(int(s) for s in args.seeds),
            ks=tuple(int(k) for k in args.ks),
            splits=tuple(str(name) for name in args.splits),
            regime=str(args.regime),
            bootstrap_replicates=int(args.bootstrap_replicates),
            bootstrap_seed=int(args.bootstrap_seed),
            bootstrap_ci=float(args.bootstrap_ci),
            device=str(args.device),
            image_sizes_path=args.image_sizes,
            images_root=args.images_root,
            image_manifest_csv=args.image_manifest,
            cosine_predictions=args.cosine_predictions,
            resume=bool(args.resume),
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            on_scoring=on_scoring,
            on_metrics=on_metrics,
            on_bootstrap=on_bootstrap,
            log=log,
        )
    finally:
        close_scoring()
        close_metrics()
        close_bootstrap()

    log(f"[run_phase0b] evaluation sentences: {result['n_records']}")
    for seed_key, payload in sorted(result["per_seed"].items()):
        log(
            f"[run_phase0b] seed {seed_key}: status={payload.get('status')} "
            f"T*={payload.get('temperature_corrected')} interior={payload.get('corrected_interior')} "
            f"oracle={payload.get('oracle_temperatures')}"
        )
    log(
        f"[run_phase0b] wrote {result['n_aggregate_rows']} aggregate rows, "
        f"{result['n_paired_rows']} paired rows (status={result['paired_status']}); "
        f"total {result['total_seconds']:.2f}s -> {result['out_dir']}"
    )
    if handle is not None:
        handle.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
