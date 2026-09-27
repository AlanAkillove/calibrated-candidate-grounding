"""Run the Phase 0A fixed-CLIP cosine audit (candidate-set reliability study).

This is a thin CLI over :func:`ccg.experiment.phase0a.run_audit`: it loads the
frozen feature cache + candidate manifests, scores every sentence's nested
candidate sets ``C_K = [target] + distractor_order[:K-1]``, runs the three hard
sanity checks (score invariance / rank monotonicity / accuracy monotonicity -
a violation writes ``VALIDATION_FAILURE.json`` and exits 2), fits the legal
global temperature on ``val_calib`` (K in {5, 10} only), and writes the full
artifact set (metrics tables, reliability bins, calibration-map shift, paired
image-clustered bootstrap CIs, the oracle diagnostic and diagnostic figures)
into ``--out``.

Only the ``random`` construction regime is audited here (the same-category
regime is a separate experiment; CLIP-hard construction is forbidden by the
protocol).  The ``device`` flag is recorded for provenance - the whole
pipeline is NumPy/CPU.

Usage
-----
    python scripts/run_phase0a.py \
        --features cache/features \
        --manifests cache/manifests \
        --refs "data/raw/refcoco+/refcoco+/refs(unc).p" \
        --instances "data/raw/refcoco+/refcoco+/instances.json" \
        --out results/phase0a_cosine \
        --regime random \
        --bootstrap-replicates 5000 \
        --device cpu

Progress reporting: two live tqdm bars (stderr, ``disable=None``).  A
stage-level bar ticks once per completed stage of :func:`run_audit` (eight
stages), and a second filling bar tracks the paired bootstrap via its
``on_bootstrap_pair`` hook - one tick per finished comparison, label = split /
metric / variant / K-pair, exact planned total.  The bootstrap is the long
stage, so it gets its own bar; per-stage log lines and the closing
stage-duration summary remain the non-TTY (redirected) progress record.

Exit codes: 0 on success; 2 when a hard sanity check STOPs the audit (the
failure JSON was written first); any other exception propagates with its
traceback.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.experiment.phase0a import (  # noqa: E402  (pure numpy / lazy h5py)
    DEFAULT_KS,
    PRIMARY_SPLITS,
    REGIME_RANDOM,
    run_audit,
)

__all__ = [
    "DEFAULT_FEATURES",
    "DEFAULT_MANIFESTS",
    "DEFAULT_REFS",
    "DEFAULT_INSTANCES",
    "DEFAULT_OUT",
    "build_parser",
    "main",
]

DEFAULT_FEATURES = Path("cache/features")
#: Either the ``manifests/`` directory itself or its parent (``cache``); the
#: loader accepts both layouts (see ``phase0a._manifest_file_for``).
DEFAULT_MANIFESTS = Path("cache/manifests")
DEFAULT_REFS = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
DEFAULT_INSTANCES = Path("data/raw/refcoco+/refcoco+/instances.json")
DEFAULT_OUT = Path("results/phase0a_cosine")

#: ``run_audit`` stages in execution order; each completion logs
#: "[phase0a] <name>: <seconds>s" through the ``log`` callback, which turns the
#: low-frequency stage signal into the CLI's stage-level progress bar.
_PHASE0A_STAGES = (
    "load_inputs",
    "score_sets",
    "hard_checks",
    "fit_global_temperature",
    "metric_tables",
    "bootstrap",
    "oracle_diagnostic",
    "write_artifacts",
)
_STAGE_DONE_RE = re.compile(r"\[phase0a\] ([a-z_]+): ([0-9.]+)s$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help=f"feature cache root written by the feature extraction (default: {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        default=DEFAULT_MANIFESTS,
        help=(
            "candidate-manifest root: either the manifests/ directory itself or its "
            f"parent (default: {DEFAULT_MANIFESTS})"
        ),
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=DEFAULT_REFS,
        help=f"path to refs(unc).p (default: {DEFAULT_REFS})",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=None,
        help=(
            "optional COCO instances.json, recorded for provenance only - the "
            "sentence set of this audit comes from refs(unc).p"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"artifact directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--regime",
        default=REGIME_RANDOM,
        choices=[REGIME_RANDOM],
        help=(
            "construction regime; only 'random' is audited in Phase 0A "
            "(same-category is a separate experiment, CLIP-hard is forbidden)"
        ),
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
        "--bootstrap-replicates",
        type=int,
        default=5000,
        help="paired image-clustered bootstrap replicates (protocol minimum: 5000)",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="bootstrap RNG seed")
    parser.add_argument("--bootstrap-ci", type=float, default=0.95, help="bootstrap CI level")
    parser.add_argument(
        "--device",
        default="cpu",
        help="recorded for provenance only; the audit itself is NumPy/CPU",
    )
    return parser


def _make_logger(bar=None) -> "object":
    """Timestamped console log; advances ``bar`` on every stage-done line.

    While ``bar`` is live the bar line is cleared before (and redrawn after)
    each message so log lines and the bar do not clobber each other; both calls
    are no-ops on a disabled bar (non-TTY).
    """

    def log(message: str) -> None:
        if bar is not None:
            bar.clear()
        print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)
        if bar is None:
            return
        match = _STAGE_DONE_RE.match(message)
        if match is not None and match.group(1) in _PHASE0A_STAGES:
            done = match.group(1)
            index = _PHASE0A_STAGES.index(done)
            bar.update(1)
            nxt = _PHASE0A_STAGES[index + 1] if index + 1 < len(_PHASE0A_STAGES) else "done"
            bar.set_postfix_str(f"{done} {match.group(2)}s, next={nxt}")
        bar.refresh()

    return log


def _require_file(path: Path, what: str, parser: argparse.ArgumentParser) -> Path:
    if not path.exists():
        parser.error(f"{what} not found: {path}")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Stage-level bar ticks on the "[phase0a] <stage>: <seconds>s" log lines;
    # the finer bootstrap bar is mounted inside _run via the on_bootstrap_pair
    # hook (stderr, disable=None -> hidden when redirected).
    with tqdm(
        total=len(_PHASE0A_STAGES),
        desc="phase0a",
        unit="stage",
        disable=None,
    ) as stage_bar:
        return _run(parser, args, stage_bar)


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace, stage_bar: "tqdm") -> int:
    log = _make_logger(stage_bar)
    bootstrap_bar: Optional["tqdm"] = None

    def on_bootstrap_pair(index: int, total: int, label: str) -> None:
        """Fill one tick per finished bootstrap comparison (exact planned total)."""
        nonlocal bootstrap_bar
        if bootstrap_bar is None:
            bootstrap_bar = tqdm(
                total=int(total),
                desc="bootstrap",
                unit="pair",
                disable=None,
                position=1,
                leave=True,
            )
        bootstrap_bar.update(1)
        bootstrap_bar.set_postfix_str(label)

    features = _require_file(args.features, "feature cache root", parser)
    manifests = _require_file(args.manifests, "manifest root", parser)
    refs = _require_file(args.refs, "refs(unc).p", parser)
    if args.instances is not None:
        _require_file(args.instances, "--instances", parser)
        log(
            "[run_phase0a] --instances given but not read by this audit "
            "(sentences come from refs(unc).p); recorded for provenance only"
        )

    log(f"[run_phase0a] features  : {features}")
    log(f"[run_phase0a] manifests : {manifests}")
    log(f"[run_phase0a] refs      : {refs}")
    log(f"[run_phase0a] out       : {args.out}")
    log(f"[run_phase0a] regime={args.regime} ks={list(args.ks)} splits={list(args.splits)}")
    log(
        f"[run_phase0a] bootstrap: {args.bootstrap_replicates} replicates, "
        f"seed={args.bootstrap_seed}, ci={args.bootstrap_ci}"
    )

    try:
        result = run_audit(
            features_root=features,
            manifests_root=manifests,
            refs=refs,
            out_dir=args.out,
            ks=tuple(int(k) for k in args.ks),
            splits=tuple(str(name) for name in args.splits),
            regime=args.regime,
            bootstrap_replicates=int(args.bootstrap_replicates),
            bootstrap_seed=int(args.bootstrap_seed),
            bootstrap_ci=float(args.bootstrap_ci),
            device=str(args.device),
            on_bootstrap_pair=on_bootstrap_pair,
            log=log,
        )
    finally:
        if bootstrap_bar is not None:
            bootstrap_bar.close()

    pooled = result["cohort_summary"]["splits"].get("__pooled__", {})
    log(
        f"[run_phase0a] pooled common cohort: "
        f"{pooled.get('n_common_sentences', '?')} sentences / "
        f"{pooled.get('n_common_refs', '?')} refs"
    )
    global_payload = result["global_temperature"]
    if "temperature" in global_payload:
        log(
            f"[run_phase0a] global temperature T* = {global_payload['temperature']:.6f} "
            f"(val_calib, K={global_payload['ks']}, n_sets={global_payload['n_sets']})"
        )
    else:
        log("[run_phase0a] global temperature: not fitted")
    for check_name, summary in result["hard_checks"].items():
        log(f"[run_phase0a] hard check {check_name}: {summary.get('status')}")
    log(
        f"[run_phase0a] wrote {result['n_bootstrap_comparisons']} bootstrap comparisons, "
        f"{len(result['figures'])} figures; total {result['total_seconds']:.2f}s"
    )
    durations = result.get("durations_seconds") or {}
    if durations:
        log(
            "[run_phase0a] stage durations: "
            + ", ".join(f"{name}={seconds:.2f}s" for name, seconds in durations.items())
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
