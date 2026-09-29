"""One-time export of the frozen RefCOCO+ reliability coefficients (protocol A11.7).

Why this script exists
----------------------
A11 forbids fitting on its production path: the RefCOCOg runner may only
``load saved coefficients -> predict``.  The Stats Logistic coefficients,
however, were never written to disk - ``phase05_score_sufficiency/stats_logistic/``
holds only ``selection.json`` / ``metrics.csv``, and the A8.4 recovery step
(:func:`ccg.semantic.frozen.recover_frozen_models`) *re-fits* both models every
time it runs.  So the coefficients have to be produced exactly once, on the
RefCOCO+ side, and frozen into a bundle the external run can read.

What this script does, in order
-------------------------------
1. pins the BLAS thread count *before* numpy is imported (A11.9) and records it;
2. runs the pre-existing A8.4 recovery, i.e. the deterministic re-fit plus its
   five ``1e-9`` consistency checks against the frozen Phase 0.5 / Phase 1
   artifacts - this is the only fitting A11 ever performs, and it happens here,
   on RefCOCO+ rows, never on RefCOCOg;
3. proves that :mod:`ccg.semantic.frozen_load`'s closed-form ``sigmoid(x @ coef + b)``
   reproduces sklearn's ``predict_proba`` for the *actual* coefficient values
   (both the 17-d Stats and the 33-d E1b vector), which is what licenses the
   no-sklearn production path;
4. writes ``models.json`` + ``models.sha256`` and re-reads them;
5. re-checks the written bundle against the artifacts it cites
   (:func:`ccg.semantic.frozen_load.verify_against_frozen_artifacts`) at the
   untouched ``_TOL`` of A8.4;
6. verifies the OpenCLIP checkpoint identity (A11.5) and hashes every frozen
   input A11.8 lists into ``frozen_artifact_manifest.json``.

It writes nothing outside ``--out`` and refuses to overwrite an existing bundle
unless ``--force`` is given: the bundle A11 runs against must be the one that was
exported before the first external prediction.
"""

from __future__ import annotations

import os

#: A11.9: the thread count has to be fixed before BLAS loads, so this block runs
#: ahead of the numpy / sklearn imports on purpose.
_THREADS = os.environ.get("A11_BLAS_THREADS", "16")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.external import frozen_identity as fi  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import frozen as sfrozen  # noqa: E402
from ccg.semantic import frozen_load as fl  # noqa: E402

DEFAULT_OUT = Path("results/phase1e_refcocog_external/frozen_models")
SEEDS: Tuple[str, ...] = ("b3_seed1", "b3_seed2", "b3_seed3")
#: Rows of synthetic features used for the sklearn / closed-form equivalence.
_EQUIV_ROWS = 4096
#: The scale block of the equivalence probe: sklearn and a stable sigmoid must
#: agree in the saturated tail as well as around zero.
_EQUIV_SCALES: Tuple[float, ...] = (1.0, 50.0)
BUNDLE_FILENAME = fl.BUNDLE_FILENAME
CHECKSUM_FILENAME = fl.CHECKSUM_FILENAME
#: Entries :data:`FROZEN_SOURCES` must cover, checked by name after hashing.
REQUIRED_LABELS = (
    "openclip_checkpoint",
    "openclip_metadata",
    "semantic_feature_code",
    "stat_feature_code",
    "split_manifest",
    "e1b_coefficients",
    "e1b_selection",
    "stats_selection",
)


def _log(message: str) -> None:
    print(f"[a11-export] {message}", flush=True)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001 - a missing git is recorded, not fatal
        return "unknown"
    return out.stdout.strip() or "unknown"


def environment_snapshot() -> Dict[str, Any]:
    """The runtime identity A11.9/A11.10 demands be recorded (not assumed)."""
    import sklearn

    version: Dict[str, str] = {}
    try:  # torch is a hard dependency of the scorer path, but stay defensive
        import torch

        version["torch"] = torch.__version__
        version["cuda_available"] = str(bool(torch.cuda.is_available()))
    except Exception:  # noqa: BLE001
        version["torch"] = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        **version,
        "blas_threads": {
            var: os.environ.get(var, "<unset>")
            for var in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "git_sha": _git_sha(),
    }


def frozen_sources(args: argparse.Namespace) -> Dict[str, Path]:
    """Every artifact A11.8 names, labelled, for the checksum manifest."""
    phase05 = Path(args.phase05_root)
    phase1 = Path(args.phase1_root)
    b3 = Path(args.b3_root)
    emb = Path(args.emb_root)
    sources: Dict[str, Path] = {
        "semantic_feature_code": ROOT / "src" / "ccg" / "semantic" / "features.py",
        "stat_feature_code": ROOT / "src" / "ccg" / "reliability" / "features.py",
        "frozen_loader_code": ROOT / "src" / "ccg" / "semantic" / "frozen.py",
        "split_manifest": phase05 / "split_manifest.json",
        "stats_selection": phase05 / "stats_logistic" / "selection.json",
        "e1b_coefficients": phase1 / "e1_logistic" / "coefficients.csv",
        "e1b_selection": phase1 / "e1_logistic" / "selection.json",
        "embeddings_K5": emb / "embeddings_K5.npz",
        "embeddings_K10": emb / "embeddings_K10.npz",
        "embedding_index": emb / "embedding_index.json",
        "openclip_metadata": Path(args.features_root) / "metadata.json",
    }
    checkpoint = fi.checkpoint_identity(args.features_root)["checkpoint_path"]
    sources["openclip_checkpoint"] = Path(checkpoint)
    for scorer in args.seeds:
        tag = scorer.split("b3_seed")[1]
        sources[f"b3_model_{scorer}"] = b3 / f"seed_{tag}" / "model.npz"
        sources[f"b3_training_{scorer}"] = b3 / f"seed_{tag}" / "training.json"
        sources[f"temperature_{scorer}"] = b3 / f"seed_{tag}" / "eval_metadata.json"
        sources[f"raw_scores_K5_{scorer}"] = b3 / f"seed_{tag}" / "raw_scores" / "K5.npz"
        sources[f"raw_scores_K10_{scorer}"] = b3 / f"seed_{tag}" / "raw_scores" / "K10.npz"
        sources[f"stats_predictions_{scorer}"] = (
            phase05 / "predictions" / scorer / "stats_logistic.csv.gz"
        )
        sources[f"stats_normalisation_{scorer}"] = (
            phase05 / "features" / f"{scorer}_normalisation.json"
        )
        sources[f"stats_features_{scorer}"] = phase05 / "features" / f"{scorer}.npz"
    for cell in ("rand5", "hard5"):
        for scorer in args.seeds:
            sources[f"phase1f_{cell}_{scorer}"] = (
                Path(args.phase1f_root) / "predictions" / f"{cell}__{scorer}.npz"
            )
    return sources


def _sklearn_predict_proba(coef: Sequence[float], intercept: float, x: np.ndarray) -> np.ndarray:
    """sklearn's own ``predict_proba`` for a *given* ``(coef, intercept)`` pair.

    No fitting happens here: the estimator's fitted attributes are populated
    directly, so this is the reference side of the equivalence proof, using the
    coefficient values that are about to be frozen.
    """
    from sklearn.linear_model import LogisticRegression

    values = np.asarray(coef, dtype=np.float64).reshape(1, -1)
    probe = LogisticRegression()
    probe.coef_ = values
    probe.intercept_ = np.asarray([float(intercept)], dtype=np.float64)
    probe.classes_ = np.asarray([0, 1])
    probe.n_features_in_ = int(values.shape[1])
    return np.asarray(probe.predict_proba(x)[:, 1], dtype=np.float64)


def equivalence_probe(
    artifact: fl.FrozenSeedArtifacts, *, rows: int = _EQUIV_ROWS, scales: Sequence[float] = _EQUIV_SCALES
) -> Dict[str, float]:
    """Closed form vs sklearn, on the real coefficients, over several input scales."""
    rng = np.random.default_rng(20260929)
    out: Dict[str, float] = {}
    for label, coef, intercept, dim in (
        ("stats17", artifact.stats_coef, artifact.stats_intercept, artifact.stats_coef.size),
        ("e1b33", artifact.e1b_coef, artifact.e1b_intercept, artifact.e1b_coef.size),
    ):
        worst = 0.0
        for scale in scales:
            x = np.ascontiguousarray(rng.normal(size=(rows, dim)) * float(scale))
            mine = artifact._apply(coef, intercept, x, label)  # noqa: SLF001 - the probe *is* this call
            theirs = _sklearn_predict_proba(coef, intercept, x)
            worst = max(worst, float(np.abs(mine - theirs).max()))
        out[label] = worst
    return out


def build_bundle(
    recovered: sfrozen.FrozenReliability,
    *,
    scorers: Sequence[str],
    source: Dict[str, Any],
    equivalence: Dict[str, Dict[str, float]],
) -> fl.FrozenExternalModels:
    """``FrozenSeedArtifacts`` per seed, straight out of the verified recovery."""
    seeds: Dict[str, fl.FrozenSeedArtifacts] = {}
    for scorer in scorers:
        models = recovered.seeds[str(scorer)]
        stats_coef, stats_intercept = models.stats_clf.coefficients()
        e1b_coef, e1b_intercept = models.e1b_clf.coefficients()
        seeds[str(scorer)] = fl.FrozenSeedArtifacts(
            scorer=str(scorer),
            temperature=float(models.temperature),
            stats_coef=stats_coef,
            stats_intercept=stats_intercept,
            e1b_coef=e1b_coef,
            e1b_intercept=e1b_intercept,
            stats_C=float(models.stats_clf.C),
            e1b_C=float(models.e1b_clf.C),
            stats_fit=models.stats_fit,
            sem_fit=models.sem_fit,
        )
    payload = dict(source)
    payload["sklearn_vs_closed_form_max_abs"] = equivalence
    return fl.FrozenExternalModels(
        seeds=seeds,
        stats_feature_names=tuple(rfeat.stat_feature_names()),
        semantic_feature_names=tuple(sfeat.SEMANTIC_STAT_NAMES),
        source=payload,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", nargs="+", default=list(SEEDS))
    parser.add_argument("--b3-root", type=Path, default=sfrozen.DEFAULT_B3_ROOT)
    parser.add_argument("--phase05-root", type=Path, default=sfrozen.DEFAULT_PHASE05)
    parser.add_argument("--phase1-root", type=Path, default=sfrozen.DEFAULT_PHASE1)
    parser.add_argument("--emb-root", type=Path, default=sfrozen.DEFAULT_EMB_ROOT)
    parser.add_argument("--phase1f-root", type=Path, default=Path("results/phase1f_hard_semantic"))
    parser.add_argument("--features-root", type=Path, default=fi.DEFAULT_FEATURES_ROOT)
    parser.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if not args.force and (out / BUNDLE_FILENAME).exists():
        _log(
            f"{out / BUNDLE_FILENAME} already exists - refusing to overwrite it without "
            "--force (the A11 run must use the bundle exported before its first prediction)"
        )
        return 2

    started = time.perf_counter()
    snapshot = environment_snapshot()
    _log(
        "threads "
        + ", ".join(f"{k}={v}" for k, v in snapshot["blas_threads"].items())
        + f"; numpy {snapshot['numpy']}; sklearn {snapshot['sklearn']}; git {snapshot['git_sha']}"
    )

    # --- 1: the one permitted fit, with A8.4's own five 1e-9 checks ----------
    recovered = sfrozen.recover_frozen_models(
        scorers=tuple(args.seeds),
        phase05_root=Path(args.phase05_root),
        phase1_root=Path(args.phase1_root),
        emb_root=Path(args.emb_root),
        b3_root=Path(args.b3_root),
        log=_log,
    )
    _log(f"A8.4 recovery verified: {json.dumps(recovered.verification['all'])}")

    # --- 2: closed form == sklearn, on the coefficients about to be frozen ---
    equivalence: Dict[str, Dict[str, float]] = {}
    for scorer in args.seeds:
        models = recovered.seeds[str(scorer)]
        probe = equivalence_probe(
            fl.FrozenSeedArtifacts(
                scorer=str(scorer),
                temperature=float(models.temperature),
                stats_coef=models.stats_clf.coefficients()[0],
                stats_intercept=models.stats_clf.coefficients()[1],
                e1b_coef=models.e1b_clf.coefficients()[0],
                e1b_intercept=models.e1b_clf.coefficients()[1],
                stats_C=float(models.stats_clf.C),
                e1b_C=float(models.e1b_clf.C),
                stats_fit=models.stats_fit,
                sem_fit=models.sem_fit,
            )
        )
        for name, value in probe.items():
            if not value <= sfrozen._TOL:
                raise AssertionError(
                    f"{scorer}: {name} closed-form vs sklearn max|delta| = {value:.3e} exceeds "
                    f"{sfrozen._TOL:.0e} - the no-sklearn production path is not equivalent"
                )
        equivalence[str(scorer)] = probe
        _log(f"{scorer}: sklearn vs closed form {probe}")

    # --- 3: write, read back, re-verify the written bundle ------------------
    bundle = build_bundle(
        recovered,
        scorers=tuple(args.seeds),
        source={
            "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": snapshot,
            "a8_4_recovery_checks": recovered.verification,
            "protocol": "A11.7",
        },
        equivalence=equivalence,
    )
    provenance = fl.save_models(out, bundle)
    _log(f"wrote {out / BUNDLE_FILENAME} ({provenance['bundle_sha256'][:16]}...)")
    written = fl.load_models(out)
    if written.scorers != tuple(sorted(str(s) for s in args.seeds)):
        raise AssertionError(f"bundle round trip lost seeds: {written.scorers}")

    verification = fl.verify_against_frozen_artifacts(
        written,
        phase05_root=Path(args.phase05_root),
        phase1_root=Path(args.phase1_root),
        phase1f_root=Path(args.phase1f_root),
        b3_root=Path(args.b3_root),
        scorers=tuple(args.seeds),
        log=_log,
    )
    (out / "bundle_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # --- 4: encoder identity + the A11.8 checksum manifest ------------------
    identity = fi.assert_openclip_identity(args.features_root)
    _log(
        "OpenCLIP identity pinned: "
        f"{identity['frozen']['model_name']}/{identity['frozen']['pretrained']} "
        f"sha256 {identity['measured_checkpoint_sha256'][:16]}..."
    )
    sources = frozen_sources(args)
    manifest = {label: fi.file_identity(path, root=ROOT).to_dict() for label, path in sources.items()}
    missing = [label for label in REQUIRED_LABELS if label not in manifest]
    if missing:
        raise AssertionError(f"frozen source manifest misses required entries: {missing}")
    report = {
        "schema_version": fl.ARTIFACT_SCHEMA,
        "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": snapshot,
        "bundle": provenance,
        "openclip_identity": identity,
        "a8_4_recovery_checks": recovered.verification,
        "sklearn_vs_closed_form_max_abs": equivalence,
        "bundle_verification": verification,
        "tolerance": float(sfrozen._TOL),
        "artifacts": manifest,
        "seconds": round(time.perf_counter() - started, 2),
    }
    (out / "frozen_artifact_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    check = fi.verify_checksum_manifest(
        {label: entry for label, entry in manifest.items()}, root=ROOT, required=REQUIRED_LABELS
    )
    if not check["ok"]:
        raise AssertionError(f"FROZEN_ARTIFACT_FAILURE: {check['mismatches']}")
    _log(
        f"exported {len(manifest)} pinned artifacts, {len(written.scorers)} seeds, "
        f"{report['seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
