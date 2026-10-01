"""P1-F3 -- Frozen Scorer Compatibility (Route F).

Question this answers (and nothing else)::

    Can the entire frozen V1 B0 reliability stack operate meaningfully on DETR
    proposals *without adaptation*?

Route F swaps only the candidate *bank* (Faster-RCNN RPN -> DETR-R50, N=64,
query-independent, the P1-F2-frozen Proposal-B).  Every other parameter is the
frozen V1 B0 stack, loaded read-only and **never** re-fitted:

* OpenCLIP ViT-B/32 (laion2b_s34b_b79k) region/text encoders,
* B3 seed1/2/3 checkpoints (``results/phase0b_independent/seed_{s}/model.npz``),
* the per-seed global calibration temperature,
* R1 Stats Logistic + E1b Stats+Semantic Logistic (closed-form bundle,
  ``FrozenExternalModels.predict`` -- load + predict, no sklearn, no fit).

The stage reports a *primary* scorer-domain compatibility gate (Random-K5
accuracy over the three seeds), an *explanatory* score-distribution compatibility
audit vs the RPN reference, and a frozen R1/E1b head evaluation on the same
cohort, so a failure can be attributed to the grounding scorer rather than the
reliability heads (or vice versa).  It never runs K>5, never builds a
same-category cohort, and never touches C1/C4.

Run::

    conda activate deepminer
    python scripts/p1_f3_route_f_compatibility.py

Zero-fit is enforced twice: :func:`ccg.semantic.frozen_load.assert_no_fit_path`
statically scans this file, and :func:`ccg.semantic.frozen_load.fit_is_forbidden`
is held open for the whole scoring loop so any refit entry point raises.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.external import frozen_identity as fi  # noqa: E402
from ccg.features import cache as fcache  # noqa: E402
from ccg.models.b3_data import B3Corpus  # noqa: E402
from ccg.reliability import evaluate as reval  # noqa: E402
from ccg.reliability import features as rfeat  # noqa: E402
from ccg.semantic import features as sfeat  # noqa: E402
from ccg.semantic import frozen_load as fl  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402

# ---------------------------------------------------------------------------
# frozen inputs (V1 B0) -- none of these are fitted here
# ---------------------------------------------------------------------------
B3_ROOT = Path("results/phase0b_independent")
FROZEN_MODELS = Path("results/phase1e_refcocog_external/frozen_models")
FROZEN_ARTIFACT_MANIFEST = FROZEN_MODELS / "frozen_artifact_manifest.json"
RAW_SCORES_ROOT = B3_ROOT  # seed_{s}/raw_scores/K5.npz
REFS_PATH = Path("data/raw/refcoco+/refcoco+/refs(unc).p")
IMAGE_SIZES_PATH = Path("cache/image_sizes.npz")
MANIFEST_SEED = 20260927

V1_FEATURES_ROOT = Path("cache/features")
V1_MANIFESTS_ROOT = Path("cache/manifests")
V1_BANK = Path("cache/proposals.h5")
DETR_FEATURES_ROOT = Path("cache/features_detr_r50")
DETR_MANIFESTS_ROOT = Path("cache/manifests_detr")
DETR_BANK = Path("cache/proposals_detr_r50.h5")

B3_SEEDS: Tuple[int, ...] = (1, 2, 3)
K = 5  # Random-K5 only
#: Pooled-test eval splits (testA + testB); val_* are calibration/selection only.
POOLED_TEST: Tuple[str, ...] = ("testA", "testB")

DEFAULT_OUT = Path("results/v2_p1_f3_route_f")
SEM_CHUNK = 4096

# ---------------------------------------------------------------------------
# gate thresholds + collapse detector parameters (protocol P1-F3 sections 4/5)
# ---------------------------------------------------------------------------
ACC_COMPATIBLE = 0.50
ACC_DEGRADED = 0.35
COLLAPSE_STD_EPS = 1e-3
COLLAPSE_INTERVAL_WIDTH = 0.02
COLLAPSE_INTERVAL_FRAC = 0.95

#: The 17-d stat column indices reused for the score-distribution audit.
DIST_METRICS: Tuple[Tuple[str, int], ...] = (
    ("top1_raw", 1),
    ("margin12_raw", 4),
    ("msp", 8),
    ("entropy", 9),
    ("norm_entropy", 10),
    ("logsumexp", 11),
)

FAMILIES: Dict[str, Dict[str, Any]] = {
    "RPN": {
        "features_root": V1_FEATURES_ROOT,
        "manifests_root": V1_MANIFESTS_ROOT,
        "bank": V1_BANK,
    },
    "DETR": {
        "features_root": DETR_FEATURES_ROOT,
        "manifests_root": DETR_MANIFESTS_ROOT,
        "bank": DETR_BANK,
    },
}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ===========================================================================
# pure, offline-testable helpers (no I/O, deterministic)
# ===========================================================================
def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def densest_interval_fraction(
    probs: Sequence[float], *, width: float = COLLAPSE_INTERVAL_WIDTH
) -> Tuple[float, float]:
    """Largest fraction of ``probs`` inside any window of ``width``.

    Deterministic 1-D sliding-window count over the sorted probabilities.  Returns
    ``(fraction, window_lo)``; the fraction is the confidence-collapse "pile-up"
    signal of protocol section 5.
    """
    arr = np.sort(np.asarray(probs, dtype=np.float64).reshape(-1))
    n = arr.size
    if n == 0:
        return 0.0, float("nan")
    best = 0
    best_lo = float(arr[0])
    hi = 0
    for lo in range(n):
        while hi < n and arr[hi] - arr[lo] <= float(width):
            hi += 1
        count = hi - lo
        if count > best:
            best = count
            best_lo = float(arr[lo])
    return float(best) / float(n), best_lo


def std_collapse(std_value: float, *, eps: float = COLLAPSE_STD_EPS) -> bool:
    """``True`` when a reliability output's standard deviation is near zero."""
    return bool(np.isfinite(std_value)) and float(std_value) < float(eps)


def confidence_collapse(
    *, msp: Sequence[float], r1: Sequence[float], e1b: Sequence[float]
) -> Dict[str, Any]:
    """Deterministic confidence-collapse detector (protocol section 5).

    Flags a collapse when any confidence stream (calibrated MSP, R1, E1b) has a
    standard deviation below :data:`COLLAPSE_STD_EPS` **or** more than
    :data:`COLLAPSE_INTERVAL_FRAC` of its values pile into a
    :data:`COLLAPSE_INTERVAL_WIDTH`-wide interval.  This is a warning only -- it
    never changes the primary accuracy gate.
    """
    streams = {"msp": msp, "r1": r1, "e1b": e1b}
    per_stream: Dict[str, Any] = {}
    triggered = False
    reasons: List[str] = []
    for name, values in streams.items():
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        mean, std = _mean_std(arr)
        frac, lo = densest_interval_fraction(arr)
        s_collapse = std_collapse(std)
        i_collapse = frac > COLLAPSE_INTERVAL_FRAC
        per_stream[name] = {
            "mean": mean,
            "std": std,
            "std_collapse": s_collapse,
            "densest_interval_fraction": frac,
            "densest_interval_lo": lo,
            "interval_collapse": i_collapse,
        }
        if s_collapse:
            triggered = True
            reasons.append(f"{name}.std={std:.3e}<{COLLAPSE_STD_EPS:g}")
        if i_collapse:
            triggered = True
            reasons.append(
                f"{name}: {frac:.1%} within {COLLAPSE_INTERVAL_WIDTH:g} "
                f"(>= {COLLAPSE_INTERVAL_FRAC:.0%})"
            )
    return {"collapse": triggered, "reasons": reasons, "streams": per_stream}


def gate_label(mean_accuracy: float) -> str:
    """Primary scorer-domain compatibility gate (protocol section 4)."""
    if mean_accuracy >= ACC_COMPATIBLE:
        return "SCORER_COMPATIBLE"
    if mean_accuracy >= ACC_DEGRADED:
        return "SCORER_DEGRADED_BUT_USABLE"
    return "SCORER_DOMAIN_FAILURE"


def interpretation_case(mean_accuracy: float, collapse: bool) -> Dict[str, str]:
    """Compatibility interpretation matrix (protocol section 7)."""
    if mean_accuracy >= ACC_COMPATIBLE and not collapse:
        return {
            "case": "A",
            "verdict": "ROUTE_F_FULLY_USABLE",
            "note": "frozen B3 transfers to DETR and reliability heads stay live; "
            "P1-F4 (C1) / P1-F5 (C4) are scientifically interpretable.",
        }
    if mean_accuracy >= ACC_COMPATIBLE and collapse:
        return {
            "case": "B",
            "verdict": "GROUNDING_TRANSFER_OK__RELIABILITY_HEAD_SHIFT_WARNING",
            "note": "grounding transfers but reliability-head inputs may have "
            "collapsed; C1/C4 allowed yet any failure must be tagged as a "
            "possible reliability-head domain shift.",
        }
    if ACC_DEGRADED <= mean_accuracy < ACC_COMPATIBLE:
        return {
            "case": "C",
            "verdict": "SCORER_DEGRADED_BUT_USABLE",
            "note": "diagnostic C1 only; C4 headline requires caution -- report "
            "first, do not auto-advance.",
        }
    return {
        "case": "D",
        "verdict": "SCORER_DOMAIN_FAILURE",
        "note": "RPN-trained scorer cannot transfer to the DETR candidate "
        "distribution; STOP and consider the P1-A1 Proposal-Specific Scorer "
        "amendment.",
    }


# ===========================================================================
# DETR text-cache reuse (protocol section 2: query text == frozen V1 cache)
# ===========================================================================
def prepare_detr_text_cache(
    detr_root: Path = DETR_FEATURES_ROOT, v1_root: Path = V1_FEATURES_ROOT
) -> Dict[str, Any]:
    """Point the DETR feature root's text block at the frozen V1 text cache.

    Region features are bank-specific and were just re-extracted for DETR; the
    query text embeddings are bank-independent, so the V1 ``text_features.h5`` /
    ``text_index.csv`` are reused verbatim (copied if the DETR root has none) and
    their SHA is pinned to V1.  ``n_sentences`` is carried into the DETR
    ``metadata.json`` so the Phase-0A cache/refs sync check applies unchanged.
    """
    v1_text = v1_root / fcache.TEXT_FILENAME
    v1_index = v1_root / "text_index.csv"
    if not v1_text.exists():
        raise FileNotFoundError(f"V1 text cache missing: {v1_text}")
    detr_root.mkdir(parents=True, exist_ok=True)

    det_text = detr_root / fcache.TEXT_FILENAME
    if not det_text.exists():
        shutil.copy2(v1_text, det_text)
    det_index = detr_root / "text_index.csv"
    if v1_index.exists() and not det_index.exists():
        shutil.copy2(v1_index, det_index)

    v1_sha = fi.sha256_file(v1_text)
    det_sha = fi.sha256_file(det_text)
    if v1_sha != det_sha:
        raise AssertionError(
            f"DETR text cache {det_sha[:12]} does not match frozen V1 {v1_sha[:12]}"
        )

    v1_meta = json.loads((v1_root / fcache.METADATA_FILENAME).read_text(encoding="utf-8"))
    det_meta_path = detr_root / fcache.METADATA_FILENAME
    det_meta = (
        json.loads(det_meta_path.read_text(encoding="utf-8")) if det_meta_path.exists() else {}
    )
    n_sentences = int(v1_meta.get("n_sentences", 0))
    if n_sentences and det_meta.get("n_sentences") != n_sentences:
        det_meta["n_sentences"] = n_sentences
        det_meta_path.write_text(json.dumps(det_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "text_features_sha256": v1_sha,
        "reused_text_index": det_index.exists(),
        "n_sentences": n_sentences,
    }


# ===========================================================================
# scoring one family through the frozen stack
# ===========================================================================
def build_corpus(family: str) -> B3Corpus:
    cfg = FAMILIES[family]
    return B3Corpus(
        cfg["features_root"],
        cfg["manifests_root"],
        REFS_PATH,
        cfg["bank"],
        image_sizes_path=IMAGE_SIZES_PATH,
        ks=(K,),
        regime="random",
        preload=True,
    )


def pooled_test_samples(corpus: B3Corpus) -> List[Any]:
    """Random-K5 pooled-test rows: target present and a full 4-distractor order."""
    records = corpus.eval_records(POOLED_TEST, ks=(K,))
    out: List[Any] = []
    for r in records:
        if r.target_index is None:
            continue
        if int(np.asarray(r.distractor_order).size) < K - 1:
            continue
        out.append(r)
    return out


def _semantic_stats_chunked(z_q: np.ndarray, z_i: np.ndarray, scores: np.ndarray) -> np.ndarray:
    n = int(z_q.shape[0])
    out = np.empty((n, len(sfeat.SEMANTIC_STAT_NAMES)), dtype=np.float64)
    for start in range(0, n, SEM_CHUNK):
        stop = min(start + SEM_CHUNK, n)
        out[start:stop] = sfeat.semantic_stats(z_q[start:stop], z_i[start:stop], scores[start:stop])
    return out


def _load_frozen_raw_scores(seed: int, want_splits: Sequence[str]) -> Dict[int, np.ndarray]:
    """``sentence_id -> raw [K] logits`` from the frozen RPN Phase-0B scores."""
    path = RAW_SCORES_ROOT / f"seed_{seed}" / "raw_scores" / f"K{K}.npz"
    with np.load(path) as z:
        scores = np.asarray(z["scores"], dtype=np.float32)
        sid = np.asarray(z["sentence_id"], dtype=np.int64).reshape(-1)
        split = np.asarray([str(s) for s in z["eval_split"]])
    want = set(want_splits)
    mapping: Dict[int, np.ndarray] = {}
    for i in range(sid.shape[0]):
        if split[i] in want:
            mapping[int(sid[i])] = scores[i]
    return mapping


def score_family(
    family: str,
    *,
    models: Mapping[str, Any],
    bundle: fl.FrozenExternalModels,
    device: str,
    batch_size: int,
    out_dir: Path,
    verify: bool,
) -> Dict[str, Any]:
    corpus = build_corpus(family)
    try:
        samples = pooled_test_samples(corpus)
        if not samples:
            raise RuntimeError(f"{family}: no pooled-test Random-K5 rows")
        _log(f"[f3] {family}: scoring {len(samples)} pooled-test K5 rows")
        batch = hscores.materialise_examples(corpus, samples, K)
        sids = [int(s.sentence_id) for s in samples]

        rows: List[Dict[str, Any]] = []
        all_scores: Dict[str, np.ndarray] = {}
        for seed in B3_SEEDS:
            scorer = f"b3_seed{seed}"
            temperature = float(bundle.seed(scorer).temperature)
            scores = hscores.score_examples(models[scorer], batch, batch_size=batch_size)
            all_scores[scorer] = np.asarray(scores, dtype=np.float32)
            scores64 = scores.astype(np.float64)
            correct = np.argmax(scores, axis=1) == 0

            scalars = rfeat.scalar_confidence(scores64, temperature=temperature)
            stats17 = np.asarray(
                rfeat.stat_features(scores64, temperature=temperature), dtype=np.float64
            )
            sem16 = _semantic_stats_chunked(batch.z_q, batch.z_i, scores64)
            r1_conf, e1b_conf = bundle.predict(scorer, stats17, sem16)

            msp = np.asarray(scalars["msp"], dtype=np.float64)
            margin = np.asarray(scalars["margin"], dtype=np.float64)
            msp_row = reval.point_metric_row(msp, correct, probability=msp)
            r1_row = reval.point_metric_row(r1_conf, correct, probability=r1_conf)
            e1b_row = reval.point_metric_row(e1b_conf, correct, probability=e1b_conf)

            dist_rows = []
            for metric, idx in DIST_METRICS:
                col = stats17[:, idx]
                mc = float(col[correct].mean()) if correct.any() else float("nan")
                mi = float(col[~correct].mean()) if (~correct).any() else float("nan")
                dist_rows.append(
                    {"metric": metric, "mean": float(col.mean()), "std": float(col.std(ddof=0)),
                     "mean_correct": mc, "mean_incorrect": mi}
                )

            collapse = confidence_collapse(msp=msp, r1=r1_conf, e1b=e1b_conf)

            np.savez(
                out_dir / "predictions" / f"{family}__{scorer}__rand5.npz",
                sentence_id=np.asarray(sids, dtype=np.int64),
                family=np.asarray([family] * len(sids)),
                eval_split=np.asarray([str(s.eval_split) for s in samples]),
                correct=np.asarray(correct, dtype=bool),
                scores=np.asarray(scores, dtype=np.float32),
                conf_msp=msp,
                conf_r1=np.asarray(r1_conf, dtype=np.float64),
                conf_e1b=np.asarray(e1b_conf, dtype=np.float64),
                margin=margin,
                stats17=stats17,
                sem16=sem16,
            )
            rows.append(
                {
                    "family": family,
                    "scorer": scorer,
                    "seed": seed,
                    "temperature": temperature,
                    "n": int(len(samples)),
                    "accuracy": float(correct.mean()),
                    "msp": msp_row,
                    "r1": r1_row,
                    "e1b": e1b_row,
                    "distribution": dist_rows,
                    "collapse": collapse,
                    "msp_mean": float(msp.mean()),
                    "msp_std": float(msp.std(ddof=0)),
                    "margin_mean": float(margin.mean()),
                    "margin_std": float(margin.std(ddof=0)),
                }
            )
    finally:
        corpus.close()

    verify_report: Dict[str, Any] = {"performed": False}
    if verify:
        # The frozen Phase-0B raw_scores predate the current manifest cohort, so
        # the drift check runs on the intersection (the shared Random-K5 rows) --
        # the A8.4 STOP guarantee, not a re-scoring of a wider set.
        deltas: Dict[str, float] = {}
        n_common: int = 0
        for seed in B3_SEEDS:
            scorer = f"b3_seed{seed}"
            frozen = _load_frozen_raw_scores(seed, POOLED_TEST)
            common_idx = [i for i, s in enumerate(sids) if s in frozen]
            if not common_idx:
                raise KeyError(f"RPN verify seed_{seed}: empty intersection with frozen raw_scores")
            n_common = len(common_idx)
            recomputed = all_scores[scorer][np.asarray(common_idx, dtype=int)]
            reference = np.stack([frozen[sids[i]] for i in common_idx], axis=0).astype(np.float32)
            d = hscores.verify_against_raw_scores(
                {scorer: recomputed}, {scorer: reference}, k=K, atol=1e-4
            )
            deltas[scorer] = d[scorer]
        verify_report = {
            "performed": True,
            "max_abs_delta": deltas,
            "atol": 1e-4,
            "n_cohort": int(len(sids)),
            "n_intersection": int(n_common),
            "coverage": round(float(n_common) / float(len(sids)), 4) if sids else 0.0,
        }

    return {
        "rows": rows,
        "verify": verify_report,
        "n": int(len(samples)),
    }


def _aggregate_over_seeds(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    accs = [float(r["accuracy"]) for r in rows]
    mean_acc, std_acc = _mean_std(accs)

    def _stream(key: str, metric: str) -> Tuple[float, float]:
        vals = [float(r[key][metric]) for r in rows]
        return _mean_std(vals)

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "msp_auroc_mean": _stream("msp", "auroc_correct")[0],
        "msp_auroc_std": _stream("msp", "auroc_correct")[1],
        "msp_e_aurc_mean": _stream("msp", "e_aurc")[0],
        "msp_rer50_mean": _stream("msp", "rer_at_50")[0],
        "r1_auroc_mean": _stream("r1", "auroc_correct")[0],
        "r1_e_aurc_mean": _stream("r1", "e_aurc")[0],
        "r1_rer50_mean": _stream("r1", "rer_at_50")[0],
        "r1_ece_mean": _stream("r1", "ece_adaptive")[0],
        "r1_brier_mean": _stream("r1", "brier_binary")[0],
        "e1b_auroc_mean": _stream("e1b", "auroc_correct")[0],
        "e1b_e_aurc_mean": _stream("e1b", "e_aurc")[0],
        "e1b_rer50_mean": _stream("e1b", "rer_at_50")[0],
        "e1b_ece_mean": _stream("e1b", "ece_adaptive")[0],
        "e1b_brier_mean": _stream("e1b", "brier_binary")[0],
        "collapse_any": bool(any(r["collapse"]["collapse"] for r in rows)),
        "collapse_reasons": sorted({reason for r in rows for reason in r["collapse"]["reasons"]}),
        # Cohort-distribution scalars averaged over the three seeds (protocol section 8):
        # "mean MSP"/"std MSP"/"mean margin"/"std margin" describe the score
        # distribution, so we average the per-seed cohort means / cohort stds.
        "_mean_msp": _mean_std([r["msp_mean"] for r in rows])[0],
        "_std_msp": _mean_std([r["msp_std"] for r in rows])[0],
        "_mean_margin": _mean_std([r["margin_mean"] for r in rows])[0],
        "_std_margin": _mean_std([r["margin_std"] for r in rows])[0],
    }


# ===========================================================================
# frozen-identity self-check (protocol section 9)
# ===========================================================================
def frozen_identity_check(bundle: fl.FrozenExternalModels) -> Dict[str, Any]:
    """Prove the grounding-invariance of Route F: only the bank changed."""
    manifest = json.loads(FROZEN_ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    arts = manifest["artifacts"]

    b3_check: Dict[str, Any] = {}
    for seed in B3_SEEDS:
        scorer = f"b3_seed{seed}"
        recorded = arts[f"b3_model_{scorer}"]
        current = fi.sha256_file(Path(recorded["path"]))
        b3_check[scorer] = {
            "recorded_sha256": recorded["sha256"],
            "current_sha256": current,
            "unchanged": bool(current == recorded["sha256"]),
        }

    temp_check: Dict[str, Any] = {}
    for seed in B3_SEEDS:
        scorer = f"b3_seed{seed}"
        meta_path = B3_ROOT / f"seed_{seed}" / "eval_metadata.json"
        recorded_t = json.loads(meta_path.read_text(encoding="utf-8"))["temperature_corrected"]
        bundle_t = float(bundle.seed(scorer).temperature)
        temp_check[scorer] = {
            "eval_metadata_temperature": float(recorded_t),
            "bundle_temperature": bundle_t,
            "unchanged": bool(abs(bundle_t - float(recorded_t)) < 1e-12),
        }

    v1_text = fi.sha256_file(V1_FEATURES_ROOT / fcache.TEXT_FILENAME)
    det_text = fi.sha256_file(DETR_FEATURES_ROOT / fcache.TEXT_FILENAME)

    openclip_recorded = manifest.get("openclip_identity", {}) or {}
    ckpt_key = arts.get("openclip_checkpoint", {})
    ckpt_sha = fi.sha256_file(Path(ckpt_key["path"])) if ckpt_key.get("path") else None

    ok_b3 = all(v["unchanged"] for v in b3_check.values())
    ok_temp = all(v["unchanged"] for v in temp_check.values())
    ok_text = bool(v1_text == det_text)
    return {
        "b3_checkpoints": b3_check,
        "temperatures": temp_check,
        "text_cache": {
            "v1_sha256": v1_text,
            "detr_sha256": det_text,
            "unchanged": ok_text,
        },
        "openclip": {
            "checkpoint_sha256": ckpt_sha,
            "recorded_identity": openclip_recorded,
        },
        "r1_e1b_bundle": {
            "note": "R1/E1b are read from the checksum-verified bundle (models.sha256); "
            "load_models(verify_checksum=True) pinned them; they are never re-fitted.",
            "bundle_checksum_verified": True,
        },
        "grounding_invariant": bool(ok_b3 and ok_temp and ok_text),
    }


# ===========================================================================
# orchestration
# ===========================================================================
def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    _log("[f3] verifying frozen identity of the encoder + bundle")
    fi.assert_openclip_identity(V1_FEATURES_ROOT)
    bundle = fl.load_models(args.frozen_models, verify_checksum=True)
    bundle.check_feature_contract()
    missing = [f"b3_seed{s}" for s in B3_SEEDS if f"b3_seed{s}" not in bundle.seeds]
    if missing:
        raise AssertionError(f"bundle missing seeds: {missing}")

    _log("[f3] static zero-fit scan of this driver")
    fl.assert_no_fit_path([Path(__file__)])

    models = hscores.load_frozen_scorers(b3_root=args.b3_root, seeds=B3_SEEDS, device=args.device)

    prepared_text = prepare_detr_text_cache()
    _log(f"[f3] DETR text cache reused: {prepared_text['text_features_sha256'][:12]}")

    identity = frozen_identity_check(bundle)

    family_results: Dict[str, Any] = {}
    with fl.fit_is_forbidden() as tripwire:
        for family in ("RPN", "DETR"):
            verify = bool(family == "RPN" and args.verify_rpn)
            _log(f"[f3] family={family} verify_vs_frozen_raw_scores={verify}")
            family_results[family] = score_family(
                family,
                models=models,
                bundle=bundle,
                device=args.device,
                batch_size=args.batch_size,
                out_dir=out_dir,
                verify=verify,
            )
    if tripwire.hits:
        raise AssertionError(f"a forbidden fit fired during scoring: {tripwire.hits}")

    aggs = {f: _aggregate_over_seeds(r["rows"]) for f, r in family_results.items()}

    detr_mean_acc = aggs["DETR"]["mean_accuracy"]
    detr_collapse = aggs["DETR"]["collapse_any"]
    gate = gate_label(detr_mean_acc)
    case = interpretation_case(detr_mean_acc, detr_collapse)

    comparison = _build_comparison(aggs)
    _write_outputs(out_dir, family_results, aggs, identity, gate, case, comparison, prepared_text)

    return {
        "gate": gate,
        "case": case,
        "detr_mean_accuracy": detr_mean_acc,
        "detr_collapse": detr_collapse,
        "aggregates": aggs,
        "identity_invariant": identity["grounding_invariant"],
    }


def _build_comparison(aggs: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        ("Accuracy", "mean_accuracy"),
        ("MSP AUROC", "msp_auroc_mean"),
        ("E-AURC", "msp_e_aurc_mean"),
        ("RER50", "msp_rer50_mean"),
        ("R1 AUROC", "r1_auroc_mean"),
        ("E1b AUROC", "e1b_auroc_mean"),
        ("mean MSP", "_mean_msp"),
        ("std MSP", "_std_msp"),
    ]
    rows: List[Dict[str, Any]] = []
    for label, key in metrics:
        rows.append(
            {"Metric": label, "RPN_K5": _num(aggs["RPN"].get(key)), "DETR_K5": _num(aggs["DETR"].get(key))}
        )
    for label, key in (("mean margin", "_mean_margin"), ("std margin", "_std_margin")):
        rows.append(
            {"Metric": label, "RPN_K5": _num(aggs["RPN"].get(key)), "DETR_K5": _num(aggs["DETR"].get(key))}
        )
    return rows


def _num(value: Any) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


def _write_outputs(
    out_dir: Path,
    family_results: Mapping[str, Any],
    aggs: Mapping[str, Any],
    identity: Mapping[str, Any],
    gate: str,
    case: Mapping[str, str],
    comparison: Sequence[Dict[str, Any]],
    prepared_text: Mapping[str, Any],
) -> None:
    compat = {
        "protocol": "V2-P1 P1-F3 Route F frozen scorer compatibility",
        "regime": "random",
        "K": K,
        "eval_scope": list(POOLED_TEST),
        "gate_thresholds": {
            "compatible": ACC_COMPATIBLE,
            "degraded": ACC_DEGRADED,
            "collapse_std_eps": COLLAPSE_STD_EPS,
            "collapse_interval_width": COLLAPSE_INTERVAL_WIDTH,
            "collapse_interval_frac": COLLAPSE_INTERVAL_FRAC,
        },
        "detr_mean_accuracy": aggs["DETR"]["mean_accuracy"],
        "detr_std_accuracy": aggs["DETR"]["std_accuracy"],
        "primary_gate": gate,
        "interpretation": dict(case),
        "confidence_collapse": {
            "RPN": aggs["RPN"]["collapse_any"],
            "DETR": aggs["DETR"]["collapse_any"],
            "DETR_reasons": aggs["DETR"]["collapse_reasons"],
            "RPN_reasons": aggs["RPN"]["collapse_reasons"],
        },
        "rpn_reference_verify_vs_frozen_raw_scores": family_results["RPN"]["verify"],
        "per_family_per_seed": {
            family: [
                {
                    "scorer": r["scorer"],
                    "temperature": r["temperature"],
                    "n": r["n"],
                    "accuracy": r["accuracy"],
                    "msp_auroc": r["msp"]["auroc_correct"],
                    "msp_e_aurc": r["msp"]["e_aurc"],
                    "msp_rer50": r["msp"]["rer_at_50"],
                    "r1_auroc": r["r1"]["auroc_correct"],
                    "r1_e_aurc": r["r1"]["e_aurc"],
                    "r1_rer50": r["r1"]["rer_at_50"],
                    "r1_ece": r["r1"]["ece_adaptive"],
                    "r1_brier": r["r1"]["brier_binary"],
                    "e1b_auroc": r["e1b"]["auroc_correct"],
                    "e1b_e_aurc": r["e1b"]["e_aurc"],
                    "e1b_rer50": r["e1b"]["rer_at_50"],
                    "e1b_ece": r["e1b"]["ece_adaptive"],
                    "e1b_brier": r["e1b"]["brier_binary"],
                    "collapse": r["collapse"]["collapse"],
                }
                for r in family_results[family]["rows"]
            ]
            for family in ("RPN", "DETR")
        },
        "detr_text_cache": dict(prepared_text),
    }
    _atomic_json(out_dir / "scorer_compatibility.json", compat)
    _atomic_json(out_dir / "frozen_identity_check.json", dict(identity))

    with (out_dir / "comparison_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Metric", "RPN_K5", "DETR_K5"])
        writer.writeheader()
        for row in comparison:
            writer.writerow(row)

    with (out_dir / "score_distribution.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "family", "scorer", "metric", "mean", "std", "mean_correct", "mean_incorrect",
            ],
        )
        writer.writeheader()
        for family in ("RPN", "DETR"):
            for r in family_results[family]["rows"]:
                for d in r["distribution"]:
                    writer.writerow(
                        {
                            "family": family,
                            "scorer": r["scorer"],
                            "metric": d["metric"],
                            "mean": round(d["mean"], 6),
                            "std": round(d["std"], 6),
                            "mean_correct": round(d["mean_correct"], 6),
                            "mean_incorrect": round(d["mean_incorrect"], 6),
                        }
                    )
    _log(f"[f3] outputs -> {out_dir}")


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P1-F3 Route F frozen scorer compatibility")
    p.add_argument("--b3-root", type=Path, default=B3_ROOT)
    p.add_argument("--frozen-models", type=Path, default=FROZEN_MODELS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--verify-rpn", action="store_true", default=True,
                   help="cross-check RPN recomputation against the frozen Phase-0B raw scores")
    p.add_argument("--no-verify-rpn", dest="verify_rpn", action="store_false")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    _log(
        f"[f3] GATE={summary['gate']} CASE={summary['case']['case']} "
        f"verdict={summary['case']['verdict']} "
        f"DETR mean acc={summary['detr_mean_accuracy']:.4f} "
        f"collapse={summary['detr_collapse']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
