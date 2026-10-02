"""RQ4-M1 - transition-confidence decomposition of the K5 -> K50 AUROC degradation.

Question (frozen in
``results/v2_rq4_mechanism/m1_transition_confidence/protocol_freeze.json``)::

    Decompose the K5->K50 AUROC degradation into (1) correctness-state transition
    and (2) confidence-ranking change, and determine which component accounts for
    the cross-proposal-family amplification.

Classification: ``DESCRIPTIVE_MECHANISM_DECOMPOSITION``.  This script reads the frozen
per-row prediction columns that the confirmatory V2-P C1 gate already used (``correct``
and ``conf_msp`` = the frozen ``global_T_corrected`` confidence) for the three proposal
families on their own common-K50 cohorts and performs one exact two-factor Shapley
accounting.  **0 new parameters, 0 GPU, 0 model forward, 0 fit, 0 new data**: every
number it emits is an AUROC of already-stored arrays, and both corners of every family
are checked against the published C1 point metrics to 1e-12.

Freeze discipline (protocol_freeze.json, commit_plan A): no real RPN / DETR / GDINO row
may be computed before the freeze commit is pushed, so the real-data path sits behind an
explicit ``--allow-real-data`` flag *and* requires the freeze file to be git-tracked.
``--synthetic-selftest`` exercises the whole algebra / bootstrap / writer plumbing on
synthetic arrays and is the only mode allowed in the freeze round.

Run::

    conda activate deepminer
    python scripts/rq4_m1_decomposition.py --write-input-manifest     # freeze round (hashes only)
    python scripts/rq4_m1_decomposition.py --synthetic-selftest      # freeze round
    python scripts/rq4_m1_decomposition.py --allow-real-data         # result round only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from ccg.rq4 import decomposition as dec  # noqa: E402
from scripts import p1_replication_core as core  # noqa: E402 - frozen loader / constants

OUT_DIR = ROOT / "results" / "v2_rq4_mechanism" / "m1_transition_confidence"
PRED_DIR = ROOT / "results" / "v2_proposal_robustness" / "predictions"
POINT_REF = ROOT / "results" / "v2_proposal_robustness" / "p2_c1_gdino" / "c1_point.csv"
FREEZE_PATH = OUT_DIR / "protocol_freeze.json"

FAMILIES: Tuple[str, ...] = ("RPN", "DETR", "GDINO")
PAIRS: Tuple[Tuple[str, str], ...] = (("GDINO", "RPN"), ("GDINO", "DETR"), ("DETR", "RPN"))
COVERAGES: Tuple[float, ...] = (0.5, 0.8)
SEEDS: Tuple[int, ...] = tuple(core.B3_SEEDS)
GROUP_ATTR = {"S": "stable_correct", "F": "lost", "E": "stable_error", "G": "gained"}
#: quantities bootstrapped per family; the point estimate uses the same estimator.
BOOT_COMPONENT_KEYS: Tuple[str, ...] = ("A00", "A11", "D_total", "D_label", "D_conf",
                                        "share_label", "share_conf", "max_abs_residual")
#: seed-averaged quantities; the signed shares are derived from the averaged numerator
#: and denominator instead of being averaged themselves (a mean of ratios is not a ratio
#: of means, and the shares must describe the reported mean decomposition).
BOOT_MEAN_KEYS: Tuple[str, ...] = tuple(k for k in BOOT_COMPONENT_KEYS
                                        if not k.startswith("share_"))
POINT_COLS = ["family", "seed", "n", "A00", "A10", "A01", "A11", "delta_total", "D_total",
              "D_label", "D_conf", "shapley_residual", "share_label", "share_conf",
              "ordering", "component_difference_label_minus_conf", "S", "F", "E", "G"]

#: frozen input-artifact manifest (protocol_freeze.json section ``input_artifact_manifest``).
#: Built in the freeze round from file metadata only: no prediction array is loaded, so no
#: real RPN / DETR / GDINO row is touched before the freeze commit is pushed.
MANIFEST_PATH = OUT_DIR / "input_artifact_manifest.csv"
MANIFEST_COLS = ["family", "K", "seed", "role", "path", "sha256", "file_size",
                 "row_count_from_metadata_if_available", "row_count_source"]
ROLE_PREDICTION = "prediction_artifact"
ROLE_REFERENCE = "published_reference"
ROW_COUNT_SOURCE_PRED = ("published c1_point.csv 'n' column (common-K50 cohort rows for this "
                         "family/seed/K); the npz's own row count is DEFERRED to the result "
                         "round - no prediction array was loaded in the freeze round")
ROW_COUNT_SOURCE_REF = ("line count of the published C1 point table (file metadata; 48 data "
                        "rows = 3 families x (3 seeds + mean) x 4 K values)")


class FreezeViolation(RuntimeError):
    """The freeze artifact is missing or untracked: real data may not be computed yet."""


class StopCondition(RuntimeError):
    """A frozen stop condition fired; the analysis must not continue or be repaired."""


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [rq4-m1] {msg}", flush=True)


def _git(args: Sequence[str]) -> Any:
    proc = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=float) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cols))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_freeze() -> Dict[str, Any]:
    if not FREEZE_PATH.exists():
        raise FreezeViolation(f"missing protocol freeze: {FREEZE_PATH}")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("protocol") != "RQ4-M1":
        raise FreezeViolation("freeze file is not the RQ4-M1 protocol")
    if freeze.get("classification") != "DESCRIPTIVE_MECHANISM_DECOMPOSITION":
        raise FreezeViolation("freeze classification drifted")
    if not freeze.get("authored_before_any_result"):
        raise FreezeViolation("freeze does not declare authorship before any result")
    return freeze


# ---------------------------------------------------------------------------
# frozen inputs (read-only)
# ---------------------------------------------------------------------------
def published_point_reference() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """``(family, seed, K) -> point metrics`` from the confirmatory C1 CSV (read-only)."""
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    with POINT_REF.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["family"], row["seed"], int(row["K"]))] = {
                "n": int(float(row["n"])),
                "auroc_correct": float(row["auroc_correct"]),
                "rer_at_50": float(row["rer_at_50"]),
                "rer_at_80": float(row["rer_at_80"]),
                "accuracy": float(row["accuracy"]),
            }
    return out


def pred_file(family: str, k: int, seed: int) -> Path:
    return core.pred_path(PRED_DIR, family, f"random_k{k}", f"b3_seed{seed}")


# ---------------------------------------------------------------------------
# frozen input-artifact manifest (file metadata + hash only, no row payload)
# ---------------------------------------------------------------------------
def manifest_rows() -> List[Dict[str, Any]]:
    """One row per frozen input artifact: 18 prediction files + the published C1 table."""
    counts = {(fam, tag, int(k)): rec["n"]
              for (fam, tag, k), rec in published_point_reference().items()}
    rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for k in (core.K_BASELINE, core.PRIMARY_KB):
            for seed in SEEDS:
                path = pred_file(family, k, seed)
                if not path.exists():
                    raise StopCondition(f"frozen prediction artifact missing: {path}")
                rows.append({
                    "family": family, "K": int(k), "seed": f"seed{seed}",
                    "role": ROLE_PREDICTION,
                    "path": str(path.relative_to(ROOT).as_posix()),
                    "sha256": _sha256(path), "file_size": int(path.stat().st_size),
                    "row_count_from_metadata_if_available":
                        counts.get((family, f"seed{seed}", int(k)), ""),
                    "row_count_source": ROW_COUNT_SOURCE_PRED,
                })
    ref = POINT_REF
    if not ref.exists():
        raise StopCondition(f"published C1 reference artifact missing: {ref}")
    with ref.open(encoding="utf-8", newline="") as handle:
        n_data_rows = sum(1 for _ in handle) - 1
    rows.append({
        "family": "ALL", "K": "5,10,20,50", "seed": "seed1,seed2,seed3,mean",
        "role": ROLE_REFERENCE, "path": str(ref.relative_to(ROOT).as_posix()),
        "sha256": _sha256(ref), "file_size": int(ref.stat().st_size),
        "row_count_from_metadata_if_available": n_data_rows,
        "row_count_source": ROW_COUNT_SOURCE_REF,
    })
    return rows


def write_input_manifest() -> Dict[str, Any]:
    """Freeze-round entry point: write the manifest and report its hash (no rows read)."""
    rows = manifest_rows()
    _write_csv(MANIFEST_PATH, rows, MANIFEST_COLS)
    summary = {
        "path": str(MANIFEST_PATH.relative_to(ROOT).as_posix()),
        "n_prediction_rows": sum(1 for r in rows if r["role"] == ROLE_PREDICTION),
        "n_reference_rows": sum(1 for r in rows if r["role"] == ROLE_REFERENCE),
        "sha256": _sha256(MANIFEST_PATH),
        "file_size": int(MANIFEST_PATH.stat().st_size),
    }
    _log(f"manifest: {summary['n_prediction_rows']} prediction + "
         f"{summary['n_reference_rows']} reference rows, sha256={summary['sha256'][:16]}...")
    return summary


def verify_input_manifest(freeze: Mapping[str, Any]) -> Dict[str, Any]:
    """Result-round gate: the analysis may only consume the manifest-hashed artifacts."""
    if not MANIFEST_PATH.exists():
        raise FreezeViolation(f"missing input artifact manifest: {MANIFEST_PATH}")
    frozen_sha = str(freeze.get("input_manifest_sha256", ""))
    actual_sha = _sha256(MANIFEST_PATH)
    if not frozen_sha or actual_sha != frozen_sha:
        raise FreezeViolation(
            f"input_artifact_manifest.csv sha256 {actual_sha[:16]}... does not match the "
            f"frozen input_manifest_sha256 {frozen_sha[:16]}...")
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prediction_rows = [r for r in rows if r["role"] == ROLE_PREDICTION]
    if len(prediction_rows) != len(FAMILIES) * len(SEEDS) * 2:
        raise FreezeViolation(
            f"manifest lists {len(prediction_rows)} prediction artifacts, expected "
            f"{len(FAMILIES) * len(SEEDS) * 2}")
    for row in rows:
        path = ROOT / row["path"]
        if not path.exists():
            raise StopCondition(f"manifest artifact no longer present: {row['path']}")
        if _sha256(path) != row["sha256"] or int(path.stat().st_size) != int(row["file_size"]):
            raise StopCondition(
                f"manifest artifact drifted (sha256 or size): {row['path']}")
    return {"manifest_sha256": actual_sha, "n_verified": len(rows),
            "n_prediction_verified": len(prediction_rows)}


def cohort_ids(family: str) -> Set[int]:
    """The family's own common-K50 cohort, taken from the frozen prediction files.

    ``random_k50`` rows are exactly the K50-eligible expressions, and the nested-K design
    makes that the common-K50 cohort; the K5 file is intersected so a stray row cannot
    slip in.  No corpus, manifest or proposal bank is opened by this analysis.
    """
    per_k: Dict[int, Set[int]] = {}
    for k in (core.K_BASELINE, core.PRIMARY_KB):
        with np.load(pred_file(family, k, SEEDS[0]), allow_pickle=False) as handle:
            per_k[k] = {int(s) for s in np.asarray(handle["sentence_id"], dtype=np.int64)}
    ids = per_k[core.K_BASELINE] & per_k[core.PRIMARY_KB]
    if not ids:
        raise StopCondition(f"{family}: empty common-K50 cohort derived from the frozen predictions")
    if per_k[core.PRIMARY_KB] != ids:
        raise StopCondition(f"{family}: K50 rows are not nested inside the K5 rows")
    return ids


def family_seed_arrays(family: str, seed: int, ids: Set[int]) -> Dict[str, np.ndarray]:
    """Aligned (r5, r50, p5, p50) + ids, loaded through the frozen ``core.load_pred``."""
    scorer = f"b3_seed{seed}"
    k5 = core.load_pred(PRED_DIR, family, f"random_k{core.K_BASELINE}", scorer, ids)
    k50 = core.load_pred(PRED_DIR, family, f"random_k{core.PRIMARY_KB}", scorer, ids)
    if not np.array_equal(k5["sentence_id"], k50["sentence_id"]):
        raise StopCondition(f"{family}/seed{seed}: K5 and K50 rows are not matched")
    if not np.array_equal(k5["image_id"], k50["image_id"]):
        raise StopCondition(f"{family}/seed{seed}: image_id drifted between K5 and K50")
    for name, arr in (("K5", k5["conf_msp"]), ("K50", k50["conf_msp"])):
        if not np.all(np.isfinite(arr)):
            raise StopCondition(f"{family}/seed{seed}: conf_msp@{name} has non-finite entries")
    return {
        "sentence_id": k5["sentence_id"].astype(np.int64),
        "image_id": k5["image_id"].astype(np.int64),
        "r5": k5["correct"].astype(np.float64),
        "r50": k50["correct"].astype(np.float64),
        "p5": k5["conf_msp"].astype(np.float64),
        "p50": k50["conf_msp"].astype(np.float64),
    }


# ---------------------------------------------------------------------------
# point analysis
# ---------------------------------------------------------------------------
def mean_over_seeds(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def analyze_family(family: str, arrays: Sequence[Dict[str, np.ndarray]],
                   ref: Mapping[Tuple[str, str, int], Dict[str, float]]) -> Dict[str, Any]:
    """Point decomposition plus every frozen identity / reproduction check for one family."""
    if len(arrays) != len(SEEDS):
        raise StopCondition(f"{family}: expected {len(SEEDS)} seeds, got {len(arrays)}")
    per_seed: List[Dict[str, Any]] = []
    for seed, arr in zip(SEEDS, arrays):
        tag = f"seed{seed}"
        groups = dec.transition_groups(arr["r5"], arr["r50"])
        invariant = dec.check_structural_invariant(groups, family=family, seed=tag)
        recon = dec.group_weight_reconstruction(arr["p5"], arr["r5"], arr["p50"], arr["r50"], groups)
        if recon["max_abs_residual"] > dec.IDENTITY_TOLERANCE:
            raise StopCondition(
                f"{family}/{tag}: group-weight reconstruction residual "
                f"{recon['max_abs_residual']:.3e} exceeds {dec.IDENTITY_TOLERANCE}")
        point = dec.decompose(arr["p5"], arr["r5"], arr["p50"], arr["r50"])
        if abs(point.shapley_residual) > dec.IDENTITY_TOLERANCE:
            raise StopCondition(
                f"{family}/{tag}: Shapley residual {point.shapley_residual:.3e} exceeds "
                f"{dec.IDENTITY_TOLERANCE}")
        for k, corner in ((core.K_BASELINE, point.a00), (core.PRIMARY_KB, point.a11)):
            published = ref[(family, tag, int(k))]["auroc_correct"]
            if abs(corner - published) > dec.IDENTITY_TOLERANCE:
                raise StopCondition(
                    f"{family}/{tag}: reproduced K{k} AUROC {corner!r} does not match the "
                    f"published C1 value {published!r}")
        shares = point.shares()
        per_seed.append({
            "seed": tag, "n": int(groups.n),
            **point.as_dict(),
            "share_label": shares["share_label"], "share_conf": shares["share_conf"],
            "ordering": dec.component_ordering(point),
            "group_counts": recon["counts"], "pairwise": recon["pairwise"],
            "reconstructed": recon["reconstructed"],
            "group_recon_residual_max": recon["max_abs_residual"],
            "selective": dec.selective_error_sources(arr["p50"], groups, coverage_levels=COVERAGES),
            **{k: v for k, v in invariant.items() if k in ("S", "F", "E", "G")},
        })
    keys = ["A00", "A10", "A01", "A11", "delta_total", "D_total", "D_label", "D_conf",
            "shapley_residual"]
    means = mean_over_seeds(per_seed, keys)
    means["share_label"] = dec.signed_share(means["D_label"], means["D_total"])
    means["share_conf"] = dec.signed_share(means["D_conf"], means["D_total"])
    means["ordering"] = dec.component_ordering(_as_decomposition(means))
    means["component_difference_label_minus_conf"] = means["D_label"] - means["D_conf"]
    return {"family": family, "n": per_seed[0]["n"], "per_seed": per_seed, "mean": means}


def _as_decomposition(comps: Mapping[str, Any]) -> dec.Decomposition:
    return dec.Decomposition(a00=float(comps["A00"]), a10=float(comps["A10"]),
                             a01=float(comps["A01"]), a11=float(comps["A11"]),
                             delta_total=float(comps["delta_total"]),
                             d_total=float(comps["D_total"]), d_label=float(comps["D_label"]),
                             d_conf=float(comps["D_conf"]),
                             shapley_residual=float(comps["shapley_residual"]))


def family_statistic(arrays: Sequence[Dict[str, np.ndarray]]) -> Callable[[np.ndarray], Mapping[str, float]]:
    """One family's bootstrapped functional: the 3-seed mean decomposition of a draw."""
    def statistic(draw: np.ndarray) -> Mapping[str, float]:
        rows: List[Dict[str, float]] = []
        for arr in arrays:
            point = dec.decompose(arr["p5"][draw], arr["r5"][draw],
                                  arr["p50"][draw], arr["r50"][draw])
            groups = dec.transition_groups(arr["r5"][draw], arr["r50"][draw])
            recon = dec.group_weight_reconstruction(arr["p5"][draw], arr["r5"][draw],
                                                    arr["p50"][draw], arr["r50"][draw], groups)
            rows.append({**point.as_dict(), "max_abs_residual": recon["max_abs_residual"]})
        means = mean_over_seeds(rows, list(BOOT_MEAN_KEYS))
        share_label = dec.signed_share(means["D_label"], means["D_total"])
        share_conf = dec.signed_share(means["D_conf"], means["D_total"])
        means["share_label"] = float("nan") if share_label is None else share_label
        means["share_conf"] = float("nan") if share_conf is None else share_conf
        return means
    return statistic


def gap_rows(families: Mapping[str, Dict[str, Any]], boots: Mapping[str, Dict[str, Any]],
             pairs: Sequence[Tuple[str, str]] = PAIRS) -> List[Dict[str, Any]]:
    """Cross-family gaps on the 3-seed means, with independent-family replicate CIs."""
    rows: List[Dict[str, Any]] = []
    for first, second in pairs:
        gap = dec.cross_family_gap(first, families[first]["mean"], second, families[second]["mean"])
        reps_a = boots[first]["_replicates"]
        reps_b = boots[second]["_replicates"]
        for label, boot_key in (("gap_total_degradation", "D_total"),
                                ("gap_label_component", "D_label"),
                                ("gap_conf_component", "D_conf")):
            ci = dec.aligned_cross_family_gap_ci(reps_a, reps_b, key=boot_key, ci=core.BOOTSTRAP_CI)
            gap[f"{label}_ci_low"] = ci["gap_ci_low"]
            gap[f"{label}_ci_high"] = ci["gap_ci_high"]
        gap["design"] = ("descriptive CI from independent-family resampling on each family's own "
                         "common-K50 cohort, replicate-index-aligned; NOT a paired-expression CI - "
                         "paired inference comes only from the matched-expression intersection "
                         "with shared image-cluster draws")
        rows.append(gap)
    return rows


# ---------------------------------------------------------------------------
# matched-expression secondary (shared cluster draws -> paired contrast)
# ---------------------------------------------------------------------------
def matched_pair(first: str, second: str, ids: Mapping[str, Set[int]],
                 arrays: Mapping[str, Sequence[Dict[str, np.ndarray]]],
                 *, replicates: int = core.BOOTSTRAP_REPLICATES) -> Dict[str, Any]:
    """Decompose both families on their shared expressions and resample them together."""
    inter = core.matched_expression_ids(ids[first], ids[second])
    if not inter:
        raise StopCondition(f"{first}/{second}: empty matched intersection")
    keep = int(len(inter))
    wanted = np.fromiter(sorted(inter), dtype=np.int64, count=keep)
    per_seed: List[Dict[str, Any]] = []
    subs: Dict[str, List[Dict[str, np.ndarray]]] = {first: [], second: []}
    for seed, (arr_a, arr_b) in zip(SEEDS, zip(arrays[first], arrays[second])):
        index = {}
        for family, arr in ((first, arr_a), (second, arr_b)):
            pos = np.searchsorted(arr["sentence_id"], wanted)
            if pos.size != keep or not np.array_equal(arr["sentence_id"][pos], wanted):
                raise StopCondition(f"{first}/{second} seed{seed}: intersection rows not found")
            sub = {key: values[pos] for key, values in arr.items()}
            subs[family].append(sub)
            index[family] = sub
        if not np.array_equal(index[first]["sentence_id"], index[second]["sentence_id"]):
            raise StopCondition(f"{first}/{second} seed{seed}: sentence_id mismatch on the intersection")
        if not np.array_equal(index[first]["image_id"], index[second]["image_id"]):
            raise StopCondition(f"{first}/{second} seed{seed}: image_id disagrees on shared expressions")
        for family in (first, second):
            point = dec.decompose(index[family]["p5"], index[family]["r5"],
                                  index[family]["p50"], index[family]["r50"])
            per_seed.append({"family": family, "seed": f"seed{seed}", "n": keep,
                             **point.as_dict(),
                             "ordering": dec.component_ordering(point)})
    keys = ["A00", "A10", "A01", "A11", "delta_total", "D_total", "D_label", "D_conf",
            "shapley_residual"]
    means = {family: mean_over_seeds([r for r in per_seed if r["family"] == family], keys)
             for family in (first, second)}
    for family in (first, second):
        means[family]["share_label"] = dec.signed_share(means[family]["D_label"], means[family]["D_total"])
        means[family]["share_conf"] = dec.signed_share(means[family]["D_conf"], means[family]["D_total"])
        means[family]["ordering"] = dec.component_ordering(_as_decomposition(means[family]))
    gap = dec.cross_family_gap(first, means[first], second, means[second])

    def paired(draw: np.ndarray) -> Mapping[str, float]:
        out: Dict[str, float] = {}
        for family in (first, second):
            rows = [dec.decompose(arr["p5"][draw], arr["r5"][draw],
                                  arr["p50"][draw], arr["r50"][draw]).as_dict()
                    for arr in subs[family]]
            block = mean_over_seeds(rows, ["D_total", "D_label", "D_conf"])
            for key, value in block.items():
                out[f"{family}|{key}"] = value
        for key in ("D_total", "D_label", "D_conf"):
            out[f"gap|{key}"] = out[f"{first}|{key}"] - out[f"{second}|{key}"]
        return out

    boot = dec.image_cluster_bootstrap(paired, subs[first][0]["image_id"],
                                       n_replicates=int(replicates),
                                       seed=core.BOOTSTRAP_SEED, ci=core.BOOTSTRAP_CI)
    return {
        "pair": f"{first}_intersection_{second}",
        "n_expressions": keep,
        "classification": "SECONDARY_MATCHED_DIAGNOSTIC",
        "per_family_seed": per_seed,
        "family_mean": means,
        "gap": gap,
        "bootstrap": {k: v for k, v in boot.items() if k != "_replicates"},
        "design": "matched expressions, SHARED image-cluster draws across both families (paired)",
    }


# ---------------------------------------------------------------------------
# figures (exactly three)
# ---------------------------------------------------------------------------
def make_figures(families: Mapping[str, Dict[str, Any]],
                 arrays: Mapping[str, Sequence[Dict[str, np.ndarray]]],
                 gaps: Sequence[Mapping[str, Any]], fig_dir: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    names = list(families)
    written: List[str] = []

    # Fig 1: D_total = D_label + D_conf, signed stack (negatives fall below zero).
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    label_part = [families[f]["mean"]["D_label"] for f in names]
    conf_part = [families[f]["mean"]["D_conf"] for f in names]
    xpos = np.arange(len(names), dtype=np.float64)
    ax.bar(xpos, label_part, width=0.55, color="#c44e52", label="D_label (correctness transition)")
    ax.bar(xpos, conf_part, width=0.55, bottom=label_part, color="#4c72b0",
           label="D_conf (confidence change)")
    for pos, lab, cn in zip(xpos, label_part, conf_part):
        ax.annotate(f"D_total={lab + cn:+.4f}", (pos, lab + cn), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_ylabel("AUROC degradation (K5 - K50)")
    ax.set_title("RQ4-M1 Fig1 - signed Shapley decomposition of the K5->K50 AUROC drop")
    ax.legend(fontsize=8)
    fig.tight_layout()
    written.append(_save(fig, fig_dir / "fig1_signed_decomposition.png"))

    # Fig 2: K50 confidence by transition group (box + 5-95% whiskers, no violins).
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    positions: List[float] = []
    values: List[np.ndarray] = []
    ticks: List[str] = []
    for index, family in enumerate(names):
        arr = arrays[family][0]
        groups = dec.transition_groups(arr["r5"], arr["r50"])
        counts = groups.counts
        for offset, group in enumerate(("S", "F", "E")):
            mask = getattr(groups, GROUP_ATTR[group])
            if not counts[group]:
                continue
            positions.append(index * 4.0 + offset)
            values.append(arr["p50"][mask])
            ticks.append(f"{family}\n{group} n={counts[group]}")
    box = ax.boxplot(values, positions=positions, widths=0.7, showfliers=False, whis=(5, 95))
    for median in box["medians"]:
        # a median line spans its box, so its two endpoints average back to the position
        xs = np.asarray(median.get_xdata(), dtype=np.float64)
        ys = np.asarray(median.get_ydata(), dtype=np.float64)
        if xs.size == 0 or ys.size == 0:
            continue
        ax.plot([float(xs.mean())], [float(ys[0])], marker="D", color="black", markersize=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_ylabel("conf_msp (global_T_corrected) at K50")
    ax.set_title("RQ4-M1 Fig2 - K50 confidence by transition group (seed 1, 5-95% whiskers)")
    fig.tight_layout()
    written.append(_save(fig, fig_dir / "fig2_transition_group_confidence.png"))

    # Fig 3: cross-family gap split into its two components.
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    names_gap = [g["comparison"] for g in gaps]
    lab = [g["gap_label_component"] for g in gaps]
    cnf = [g["gap_conf_component"] for g in gaps]
    xpos = np.arange(len(names_gap), dtype=np.float64)
    ax.bar(xpos - 0.18, lab, width=0.34, color="#c44e52", label="label-gap component")
    ax.bar(xpos + 0.18, cnf, width=0.34, color="#4c72b0", label="confidence-gap component")
    for pos, value in zip(xpos - 0.18, lab):
        ax.annotate(f"{value:+.4f}", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7)
    for pos, value in zip(xpos + 0.18, cnf):
        ax.annotate(f"{value:+.4f}", (pos, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names_gap, fontsize=8)
    ax.set_ylabel("difference in AUROC degradation")
    ax.set_title("RQ4-M1 Fig3 - cross-family degradation gap, split by component")
    ax.legend(fontsize=8)
    fig.tight_layout()
    written.append(_save(fig, fig_dir / "fig3_cross_family_gap_components.png"))
    return written


def _save(fig: Any, path: Path) -> str:
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:                                  # rendered outside the repo (tests)
        return str(path)


# ---------------------------------------------------------------------------
# result round
# ---------------------------------------------------------------------------
def run_real(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    freeze = load_freeze()
    if not args.allow_real_data:
        raise FreezeViolation(
            "RQ4-M1 real data may only be computed after the freeze commit is pushed; "
            "pass --allow-real-data in the result round (protocol_freeze.json commit_plan A/B)")
    if _git(["ls-files", "--error-unmatch", str(FREEZE_PATH.relative_to(ROOT).as_posix())]) is None:
        raise FreezeViolation("protocol_freeze.json is not tracked by git - freeze round incomplete")
    manifest = verify_input_manifest(freeze)
    _log(f"input manifest verified: {manifest['n_verified']} artifacts "
         f"({manifest['n_prediction_verified']} prediction files)")

    ref = published_point_reference()
    ids: Dict[str, Set[int]] = {}
    arrays: Dict[str, List[Dict[str, np.ndarray]]] = {}
    families: Dict[str, Dict[str, Any]] = {}
    boots: Dict[str, Dict[str, Any]] = {}
    files_read: List[str] = []

    for family in FAMILIES:
        ids[family] = cohort_ids(family)
        published_n = ref[(family, f"seed{SEEDS[0]}", core.PRIMARY_KB)]["n"]
        if published_n != len(ids[family]):
            raise StopCondition(
                f"{family}: derived cohort {len(ids[family])} != published C1 n {published_n}")
        per_seed = []
        for seed in SEEDS:
            for k in (core.K_BASELINE, core.PRIMARY_KB):
                files_read.append(str(pred_file(family, k, seed).relative_to(ROOT)))
            per_seed.append(family_seed_arrays(family, seed, ids[family]))
        if len({int(a["r5"].shape[0]) for a in per_seed}) != 1:
            raise StopCondition(f"{family}: cohort size differs across seeds")
        arrays[family] = per_seed
        families[family] = analyze_family(family, per_seed, ref)
        families[family]["cohort_rows"] = int(len(ids[family]))
        boots[family] = dec.image_cluster_bootstrap(
            family_statistic(per_seed), per_seed[0]["image_id"],
            n_replicates=core.BOOTSTRAP_REPLICATES, seed=core.BOOTSTRAP_SEED,
            ci=core.BOOTSTRAP_CI, keep_replicates=True)
        mean = families[family]["mean"]
        _log(f"{family}: cohort={len(ids[family])} D_total={mean['D_total']:+.6f} "
             f"(label {mean['D_label']:+.6f} / conf {mean['D_conf']:+.6f}) -> {mean['ordering']}")

    gaps = gap_rows(families, boots)
    matched = [matched_pair(first, second, ids, arrays) for first, second in PAIRS]

    # Numeric artifacts are written before any rendering: every frozen stop condition has
    # already been evaluated inside analyze_family / bootstrap / matched_pair, so a plotting
    # or report failure can never destroy a computed decomposition, and no number is written
    # that has not passed the identity, reproduction and invariant checks.
    _write_point_rows(families, gaps)
    _write_support_rows(families, arrays, boots, gaps, matched)

    residual_max = max(max(abs(r["reconstructed"][c] - r[c]) for c in ("A00", "A01", "A10", "A11"))
                       for family in FAMILIES for r in families[family]["per_seed"])
    verdict = {
        "protocol": "RQ4-M1",
        "classification": freeze["classification"],
        "primary_verdict": "DECOMPOSITION_REPORTED",
        "primary_verdict_meaning": "an exact accounting cannot succeed or fail; the identities "
                                   "were checked and both components are reported as they are",
        "per_family_ordering": {f: families[f]["mean"]["ordering"] for f in FAMILIES},
        "per_family_components": {
            f: {k: families[f]["mean"][k] for k in
                ("D_total", "D_label", "D_conf", "share_label", "share_conf",
                 "component_difference_label_minus_conf")} for f in FAMILIES},
        "cross_family_gaps": {g["comparison"]: {k: g[k] for k in
                                                ("gap_total_degradation", "gap_label_component",
                                                 "gap_conf_component", "gap_residual",
                                                 "heavier_component", "share_of_gap_from_label",
                                                 "share_of_gap_from_confidence",
                                                 "gap_total_degradation_ci_low",
                                                 "gap_total_degradation_ci_high")} for g in gaps},
        "matched_secondary": {b["pair"]: {"n_expressions": b["n_expressions"],
                                          "classification": b["classification"],
                                          "ordering": {f: b["family_mean"][f]["ordering"]
                                                       for f in FAMILIES if f in b["family_mean"]}}
                              for b in matched},
        "identity_checks": {
            "max_abs_shapley_residual": float(max(
                abs(families[f]["mean"]["shapley_residual"]) for f in FAMILIES)),
            "max_abs_group_reconstruction_residual": float(residual_max),
            "tolerance": dec.IDENTITY_TOLERANCE,
            "all_within_tolerance": bool(
                residual_max <= dec.IDENTITY_TOLERANCE
                and all(abs(families[f]["mean"]["shapley_residual"]) <= dec.IDENTITY_TOLERANCE
                        for f in FAMILIES)),
        },
        "forbidden_labels_used": [],
        "stop_conditions_triggered": [],
        "stop_conditions_note": "every frozen stop condition raises and aborts the run, so an "
                                "empty list means none fired rather than one being ignored",
        "interpretation_boundary": freeze["interpretation_boundary"],
        "out_of_scope_confirmed": [
            "no R1/R2/R4/R5/H2a quantity enters the primary decomposition",
            "no added-candidate tail-pressure metric is reported (freeze NOT USED: per-candidate "
            "raw scores are absent from the frozen prediction artifacts and recovering them "
            "would need a forbidden new model forward)",
            "no new threshold, gate, model, dataset, seed or parameter",
            "no V2 or V2-P confirmatory number, gate or threshold was touched",
        ],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(OUT_DIR / "verdict.json", verdict)

    figures = make_figures(families, arrays, gaps, OUT_DIR / "figures") if args.figures else []
    write_report(freeze, families, boots, gaps, matched, verdict, figures, FAMILIES)

    metadata = {
        "artifact": "rq4_m1_transition_confidence_decomposition",
        "protocol": "RQ4-M1",
        "classification": freeze["classification"],
        "config_freeze": str(FREEZE_PATH.relative_to(ROOT)),
        "config_freeze_sha256": _sha256(FREEZE_PATH),
        "input_artifact_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "input_manifest_sha256": freeze.get("input_manifest_sha256"),
        "input_manifest_verified": manifest,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["rev-parse", "HEAD"]),
        "parent_artifacts": freeze["parent_artifacts"],
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "cohorts": {f: families[f]["cohort_rows"] for f in FAMILIES},
        "inputs_read": sorted(set(files_read)),
        "input_sha256": {p: _sha256(ROOT / p) for p in sorted(set(files_read))},
        "confidence_column": "conf_msp (frozen global_T_corrected), read as stored",
        "new_training_parameters": 0,
        "model_forward_passes": 0,
        "gpu_used": False,
        "temperature_refit": 0,
        "bootstrap": {"replicates": core.BOOTSTRAP_REPLICATES, "seed": core.BOOTSTRAP_SEED,
                      "ci": core.BOOTSTRAP_CI, "resample_unit": "image"},
        "figures": figures,
        "wall_seconds": round(time.time() - started, 2),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(OUT_DIR / "metadata.json", metadata)
    _log(f"verdict={verdict['primary_verdict']} ordering={verdict['per_family_ordering']} "
         f"wall={metadata['wall_seconds']}s")
    return {"verdict": verdict, "metadata": metadata, "families": families,
            "gaps": gaps, "matched": matched, "arrays": arrays}


def _write_point_rows(families: Mapping[str, Dict[str, Any]],
                      gaps: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for seed_row in families[family]["per_seed"]:
            rows.append({"family": family, "seed": seed_row["seed"], "n": seed_row["n"],
                         **{key: seed_row[key] for key in POINT_COLS
                            if key in seed_row and key not in ("family", "seed", "n")}})
        mean_row = {"family": family, "seed": "mean3", "n": families[family]["n"],
                    **families[family]["mean"]}
        rows.append(mean_row)
    for gap in gaps:
        rows.append({"family": gap["comparison"], "seed": "gap", "n": "", **gap})
    _write_csv(OUT_DIR / "point_decomposition.csv", rows, POINT_COLS + [
        "comparison", "gap_total_degradation", "gap_label_component", "gap_conf_component",
        "gap_residual", "share_of_gap_from_label", "share_of_gap_from_confidence",
        "heavier_component"])


def _write_support_rows(families: Mapping[str, Dict[str, Any]],
                        arrays: Mapping[str, Sequence[Dict[str, np.ndarray]]],
                        boots: Mapping[str, Dict[str, Any]],
                        gaps: Sequence[Mapping[str, Any]],
                        matched: Sequence[Mapping[str, Any]]) -> None:
    group_rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for seed, arr in zip(SEEDS, arrays[family]):
            groups = dec.transition_groups(arr["r5"], arr["r50"])
            counts = groups.counts
            for group in ("S", "F", "E", "G"):
                mask = getattr(groups, GROUP_ATTR[group])
                n_group = int(mask.sum())
                group_rows.append({
                    "family": family, "seed": f"seed{seed}", "group": group,
                    "n": n_group, "rate": n_group / float(groups.n),
                    "mean_p5": float(np.mean(arr["p5"][mask])) if n_group else "",
                    "median_p5": float(np.median(arr["p5"][mask])) if n_group else "",
                    "mean_p50": float(np.mean(arr["p50"][mask])) if n_group else "",
                    "median_p50": float(np.median(arr["p50"][mask])) if n_group else "",
                })
    _write_csv(OUT_DIR / "transition_groups.csv", group_rows, [
        "family", "seed", "group", "n", "rate", "mean_p5", "median_p5", "mean_p50", "median_p50"])

    component_rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for seed_row in families[family]["per_seed"]:
            counts = seed_row["group_counts"]
            for corner in ("A00", "A01", "A10", "A11"):
                component_rows.append({
                    "family": family, "seed": seed_row["seed"], "corner": corner,
                    "direct": seed_row[corner], "reconstructed": seed_row["reconstructed"][corner],
                    "residual": seed_row["reconstructed"][corner] - seed_row[corner],
                    "nS": counts["S"], "nF": counts["F"], "nE": counts["E"], "nG": counts["G"],
                    **seed_row["pairwise"],
                })
    _write_csv(OUT_DIR / "pairwise_auc_components.csv", component_rows, [
        "family", "seed", "corner", "direct", "reconstructed", "residual", "nS", "nF", "nE", "nG",
        "AUC_SE_p5", "AUC_FE_p5", "AUC_SF_p5", "AUC_SE_p50", "AUC_FE_p50", "AUC_SF_p50"])

    selective_rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for seed_row in families[family]["per_seed"]:
            for entry in seed_row["selective"]:
                selective_rows.append({"family": family, "seed": seed_row["seed"], **entry})
    _write_csv(OUT_DIR / "selective_error_sources.csv", selective_rows, [
        "family", "seed", "coverage", "n_accepted", "accepted_S", "accepted_F_new_errors",
        "accepted_E_persistent_errors", "accepted_G", "accepted_error_count", "risk",
        "fraction_of_accepted_errors_from_F", "fraction_of_accepted_errors_from_E"])

    boot_rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for key in BOOT_COMPONENT_KEYS:
            boot_rows.append({"scope": family, "quantity": key, **boots[family][key],
                              "design": "image-cluster bootstrap on the family's own "
                                        "common-K50 cohort; shared draws across r/p arrays"})
    _write_csv(OUT_DIR / "bootstrap_decomposition.csv", boot_rows, [
        "scope", "quantity", "point", "boot_mean", "ci_low", "ci_high", "std",
        "n_valid_replicates", "n_replicates", "ci_level", "resample_unit", "n_clusters", "n",
        "seed", "design"])

    _write_csv(OUT_DIR / "cross_family_gap_decomposition.csv", list(gaps), [
        "comparison", "gap_total_degradation", "gap_label_component", "gap_conf_component",
        "gap_residual", "share_of_gap_from_label", "share_of_gap_from_confidence",
        "heavier_component", "gap_total_degradation_ci_low", "gap_total_degradation_ci_high",
        "gap_label_component_ci_low", "gap_label_component_ci_high",
        "gap_conf_component_ci_low", "gap_conf_component_ci_high", "design"])

    matched_rows: List[Dict[str, Any]] = []
    for block in matched:
        for row in block["per_family_seed"]:
            matched_rows.append({"pair": block["pair"], "n_expressions": block["n_expressions"],
                                 "classification": block["classification"], "level": "seed",
                                 **row})
        for family, comps in block["family_mean"].items():
            matched_rows.append({"pair": block["pair"], "n_expressions": block["n_expressions"],
                                 "classification": block["classification"], "level": "family_mean",
                                 "family": family, "seed": "mean3",
                                 "n": block["n_expressions"], **comps})
        matched_rows.append({"pair": block["pair"], "n_expressions": block["n_expressions"],
                             "classification": block["classification"], "level": "paired_gap",
                             "family": block["pair"], "seed": "mean3",
                             "n": block["n_expressions"], **block["gap"]})
    _write_csv(OUT_DIR / "matched_intersection_decomposition.csv", matched_rows, [
        "pair", "n_expressions", "classification", "level", "family", "seed", "n",
        "A00", "A10", "A01", "A11", "delta_total", "D_total", "D_label", "D_conf",
        "shapley_residual", "share_label", "share_conf", "ordering",
        "comparison", "gap_total_degradation", "gap_label_component", "gap_conf_component",
        "gap_residual", "share_of_gap_from_label", "share_of_gap_from_confidence",
        "heavier_component"])


# ---------------------------------------------------------------------------
# report.md (the frozen prose of §21 is the only admissible wording)
# ---------------------------------------------------------------------------
def _sig(value: Any, digits: int = 6) -> str:
    """Signed fixed-point rendering; a missing or non-finite value says so instead of hiding."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "nan"
    return f"{number:+.{digits}f}"


def _plain(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "nan" if not np.isfinite(number) else f"{number:.{digits}f}"


def _ci(boot: Mapping[str, Any], key: str) -> str:
    block = boot.get(key)
    if not isinstance(block, Mapping):
        return "n/a"
    return f"[{_plain(block['ci_low'])}, {_plain(block['ci_high'])}]"


def write_report(freeze: Mapping[str, Any], families: Mapping[str, Dict[str, Any]],
                 boots: Mapping[str, Dict[str, Any]], gaps: Sequence[Mapping[str, Any]],
                 matched: Sequence[Mapping[str, Any]], verdict: Mapping[str, Any],
                 figures: Sequence[str],
                 names: Sequence[str] = FAMILIES) -> str:
    """Render ``report.md`` from the frozen artifacts; it computes nothing of its own."""
    boundary = verdict["interpretation_boundary"]
    lines: List[str] = ["# RQ4-M1 — Transition–Confidence Decomposition",
                        "",
                        f"Classification: `{freeze['classification']}` — descriptive mechanism "
                        "decomposition of an already-settled confirmatory effect (V2-P C1). "
                        "No new existence claim, no intervention test, no new method.",
                        "",
                        f"Protocol: `{freeze['protocol']}` (frozen before any RQ4-M1 result: "
                        "`protocol_freeze.json`).",
                        "",
                        "## 1. Identity checks (all frozen tolerances)",
                        "",
                        f"- max |Shapley residual| over families: "
                        f"`{verdict['identity_checks']['max_abs_shapley_residual']:.3e}`",
                        f"- max |group-weight reconstruction residual|: "
                        f"`{verdict['identity_checks']['max_abs_group_reconstruction_residual']:.3e}`",
                        f"- tolerance: `{verdict['identity_checks']['tolerance']:.1e}`, "
                        f"within tolerance: `{verdict['identity_checks']['all_within_tolerance']}`",
                        f"- A00 / A11 reproduce the published C1 K5 / K50 AUROC per (family, seed) "
                        "(checked at computation time; a drift would have stopped the run)",
                        f"- structural invariant G == 0 held for every family and seed "
                        "(checked in `transition_groups.csv`)",
                        f"- primary verdict: `{verdict['primary_verdict']}` — a constant, because an "
                        "exact accounting decomposition cannot succeed or fail",
                        "",
                        "## 2. Per-family decomposition of the K5→K50 AUROC degradation",
                        "",
                        "`D_total = D_label + D_conf`, both components **signed** (a negative "
                        "component means that factor improved reliability while the other worsened "
                        "it more). Shares are signed descriptive shares, never clipped.",
                        "",
                        "| family | cohort n | D_total | D_label | D_conf | share_label | "
                        "share_conf | D_label CI | D_conf CI | D_total CI | ordering |",
                        "|---|---|---|---|---|---|---|---|---|---|"]
    for family in names:
        mean = families[family]["mean"]
        boot = boots[family]
        lines.append(
            f"| {family} | {families[family]['cohort_rows']} | {_sig(mean['D_total'])} | "
            f"{_sig(mean['D_label'])} | {_sig(mean['D_conf'])} | {_plain(mean['share_label'], 4)} | "
            f"{_plain(mean['share_conf'], 4)} | {_ci(boot, 'D_label')} | {_ci(boot, 'D_conf')} | "
            f"{_ci(boot, 'D_total')} | `{mean['ordering']}` |")
    lines += ["", "Cohorts: " + ", ".join(f"{f} = {families[f]['cohort_rows']} rows "
                                          f"(per-seed n = {families[f]['n']})" for f in names) + ".",
              "", "Per-seed values are in `point_decomposition.csv` (rows `seed1/seed2/seed3`), "
                  "bootstrap details in `bootstrap_decomposition.csv`.",
              "",
              "## 3. Cross-family exact gap decomposition",
              "",
              "`D_total(a) - D_total(b) = (D_label(a) - D_label(b)) + (D_conf(a) - D_conf(b))`, "
              "exact by subtraction of two exact identities. The gap CIs are **descriptive under "
              "independent-family resampling** on each family's own cohort and are **not** "
              "paired-expression CIs.",
              "",
              "| comparison | gap D_total | gap CI | gap D_label | label-gap CI | "
              "gap D_conf | conf-gap CI | heavier component | share from label | "
              "share from confidence |",
              "|---|---|---|---|---|---|---|---|---|"]
    for gap in gaps:
        lines.append(
            f"| {gap['comparison']} | {_sig(gap['gap_total_degradation'])} | "
            f"[{_plain(gap['gap_total_degradation_ci_low'])}, "
            f"{_plain(gap['gap_total_degradation_ci_high'])}] | "
            f"{_sig(gap['gap_label_component'])} | "
            f"[{_plain(gap['gap_label_component_ci_low'])}, "
            f"{_plain(gap['gap_label_component_ci_high'])}] | "
            f"{_sig(gap['gap_conf_component'])} | "
            f"[{_plain(gap['gap_conf_component_ci_low'])}, "
            f"{_plain(gap['gap_conf_component_ci_high'])}] | `{gap['heavier_component']}` | "
            f"{_plain(gap['share_of_gap_from_label'], 4)} | "
            f"{_plain(gap['share_of_gap_from_confidence'], 4)} |")
    lines += ["", "## 4. Transition-group composition (S / F / E, G == 0 everywhere)", "",
              "Counts and confidence summaries per family × seed are in `transition_groups.csv`; "
              "the key ranking quantity `AUC(S,F;p50)` (newly introduced errors keep high "
              "confidence at K50?) is reported next to `AUC(S,E;p50)` (persistent errors) in "
              "`pairwise_auc_components.csv`.", "",
              "| family | seed | S | F | E | G | AUC(S,F;p50) | AUC(S,E;p50) |",
              "|---|---|---|---|---|---|---|---|"]
    for family in names:
        for row in families[family]["per_seed"]:
            counts = row["group_counts"]
            pairwise = row["pairwise"]
            lines.append(f"| {family} | {row['seed']} | {counts['S']} | {counts['F']} | "
                         f"{counts['E']} | {counts['G']} | {_plain(pairwise['AUC_SF_p50'])} | "
                         f"{_plain(pairwise['AUC_SE_p50'])} |")
    lines += ["", "## 5. Accepted-error sources behind the published RER@50 / RER@80 degradation",
              "",
              "No new gate: this only describes the composition of the accepted set under the "
              "frozen selective convention (`n_keep = max(1, ceil(coverage * n))`, descending "
              "`conf_msp`@K50, stable sort). Full table in `selective_error_sources.csv`.", "",
              "| family | seed | coverage | accepted F (new) | accepted E (persistent) | "
              "fraction of accepted errors from F | risk |",
              "|---|---|---|---|---|---|---|"]
    for family in names:
        for row in families[family]["per_seed"]:
            for entry in row["selective"]:
                lines.append(
                    f"| {family} | {row['seed']} | {entry['coverage']} | "
                    f"{entry['accepted_F_new_errors']} | "
                    f"{entry['accepted_E_persistent_errors']} | "
                    f"{_plain(entry['fraction_of_accepted_errors_from_F'], 4)} | "
                    f"{_plain(entry['risk'], 4)} |")
    lines += ["", "## 6. Matched-expression secondary (paired, SECONDARY_MATCHED_DIAGNOSTIC)",
              "",
              "On each intersection both families are restricted to identical expression rows and "
              "resampled with **shared** image-cluster draws, so the gap there is genuinely "
              "paired-expression. It never overrides the primary own-cohort result; it reports "
              "whether the same component ordering survives cohort matching.", "",
              "| pair | n expressions | family | D_total | D_label | D_conf | ordering |",
              "|---|---|---|---|---|---|---|"]
    for block in matched:
        for family, mean in block["family_mean"].items():
            lines.append(
                f"| {block['pair']} | {block['n_expressions']} | {family} | "
                f"{_sig(mean['D_total'])} | {_sig(mean['D_label'])} | {_sig(mean['D_conf'])} | "
                f"`{mean['ordering']}` |")
    lines += ["", "Paired gap on the same intersection (shared image-cluster draws, so this CI is "
              "the genuine paired-expression contrast):", "",
              "| comparison | gap D_total | CI | gap D_label | CI | gap D_conf | CI |",
              "|---|---|---|---|---|---|---|"]
    for block in matched:
        boot = block["bootstrap"]
        gap = block["gap"]
        lines.append(
            f"| {block['pair']} | {_sig(gap['gap_total_degradation'])} | {_ci(boot, 'gap|D_total')} | "
            f"{_sig(gap['gap_label_component'])} | {_ci(boot, 'gap|D_label')} | "
            f"{_sig(gap['gap_conf_component'])} | {_ci(boot, 'gap|D_conf')} |")
    lines += ["", "## 7. Figures (exactly three, per the frozen figure policy)", ""]
    lines += [f"- `{name}`" for name in figures] or ["- figures disabled in this run"]
    lines += ["", "## 8. Interpretation boundary (frozen wording)", "",
              "Allowed:", ""]
    lines += [f"- {sentence}" for sentence in boundary["allowed"]]
    lines += ["", "Forbidden:", ""]
    lines += [f"- {sentence}" for sentence in boundary["forbidden"]]
    lines += ["", f"Phrase rule: {boundary['phrase_rule']}", "",
              "The two research questions are answered only as far as this decomposition can answer "
              "them: Q1 by the reported components per family, Q2 by the exact cross-family gap "
              "split. `D_conf` is a confidence-**ranking**-change contribution, not a temperature or "
              "calibration-scale change (AUROC is invariant to monotone score transforms).", "",
              "Out of scope and not reported: " + "; ".join(verdict["out_of_scope_confirmed"]) + ".",
              "",
              "## 9. Provenance", "",
              "- inputs: the 18 hashed prediction artifacts plus the published C1 reference listed in "
              "`input_artifact_manifest.csv`; every real analysis consumes exactly those hashed files",
              "- no model forward pass, no training, no feature extraction, no GPU, no temperature "
              "refit, no new cohort / seed / family / threshold",
              "- full machine-readable detail in `verdict.json` and `metadata.json`", ""]
    path = OUT_DIR / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        return str(path.relative_to(ROOT))
    except ValueError:                                  # rendered outside the repo (tests)
        return str(path)


# ---------------------------------------------------------------------------
# freeze-round selftest: synthetic arrays only (never reads a frozen row)
# ---------------------------------------------------------------------------
def synthetic_arrays(n: int = 4000, seed: int = 7, flip_rate: float = 0.35) -> List[Dict[str, np.ndarray]]:
    """Three synthetic 'seeds' that obey the nested-K structure (so G == 0 by construction).

    ``image_id`` comes from a fixed side-generator so two synthetic families share the
    same expression -> image mapping, which is what the matched-intersection path expects.
    """
    rng = np.random.default_rng(seed)
    images = np.random.default_rng(91).integers(0, 600, n).astype(np.int64)
    out: List[Dict[str, np.ndarray]] = []
    for _ in range(len(SEEDS)):
        r5 = (rng.random(n) < 0.75).astype(np.float64)
        correct_pos = np.nonzero(r5 == 1.0)[0]
        flip = rng.random(correct_pos.shape[0]) < flip_rate
        r50 = r5.copy()
        r50[correct_pos[flip]] = 0.0
        p5 = np.clip(rng.normal(0.72, 0.18, n), 1e-4, 1.0) * (0.6 + 0.4 * r5)
        p50 = np.clip(rng.normal(0.55, 0.22, n), 1e-4, 1.0) * (0.6 + 0.4 * r50)
        out.append({"sentence_id": np.arange(n, dtype=np.int64),
                    "image_id": images.copy(),
                    "r5": r5, "r50": r50, "p5": p5, "p50": p50})
    return out


def synthetic_reference(arrays: Sequence[Dict[str, np.ndarray]],
                        family: str) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """The selftest reference: corners come from the synthetic arrays themselves."""
    ref: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for seed, arr in zip(SEEDS, arrays):
        a00, _, _, a11 = dec.corner_aurocs(arr["p5"], arr["r5"], arr["p50"], arr["r50"])
        for k, corner in ((core.K_BASELINE, a00), (core.PRIMARY_KB, a11)):
            ref[(family, f"seed{seed}", int(k))] = {
                "auroc_correct": corner, "n": int(arr["r5"].shape[0]),
                "rer_at_50": 0.0, "rer_at_80": 0.0, "accuracy": 0.0}
    return ref


def run_synthetic_selftest(args: argparse.Namespace) -> Dict[str, Any]:
    """Exercise the full algebra + bootstrap + matched-pair plumbing on synthetic rows only.

    Nothing is written and no frozen prediction file is opened: this is the only mode the
    freeze round is allowed to run (``protocol_freeze.json`` tests.freeze_round_rule).
    """
    arrays: Dict[str, List[Dict[str, np.ndarray]]] = {
        "SYNTH": synthetic_arrays(n=args.n, seed=11),
        "SYNTH2": synthetic_arrays(n=args.n, seed=23, flip_rate=0.15),
    }
    families: Dict[str, Dict[str, Any]] = {}
    boots: Dict[str, Dict[str, Any]] = {}
    ids: Dict[str, Set[int]] = {}
    for family, per_seed in arrays.items():
        families[family] = analyze_family(family, per_seed, synthetic_reference(per_seed, family))
        boots[family] = dec.image_cluster_bootstrap(
            family_statistic(per_seed), per_seed[0]["image_id"],
            n_replicates=args.reps, seed=core.BOOTSTRAP_SEED, ci=core.BOOTSTRAP_CI,
            keep_replicates=True)
        ids[family] = {int(v) for v in per_seed[0]["sentence_id"]}

    gaps = gap_rows(families, boots, (("SYNTH2", "SYNTH"),))
    matched = matched_pair("SYNTH2", "SYNTH", ids, arrays, replicates=args.reps)
    residual_max = max(
        max(abs(r["reconstructed"][c] - r[c]) for c in ("A00", "A01", "A10", "A11"))
        for family in families for r in families[family]["per_seed"])
    means = families["SYNTH"]["mean"]
    report = {
        "classification": "DESCRIPTIVE_MECHANISM_DECOMPOSITION",
        "mode": "synthetic selftest (no frozen RPN/DETR/GDINO row was read, nothing written)",
        "mean": means,
        "gap": gaps[0],
        "matched_classification": matched["classification"],
        "matched_n_expressions": matched["n_expressions"],
        "max_abs_group_reconstruction_residual": residual_max,
        "identities_within_tolerance": bool(
            abs(means["shapley_residual"]) <= dec.IDENTITY_TOLERANCE
            and residual_max <= dec.IDENTITY_TOLERANCE
            and abs(gaps[0]["gap_residual"]) <= dec.IDENTITY_TOLERANCE),
        "D_total_ci": [boots["SYNTH"]["D_total"]["ci_low"], boots["SYNTH"]["D_total"]["ci_high"]],
    }
    _log(f"selftest D_total={means['D_total']:+.6f} label={means['D_label']:+.6f} "
         f"conf={means['D_conf']:+.6f} ordering={means['ordering']} "
         f"identities_ok={report['identities_within_tolerance']}")
    if not report["identities_within_tolerance"]:
        raise StopCondition("synthetic selftest identities failed")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ4-M1 transition-confidence decomposition")
    parser.add_argument("--allow-real-data", action="store_true",
                        help="required in the result round; without it the real-data path refuses")
    parser.add_argument("--synthetic-selftest", action="store_true",
                        help="run the algebra / bootstrap plumbing on synthetic arrays only")
    parser.add_argument("--write-input-manifest", action="store_true",
                        help="freeze-round entry point: hash the 18 frozen prediction files "
                             "plus the published C1 table (file metadata only, no row loaded)")
    parser.add_argument("--no-figures", dest="figures", action="store_false", default=True)
    parser.add_argument("--n", type=int, default=4000, help="synthetic selftest row count")
    parser.add_argument("--reps", type=int, default=200, help="synthetic selftest replicates")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_input_manifest:
        write_input_manifest()
        return 0
    if args.synthetic_selftest:
        run_synthetic_selftest(args)
        return 0
    run_real(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
