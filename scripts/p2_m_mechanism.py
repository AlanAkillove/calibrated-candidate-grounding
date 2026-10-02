#!/usr/bin/env python
"""V2-P2-M mechanism diagnostic: composition, not count.

Protocol: ``results/v2_proposal_robustness/p2_m_mechanism_config_freeze.json``
(commit ``87009ac``, authored before any P2-M number exists).

Question: V2-P2-C1 measured dAUROC(K5->K50) = +0.0517 / +0.0852 / +0.2366 for
RPN / DETR / GDINO under one identical frozen scorer stack.  All three banks hold
exactly 64 proposals per image and every family is evaluated at the same presented
K, so candidate *count* is ruled out arithmetically.  This diagnostic asks whether
the remaining candidate *composition* channel is measurable and ordered with the
harm:

  (a) within each family separately, does per-expression pool-growth harm rise with
      per-expression pool redundancy?  (Spearman, image-cluster bootstrap CI)
  (b) does the GDINO-minus-RPN / GDINO-minus-DETR harm gap survive inside matched
      redundancy quintiles?  (stratified shrinkage)

Reads only: committed prediction npz, the three proposal banks, COCO GT.  No model
forward pass, no fit, no new feature function, no new seed, no C4/hard artifact.

Usage::

    python scripts/p2_m_mechanism.py            # ~10 min, bootstrap-bound
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg.data import audit, bank, proposals  # noqa: E402  (frozen primitives)
from ccg.experiment import phase0a as p0a  # noqa: E402  (frozen cluster sampler)
import run_proposal_audit as v1  # noqa: E402
from scripts import p1_f4_cardinality as f4  # noqa: E402
from scripts import p1_replication_core as core  # noqa: E402

__all__ = ["run", "build_parser", "main"]

FAMILIES: Tuple[str, ...] = ("RPN", "DETR", "GDINO")
OUT_DIR = _ROOT / "results" / "v2_proposal_robustness" / "p2_m_mechanism"
FREEZE_PATH = _ROOT / "results" / "v2_proposal_robustness" / "p2_m_mechanism_config_freeze.json"
PRED_DIR = _ROOT / "results" / "v2_proposal_robustness" / "predictions"
ANNOTATIONS = _ROOT / "data" / "raw" / "annotations" / "instances_train2014.json"
IOU_THRESH = 0.5  # frozen V1 GT-matching level

# frozen decision constants (config freeze)
MIN_COHORT_ROWS = 8000
MIN_BIN_ROWS = 200
MAX_UNKNOWN_SHARE = 0.20
SHRINKAGE_MIN = 0.50
RHO_FAMILIES_REQUIRED = 2
N_BINS = 5

#: The frozen A1 grid is these 3 harm measures x the 5 redundancy measures.
A1_HARM_MEASURES: Tuple[str, ...] = ("H1a_lost_flip", "H1c_net_harm", "H2a_delta_margin")
#: Confidence-separation deltas: point columns only, never in a decision rule.
SECONDARY_HARM_MEASURES: Tuple[str, ...] = ("H2b_delta_conf_msp", "H2c_delta_clip_separation")
REDUNDANCY_MEASURES: Tuple[str, ...] = (
    "R1_same_class_distractors", "R2_unmatched_fraction",
    "R4_frac_pairs_gt_mid", "R4_frac_pairs_gt_high", "R5_query_cos_spread",
)
#: R1 is undefined when the target has no matchable COCO category, so it is
#: estimated on its own row mask (the freeze's unknown-category stop condition).
R1_NAME = "R1_same_class_distractors"
#: Class II (winner-anchored, endogenous): reported, never used in a decision rule.
SEM_CLASS_II: Tuple[str, ...] = (
    "winner_competitor_max_cos", "winner_competitor_mean_cos",
    "winner_competitor_std_cos", "winner_top5_mean_cos",
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p2-m] {msg}", flush=True)


def _rank_std(a: np.ndarray) -> np.ndarray:
    """Row-wise average ranks (ties resolved by ``scipy``), centred, unit-norm.

    After centring and scaling, a Spearman rho is a plain dot product, which lets
    one bootstrap replicate rank every vector once and score the whole grid.
    """
    r = rankdata(np.asarray(a, dtype=np.float64), axis=1)
    r = r - r.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->i", r, r))[:, None]
    norm = np.where(norm == 0.0, np.nan, norm)
    return r / norm


def rho_block(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Pairwise Spearman rho of *already standardised* rank rows, ``[len(rb), len(ra)]``.

    ``_rank_std`` centred and unit-normalised each row, so a rho is exactly the
    dot product of two rows - dividing by ``n`` again would shrink every estimate
    by ``1/n``.
    """
    return rb @ ra.T


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(rho_block(_rank_std(a[mask][None, :]), _rank_std(b[mask][None, :]))[0, 0])


# ---------------------------------------------------------------------------
# pool-level redundancy (score-independent: bank boxes + COCO GT only)
# ---------------------------------------------------------------------------
def pool_redundancy(family: str, frame: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """R1 / R2 / R4 per expression, exactly as frozen in the config freeze."""
    bank_path = core.FAMILIES[family]["bank"]
    ids = np.asarray(frame["image_id"], dtype=np.int64)
    pool = np.asarray(frame["pool"], dtype=np.int64)          # [n, K50]
    target = np.asarray(frame["target_index"], dtype=np.int64)
    uniq = np.unique(ids)
    t_red = time.perf_counter()
    gt_index = v1.load_gt_index(ANNOTATIONS, [int(i) for i in uniq])
    _log(f"    {family}: GT index over {uniq.size} images in {time.perf_counter() - t_red:.1f} s")

    n = ids.size
    t_red = time.perf_counter()
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r4m = np.full(n, np.nan)
    r4h = np.full(n, np.nan)
    unknown_target = np.zeros(n, dtype=np.int64)

    for image_id in uniq:
        rows = np.nonzero(ids == image_id)[0]
        boxes, _obj = bank.read_bank_image(bank_path, int(image_id))
        boxes = np.asarray(boxes, dtype=np.float64)
        gt = gt_index.objects(int(image_id)).non_crowd()
        cats, _ious = audit.assign_gt_category(
            boxes, np.asarray(gt.boxes), np.asarray(gt.categories), iou_thresh=IOU_THRESH
        )
        cats = np.asarray(cats, dtype=np.int64)
        for r in rows:
            cand = pool[r]
            t = int(target[r])
            ious = proposals.iou_matrix(boxes[t][None, :], boxes[cand])[0]
            others = cand != t                                  # freeze R2: target only
            members = cand[others & (ious < IOU_THRESH)]        # freeze R1: + duplicates
            if others.any():
                r2[r] = float(np.mean(cats[cand[others]] == -1))
            tc = int(cats[t])
            if tc == -1:
                unknown_target[r] = 1
            elif members.size:
                r1[r] = float(np.sum(cats[members] == tc))
            stats = audit.redundancy_stats(boxes[cand])
            r4m[r] = float(stats["frac_pairs_gt_mid"])
            r4h[r] = float(stats["frac_pairs_gt_high"])
    _log(f"    {family}: pool redundancy over {n} expressions in "
         f"{time.perf_counter() - t_red:.1f} s")
    return {
        "R1_same_class_distractors": r1,
        "R2_unmatched_fraction": r2,
        "R4_frac_pairs_gt_mid": r4m,
        "R4_frac_pairs_gt_high": r4h,
        "unknown_target": unknown_target,
    }



# ---------------------------------------------------------------------------
# cohort, pools and per-expression harm (from committed predictions)
# ---------------------------------------------------------------------------
def load_frame(family: str, limit: int = 0) -> Dict[str, np.ndarray]:
    """Common-K50 cohort rows (sorted by sentence_id) with their K50 pools.

    ``limit`` is a development-only smoke cut (first ``limit`` rows); it never
    applies to a run that writes artifacts.
    """
    t_load = time.perf_counter()
    keep = f4.common_k50_ids(family)
    corpus = core.build_corpus(family, "random")
    try:
        records = [r for r in corpus.eval_records(core.POOLED_TEST, ks=core.KS)
                   if int(r.sentence_id) in keep]
    finally:
        corpus.close()
    records.sort(key=lambda r: int(r.sentence_id))
    if limit:
        records = records[:limit]
    elif len(records) < MIN_COHORT_ROWS:  # frozen stop condition
        raise RuntimeError(f"{family}: common-K50 cohort {len(records)} < {MIN_COHORT_ROWS}")
    pools = np.empty((len(records), core.PRIMARY_KB), dtype=np.int64)
    for i, r in enumerate(records):
        distractors = np.asarray(r.distractor_order, dtype=np.int64)[: core.PRIMARY_KB - 1]
        pools[i] = np.concatenate([[int(r.target_index)], distractors])
    frame = {
        "sentence_id": np.array([int(r.sentence_id) for r in records], dtype=np.int64),
        "ref_id": np.array([int(r.ref_id) for r in records], dtype=np.int64),
        "image_id": np.array([int(r.image_id) for r in records], dtype=np.int64),
        "target_index": np.array([int(r.target_index) for r in records], dtype=np.int64),
        "pool": pools,
    }
    _log(f"    {family}: cohort rows={len(records)} in {time.perf_counter() - t_load:.1f} s")
    return frame


def _job(k: int) -> str:
    return core.job_name("random", k)


def harm_measures(family: str, frame: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-seed harm columns, aligned to ``frame`` row order (asserted)."""
    keep = set(int(s) for s in frame["sentence_id"])
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for seed in core.B3_SEEDS:
        scorer = f"b3_seed{seed}"
        a = core.load_pred(PRED_DIR, family, _job(core.K_BASELINE), scorer, keep)
        b = core.load_pred(PRED_DIR, family, _job(core.PRIMARY_KB), scorer, keep)
        if not np.array_equal(a["sentence_id"], b["sentence_id"]):
            raise RuntimeError(f"{family}/{scorer}: K5 and K50 rows are not aligned")
        ref = np.asarray(frame["sentence_id"], dtype=np.int64)
        if not np.array_equal(b["sentence_id"], ref):
            raise RuntimeError(f"{family}/{scorer}: predictions do not match the cohort order")
        idx5 = core.V2_SEM_INDEX["q_margin12"]
        idx_spread = core.V2_SEM_INDEX["query_cos_spread"]
        correct5 = np.asarray(a["correct"], dtype=np.float64)
        correct50 = np.asarray(b["correct"], dtype=np.float64)
        d = {
            "H1a_lost_flip": ((correct5 == 1) & (correct50 == 0)).astype(np.float64),
            "H1b_gained_flip": ((correct5 == 0) & (correct50 == 1)).astype(np.float64),
            "H1c_net_harm": None,  # filled below
            "H2a_delta_margin": np.asarray(a["raw_margin12"], dtype=np.float64)
            - np.asarray(b["raw_margin12"], dtype=np.float64),
            "H2b_delta_conf_msp": np.asarray(a["conf_msp"], dtype=np.float64)
            - np.asarray(b["conf_msp"], dtype=np.float64),
            "H2c_delta_clip_separation": np.asarray(a["sem14"], dtype=np.float64)[:, idx5]
            - np.asarray(b["sem14"], dtype=np.float64)[:, idx5],
            "R5_query_cos_spread": np.asarray(b["sem14"], dtype=np.float64)[:, idx_spread],
            "accuracy_K5": correct5,
            "accuracy_K50": correct50,
        }
        d["H1c_net_harm"] = d["H1a_lost_flip"] - d["H1b_gained_flip"]
        for name in SEM_CLASS_II:
            d[name] = np.asarray(b["sem14"], dtype=np.float64)[:, core.V2_SEM_INDEX[name]]
        out[scorer] = d
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def association_group(
    harm_cols: Dict[str, Sequence[np.ndarray]],
    red_cols: Dict[str, np.ndarray],
    clusters: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
) -> Dict[str, Dict[str, Any]]:
    """One family, one row mask: the frozen A1 grid of within-family Spearman rho.

    Point estimate = mean over the three B3 seeds of the per-seed within-family
    rho (``rho_std`` reports the spread over seeds).  The CI resamples whole
    images with the frozen design (:class:`ccg.experiment.phase0a._ClusterSampler`,
    ``default_rng(core.BOOTSTRAP_SEED)``, ``core.BOOTSTRAP_REPLICATES``,
    ``core.BOOTSTRAP_CI``) and re-runs **the same** mean-over-seeds estimator on
    each replicate, so point and interval estimate one functional.  The frozen
    ``phase0a.paired_cluster_bootstrap`` helper is hard-wired to
    ``metric(a) - metric(b)`` over two prediction sets, so the replicate loop
    lives here; only the sampling design is reused.

    ``replicates = 0`` returns point estimates with ``ci_low = ci_high = nan``.
    """
    h_names = list(A1_HARM_MEASURES)
    r_names = list(red_cols)
    n_seed = len(core.B3_SEEDS)
    H = np.vstack([np.vstack([np.asarray(v)[mask] for v in harm_cols[h]]) for h in h_names])
    R = np.vstack([np.asarray(red_cols[r])[mask] for r in r_names])
    ok = np.isfinite(H).all(axis=0) & np.isfinite(R).all(axis=0)
    H, R = H[:, ok], R[:, ok]
    rows = np.asarray(clusters).reshape(-1)[mask][ok]
    if int(H.shape[1]) < 3:
        raise RuntimeError(f"A1 group degenerate: only {H.shape[1]} finite rows")
    point = rho_block(_rank_std(H), _rank_std(R)).reshape(len(r_names), len(h_names), n_seed)
    reps = np.empty((max(int(replicates), 0), len(r_names), len(h_names)), dtype=np.float64)
    if reps.shape[0]:
        sampler = p0a._ClusterSampler(rows)
        rng = np.random.default_rng(int(core.BOOTSTRAP_SEED))
        for i in range(reps.shape[0]):
            draw = sampler.draw(rng)
            ra = _rank_std(H[:, draw])
            rb = _rank_std(R[:, draw])
            reps[i] = rho_block(ra, rb).reshape(len(r_names), len(h_names), n_seed).mean(axis=2)
    alpha = (1.0 - float(core.BOOTSTRAP_CI)) / 2.0
    out: Dict[str, Dict[str, Any]] = {}
    for ri, rname in enumerate(r_names):
        for hi, hname in enumerate(h_names):
            series = reps[:, ri, hi]
            finite = series[np.isfinite(series)] if series.size else np.zeros(0)
            out[f"{hname}|{rname}"] = {
                "rho_mean_of_seeds": float(point[ri, hi].mean()),
                "rho_std_of_seeds": float(point[ri, hi].std()),
                "ci_low": float(np.quantile(finite, alpha)) if finite.size else float("nan"),
                "ci_high": float(np.quantile(finite, 1.0 - alpha)) if finite.size else float("nan"),
                "per_seed_rho": [float(v) for v in point[ri, hi]],
                "n": int(H.shape[1]),
            }
    return out


def stratify(
    r1_by_family: Dict[str, np.ndarray],
    harm_by_family: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Quintiles of R1 on the pooled 3-family sample; matched harm gap per bin."""
    pooled = np.concatenate([r1_by_family[f] for f in FAMILIES])
    pooled = pooled[np.isfinite(pooled)]
    edges = [float(q) for q in np.quantile(pooled, [i / N_BINS for i in range(N_BINS + 1)])]
    edges[0] = -np.inf
    edges[-1] = np.inf
    bins: List[Dict[str, Any]] = []
    weights: List[float] = []
    gaps_rpn: List[float] = []
    gaps_detr: List[float] = []
    dropped = 0
    for bi in range(N_BINS):
        lo, hi = edges[bi], edges[bi + 1]
        cell: Dict[str, Any] = {"bin": bi, "R1_lo": lo, "R1_hi": hi}
        counts = {}
        for fam in FAMILIES:
            m = np.isfinite(r1_by_family[fam]) & (r1_by_family[fam] >= lo) & (r1_by_family[fam] < hi)
            counts[fam] = int(m.sum())
            cell[f"n_{fam}"] = counts[fam]
            cell[f"H1c_{fam}"] = float(np.nanmean(harm_by_family[fam][m])) if counts[fam] else float("nan")
        if min(counts.values()) < MIN_BIN_ROWS:
            cell["dropped"] = True
            dropped += 1
            bins.append(cell)
            continue
        cell["dropped"] = False
        w = float(sum(counts.values()))
        weights.append(w)
        gaps_rpn.append(cell["H1c_GDINO"] - cell["H1c_RPN"])
        gaps_detr.append(cell["H1c_GDINO"] - cell["H1c_DETR"])
        bins.append(cell)

    unmatched = {
        "GDINO_minus_RPN": float(np.nanmean(harm_by_family["GDINO"]) - np.nanmean(harm_by_family["RPN"])),
        "GDINO_minus_DETR": float(np.nanmean(harm_by_family["GDINO"]) - np.nanmean(harm_by_family["DETR"])),
    }
    out: Dict[str, Any] = {"bin_edges_R1": [float(e) for e in edges[1:-1]], "bins": bins,
                           "bins_dropped": dropped, "unmatched_gap_H1c": unmatched}
    if dropped <= 1 and weights:
        wsum = float(sum(weights))
        matched = {
            "GDINO_minus_RPN": float(np.dot(weights, gaps_rpn) / wsum),
            "GDINO_minus_DETR": float(np.dot(weights, gaps_detr) / wsum),
        }
        out["matched_gap_H1c"] = matched
        out["shrinkage"] = {
            k: (1.0 - matched[k] / unmatched[k]) if unmatched[k] != 0 else float("nan")
            for k in matched
        }
    else:
        out["matched_gap_H1c"] = None
        out["shrinkage"] = None
        out["shrinkage_note"] = "not reported: more than one redundancy bin fell below the frozen minimum"
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    freeze = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    limit = int(args.limit)
    write_outputs = limit == 0
    replicates = int(core.BOOTSTRAP_REPLICATES) if write_outputs and not args.no_bootstrap else 0
    if not write_outputs:
        _log(f"SMOKE MODE: first {limit} cohort rows, replicates=0, nothing is written")
    elif replicates == 0:
        _log("--no-bootstrap: point estimates only, CIs are nan")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, Dict[str, np.ndarray]] = {}
    red: Dict[str, Dict[str, np.ndarray]] = {}
    harm: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for family in FAMILIES:
        frames[family] = load_frame(family, limit=limit)
        red[family] = pool_redundancy(family, frames[family])
        harm[family] = harm_measures(family, frames[family])
        n = frames[family]["sentence_id"].size
        unk = float(red[family]["unknown_target"].mean())
        spread = [harm[family][f"b3_seed{s}"]["R5_query_cos_spread"] for s in core.B3_SEEDS]
        dev = float(np.max([np.nanmax(np.abs(spread[0] - s)) for s in spread[1:]]))
        _log(f"{family}: cohort={n} unknown-target share={unk:.4f} "
             f"R1 mean={np.nanmean(red[family]['R1_same_class_distractors']):.3f} "
             f"Class-I seed deviation={dev:.2e}")

    # ---- point table -------------------------------------------------------
    point_rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        seeds = [f"b3_seed{s}" for s in core.B3_SEEDS]
        for name in REDUNDANCY_MEASURES:
            col = (red[family]["R1_same_class_distractors"] if name == "R1_same_class_distractors"
                   else red[family]["R2_unmatched_fraction"] if name == "R2_unmatched_fraction"
                   else red[family]["R4_frac_pairs_gt_mid"] if name == "R4_frac_pairs_gt_mid"
                   else red[family]["R4_frac_pairs_gt_high"] if name == "R4_frac_pairs_gt_high"
                   else np.mean([harm[family][s]["R5_query_cos_spread"] for s in seeds], axis=0))
            point_rows.append({
                "family": family, "kind": "redundancy", "measure": name,
                "n": int(np.isfinite(col).sum()),
                "mean": float(np.nanmean(col)), "std": float(np.nanstd(col)),
                "p50": float(np.nanmedian(col)),
                "p90": float(np.nanquantile(col[np.isfinite(col)], 0.9)),
                "seed_spread": "",
            })
        for name in (list(A1_HARM_MEASURES) + list(SECONDARY_HARM_MEASURES)
                     + ["H1b_gained_flip", "accuracy_K5", "accuracy_K50"] + list(SEM_CLASS_II)):
            cols = [harm[family][s][name] for s in seeds]
            means = [float(np.nanmean(c)) for c in cols]
            point_rows.append({
                "family": family, "kind": "harm" if name.startswith("H") else "secondary",
                "measure": name, "n": int(np.isfinite(cols[0]).sum()),
                "mean": float(np.mean(means)), "std": float(np.std(means)),
                "p50": "", "p90": "",
                "seed_spread": ";".join(f"{m:.6f}" for m in means),
            })
    if write_outputs:
        _write_csv(OUT_DIR / "mechanism_point.csv", point_rows)

    # ---- A1: within-family association (frozen 3 harm x 5 redundancy grid) --
    seeds = [f"b3_seed{s}" for s in core.B3_SEEDS]
    assoc_rows: List[Dict[str, Any]] = []
    assoc: Dict[str, Dict[str, Dict[str, Any]]] = {f: {} for f in FAMILIES}
    a1_cohorts: Dict[str, Dict[str, int]] = {}
    for family in FAMILIES:
        clusters = frames[family]["image_id"]
        harm_cols = {name: [harm[family][s][name] for s in seeds] for name in A1_HARM_MEASURES}
        red_cols = {
            "R2_unmatched_fraction": red[family]["R2_unmatched_fraction"],
            "R4_frac_pairs_gt_mid": red[family]["R4_frac_pairs_gt_mid"],
            "R4_frac_pairs_gt_high": red[family]["R4_frac_pairs_gt_high"],
            "R5_query_cos_spread": np.mean(
                [harm[family][s]["R5_query_cos_spread"] for s in seeds], axis=0),
        }
        mask_a = (np.isfinite(np.vstack([np.vstack(v) for v in harm_cols.values()])).all(axis=0)
                  & np.isfinite(np.vstack(list(red_cols.values()))).all(axis=0))
        mask_b = mask_a & np.isfinite(red[family][R1_NAME])
        a1_cohorts[family] = {"group_a_rows": int(mask_a.sum()),
                              "group_b_r1_finite_rows": int(mask_b.sum())}
        cells = association_group(harm_cols, red_cols, clusters, mask_a, replicates=replicates)
        cells.update(association_group(harm_cols, {R1_NAME: red[family][R1_NAME]}, clusters,
                                       mask_b, replicates=replicates))
        for key, est in cells.items():
            assoc[family][key] = est
            hname, rname = key.split("|")
            assoc_rows.append({"family": family, "harm": hname, "redundancy": rname,
                               "rho_mean_of_seeds": est["rho_mean_of_seeds"],
                               "rho_std_of_seeds": est["rho_std_of_seeds"],
                               "ci_low": est["ci_low"], "ci_high": est["ci_high"],
                               "n": est["n"]})
            _log(f"{family} rho({hname}, {rname}) = {est['rho_mean_of_seeds']:+.4f} "
                 f"[{est['ci_low']:+.4f}, {est['ci_high']:+.4f}] n={est['n']}")
    if write_outputs:
        _write_csv(OUT_DIR / "mechanism_association.csv", assoc_rows)

    # ---- S1: matched-strata shrinkage -------------------------------------
    h1c = {f: np.mean([harm[f][s]["H1c_net_harm"] for s in seeds], axis=0) for f in FAMILIES}
    r1 = {f: red[f][R1_NAME] for f in FAMILIES}
    unknown_share = {f: float(red[f]["unknown_target"].mean()) for f in FAMILIES}
    strata = stratify(r1, h1c)
    if write_outputs:
        _write_csv(OUT_DIR / "mechanism_strata.csv", _strata_rows(strata))

    # ---- decision ---------------------------------------------------------
    rule_a_families = []
    for family in FAMILIES:
        est = assoc[family].get(f"H1c_net_harm|{R1_NAME}")
        rho = est["rho_mean_of_seeds"] if est else float("nan")
        if est and np.isfinite(rho) and rho > 0 and np.isfinite(est["ci_low"]) \
                and est["ci_low"] > 0:
            rule_a_families.append(family)
    rule_a = len(rule_a_families) >= RHO_FAMILIES_REQUIRED
    shrink = strata.get("shrinkage")
    rule_b = bool(shrink) and all(
        np.isfinite(shrink.get(k, float("nan"))) and shrink[k] >= SHRINKAGE_MIN
        for k in ("GDINO_minus_RPN", "GDINO_minus_DETR")
    )
    blocked = [f for f in FAMILIES if unknown_share[f] > MAX_UNKNOWN_SHARE]
    if blocked:
        rule_a = False
        rule_b = False
    verdict = ("MECHANISM_COMPOSITION_LINKED" if rule_a and rule_b
               else "MECHANISM_PARTIAL" if rule_a
               else "MECHANISM_GAP_PERSISTS" if rule_b
               else "MECHANISM_NOT_SUPPORTED")
    payload = {
        "artifact": "v2_p2_m_mechanism",
        "protocol": freeze["protocol"],
        "classification": freeze["classification"],
        "config_freeze": str(args.freeze),
        "families": list(FAMILIES),
        "cohorts": {f: int(frames[f]["sentence_id"].size) for f in FAMILIES},
        "a1_cohorts": a1_cohorts,
        "unknown_target_share": unknown_share,
        "note_on_binary_harm": freeze["measures"]["note_on_binary_harm"],
        "rule_a_within_family_rho": {
            "measure": f"rho(H1c_net_harm, {R1_NAME})",
            "per_family": {f: assoc[f].get(f"H1c_net_harm|{R1_NAME}") for f in FAMILIES},
            "families_passing": rule_a_families,
            "required": RHO_FAMILIES_REQUIRED,
            "passed": rule_a,
            "blocked_by_unknown_share": blocked,
        },
        "rule_b_matched_strata": {
            "bin_edges_R1": strata["bin_edges_R1"], "bins_dropped": strata["bins_dropped"],
            "unmatched_gap_H1c": strata["unmatched_gap_H1c"],
            "matched_gap_H1c": strata["matched_gap_H1c"],
            "shrinkage": strata.get("shrinkage"),
            "shrinkage_min": SHRINKAGE_MIN, "passed": rule_b,
            "note": strata.get("shrinkage_note", ""),
        },
        "verdict": verdict,
        "honesty_clause": freeze["decision_labels"]["honesty_clause"],
        "bootstrap": {
            "convention": "frozen: image-cluster sampler, replicates/seed/ci from core",
            "replicates": int(replicates), "seed": int(core.BOOTSTRAP_SEED),
            "ci": float(core.BOOTSTRAP_CI),
            "deviation": "replicate loop is local because phase0a.paired_cluster_bootstrap "
                         "is hard-wired to metric(a)-metric(b) over two prediction sets",
            "estimator": "the replicate resamples images and re-runs the point functional "
                         "(mean over B3 seeds of the within-family per-seed Spearman rho)",
        },
        "new_training_parameters": 0,
        "model_forward_passes": 0,
        "out_of_scope_confirmed": freeze["out_of_scope"],
        "wall_seconds": round(time.perf_counter() - started, 2),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not write_outputs:
        _log(f"{verdict} (smoke run, no artifacts written)")
        return payload
    (OUT_DIR / "p2_m_verdict.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _log(f"{verdict}  (rule_a={rule_a} families={rule_a_families}, rule_b={rule_b})")
    return payload


def _strata_rows(strata: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for cell in strata["bins"]:
        rows.append({k: ("" if v is None else v) for k, v in cell.items()})
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--freeze", default=str(FREEZE_PATH))
    p.add_argument("--no-bootstrap", action="store_true",
                   help="point estimates only (development; the verdict file is still written)")
    p.add_argument("--limit", type=int, default=0,
                   help="smoke: first N cohort rows per family, no CI, writes nothing")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
