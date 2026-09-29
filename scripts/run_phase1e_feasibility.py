"""Phase 1E feasibility driver: FineCops-Ref external audit, stages F0-F4.

Run stages in the pre-registered order (instruction section 31)::

    F0/F1  parse     - annotations, scene-graph join, distributions, audit subset
    F0b    images    - fetch exactly the test images the audit needs from GQA
    F2     rpn       - frozen N=64 RPN audit on the deterministic subset
    F3/F4  decision  - candidate availability, engineering gate, branch choice

``--stages parse`` alone must be enough to see the metadata facts; the run then
stops before any model inference, which is where this round ends (section 36).

Outputs (``results/phase1e_finecops/``)::

    feasibility_protocol.json      frozen criteria + inputs + environment
    metadata_audit.json            F0/F1 facts and cross-checks
    difficulty_distribution.csv    official level groups (section 13)
    tuple_type_distribution.csv    section 22 groups
    negative_distribution.csv      section 24 future-abstention counts
    cohort_inventory.csv           one row per positive expression
    audit_subset.csv               deterministic image sample (seeded)
    image_dims_check.csv           released vs decoded image size
    rpn_audit.csv                  one row per audited expression
    rpn_audit_summary.json         F2 aggregate rates
    candidate_availability.csv     per-level / per-K constructibility
    external_branch_decision.json  F4 verdict + section 32 answers
    figures/*.png                  diagnostics
    metadata.json                  run record, written last
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccg.external import finecops as fc  # noqa: E402
from ccg.external import gqa_images as gi  # noqa: E402
from ccg.external import feasibility as fe  # noqa: E402

DEFAULT_OUT = Path("results/phase1e_finecops")
DEFAULT_CACHE = Path("cache/phase1e_finecops")
#: Fixed before the audit ran; the subset is reproducible from this number.
DEFAULT_SEED = 20260929
DEFAULT_AUDIT_IMAGES = 1000
STAGES = ("parse", "images", "rpn", "decision")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[phase1e] {msg}", flush=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Union-fieldname CSV with atomic replace (repo convention)."""
    rows = [dict(r) for r in rows]
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path


def _sha(path: Path) -> str:
    if not Path(path).exists():
        return ""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - bookkeeping must never break a run
        return "unknown"


def _env() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        info["torch"] = "unavailable"
    return info


def _frozen_artifacts() -> Dict[str, Any]:
    """Checksums of the RefCOCO+ artifacts the external run must reuse verbatim.

    Phase 1E evaluates FineCops with the *frozen* Stats Logistic / E1b pair
    recovered by :func:`ccg.semantic.frozen.recover_frozen_models` (instruction
    section 14).  Pinning those inputs here means a later external run can prove
    it consumed the same coefficients, hyper-parameters and train-only
    normalisation - the A8.4 recovery raises on any ``max|delta| > 1e-9``.
    """
    from ccg.semantic import frozen as sf

    candidates: List[Path] = [
        ROOT / sf.DEFAULT_PHASE05 / "stats_logistic" / "selection.json",
        ROOT / sf.DEFAULT_PHASE1 / "e1_logistic" / "selection.json",
        ROOT / sf.DEFAULT_PHASE1 / "e1_logistic" / "coefficients.csv",
        ROOT / sf.DEFAULT_PHASE05 / "split_manifest.json",
    ]
    for scorer in ("b3_seed1", "b3_seed2", "b3_seed3"):
        candidates.append(ROOT / sf.DEFAULT_PHASE05 / "features" / f"{scorer}_normalisation.json")
        candidates.append(
            ROOT / sf.DEFAULT_PHASE05 / "predictions" / scorer / "stats_logistic.csv.gz"
        )
    files: Dict[str, str] = {}
    for path in candidates:
        if path.exists():
            files[str(path.relative_to(ROOT)).replace("\\", "/")] = _sha(path)
    return {
        "recovery_module": "ccg.semantic.frozen.recover_frozen_models",
        "recovery_tolerance": sf._TOL,  # noqa: SLF001 - the A8.4 constant is the contract
        "checks": list(sf._CHECK_KEYS),  # noqa: SLF001
        "scorers": ["b3_seed1", "b3_seed2", "b3_seed3"],
        "train_ks": list(sf.TRAIN_KS),
        "files": files,
        "n_files": len(files),
    }


def _refcoco_reference() -> Dict[str, Any]:
    """The RefCOCO+ side of the domain-shift comparison, read from its own audit.

    Instruction sections 5/28 ask how the FineCops proposal audit compares to the
    one that produced every frozen RefCOCO+ number.  Those figures already exist
    on disk (same frozen RPN, same N=64, 1500 images), so they are *read*, never
    retyped - the protocol forbids hard-coding numbers it can measure.
    """
    audit = ROOT / "results" / "proposal_audit"
    out: Dict[str, Any] = {"source_dir": str(audit.relative_to(ROOT)).replace("\\", "/")}
    rows = [r for r in _read_csv(audit / "recall_by_N.csv") if r.get("N") == str(fe.N_PROPOSALS)]
    out["recall"] = dict(rows[0]) if rows else {"error": f"no N={fe.N_PROPOSALS} row"}
    availability: Dict[str, Any] = {}
    for row in _read_csv(audit / "same_category_availability.csv"):
        if row.get("N") != str(fe.N_PROPOSALS) or not row.get("threshold", "").isdigit():
            continue
        availability[f"k{int(row['threshold']) + 1}"] = dict(row)
    out["availability"] = availability or {"error": f"no N={fe.N_PROPOSALS} rows"}
    out["primary_k"] = fe.PRIMARY_K
    out["comparison_caveat"] = (
        "RefCOCO+ same-category counts come from COCO GT categories, the FineCops side from "
        "exact GQA scene-graph name equality (A9.4); the recall columns are directly "
        "comparable because both used the same frozen COCO-pretrained RPN at N=64 with "
        "target = argmax-IoU proposal."
    )
    return out


# ---------------------------------------------------------------------------
# F0/F1 - parse
# ---------------------------------------------------------------------------
def stage_parse(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    expressions = fc.load_test_expressions(
        Path(args.annotation_dir), Path(args.scene_graph), require_scene_graph=False
    )
    _log(f"positive test expressions={len(expressions)} images={len({e.image_id for e in expressions})}")

    graph = fc.load_scene_graph(Path(args.scene_graph))
    cross = fc.cross_check_coco_against_vanilla(expressions, Path(args.annotation_dir))
    difficulty = fc.difficulty_distribution(expressions)
    tuple_types = fc.tuple_type_distribution(expressions)
    negatives = fc.negative_feasibility_counts(Path(args.annotation_dir))
    geometry = fc.boxes_inside_image(expressions)
    overlaps = fc.target_pair_overlaps(expressions)

    unresolved = [e for e in expressions if e.target_name_source == "unresolved"]
    fallback = [e for e in expressions if e.target_name_source == "objects_id_first"]
    graph_missing = [e for e in expressions if not e.graph_present]
    level_agreement: Dict[str, Any] = {}
    for level in fc.LEVELS:
        sel = [e for e in expressions if e.level == level]
        if not sel:
            continue
        level_agreement[str(level)] = {
            "n": len(sel),
            "share_graph_says_no_same_name": round(
                sum(1 for e in sel if e.n_same_name <= 1) / len(sel), 6
            ),
            "share_graph_says_some_same_name": round(
                sum(1 for e in sel if e.n_same_name >= 2) / len(sel), 6
            ),
            "mean_same_name_objects": round(float(np.mean([e.n_same_name for e in sel])), 4),
        }

    subset = fc.select_audit_subset(expressions, args.audit_images, args.seed)
    by_image: Dict[int, List[fc.FineCopsExpression]] = {}
    for e in expressions:
        by_image.setdefault(e.image_id, []).append(e)
    subset_rows = [
        {
            "image_id": iid,
            "gqa_image_id": by_image[iid][0].gqa_image_id,
            "n_expressions": len(by_image[iid]),
            "levels": ",".join(str(sorted({e.level for e in by_image[iid]}))),
            "width": by_image[iid][0].width,
            "height": by_image[iid][0].height,
        }
        for iid in subset
    ]

    neg_rows: List[Dict[str, Any]] = []
    for key in ("negative_cate", "negative_type", "negative_level"):
        for value, count in (negatives.get(key) or {}).items():
            neg_rows.append({"dimension": key, "value": value, "n_expressions": count})

    metadata = {
        "stage": "F0_F1_parse",
        "n_positive_test": len(expressions),
        "n_positive_images": len({e.image_id for e in expressions}),
        "scene_graph_file": str(args.scene_graph),
        "scene_graph_images": len(graph),
        "scene_graph_coverage": round(
            sum(1 for e in expressions if e.graph_present) / max(1, len(expressions)), 6
        ),
        "n_images_without_graph": len({e.image_id for e in graph_missing}),
        "coco_vs_vanilla": cross,
        "target_name_resolution": {
            "from_graph_box": len(expressions) - len(unresolved) - len(fallback),
            "from_objects_id_fallback": len(fallback),
            "unresolved": len(unresolved),
            "iou_min_for_graph_match": fc.GRAPH_IOU_MIN,
        },
        "level_vs_scene_graph_agreement": level_agreement,
        "target_geometry": geometry,
        "same_image_targets": overlaps,
        "negative_feasibility": negatives,
        "audit_subset": {
            "n_images": len(subset),
            "n_expressions": sum(len(by_image[i]) for i in subset),
            "seed": int(args.seed),
            "rule": "numpy default_rng(seed) over the sorted unique image ids",
        },
        "seconds": round(time.time() - t0, 2),
    }

    _write_csv(
        out / "difficulty_distribution.csv",
        difficulty,
    )
    _write_csv(out / "tuple_type_distribution.csv", tuple_types)
    _write_csv(out / "negative_distribution.csv", neg_rows)
    _write_csv(out / "audit_subset.csv", subset_rows)
    inventory = [e.to_row() for e in expressions]
    for row in inventory:  # keep the committed artefact lean; text goes to the sample file
        row.pop("expression", None)
    _write_csv(out / "cohort_inventory.csv", inventory)
    samples: List[Dict[str, Any]] = []
    for level in fc.LEVELS:
        sel = [e for e in expressions if e.level == level][:5]
        samples.extend(
            {
                "level": level,
                "expr_id": e.expr_id,
                "image_id": e.image_id,
                "tuple_type": e.tuple_type,
                "target_name": e.target_name or "",
                "n_same_name": e.n_same_name,
                "expression": e.text,
            }
            for e in sel
        )
    _write_csv(out / "difficulty_examples.csv", samples)
    (out / "metadata_audit.json").parent.mkdir(parents=True, exist_ok=True)
    _dump(out / "metadata_audit.json", metadata)

    protocol = {
        "phase": "1E",
        "purpose": "FineCops-Ref external semantic confirmation - feasibility audit only (F0-F4)",
        "frozen_before_run": {
            "n_proposals": fe.N_PROPOSALS,
            "primary_k": fe.PRIMARY_K,
            "secondary_k": fe.SECONDARY_K,
            "recall_ious": list(fe.RECALL_IOUS),
            "go_recall_min": fe.GO_RECALL_MIN,
            "go_availability_min": fe.GO_AVAILABILITY_MIN,
            "stop_recall_max": fe.STOP_RECALL_MAX,
            "same_category_primary_min_availability": fe.SAME_CATEGORY_PRIMARY_MIN_AVAILABILITY,
            "same_category_diagnostic_min_availability": fe.SAME_CATEGORY_DIAGNOSTIC_MIN_AVAILABILITY,
            "min_group_for_tuple_type": fe.MIN_GROUP_FOR_TUPLE_TYPE,
            "invalid_crop_min_side": fe.INVALID_CROP_MIN_SIDE,
            "graph_iou_min": fc.GRAPH_IOU_MIN,
            "audit_seed": int(args.seed),
            "audit_images": int(args.audit_images),
        },
        "model_pipeline_reused_unchanged": [
            "torchvision fasterrcnn_resnet50_fpn RPN stage (COCO_V1, N=64)",
            "OpenCLIP ViT-B/32 laion2b_s34b_b79k",
            "B3 independent scorer seeds 1/2/3",
            "Stats Logistic, E1b Stats+Semantic Logistic (RefCOCO+ frozen)",
        ],
        "frozen_refcoco_artifacts": _frozen_artifacts(),
        "forbidden": [
            "FineCops train/val labels",
            "FineCops calibration or threshold fitting",
            "detector swap / N re-selection / reranking / MLLM CRS",
            "mixing negative (target-absent) rows into the positive gate",
        ],
        "inputs": {
            "annotation_dir": str(args.annotation_dir),
            "files": {
                p.name: {"bytes": p.stat().st_size, "sha256_16": _sha(p)}
                for p in sorted(Path(args.annotation_dir).glob("*.json"))
            },
            "scene_graph": {
                "path": str(args.scene_graph),
                "bytes": Path(args.scene_graph).stat().st_size,
                "sha256_16": _sha(Path(args.scene_graph)),
            },
        },
        "environment": _env(),
    }
    _dump(out / "feasibility_protocol.json", protocol)
    _log(f"parse done in {metadata['seconds']}s -> {out / 'metadata_audit.json'}")
    return metadata


def _dump(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=float),
                   encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# F0b - images
# ---------------------------------------------------------------------------
def stage_images(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    subset_rows = _read_csv(out / "audit_subset.csv")
    if not subset_rows:
        raise FileNotFoundError("run the parse stage first (audit_subset.csv missing)")
    ids = [r["gqa_image_id"] for r in subset_rows]
    t0 = time.time()
    report = gi.fetch_images(ids, Path(args.image_dir), workers=args.workers, log=_log)
    _log(
        f"images fetched={report.fetched} skipped={report.skipped_existing} "
        f"failed={len(report.failed)} missing={len(report.missing_in_archive)} "
        f"bytes={report.bytes_written/1e6:.1f}MB in {time.time()-t0:.1f}s"
    )
    expressions = fc.load_test_expressions(
        Path(args.annotation_dir), Path(args.scene_graph), require_scene_graph=False
    )
    dims = gi.local_dims(Path(args.image_dir), ids)
    rows = fc.image_dim_rows(expressions, dims)
    _write_csv(out / "image_dims_check.csv", rows)
    n = len(rows)
    mismatch_ann = sum(1 for r in rows if not r["ann_matches_actual"])
    mismatch_graph = sum(1 for r in rows if not r["ann_matches_graph"])
    payload = {
        "stage": "F0b_images",
        "fetch": report.to_dict(),
        "images_decoded": n,
        "annotation_vs_actual_mismatch": mismatch_ann,
        "annotation_vs_graph_mismatch": mismatch_graph,
        "dims_source_of_truth": "actual JPEG",
        "verdict": (
            "IMAGE GEOMETRY OK"
            if n and mismatch_ann == 0
            else ("NO IMAGES" if n == 0 else "IMAGE GEOMETRY MISMATCH")
        ),
        "seconds": round(time.time() - t0, 2),
    }
    _dump(out / "image_fetch_report.json", payload)
    _log(f"images stage done: {payload['verdict']} ({n} decoded, {mismatch_ann} mismatched)")
    return payload


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _rate(flags: Sequence[bool]) -> float:
    return float(sum(1 for f in flags if f) / len(flags)) if flags else 0.0


# ---------------------------------------------------------------------------
# F2 - frozen RPN audit
# ---------------------------------------------------------------------------
# Report-only stratification used to interpret a recall gap (instruction section 32
# item 10): is the missing target a small-object problem or a general perceptual
# gap?  It changes nothing about the frozen proposals.
_SIZE_EDGES = (32.0, 64.0, 128.0, 256.0)
_SIZE_LABELS = ("<32px", "<64px", "<128px", "<256px", ">=256px")


def _size_bucket(side: float) -> str:
    for edge, label in zip(_SIZE_EDGES, _SIZE_LABELS[:-1]):
        if side < edge:
            return label
    return _SIZE_LABELS[-1]


def _by_target_size(
    rows: Sequence[fe.ProposalAuditRow], area_by_expr: Mapping[int, Tuple[float, float]]
) -> List[Dict[str, Any]]:
    """Recall / availability per equivalent-square-side bucket of the target box."""
    out: List[Dict[str, Any]] = []
    for bucket in _SIZE_LABELS:
        sub = [r for r in rows if _size_bucket(area_by_expr.get(r.expr_id, (0.0, 0.0))[0]) == bucket]
        if not sub:
            continue
        n = len(sub)
        out.append(
            {
                "target_side_bucket": bucket,
                "n_rows": n,
                "share_of_rows": round(n / float(len(rows)), 6),
                "mean_target_side_px": round(
                    float(np.mean([area_by_expr[r.expr_id][0] for r in sub])), 2
                ),
                "mean_rel_area": round(
                    float(np.mean([area_by_expr[r.expr_id][1] for r in sub])), 6
                ),
                "recall_at_05": round(_rate([r.target_present for r in sub]), 6),
                "recall_at_07": round(_rate([r.target_best_iou >= 0.7 for r in sub]), 6),
                "k5_availability": round(
                    _rate([r.available.get(fe.PRIMARY_K, False) for r in sub]), 6
                ),
                "k5_same_name_availability": round(
                    _rate([r.same_name_available.get(fe.PRIMARY_K, False) for r in sub]), 6
                ),
                "mean_best_iou": round(float(np.mean([r.target_best_iou for r in sub])), 6),
            }
        )
    return out


def stage_rpn(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from PIL import Image

    from ccg.data.proposals import read_bank, write_bank
    from ccg.data.rpn import build_rpn_model, extract_proposal_bank

    t0 = time.time()
    subset = {int(r["image_id"]) for r in _read_csv(out / "audit_subset.csv")}
    if not subset:
        raise FileNotFoundError("run the parse stage first (audit_subset.csv missing)")
    expressions = [
        e
        for e in fc.load_test_expressions(
            Path(args.annotation_dir), Path(args.scene_graph), require_scene_graph=False
        )
        if e.image_id in subset
    ]
    graph = fc.load_scene_graph(Path(args.scene_graph))
    names_by_image = {
        e.gqa_image_id: sorted(
            {str(o.get("name")) for o in graph.get(e.gqa_image_id, {}).get("objects", {}).values()}
        )
        for e in expressions
    }
    codes = fe.build_name_code_table(expressions, names_by_image)

    bank_path = Path(args.cache_dir) / "proposals_gqa.h5"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    cached: Dict[int, Any] = {}
    if bank_path.exists():
        from ccg.data.proposals import list_image_ids

        cached = {i: None for i in list_image_ids(bank_path)}
    device = args.device
    model = None
    if len(cached) < len(subset):
        torch.manual_seed(0)
        model = build_rpn_model(device=device)
        _log(f"RPN loaded on {device} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    rows: List[fe.ProposalAuditRow] = []
    area_by_expr: Dict[int, Tuple[float, float]] = {}
    n_extracted = 0
    skipped_images: List[str] = []
    dim_violations: List[str] = []
    image_ids = sorted({e.image_id for e in expressions})
    for pos, iid in enumerate(image_ids, start=1):
        gqa_id = str(iid)
        path = gi.local_image_path(Path(args.image_dir), gqa_id)
        if not path.exists():
            skipped_images.append(gqa_id)
            continue
        # the JPEG is always decoded: it is the source of truth for the released boxes
        dims = gi.decode_dims(path)
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        if dims != (image.shape[1], image.shape[0]):  # pragma: no cover - PIL is consistent
            dim_violations.append(gqa_id)
        exprs = [e for e in expressions if e.image_id == iid]
        for e in exprs:
            if (e.width, e.height) != (image.shape[1], image.shape[0]):
                dim_violations.append(f"{gqa_id}:expr{e.expr_id}")
        objects = graph.get(gqa_id, {}).get("objects", {}) or {}
        gt_boxes, gt_codes, _ids = fe.graph_object_boxes(objects, codes)
        if iid in cached:
            bank = read_bank(bank_path, iid)
        else:
            bank, _meta = extract_proposal_bank(image, model, int(iid), top_n=fe.N_PROPOSALS)
            write_bank(bank_path, bank)
            cached[iid] = None
            n_extracted += 1
        for e in exprs:
            area = float(max(0.0, (e.gt_box[2] - e.gt_box[0])) * max(0.0, (e.gt_box[3] - e.gt_box[1])))
            rel = area / float(max(1, e.width * e.height))
            area_by_expr[e.expr_id] = (float(np.sqrt(area)), rel)
            rows.append(
                fe.audit_expression(
                    e,
                    bank,
                    gt_object_boxes_xyxy=gt_boxes,
                    gt_object_name_codes=gt_codes,
                )
            )
        if pos % 50 == 0 or pos == len(image_ids):
            mem = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
            _log(
                f"[rpn] {pos}/{len(image_ids)} images, {len(rows)} expressions, "
                f"extracted={n_extracted}, gpu_peak={mem:.0f}MB"
            )

    summary = fe.summarise_rows(rows)
    summary.update(
        {
            "n_images_requested": len(subset),
            "n_images_audited": len({r.image_id for r in rows}),
            "n_images_skipped_missing_file": len(skipped_images),
            "n_banks_extracted_this_run": n_extracted,
            "n_banks_reused_from_cache": len({r.image_id for r in rows}) - n_extracted,
            "n_distinct_target_names": len({r.target_name for r in rows if r.target_name}),
            "n_dimension_violations": len(dim_violations),
            "dimension_violations_sample": dim_violations[:20],
            "geometry_verdict": (
                "RELEASED BOXES IN JPEG PIXEL SPACE" if not dim_violations else "PIXEL SPACE MISMATCH"
            ),
            "seconds": round(time.time() - t0, 2),
        }
    )
    summary["by_target_size"] = _by_target_size(rows, area_by_expr)
    _write_csv(out / "rpn_audit.csv", fe.audit_cohort(rows))
    _write_csv(out / "recall_by_target_size.csv", summary["by_target_size"])
    _dump(out / "rpn_audit_summary.json", summary)
    _log(
        f"rpn stage done in {summary['seconds']}s: recall@0.5="
        f"{summary.get('recall_at_05', {}).get('rate')} "
        f"K5_avail={summary.get('k5_availability', {}).get('rate')} "
        f"same_name_K5={summary.get('k5_availability', {}).get('same_name_rate')}"
    )
    return summary


# ---------------------------------------------------------------------------
# F3/F4 - decision
# ---------------------------------------------------------------------------
def stage_decision(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    summary_path = out / "rpn_audit_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run the rpn stage first (rpn_audit_summary.json missing)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads((out / "metadata_audit.json").read_text(encoding="utf-8"))
    verdict = fe.engineering_verdict(summary)
    branch = fe.candidate_regime_decision(summary)

    availability_rows: List[Dict[str, Any]] = []
    for level, stats in sorted((summary.get("by_level") or {}).items()):
        availability_rows.append({"grouping": "level", "group": level, **_flat(stats)})
    for k, stats in sorted((summary.get("by_tuple_type") or {}).items()):
        availability_rows.append({"grouping": "tuple_type", "group": k, **_flat(stats)})
    for key in ("recall_at_05", "recall_at_07", "k5_availability", "k10_availability"):
        val = summary.get(key)
        availability_rows.append(
            {
                "grouping": "overall",
                "group": key,
                "rate": val.get("rate") if isinstance(val, dict) else val,
                "wilson_low": val.get("wilson_low") if isinstance(val, dict) else "",
                "wilson_high": val.get("wilson_high") if isinstance(val, dict) else "",
            }
        )
    for key, val in sorted((summary.get("same_name_distractors") or {}).items()):
        availability_rows.append({"grouping": "same_name_supply", "group": key, "rate": val})
    _write_csv(out / "candidate_availability.csv", availability_rows)

    domain_shift = {
        "note": "B3 accuracy / MSP / margin / entropy on FineCops are measured in F7/F8, "
                "not in the feasibility round; recorded here as pending",
        "severe_domain_shift_threshold_b3_accuracy": 0.30,
        "status": "PENDING_MODEL_INFERENCE",
    }

    payload = {
        "stage": "F3_F4_decision",
        "engineering_gate": verdict,
        "candidate_regime": branch,
        "b3_input_pipeline_compatible": {
            "criterion": "N=64 frozen RPN produces >=64 proposals and >=90% of targets are proposable",
            "measured_min_bank_size": summary.get("min_bank_size"),
            "measured_mean_bank_size": summary.get("mean_bank_size"),
            "recall_at_05": (summary.get("recall_at_05") or {}).get("rate"),
        },
        "metadata_facts": {
            "n_positive_test": metadata.get("n_positive_test"),
            "n_positive_images": metadata.get("n_positive_images"),
            "scene_graph_coverage": metadata.get("scene_graph_coverage"),
            "scene_graph_images": metadata.get("scene_graph_images"),
            "coco_vs_vanilla": metadata.get("coco_vs_vanilla"),
            "target_name_resolution": metadata.get("target_name_resolution"),
            "level_vs_scene_graph_agreement": metadata.get("level_vs_scene_graph_agreement"),
            "target_geometry": metadata.get("target_geometry"),
            "same_image_targets": metadata.get("same_image_targets"),
            "negative_feasibility_counts": {
                k: v
                for k, v in (metadata.get("negative_feasibility") or {}).items()
                if k.startswith("n_")
            },
            "difficulty_distribution": {
                str(r["level"]): r for r in _read_csv(out / "difficulty_distribution.csv")
            },
            "rpn_audit": {
                k: summary.get(k)
                for k in (
                    "n_rows",
                    "n_images_requested",
                    "n_images_audited",
                    "recall_at_05",
                    "recall_at_07",
                    "natural_omission_at_05",
                    "k5_availability",
                    "k10_availability",
                    "same_name_distractors",
                    "invalid_crop_rate",
                    "redundancy_frac_pairs_gt_07",
                    "geometry_verdict",
                )
            },
        },
        "domain_shift": domain_shift,
        "refcoco_plus_reference": _refcoco_reference(),
        "answers": {
            "q_is_finecops_suitable": None,  # filled below
            "q_primary_regime": branch["primary_hard_regime"],
        },
        "audit_subset": metadata.get("audit_subset"),
        "environment": _env(),
    }
    go = verdict["verdict"] == "EXTERNAL GO"
    payload["answers"]["q_is_finecops_suitable"] = (
        "YES - frozen pipeline applies unchanged"
        if go
        else f"NO / CONDITIONAL - engineering gate says {verdict['verdict']}"
    )
    _dump(out / "external_branch_decision.json", payload)
    _log(f"decision: engineering={verdict['verdict']} branch={branch['branch']}")
    return payload


def _flat(stats: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# figures + metadata
# ---------------------------------------------------------------------------
def write_figures(out: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        _log(f"figures skipped ({type(exc).__name__}: {exc})")
        return []
    rows = _read_csv(out / "rpn_audit.csv")
    if not rows:
        return []
    summary = json.loads((out / "rpn_audit_summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ious = np.asarray([float(r["target_best_iou"]) for r in rows], dtype=float)
    axes[0].hist(ious, bins=40, range=(0.0, 1.0), color="#3b6ea5")
    axes[0].axvline(0.5, ls="--", c="k", lw=1)
    axes[0].axvline(0.7, ls=":", c="k", lw=1)
    axes[0].set_title(
        "FineCops target max-IoU (N=64)\n"
        f"recall@0.5={summary.get('recall_at_05', {}).get('rate', float('nan')):.3f}  "
        f"@0.7={summary.get('recall_at_07', {}).get('rate', float('nan')):.3f}"
    )
    axes[0].set_xlabel("IoU(target box, best proposal)")
    axes[0].set_ylabel("expressions")

    same = np.asarray([int(r["n_same_name_distractors"]) for r in rows], dtype=float)
    for level, colour in ((1, "#c0504d"), (2, "#e8a33d"), (3, "#4f8a4f")):
        sel = same[[i for i, r in enumerate(rows) if int(r["level"]) == level]]
        if sel.size:
            axes[1].hist(
                sel, bins=np.arange(-0.5, 12.5, 1), alpha=0.6, label=f"level {level}", color=colour
            )
    axes[1].axvline(fe.PRIMARY_K - 1, ls="--", c="k", lw=1)
    axes[1].set_title("same-name distractor supply (K=5 needs >=4)")
    axes[1].set_xlabel("# proposals matching another object of the target's name")
    axes[1].legend(fontsize=8)

    labels = ["recall@0.5", "K5 random", "K5 same-name"]
    vals = [
        summary.get("recall_at_05", {}).get("rate", 0.0),
        summary.get("k5_availability", {}).get("rate", 0.0),
        summary.get("k5_availability", {}).get("same_name_rate", 0.0),
    ]
    bars = axes[2].bar(labels, vals, color=["#3b6ea5", "#4f8a4f", "#c0504d"])
    axes[2].axhline(fe.GO_AVAILABILITY_MIN, ls="--", c="k", lw=1, label="GO threshold 0.90")
    axes[2].axhline(fe.STOP_RECALL_MAX, ls=":", c="k", lw=1, label="STOP threshold 0.80")
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[2].set_ylim(0, 1.12)
    axes[2].set_title("frozen-pipeline constructibility on FineCops")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    path = out / "figures" / "fig1_finecops_feasibility.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    _log(f"figure -> {path}")
    return [str(path)]


STAGE_FUNCS = {
    "parse": stage_parse,
    "images": stage_images,
    "rpn": stage_rpn,
    "decision": stage_decision,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list of {STAGES}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--annotation-dir", type=Path, default=fc.ANNOTATION_DIR)
    ap.add_argument("--scene-graph", type=Path, default=fc.SCENE_GRAPH_PATH)
    ap.add_argument("--image-dir", type=Path, default=fc.IMAGE_DIR)
    ap.add_argument("--audit-images", type=int, default=DEFAULT_AUDIT_IMAGES)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--workers", type=int, default=gi.DEFAULT_WORKERS,
                    help="parallel range streams (the GQA host 503s under load)")
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGE_FUNCS:
            raise SystemExit(f"unknown stage {s!r}; expected one of {STAGES}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}
    for s in stages:
        _log(f"=== stage {s} ===")
        results[s] = STAGE_FUNCS[s](out, args)
    if "decision" in stages:
        figures = write_figures(out)
        meta = {
            "phase": "1E_feasibility",
            "stages_run": stages,
            "seed": int(args.seed),
            "audit_images": int(args.audit_images),
            "git_commit": _git_commit(),
            "environment": _env(),
            "artifacts": sorted(
                str(p.resolve().relative_to(ROOT.resolve())) for p in out.rglob("*") if p.is_file()
            ),
            "figures": figures,
            "verdict": (results.get("decision") or {}).get("engineering_gate", {}).get("verdict"),
            "branch": (results.get("decision") or {}).get("candidate_regime", {}).get("branch"),
            "total_wallclock_s": None,
        }
        _dump(out / "metadata.json", meta)
        print(
            "PHASE1E_FEASIBILITY_COMPLETE "
            f"verdict={meta['verdict']} branch={meta['branch']} out={out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
