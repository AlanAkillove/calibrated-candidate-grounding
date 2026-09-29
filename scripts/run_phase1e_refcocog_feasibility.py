"""Phase 1E / Amendment A10 feasibility driver: RefCOCOg (UMD) as an external set.

Pre-registered stage order (instruction section 5)::

    G0  parse      - refs(umd).p, integrity facts, hashes, target-box source control
    G1  overlap    - image-overlap matrix against every RefCOCO+ development set
    G2  subsets    - rg_external_strict / rg_external_devdisjoint + the size gate
    --  images     - fetch exactly the COCO train2014 JPEGs the audit needs
    G3  rpn        - frozen N=64 class-agnostic RPN proposal audit
    G4  candidates - COCO same-category supply, hard cohort, distributions
    G5  decision   - branch A / B / C + figures

This round *stops* at G5 (section 29).  Nothing here runs B3, CLIP, Stats
Logistic or E1b: the only model that executes is the frozen RPN, which is a
property of the candidate generator, not of the reliability model under test.

Reuse is the protocol (sections 10/13)
--------------------------------------
The proposal bank comes from :func:`ccg.data.rpn.build_rpn_model` /
:func:`ccg.data.rpn.extract_proposal_bank` at ``top_n = N_PROPOSALS = 64``; the
images RefCOCO+ already exposed are read out of the *frozen*
``cache/proposals.h5`` instead of being re-extracted, so the external rows and
the development rows share one generator byte-for-byte where they can.  Every
measurement is :func:`ccg.external.feasibility.audit_expression` - the same
target assignment, the same equivalent-proposal removal, the same
``assign_gt_category`` / ``same_category_counts`` pair that produced the A8
RefCOCO+ cohort - and the RefCOCO+ comparison numbers are read from
``results/proposal_audit``, never retyped.

Outputs (``results/phase1e_refcocog_feasibility/``)::

    feasibility_protocol.json   frozen criteria + inputs + hashes + environment
    dataset_summary.json        G0 counts and integrity cross-checks
    image_overlap.csv           G1 overlap matrix (section 7)
    external_subsets.json       G2 subset sizes + section 9 gate
    audit_expressions.csv       the expression-level audit input, deterministically
    expression_distribution.csv section 18 language shift, both datasets
    image_fetch_report.json     which JPEGs were fetched / already present
    image_dims_check.csv        archive vs official COCO vs decoded JPEG
    rpn_audit.csv               one row per audited expression
    rpn_summary.json            G3 rates, per subset, + RefCOCO+ reference
    candidate_availability.csv  section 14 same-category coverage
    category_distribution.csv   section 19 target-category shift
    hard_cohort.csv             section 17 matched random / same-category cohort
    hard_cohort_summary.json    section 16 power gate
    branch_decision.json        G5 verdict + section 30 answers
    figures/*.png               diagnostics
    metadata.json               run record, written last
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ccg.data.splits import SPLIT_TEST_A, SPLIT_TEST_B  # noqa: E402
from ccg.external import coco_images as ci  # noqa: E402
from ccg.external import feasibility as fe  # noqa: E402
from ccg.external import refcocog as rg  # noqa: E402

DEFAULT_OUT = Path("results/phase1e_refcocog_feasibility")
DEFAULT_CACHE = Path("cache/phase1e_refcocog")
DEFAULT_IMAGE_ROOT = Path("data/raw/refcocog/images")
#: The frozen RefCOCO+ proposal bank - read-only input, never written by this run.
FROZEN_BANK = Path("cache/proposals.h5")
REFCOCOG_DIR = Path("data/raw/refcocog/refcocog")
REFCOCO_PLUS_DIR = Path("data/raw/refcoco+/refcoco+")
COCO_GT = Path("data/raw/annotations/instances_train2014.json")
MANIFEST_DIR = Path("cache/manifests")
PROPOSAL_AUDIT_DIR = Path("results/proposal_audit")
HARD_MANIFEST = Path("results/phase1f_hard_semantic/manifests/hard5_candidates.npz")
STAGES = ("parse", "overlap", "subsets", "images", "rpn", "candidates", "decision")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[phase1e-refcocog] {msg}", flush=True)


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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _dump(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=float),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def _sha(path: Path) -> str:
    if not Path(path).exists():
        return ""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha16(path: Path) -> str:
    return _sha(path)[:16]


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
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        info["torchvision"] = "unavailable"
    return info


def _rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _frozen_inputs() -> Dict[str, Any]:
    """The RefCOCO+ artifacts this round must reuse unchanged (hash-pinned)."""
    from ccg.semantic import frozen as sf

    candidates: List[Path] = [
        ROOT / FROZEN_BANK,
        ROOT / PROPOSAL_AUDIT_DIR / "recall_by_N.csv",
        ROOT / PROPOSAL_AUDIT_DIR / "same_category_availability.csv",
        ROOT / MANIFEST_DIR / "random_testA.jsonl",
        ROOT / MANIFEST_DIR / "random_testB.jsonl",
        ROOT / MANIFEST_DIR / "random_train.jsonl",
        ROOT / MANIFEST_DIR / "random_val_select.jsonl",
        ROOT / MANIFEST_DIR / "random_val_calib.jsonl",
        ROOT / HARD_MANIFEST,
        ROOT / sf.DEFAULT_PHASE05 / "stats_logistic" / "selection.json",
        ROOT / sf.DEFAULT_PHASE1 / "e1_logistic" / "coefficients.csv",
        ROOT / sf.DEFAULT_PHASE05 / "split_manifest.json",
    ]
    files: Dict[str, str] = {}
    for path in candidates:
        if path.exists():
            files[_rel(path)] = _sha16(path)
    return {
        "note": (
            "read-only inputs of the feasibility round; A10 forwards no reliability "
            "model, so these hashes exist to prove that a later frozen external run "
            "consumes the identical artifacts (instruction sections 21/23)"
        ),
        "model_pipeline_reused_unchanged": [
            "torchvision fasterrcnn_resnet50_fpn RPN stage (COCO_V1, class-agnostic, N=64)",
            "ccg.data.proposals.assign_target (IoU >= 0.5, argmax target, equivalent removal)",
            "ccg.data.audit.assign_gt_category + same_category_counts (A8 rule)",
            "ccg.external.feasibility.audit_expression / summarise_rows",
        ],
        "files": files,
        "n_files": len(files),
    }


def _refcoco_reference() -> Dict[str, Any]:
    """RefCOCO+ frozen numbers, loaded from disk (section 12 forbids retyping).

    ``results/proposal_audit/recall_by_N.csv`` names its two recall families
    differently: ``ref_target_recall@*`` is the *referred target* recall (the
    quantity A10 gates on) while ``gt_object_recall@*`` is recall over every COCO
    GT object.  The mapping is done here, once, so the comparison columns cannot
    be mixed up.
    """
    audit = ROOT / PROPOSAL_AUDIT_DIR
    out: Dict[str, Any] = {"source_dir": _rel(audit)}
    rows = [r for r in _read_csv(audit / "recall_by_N.csv") if r.get("N") == str(fe.N_PROPOSALS)]
    row = dict(rows[0]) if rows else {"error": f"no N={fe.N_PROPOSALS} row"}
    out["recall"] = row
    out["target_recall_at_05"] = _num(row.get("ref_target_recall@0.5"))
    out["target_recall_at_07"] = _num(row.get("ref_target_recall@0.7"))
    out["gt_object_recall_at_05"] = _num(row.get("gt_object_recall@0.5"))
    out["n_ref_targets"] = _num(row.get("num_ref_targets"))
    availability: Dict[str, Any] = {}
    for r in _read_csv(audit / "same_category_availability.csv"):
        if r.get("N") != str(fe.N_PROPOSALS) or not _is_distractor_row(r):
            continue
        availability[f"at_least_{int(r['threshold'])}"] = dict(r)
    out["same_category_availability"] = availability or {"error": f"no N={fe.N_PROPOSALS} rows"}
    #: K = 5 needs four distractors, so the ``threshold = 4`` row *is* the RefCOCO+
    #: same-category K5 availability the A10 bar is compared against.
    out["same_category_k5_availability"] = _num(
        (availability.get("at_least_4") or {}).get("frac")
    )
    cohort_path = ROOT / "results" / "phase1f_hard_semantic" / "cohort_summary.json"
    if cohort_path.exists():
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        out["phase1f_hard_cohort"] = {
            "source": _rel(cohort_path),
            "same4": cohort.get("cohorts", {}).get("same4"),
            "base": cohort.get("cohorts", {}).get("base"),
        }
    out["primary_k"] = fe.PRIMARY_K
    out["comparison_caveat"] = (
        "RefCOCO+ recall and same-category numbers were produced by the same frozen "
        "RPN at N=64 with the same target assignment and the same COCO-GT category "
        "rule, so the columns are directly comparable; the RefCOCO+ figures cover its "
        "own 1500-image audit subset while A10 audits every surviving external image."
    )
    return out


def _is_distractor_row(row: Mapping[str, str]) -> bool:
    return bool(row.get("threshold", "").isdigit())


def _num(value: Any) -> Optional[float]:
    """CSV string -> finite float, or ``None`` (never a JSON ``NaN``)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ---------------------------------------------------------------------------
# shared loaders
# ---------------------------------------------------------------------------
def load_regions(annotation_dir: Path) -> List[rg.RefCOCOGRegion]:
    """Parse ``refs(umd).p`` (the only form the UMD split exists in)."""
    pickle_path = Path(annotation_dir) / "refs(umd).p"
    return rg.parse_regions(rg.load_refs_umd(pickle_path))


def load_archive_instances(annotation_dir: Path) -> Dict[str, Any]:
    """The archive's own ``instances.json`` (boxes / sizes / category names)."""
    path = Path(annotation_dir) / "instances.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def archive_join(instances: Mapping[str, Any]) -> Tuple[
    Dict[int, List[float]], Dict[int, Tuple[int, int]], Dict[int, str]
]:
    """``ann_id -> xywh box``, ``image_id -> (w, h)``, ``category_id -> name``."""
    boxes = {int(a["id"]): [float(v) for v in a["bbox"]] for a in instances["annotations"]}
    meta = {int(i["id"]): (int(i["width"]), int(i["height"])) for i in instances["images"]}
    names = {int(c["id"]): str(c.get("name", "")) for c in instances.get("categories", [])}
    return boxes, meta, names


def official_gt(image_ids: Iterable[int], gt_file: Path):
    """Official COCO index restricted to ``image_ids`` (A8's GT source)."""
    from ccg.data.coco import load_coco_index

    return load_coco_index([Path(gt_file)], only_image_ids=list(image_ids))


def load_refcoco_plus_regions(path: Path) -> List[dict]:
    from ccg.data.refcoco import load_refs_pickle

    return load_refs_pickle(Path(path))


# ---------------------------------------------------------------------------
# G0 - parse
# ---------------------------------------------------------------------------
def stage_parse(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    adir = Path(args.annotation_dir)
    regions = load_regions(adir)
    instances = load_archive_instances(adir)
    boxes_by_ann, meta_by_image, cat_names = archive_join(instances)

    n_unmatched = sum(1 for r in regions if r.ann_id not in boxes_by_ann)
    provenance = _provenance(adir)
    summary = rg.summarise_dataset(
        regions,
        instances_member=instances,
        provenance=provenance,
        boxes_by_ann=boxes_by_ann,
        n_unmatched_ann=n_unmatched,
    )

    # --- section 11 debug evidence: which box *is* the target? ---------------
    test_images = sorted({r.image_id for r in regions if r.split == rg.UMD_EVAL_SPLIT})
    off = official_gt(test_images, Path(args.coco_gt))
    official_by_ann = _official_ann_map(off)
    pairs_g, off_cat_mismatch, off_img_missing = [], 0, 0
    for reg in regions:
        if reg.split != rg.UMD_EVAL_SPLIT:
            continue
        official = official_by_ann.get(int(reg.ann_id))
        if official is None:
            off_img_missing += 1
            continue
        archive_box = boxes_by_ann.get(reg.ann_id)
        if archive_box is None:
            continue
        pairs_g.append((archive_box, official["bbox"]))
        if int(official["category_id"]) != int(reg.category_id):
            off_cat_mismatch += 1
    summary["target_box_source"] = rg.target_box_agreement(pairs_g)
    summary["target_box_source"].update(
        {
            "official_gt_file": _rel(args.coco_gt),
            "n_refs_missing_in_official_coco": off_img_missing,
            "n_category_mismatch_vs_official_coco": off_cat_mismatch,
            "precedent": (
                "scripts/run_proposal_audit.py: RefCOCO+ joins the *target* box from its "
                "own instances.json and takes the GT *object set* (for category "
                "assignment) from the official instances_train2014.json; A10 does the same"
            ),
        }
    )

    # --- file-name / id canonicalisation check -------------------------------
    canon_bad, dims_mismatch = _canonicalisation_check(regions, off, meta_by_image)
    summary["image_identity"] = {
        "n_file_name_mismatch_vs_official_coco": canon_bad,
        "n_dims_mismatch_vs_official_coco": dims_mismatch,
        "rule": rg.canonical_image_file_name.__doc__.splitlines()[0].strip(),
        "coco_folders": sorted({str(Path(r.file_name).name).split("_")[1] for r in regions}),
    }

    _dump(out / "dataset_summary.json", summary)
    _log(
        f"G0 refs={summary['records']['n_refs_total']} "
        f"expressions={summary['records']['n_expressions_total']} "
        f"images={summary['records']['n_images_total']} "
        f"image_level_split={summary['records']['image_level_split']}"
    )

    protocol = {
        "phase": "1E",
        "amendment": "A10",
        "purpose": (
            "RefCOCOg (UMD split) image-disjoint external confirmation - feasibility only "
            "(G0-G5); cross-dataset replication under a SHARED COCO visual domain, not "
            "cross-domain visual generalisation (instruction section 24)"
        ),
        "predecessor": {
            "amendment": "A9",
            "dataset": "FineCops-Ref",
            "decision": "EXTERNAL STOP",
            "frozen_reason": {
                "target_proposal_recall_at_05": 0.7579,
                "stop_threshold": 0.80,
                "same_name_k5_availability": 0.1861,
            },
            "note": (
                "A9's stop stands unchanged; A10 does not lower, reinterpret or continue "
                "it (instruction preamble).  RefCOCOg was chosen because it removes the "
                "GQA perception shift that caused it while keeping a language / annotation "
                "distribution shift to test"
            ),
        },
        "frozen_before_run": {
            "n_proposals": fe.N_PROPOSALS,
            "primary_k": fe.PRIMARY_K,
            "secondary_k": fe.SECONDARY_K,
            "recall_ious": list(fe.RECALL_IOUS),
            "target_iou_thresh": 0.5,
            "strict_min_expressions": rg.STRICT_MIN_EXPRESSIONS,
            "strict_min_images": rg.STRICT_MIN_IMAGES,
            "dev_min_expressions": rg.DEV_MIN_EXPRESSIONS,
            "dev_min_images": rg.DEV_MIN_IMAGES,
            "go_recall_min": rg.A10_GO_RECALL_MIN,
            "go_random_k5_min": rg.A10_GO_RANDOM_K5_MIN,
            "gray_recall_min": rg.A10_GRAY_RECALL_MIN,
            "same_category_primary_min": rg.A10_SAMECAT_PRIMARY_MIN,
            "same_category_good_min": rg.A10_SAMECAT_GOOD_MIN,
            "same_category_stop_max": rg.A10_SAMECAT_STOP_MAX,
            "hard_min_expressions": rg.A10_HARD_MIN_EXPRESSIONS,
            "hard_min_images": rg.A10_HARD_MIN_IMAGES,
            "distractor_coverage_levels": list(rg.DISTRACTOR_COVERAGE_LEVELS),
            "umd_eval_split": rg.UMD_EVAL_SPLIT,
        },
        "forbidden": [
            "RefCOCOg train / val annotations (0 training parameters this round)",
            "any B3 / CLIP / Stats Logistic / E1b forward pass or re-calibration",
            "detector swap / N re-selection / reranking / threshold re-tuning",
            "weakening or revisiting the A9 FineCops stop",
            "describing the result as cross-visual-domain generalisation",
        ],
        "inputs": {
            "annotation_dir": _rel(args.annotation_dir),
            "files": {
                _rel(p): {"bytes": p.stat().st_size, "sha256_16": _sha16(p)}
                for p in [
                    Path(args.annotation_dir) / "refs(umd).p",
                    Path(args.annotation_dir) / "instances.json",
                    Path(args.coco_gt),
                ]
                if p.exists()
            },
            "dataset_card": provenance.get("dataset_card"),
        },
        "frozen_refcoco_artifacts": _frozen_inputs(),
        "environment": _env(),
    }
    _dump(out / "feasibility_protocol.json", protocol)
    summary["seconds"] = round(time.time() - t0, 2)
    return summary


def _provenance(adir: Path) -> Dict[str, Any]:
    """Where the annotation came from, straight from the download's dataset card."""
    card = Path(adir).parent / "dataset_card.json"
    out: Dict[str, Any] = {"dataset_card": _rel(card) if card.exists() else None}
    files: Dict[str, Any] = {}
    for name in ("refs(umd).p", "refs(google).p", "instances.json"):
        p = Path(adir) / name
        if p.exists():
            files[name] = {"bytes": p.stat().st_size, "sha256": _sha(p)}
    out["files"] = files
    if card.exists():
        try:
            payload = json.loads(card.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken card must not block G0
            payload = {}
        for key in ("sources", "urls", "archive", "official_url", "snapshot", "retrieved_utc"):
            if key in payload:
                out[key] = payload[key]
    return out


def _official_ann_map(index) -> Dict[int, Dict[str, Any]]:
    """``ann_id -> {image_id, bbox xywh, category_id}`` over an official COCO index.

    Built once per image set: the RefCOCOg / official-COCO join questions (does our
    target box equal the official box, does our category equal the official one) are
    per-annotation lookups, and a linear scan per region would turn a two-second
    check into an hour.
    """
    from ccg.data.proposals import xyxy_to_xywh

    out: Dict[int, Dict[str, Any]] = {}
    for image_id in index.image_ids:
        gt = index.objects(int(image_id))
        if not len(gt):
            continue
        xywh = xyxy_to_xywh(gt.boxes)
        for row, obj_id in enumerate(gt.object_ids):
            out[int(obj_id)] = {
                "image_id": int(image_id),
                "bbox": [float(v) for v in xywh[row]],
                "category_id": int(gt.categories[row]),
                "iscrowd": bool(gt.crowd_flags[row]),
            }
    return out


def _canonicalisation_check(
    regions: Sequence[rg.RefCOCOGRegion],
    index,
    meta_by_image: Mapping[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """RefCOCOg ``file_name``/size vs the official COCO record, per region."""
    canon_bad = dims_bad = 0
    for reg in regions:
        if reg.image_id not in index:
            continue
        try:
            canon = rg.canonical_image_file_name(reg.file_name, reg.image_id)
        except ValueError:
            canon_bad += 1
            continue
        if canon != index.file_name(reg.image_id):
            canon_bad += 1
        meta = index.image(reg.image_id)
        archive = meta_by_image.get(int(reg.image_id))
        if archive is not None and archive != (meta.width, meta.height):
            dims_bad += 1
    return canon_bad, dims_bad


# ---------------------------------------------------------------------------
# G1 - overlap
# ---------------------------------------------------------------------------
def stage_overlap(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    regions = load_regions(Path(args.annotation_dir))
    sets = rg.load_development_image_sets(Path(args.manifest_dir))
    by_split = rg.image_ids_by_split(regions)

    expr_by_image: Dict[int, int] = {}
    refs_by_image: Dict[int, int] = {}
    for reg in regions:
        refs_by_image[reg.image_id] = refs_by_image.get(reg.image_id, 0) + 1
        expr_by_image[reg.image_id] = expr_by_image.get(reg.image_id, 0) + reg.n_sentences
    rows = rg.overlap_rows(by_split, sets, expr_by_image, refs_by_image)
    _write_csv(out / "image_overlap.csv", rows)

    payload = {
        "stage": "G1_image_overlap",
        "development_sets": {
            name: len(ids) for name, ids in sorted(sets.items()) if name in {
                SPLIT_TEST_A, SPLIT_TEST_B, "train", "val_select", "val_calib",
                "development", "all_refcoco_plus",
            }
        },
        "development_sets_source": _rel(args.manifest_dir),
        "refcocog_test_images": len(by_split.get(rg.UMD_EVAL_SPLIT, set())),
        "rows": rows,
        "max_overlap": max((r["overlapping_images"] for r in rows), default=0),
        "seconds": round(time.time() - t0, 2),
    }
    _dump(out / "image_overlap_summary.json", payload)
    for r in rows:
        _log(
            f"G1 test vs {r['refcoco_plus_set']}: overlap={r['overlapping_images']} "
            f"remaining_images={r['non_overlapping_images']} "
            f"remaining_expr={r['expressions_remaining']} remaining_refs={r['refs_remaining']}"
        )
    return payload


# ---------------------------------------------------------------------------
# G2 - external subsets
# ---------------------------------------------------------------------------
def stage_subsets(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    adir = Path(args.annotation_dir)
    regions = load_regions(adir)
    instances = load_archive_instances(adir)
    boxes_by_ann, meta_by_image, cat_names = archive_join(instances)
    sets = rg.load_development_image_sets(Path(args.manifest_dir))
    membership = rg.subset_membership(rg.image_ids_by_split(regions), sets)

    expressions = rg.expand_expressions(
        regions,
        boxes_by_ann=boxes_by_ann,
        image_meta=meta_by_image,
        category_names=cat_names,
        splits=(rg.UMD_EVAL_SPLIT,),
        subsets_by_image=membership,
    )
    stats = rg.build_subsets(expressions, regions, membership)
    gate = rg.size_gate(stats)
    external = [e for e in expressions if e.level > 0]

    _write_csv(out / "audit_expressions.csv", [e.to_row() for e in external])
    payload = {
        "stage": "G2_external_subsets",
        "definitions": {
            rg.SUBSET_STRICT: (
                "RefCOCOg UMD test minus ALL RefCOCO+ images (train, val_select, "
                "val_calib, testA, testB) - the study has looked at testA/testB "
                "results even though it never trained on them"
            ),
            rg.SUBSET_DEV: (
                "RefCOCOg UMD test minus the development-exposed images (train, "
                "val_select, val_calib); reported as the superset it is"
            ),
        },
        "cumulative": True,
        "size_gate": gate,
        "test_expressions_total": len(expressions),
        "external_expressions_total": len(external),
        "test_images_total": len({r.image_id for r in regions if r.split == rg.UMD_EVAL_SPLIT}),
        "excluded_by_strict": stats[rg.SUBSET_STRICT].excluded_images,
        "excluded_by_development": stats[rg.SUBSET_DEV].excluded_images,
        "membership_disjoint_check": {
            # ``level`` is a partition, so strict must be a subset of the looser
            # development-disjoint definition; an image classified as strict but
            # not as dev would mean the two exclusion sets were wired backwards
            # (and defaulting an excluded image to "strict" would silently
            # advertise the whole RefCOCOg test split as external material)
            "strict_without_development": len(
                {
                    iid
                    for iid, name in membership.items()
                    if name == rg.SUBSET_STRICT
                    and iid not in rg.image_ids_by_split(regions)[rg.UMD_EVAL_SPLIT] - sets["development"]
                }
            ),
            "membership_values_outside_the_two_subsets": sorted(
                {name for name in membership.values()} - {rg.SUBSET_STRICT, rg.SUBSET_DEV}
            ),
            "expressions_outside_both_subsets": sum(1 for e in expressions if e.level == 0),
        },
        "seconds": round(time.time() - t0, 2),
    }
    _dump(out / "external_subsets.json", payload)

    # --- section 18: language shift, measured identically on both datasets ---
    groups: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
        f"refcocog_umd_test::{rg.SUBSET_STRICT}": [
            (e.text, rg.whitespace_tokens(e.text)) for e in expressions if e.level == 1
        ],
        f"refcocog_umd_test::{rg.SUBSET_DEV}": [
            (e.text, rg.whitespace_tokens(e.text)) for e in expressions if e.level in (1, 2)
        ],
        "refcocog_umd_test::all": [
            (e.text, rg.whitespace_tokens(e.text)) for e in expressions
        ],
    }
    groups.update(_refcoco_plus_language(Path(args.refcoco_plus_dir)))
    dist_rows = rg.expression_length_rows(groups)
    for row in dist_rows:
        flat = {
            "group": row.get("group"),
            "n_expressions": row.get("n_expressions"),
            **{
                f"tokens_{k}": v for k, v in (row.get("tokens_per_expression") or {}).items()
            },
            **{
                k: v
                for k, v in row.items()
                if k not in ("group", "n_expressions", "tokens_per_expression")
            },
        }
        row.clear()
        row.update(flat)
    _write_csv(out / "expression_distribution.csv", dist_rows)

    primary = gate["primary_subset"]
    n_img = stats[primary].n_images if primary else 0
    n_expr = stats[primary].n_expressions if primary else 0
    _log(
        f"G2 strict={stats[rg.SUBSET_STRICT].to_dict()} "
        f"dev={stats[rg.SUBSET_DEV].to_dict()} -> {gate['decision']} primary={primary or '-'}"
    )
    _log(f"G2 auditable expressions in the primary subset: {n_expr} over {n_img} images")
    return payload


def _refcoco_plus_language(plus_dir: Path) -> Dict[str, List[Tuple[str, Tuple[str, ...]]]]:
    """RefCOCO+ testA/testB expressions, tokenised the same whitespace way."""
    records = load_refcoco_plus_regions(Path(plus_dir) / "refs(unc).p")
    groups: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
        "refcoco_plus_testA_testB::all_sentences": [],
        "refcoco_plus_testA_testB::first_sentence_per_region": [],
    }
    for rec in records:
        if str(rec.get("split")) not in (SPLIT_TEST_A, SPLIT_TEST_B):
            continue
        texts: List[str] = []
        for sent in rec.get("sentences") or []:
            if isinstance(sent, str):
                texts.append(sent.strip())
            elif isinstance(sent, Mapping):
                for key in ("sent", "raw", "sentence", "text", "phrase"):
                    value = sent.get(key)
                    if isinstance(value, str) and value.strip():
                        texts.append(value.strip())
                        break
        texts = [t for t in texts if t]
        groups["refcoco_plus_testA_testB::all_sentences"].extend(
            (t, rg.whitespace_tokens(t)) for t in texts
        )
        if texts:
            groups["refcoco_plus_testA_testB::first_sentence_per_region"].append(
                (texts[0], rg.whitespace_tokens(texts[0]))
            )
    return groups


# ---------------------------------------------------------------------------
# image fetch
# ---------------------------------------------------------------------------
def stage_images(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    inv = _read_csv(out / "audit_expressions.csv")
    if not inv:
        raise FileNotFoundError("run the subsets stage first (audit_expressions.csv missing)")
    adir = Path(args.annotation_dir)
    regions = load_regions(adir)
    instances = load_archive_instances(adir)
    _boxes, meta_by_image, _names = archive_join(instances)
    by_ref = {(r.image_id, r.ann_id): r for r in regions}

    images = sorted({int(r["image_id"]) for r in inv})
    ann_by_image: Dict[int, int] = {}
    for r in inv:
        ann_by_image.setdefault(int(r["image_id"]), int(r["ann_id"]))
    names: Dict[int, str] = {}
    for iid in images:
        reg = by_ref.get((iid, ann_by_image[iid]))
        if reg is None:
            raise KeyError(f"audit_expressions.csv row for image {iid} has no refs(umd) region")
        names[iid] = rg.canonical_image_file_name(reg.file_name, iid)
    frozen = {int(i) for i in _frozen_bank_image_ids(Path(args.frozen_bank))}
    need = [n for iid, n in names.items() if iid not in frozen]
    report = ci.fetch_coco_images(need, Path(args.image_root), workers=args.workers, log=_log)

    rows: List[Dict[str, Any]] = []
    dims_bad = 0
    for iid in images:
        name = names[iid]
        archive = meta_by_image.get(iid)
        path = ci.local_image_path(Path(args.image_root), name)
        frozen_copy = ci.local_image_path(Path(args.mscoco_root), name)
        if path.exists():
            source = "refcocog_download"
        elif iid in frozen and frozen_copy.exists():
            # the pixel this bank row was built from - decode it to prove the
            # RefCOCOg size claim matches the image RefCOCO+ was developed on
            source = "frozen_refcoco_plus_bank"
            path = frozen_copy
        elif iid in frozen:
            source = "frozen_refcoco_plus_bank"
            path = None
        else:
            source = "missing"
            path = None
        actual = ci.decode_dims(path) if path is not None else None
        if actual is not None and archive is not None and tuple(actual) != tuple(archive):
            dims_bad += 1
            verdict = "MISMATCH"
        elif actual is None:
            verdict = "NOT_DECODED"
        else:
            verdict = "OK"
        rows.append(
            {
                "image_id": iid,
                "file_name": name,
                "source": source,
                "archive_width": archive[0] if archive else "",
                "archive_height": archive[1] if archive else "",
                "decoded_width": actual[0] if actual else "",
                "decoded_height": actual[1] if actual else "",
                "verdict": verdict,
            }
        )
    _write_csv(out / "image_dims_check.csv", rows)
    payload = {
        "stage": "images",
        "fetch": report.to_dict(),
        "images_required": len(images),
        "images_reused_from_frozen_bank": sum(1 for r in rows if r["source"] == "frozen_refcoco_plus_bank"),
        "images_downloaded": sum(1 for r in rows if r["source"] == "refcocog_download"),
        "images_missing": sum(1 for r in rows if r["source"] == "missing"),
        "dimension_mismatches": dims_bad,
        "verdict": (
            "IMAGE GEOMETRY OK"
            if report.requested and dims_bad == 0 and not report.failed
            else ("NO IMAGES" if report.requested == 0 else "IMAGE PROBLEM - SEE FETCH/GEOMETRY")
        ),
        "frozen_bank_note": (
            "RefCOCOg UMD test images that are also RefCOCO+ testA/testB images keep "
            "their frozen proposal bank; only the disjoint remainder is fetched and "
            "re-extracted with the same model (instruction section 10)"
        ),
        "seconds": round(time.time() - t0, 2),
    }
    _dump(out / "image_fetch_report.json", payload)
    _log(
        f"images fetched={report.fetched} existing={report.skipped_existing} "
        f"failed={len(report.failed)} reused_from_frozen_bank="
        f"{payload['images_reused_from_frozen_bank']} in {payload['seconds']}s"
    )
    return payload


def _frozen_bank_image_ids(path: Path) -> List[int]:
    """Image ids of a ``bank-v1`` file (``image_{id}`` groups, frozen layout)."""
    if not Path(path).exists():
        return []
    from ccg.data.bank import image_ids

    return [int(i) for i in image_ids(path).tolist()]


def _read_bank_as_proposal(path: Path, image_id: int) -> Any:
    """One ``bank-v1`` group as the :class:`ccg.data.types.ProposalBank` the audit reads.

    The frozen RefCOCO+ bank and the bank this run appends share the ``bank-v1``
    layout written by ``scripts/extract_proposals.py``, so both sides go through
    exactly one reader here - the reuse of the development bank cannot be a
    reading-convention accident.
    """
    from ccg.data.bank import read_bank_image
    from ccg.data.types import ProposalBank

    boxes, objectness = read_bank_image(path, int(image_id))
    return ProposalBank(image_id=int(image_id), boxes=boxes, objectness=objectness)


# ---------------------------------------------------------------------------
# G3 - frozen proposal audit
# ---------------------------------------------------------------------------
def stage_rpn(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from PIL import Image

    import h5py

    from ccg.data.bank import finalize_bank, write_bank_entry
    from ccg.data.rpn import MODEL_NAME, build_rpn_model, extract_proposal_bank

    t0 = time.time()
    inv = _read_csv(out / "audit_expressions.csv")
    if not inv:
        raise FileNotFoundError("run the subsets stage first (audit_expressions.csv missing)")
    adir = Path(args.annotation_dir)
    regions = load_regions(adir)
    by_ref = {(r.image_id, r.ann_id): r for r in regions}
    expressions = _expressions_from_inventory(inv, by_ref, adir)
    images = sorted({e.image_id for e in expressions})
    gt_index = official_gt(images, Path(args.coco_gt))

    bank_path = Path(args.cache_dir) / "proposals_refcocog.h5"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    cached = set(_frozen_bank_image_ids(bank_path)) if bank_path.exists() else set()
    frozen_ids = set(_frozen_bank_image_ids(Path(args.frozen_bank)))
    need_extraction = [i for i in images if i not in cached and i not in frozen_ids]

    torch.manual_seed(0)
    model = None
    if need_extraction:
        model = build_rpn_model(device=args.device)
        _log(
            f"RPN loaded on {args.device} "
            f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params) for "
            f"{len(need_extraction)} new image(s); {len(images) - len(need_extraction)} "
            "bank(s) are read from cache"
        )

    rows: List[fe.ProposalAuditRow] = []
    bank_size_violations: List[str] = []
    skipped: List[int] = []
    n_extracted = n_reused_frozen = n_reused_local = 0
    handle = h5py.File(bank_path, "a") if need_extraction else None
    for pos, iid in enumerate(images, start=1):
        exprs = [e for e in expressions if e.image_id == iid]
        gt = gt_index.objects(iid).non_crowd()  # A8: category supply excludes iscrowd
        if iid in cached:
            bank = _read_bank_as_proposal(bank_path, iid)
            n_reused_local += 1
        elif iid in frozen_ids:
            bank = _read_bank_as_proposal(Path(args.frozen_bank), iid)  # read-only reuse
            n_reused_frozen += 1
        else:
            name = rg.canonical_image_file_name(by_ref[(iid, exprs[0].ann_id)].file_name, iid)
            path = ci.local_image_path(Path(args.image_root), name)
            if not path.exists():
                skipped.append(iid)
                continue
            with Image.open(path) as im:
                image = np.asarray(im.convert("RGB"))
            bank, meta = extract_proposal_bank(image, model, int(iid), top_n=fe.N_PROPOSALS)
            write_bank_entry(
                handle,
                iid,
                bank.boxes,
                bank.objectness,
                {"n_raw_post_nms": int(meta.get("n_raw_post_nms", bank.boxes.shape[0]))},
                overwrite=False,
            )
            n_extracted += 1
        if int(np.asarray(bank.boxes).shape[0]) != fe.N_PROPOSALS:
            bank_size_violations.append(f"{iid}:{int(np.asarray(bank.boxes).shape[0])}")
        for e in exprs:
            rows.append(
                fe.audit_expression(
                    e,
                    bank,
                    gt_object_boxes_xyxy=gt.boxes,
                    gt_object_name_codes=gt.categories,
                )
            )
        if pos % 50 == 0 or pos == len(images):
            mem = (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else 0.0
            )
            _log(
                f"[G3] {pos}/{len(images)} images, {len(rows)} expressions, "
                f"extracted={n_extracted}, gpu_peak={mem:.0f}MB"
            )

    if handle is not None:
        handle.flush()
        handle.close()
        import torchvision

        finalize_bank(
            bank_path,
            {
                "model_name": MODEL_NAME,
                "weights": "COCO_V1",
                "proposal_type": "class-agnostic post-NMS RPN",
                "top_n": fe.N_PROPOSALS,
                "torchvision_version": torchvision.__version__,
                "images_root": str(args.image_root),
                "note": "A10 RefCOCOg external audit bank, same generator as cache/proposals.h5",
            },
        )

    summary = fe.summarise_rows(rows)
    by_subset = {
        rg.SUBSET_STRICT: [r for r in rows if r.level == 1],
        rg.SUBSET_DEV: [r for r in rows if r.level in (1, 2)],
        "dev_disjoint_only_extra": [r for r in rows if r.level == 2],
    }
    summary["by_subset"] = {name: fe.summarise_rows(sub) for name, sub in by_subset.items()}
    summary["n_images_requested"] = len(images)
    summary["n_images_audited"] = len({r.image_id for r in rows})
    summary["n_images_skipped_missing_file"] = len(skipped)
    summary["skipped_image_ids"] = skipped[:20]
    summary["n_banks_extracted_this_run"] = n_extracted
    summary["n_banks_reused_from_frozen_refcoco_bank"] = n_reused_frozen
    summary["n_banks_reused_from_refcocog_cache"] = n_reused_local
    summary["frozen_bank"] = _frozen_bank_attrs(Path(args.frozen_bank))
    summary["n_bank_size_violations"] = len(bank_size_violations)
    summary["bank_size_violations_sample"] = bank_size_violations[:20]
    summary["frozen_generator"] = {
        "model": "fasterrcnn_resnet50_fpn",
        "weights": "COCO_V1",
        "stage": "class-agnostic RPN, post-NMS top-64",
        "n_proposals": fe.N_PROPOSALS,
        "target_iou_thresh": 0.5,
        "note": "identical to the RefCOCO+ audit; nothing was re-selected for RefCOCOg",
    }
    summary["refcoco_plus_reference"] = _refcoco_reference()
    summary["seconds"] = round(time.time() - t0, 2)

    _write_csv(out / "rpn_audit.csv", fe.audit_cohort(rows))
    _dump(out / "rpn_summary.json", summary)
    _log(
        f"G3 done in {summary['seconds']}s: rows={summary.get('n_rows')} "
        f"recall@0.5={_rate_of(summary, 'recall_at_05')} "
        f"K5={_rate_of(summary, f'k{fe.PRIMARY_K}_availability')} "
        f"same-cat K5={_same_of(summary)}"
    )
    return summary


def _rate_of(summary: Mapping[str, Any], key: str) -> Any:
    val = summary.get(key)
    return val.get("rate") if isinstance(val, dict) else val


def _same_of(summary: Mapping[str, Any]) -> Any:
    val = summary.get(f"k{fe.PRIMARY_K}_availability")
    return val.get("same_name_rate") if isinstance(val, dict) else None


def _expressions_from_inventory(
    rows: Sequence[Mapping[str, str]],
    by_ref: Mapping[Tuple[int, int], rg.RefCOCOGRegion],
    adir: Path,
) -> List[rg.RefCOCOGExpression]:
    """Rebuild the audit inputs from ``audit_expressions.csv`` (single source of truth).

    The CSV is written by G2 and hashed into the run record, so the audited set
    cannot silently differ from the set the size gate counted.
    """
    instances = load_archive_instances(adir)
    boxes_by_ann, meta_by_image, cat_names = archive_join(instances)
    out: List[rg.RefCOCOGExpression] = []
    for r in rows:
        image_id, ann_id = int(r["image_id"]), int(r["ann_id"])
        box = boxes_by_ann.get(ann_id)
        meta = meta_by_image.get(image_id)
        if box is None or meta is None or (image_id, ann_id) not in by_ref:
            continue
        x, y, w, h = box
        out.append(
            rg.RefCOCOGExpression(
                expr_id=int(r["expr_id"]),
                image_id=image_id,
                level=int(r["level"]),
                tuple_type=str(r["tuple_type"]),
                target_name=str(r["target_name"]),
                width=int(meta[0]),
                height=int(meta[1]),
                gt_box=(x, y, x + w, y + h),
                ann_id=ann_id,
                category_id=int(r["category_id"]),
                ref_id=int(r["region_ref_id"]),
                region_ref_id=int(r["region_ref_id"]),
                split=str(r["split"]),
                subset=str(r["subset"]),
            )
        )
    return out


def _frozen_bank_attrs(path: Path) -> Dict[str, Any]:
    if not Path(path).exists():
        return {"path": _rel(path), "error": "missing"}
    import h5py

    with h5py.File(path, "r") as handle:
        attrs = {
            k: (v.decode() if isinstance(v, bytes) else v.item() if hasattr(v, "item") else v)
            for k, v in handle.attrs.items()
        }
    attrs["path"] = _rel(path)
    return attrs


# ---------------------------------------------------------------------------
# G4 - candidates, cohort, categories
# ---------------------------------------------------------------------------
def stage_candidates(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    summary_path = out / "rpn_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run the rpn stage first (rpn_summary.json missing)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inv = _read_csv(out / "audit_expressions.csv")
    adir = Path(args.annotation_dir)
    regions = load_regions(adir)
    by_ref = {(r.image_id, r.ann_id): r for r in regions}
    expressions = _expressions_from_inventory(inv, by_ref, adir)
    ref_id_by_expr = {e.expr_id: e.region_ref_id for e in expressions}
    audit_rows = _rows_from_csv(_read_csv(out / "rpn_audit.csv"))

    primary = _primary_subset(out)
    prim_rows = [r for r in audit_rows if r.level == 1] if primary == rg.SUBSET_STRICT else audit_rows
    #: every "overall" figure below is the *primary* subset's summary, never the
    #: pooled one - a dev-disjoint rate labelled as the strict subset would move
    #: the reported gate without moving any data
    prim_summary = (summary.get("by_subset") or {}).get(primary) or summary

    # --- section 14: same-category distractor supply ------------------------
    n_same = [r.n_same_name_distractors for r in prim_rows]
    n_valid = [r.n_valid_distractors for r in prim_rows]
    coverage = rg.same_category_coverage_rows(n_same, n_valid=n_valid, group=primary or "all")
    for level, label in ((2, "dev_disjoint_only_extra"),):
        sub = [r for r in audit_rows if r.level == level]
        if sub:
            coverage.extend(
                rg.same_category_coverage_rows(
                    [r.n_same_name_distractors for r in sub],
                    n_valid=[r.n_valid_distractors for r in sub],
                    group=label,
                )
            )
    availability_rows: List[Dict[str, Any]] = [dict(r) for r in coverage]
    for key in (
        "recall_at_05",
        "recall_at_07",
        f"k{fe.PRIMARY_K}_availability",
        f"k{fe.SECONDARY_K}_availability",
    ):
        val = prim_summary.get(key) or {}
        availability_rows.append(
            {
                "group": f"{primary or 'all'}::overall",
                "distractor_level": "",
                "cohort_size": key,
                "n_rows": prim_summary.get("n_rows"),
                "availability": _num(val.get("rate") if isinstance(val, dict) else val),
                "wilson_low": val.get("wilson_low") if isinstance(val, dict) else "",
                "wilson_high": val.get("wilson_high") if isinstance(val, dict) else "",
                "same_category_availability": _num(
                    val.get("same_name_rate") if isinstance(val, dict) else None
                ),
            }
        )
    _write_csv(out / "candidate_availability.csv", availability_rows)

    k5_same = float(
        ((summary.get("by_subset") or {}).get(primary or "") or {})
        .get(f"k{fe.PRIMARY_K}_availability", {})
        .get("same_name_rate", float("nan"))
    )
    if not np.isfinite(k5_same):
        k5_same = float(
            (summary.get(f"k{fe.PRIMARY_K}_availability") or {}).get(
                "same_name_rate", float("nan")
            )
        )
    same_category = rg.same_category_verdict_a10(k5_same)

    # --- section 16/17: the matched hard cohort ------------------------------
    hard = rg.matched_hard_rows(prim_rows, ref_id_by_expr, k=fe.PRIMARY_K)
    _write_csv(out / "hard_cohort.csv", hard)
    n_hard_expr = len(hard)
    n_hard_img = len({h["image_id"] for h in hard})
    n_hard_ref = len({h["ref_id"] for h in hard})
    n_hard_cat = len({h["category"] for h in hard})
    power = rg.hard_cohort_gate_a10(n_hard_expr, n_hard_img)
    matched_identity = sum(1 for h in hard if h["matched_random_k5_available"] == 1)
    hard_payload = {
        "stage": "G4_hard_cohort",
        "primary_subset": primary,
        "definition": (
            "expression is hard-eligible when a K=5 set exists whose target is the "
            "argmax-IoU proposal (IoU >= 0.5), whose 4 distractors are COCO same-category "
            "objects (proposal -> highest-IoU COCO GT with IoU >= 0.5) and whose "
            "target-equivalent proposals were removed - the A8 rule verbatim"
        ),
        "counts": {
            "n_expressions": n_hard_expr,
            "n_images": n_hard_img,
            "n_refs": n_hard_ref,
            "n_categories": n_hard_cat,
            "n_candidate_rows": n_hard_expr,
        },
        "matched_control": {
            "criterion": "every hard-eligible expression must also admit the random K=5 view "
            "of the same sentence/image/target (section 17)",
            "n_matched_random_available": matched_identity,
            "identity_ok": bool(hard) and matched_identity == n_hard_expr,
        },
        "power_gate": power,
        "mean_same_category_distractors": round(
            float(np.mean([h["n_same_category_distractors"] for h in hard])), 4
        )
        if hard
        else None,
        "refcoco_plus_phase1f_reference": summary.get("refcoco_plus_reference", {}).get(
            "phase1f_hard_cohort"
        ),
        "seconds": round(time.time() - t0, 2),
    }
    _dump(out / "hard_cohort_summary.json", hard_payload)

    # --- section 19: target-category distribution ---------------------------
    groups: Dict[str, List[Tuple[str, int]]] = {
        f"refcocog_umd_test::{primary or 'all'}": [
            (r.target_name, r.image_id) for r in prim_rows
        ],
        "refcocog_umd_test::hard_cohort": [(h["category"], h["image_id"]) for h in hard],
    }
    if primary == rg.SUBSET_STRICT:
        groups["refcocog_umd_test::rg_external_devdisjoint"] = [
            (r.target_name, r.image_id) for r in audit_rows
        ]
    groups["refcoco_plus::phase1f_hard_cohort"] = _refcoco_plus_hard_categories(
        Path(args.refcoco_plus_dir), Path(args.coco_gt)
    )
    _write_csv(out / "category_distribution.csv", rg.category_rows(groups))

    _log(
        f"G4 same-cat K5={k5_same:.4f} ({same_category['verdict']}) hard cohort="
        f"{n_hard_expr} expr / {n_hard_img} img / {n_hard_ref} refs ({power['verdict']})"
    )
    return {
        "availability_rows": len(availability_rows),
        "same_category": same_category,
        "hard_cohort": hard_payload,
        "category_groups": sorted(groups),
    }


def _rows_from_csv(rows: Sequence[Mapping[str, str]]) -> List[fe.ProposalAuditRow]:
    """Re-read ``rpn_audit.csv`` into audit rows (keeps G4 off the GPU path)."""
    out: List[fe.ProposalAuditRow] = []
    for r in rows:
        available = {}
        same_available = {}
        for key, value in r.items():
            if key.startswith("available_K") and value != "":
                available[int(key[11:])] = bool(int(value))
            elif key.startswith("same_name_available_K") and value != "":
                same_available[int(key[21:])] = bool(int(value))
        out.append(
            fe.ProposalAuditRow(
                expr_id=int(r["expr_id"]),
                image_id=int(r["image_id"]),
                level=int(r["level"]),
                tuple_type=str(r["tuple_type"]),
                target_name=str(r["target_name"]),
                n_proposals=int(r["n_proposals"]),
                target_best_iou=float(r["target_best_iou"]),
                target_present=bool(int(r["target_present_at_05"])),
                n_equivalent=int(r["n_equivalent"]),
                n_valid_distractors=int(r["n_valid_distractors"]),
                n_invalid_crop=int(r["n_invalid_crop"]),
                n_same_name_distractors=int(r["n_same_name_distractors"]),
                available=available,
                same_name_available=same_available,
            )
        )
    return out


def _primary_subset(out: Path) -> str:
    path = Path(out) / "external_subsets.json"
    if not path.exists():
        return ""
    return str(json.loads(path.read_text(encoding="utf-8")).get("size_gate", {}).get("primary_subset", ""))


def _refcoco_plus_hard_categories(plus_dir: Path, coco_gt: Path) -> List[Tuple[str, int]]:
    """COCO categories of the Phase 1F same-category cohort (the comparison side).

    Read from the committed Phase 1F manifest rather than recomputed: the point of
    section 19 is whether *the cohort that produced the RefCOCO+ result* has a
    different category mix from the external one, so its membership is an input.
    """
    path = ROOT / HARD_MANIFEST
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=False)  # int64 arrays only, no object deserialisation
    ref_ids = {int(v) for v in np.asarray(data["ref_id"]).reshape(-1)}
    records = load_refcoco_plus_regions(Path(plus_dir) / "refs(unc).p")
    ann_by_ref = {
        int(r["ref_id"]): (int(r["ann_id"]), int(r["image_id"]))
        for r in records
        if int(r["ref_id"]) in ref_ids
    }
    gt_index = official_gt(
        sorted({img for _ann, img in ann_by_ref.values()}), coco_gt
    )
    official = _official_ann_map(gt_index)
    rows: List[Tuple[str, int]] = []
    for _ref_id, (ann_id, image_id) in sorted(ann_by_ref.items()):
        hit = official.get(ann_id)
        if hit is None:
            continue
        name = gt_index.category_names.get(int(hit["category_id"]))
        rows.append((str(name if name else f"cat_{hit['category_id']}"), int(image_id)))
    return rows


# ---------------------------------------------------------------------------
# G5 - decision
# ---------------------------------------------------------------------------
def stage_decision(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    subsets = json.loads((out / "external_subsets.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "rpn_summary.json").read_text(encoding="utf-8"))
    hard = json.loads((out / "hard_cohort_summary.json").read_text(encoding="utf-8"))
    gate = subsets.get("size_gate") or {}
    primary = gate.get("primary_subset") or ""
    prim = (summary.get("by_subset") or {}).get(primary, {}) or {}

    recall = float(prim.get("recall_at_05", {}).get("rate", float("nan")))
    k5_rand = float(prim.get(f"k{fe.PRIMARY_K}_availability", {}).get("rate", float("nan")))
    k5_same = float(
        prim.get(f"k{fe.PRIMARY_K}_availability", {}).get("same_name_rate", float("nan"))
    )
    engineering = rg.engineering_verdict_a10(recall, k5_rand)
    same_category = rg.same_category_verdict_a10(k5_same)
    power = hard.get("power_gate") or rg.hard_cohort_gate_a10(0, 0)
    branch = rg.branch_decision_a10(
        size_decision=primary,
        engineering=engineering,
        same_category=same_category,
        hard_cohort=power,
    )

    ref = summary.get("refcoco_plus_reference") or {}
    ref_recall = _num(ref.get("target_recall_at_05"))
    ref_recall = float("nan") if ref_recall is None else ref_recall
    ref_same = _num(ref.get("same_category_k5_availability"))
    comparison = {
        "refcoco_plus_target_recall_at_05_frozen_reference": ref_recall,
        "refcocog_external_target_recall_at_05": recall,
        "delta_recall_at_05": round(recall - ref_recall, 6) if np.isfinite(ref_recall) else None,
        "refcoco_plus_target_recall_at_07": _num(ref.get("target_recall_at_07")),
        "refcocog_external_target_recall_at_07": _num(
            float(prim.get("recall_at_07", {}).get("rate", float("nan")))
        )
        if np.isfinite(float(prim.get("recall_at_07", {}).get("rate", float("nan"))))
        else None,
        "refcoco_plus_same_category_k5_availability": ref_same,
        "refcocog_external_same_category_k5_availability": _num(k5_same),
        "delta_same_category_k5_availability": (
            round(k5_same - ref_same, 6) if ref_same is not None else None
        ),
        "refcoco_plus_gt_object_recall_at_05": _num(ref.get("gt_object_recall_at_05")),
        "source": ref.get("source_dir"),
        "note": (
            "loaded from results/proposal_audit at run time (section 12); "
            "ref_target_recall@* is the referred-target recall the A10 gate uses, "
            "gt_object_recall@* is recall over every COCO GT object"
        ),
    }

    payload = {
        "stage": "G5_decision",
        "amendment": "A10",
        "size_gate": {"primary_subset": primary, "decision": gate.get("decision")},
        "engineering_gate": engineering,
        "same_category_gate": same_category,
        "hard_cohort_gate": power,
        "branch": branch,
        "refcoco_plus_comparison": comparison,
        "measured": {
            "primary_subset": primary,
            "n_rows_audited": prim.get("n_rows"),
            "n_images_audited": prim.get("n_images"),
            "recall_at_05": recall,
            "recall_at_07": float(prim.get("recall_at_07", {}).get("rate", float("nan"))),
            "natural_omission_at_05": prim.get("natural_omission_at_05"),
            "k5_random_availability": k5_rand,
            "k10_random_availability": float(
                prim.get(f"k{fe.SECONDARY_K}_availability", {}).get("rate", float("nan"))
            ),
            "k5_same_category_availability": k5_same,
            "same_category_coverage": prim.get("same_name_distractors"),
            "gt_object_recall_at_05": prim.get("gt_object_recall_at_05"),
            "hard_cohort": hard.get("counts"),
            "frozen_generator": summary.get("frozen_generator"),
            "n_banks_reused_from_frozen_refcoco_bank": summary.get(
                "n_banks_reused_from_frozen_refcoco_bank"
            ),
            "n_banks_extracted_this_run": summary.get("n_banks_extracted_this_run"),
        },
        "boundary": (
            "cross-dataset external confirmation under a shared COCO visual domain and a "
            "shared COCO object ontology; NOT cross-visual-domain generalisation "
            "(instruction section 24)"
        ),
        "not_run_this_round": [
            "CLIP / OpenCLIP feature extraction",
            "B3 inference",
            "Stats Logistic / E1b inference",
            "bootstrap of any external model statistic",
            "any FineCops (A9) continuation",
            "any training or calibration of any kind",
        ],
        "next_stage_allowed": branch.get("next_stage_allowed"),
        "seconds": round(time.time() - t0, 2),
    }
    _dump(out / "branch_decision.json", payload)
    figures = write_figures(out)
    meta = {
        "phase": "1E_A10_refcocog_feasibility",
        "stages_run": list(STAGES),
        "git_commit": _git_commit(),
        "environment": _env(),
        "figures": figures,
        "artifacts": sorted(
            _rel(p) for p in Path(out).rglob("*") if p.is_file() and p.suffix != ".tmp"
        ),
        "primary_subset": primary,
        "size_decision": gate.get("decision"),
        "engineering_verdict": engineering.get("verdict"),
        "branch": branch.get("branch"),
        "total_wallclock_s": None,
    }
    _dump(out / "metadata.json", meta)
    _log(
        f"G5 branch={branch['branch']} ({branch['reason']}) "
        f"recall@0.5={recall:.4f} K5rand={k5_rand:.4f} K5same={k5_same:.4f}"
    )
    print(
        "PHASE1E_REFCOCOG_FEASIBILITY_COMPLETE "
        f"branch={branch['branch']} primary={primary or '-'} out={out}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# figures
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
    summary = json.loads((out / "rpn_summary.json").read_text(encoding="utf-8"))
    primary = _primary_subset(out)
    prim = (summary.get("by_subset") or {}).get(primary, {}) or {}
    if not rows:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ious = np.asarray([float(r["target_best_iou"]) for r in rows], dtype=float)
    axes[0].hist(ious, bins=40, range=(0.0, 1.0), color="#3b6ea5")
    axes[0].axvline(0.5, ls="--", c="k", lw=1)
    axes[0].axvline(0.7, ls=":", c="k", lw=1)
    axes[0].set_title(
        "RefCOCOg external target max-IoU (frozen N=64)\n"
        f"recall@0.5={prim.get('recall_at_05', {}).get('rate', float('nan')):.3f}  "
        f"@0.7={prim.get('recall_at_07', {}).get('rate', float('nan')):.3f}"
    )
    axes[0].set_xlabel("IoU(target box, best proposal)")
    axes[0].set_ylabel("expressions")

    same = np.asarray([int(r["n_same_name_distractors"]) for r in rows], dtype=float)
    for level, colour, label in (
        (1, "#4f8a4f", rg.SUBSET_STRICT),
        (2, "#e8a33d", "dev-disjoint-only extra"),
    ):
        sel = same[[i for i, r in enumerate(rows) if int(r["level"]) == level]]
        if sel.size:
            axes[1].hist(
                sel, bins=np.arange(-0.5, 24.5, 1), alpha=0.6, label=label, color=colour
            )
    axes[1].axvline(fe.PRIMARY_K - 1, ls="--", c="k", lw=1)
    axes[1].set_title("COCO same-category distractor supply (K=5 needs >= 4)")
    axes[1].set_xlabel("# proposals mapped to another COCO GT object of the target category")
    axes[1].legend(fontsize=8)

    ref = summary.get("refcoco_plus_reference") or {}
    ref_recall = _num(ref.get("target_recall_at_05"))
    ref_recall = float("nan") if ref_recall is None else ref_recall
    labels = ["recall@0.5", "K5 random", "K5 same-cat"]
    vals = [
        prim.get("recall_at_05", {}).get("rate", 0.0),
        prim.get(f"k{fe.PRIMARY_K}_availability", {}).get("rate", 0.0),
        prim.get(f"k{fe.PRIMARY_K}_availability", {}).get("same_name_rate", 0.0),
    ]
    bars = axes[2].bar(labels, vals, color=["#3b6ea5", "#4f8a4f", "#c0504d"])
    axes[2].axhline(rg.A10_GO_RECALL_MIN, ls="--", c="k", lw=1, label=f"A10 GO {rg.A10_GO_RECALL_MIN}")
    axes[2].axhline(rg.A10_GRAY_RECALL_MIN, ls=":", c="k", lw=1, label=f"A10 STOP <{rg.A10_GRAY_RECALL_MIN}")
    if np.isfinite(ref_recall):
        axes[2].axhline(ref_recall, ls="-.", c="#666666", lw=1, label=f"RefCOCO+ {ref_recall:.3f}")
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[2].set_ylim(0, 1.15)
    axes[2].set_title(f"frozen-pipeline constructibility on {primary or 'RefCOCOg external'}")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    path = out / "figures" / "fig1_refcocog_feasibility.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    dist = _read_csv(out / "expression_distribution.csv")
    dist = [r for r in dist if r.get("n_expressions") not in ("", "0", None)]
    if dist:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        names = [str(r["group"]) for r in dist]
        means = [float(r.get("tokens_mean") or 0.0) for r in dist]
        meds = [float(r.get("tokens_median") or 0.0) for r in dist]
        absr = [float(r.get("absolute_position_rate") or 0.0) for r in dist]
        x = np.arange(len(names))
        ax.bar(x - 0.2, means, width=0.2, label="mean tokens", color="#3b6ea5")
        ax.bar(x, meds, width=0.2, label="median tokens", color="#4f8a4f")
        ax.bar(x + 0.2, absr, width=0.2, label="absolute-position rate", color="#c0504d")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_title("annotation-distribution shift (section 18): one tokenizer for both sets")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path2 = out / "figures" / "fig2_language_shift.png"
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        _log(f"figure -> {path2}")
    _log(f"figure -> {path}")
    return [str(path)]


STAGE_FUNCS = {
    "parse": stage_parse,
    "overlap": stage_overlap,
    "subsets": stage_subsets,
    "images": stage_images,
    "rpn": stage_rpn,
    "candidates": stage_candidates,
    "decision": stage_decision,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list of {STAGES}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--annotation-dir", type=Path, default=REFCOCOG_DIR)
    ap.add_argument("--refcoco-plus-dir", type=Path, default=REFCOCO_PLUS_DIR)
    ap.add_argument("--coco-gt", type=Path, default=COCO_GT)
    ap.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument(
        "--mscoco-root",
        type=Path,
        default=Path("data/raw/mscoco"),
        help="the frozen RefCOCO+ image tree (read-only, geometry cross-check only)",
    )
    ap.add_argument("--frozen-bank", type=Path, default=FROZEN_BANK)
    ap.add_argument("--workers", type=int, default=ci.DEFAULT_WORKERS)
    ap.add_argument("--device", default="cuda")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.annotation_dir / "refs(umd).p", args.annotation_dir / "instances.json"):
        if not Path(path).exists():
            raise SystemExit(
                f"{path} not found - run tools/download_refcocog.py first (A10 G0)"
            )
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGE_FUNCS:
            raise SystemExit(f"unknown stage {s!r}; expected one of {STAGES}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for s in stages:
        _log(f"=== stage {s} ===")
        STAGE_FUNCS[s](out, args)
    _log(f"stages {','.join(stages)} done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
