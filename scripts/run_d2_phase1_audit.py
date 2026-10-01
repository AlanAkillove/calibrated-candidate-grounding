"""V2-D2 Phase 1 - data / split audit and cohort redefinition.

The D2 brief asked for a *strict image-disjoint* RefCOCO cohort.  That premise is
false on this data, and this artifact proves it rather than papering over it:

* every RefCOCO ``testA``/``testB`` image is, bar one, the same physical COCO
  ``train2014`` image the RefCOCO+ study was trained / selected / evaluated on, so
  ``|testA_testB \ all_refcoco_plus| = 1`` and a non-empty image-disjoint cohort
  does not exist;
* what *is* new is the **expression language distribution** - RefCOCO keeps the
  spatial-mention words RefCOCO+ deleted - on the frozen proposal system.

So Phase 1 redefines the D2 cohort as the honest transfer set and freezes it:
``RefCOCO testA ∪ testB`` expressions on the shared COCO images, the frozen
``N = 64`` RPN bank, target-present rows only.  It also runs the construction end to
end (expressions -> target-present filter -> matched ``K = 5/10/20/50`` cohort) so
the downstream inference stages read a persisted cohort instead of re-deriving it,
and records the environment versions the reproducibility review required.

    conda activate deepminer
    python scripts/run_d2_phase1_audit.py

Outputs (under ``results/v2_d2_refcoco_lang/phase1``):
  * ``cohort_audit.json``   - material counts, honest overlap matrix, cohort sizes;
  * ``cohort.csv``          - the frozen matched cohort sidecar;
  * ``image_sizes.json``    - ``(W, H)`` of the cohort images;
  * ``environment.json``    - Python / PyTorch / sklearn / transformers / CUDA versions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.external import refcoco_lang as L  # noqa: E402

DEFAULT_OUT = Path("results/v2_d2_refcoco_lang/phase1")


def environment_info() -> Dict[str, Any]:
    """Spec-environment versions (the reproducibility requirement before D2)."""
    import importlib

    def _ver(mod: str):
        try:
            return str(importlib.import_module(mod).__version__)
        except Exception as exc:  # pragma: no cover - environment dependent
            return f"unavailable:{type(exc).__name__}"

    info: Dict[str, Any] = {"python": sys.version.split()[0], "numpy": _ver("numpy")}
    try:
        import torch

        info.update(
            torch=_ver("torch"),
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=(torch.version.cuda if torch.cuda.is_available() else None),
            device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            gpu_count=int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        )
    except Exception as exc:  # pragma: no cover
        info["torch"] = f"unavailable:{type(exc).__name__}"
    info.update(sklearn=_ver("sklearn"), transformers=_ver("transformers"),
                open_clip=_ver("open_clip"))
    return info


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap_report(refcoco_images: List[int], plus_sets: Dict[str, Any]) -> Dict[str, Any]:
    """RefCOCO testA/testB image set vs every RefCOCO+ exposure set (the honest matrix)."""
    rc = set(int(i) for i in refcoco_images)
    out: Dict[str, Any] = {"n_refcoco_test_images": len(rc)}
    per_set: Dict[str, Any] = {}
    for key in ("train", "val_select", "val_calib", "testA", "testB", "development",
                "all_refcoco_plus", "all_refcoco_plus_ids"):
        if key not in plus_sets:
            continue
        other = set(int(i) for i in plus_sets[key])
        per_set[key] = {
            "plus_images": len(other),
            "overlap_with_refcoco_test": len(rc & other),
            "refcoco_test_disjoint_from_this": len(rc - other),
        }
    all_plus = set(int(i) for i in plus_sets["all_refcoco_plus_ids"])
    out["per_set"] = per_set
    out["strict_image_disjoint_survivors"] = len(rc - all_plus)
    out["image_disjoint_premise_holds"] = (len(rc - all_plus) > 0)
    return out


def cohort_sizes(cohort: L.LangCohort) -> Dict[str, Any]:
    """Cohort counts + per-K availability (material, not yet the feasibility audit)."""
    common = L.common_cohort_rows(cohort)
    hard = L.matched_hard_rows(cohort)
    avail = {}
    for k in L.C1_KS:
        avail[f"available_K{k}"] = int(np.count_nonzero(cohort.n_valid_distractors >= k - 1))
    return {
        "n_rows": len(cohort),
        "n_images": cohort.n_images,
        "n_refs": cohort.n_refs,
        "common_cohort_K50_eligible": int(common.size),
        "matched_hard_K5_eligible": int(hard.size),
        "per_K_availability": avail,
        "mean_target_best_iou": float(np.mean(cohort.target_best_iou)) if len(cohort) else None,
        "mean_n_valid_distractors": float(np.mean(cohort.n_valid_distractors)) if len(cohort) else None,
        "mean_n_same_category": float(np.mean(cohort.n_same_category)) if len(cohort) else None,
    }


def write_cohort_csv(cohort: L.LangCohort, path: Path) -> None:
    header = [
        "expr_id", "ref_id", "image_id", "ann_id", "split", "category", "category_id",
        "target_index", "rand_distractors", "hard_distractors", "target_best_iou",
        "n_valid_distractors", "n_same_category", "n_target_equiv", "text",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in range(len(cohort)):
            rand = [int(v) for v in cohort.rand_order[row] if int(v) >= 0]
            hard = [int(v) for v in cohort.hard_order[row] if int(v) >= 0]
            writer.writerow([
                int(cohort.expr_id[row]), int(cohort.ref_id[row]), int(cohort.image_id[row]),
                int(cohort.ann_id[row]), str(cohort.split[row]), str(cohort.category[row]),
                int(cohort.category_id[row]), int(cohort.target_index[row]),
                json.dumps(rand), json.dumps(hard), repr(float(cohort.target_best_iou[row])),
                int(cohort.n_valid_distractors[row]), int(cohort.n_same_category[row]),
                int(cohort.n_target_equiv[row]), str(cohort.text[row]),
            ])


def stage_all(out_dir: Path, log) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = L.load_image_sizes_map()
    log(f"[d2-p1] image_sizes.npz loaded: {len(sizes)} images")

    # 1. honest overlap: RefCOCO testA/testB vs RefCOCO+ exposure sets
    plus_sets = L.load_refcoco_plus_image_sets()
    expressions = L.load_refcoco_test_expressions(sizes=sizes)
    refcoco_images = sorted({int(e.image_id) for e in expressions})
    overlap = overlap_report(refcoco_images, plus_sets)
    log(f"[d2-p1] test images={overlap['n_refcoco_test_images']} "
        f"strict-disjoint survivors={overlap['strict_image_disjoint_survivors']} "
        f"premise_holds={overlap['image_disjoint_premise_holds']}")

    # 2. target-present filter (natural omission reported, never silently dropped)
    present, omission = L.target_present_expressions(expressions)
    log(f"[d2-p1] expressions={len(expressions)} target-present={len(present)} "
        f"omission_rate={omission['natural_omission_rate']:.4f}")

    # 3. matched cohort for every K (reuses the frozen manifest builder)
    image_ids = sorted({int(e.image_id) for e in present})
    gt_index = L.official_gt_index(image_ids)
    cohort, rand_manifest, hard_manifest = L.build_cohorts(present, gt_index=gt_index, log=log)

    # 4. freeze artifacts
    write_cohort_csv(cohort, out_dir / "cohort.csv")
    (out_dir / "image_sizes.json").write_text(json.dumps(
        {str(int(i)): [int(w), int(h)] for i, (w, h) in
        sorted((int(k), v) for k, v in sizes.items() if int(k) in set(int(x) for x in cohort.image_id))},
        indent=2) + "\n", encoding="utf-8")

    report = {
        "protocol": "V2-D2 Phase 1 (language-distribution transfer cohort)",
        "cohort_redefinition": (
            "D2 is expression-language-distribution transfer on shared COCO testA/testB "
            "images under the frozen N=64 RPN; a strict image-disjoint RefCOCO cohort does "
            "not exist because RefCOCO and RefCOCO+ re-annotate the same COCO regions."
        ),
        "boundary": "cross-dataset transfer under a shared COCO visual domain; NOT cross-visual-domain generalisation",
        "dataset": {
            "source": "UNC refcoco refs(unc).p + instances.json (data/raw/refcoco)",
            "splits": list(L.TEST_SPLITS),
            "checksums": {
                "refs(unc).p": _sha(L.REFCOCO_REFS_PICKLE),
                "instances.json": _sha(L.REFCOCO_INSTANCES_JSON),
            },
        },
        "material": {
            "original_images": overlap["n_refcoco_test_images"],
            "original_expressions": len(expressions),
            # "strict" here = the honest redefined cohort (image-disjointness is
            # impossible, so the surviving material is the target-present cohort);
            # reported next to the requested labels so the mapping is explicit.
            "strict_images": cohort.n_images,
            "strict_expressions": len(cohort),
            "images_dropped_in_cohort_build": overlap["n_refcoco_test_images"] - cohort.n_images,
            "target_present_expressions": len(present),
            "natural_omission": omission,
        },
        "overlap": overlap,
        "cohort": cohort_sizes(cohort),
        "environment": environment_info(),
    }
    (out_dir / "cohort_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "cohort.csv.sha256").write_text(_sha(out_dir / "cohort.csv") + "\n", encoding="utf-8")
    log(f"[d2-p1] cohort rows={len(cohort)} images={cohort.n_images} "
        f"common(K50)={cohort_sizes(cohort)['common_cohort_K50_eligible']} "
        f"matched_hard(K5)={cohort_sizes(cohort)['matched_hard_K5_eligible']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    report = stage_all(Path(args.out_dir), print)
    print(f"[d2-p1] done in {round(time.time() - started, 1)}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
