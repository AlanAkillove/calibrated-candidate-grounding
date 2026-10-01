"""V2-D2 Phase 2 - proposal feasibility on the frozen ``N = 64`` COCO RPN.

The proposal generator is *not* touched: same frozen COCO-pretrained RPN, same
``N = 64`` bank (``cache/proposals.h5``), same target-assignment / same-category
rules as RefCOCO+ (they are imported from :mod:`ccg.external.feasibility` and
:mod:`ccg.data`).  What changes is only the expression language distribution, so a
feasibility collapse here would mean the *frozen* system cannot even propose the
RefCOCO target and any downstream reliability number would be uninterpretable.

Measured with :func:`ccg.external.feasibility.audit_expression` / :func:`summarise_rows`
over ``K = 5, 10, 20, 50``:

  * target recall @ IoU 0.5 / 0.7;
  * random availability at ``K = 5/10/20/50``;
  * same-category availability (``>= 1/2/4/9`` same-COCO-category distractors);
  * natural-omission rate.

The gate is D2's own, frozen *before* the audit runs (the GPT brief; stricter than
the FineCops ``0.90 / 0.80`` default and deliberately not :func:`engineering_verdict`):

  * ``recall@0.5 >= 0.95`` -> ``FULL``;
  * ``0.90 <= recall@0.5 < 0.95`` -> ``GRAY``;
  * ``recall@0.5 < 0.90`` -> ``STOP``.

If ``K = 50`` availability is short of the cohort it is reported as-is; nothing is
re-sampled or padded.

    conda activate deepminer
    python scripts/run_d2_phase2_feasibility.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.data.bank import read_bank_image  # noqa: E402
from ccg.data.types import ProposalBank  # noqa: E402
from ccg.external import feasibility as feas  # noqa: E402
from ccg.external import refcoco_lang as L  # noqa: E402
from ccg.external import refcocog_manifests as rman  # noqa: E402

DEFAULT_OUT = Path("results/v2_d2_refcoco_lang/phase2")
PHASE1_COHORT = Path("results/v2_d2_refcoco_lang/phase1/cohort_audit.json")

#: D2's frozen feasibility gate (the GPT brief), NOT the FineCops engineering gate.
FULL_RECALL_MIN = 0.95
GRAY_RECALL_MIN = 0.90


def _gate(recall_at_05: float) -> Dict[str, Any]:
    if not np.isfinite(recall_at_05):
        verdict = "AUDIT INCOMPLETE"
    elif recall_at_05 >= FULL_RECALL_MIN:
        verdict = "FULL"
    elif recall_at_05 >= GRAY_RECALL_MIN:
        verdict = "GRAY"
    else:
        verdict = "STOP"
    return {
        "criteria": {"full_recall_min": FULL_RECALL_MIN, "gray_recall_min": GRAY_RECALL_MIN},
        "measured_recall_at_05": recall_at_05,
        "verdict": verdict,
        "note": "proposal generator frozen; a STOP is reported, never answered by tuning the RPN",
    }


def audit_all_expressions(
    expressions: List[L.LangExpression],
    *,
    bank_path: Path,
    gt_index: Any,
    log,
) -> List[feas.ProposalAuditRow]:
    """Run the frozen proposal audit over every expression (target-present or not)."""
    rows: List[feas.ProposalAuditRow] = []
    bank_cache: Dict[int, ProposalBank] = {}
    gt_cache: Dict[int, Any] = {}
    for i, expr in enumerate(expressions):
        image_id = int(expr.image_id)
        bank = bank_cache.get(image_id)
        if bank is None:
            boxes, objectness = read_bank_image(Path(bank_path), image_id)
            bank = ProposalBank(image_id=image_id, boxes=boxes, objectness=objectness)
            bank_cache[image_id] = bank
            gt_boxes, gt_cats = rman.read_gt_basis(gt_index, image_id, basis=rman.GT_BASIS_PRIMARY)
            gt_cache[image_id] = (gt_boxes, gt_cats)
        gt_boxes, gt_cats = gt_cache[image_id]
        # audit_expression applies assign_gt_category internally, so it needs the
        # per-GT-object boxes + COCO category ids (aligned), NOT per-bank categories.
        rows.append(
            feas.audit_expression(
                expr, bank,
                gt_object_boxes_xyxy=gt_boxes,
                gt_object_name_codes=gt_cats,
                Ks=L.C1_KS,
            )
        )
        if (i + 1) % 2000 == 0:
            log(f"[d2-p2] audited {i + 1}/{len(expressions)}")
    return rows


def write_per_expression(rows: List[feas.ProposalAuditRow], path: Path) -> None:
    bodies = [r.to_row() for r in rows]
    if not bodies:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(bodies[0].keys()))
        writer.writeheader()
        writer.writerows(bodies)


def stage_all(out_dir: Path, log) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = L.load_image_sizes_map()
    expressions = L.load_refcoco_test_expressions(sizes=sizes)
    image_ids = sorted({int(e.image_id) for e in expressions})
    gt_index = L.official_gt_index(image_ids)
    log(f"[d2-p2] auditing {len(expressions)} expressions on {len(image_ids)} images")

    rows = audit_all_expressions(expressions, bank_path=L.FROZEN_BANK, gt_index=gt_index, log=log)
    write_per_expression(rows, out_dir / "feasibility_per_expression.csv")

    summary = feas.summarise_rows(rows, Ks=L.C1_KS)
    recall_05 = float(summary.get("recall_at_05", {}).get("rate", float("nan")))
    gate = _gate(recall_05)

    report = {
        "protocol": "V2-D2 Phase 2 (proposal feasibility, frozen N=64 RPN)",
        "n_images": len(image_ids),
        "n_expressions": len(expressions),
        "Ks": list(L.C1_KS),
        "summary": summary,
        "gate": gate,
        "note": (
            "same_name_* keys carry same-COCO-category supply (RefCOCO/RefCOCO+ share "
            "the COCO ontology, unlike FineCops' GQA names); 'share_ge_4' is the K=5 "
            "hard-cohort constructibility rate reused for C4"
        ),
    }
    (out_dir / "feasibility.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"[d2-p2] recall@0.5={recall_05:.4f} -> gate {gate['verdict']}; "
        f"K50 availability={summary.get('k50_availability', {}).get('rate', float('nan')):.4f}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    stage_all(Path(args.out_dir), print)
    print(f"[d2-p2] done in {round(time.time() - started, 1)}s -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
