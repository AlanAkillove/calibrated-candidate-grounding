"""P1 frozen inference - produce the per-row predictions P1-F4 / P1-F5 analyse.

One read-only pass over the frozen V1 B0 stack scores **both** proposal families
(RPN reference + DETR proposal-B) on the candidate constructions the two confirmatory
stages need:

* ``random_k5`` / ``random_k10`` / ``random_k20`` / ``random_k50`` - the nested
  seeded-random cohorts (P1-A0: the primary random regime is the exact V1
  seeded-random construction, *not* DETR confidence top-K);
* ``hard_k5`` - the same-category cohort (target + 4 same-GT-category distractors).

For every ``(family, job, scorer)`` it writes a prediction npz aligned by
``sentence_id`` with ``conf_msp`` (temperature-scaled B3), ``conf_stats`` (frozen R1),
``conf_e1b`` (frozen E1b), the V2 14-d backbone-neutral ``sem14`` (manipulation check),
and the calibration-independent ``raw_top1`` / ``raw_margin12`` stat columns (P1-F4
secondary diagnostic).  The three seeds are scored separately.

Nothing is fitted: the whole loop runs inside
``ccg.semantic.frozen_load.fit_is_forbidden()`` and this file is scanned by
``assert_no_fit_path``.  The RPN reproduction of the frozen Phase-0B logits is verified
at every nested K (the A8.4 STOP guarantee).

Run::

    conda activate deepminer
    python scripts/p1_frozen_inference.py                 # both families (default)
    python scripts/p1_frozen_inference.py --family DETR   # DETR only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg.external import frozen_identity as fi  # noqa: E402
from ccg.semantic import frozen_load as fl  # noqa: E402
from ccg.semantic import hard_scores as hscores  # noqa: E402

from scripts import p1_replication_core as core  # noqa: E402
import scripts.p1_f3_route_f_compatibility as f3  # noqa: E402

DEFAULT_OUT = Path("results/v2_proposal_robustness")
RANDOM_JOBS: Dict[str, int] = {
    "random_k5": 5, "random_k10": 10, "random_k20": 20, "random_k50": 50,
}
HARD_JOB = "hard_k5"
HARD_K = 5


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [p1-infer] {msg}", flush=True)


def score_job(
    family: str,
    regime: str,
    k: int,
    job: str,
    records,
    *,
    models,
    bundle,
    pred_dir: Path,
    batch_size: int,
    verify_rpn: bool,
) -> Dict[str, Any]:
    _log(f"{family}/{job}: cohort rows={len(records)}")
    per_seed = core.score_cohort(
        family, regime, k, records, models=models, bundle=bundle, batch_size=batch_size
    )
    written = core.write_predictions(pred_dir, family, job, per_seed)

    report: Dict[str, Any] = {
        "family": family, "regime": regime, "job": job, "K": int(k),
        "n_rows": int(len(records)),
        "images": int(len({int(r.image_id) for r in records})),
        "refs": int(len({int(r.ref_id) for r in records})),
        "n_predictions_written": len(written),
    }
    if verify_rpn and regime == "random":
        report["rpn_reproduction_verify"] = core.verify_rpn_reproduction(
            family, regime, k, records, models, bundle, batch_size=batch_size
        )
        _log(f"{family}/{job}: RPN reproduction max|delta|="
             f"{max(report['rpn_reproduction_verify']['max_abs_delta'].values()):.2e}")
    return report


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    _log("verifying frozen identity of the encoder + bundle")
    fi.assert_openclip_identity(core.FAMILIES["RPN"]["features_root"])
    bundle = fl.load_models(args.frozen_models, verify_checksum=True)
    bundle.check_feature_contract()

    _log("static zero-fit scan of the replication core + this driver")
    fl.assert_no_fit_path([Path(__file__), ROOT / "scripts" / "p1_replication_core.py"])

    models = hscores.load_frozen_scorers(
        b3_root=core.B3_ROOT, seeds=core.B3_SEEDS, device=args.device
    )
    prepared_text = f3.prepare_detr_text_cache()
    _log(f"DETR text cache reused: {prepared_text['text_features_sha256'][:12]}")
    identity = f3.frozen_identity_check(bundle)

    families = [args.family] if args.family != "both" else ["RPN", "DETR"]
    reports: List[Dict[str, Any]] = []
    with fl.fit_is_forbidden() as tripwire:
        for family in families:
            job_report: Dict[str, Any] = {"family": family, "attrition": {}, "jobs": []}
            # nested random cohorts + attrition
            c = core.build_corpus(family, "random")
            try:
                job_report["attrition"]["random"] = core.attrition_report(c, "random")
                rand_records = {k: core.cohort_records(c, k) for k in core.KS}
            finally:
                c.close()
            for job, k in RANDOM_JOBS.items():
                job_report["jobs"].append(score_job(
                    family, "random", k, job, rand_records[k],
                    models=models, bundle=bundle, pred_dir=pred_dir,
                    batch_size=args.batch_size,
                    verify_rpn=bool(args.verify_rpn and family == "RPN"),
                ))
            # same-category (hard) cohort built from the manifests
            hard = core.build_hard_samples(family)
            job_report["attrition"]["same_category"] = hard["attrition"]
            job_report["jobs"].append(score_job(
                family, "same_category", HARD_K, HARD_JOB, hard["hard_samples"],
                models=models, bundle=bundle, pred_dir=pred_dir,
                batch_size=args.batch_size,
                verify_rpn=False,
            ))
            reports.append(job_report)

    if tripwire.hits:
        raise AssertionError(f"a forbidden fit fired during inference: {tripwire.hits}")

    payload = {
        "artifact": "v2_p1_frozen_inference",
        "protocol": "V2-P1 P1-F4/P1-F5 frozen inference (both proposal families)",
        "families": reports,
        "detr_text_cache": prepared_text,
        "frozen_identity": identity,
        "new_training_parameters": 0,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "inference_report.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    _log(f"wrote inference_report.json -> {out_dir}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P1 frozen inference (nested-K + hard/random)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--frozen-models", type=Path, default=core.FROZEN_MODELS)
    p.add_argument("--family", default="both", choices=["both", "RPN", "DETR"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--verify-rpn", action="store_true", default=True)
    p.add_argument("--no-verify-rpn", dest="verify_rpn", action="store_false")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
