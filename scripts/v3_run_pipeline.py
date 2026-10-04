#!/usr/bin/env python
"""Prepare and score the frozen V3 FineCops pipeline.

Preparation (RPN/CLIP/candidate identity) and scorer inference are separate
commands. Confirmation preparation requires the parent-issued authorization;
confirmation scoring requires a sealed freeze and the write-once ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ccg.v3.inference import (  # noqa: E402
    CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    EMPTY_PROPOSAL_FILENAME,
    cohort_bbox_boundary_report,
    ensure_feature_caches,
    ensure_proposal_bank,
    frozen_model_paths,
    _load_bank_data,
    prepare_candidates_from_caches,
    run_frozen_inference,
    sha256_file,
    source_code_paths,
    unique_images,
    validate_cohort_geometry,
    validate_config_file,
    validate_exposure_manifest,
    verify_prepare_authorization,
    write_preparation_manifest,
)
from ccg.v3.candidates import REQUESTED_KS, load_cohort  # noqa: E402


def _validate_development_prediction_grid(
    record_ids: list[str], prediction_rows: list[dict], logits: object, offsets: object
) -> dict[str, int]:
    """Check lossless dev output alignment and the complete scorer/mode/K grid."""
    import numpy as np

    values = np.asarray(logits, dtype=np.float32).reshape(-1)
    boundaries = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if boundaries.shape != (len(prediction_rows) + 1,) or boundaries[0] != 0:
        raise ValueError("scorer-logit offsets do not contain one boundary per prediction row")
    if np.any(np.diff(boundaries) < 0) or boundaries[-1] != values.size or not np.isfinite(values).all():
        raise ValueError("scorer-logit artifact has invalid offsets or non-finite values")
    expected = {
        (record_id, mode, k, backbone, f"b3_seed{seed}")
        for record_id in record_ids
        for mode in ("natural", "controlled")
        for k in REQUESTED_KS
        for backbone in ("b0", "b16")
        for seed in (1, 2, 3)
    }
    observed: dict[tuple[str, str, int, str, str], dict] = {}
    identity_by_cell: dict[tuple[str, str, int], list[int]] = {}
    for index, row in enumerate(prediction_rows):
        key = (
            str(row["record_id"]), str(row["mode"]), int(row["requested_k"]),
            str(row["backbone"]), str(row["seed"]),
        )
        if key in observed:
            raise ValueError(f"development predictions repeat cell {key}")
        observed[key] = row
        record_id, mode, requested_k, backbone, seed = key
        if record_id not in record_ids or mode not in {"natural", "controlled"}:
            raise ValueError(f"development prediction has an unexpected source identity or mode: {key}")
        if requested_k not in REQUESTED_KS or backbone not in {"b0", "b16"} or seed not in {
            "b3_seed1", "b3_seed2", "b3_seed3"
        }:
            raise ValueError(f"development prediction is outside the frozen scorer grid: {key}")
        if row.get("source_split") != "train":
            raise ValueError(f"development prediction is not from the official positive train split: {key}")
        if int(row.get("raw_logits_row", -1)) != index:
            raise ValueError(f"raw scorer row pointer is misaligned for {key}")
        k_eff = int(row["k_eff"])
        candidate_ids = [int(value) for value in row["candidate_indices"]]
        if k_eff != len(candidate_ids) or k_eff > requested_k or len(set(candidate_ids)) != k_eff:
            raise ValueError(f"candidate identity width/padding is invalid for {key}")
        if int(boundaries[index + 1] - boundaries[index]) != k_eff:
            raise ValueError(f"raw scorer logits do not losslessly align with candidate identities for {key}")
        if row.get("valid_target_count") != row.get("bank_valid_target_count"):
            raise ValueError(f"bank-level target count compatibility alias changed for {key}")
        if int(row["presented_valid_target_count"]) > int(row["bank_valid_target_count"]):
            raise ValueError(f"presented target count exceeds bank target count for {key}")
        if bool(row["target_coverage"]) != (int(row["presented_valid_target_count"]) > 0):
            raise ValueError(f"presented coverage disagrees with selected valid-target count for {key}")
        if bool(row["bank_target_coverage"]) != (int(row["bank_valid_target_count"]) > 0):
            raise ValueError(f"bank coverage disagrees with bank valid-target count for {key}")
        identity_key = (record_id, mode, requested_k)
        previous = identity_by_cell.setdefault(identity_key, candidate_ids)
        if previous != candidate_ids:
            raise ValueError(f"backbones/seeds did not use shared candidate identities for {identity_key}")
        confidence = row.get("confidence") or {}
        missing = row.get("confidence_missing_reason") or {}
        expected_models = {"MSP", "S", "S+Q", "S+V", "Full"}
        if backbone == "b0":
            expected_models |= {"SDS_small", "SDS_large"}
        if k_eff >= 5:
            if set(confidence) != expected_models:
                raise ValueError(f"scorable cell has incomplete confidence families for {key}: {sorted(confidence)}")
            if any(not np.isfinite(float(confidence[name])) for name in expected_models):
                raise ValueError(f"scorable cell has non-finite confidence for {key}")
        elif confidence:
            raise ValueError(f"K_eff<5 must report basic grounding only for {key}")
        elif set(missing) != expected_models or any(
            value not in {"empty_candidate_set", "k_eff_below_5"} for value in missing.values()
        ):
            raise ValueError(f"basic/empty grounding row lacks explicit confidence-unavailable reasons for {key}")
        if k_eff == 0:
            if row.get("correct") is not None or row.get("winner_proposal_index") is not None:
                raise ValueError(f"empty grounding flow must retain null correctness/winner for {key}")
        elif not isinstance(row.get("correct"), bool) or row.get("winner_proposal_index") not in candidate_ids:
            raise ValueError(f"nonempty grounding row is missing a binary result or candidate winner for {key}")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))[:5]
        extra = sorted(set(observed) - expected)[:5]
        raise ValueError(f"development output grid is incomplete: missing={missing}, extra={extra}")
    for record_id in record_ids:
        for mode in ("natural", "controlled"):
            previous: list[int] | None = None
            for k in REQUESTED_KS:
                current = identity_by_cell[(record_id, mode, k)]
                if previous is not None and current[:len(previous)] != previous:
                    raise ValueError(f"{mode} candidate identities are not nested for {record_id}")
                previous = current
    return {
        "prediction_rows": len(prediction_rows),
        "candidate_cells": len(identity_by_cell),
        "raw_logit_values": int(values.size),
        "empty_candidate_rows": sum(int(row["k_eff"]) == 0 for row in prediction_rows),
        "basic_grounding_only_rows": sum(0 < int(row["k_eff"]) < 5 for row in prediction_rows),
    }


def _write_development_acceptance(
    *, rows: list, cohort_path: Path, config_path: Path, exposure_paths: list[Path],
    candidate_path: Path, bank_path: Path, cache_stage: Path, output: Path,
    anchor_path: Path, preparation_path: Path,
) -> Path:
    """Write the actual PASS report consumed by the parent freeze builder."""
    import numpy as np

    predictions_path = output / "predictions.jsonl"
    logits_path = output / "scorer_logits.npz"
    summary_path = output / "summary.json"
    if not anchor_path.is_file() or not preparation_path.is_file():
        raise FileNotFoundError("development acceptance requires source anchors and a completed preparation manifest")
    prediction_rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with np.load(logits_path, allow_pickle=False) as archive:
        grid = _validate_development_prediction_grid(
            [row.record_id for row in rows], prediction_rows, archive["logits"], archive["offsets"]
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("stage") != "development" or summary.get("split") != "train":
        raise ValueError("development inference summary reports an unexpected stage or split")
    if summary.get("n_expression_rows") != len(rows) or summary.get("n_unique_images") != len(unique_images(rows)):
        raise ValueError("development inference summary has a changed cohort size")
    if summary.get("full_confirmatory_bootstrap") != "not run by V3 pipeline; requires parent resource grant":
        raise ValueError("V3 dev runner must not claim to have run formal confirmation bootstrap")
    boundary_report = summary.get("cohort_boundary_annotation_report", {})
    if (
        boundary_report.get("raw_released_boxes_preserved") is not True
        or boundary_report.get("ground_truth_clipping_or_rescaling") is not False
        or boundary_report.get("n_expression_rows") != len(rows)
    ):
        raise ValueError("development summary is missing the unmodified released-box boundary report")
    grouped = summary.get("summary_by_mode_backbone_seed", {})
    if len(grouped) != 12:
        raise ValueError("development summary must preserve all mode/backbone/seed groups separately")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if anchor.get("status") != "PASS" or not anchor.get("source_only"):
        raise ValueError("source-model replay anchor did not pass as a source-only check")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if (
        preparation.get("stage") != "dev"
        or preparation.get("scorer_forward") is not False
        or preparation.get("reliability_forward") is not False
        or preparation.get("performance_aggregation") is not False
    ):
        raise ValueError("development preparation manifest does not describe a prepare-only phase")

    model_paths = frozen_model_paths()
    code_paths = source_code_paths()
    input_roles: dict[Path, str] = {
        cohort_path.resolve(): "cohort",
        config_path.resolve(): "config",
        candidate_path.resolve(): "candidate_manifest",
        bank_path.resolve(): "proposal_bank",
        (bank_path.parent / EMPTY_PROPOSAL_FILENAME).resolve(): "proposal_flow",
        preparation_path.resolve(): "preparation_manifest",
        predictions_path.resolve(): "predictions",
        logits_path.resolve(): "scorer_logits",
        summary_path.resolve(): "summary",
        anchor_path.resolve(): "source_anchor",
    }
    for path in exposure_paths:
        input_roles[path.resolve()] = "exposure"
    for path in unique_images(rows).values():
        input_roles[path.resolve()] = "image"
    for backbone in ("b0", "b16"):
        root = cache_stage / f"features_{backbone}"
        for name in ("metadata.json", "region_features.h5", "text_features.h5", "invalid_crops.csv"):
            input_roles[(root / name).resolve()] = "feature_cache"
    for path in code_paths:
        input_roles[path.resolve()] = "code"
    encoder_meta = {path.resolve() for path in (ROOT / "cache/features/metadata.json", ROOT / "cache/v2_backbones/openclip_b16/metadata.json")}
    encoder_weights = {
        (ROOT / json.loads(path.read_text(encoding="utf-8"))["backbone"]["checkpoint_path"].replace("\\", "/")).resolve()
        for path in (ROOT / "cache/features/metadata.json", ROOT / "cache/v2_backbones/openclip_b16/metadata.json")
    }
    for path in model_paths:
        input_roles[path.resolve()] = "encoder_model" if path.resolve() in encoder_meta | encoder_weights else "model"
    missing = [str(path) for path in input_roles if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"development acceptance inputs missing: {missing[:8]}")

    def root_relative(path: Path) -> str:
        return path.relative_to(ROOT).as_posix()

    payload = {
        "schema": "v3-development-acceptance-v1",
        "status": "PASS",
        "stage": "dev",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": {"path": root_relative(cohort_path.resolve()), "n_expression_rows": len(rows), "n_images": len(unique_images(rows))},
        "checks": {
            "source_model_anchors": "PASS",
            "development_prepare_only_manifest": "PASS",
            "complete_natural_and_controlled_grid": "PASS",
            "shared_nested_candidate_identities": "PASS",
            "natural_cohort_rows_retained_without_N_or_target_presence_filter": "PASS",
            "released_gt_overhangs_reported_and_unclipped": "PASS",
            "empty_and_basic_grounding_flows_preserved": "PASS",
            "lossless_candidate_to_raw_logit_alignment": "PASS",
            "separate_backbone_seed_mode_summaries": "PASS",
            "confirmation_bootstrap_not_run": "PASS",
        },
        "grid": grid,
        "cohort_boundary_annotation_report": boundary_report,
        "source_anchor": anchor,
        "preparation_sha256": sha256_file(preparation_path),
        "inference_summary_sha256": sha256_file(summary_path),
        "inputs": [
            {"role": role, "path": root_relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path, role in sorted(input_roles.items(), key=lambda pair: root_relative(pair[0]))
        ],
    }
    destination = DEFAULT_OUTPUT_ROOT / "dev_acceptance.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


@contextmanager
def gpu_process_lock(lock_path: Path) -> Iterator[None]:
    """Take a nonblocking cross-platform exclusive lock for the whole prep."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(1)
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\n".encode("ascii"))
        stream.flush()
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _default_cohort(stage: str) -> Path:
    name = "dev_cohort_manifest.jsonl" if stage == "dev" else "confirmation_cohort_manifest.jsonl"
    return ROOT / "results/v3_final_validation/exposure" / name


def _default_exposure(stage: str) -> list[Path]:
    # Keep the input audit manifest explicit on the command line if the
    # exposure pipeline changes its output name; never infer eligibility from
    # a broad folder scan.
    return [ROOT / "results/v3_final_validation/exposure/exposure_cohort_manifest.csv"]


def _under_root(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"{label} must be inside the workspace: {resolved}")
    return resolved


def _prepare(args: argparse.Namespace) -> int:
    stage = args.stage
    if stage not in {"dev", "confirmation"}:
        raise ValueError("prepare stage must be dev or confirmation")
    cohort_path = _under_root(Path(args.cohort or _default_cohort(stage)), "cohort manifest")
    config_path = _under_root(Path(args.config), "V3 config")
    exposure_values = args.exposure or _default_exposure(stage)
    exposure_paths = [_under_root(Path(path), "exposure manifest") for path in exposure_values]
    if not exposure_paths:
        raise ValueError("at least one content-bound image exposure manifest is required")

    authorization_path: Path | None = None
    if stage == "confirmation":
        if not args.authorization:
            raise ValueError("confirmation preparation is refused without a parent-issued authorization")
        authorization_path = _under_root(Path(args.authorization), "preparation authorization")
    expected_split = "train" if stage == "dev" else "val"
    rows = load_cohort(cohort_path, expected_split=expected_split, root=ROOT)
    image_paths = list(unique_images(rows).values())
    if stage == "confirmation":
        assert authorization_path is not None
        # This check happens before opening image pixels, bank or feature
        # caches, and before any RPN/CLIP work.
        verify_prepare_authorization(
            authorization_path,
            config_path=config_path,
            cohort_path=cohort_path,
            exposure_paths=exposure_paths,
            image_paths=image_paths,
        )
    elif args.authorization:
        raise ValueError("an authorization is only accepted for the confirmation preparation stage")

    config = validate_config_file(config_path)
    image_sizes = validate_cohort_geometry(rows)
    exposure_report = validate_exposure_manifest(rows, exposure_paths)
    if stage == "dev":
        images = unique_images(rows)
        if len(images) > int(config["development_max_images"]):
            raise ValueError(
                f"development cohort has {len(images)} images; frozen cap is {config['development_max_images']}"
            )

    cache_stage = _under_root(ROOT / "cache/v3" / stage, "V3 cache root")
    output_stage = _under_root(DEFAULT_OUTPUT_ROOT / stage, "V3 pipeline output")
    bank_path = cache_stage / "proposals.h5"
    candidate_path = output_stage / "candidate_manifest.jsonl"
    feature_roots = {backbone: cache_stage / f"features_{backbone}" for backbone in ("b0", "b16")}

    if args.device.startswith("cuda"):
        try:
            import torch

            torch.set_num_threads(2)
        except Exception as exc:  # pragma: no cover - clear runtime diagnostic
            raise RuntimeError("PyTorch is required for V3 proposal/feature preparation") from exc
        lock = cache_stage.parent / ".gpu_process.lock"
        with gpu_process_lock(lock):
            ensure_proposal_bank(rows, bank_path, device=args.device)
            ensure_feature_caches(rows, bank_path, cache_stage, device=args.device)
    else:
        ensure_proposal_bank(rows, bank_path, device=args.device)
        ensure_feature_caches(rows, bank_path, cache_stage, device=args.device)

    prepare_candidates_from_caches(rows, bank_path, feature_roots, candidate_path)
    manifest_path = write_preparation_manifest(
        output_stage / "preparation_manifest.json",
        stage=stage,
        config_path=config_path,
        cohort_path=cohort_path,
        exposure_paths=exposure_paths,
        image_paths=image_paths,
        bank_path=bank_path,
        feature_roots=feature_roots,
        candidate_path=candidate_path,
        authorization_path=authorization_path,
    )
    result = {
        "status": "PASS",
        "stage": stage,
        "cohort_path": cohort_path.relative_to(ROOT).as_posix(),
        "cohort_sha256": sha256_file(cohort_path),
        "n_expressions": len(rows),
        "n_images": len(unique_images(rows)),
        "exposure_audit": exposure_report,
        "cohort_boundary_annotation_report": cohort_bbox_boundary_report(rows, image_sizes),
        "bank_path": bank_path.relative_to(ROOT).as_posix(),
        "candidate_manifest": candidate_path.relative_to(ROOT).as_posix(),
        "preparation_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "preparation_sha256": sha256_file(manifest_path),
        "scorer_forward": False,
        "reliability_forward": False,
        "performance_aggregation": False,
    }
    print(json.dumps(result, indent=2))
    return 0


def _infer(args: argparse.Namespace) -> int:
    stage = args.stage
    if stage == "confirmation" and (not args.freeze or not args.ledger):
        raise ValueError("confirmation inference is refused without both a parent freeze and run ledger")
    cohort_path = _under_root(Path(args.cohort or _default_cohort(stage)), "cohort manifest")
    config_path = _under_root(Path(args.config), "V3 config")
    exposure_values = args.exposure or _default_exposure(stage)
    exposure_paths = [_under_root(Path(path), "exposure manifest") for path in exposure_values]
    rows = load_cohort(cohort_path, expected_split="train" if stage == "dev" else "val", root=ROOT)
    image_sizes = validate_cohort_geometry(rows)
    cache_stage = _under_root(ROOT / "cache/v3" / stage, "V3 cache root")
    bank_path = _under_root(Path(args.bank or cache_stage / "proposals.h5"), "proposal bank")
    candidate_path = _under_root(Path(args.candidates or DEFAULT_OUTPUT_ROOT / stage / "candidate_manifest.jsonl"), "candidate manifest")
    output = _under_root(Path(args.output or DEFAULT_OUTPUT_ROOT / stage / "inference"), "inference output")
    feature_roots = {backbone: cache_stage / f"features_{backbone}" for backbone in ("b0", "b16")}

    if stage == "dev":
        if args.freeze or args.ledger:
            raise ValueError("development inference does not accept confirmation freeze or ledger paths")
        anchor = __import__("ccg.v3.inference", fromlist=["verify_source_model_anchors"]).verify_source_model_anchors
        anchor_path = output.parent / "source_model_anchors.json"
        anchor_report = anchor()
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        anchor_path.write_text(json.dumps(anchor_report, indent=2) + "\n", encoding="utf-8")
        result = run_frozen_inference(
            rows, candidate_path, bank_path, feature_roots, image_sizes, output,
            cohort_path=cohort_path,
            exposure_paths=exposure_paths,
            config_path=config_path,
        )
        result["source_anchor_path"] = anchor_path.relative_to(ROOT).as_posix()
        result["source_anchor_sha256"] = sha256_file(anchor_path)
        acceptance_path = _write_development_acceptance(
            rows=rows,
            cohort_path=cohort_path,
            config_path=config_path,
            exposure_paths=exposure_paths,
            candidate_path=candidate_path,
            bank_path=bank_path,
            cache_stage=cache_stage,
            output=output,
            anchor_path=anchor_path,
            preparation_path=DEFAULT_OUTPUT_ROOT / "dev" / "preparation_manifest.json",
        )
        print(json.dumps({
            "status": "PASS",
            "stage": "dev",
            "output": output.relative_to(ROOT).as_posix(),
            "summary": result,
            "acceptance": acceptance_path.relative_to(ROOT).as_posix(),
            "acceptance_sha256": sha256_file(acceptance_path),
        }, indent=2, ensure_ascii=False))
        return 0

    result = run_frozen_inference(
        rows, candidate_path, bank_path, feature_roots, image_sizes, output,
        cohort_path=cohort_path,
        allowed_confirmation=True,
        freeze_path=_under_root(Path(args.freeze), "freeze manifest"),
        ledger_path=_under_root(Path(args.ledger), "confirmation run ledger"),
        exposure_paths=exposure_paths,
        config_path=config_path,
    )
    print(json.dumps({"status": "PASS", "stage": "confirmation", "output": str(output), "summary": result}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare frozen proposal, CLIP, and candidate identity caches")
    prepare.add_argument("--stage", choices=("dev", "confirmation"), required=True)
    prepare.add_argument("--cohort", type=Path)
    prepare.add_argument("--exposure", type=Path, action="append", default=[])
    prepare.add_argument("--config", type=Path, default=CONFIG_PATH)
    prepare.add_argument("--authorization", type=Path)
    prepare.add_argument("--device", default="cuda")
    prepare.set_defaults(handler=_prepare)

    infer = subparsers.add_parser("infer", help="run frozen grounding/reliability inference")
    infer.add_argument("--stage", choices=("dev", "confirmation"), required=True)
    infer.add_argument("--cohort", type=Path)
    infer.add_argument("--exposure", type=Path, action="append", default=[])
    infer.add_argument("--config", type=Path, default=CONFIG_PATH)
    infer.add_argument("--bank", type=Path)
    infer.add_argument("--candidates", type=Path)
    infer.add_argument("--output", type=Path)
    infer.add_argument("--freeze", type=Path)
    infer.add_argument("--ledger", type=Path)
    infer.set_defaults(handler=_infer)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"V3 pipeline refused/failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
