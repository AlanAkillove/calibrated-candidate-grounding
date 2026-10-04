from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ccg.repairs.mechanism import (
    _analysis_status,
    _write_family_artifacts,
    derive_paths_from_corners,
    four_corner_paths,
)


def test_four_corner_signed_paths_interaction_and_shapley_identity() -> None:
    # AUROC is 1 for perfect separation and 0 for reversed separation.
    result = four_corner_paths(
        p5=[0.1, 0.9, 0.2, 0.8], r5=[0, 1, 0, 1],
        p50=[0.9, 0.1, 0.8, 0.2], r50=[1, 0, 1, 0],
    )
    assert result["A00"] == 1.0
    assert result["A10"] == 0.0
    assert result["A01"] == 0.0
    assert result["A11"] == 1.0
    assert result["L_at_p5"] == -1.0
    assert result["L_at_p50"] == 1.0
    assert result["C_at_r5"] == -1.0
    assert result["C_at_r50"] == 1.0
    assert result["interaction_I"] == 2.0
    assert result["shapley_label_delta"] == 0.0
    assert result["shapley_confidence_delta"] == 0.0
    assert result["shapley_identity_residual"] == 0.0


def test_undefined_corner_only_invalidates_derived_estimands_that_need_it() -> None:
    result = derive_paths_from_corners(float("nan"), 0.7, 0.6, 0.8)
    assert math.isnan(result["A00"])
    assert math.isnan(result["L_at_p5"])
    assert math.isnan(result["C_at_r5"])
    assert result["L_at_p50"] == pytest.approx(0.2)
    assert result["C_at_r50"] == pytest.approx(0.1)
    assert math.isnan(result["interaction_I"])
    assert math.isnan(result["shapley_label_delta"])
    assert math.isnan(result["shapley_confidence_delta"])
    assert math.isnan(result["shapley_identity_residual"])


def test_four_corner_rejects_nonbinary_labels_and_misaligned_rows() -> None:
    with np.testing.assert_raises_regex(ValueError, "identical row counts"):
        four_corner_paths([0.1], [0], [0.1, 0.2], [0, 1])
    with np.testing.assert_raises_regex(ValueError, "binary"):
        four_corner_paths([0.1, 0.2], [0, 2], [0.1, 0.2], [0, 1])


def test_partial_invalid_family_artifacts_keep_nan_replicates_and_never_claim_formal_complete(
    tmp_path,
) -> None:
    corners = derive_paths_from_corners(float("nan"), 0.7, 0.6, 0.8)
    assert corners["L_at_p50"] == pytest.approx(0.2)
    assert math.isnan(corners["A00"])
    assert math.isnan(corners["interaction_I"])
    summary = {
        "schema": "research-repair-v1-mechanism-family-summary-v1",
        "status": "FAMILY_COMPLETE",
        "analysis_kind": "PILOT",
        "family": "RPN",
        "estimates": {
            "A00": {"point": corners["A00"], "valid_replicates": 0, "invalid_replicates": 4},
            "L_at_p50": {"point": corners["L_at_p50"], "valid_replicates": 4, "invalid_replicates": 0},
        },
    }
    _write_family_artifacts(
        tmp_path, "RPN", summary=summary,
        corner_rows=[{
            "family": "RPN", "seed": "mean3", "n_rows": 4, "n_images": 2,
            **corners, "S": 1, "F": 1, "E": 1, "G": 0,
        }],
        ci_rows=[{
            "family": "RPN", "estimate": "A00", "point_mean3": corners["A00"],
            "ci_low_mean3": None, "ci_high_mean3": None,
            "valid_replicates": 0, "invalid_replicates": 4,
        }],
        raw_replicates={
            "RPN__A00__mean3": np.asarray([np.nan, np.nan, np.nan, np.nan]),
            "RPN__L_at_p50__mean3": np.asarray([0.2, 0.1, 0.3, 0.2]),
        },
    )
    family_summary_path = tmp_path / "families/RPN/summary.json"
    written = json.loads(family_summary_path.read_text(encoding="utf-8"))
    assert written["status"] == "FAMILY_COMPLETE"
    assert written["analysis_kind"] == "PILOT"
    assert written["estimates"]["A00"]["point"] is None
    assert written["estimates"]["A00"]["valid_replicates"] == 0
    assert written["estimates"]["A00"]["invalid_replicates"] == 4
    assert written["estimates"]["L_at_p50"]["point"] == pytest.approx(0.2)
    assert _analysis_status(formal_protocol=False, all_families_complete=True) == "PILOT"
    assert _analysis_status(formal_protocol=True, all_families_complete=False) == "PARTIAL"
    assert _analysis_status(formal_protocol=True, all_families_complete=True) == "COMPLETE"
    with np.load(tmp_path / "families/RPN/bootstrap_raw_replicates.npz", allow_pickle=False) as archive:
        assert np.isnan(archive["RPN__A00__mean3"]).all()
        assert np.allclose(archive["RPN__L_at_p50__mean3"], [0.2, 0.1, 0.3, 0.2])
