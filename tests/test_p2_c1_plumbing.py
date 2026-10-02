"""V2-P2-C1 commit-B plumbing tests: extractor + FAMILIES entry + CLI wiring.

Every expected value is read from the result-before-config freeze
(results/v2_proposal_robustness/p2_c1_gdino_config_freeze.json), never
re-specified here, so the plumbing cannot silently drift from the frozen
protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FREEZE = ROOT / "results" / "v2_proposal_robustness" / "p2_c1_gdino_config_freeze.json"


def _freeze() -> dict:
    return json.loads(FREEZE.read_text(encoding="utf-8"))


def test_freeze_file_exists_and_is_the_c1_only_scope():
    f = _freeze()
    assert f["protocol"] == "V2-P2-C1"
    assert any("C4" in s or "hard" in s for s in f["scope"]["explicitly_out_of_scope"])


def test_core_families_gdino_entry_matches_the_freeze():
    import p1_replication_core as core

    assert "GDINO" in core.FAMILIES
    entry = core.FAMILIES["GDINO"]
    fz = _freeze()["gdino_specific_freeze"]
    assert entry["features_root"] == Path(fz["features"]["path"])
    assert entry["manifests_root"] == Path(fz["manifests"]["path"])
    assert entry["bank"] == Path(fz["bank"]["path"])
    # the two frozen families are untouched
    assert core.FAMILIES["RPN"]["bank"] == Path("cache/proposals.h5")
    assert core.FAMILIES["DETR"]["bank"] == Path("cache/proposals_detr_r50.h5")


def test_frozen_inference_cli_knows_gdino_but_both_stays_rpn_detr():
    import p1_frozen_inference as inf

    p = inf.build_parser()
    assert p.parse_args(["--family", "GDINO"]).family == "GDINO"
    assert p.parse_args([]).family == "both"
    src = Path(inf.__file__).read_text(encoding="utf-8")
    # "both" semantics must remain exactly the P1 pair (GDINO never sneaks in)
    assert '"RPN", "DETR"]' in src


def test_frozen_inference_skips_gdino_hard_jobs():
    """The driver must never call build_hard_samples for GDINO (C1-only freeze)."""
    import p1_frozen_inference as inf

    src = Path(inf.__file__).read_text(encoding="utf-8")
    guard = src.split('if family == "GDINO"')[1].split("hard = core.build_hard_samples")[0]
    assert "continue" in guard
    assert "build_hard_samples" not in guard


def test_extractor_defaults_match_the_freeze():
    import p2_extract_gdino_proposals as ex

    fz = _freeze()["gdino_specific_freeze"]
    assert str(ex.DEFAULT_OUT) == str(ROOT / fz["bank"]["path"]).replace("\\", "/") or \
        Path(ex.DEFAULT_OUT) == ROOT / fz["bank"]["path"]
    assert Path(ex.IDENTITY_PATH) == ROOT / fz["checkpoint"]["identity_file"]
    assert ex.DEFAULT_TOP_N == int(fz["bank"]["n"])
    assert Path(ex.DEFAULT_MANIFEST) == ROOT / "data" / "full_image_manifest.csv"


def test_extractor_reads_identity_from_the_pinned_json():
    import p2_extract_gdino_proposals as ex

    identity = json.loads(Path(ex.IDENTITY_PATH).read_text(encoding="utf-8"))
    assert ex._identity_revision(identity) == "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    sha = ex._identity_weights_sha(identity)
    assert sha and sha.startswith("5548f844c928c4b6")  # freeze-pinned prefix


def test_extractor_writes_nothing_but_the_frozen_gdino_score_definition():
    import p2_extract_gdino_proposals as ex

    src = Path(ex.__file__).read_text(encoding="utf-8")
    assert "max class-token sigmoid prob" in src          # frozen score definition
    assert "detr.extract_detr_proposals" not in src        # generator is gdino only
    assert "from ccg.data import gdino" in src


def test_no_fit_path_in_p2_extractor():
    """Zero-fit discipline (mirrors tests/test_p2_gdino.py static scan)."""
    import p2_extract_gdino_proposals as ex

    src = Path(ex.__file__).read_text(encoding="utf-8")
    for needle in (".fit(", "fit_transform", "requires_grad_(True)", ".train()",
                   "backward(", "optimizer"):
        assert needle not in src, f"forbidden fit needle {needle!r} in extractor"
