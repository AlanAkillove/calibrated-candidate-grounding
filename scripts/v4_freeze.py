"""Seal V4 stages; never write to V3 or repair evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v4_targeted_strengthening'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def seal(stage, paths):
    target = OUT / f'{stage}_freeze.json'
    if target.exists():
        raise FileExistsError(f'freeze already exists: {target}')
    files = []
    for name in sorted(set(paths)):
        path = (ROOT / name).resolve()
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(dict(path=relative, sha256=digest(path), size_bytes=path.stat().st_size))
    payload = dict(schema='v4-stage-freeze-v1', stage=stage,
                   created_utc=datetime.now(timezone.utc).isoformat(), files=files,
                   baseline='a203744cba2a44b12514012a966a753f028ab1e2')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(stage=stage, n_files=len(files), sha256=digest(target))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['p0', 'p1', 'c_pilot', 'c_final'], required=True)
    parser.add_argument('--paths-json', type=Path)
    args = parser.parse_args()
    if args.stage == 'p0':
        audit = json.loads((OUT / 'baseline_audit_initial.json').read_text(encoding='utf-8'))
        if audit['errors']:
            raise AssertionError('baseline audit errors')
        config = dict(schema='v4-config-v1', baseline='a203744cba2a44b12514012a966a753f028ab1e2',
            backbones=['b0', 'b16'], scorer_seeds=[1,2,3],
            development_status='result-driven follow-up on already-observed FineCops',
            train_images=523, tune_images=224, train_K=[5,10], C=[0.1,1,10], selection_tie_tolerance=.002,
            reuse_frozen_controls=['S','S+Q'], new_groups=['S+V_pure','Full_pure'],
            formal_controlled_images=2888, formal_controlled_expressions=7004,
            coverages=[.2,.3,.4,.5,.6,.7,.8,.9],
            ambiguity='nn_cos_mean', ambiguity_source='image-balanced source train controlled K50',
            bootstrap=dict(draws=5000, seed=0, unit='image', marginal_ci=.95,
                           family_size=6, adjusted_ci=1-.05/6, estimator='mean within-seed effects'),
            primary=[f'{work}_{backbone}' for work in ['A_delta_AURC','A4_high_low_delta_AUROC','B_pure_delta_AUROC'] for backbone in ['b0','b16']],
            secondary_controlled_K=[5,10,20], secondary_natural_K=[5,20,50],
            gpu_budget_hours=20, gpu_warning_hours=12, blas_threads=2,
            prohibited=['new architecture','new seed','feature rescue','post-result dataset switching','extra heterogeneity'],
            external_initial_status='NO_GO pending actual source/domain/access/permission/size gates',
            external_priority=['ReferIt','RefEgo','Talk2Car','Flickr30k'],
            external_audit_images=400, external_seed=20261005,
            external_gate=dict(bank_recall_05=.70, controlled_usable_fraction=.50,
                               controlled_audit_images=200, formal_images_min=500, formal_expressions_min=1000))
        path = OUT / 'config.json'
        with path.open('x', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            f.write('\n')
        paths = ['reviews/v4_protocol.md','reviews/v4_feature_dependency_audit.md',
                 'src/ccg/v4/pure_features.py','src/ccg/v4/practical.py',
                 'tests/test_v4_pure_features.py','tests/test_v4_practical.py',
                 'scripts/v4_freeze.py','scripts/v4_audit_baseline.py',
                 'results/v4_targeted_strengthening/config.json',
                 'results/v4_targeted_strengthening/baseline_audit_initial.json']
    else:
        if args.paths_json is None:
            raise ValueError('later stages require explicit paths JSON')
        paths = json.loads(args.paths_json.read_text(encoding='utf-8'))
    seal(args.stage, paths)

if __name__ == '__main__':
    main()
