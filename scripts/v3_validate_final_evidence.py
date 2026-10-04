"""Read-only scientific acceptance of actual V3 predictions and bootstrap arrays."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ccg.v3.protocol import sha256
from ccg.v3.statistics import load_prediction_grid, SEEDS, KS
from ccg.v3.candidates import load_cohort, box_iou_one_to_many
from ccg.v3.inference import _load_bank_data, load_candidate_manifest


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_distribution(folder):
    summary = read(folder / 'summary.json')
    assert summary['status'] == 'COMPLETE' and summary['n_replicates'] == 5000 and summary['seed'] == 0
    assert sha256(folder / summary['raw_draws']['path']) == summary['raw_draws']['sha256']
    invalid = 0
    with np.load(folder / summary['raw_draws']['path']) as draws:
        for name, keys in summary['raw_draws']['keys'].items():
            item = summary['estimates'][name]
            values = draws[keys['mean']]
            assert values.shape == (5000,)
            per_seed = np.stack([draws[keys['per_seed'][seed]] for seed in SEEDS])
            assert np.allclose(values, per_seed.mean(axis=0), atol=1e-14, rtol=0, equal_nan=True), name
            valid = values[np.isfinite(values)]
            assert item['invalid_replicates'] == 5000-len(valid), name
            invalid += item['invalid_replicates']
            if len(valid) >= 2:
                assert np.allclose([item['ci_low'], item['ci_high']], np.quantile(valid, [.025, .975]), atol=1e-14, rtol=0), name
                if 'bonferroni_9875' in item:
                    ci = item['bonferroni_9875']
                    assert np.allclose([ci['ci_low'], ci['ci_high']], np.quantile(valid, [.00625,.99375]), atol=1e-14, rtol=0), name
            points = [item['per_seed'][seed]['point'] for seed in SEEDS]
            if all(point is not None for point in points):
                assert np.isclose(item['point'], np.mean(points), atol=1e-14, rtol=0), name
                assert np.isclose(item['seed_standard_deviation'], np.std(points, ddof=1), atol=1e-14, rtol=0), name
        if folder.name == 'natural':
            for backbone in ('b0', 'b16'):
                for k in KS:
                    prefix = f'{backbone}/K{k}'
                    names = [f'{prefix}/paired_overall/Full_minus_SQ/auroc_correct',
                             f'{prefix}/ranking_audit/covered_ranking_contribution',
                             f'{prefix}/ranking_audit/uncovered_ranking_contribution']
                    for seed in (None, *SEEDS):
                        arrays = [draws[summary['raw_draws']['keys'][name]['mean' if seed is None else 'per_seed']]
                                  if seed is None else draws[summary['raw_draws']['keys'][name]['per_seed'][seed]] for name in names]
                        finite = np.isfinite(np.stack(arrays)).all(axis=0)
                        assert np.allclose(arrays[0][finite], arrays[1][finite]+arrays[2][finite], atol=1e-12, rtol=0), (prefix,seed)
    return summary, {'estimates':len(summary['estimates']), 'total_invalid_estimate_draws':invalid,
                     'summary_sha256':sha256(folder/'summary.json'), 'draws_sha256':summary['raw_draws']['sha256']}


def main():
    base = ROOT / 'results/v3_final_validation'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-bank',type=Path,default=base/'pipeline/confirmation/frozen_proposals.h5',
                        help='Byte-identical saved bank; no detector runs and no image pixels are required')
    args=parser.parse_args()
    ledger = read(base/'confirmation_run.json')
    assert ledger['status'] in ('INFERENCE_COMPLETE','COMPLETE')
    assert ledger['freeze_sha256'] == sha256(base/'protocol_freeze.json')
    assert not ledger['corrections'], 'Original confirmation must be distinguished from a correction rerun'
    predictions = ROOT / ledger['predictions']['path']
    assert sha256(predictions) == ledger['predictions']['sha256']
    index = read(base/'pipeline/confirmation/bootstrap/index.json')
    assert index['status']=='COMPLETE' and index['predictions_sha256']==sha256(predictions)
    formal = base/'pipeline/confirmation/bootstrap'
    controlled,c_checks = validate_distribution(formal/'controlled')
    natural,n_checks = validate_distribution(formal/'natural')
    order,images,grid = load_prediction_grid(predictions)
    cohort = load_cohort(base/'exposure/confirmation_cohort_manifest.jsonl',expected_split='val',root=ROOT)
    freeze = read(base/'protocol_freeze.json')
    bank_identity = next(item for item in freeze['inputs'] if item['path']=='cache/v3/confirmation/proposals.h5')
    assert sha256(args.proposal_bank)==bank_identity['sha256']
    bank = _load_bank_data(args.proposal_bank)
    assignments = load_candidate_manifest(base/'pipeline/confirmation/candidate_manifest.jsonl',cohort)
    ious = {row.record_id:box_iou_one_to_many(row.gt_boxxyxy,bank[row.image_id][0]) for row in cohort}
    seen_logits = set()
    with np.load(predictions.parent/'scorer_logits.npz') as packed:
        logits,offsets = packed['logits'],packed['offsets']
        assert len(offsets)==48*len(order)+1 and offsets[0]==0 and offsets[-1]==len(logits)
        for cells in grid.values():
            for record,row in cells.items():
                pointer=int(row['raw_logits_row'])
                assert pointer not in seen_logits
                seen_logits.add(pointer)
                candidate_ids=np.asarray(row['candidate_indices'],int)
                assert np.array_equal(candidate_ids,assignments[record][row['mode']][int(row['requested_k'])])
                scores=logits[offsets[pointer]:offsets[pointer+1]]
                assert len(scores)==row['k_eff']==len(candidate_ids)
                expected_covered=int((ious[record][candidate_ids]>=.5).sum())
                assert expected_covered==row['presented_valid_target_count']
                assert bool(expected_covered)==bool(row['target_coverage'])
                if len(scores):
                    winner=int(row['winner_proposal_index'])
                    position=int(np.flatnonzero(candidate_ids==winner)[0])
                    assert scores[position]==scores.max(), 'Recorded winner is not maximal in the saved score trace'
                    assert position==int(np.argmax(scores)), 'Recorded winner violates frozen stable float32 tie rule'
                    assert bool(row['correct'])==bool(ious[record][winner]>=.5)
                    assert np.isclose(row['winner_iou'],ious[record][winner],atol=1e-12,rtol=0)
                else:
                    assert row['correct'] is None and row['winner_proposal_index'] is None
    assert seen_logits==set(range(48*len(order)))
    common = [record for record in order if all(grid['controlled',b,s,50][record]['k_eff']==50 and
              grid['controlled',b,s,50][record]['target_present'] and not grid['controlled',b,s,50][record].get('missing_reason')
              for b in ('b0','b16') for s in SEEDS)]
    assert len(common)==controlled['flow']['common_expressions']
    exclusion_reasons = Counter(str(grid['controlled','b0',SEEDS[0],50][record].get('missing_reason'))
                                for record in order if record not in set(common))
    assert sum(exclusion_reasons.values())==controlled['flow']['excluded_expressions']
    accuracy = {}
    for backbone in ('b0','b16'):
        for seed in SEEDS:
            previous = None
            for k in KS:
                labels = np.array([grid['controlled',backbone,seed,k][r]['correct'] for r in common],bool)
                if previous is not None:
                    assert not np.any(~previous & labels), (backbone,seed,k,'impossible controlled 0-to-1 correctness transition')
                previous = labels
                accuracy.setdefault(f'{backbone}/K{k}',{})[seed]=float(labels.mean())
        for k in KS:
            flow = natural['flow'][f'{backbone}/b3_seed1/K{k}']
            assert flow['n_all']==len(order) and sum(flow['actual_k_histogram'].values())==len(order)
            for seed in SEEDS:
                other = natural['flow'][f'{backbone}/{seed}/K{k}']
                assert all(other[key]==flow[key] for key in ('n_all','n_covered','n_empty','n_basic_only','actual_k_histogram'))
    tests = read(base/'tests_before_confirmation.json')
    assert tests['status']=='PASS' and tests['exit_code']==0 and not tests['changed_during_tests']
    assert all(sha256(ROOT/path)==digest for path,digest in tests['tested_source_inputs'].items())
    legacy = read(base/'legacy_inputs_final.json')
    assert legacy['status']=='PASS' and legacy['unique_files']==834 and not legacy['changed']
    source = read(base/'pipeline/confirmation/inference/summary.json')['controlled_primary_point_estimates']
    assert source['n_rows']==len(common)
    named = ('C1_B0_MSP_AUROC_K5_minus_K50','C2_B0_Full_minus_SQ_AUROC_K50',
             'C3_B16_Full_minus_SQ_AUROC_K50','C4_B0_SDS_large_minus_small_AUROC_K50')
    for i,name in enumerate(named,1):
        expected = np.mean([source['seed_contrasts'][seed][name] for seed in SEEDS])
        assert np.isclose(expected,controlled['estimates'][f'C{i}/auroc_correct']['point'],atol=1e-12,rtol=0)
    report={'status':'PASS','verified_utc':datetime.now(timezone.utc).isoformat(),
            'expressions':len(order),'images':int(np.unique(images).size),'prediction_rows':48*len(order),
            'common_expressions':len(common),'controlled':c_checks,'natural':n_checks,
            'raw_logits_pointers_verified':len(seen_logits),'controlled_exclusion_reasons':dict(exclusion_reasons),
            'controlled_accuracy_points':{key:{'mean':float(np.mean(list(value.values()))),'per_seed':value} for key,value in accuracy.items()},
            'freeze_sha256':sha256(base/'protocol_freeze.json'),'predictions_sha256':sha256(predictions),
            'saved_proposal_bank_sha256':sha256(args.proposal_bank),'saved_proposal_bank_original_identity':bank_identity,
            'checks':['complete sentence/seed/configuration grid','all candidate IDs match prescore frozen manifest across seeds/configurations','own winner correctness independently recomputed against GT and saved logits with stable float32 ties','same-draw seed means and both percentile levels',
                      'raw signed ranking identity per seed and draw','controlled nested correctness','whole natural flow',
                      'source point estimators agree','tested source unchanged','834 legacy inputs unchanged'],
            'limits':'Acceptance checks computation and recorded identities, not causal identification, foundation pretraining exposure, or correctness of all source annotations.'}
    (base/'final_acceptance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','expressions','images','prediction_rows','common_expressions','controlled','natural')},indent=2))


if __name__ == '__main__':
    main()
