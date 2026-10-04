"""Export per-seed intervals and theory tables from the saved formal distributions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ccg.repairs.mechanism import _write_csv, _write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research_repair_v1/theory_analysis/v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    manifest = json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    assert manifest['status']=='COMPLETE' and manifest['kind']=='FORMAL'
    assert manifest['n_replicates']==5000 and manifest['bootstrap_seed']==0
    for row in manifest['outputs']:
        assert sha(OUT/row['path'])==row['sha256'],row['path']
    for row in manifest['analysis_sources']:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    ci=read(OUT/'bootstrap_ci.csv')
    points=read(OUT/'point_components.csv')
    point_map={(r['family'],r['seed']):r for r in points}
    mean={(r['family'],r['quantity']):r for r in ci}
    per_seed=[]
    with np.load(OUT/'raw_replicates.npz',allow_pickle=False) as raw:
        for row in ci:
            family,key=row['family'],row['quantity']
            for seed in ('seed1','seed2','seed3'):
                values=raw[f'{family}__{key}__{seed}']
                valid=values[np.isfinite(values)]
                low,high=np.quantile(valid,[.025,.975]) if len(valid)>=2 else [float('nan')]*2
                per_seed.append({'family':family,'seed':seed,'quantity':key,
                                 'point':float(point_map[(family,seed)][key]),
                                 'ci_low':float(low),'ci_high':float(high),
                                 'valid_replicates':len(valid),'invalid_replicates':len(values)-len(valid),
                                 'n_replicates':len(values),'bootstrap_seed':0,'ci_level':.95})
                assert len(values)==5000
        # Each mean replicate must be the mean of the aligned three fixed seed effects.
        for row in ci:
            prefix=f"{row['family']}__{row['quantity']}"
            reconstructed=np.mean([raw[prefix+'__'+seed] for seed in ('seed1','seed2','seed3')],axis=0)
            np.testing.assert_allclose(reconstructed,raw[prefix+'__mean3'],atol=1e-12,rtol=0,equal_nan=True)
        # All new identities must hold within each shared draw, not only at the point.
        residuals=[]
        for family in ('RPN','DETR','GDINO'):
            for seed in ('seed1','seed2','seed3','mean3'):
                get=lambda k:raw[f'{family}__{k}__{seed}']
                for p in ('p5','p50'):
                    residuals.append(float(np.max(np.abs(get('label_path_'+p)-get('new_comparison_'+p)-get('removed_comparison_'+p)))))
                residuals.append(float(np.max(np.abs(get('C0')-get('C0_SE')-get('C0_FE')))))
                residuals.append(float(np.max(np.abs(get('C1')-get('C1_SE')-get('C1_SF')))))
                residuals.append(float(np.max(np.abs(get('interaction_I')-get('I_new_comparison')-get('I_removed_comparison')))))
        assert max(residuals)<=1e-12
    _write_csv(OUT/'bootstrap_per_seed_ci.csv',per_seed,list(per_seed[0]))
    def number(f,k,interval=False):
        row=mean[(f,k)]
        result=f"{float(row['point_mean3']):+.6f}"
        return result+f" [{float(row['ci_low_mean3']):+.6f}, {float(row['ci_high_mean3']):+.6f}]" if interval else result
    lines=['### 已有数据支持的路径解释','',
           '| Family | T_N(p5) | T_R(p5) | L0 | T_N(p50) | T_R(p50) | L1 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for f in ('RPN','DETR','GDINO'):
        lines.append('| '+f+' | '+' | '.join(number(f,k) for k in
            ('new_comparison_p5','removed_comparison_p5','label_path_p5',
             'new_comparison_p50','removed_comparison_p50','label_path_p50'))+' |')
    lines+=['','数值为逐 seed 计算后的均值；完整 CI 见配对报告。所有 p5 路径中负的新比较项超过正的移除比较项。',
            'RPN 的 p50 新比较项仍为负，但移除比较项更大，导致 L1 为正。DETR 的 p50 两项均稳定为正。',
            'GDINO 的 p50 移除比较项为正；新比较项点估计为正但区间跨零，不宣称该项稳定为正。','',
            '| Family | delta U_SE（95% CI） | delta U_FE（95% CI） | delta U_SF（95% CI） |',
            '|---|---:|---:|---:|']
    for f in ('RPN','DETR','GDINO'):
        lines.append('| '+f+' | '+' | '.join(number(f,'delta_U_'+pair,True) for pair in ('SE','FE','SF'))+' |')
    lines+=['','RPN/DETR 的 SF 排序改善，但 SE/FE 排序变差；GDINO 三种排序都变差。',
            '这一组间变化通过命题六精确形成 C0/C1 的不同方向。它不是平均置信度变化，也不是内部原因的唯一识别。',
            '尤其是 GDINO：I 为正并不表示 SF 排序改善，而是 SF 比 SE 下降得慢；正交互不能直接写成语义协同增强。','',
            '![按结果组划分的配对排序](../results/research_repair_v1/theory_analysis/v1/figure_pair_ranking.png)',
            '', '图中每个配对把第一个组作为分析正组、第二个组作为分析负组；这只是 U_AB 的比较方向。',
            '它不表示两种 K 下的真实正确性标签始终相反，例如 SF 在小 K 下都属于正确预测。']
    doc=ROOT/'docs/theory_analysis_v1.md'
    content=doc.read_text(encoding='utf-8')
    start='<!-- GENERATED_INTERPRETATION_START -->'; end='<!-- GENERATED_INTERPRETATION_END -->'
    assert content.count(start)==content.count(end)==1
    i=content.index(start)+len(start); j=content.index(end)
    doc.write_text(content[:i]+'\n\n'+'\n'.join(lines)+'\n\n'+content[j:],encoding='utf-8',newline='\n')
    integration={'status':'COMPLETE','role':'derived exports only; no new resampling or model evaluation',
                 'analysis_manifest_sha256':sha(OUT/'manifest.json'),
                 'source_script_sha256':sha(Path(__file__)),
                 'per_seed_rows':len(per_seed),'all_seed_draws_valid':all(r['invalid_replicates']==0 for r in per_seed),
                 'maximum_identity_residual_over_all_saved_draws':max(residuals),
                 'outputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in
                            (OUT/'bootstrap_per_seed_ci.csv',doc)]}
    _write_json(OUT/'integration_manifest.json',integration)
    print(json.dumps(integration,ensure_ascii=False))


if __name__=='__main__':
    main()
