"""Analyze existing predictions; no model training, candidates or gates are changed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from ccg.repairs.bootstrap import shared_image_cluster_bootstrap
from ccg.repairs.mechanism import (
    FAMILIES, SEEDS, _read_c1_point_reference, _validate_frozen_manifest,
    _write_csv, _write_json, _write_npz_atomic, load_family_cohort,
)
from ccg.repairs.pairwise_analysis import FrozenPairAccounting

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/research_repair_v1/theory_analysis/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def render_report(out, rows, checks):
    lookup = {(r["family"], r["quantity"]): r for r in rows}
    def val(family, key, ci=False):
        r = lookup[(family, key)]
        text = f"{r['point_mean3']:+.6f}"
        return (text + f" [{r['ci_low_mean3']:+.6f}, {r['ci_high_mean3']:+.6f}]") if ci else text
    text = ["# 已有预测的配对排序分析", "", "这是结果驱动的描述性补充分析。S=两端正确，F=正确转为错误，E=两端错误。",
            "输入为各 family 自己的共同 K50 cohort；不同 family 不是共同样本上的配对比较。",
            "三个固定 seed 的效应均值；共享 image-cluster 抽样，seed=0，95% percentile CI。",
            "本文件所有表由 CSV 生成。分量名称表示指标比较对象，不表示干预机制。", "",
            "## 标签路径的两个有符号项", "",
            "`alpha=nF/(nS+nF)`，`beta=nF/(nF+nE)`。",
            "`L(p)=beta*(U_SF-U_SE)+alpha*(U_SE-U_FE)`。", "",
            "| Family | p | 新比较项 beta*(SF-SE) | 移除比较项 alpha*(SE-FE) | L(p) |",
            "|---|---|---:|---:|---:|"]
    for family in FAMILIES:
        for p in ("p5", "p50"):
            text.append(f"| {family} | {p} | {val(family,'new_comparison_'+p,True)} | "
                        f"{val(family,'removed_comparison_'+p,True)} | {val(family,'label_path_'+p,True)} |")
    text += ["", "## 排序胜率的变化", "",
             "| Family | delta U_SE | delta U_FE | delta U_SF |", "|---|---:|---:|---:|"]
    for family in FAMILIES:
        text.append(f"| {family} | " + " | ".join(val(family,'delta_U_'+pair,True) for pair in ("SE","FE","SF")) + " |")
    text += ["", "## 四角交互的两项分解", "",
             "`I=delta new_comparison + delta removed_comparison`。", "",
             "| Family | delta new_comparison | delta removed_comparison | I |", "|---|---:|---:|---:|"]
    for family in FAMILIES:
        text.append(f"| {family} | " + " | ".join(val(family,k,True) for k in
                    ("I_new_comparison","I_removed_comparison","interaction_I")) + " |")
    text += ["", "## 置信度路径的组间来源", "",
             "| Family | C0_SE | C0_FE | C0 | C1_SE | C1_SF | C1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        text.append(f"| {family} | " + " | ".join(val(family,k,True) for k in
                    ("C0_SE","C0_FE","C0","C1_SE","C1_SF","C1")) + " |")
    text += ["", "## 复核与边界", "", "```json", json.dumps(checks, indent=2, ensure_ascii=False), "```", "",
             "这些组由两个端点的结果定义，不能直接作为预处理因果分层。",
             "平均值是先按 seed 计算再平均，不将平均权重乘平均胜率冒充同一估计量。",
             "每个区间是逐项、条件于既定模型的区间，不是同时覆盖保证；补充分析不是独立确认。"]
    (out / "analysis_report.md").write_text("\n".join(text)+"\n", encoding="utf-8")


def render_figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["family"],r["quantity"]):r for r in rows}
    fig, axes = plt.subplots(1,3,figsize=(11,3.8),sharey=True)
    colors = ["#336699", "#aa5533", "#339977"]
    for ax, family in zip(axes,FAMILIES):
        for index,pair in enumerate(("SE","FE","SF")):
            vals=[lookup[(family,f"U_{pair}_{tag}")] for tag in ("p5","p50")]
            ys=[r["point_mean3"] for r in vals]
            bounds=np.asarray([[r['ci_low_mean3'],r['ci_high_mean3']] for r in vals])
            ax.plot([0,1],ys,marker="o",color=colors[index],label=pair)
            ax.vlines([0,1],bounds[:,0],bounds[:,1],color=colors[index],linewidth=1.5)
        ax.axhline(.5,color="gray",linestyle="--",linewidth=.7)
        ax.set(title=family,xticks=[0,1],xticklabels=["p5","p50"],ylim=(0,1),xlabel="Confidence vector")
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel("Mean pair ranking probability (95% CI)")
    axes[-1].legend(title="Positive vs negative group",fontsize=9,loc="lower left")
    fig.suptitle("S: correct at both K; F: flipped to error; E: persistent error",fontsize=10)
    fig.tight_layout()
    for extension in ("png","svg","pdf"):
        fig.savefig(out/f"figure_pair_ranking.{extension}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates",type=int,default=5000)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    out=args.output.resolve()
    out.relative_to((ROOT/"results/research_repair_v1/theory_analysis").resolve())
    if out.exists():
        raise ValueError("use a new output directory; existing analysis is retained")
    out.mkdir(parents=True)
    frozen=_validate_frozen_manifest()
    reference=_read_c1_point_reference()
    historical=read_csv(ROOT/"results/v2_rq4_mechanism/m1_transition_confidence/pairwise_auc_components.csv")
    repair=read_csv(ROOT/"results/research_repair_v1/mechanism/four_corners_paths_interaction.csv")
    repair_raw=np.load(ROOT/"results/research_repair_v1/mechanism/bootstrap_raw_replicates.npz",allow_pickle=False)
    sources=["scripts/analyze_repair_pairwise.py","src/ccg/repairs/pairwise_analysis.py",
             "src/ccg/repairs/mechanism.py","src/ccg/repairs/bootstrap.py",
             "results/v2_rq4_mechanism/m1_transition_confidence/pairwise_auc_components.csv",
             "results/research_repair_v1/mechanism/four_corners_paths_interaction.csv",
             "results/research_repair_v1/mechanism/bootstrap_raw_replicates.npz"]
    manifest={"kind":"FORMAL" if args.replicates==5000 else "PILOT", "status":"RUNNING",
              "analysis_role":"result-driven descriptive analysis on already observed tests",
              "n_replicates":args.replicates,"bootstrap_seed":0,"ci_level":.95,
              "group_definition":{"S":"r5=1,r50=1","F":"r5=1,r50=0","E":"r5=0,r50=0"},
              "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "frozen_inputs":frozen,"analysis_sources":[{"path":p,"sha256":digest(ROOT/p)} for p in sources],
              "families":{},"created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    _write_json(out/"manifest.json",manifest)
    all_points=[]; all_ci=[]; raw={}; checks={"point_anchor_tolerance":1e-12,"max_point_residual":0.,
        "max_historical_pair_residual":0.,"max_bootstrap_anchor_residual":0.,"G_total":0}
    started=time.monotonic()
    for family in FAMILIES:
        arrays,images,metadata=load_family_cohort(family,reference)
        caches=[FrozenPairAccounting(a["p5"],a["r5"],a["p50"],a["r50"]) for a in arrays]
        points=[cache.evaluate() for cache in caches]
        for seed,point in zip(SEEDS,points):
            original=next(r for r in repair if r['family']==family and r['seed']==f'seed{seed}')
            old=next(r for r in historical if r['family']==family and r['seed']==f'seed{seed}' and r['corner']=='A00')
            for key in ("A00","A10","A01","A11","interaction_I"):
                residual=abs(point[key]-float(original[key]))
                checks['max_point_residual']=max(checks['max_point_residual'],residual)
                assert residual<=1e-12,(family,seed,key,residual)
            for tag in ("p5","p50"):
                for pair in ("SE","FE","SF"):
                    residual=abs(point[f'U_{pair}_{tag}']-float(old[f'AUC_{pair}_{tag}']))
                    checks['max_historical_pair_residual']=max(checks['max_historical_pair_residual'],residual)
                    assert residual<=1e-12
            for group in ("S","F","E"):
                assert point['n'+group]==int(original[group])==int(old['n'+group])
            all_points.append({"family":family,"seed":f"seed{seed}",**point})
        all_points.append({"family":family,"seed":"mean3",**{k:float(np.mean([p[k] for p in points])) for k in points[0]}})
        def statistic(indices):
            weights=np.bincount(indices,minlength=images.size)
            values=[cache.evaluate(weights) for cache in caches]
            return {k:{seed:v[k] for seed,v in zip(SEEDS,values)} for k in values[0]}
        print(f"{family}: {metadata['n_rows']} rows; {args.replicates} shared draws",flush=True)
        boot=shared_image_cluster_bootstrap(images,statistic,n_replicates=args.replicates,seed=0,ci=.95)
        for key,summary in boot["estimates"].items():
            row={"family":family,"quantity":key,"point_mean3":summary["point"],
                 "ci_low_mean3":summary["ci_low"],"ci_high_mean3":summary["ci_high"],
                 "seed_standard_deviation":summary["seed_standard_deviation"],
                 "valid_replicates":summary["valid_replicates"],"invalid_replicates":summary["invalid_replicates"],
                 "n_replicates":args.replicates,"bootstrap_seed":0,"ci_level":.95,"resample_unit":"image_cluster"}
            all_ci.append(row)
            raw[f'{family}__{key}__mean3']=summary['replicates']
            for seed in SEEDS:
                raw[f'{family}__{key}__seed{seed}']=summary['per_seed_replicates'][seed]
            if key in ("A00","A10","A01","A11","interaction_I"):
                anchor=repair_raw[f'{family}__{key}__mean3'][:args.replicates]
                error=float(np.max(np.abs(summary['replicates']-anchor)))
                checks['max_bootstrap_anchor_residual']=max(checks['max_bootstrap_anchor_residual'],error)
                assert error<=1e-12,(family,key,error)
        manifest['families'][family]={**metadata,"status":"COMPLETE"}
        _write_csv(out/'point_components.csv',all_points,list(all_points[0]))
        _write_csv(out/'bootstrap_ci.csv',all_ci,list(all_ci[0]))
        _write_npz_atomic(out/'raw_replicates.npz',raw)
        _write_json(out/'manifest.json',manifest)
        print(f"{family}: complete, elapsed {time.monotonic()-started:.1f}s",flush=True)
    # Verify every frozen source again after all calculations; no tolerance relaxation.
    assert frozen==_validate_frozen_manifest()
    for entry in manifest['analysis_sources']:
        assert digest(ROOT/entry['path'])==entry['sha256']
    checks['frozen_input_hashes_unchanged']=True
    checks['all_draws_valid']=all(r['invalid_replicates']==0 for r in all_ci)
    checks['n_quantities']=len(all_ci)
    manifest.update(status="COMPLETE",checks=checks,elapsed_seconds=time.monotonic()-started)
    render_report(out,all_ci,checks)
    render_figures(out,all_ci)
    manifest['outputs']=[{'path':p.name,'sha256':digest(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json']
    _write_json(out/'manifest.json',manifest)
    print(json.dumps(checks),flush=True)


if __name__ == "__main__":
    main()
