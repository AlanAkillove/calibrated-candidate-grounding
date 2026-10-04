"""Generate a V3 evidence index from completed stored analyses, never resample."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccg.v3.protocol import sha256

OUT = ROOT / "results/v3_final_validation"


def read(relative):
    path = OUT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def interval(item, adjusted=False):
    ci = item.get("bonferroni_9875", {}) if adjusted else item
    point, low, high = item.get("point"), ci.get("ci_low"), ci.get("ci_high")
    if point is None:
        return "未定义"
    if low is None or high is None:
        return f"{point:+.5f}；区间不可用"
    return f"{point:+.5f} [{low:+.5f}, {high:+.5f}]"


def main():
    report = ["# V3 产物生成的证据索引", "", "本文件只读取已保存产物。未执行、未验收或非正式抽样不生成确认结论。", ""]
    entries = []
    for relative in ("legacy_inputs_initial.json", "legacy_inputs_final.json", "config.json", "STATUS.json",
                     "exposure/source_identity_counts.json", "exposure/historical_fingerprint_status.json", "exposure/final_acceptance.json",
                     "replication/replication_summary.json", "replication/bootstrap_summary.json", "replication/replication_acceptance.json",
                     "pipeline/dev_acceptance.json", "pipeline/dev_revalidation.json", "pipeline/confirmation/preparation_acceptance.json",
                     "tests_before_confirmation.json", "protocol_freeze.json", "confirmation_run.json", "final_acceptance.json",
                     "model_replay/manifest.json", "pipeline/confirmation/candidate_geometry_identity.json",
                     "publication/visual_acceptance.json", "pipeline/dev/inference/predictions_package.json",
                     "pipeline/confirmation/inference/predictions_package.json",
                     "pipeline/confirmation/bootstrap/index.json", "pipeline/confirmation/bootstrap/controlled/summary.json",
                     "pipeline/confirmation/bootstrap/natural/summary.json"):
        path = OUT / relative
        if path.is_file():
            entries.append({"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    exposure = read("exposure/final_acceptance.json")
    report += ["## 暴露与来源审计", "", f"最终暴露验收：{exposure.get('status') if exposure else '尚未完成'}。", "",
               "确认身份独立与视觉来源不同分开判断。VG/GQA 命名空间不同不能证明非 COCO 来源；基础模型预训练暴露不在本项目审计能力内。", ""]
    if exposure:
        for key in ("official_positive_val_identity_eligible_image_count", "val_images_downloaded_and_hashed", "val_images_final_eligible",
                    "val_expressions_final_eligible", "train_images_selected", "train_expressions_selected"):
            if key in exposure:
                report.append(f"- {key}: {exposure[key]}")
        report.append("")
    replication = read("replication/bootstrap_summary.json")
    report += ["## B/16 特征来源复制", "", "已观察 COCO 测试数据上的配置复制；不属于独立确认。", ""]
    if replication and replication.get("status") == "COMPLETE" and replication.get("replicates") == 5000 and replication.get("seed") == 0:
        report += ["效应均为三个固定 seed 内比较后取均值，95% 共享图像抽样区间。AUROC 正向表示增益，Risk 差分负向表示改善。", "",
                   "| 比较 | AUROC 效应 [95% CI] | Risk@50 效应 [95% CI] | Risk@80 效应 [95% CI] |", "|---|---:|---:|---:|"]
        for cohort, prefix, label in (("random_all", "Full_minus_SQ__randomK5", "random K5 Full−SQ"),
                                       ("random_all", "Full_minus_SQ__randomK50", "random K50 Full−SQ"),
                                       ("same4", "Full_minus_SQ__hard5", "matched hard K5 Full−SQ"),
                                       ("same8", "Full_minus_SQ__expb_m8", "K10 m8 Full−SQ"),
                                       ("same4", "Hard_minus_random__Full_minus_SQ", "hard−random 增量差"),
                                       ("same8", "m8_minus_m0__Full_minus_SQ", "m8−m0 增量差")):
            estimates = replication["cohorts"][cohort]["estimates"]
            values = [interval(estimates[f"{prefix}__{metric}"]) for metric in ("auroc_correct", "risk_at_50", "risk_at_80")]
            report.append(f"| {label} | {' | '.join(values)} |")
        report.append("")
    else:
        report += ["正式区间尚未完成。点估计不代替区间或复制验收。", ""]
    controlled = read("pipeline/confirmation/bootstrap/controlled/summary.json")
    report += ["## 四个主要独立确认比较", ""]
    if controlled and controlled.get("status") == "COMPLETE" and controlled.get("n_replicates") == 5000 and controlled.get("seed") == 0:
        report += ["| 比较 | AUROC 效应 [95% CI] | Bonferroni 98.75% CI | 无效抽样 |", "|---|---:|---:|---:|"]
        for comparison in ("C1", "C2", "C3", "C4"):
            item = controlled["estimates"][f"{comparison}/auroc_correct"]
            ci = item["bonferroni_9875"]
            adjusted = "不可用" if ci["ci_low"] is None else f"[{ci['ci_low']:+.5f}, {ci['ci_high']:+.5f}]"
            report.append(f"| {comparison} | {interval(item)} | {adjusted} | {item['invalid_replicates']} |")
        report += ["", "区间覆盖零不证明无效或等效；含无效抽样的区间条件于有限抽样。C2 与 C3 分开判断。",
                   "C1 正向表示 K50 的判别退化；C2–C4 正向表示指定左侧模型优于右侧。", "",
                   "C1 = B0 MSP K5−K50；C2 = B0 Full−(S+Q) K50；C3 = B/16 Full−(S+Q) K50；C4 = B0 ScoreDeepSets large-K trained−small-K trained K50。", "",
                   f"受控共同样本流：`{json.dumps(controlled['flow'], ensure_ascii=False)}`。", "",
                   "### 同一受控共同样本上的绝对端点", "",
                   "| 配置/模型/K | Correctness AUROC [95% CI] | Risk@50 [95% CI] | Risk@80 [95% CI] | Brier [95% CI] |",
                   "|---|---:|---:|---:|---:|"]
        endpoints = (("b0", "MSP", 5), ("b0", "MSP", 50), ("b0", "S", 50), ("b0", "S+Q", 50),
                     ("b0", "S+V", 50), ("b0", "Full", 50), ("b16", "S", 50), ("b16", "S+Q", 50),
                     ("b16", "S+V", 50), ("b16", "Full", 50), ("b0", "SDS_small", 50), ("b0", "SDS_large", 50))
        for backbone, model, k in endpoints:
            values = [interval(controlled["estimates"][f"absolute/{backbone}/{model}/K{k}/{metric}"])
                      for metric in ("auroc_correct", "risk50", "risk80", "brier_binary")]
            report.append(f"| {backbone}/{model}/K{k} | {' | '.join(values)} |")
        report.append("")
        acceptance = read("final_acceptance.json")
        if acceptance and acceptance.get("status") == "PASS":
            report += ["### 受控共同样本的定位准确率", "", "此表是三个固定 seed 的描述性均值，不将其当作额外确认假设；逐 seed 值保存于最终验收。", "",
                       "| 配置/K | Accuracy |", "|---|---:|"]
            for name, item in acceptance["controlled_accuracy_points"].items():
                report.append(f"| {name} | {item['mean']:.5f} |")
            report.append("")
    else:
        report += ["尚无完成的正式确认统计；不得将现有 COCO 结果填入此表。", ""]
    natural = read("pipeline/confirmation/bootstrap/natural/summary.json")
    report += ["## 自然候选外部有效性", ""]
    if natural and natural.get("status") == "COMPLETE" and natural.get("n_replicates") == 5000 and natural.get("seed") == 0:
        report += ["以下使用 V3 fractional_boundary_tie 风险定义，非历史 ceil/stable 指标。覆盖条件与全部输入分别报告。", "",
                   "| 配置/K | 覆盖率 | 全部定位准确率 | 覆盖条件准确率 | Full−SQ 整体 AUROC | Full−SQ 覆盖条件 AUROC |", "|---|---:|---:|---:|---:|---:|"]
        for backbone in ("b0", "b16"):
            for k in (5, 10, 20, 50):
                prefix = f"{backbone}/K{k}"
                names = (f"{prefix}/flow/target_coverage", f"{prefix}/flow/accuracy_all", f"{prefix}/flow/accuracy_covered",
                         f"{prefix}/paired_overall/Full_minus_SQ/auroc_correct", f"{prefix}/paired_covered/Full_minus_SQ/auroc_correct")
                report.append(f"| {backbone}/K{k} | {' | '.join(interval(natural['estimates'][name]) for name in names)} |")
        report += ["", "完整 MSP/四组绝对性能、选择性风险、可评分分母、流失与排名分解均保存在自然统计产物中。排名分解不识别因果机制。", ""]
        report += ["### 自然样本流与分母", "", "候选身份在配置和 seed 之间共享；此表读取 B0 seed1 的共同候选流，完整逐 seed 记录仍保留。", "",
                   "| K | 全部表达 | 覆盖表达 | bank 覆盖表达 | 空候选 | 1–4 候选基础定位 | 配对可评分 | 覆盖且配对可评分 |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k in (5, 10, 20, 50):
            row = natural["flow"][f"b0/b3_seed1/K{k}"]
            values = [row[name] for name in ("n_all", "n_covered", "n_bank_covered", "n_empty", "n_basic_only")]
            values += [row["paired_denominators"][name] for name in ("overall", "covered")]
            report.append(f"| {k} | {' | '.join(map(str, values))} |")
        report += ["", "### 自然接口的绝对性能", "", "所有风险采用 fractional_boundary_tie；每个 scope 使用自己的明确分母。", "",
                   "| 配置/K/scope/模型 | AUROC [95% CI] | Risk@50 [95% CI] | Risk@80 [95% CI] | Brier [95% CI] |",
                   "|---|---:|---:|---:|---:|"]
        for backbone in ("b0", "b16"):
            for k in (5, 10, 20, 50):
                for scope in ("overall", "covered"):
                    for model in ("MSP", "S", "S+Q", "S+V", "Full"):
                        values = [interval(natural["estimates"][f"{backbone}/K{k}/{scope}/{model}/{metric}"])
                                  for metric in ("auroc_correct", "risk50", "risk80", "brier_binary")]
                        report.append(f"| {backbone}/K{k}/{scope}/{model} | {' | '.join(values)} |")
        report += ["", "### Full−SQ 排名增量的有符号分解", "",
                   "整体增量 = 覆盖错误成对贡献 + 未覆盖错误成对贡献。权重是各 seed 中错误样本组成；先分别算贡献再平均，不能把均值权重与均值增量相乘。", "",
                   "| 配置/K | 未覆盖错误权重 [95% CI] | 覆盖错误贡献 [95% CI] | 未覆盖错误贡献 [95% CI] |",
                   "|---|---:|---:|---:|"]
        for backbone in ("b0", "b16"):
            for k in (5, 10, 20, 50):
                values = [interval(natural["estimates"][f"{backbone}/K{k}/ranking_audit/{name}"])
                          for name in ("uncovered_negative_weight", "covered_ranking_contribution", "uncovered_ranking_contribution")]
                report.append(f"| {backbone}/K{k} | {' | '.join(values)} |")
        report.append("")
    else:
        report += ["尚无完成的自然候选确认测量。不得依据受控结果预设自然 K 趋势。", ""]
    report += ["## 可核对产物", "", "| 路径 | SHA256 | 字节 |", "|---|---|---:|"]
    report += [f"| [{item['path']}]({item['path']}) | {item['sha256']} | {item['size_bytes']} |" for item in entries]
    (OUT / "evidence_index.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "evidence_index.json").write_text(json.dumps({"schema": "v3-evidence-index-v1", "sources": entries,
        "publication_rule": "completed stored formal 5000 seed0 outputs only; developmental and confirmation kept separate"}, indent=2) + "\n", encoding="utf-8")
    acceptance = read("final_acceptance.json")
    if acceptance and acceptance.get("status") == "PASS" and controlled and natural and controlled.get("status") == natural.get("status") == "COMPLETE":
        findings = ["# V3 正文结果表：由正式产物生成", "", "来源：首次冻结确认的保存预测与 5000 次共享图像抽样。四项主要 AUROC 比较给出 Bonferroni 98.75% 区间；其余为边际 95% 区间。", "",
                    f"确认输入：{acceptance['images']} 图像、{acceptance['expressions']} 表达；受控共同样本：{controlled['flow']['common_images']} 图像、{controlled['flow']['common_expressions']} 表达；受控排除 {controlled['flow']['excluded_expressions']} 表达。", "",
                    "| 预先固定的比较 | AUROC 差分 [98.75% CI] | Risk@50 差分 [95% CI] | 判读 |", "|---|---:|---:|---|"]
        labels = ("C1: B0 MSP K5−K50", "C2: B0 Full−SQ K50", "C3: B/16 Full−SQ K50", "C4: B0 SDS large−small K50")
        for i, label in enumerate(labels, 1):
            auc = controlled['estimates'][f'C{i}/auroc_correct']
            low, high = (auc['bonferroni_9875'][name] for name in ('ci_low', 'ci_high'))
            judgment = '区间支持指定正向效应' if low is not None and low > 0 else '区间支持反向效应' if high is not None and high < 0 else '未确认方向；不代表等效'
            findings.append(f"| {label} | {interval(auc, adjusted=True)} | {interval(controlled['estimates'][f'C{i}/risk50'])} | {judgment} |")
        findings += ["", "C1 的 Risk 差分为 K5−K50，负号表示 K50 风险更高；C2–C4 的负向 Risk 差分表示指定左侧模型改善。不能把 C1 的风险变化等同于 AUROC 退化。", "",
                     "### 自然 K50 的边界", "", "| 配置 | 整体 Full−SQ AUROC [95% CI] | 覆盖条件 Full−SQ AUROC [95% CI] | 覆盖错误贡献 [95% CI] | 未覆盖错误贡献 [95% CI] |", "|---|---:|---:|---:|---:|"]
        for backbone in ('b0', 'b16'):
            prefix = f'{backbone}/K50'
            names = ('paired_overall/Full_minus_SQ/auroc_correct', 'paired_covered/Full_minus_SQ/auroc_correct',
                     'ranking_audit/covered_ranking_contribution', 'ranking_audit/uncovered_ranking_contribution')
            findings.append(f"| {backbone} | {' | '.join(interval(natural['estimates'][f'{prefix}/{name}']) for name in names)} |")
        findings += ["", "### 同一批自然输入的覆盖和准确率", "", "| K | 覆盖率 [95% CI] | B0 全部 Accuracy [95% CI] | B/16 全部 Accuracy [95% CI] |", "|---|---:|---:|---:|"]
        for k in (5, 10, 20, 50):
            names = (f'b0/K{k}/flow/target_coverage', f'b0/K{k}/flow/accuracy_all', f'b16/K{k}/flow/accuracy_all')
            findings.append(f"| {k} | {' | '.join(interval(natural['estimates'][name]) for name in names)} |")
        findings += ["", "自然不同 K 的覆盖条件样本会改变；其条件 AUROC 曲线不能自动解释成同一样本上的 K 因果效应。准确率和可靠性排序是不同性质。", "",
                     "V 的增量不等于 Full 普遍最佳。大 K 训练优势不等于 ScoreDeepSets 超越简单基线；绝对端点、全部 K、逐 seed、无效抽样及模型分母见 [完整证据索引](../evidence_index.md)。"]
        (OUT / 'publication').mkdir(exist_ok=True)
        (OUT / 'publication/key_findings.md').write_text('\n'.join(findings)+'\n', encoding='utf-8')
    print(json.dumps({"status": "INDEX_GENERATED", "sources": len(entries)}))


if __name__ == "__main__":
    main()
