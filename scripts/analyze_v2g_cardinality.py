"""V2-G **G3 cardinality reliability replication gate** (protocol V2-A1).

只读 phase0b 的 per-seed ``bootstrap.csv``（image-clustered paired bootstrap,
5000 reps, global_T_corrected variant, __pooled__ common cohort），按已冻结的
V2-A1 逻辑逐 backbone 判定 cardinality 现象是否在 B1 (OpenCLIP B/16) 与
B2 (SigLIP B/16) 上复现。

符号处理（关键，两套审计约定相反）：
  * bootstrap 的 ``diff = mean_a - mean_b``，a=K5、b=K_b。
  * AUROC / RER 的 "drop" 直接 = diff（随 K 增大而下降，故为正）。
  * E-AURC 的 "worsening K5->K_b"（基准 K5）= -r/(1+r)，其中
    r = relative diff = (a-b)/b（基准 K_b）。CI 两端同样变换后按单调性重排。

Gate（逐 backbone，基于主对比 K5 vs K50，3 seeds 全部要求 CI 排除 0 且方向一致）：
  Route A: mean ΔAUROC(K5->K50) >= 0.03  且  每 seed paired CI 排除 0（ci_low>0）
  Route B: mean E-AURC worsening >= 20%  且  mean RER@50 drop >= 10pp
           且  E-AURC 每 seed CI 排除 0  且  RER@50 每 seed CI 排除 0
  CARDINALITY_REPLICATED = Route A OR Route B

用法:
    python scripts/analyze_v2g_cardinality.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "v2_backbone_generalization"

BACKBONES = {
    "B1_openclip_b16": R / "b1_phase0b",
    "B2_siglip_b16": R / "b2_phase0b",
}
SEEDS = (1, 2, 3)
VARIANT = "global_T_corrected"
SPLIT = "__pooled__"
PRIMARY_KB = 50          # gate 主对比 K5 -> K50
AUC_THRESHOLD = 0.03     # Route A: AUROC drop
EAURC_WORSEN_THRESHOLD = 0.20   # Route B: relative worsening
RER50_DROP_THRESHOLD = 0.10      # Route B: RER@50 drop (pp -> fraction)

ROUTE_A = "Route_A_ΔAUROC>=0.03 & CI excludes 0"
ROUTE_B = "Route_B_E-AURC_worsen>=20% & RER50_drop>=10pp & CIs exclude 0"


def _num(x):
    return float(x) if x not in ("", None) else None


def _boot_rows(seed_dir: Path, kb: int):
    """返回 {metric|diff_kind: row} for split/variant/K_b 过滤后的 bootstrap 行。"""
    path = seed_dir / "bootstrap.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    out = {}
    with open(path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if (r["eval_split"] == SPLIT and r["variant"] == VARIANT
                    and r["K_a"] == "5" and int(r["K_b"]) == kb):
                out["{}|{}".format(r["metric"], r["diff_kind"])] = r
    return out


def _worsening_from_rel(mean_a, mean_b, ci_low_rel, ci_high_rel):
    """把 (a-b)/b 基准 K_b 的 relative 转成 (b-a)/a 基准 K5 的 worsening + CI。"""
    r = (mean_a - mean_b) / mean_b
    w = -r / (1.0 + r)
    # -x/(1+x) 在 x<0 区间对 x 单调递减 -> 边界映射后重排
    t = lambda x: -x / (1.0 + x)
    lo, hi = t(ci_low_rel), t(ci_high_rel)
    return w, min(lo, hi), max(lo, hi)


def _sign_excludes_zero(ci_low, ci_high):
    """paired CI 是否排除 0，并返回正/负方向。"""
    if ci_low is None or ci_high is None:
        return None, None
    if ci_low > 0:
        return True, "positive"
    if ci_high < 0:
        return True, "negative"
    return False, ("positive" if ci_low + ci_high > 0 else "negative")


def _metric_summary(per_seed, key, transform=None):
    """跨 seed 汇总某 bootstrap 指标；transform(row)->(effect,ci_low,ci_high)。"""
    effects, ci_lows, ci_highs, excl, signs = [], [], [], [], []
    for row in per_seed:
        if key not in row:
            continue
        r = row[key]
        if transform is None:
            eff, lo, hi = _num(r["diff"]), _num(r["ci_low"]), _num(r["ci_high"])
        else:
            eff, lo, hi = transform(r)
        effects.append(eff)
        ci_lows.append(lo)
        ci_highs.append(hi)
        e, s = _sign_excludes_zero(lo, hi)
        excl.append(e)
        signs.append(s)
    if not effects:
        return None
    m = sum(effects) / len(effects)
    sd = (sum((x - m) ** 2 for x in effects) / len(effects)) ** 0.5 if len(effects) > 1 else 0.0
    return {
        "effect_mean": m,
        "effect_std": sd,
        "per_seed_effects": effects,
        "per_seed_ci_low": ci_lows,
        "per_seed_ci_high": ci_highs,
        "all_seeds_ci_exclude_0": all(bool(x) for x in excl) and len(excl) == len(SEEDS),
        "direction": signs[0] if signs else None,
    }


def _auc_transform(row):
    # drop = AUROC_K5 - AUROC_K50 = diff（正=下降）
    return _num(row["diff"]), _num(row["ci_low"]), _num(row["ci_high"])


def _make_eaurc_transform():
    def t(row):
        return _worsening_from_rel(
            _num(row["mean_a"]), _num(row["mean_b"]), _num(row["ci_low"]), _num(row["ci_high"]))
    return t


def _rer_transform(row):
    return _num(row["diff"]), _num(row["ci_low"]), _num(row["ci_high"])


def evaluate_backbone(out_dir: Path):
    per_seed = [_boot_rows(out_dir / "seed_{}".format(s), PRIMARY_KB) for s in SEEDS]

    auc = _metric_summary(per_seed, "auroc_correct|absolute", _auc_transform)
    eaurc = _metric_summary(per_seed, "e_aurc|relative", _make_eaurc_transform())
    rer = _metric_summary(per_seed, "rer_at_50|absolute", _rer_transform)

    route_a = bool(auc and auc["effect_mean"] >= AUC_THRESHOLD
                   and auc["all_seeds_ci_exclude_0"] and auc["direction"] == "positive")
    route_b = bool(eaurc and rer
                   and eaurc["effect_mean"] >= EAURC_WORSEN_THRESHOLD
                   and rer["effect_mean"] >= RER50_DROP_THRESHOLD
                   and eaurc["all_seeds_ci_exclude_0"]
                   and rer["all_seeds_ci_exclude_0"])

    return {
        "primary_comparison": "K5_vs_K50",
        "variant": VARIANT,
        "split": SPLIT,
        "auc_drop_K5_K50": auc,
        "eaurc_worsening_K5_K50": eaurc,
        "rer50_drop_K5_K50": rer,
        "route_a_passed": route_a,
        "route_b_passed": route_b,
        "CARDINALITY_REPLICATED": bool(route_a or route_b),
    }


def main():
    payload = {
        "artifact": "v2g_g3_cardinality_gate",
        "protocol": "V2-A1 (frozen)",
        "bootstrap": "image-clustered paired, 5000 reps, ci=0.95",
        "thresholds": {
            "route_a_auc_drop_min": AUC_THRESHOLD,
            "route_b_eaurc_worsen_min": EAURC_WORSEN_THRESHOLD,
            "route_b_rer50_drop_min": RER50_DROP_THRESHOLD,
        },
        "route_a": ROUTE_A,
        "route_b": ROUTE_B,
        "backbones": {},
    }

    for name, out_dir in BACKBONES.items():
        payload["backbones"][name] = evaluate_backbone(out_dir)

    # 逐 backbone 摘要打印
    print("=" * 78)
    print("V2-G G3 CARDINALITY RELIABILITY REPLICATION GATE")
    print("=" * 78)
    for name, res in payload["backbones"].items():
        auc, eaurc, rer = res["auc_drop_K5_K50"], res["eaurc_worsening_K5_K50"], res["rer50_drop_K5_K50"]
        print("\n[{}]".format(name))
        print("  ΔAUROC K5->K50  = {:+.4f} ±{:.4f}  CI-excl-0(3 seeds)={} dir={}".format(
            auc["effect_mean"], auc["effect_std"], auc["all_seeds_ci_exclude_0"], auc["direction"]))
        print("  E-AURC worsen   = {:+.1f}%  (CI {:.1f}% .. {:.1f}%)  CI-excl-0={}".format(
            100 * eaurc["effect_mean"],
            100 * min(eaurc["per_seed_ci_low"]), 100 * max(eaurc["per_seed_ci_high"]),
            eaurc["all_seeds_ci_exclude_0"]))
        print("  RER@50 drop     = {:+.1f}pp  CI-excl-0={}".format(
            100 * rer["effect_mean"], rer["all_seeds_ci_exclude_0"]))
        print("  Route A: {} | Route B: {}  =>  {}".format(
            "PASS" if res["route_a_passed"] else "fail",
            "PASS" if res["route_b_passed"] else "fail",
            "CARDINALITY_REPLICATED" if res["CARDINALITY_REPLICATED"] else "NOT_REPLICATED"))

    out_path = R / "g3_cardinality_gate.json"
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    print("\nwrote", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
