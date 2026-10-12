# -*- coding: utf-8 -*-
"""
领域层 · 判定规则
=======================================================================
把连续概率翻译成可读结论。规则集中在此，接口层与服务层不重复实现。

分档口径与论文第 6 章一致：
    全站爆款基线 20%（热度数值进入全量 Top 20% 者记为爆款）
    ≥50%      高爆款潜力
    30%–50%   中等偏上
    20%–30%   接近全站均值
    <20%      低于平均
"""
from __future__ import annotations

BASELINE_HOT_RATE = 20.0            # 全站爆款率基线（%）

_TIERS: tuple[tuple[float, str, str], ...] = (
    (50.0, "高爆款潜力", "high"),
    (30.0, "中等偏上", "mid-high"),
    (20.0, "接近全站均值", "baseline"),
    (0.0, "低于平均", "low"),
)

__all__ = ["judge", "BASELINE_HOT_RATE"]


def judge(probability: float, baseline: float = BASELINE_HOT_RATE) -> dict:
    """概率（0–1）→ {verdict, level, lift, advice}。

    lift 为相对全站爆款基线的倍数，用于回答「比平均水平好多少」。
    """
    pct = probability * 100.0
    verdict, level = _TIERS[-1][1], _TIERS[-1][2]
    for bound, label, key in _TIERS:
        if pct >= bound:
            verdict, level = label, key
            break

    lift = (pct / baseline) if baseline else 0.0
    if level == "high":
        advice = "选题与内容配置均处于优势区间，建议优先立项"
    elif level == "mid-high":
        advice = "高于全站均值，可作为候选项目推进"
    elif level == "baseline":
        advice = "与全站水平相当，需靠题材组合或集数配置拉开差距"
    else:
        advice = "低于全站均值，建议调整题材或内容配置后再评估"

    return {
        "verdict": verdict,
        "level": level,
        "probability_pct": round(pct, 1),
        "baseline_pct": round(baseline, 1),
        "lift": round(lift, 2),
        "advice": advice,
    }
