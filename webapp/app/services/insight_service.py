# -*- coding: utf-8 -*-
"""
服务层 · 洞察服务
=======================================================================
职责：把仓储层读到的统计产物整理成「可直接呈现的结论」——
题材先验、选题分档、双标签组合、全局归因、论文口径评测指标。

与 ModelService 的分工：
    ModelService 负责「这一部剧会怎样」；
    InsightService 负责「整体规律是什么」。
"""
from __future__ import annotations

from app.repositories.artifact_repository import ArtifactRepository

# 分档配色交由前端决定，这里只输出语义化 level，便于换肤
TIER_ORDER = ("首选", "优先", "中性", "规避")

_TIER_RULE = {
    "首选": "置信区间下界高于全站基准",
    "优先": "点估计明显高于基准，样本量尚不足支撑显著性",
    "中性": "与全站基准无实质差异",
    "规避": "点估计明显低于基准",
}


class InsightService:
    """整体统计规律的整理与对外输出。"""

    # ------------------------------------------------------------------ #
    # 概览
    # ------------------------------------------------------------------ #
    @staticmethod
    def overview() -> dict:
        payload = ArtifactRepository.webmodel()
        meta = payload.get("meta", {})
        kpi = payload.get("kpi", {})
        return {
            "meta": meta,
            "kpi": kpi,
            "study_scale": {
                "n_samples": meta.get("n_samples"),
                "n_features": 64,
                "n_models_compared": len(
                    ArtifactRepository.metrics().get("classification", [])),
            },
        }

    # ------------------------------------------------------------------ #
    # 题材
    # ------------------------------------------------------------------ #
    @staticmethod
    def genre_options() -> list[dict]:
        """题材选择器数据源：题材名 + 样本量 + 历史爆款率 + 分档。"""
        cat = ArtifactRepository.webmodel().get("cat_stats", {})
        out = []
        for name, s in cat.items():
            out.append({
                "genre": name,
                "n": s.get("n", 0),
                "hot_rate": round(float(s.get("hot_rate", 0.0)), 1),
                "ci_low": round(float(s.get("ci", [0, 0])[0]), 1),
                "ci_high": round(float(s.get("ci", [0, 0])[1]), 1),
                "tier": s.get("tier", "中性"),
                "episodes": s.get("episodes"),
                "rating": s.get("rating"),
            })
        out.sort(key=lambda d: d["hot_rate"], reverse=True)
        return out

    @staticmethod
    def genre_ranking() -> dict:
        """题材爆款率排名（含置信区间），供对标图使用。"""
        items = InsightService.genre_options()
        return {
            "baseline": ArtifactRepository.webmodel().get("kpi", {}).get("hot_rate"),
            "items": sorted(items, key=lambda d: d["hot_rate"]),
        }

    @staticmethod
    def selection_tiers() -> dict:
        tiers = ArtifactRepository.webmodel().get("genre_tier", {})
        cat = ArtifactRepository.webmodel().get("cat_stats", {})
        groups = []
        for name in TIER_ORDER:
            members = tiers.get(name, [])
            if not members:
                continue
            groups.append({
                "tier": name,
                "rule": _TIER_RULE[name],
                "count": len(members),
                "members": [
                    {
                        "genre": g,
                        "hot_rate": round(float(cat.get(g, {}).get("hot_rate", 0.0)), 1),
                        "n": cat.get(g, {}).get("n", 0),
                    }
                    for g in members
                ],
            })
        return {"baseline": ArtifactRepository.webmodel().get("kpi", {}).get("hot_rate"),
                "groups": groups}

    @staticmethod
    def combos() -> dict:
        cs = ArtifactRepository.webmodel().get("combo_stats", {})
        return {
            "base_hot_rate": cs.get("base"),
            "min_samples": cs.get("min_n"),
            "n_pairs": cs.get("n_pairs"),
            "best": cs.get("best", []),
            "worst": cs.get("worst", []),
        }

    @staticmethod
    def plot_stats() -> list[dict]:
        stats = ArtifactRepository.webmodel().get("plot_stats", {})
        items = [
            {"tag": k, "n": v.get("n", 0), "hot_rate": round(float(v.get("hot_rate", 0.0)), 1)}
            for k, v in stats.items()
        ]
        items.sort(key=lambda d: d["hot_rate"], reverse=True)
        return items

    # ------------------------------------------------------------------ #
    # 归因与评测
    # ------------------------------------------------------------------ #
    @staticmethod
    def importance_overall() -> dict:
        return ArtifactRepository.webmodel().get("importance_overall", {})

    @staticmethod
    def paper_metrics() -> dict:
        """论文口径的评测结果，与正文表格一一对应。"""
        m = ArtifactRepository.metrics()
        return {
            "classification": m.get("classification", []),
            "regression": m.get("regression", []),
            "ablation": m.get("ablation", []),
            "ablation_total_gain": m.get("ablation_total_gain"),
            "robustness": m.get("robustness", {}),
            "importance_group": m.get("importance_group", {}),
            "shap_top": m.get("shap_top", {}),
            "shap_direction": m.get("shap_direction", {}),
        }

    @staticmethod
    def roc_curves() -> dict:
        payload = ArtifactRepository.webmodel()
        out = {}
        for name, m in payload.get("models", {}).items():
            curves = m.get("curves", {})
            out[name] = {
                "label": m["name"],
                "auc": round(float(m["metrics"]["auc"]), 4),
                "roc": curves.get("roc", {}),
                "pr": curves.get("pr", {}),
            }
        return out
