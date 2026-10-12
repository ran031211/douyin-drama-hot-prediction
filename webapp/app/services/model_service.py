# -*- coding: utf-8 -*-
"""
服务层 · 模型服务
=======================================================================
职责：编排一次预测请求的完整流程 —— 特征构建 → 双模型推理 → 判定 → 局部归因。

本层是接口层与领域层之间的粘合剂：接口层只管 HTTP，
领域层只管「给一个向量算出概率」，而「什么时候用哪个模型、
结果怎么组织成一屏结论」由本层决定。
"""
from __future__ import annotations

from typing import Sequence

from app.domain.forest import GradientBoostedForest
from app.domain.verdict import BASELINE_HOT_RATE, judge
from app.repositories.artifact_repository import ArtifactRepository
from app.services.feature_service import DramaSpec, FeatureService

# 模型展示名：冷启动版面向「新剧未上线」，全特征版需要上线后观测
MODEL_LABELS = {
    "cold": "冷启动版（仅内容属性）",
    "full": "全特征版（内容 + 观众反馈）",
}


class ModelService:
    """双模型推理与结论组织。"""

    def __init__(self, feature_service: FeatureService) -> None:
        self._features = feature_service
        self._forests: dict[str, GradientBoostedForest] = {
            name: GradientBoostedForest(ArtifactRepository.model_payload(name))
            for name in ("cold", "full")
        }

    # ------------------------------------------------------------------ #
    @property
    def forests(self) -> dict[str, GradientBoostedForest]:
        return self._forests

    def baseline_hot_rate(self) -> float:
        kpi = ArtifactRepository.webmodel().get("kpi", {})
        return float(kpi.get("hot_rate", BASELINE_HOT_RATE))

    # ------------------------------------------------------------------ #
    def predict(self, spec: DramaSpec, top_k: int = 10) -> dict:
        """对一部剧给出双模型结论。全量模型缺少上线后观测时不参与。"""
        built = self._features.build(spec)
        baseline = self.baseline_hot_rate()

        cold_forest = self._forests["cold"]
        p_cold = cold_forest.predict_proba(built["cold"])

        result: dict = {
            "cold": self._pack("cold", p_cold, baseline),
            "full": None,
            "delta": None,
            "factors": [],
            "plot_tags": built["plot_tags"],
            "flags": built["flags"],
        }

        # 局部归因：冷启动模型对所有样本都可用
        active_forest, active_vec = cold_forest, built["cold"]

        if built["full"] is not None:
            full_forest = self._forests["full"]
            p_full = full_forest.predict_proba(built["full"])
            result["full"] = self._pack("full", p_full, baseline)
            result["delta"] = self._delta(p_cold, p_full, baseline)
            active_forest, active_vec = full_forest, built["full"]

        result["factors"] = active_forest.local_contributions(active_vec, top_k)
        result["explained_by"] = active_forest.name
        return result

    # ------------------------------------------------------------------ #
    def score_matrix(self, rows: Sequence[Sequence[float]], model: str = "cold") -> list[float]:
        """批量打分（用于数据集回测）。"""
        forest = self._forests[model]
        return [forest.predict_proba(r) for r in rows]

    # ------------------------------------------------------------------ #
    def _pack(self, key: str, probability: float, baseline: float) -> dict:
        forest = self._forests[key]
        verdict = judge(probability, baseline)
        return {
            "model": key,
            "label": MODEL_LABELS[key],
            "auc": round(float(forest.metrics.get("auc", 0.0)), 4),
            "decision_threshold": round(forest.threshold, 4),
            **verdict,
        }

    @staticmethod
    def _delta(p_cold: float, p_full: float, baseline: float) -> dict:
        lift_pp = (p_full - p_cold) * 100.0
        return {
            "probability_gain_pp": round(lift_pp, 1),
            "relative_gain": round(
                (p_full / p_cold - 1.0) if p_cold > 0 else 0.0, 3
            ),
            "vs_baseline_times": round(p_full * 100.0 / baseline, 2) if baseline else 0.0,
            "note": "追加观众反馈特征后的概率变化",
        }
