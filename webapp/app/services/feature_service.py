# -*- coding: utf-8 -*-
"""
服务层 · 特征工程
=======================================================================
职责：把「一部新剧的元信息 + 简介文本」翻译成模型可消费的特征向量。

这是本系统最能体现「业务 → 特征」转换的一层，也是新剧实时预测的价值所在：
新剧尚未上线，没有评论、点赞、评分，**唯一可得的是内容属性**。
因此本模块只依赖剧名/简介/题材/集数四类输入，自动派生 40 维冷启动特征。

特征口径与训练侧完全一致（data/processed/ml_dataset.csv 的同名列）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.training_config import plot_keywords
from app.repositories.artifact_repository import ArtifactRepository

# 剧名含季数：「第二季」「第 3 季」「Season 2」「第三部」
_RE_SEASON = re.compile(r"(第\s*[0-9一二三四五六七八九十]+\s*[季部]|第\s*[0-9]+\s*季"
                        r"|season\s*\d+|S\d{1,2}\b)", re.I)
# 剧名含数字（阿拉伯或中文，如「重生之1988」「九零年代」）
_RE_NUMBER = re.compile(r"(\d|[零一二三四五六七八九十百千万]{1,3}(?=年代|年|零|九零|八零))")
# 全角/半角问号、感叹号
_RE_QUESTION = re.compile(r"[?？]")
_RE_EXCLAIM = re.compile(r"[!！]")


@dataclass
class DramaSpec:
    """一部待预测剧集的输入规格。"""

    title: str = ""
    synopsis: str = ""
    genres: list[str] = field(default_factory=list)
    episodes: int = 0
    # —— 以下为上线后早期观测，冷启动模式下不使用 ——
    hot_rating_mean: float | None = None
    hot_rating_good: float | None = None
    hot_rating_bad: float | None = None
    hot_digg_mean: float | None = None
    ep1_digg_mean: float | None = None
    ep1_hot_ratio: float | None = None
    ep1_urge_ratio: float | None = None
    ep1_pos_ratio: float | None = None
    ep1_neg_ratio: float | None = None

    def has_engagement(self) -> bool:
        """是否提供了足够的上线后观测，可用于全量模型。"""
        required = ("hot_rating_mean", "hot_digg_mean", "ep1_digg_mean",
                    "ep1_pos_ratio")
        return all(getattr(self, k) is not None for k in required)


class FeatureService:
    """剧集规格 → 特征向量。"""

    def __init__(self) -> None:
        self._model_features = {
            name: ArtifactRepository.model_payload(name)["features"]
            for name in ("cold", "full")
        }

    # ------------------------------------------------------------------ #
    # 文本派生
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_flags(title: str, synopsis: str) -> dict:
        """从剧名与简介提取结构标记（与训练侧同名同口径）。"""
        return {
            "title_len": len(title),
            "title_has_season": 1 if _RE_SEASON.search(title) else 0,
            "title_has_number": 1 if _RE_NUMBER.search(title) else 0,
            "summary_len": len(synopsis),
            "summary_has_question": 1 if _RE_QUESTION.search(synopsis) else 0,
            "summary_has_exclaim": 1 if _RE_EXCLAIM.search(synopsis) else 0,
        }

    @staticmethod
    def extract_plot_tags(synopsis: str) -> list[str]:
        """按关键词词典从简介中抽取爽点标签（多标签，可命中 0 到多个）。"""
        hits: list[str] = []
        for tag, words in plot_keywords().items():
            if any(w in synopsis for w in words):
                hits.append(tag)
        return hits

    # ------------------------------------------------------------------ #
    # 向量构建
    # ------------------------------------------------------------------ #
    def build(self, spec: DramaSpec) -> dict:
        """返回 {cold: [...], full: [...] | None, derived: {...}}。"""
        base = self._base_values(spec)
        cold = [float(base.get(f, 0.0)) for f in self._model_features["cold"]]

        full: list[float] | None = None
        if spec.has_engagement():
            full = [float(base.get(f, 0.0)) for f in self._model_features["full"]]

        return {
            "cold": cold,
            "full": full,
            "derived": base,
            "plot_tags": self.extract_plot_tags(spec.synopsis),
            "flags": self.extract_flags(spec.title, spec.synopsis),
        }

    # ------------------------------------------------------------------ #
    def _base_values(self, spec: DramaSpec) -> dict[str, float]:
        """构造「特征名 → 取值」的中间字典，冷/全量模型共用。"""
        episodes = float(spec.episodes or 0)
        plot_tags = self.extract_plot_tags(spec.synopsis)
        flags = self.extract_flags(spec.title, spec.synopsis)

        values: dict[str, float] = {
            "episodes": episodes,
            "log_episodes": _log1p(episodes),
            "is_long_series": 1.0 if episodes >= 100 else 0.0,
            "n_genre_tags": float(len(spec.genres)),
            **{k: float(v) for k, v in flags.items()},
        }

        # 题材多热编码
        for genre in spec.genres:
            values[f"tag_{genre}"] = 1.0
        # 爽点多热编码
        for tag in plot_tags:
            values[f"plot_{tag}"] = 1.0

        # 上线后观测（缺省不写入，冷启动向量自然取 0）
        for name in ("hot_rating_mean", "hot_rating_good", "hot_rating_bad",
                     "hot_digg_mean", "ep1_digg_mean", "ep1_hot_ratio",
                     "ep1_urge_ratio", "ep1_pos_ratio", "ep1_neg_ratio"):
            value = getattr(spec, name)
            if value is not None:
                values[name] = float(value)

        # 热评点赞合计：用户只给平均点赞时按 9 条热评估算（与前端口径一致）
        if spec.hot_digg_mean is not None:
            values["log_hot_digg_sum"] = _log1p(float(spec.hot_digg_mean) * 9.0)

        return values


def _log1p(value: float) -> float:
    import math
    return math.log1p(max(value, 0.0))
