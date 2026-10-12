# -*- coding: utf-8 -*-
"""
服务层 · 数据集服务
=======================================================================
职责：
  1. 数据集浏览（分页 / 检索 / 排序）—— 委托仓储层；
  2. **全量回测**：用部署模型给 800 部剧逐条打分，与真实标签对比。

回测的关键在于「用哪个子集算指标」：
  · 在全部 800 条上算 AUC 会因含训练集而虚高（实测 ≈0.91），
    那是**拟合度**，不是泛化性能，不能与论文指标并列；
  · 正确做法是复现训练时的 hold-out 划分，只在测试集上算。
    为此本模块从 src/config.py 取回 RANDOM_SEED 与 TEST_SIZE，
    用同样的种子与分层参数重放划分 —— 复算出的 AUC 必须与产物里
    记录的 AUC 一致，否则说明导出链路或特征口径发生了漂移。

这使回测成为一次**在线自检**，而不只是一张展示图。
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from app.domain import training_config
from app.repositories.artifact_repository import ArtifactRepository
from app.repositories.drama_repository import DramaRepository
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)


def _auc(scores: list[float], labels: list[int]) -> float:
    """ROC-AUC（Mann–Whitney U，含并列秩平均）。"""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


class DatasetService:
    """数据集浏览 + 全量回测。"""

    _backtest: dict[str, Any] | None = None
    _lock = threading.Lock()

    def __init__(self, model_service: ModelService) -> None:
        self._models = model_service

    # ------------------------------------------------------------------ #
    # 浏览
    # ------------------------------------------------------------------ #
    @staticmethod
    def browse(limit: int, offset: int, keyword: str, sort: str,
               descending: bool) -> dict:
        page = DramaRepository.list_dramas(limit, offset, keyword, sort, descending)
        tiers = ArtifactRepository.webmodel().get("genre_tier", {})
        tier_of = {g: t for t, gs in tiers.items() for g in gs}
        for item in page["items"]:
            genres = [g for g in (item["category"] or "").split("/") if g]
            item["genres"] = genres
            item["tier"] = next((tier_of[g] for g in genres if g in tier_of), "中性")
        return page

    @staticmethod
    def detail(drama_id: str) -> dict | None:
        return DramaRepository.get_drama(drama_id)

    # ------------------------------------------------------------------ #
    # 回测
    # ------------------------------------------------------------------ #
    def is_ready(self) -> bool:
        return DatasetService._backtest is not None

    def backtest(self) -> dict:
        if DatasetService._backtest is not None:
            return DatasetService._backtest
        with DatasetService._lock:
            if DatasetService._backtest is None:
                DatasetService._backtest = self._run_backtest()
        return DatasetService._backtest

    def warm(self) -> None:
        """预热回测缓存（应用启动时在后台线程调用）。"""
        try:
            self.backtest()
        except Exception:                                     # pragma: no cover
            logger.exception("回测预热失败，将在首次请求时重试")

    # ------------------------------------------------------------------ #
    def _run_backtest(self) -> dict:
        df = ArtifactRepository.ml_dataset()
        labels = df["is_hot"].astype(int).tolist()
        n = len(labels)

        cold_feats = self._models.forests["cold"].features
        full_feats = self._models.forests["full"].features

        p_cold = self._models.score_matrix(
            df[cold_feats].astype(float).values.tolist(), "cold")
        p_full = self._models.score_matrix(
            df[full_feats].astype(float).values.tolist(), "full")

        payload = ArtifactRepository.webmodel()
        recorded = {
            "cold": payload["models"]["cold"]["metrics"]["auc"],
            "full": payload["models"]["full"]["metrics"]["auc"],
        }

        test_idx = self._holdout_index(n, labels)
        result: dict[str, Any] = {
            "n_samples": n,
            "actual_hot_rate": round(sum(labels) / n * 100, 1),
            "recorded_auc": recorded,
            "holdout": None,
            "full_sample_fit": {
                "auc_cold": round(_auc(p_cold, labels), 4),
                "auc_full": round(_auc(p_full, labels), 4),
                "note": "含训练集，属拟合度而非泛化指标",
            },
            "top_predicted": self._top_rows(df, labels, p_cold, test_idx, 10),
        }

        if test_idx is not None:
            y_te = [labels[i] for i in test_idx]
            auc_c = _auc([p_cold[i] for i in test_idx], y_te)
            auc_f = _auc([p_full[i] for i in test_idx], y_te)
            thr = self._models.forests["cold"].threshold
            pred = [1 if p_cold[i] >= thr else 0 for i in test_idx]
            tp = sum(1 for a, b in zip(pred, y_te) if a == 1 and b == 1)
            fp = sum(1 for a, b in zip(pred, y_te) if a == 1 and b == 0)
            fn = sum(1 for a, b in zip(pred, y_te) if a == 0 and b == 1)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            result["holdout"] = {
                "n_test": len(test_idx),
                "test_size": training_config.test_size(),
                "random_seed": training_config.random_seed(),
                "auc_cold": round(auc_c, 4),
                "auc_full": round(auc_f, 4),
                "auc_gain": round(auc_f - auc_c, 4),
                "auc_cold_recorded": recorded["cold"],
                "auc_full_recorded": recorded["full"],
                "auc_cold_match": abs(auc_c - recorded["cold"]) < 5e-4,
                "auc_full_match": abs(auc_f - recorded["full"]) < 5e-4,
                "threshold": round(thr, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4)
                      if precision + recall else 0.0,
                "actual_hot_rate": round(sum(y_te) / len(y_te) * 100, 1),
            }
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _holdout_index(n: int, labels: list[int]) -> list[int] | None:
        """复现训练时的 hold-out 划分。

        train_test_split 的划分只取决于 (n, y, test_size, random_state)，
        与 X 的取值无关，因此可以用 np.arange(n) 作为占位 X 精确重放。
        sklearn 缺失时返回 None，回测降级为仅报告拟合度。
        """
        try:
            import numpy as np
            from sklearn.model_selection import train_test_split
        except ImportError:                                   # pragma: no cover
            logger.warning("未安装 scikit-learn，回测降级为全样本拟合度")
            return None
        idx_tr, idx_te = train_test_split(
            np.arange(n),
            test_size=training_config.test_size(),
            stratify=np.asarray(labels),
            random_state=training_config.random_seed(),
        )
        return [int(i) for i in idx_te]

    @staticmethod
    def _top_rows(df, labels, probs, test_idx, k: int) -> list[dict]:
        """预测概率最高的 k 部剧（优先在 hold-out 测试集内取）。"""
        pool = test_idx if test_idx else range(len(labels))
        order = sorted(pool, key=lambda i: probs[i], reverse=True)[:k]
        return [
            {
                "drama_name": str(df.at[i, "drama_name"]),
                "category": str(df.at[i, "category"]),
                "prob": round(probs[i] * 100, 1),
                "is_hot": int(labels[i]),
                "heat_wan": round(float(df.at[i, "heat_value"]) / 1e4, 0),
            }
            for i in order
        ]
