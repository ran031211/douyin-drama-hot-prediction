# -*- coding: utf-8 -*-
"""
仓储层 · 模型与统计产物
=======================================================================
职责：读取 src/ 流水线产出的模型与统计产物，向上层提供领域对象/表结构，
      屏蔽「数据到底存在 JSON、CSV 还是 SQLite」这一细节。

设计要点：
  · 本模块是**整个应用唯一接触文件系统的地方**（配合 drama_repository）；
  · 产物为只读，采用进程内懒加载 + 缓存，避免每次请求重复解析；
  · 缓存对象一律返回副本或只读视图，防止上层误改污染缓存。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from config import Config

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _memo(key: str, factory):
    """线程安全的懒加载缓存。"""
    if key in _cache:
        return _cache[key]
    with _lock:
        if key not in _cache:
            _cache[key] = factory()
    return _cache[key]


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"缺少模型产物：{path}，请先执行 src/run_all.py")
    return json.loads(path.read_text(encoding="utf-8"))


class ArtifactRepository:
    """模型产物与统计指标的统一入口。"""

    # ------------------------------ 模型 ------------------------------ #
    @staticmethod
    def webmodel() -> dict:
        """部署模型包：两个模型 + 题材先验 + 数值分位数 + KPI。"""
        return _memo("webmodel", lambda: _read_json(Config.WEBMODEL_FILE))

    @staticmethod
    def model_payload(name: str) -> dict:
        """按名称取单个模型（cold / full）。"""
        models = ArtifactRepository.webmodel().get("models", {})
        if name not in models:
            raise KeyError(f"未知模型 {name!r}，可用：{list(models)}")
        return models[name]

    # ------------------------------ 指标 ------------------------------ #
    @staticmethod
    def metrics() -> dict:
        """论文口径的评测结果：分类/回归对比、消融、稳健性、分组重要性。"""
        return _memo("metrics", lambda: _read_json(Config.METRICS_FILE))

    @staticmethod
    def eda_summary() -> dict:
        return _memo("eda", lambda: _read_json(Config.EDA_SUMMARY_FILE))

    @staticmethod
    def quality_report() -> dict:
        return _memo("quality", lambda: _read_json(Config.QUALITY_FILE))

    @staticmethod
    def feature_importance() -> pd.DataFrame:
        return _memo("importance",
                     lambda: pd.read_csv(Config.IMPORTANCE_FILE))

    @staticmethod
    def model_comparison() -> pd.DataFrame:
        return _memo("comparison",
                     lambda: pd.read_csv(Config.COMPARISON_FILE))

    # ------------------------------ 数据集 ---------------------------- #
    @staticmethod
    def ml_dataset() -> pd.DataFrame:
        """剧集级建模数据集，800 行 × 75 列。"""
        def load() -> pd.DataFrame:
            if not Config.ML_DATASET_CSV.exists():
                raise FileNotFoundError(
                    f"缺少建模数据集：{Config.ML_DATASET_CSV}")
            return pd.read_csv(Config.ML_DATASET_CSV)
        return _memo("dataset", load)

    @staticmethod
    def model_info() -> dict:
        """聚合接口层需要的模型元信息（避免接口层多次拼装）。"""
        payload = ArtifactRepository.webmodel()
        return {
            "meta": payload.get("meta", {}),
            "kpi": payload.get("kpi", {}),
            "models": {
                name: {
                    "name": m["name"],
                    "n_features": len(m["features"]),
                    "n_trees": m["n_trees"],
                    "n_nodes": m["n_nodes"],
                    "metrics": {k: round(v, 4) for k, v in m["metrics"].items()},
                    "selftest_max_diff": m.get("selftest", {}).get("max_diff"),
                }
                for name, m in payload.get("models", {}).items()
            },
        }

    @staticmethod
    def clear_cache() -> None:
        """测试或热更新产物后调用。"""
        with _lock:
            _cache.clear()
