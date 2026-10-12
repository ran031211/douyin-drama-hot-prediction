# -*- coding: utf-8 -*-
"""
webapp 配置层
=======================================================================
所有可变参数集中在此，代码其余部分不出现硬编码路径与魔法值。

分层约定：
    配置层（本文件）→ 应用工厂（app/__init__.py）→ 接口层 → 服务层
    → 领域层 / 仓储层

数据来源说明：本服务不重新训练模型，只消费 src/ 流水线已产出的
「模型产物」（outputs/），因此配置项以产物路径为主。
"""
from __future__ import annotations

import os
from pathlib import Path


class Config:
    """基础配置。"""

    # ---------- 应用元信息 ----------
    APP_NAME = "微短剧热度预测系统"
    APP_SUBTITLE = "基于 XGBoost 的抖音 AI 微短剧热度预测与影响因素分析"
    VERSION = "1.0.0"

    # ---------- 路径 ----------
    # webapp/config.py → 上溯两级即项目根
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    DASHBOARD_DIR = OUTPUTS_DIR / "dashboard"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

    # ---------- 模型产物（只读） ----------
    WEBMODEL_FILE = DASHBOARD_DIR / "webmodel.json"
    METRICS_FILE = OUTPUTS_DIR / "model_metrics.json"
    IMPORTANCE_FILE = OUTPUTS_DIR / "feature_importance.csv"
    COMPARISON_FILE = OUTPUTS_DIR / "model_comparison.csv"
    EDA_SUMMARY_FILE = OUTPUTS_DIR / "eda_summary.json"
    QUALITY_FILE = OUTPUTS_DIR / "data_quality_report.json"

    # ---------- 数据集 ----------
    ML_DATASET_CSV = DATA_PROCESSED_DIR / "ml_dataset.csv"
    SQLITE_FILE = OUTPUTS_DIR / "douyin_drama.sqlite"

    # ---------- 推理服务 ----------
    # 冷启动模型上线前即可计算，是「新剧实时预测」的默认口径；
    # 全量模型额外需要上线后的评论数据。
    DEFAULT_MODEL = "cold"
    # 爆款阈值口径：热度数值进入全量 Top X%
    HOT_QUANTILE = 0.80
    # 单次请求返回的局部归因因子数量
    LOCAL_FACTOR_TOPK = 10

    # ---------- 请求限制 ----------
    MAX_CONTENT_LENGTH = 256 * 1024        # 请求体上限 256KB
    MAX_SYNOPSIS_LEN = 4000                # 简介长度上限（字符）
    MAX_TITLE_LEN = 80
    MAX_GENRES = 3                         # 题材标签最多选 3 个（与训练口径一致）

    # ---------- 服务运行 ----------
    HOST = os.environ.get("WEBAPP_HOST", "0.0.0.0")
    PORT = int(os.environ.get("WEBAPP_PORT", "5000"))
    DEBUG = os.environ.get("WEBAPP_DEBUG", "0") == "1"
    JSON_AS_ASCII = False

    # 数据目录只读校验：缺产物时给出明确报错，而不是让请求 500
    REQUIRED_ARTIFACTS = ("WEBMODEL_FILE", "METRICS_FILE", "ML_DATASET_CSV")


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = "development"


class ProductionConfig(Config):
    DEBUG = False
    ENV = "production"


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}


def load_config(name: str | None = None) -> type[Config]:
    """按名称取配置类，未指定时读环境变量 WEBAPP_ENV。"""
    key = name or os.environ.get("WEBAPP_ENV", "default")
    return CONFIG_MAP.get(key, DevelopmentConfig)
