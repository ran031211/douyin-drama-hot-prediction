# -*- coding: utf-8 -*-
"""
接口层 · 元信息
=======================================================================
GET /api/health          服务与产物健康检查
GET /api/meta            概览：KPI + 模型元信息 + 选项字典
GET /api/insights/genres 题材爆款率排名与选题分档
GET /api/insights/combos 双标签组合
GET /api/insights/model  全局归因、ROC/PR 曲线、论文口径评测指标
"""
from __future__ import annotations

from flask import jsonify

from app import get_container
from app.api import api_bp
from app.domain import training_config
from config import Config


@api_bp.get("/health")
def health():
    """健康检查：报告产物可用性、训练配置同源性与回测预热状态。"""
    container = get_container()
    training = training_config.load()
    artifacts = {}
    for name in Config.REQUIRED_ARTIFACTS:
        path = getattr(Config, name)
        artifacts[name] = {"path": str(path), "exists": path.exists()}

    return jsonify({
        "status": "ok" if all(a["exists"] for a in artifacts.values()) else "degraded",
        "version": Config.VERSION,
        "artifacts": artifacts,
        "training_config": {
            "source": training.get("source"),
            "degraded": training["degraded"],
            "reason": training.get("reason"),
            "random_seed": training.get("RANDOM_SEED"),
            "test_size": training.get("TEST_SIZE"),
        },
        "backtest_ready": container.dataset.is_ready(),
    })


@api_bp.get("/meta")
def meta():
    """概览页所需的全部元信息，一次请求取回，减少前端串行等待。"""
    container = get_container()
    return jsonify({
        "overview": container.insight.overview(),
        "models": container.model.forests["cold"].describe(),
        "models_full": container.model.forests["full"].describe(),
        "genres": container.insight.genre_options(),
        "plots": container.insight.plot_stats(),
        "app": {
            "name": Config.APP_NAME,
            "subtitle": Config.APP_SUBTITLE,
            "version": Config.VERSION,
            "max_genres": Config.MAX_GENRES,
        },
    })


@api_bp.get("/insights/genres")
def genres():
    container = get_container()
    return jsonify({
        "ranking": container.insight.genre_ranking(),
        "tiers": container.insight.selection_tiers(),
    })


@api_bp.get("/insights/combos")
def combos():
    return jsonify(get_container().insight.combos())


@api_bp.get("/insights/model")
def model_insights():
    container = get_container()
    return jsonify({
        "importance": container.insight.importance_overall(),
        "curves": container.insight.roc_curves(),
        "paper_metrics": container.insight.paper_metrics(),
    })
