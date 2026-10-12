# -*- coding: utf-8 -*-
"""
接口层 · 数据集浏览与回测
=======================================================================
GET /api/dramas            分页浏览（keyword / sort / order / limit / offset）
GET /api/dramas/<id>       单部剧详情
GET /api/backtest          部署模型在 800 部样本上的回测结果
"""
from __future__ import annotations

from flask import jsonify, request

from app import get_container
from app.api import api_bp
from app.repositories.artifact_repository import ArtifactRepository
from app.utils.errors import NotFoundError


def _int_arg(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


@api_bp.get("/dramas")
def list_dramas():
    container = get_container()
    page = container.dataset.browse(
        limit=_int_arg("limit", 20, 1, 100),
        offset=_int_arg("offset", 0, 0, 100000),
        keyword=(request.args.get("keyword") or "").strip(),
        sort=request.args.get("sort", "heat"),
        descending=request.args.get("order", "desc").lower() != "asc",
    )
    return jsonify(page)


@api_bp.get("/dramas/<drama_id>")
def drama_detail(drama_id: str):
    item = get_container().dataset.detail(drama_id)
    if item is None:
        raise NotFoundError(f"未找到剧集 {drama_id}")
    return jsonify(item)


@api_bp.get("/backtest")
def backtest():
    return jsonify(get_container().dataset.backtest())


@api_bp.get("/dataset/stats")
def dataset_stats():
    """数据集层面的分布统计，供浏览页顶部概览使用。"""
    df = ArtifactRepository.ml_dataset()
    payload = ArtifactRepository.webmodel()
    return jsonify({
        "n_samples": int(len(df)),
        "n_features": int(len(payload["models"]["full"]["features"])),
        "kpi": payload.get("kpi", {}),
        "quality": ArtifactRepository.quality_report(),
    })
