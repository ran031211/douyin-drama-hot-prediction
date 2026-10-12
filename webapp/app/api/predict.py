# -*- coding: utf-8 -*-
"""
接口层 · 新剧实时预测
=======================================================================
POST /api/predict
    body: {
        "title":     "剧名",
        "synopsis":  "剧情简介",
        "genres":    ["逆袭", "都市"],      // 最多 3 个
        "episodes":  120,
        // 以下为可选项：提供后追加「全特征版」结论
        "hot_rating_mean": 9.2, "hot_digg_mean": 38, "ep1_digg_mean": 46,
        "ep1_pos_ratio": 0.42, ...
    }
    resp: 冷启动结论 + （可选）全特征结论 + 差异 + 局部归因 + 题材先验

接口层只做三件事：取 JSON → 交给模式层校验 → 调模型服务。
"""
from __future__ import annotations

from flask import jsonify, request

from app import get_container
from app.api import api_bp
from app.schemas.predict_schema import parse_predict_payload
from app.utils.errors import ValidationError
from config import Config


@api_bp.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    spec = parse_predict_payload(payload)          # 失败由错误处理层转 400

    container = get_container()
    result = container.model.predict(spec, top_k=Config.LOCAL_FACTOR_TOPK)

    # 题材先验：把用户所选题材的历史爆款率一并返回，便于与预测结论交叉验证
    stat_map = {g["genre"]: g for g in container.insight.genre_options()}
    prior = [stat_map[g] for g in spec.genres if g in stat_map]

    result["input"] = {
        "title": spec.title,
        "genres": spec.genres,
        "episodes": spec.episodes,
        "synopsis_len": len(spec.synopsis),
        "mode": "full" if result["full"] else "cold",
    }
    result["genre_prior"] = prior
    return jsonify(result)


@api_bp.post("/extract")
def extract():
    """从剧名与简介中实时抽取结构标记与爽点标签。

    供前端在输入过程中做即时反馈：词典留在服务端，
    前端不需要维护一份会与训练侧漂移的关键词表。
    """
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title") or "")[:Config.MAX_TITLE_LEN]
    synopsis = str(payload.get("synopsis") or "")[:Config.MAX_SYNOPSIS_LEN]

    service = get_container().feature
    return jsonify({
        "plot_tags": service.extract_plot_tags(synopsis),
        "flags": service.extract_flags(title, synopsis),
    })


@api_bp.post("/predict/batch")
def predict_batch():
    """批量预测：一次提交多部候选剧，用于选题比稿。"""
    payload = request.get_json(silent=True) or {}
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValidationError({"items": "应为非空数组"})
    if len(items) > 20:
        raise ValidationError({"items": "单次最多 20 部"})

    container = get_container()
    out = []
    for idx, item in enumerate(items):
        try:
            spec = parse_predict_payload(item)
        except ValidationError as exc:
            out.append({"index": idx, "error": exc.detail})
            continue
        res = container.model.predict(spec, top_k=0)
        out.append({
            "index": idx,
            "title": spec.title,
            "probability_pct": res["cold"]["probability_pct"],
            "verdict": res["cold"]["verdict"],
            "level": res["cold"]["level"],
            "full_probability_pct": (res["full"] or {}).get("probability_pct"),
        })
    out.sort(key=lambda d: d.get("probability_pct", -1), reverse=True)
    for rank, row in enumerate(out, 1):
        row["rank"] = rank
    return jsonify({"items": out})
