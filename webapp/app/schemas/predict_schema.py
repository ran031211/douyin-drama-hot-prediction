# -*- coding: utf-8 -*-
"""
模式层 · 请求校验
=======================================================================
接口层不直接读 request.json，而是交给本模块解析成领域对象。
好处：校验规则只有一处，且接口层保持「薄」——只做 HTTP 语义转换。

校验失败统一抛 ValidationError，由错误处理层映射为 400 + 字段级提示。
"""
from __future__ import annotations

from app.domain.training_config import genres as known_genres
from app.services.feature_service import DramaSpec
from app.utils.errors import ValidationError
from config import Config


def _as_float(payload: dict, key: str, lo: float, hi: float,
              detail: dict[str, str]) -> float | None:
    raw = payload.get(key)
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        detail[key] = "应为数值"
        return None
    if not (lo <= value <= hi):
        detail[key] = f"应在 {lo} ~ {hi} 之间"
        return None
    return value


def parse_predict_payload(payload: dict) -> DramaSpec:
    """把请求体解析为 DramaSpec，并做范围校验。"""
    if not isinstance(payload, dict):
        raise ValidationError({"body": "请求体应为 JSON 对象"})

    detail: dict[str, str] = {}

    title = str(payload.get("title") or "").strip()
    if not title:
        detail["title"] = "剧名不可为空"
    elif len(title) > Config.MAX_TITLE_LEN:
        detail["title"] = f"剧名不超过 {Config.MAX_TITLE_LEN} 字"

    synopsis = str(payload.get("synopsis") or "").strip()
    if len(synopsis) > Config.MAX_SYNOPSIS_LEN:
        detail["synopsis"] = f"简介不超过 {Config.MAX_SYNOPSIS_LEN} 字"

    raw_genres = payload.get("genres") or []
    if isinstance(raw_genres, str):
        raw_genres = [g for g in raw_genres.replace("，", ",").split(",") if g.strip()]
    if not isinstance(raw_genres, list):
        detail["genres"] = "应为题材数组"
        raw_genres = []

    valid = set(known_genres())
    picked: list[str] = []
    for g in raw_genres:
        name = str(g).strip()
        if not name:
            continue
        if valid and name not in valid:
            detail["genres"] = f"未知题材 {name!r}"
            continue
        if name not in picked:
            picked.append(name)
    if len(picked) > Config.MAX_GENRES:
        picked = picked[:Config.MAX_GENRES]

    episodes = _as_float(payload, "episodes", 1, 9999, detail)

    # 上线后观测（可选）：给了就走全量模型
    engagement = {
        "hot_rating_mean": _as_float(payload, "hot_rating_mean", 0, 10, detail),
        "hot_rating_good": _as_float(payload, "hot_rating_good", 0, 1, detail),
        "hot_rating_bad": _as_float(payload, "hot_rating_bad", 0, 1, detail),
        "hot_digg_mean": _as_float(payload, "hot_digg_mean", 0, 1e7, detail),
        "ep1_digg_mean": _as_float(payload, "ep1_digg_mean", 0, 1e7, detail),
        "ep1_hot_ratio": _as_float(payload, "ep1_hot_ratio", 0, 1, detail),
        "ep1_urge_ratio": _as_float(payload, "ep1_urge_ratio", 0, 1, detail),
        "ep1_pos_ratio": _as_float(payload, "ep1_pos_ratio", 0, 1, detail),
        "ep1_neg_ratio": _as_float(payload, "ep1_neg_ratio", 0, 1, detail),
    }

    if detail:
        raise ValidationError(detail)

    return DramaSpec(
        title=title,
        synopsis=synopsis,
        genres=picked,
        episodes=int(episodes or 0),
        **engagement,
    )
