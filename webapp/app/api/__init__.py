# -*- coding: utf-8 -*-
"""
接口层 · API 蓝图注册
=======================================================================
按资源切分三个子模块，各自只负责「解析请求 → 调服务 → 序列化响应」：
    meta.py     系统与统计元信息
    predict.py  新剧实时预测
    dataset.py  数据集浏览与回测
"""
from __future__ import annotations

from flask import Blueprint

api_bp = Blueprint("api", __name__, url_prefix="/api")

from app.api import dataset, meta, predict      # noqa: E402,F401  注册路由
