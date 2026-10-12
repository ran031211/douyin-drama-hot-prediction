# -*- coding: utf-8 -*-
"""
视图层 · 页面蓝图
=======================================================================
四个页面：总览 / 新剧预测 / 模型评估 / 数据集浏览。
页面只渲染布局壳，页面数据由前端调用 /api/* 异步获取 ——
模板不持有业务数据，后端接口变化不会让模板失效。
"""
from __future__ import annotations

from flask import Blueprint

page_bp = Blueprint("pages", __name__)

from app.views import pages      # noqa: E402,F401  注册路由
