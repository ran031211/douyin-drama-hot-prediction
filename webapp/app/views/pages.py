# -*- coding: utf-8 -*-
"""
视图层 · 页面蓝图
=======================================================================
只负责「把页面壳渲染出来」，页面数据一律由前端通过 /api/* 异步获取。
这样页面模板里没有业务数据，模板改动不会牵动后端，后端改动也不会
让模板失效 —— 前后端通过 JSON 契约而非渲染上下文耦合。
"""
from __future__ import annotations

from flask import render_template

from app.views import page_bp
from config import Config

NAV = [
    {"endpoint": "pages.dashboard", "url": "/", "label": "总览"},
    {"endpoint": "pages.predict_page", "url": "/predict", "label": "新剧预测"},
    {"endpoint": "pages.model_page", "url": "/model", "label": "模型评估"},
    {"endpoint": "pages.dataset_page", "url": "/dataset", "label": "数据集"},
]


def _ctx(active: str, page_title: str) -> dict:
    return {
        "nav": NAV,
        "active": active,
        "page_title": page_title,
        "app_name": Config.APP_NAME,
        "max_genres": Config.MAX_GENRES,
    }


@page_bp.get("/")
def dashboard():
    return render_template("dashboard.html", **_ctx("pages.dashboard", "总览"))


@page_bp.get("/predict")
def predict_page():
    return render_template("predict.html", **_ctx("pages.predict_page", "新剧预测"))


@page_bp.get("/model")
def model_page():
    return render_template("model.html", **_ctx("pages.model_page", "模型评估"))


@page_bp.get("/dataset")
def dataset_page():
    return render_template("dataset.html", **_ctx("pages.dataset_page", "数据集"))
