# -*- coding: utf-8 -*-
"""
工具层 · 统一错误处理
=======================================================================
把「Python 异常」翻译成「稳定的 JSON 错误契约」：
    {"error": {"code": "...", "message": "...", "detail": {...}}}
前端只需按 code 分支，不依赖 Flask 默认的 HTML 错误页。
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """业务异常基类。"""

    status = 400
    code = "bad_request"

    def __init__(self, message: str, detail: dict | None = None,
                 status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}
        if status is not None:
            self.status = status


class NotFoundError(ApiError):
    status = 404
    code = "not_found"


class ValidationError(ApiError):
    """请求体字段校验失败：detail 为「字段 → 原因」映射。"""

    status = 400
    code = "validation_error"

    def __init__(self, detail: dict[str, str]) -> None:
        super().__init__("请求参数校验失败", detail, status=400)


class ArtifactMissingError(ApiError):
    status = 503
    code = "artifact_missing"


def register_error_handlers(app: Flask) -> None:
    """在应用工厂中调用一次。"""

    @app.errorhandler(ApiError)
    def _api_error(exc: ApiError):
        return _payload(exc.code, exc.message, exc.detail, exc.status)

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        return _payload(
            (exc.name or "http_error").lower().replace(" ", "_"),
            exc.description or "请求失败",
            {},
            exc.code or 500,
        )

    @app.errorhandler(FileNotFoundError)
    def _missing_artifact(exc: FileNotFoundError):
        logger.error("模型产物缺失：%s", exc)
        return _payload("artifact_missing", str(exc), {}, 503)

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):                       # pragma: no cover
        logger.exception("未捕获异常")
        return _payload("internal_error", "服务内部错误", {}, 500)


def _payload(code: str, message: str, detail: dict, status: int):
    return jsonify({"error": {"code": code, "message": message, "detail": detail}}), status
