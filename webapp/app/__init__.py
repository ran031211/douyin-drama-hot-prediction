# -*- coding: utf-8 -*-
"""
应用工厂
=======================================================================
分层架构的组装入口，依赖方向自外向内：

    视图层 (templates/static)
        ↓  HTTP
    接口层 (app/api, app/views)      路由、参数、状态码
        ↓  调用服务
    服务层 (app/services)            业务编排
        ↓  ↓
    领域层 (app/domain)   仓储层 (app/repositories)
        ↓                    ↓
    纯算法，无框架        唯一接触文件系统

采用工厂函数而非模块级 app，是为了让「多环境配置」与「测试时替换容器」
都成为可能。
"""
from __future__ import annotations

import logging
import threading

from flask import Flask, jsonify

from app.container import ServiceContainer
from app.utils.errors import register_error_handlers
from config import Config, load_config


def create_app(config_name: str | None = None, warm: bool = True) -> Flask:
    config = load_config(config_name)

    app = Flask(__name__,
                static_folder=str(Config.PROJECT_ROOT / "webapp" / "static"),
                template_folder=str(Config.PROJECT_ROOT / "webapp" / "templates"))
    app.config.from_object(config)
    # 模板改动即时生效：否则在非 debug 启动下会被 Jinja 缓存，出现「改了没反应」
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.json.ensure_ascii = False
    app.json.sort_keys = False

    _setup_logging(app)

    # ---- 装配服务容器 ----
    container = ServiceContainer()
    app.extensions["container"] = container

    # ---- 接口层 ----
    from app.api import api_bp
    from app.views import page_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(page_bp)
    register_error_handlers(app)

    # ---- 预热：回测要遍历 800 条样本，放后台线程避免阻塞启动 ----
    if warm:
        threading.Thread(target=container.dataset.warm,
                         name="backtest-warmup", daemon=True).start()

    _log_banner(app, config)
    return app


def _setup_logging(app: Flask) -> None:
    logging.basicConfig(
        level=logging.DEBUG if app.config.get("DEBUG") else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _log_banner(app: Flask, config: type[Config]) -> None:
    logger = logging.getLogger("webapp")
    info = app.extensions["container"].model.forests
    logger.info("=" * 62)
    logger.info(" %s v%s", config.APP_NAME, config.VERSION)
    logger.info(" 冷启动模型：%d 维 / %d 棵树 / AUC %.4f",
                len(info["cold"].features), info["cold"].n_trees,
                info["cold"].metrics.get("auc", 0))
    logger.info(" 全特征模型：%d 维 / %d 棵树 / AUC %.4f",
                len(info["full"].features), info["full"].n_trees,
                info["full"].metrics.get("auc", 0))
    logger.info(" 监听地址：http://%s:%s", config.HOST, config.PORT)
    logger.info("=" * 62)


def get_container() -> ServiceContainer:
    """从当前应用上下文取服务容器。"""
    from flask import current_app
    return current_app.extensions["container"]
