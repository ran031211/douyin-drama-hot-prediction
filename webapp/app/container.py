# -*- coding: utf-8 -*-
"""
应用装配容器
=======================================================================
集中构造服务对象并管理其生命周期，替代在各处散落的模块级单例。

依赖方向是单向的：
    Container → Service → Repository / Domain
接口层只通过 `current_app.extensions["container"]` 取服务，
不自己 new 任何东西 —— 这是本系统「可测试」的关键：
测试时替换容器即可整体换掉数据来源。
"""
from __future__ import annotations

from app.services.dataset_service import DatasetService
from app.services.feature_service import FeatureService
from app.services.insight_service import InsightService
from app.services.model_service import ModelService


class ServiceContainer:
    """服务对象的装配与持有。"""

    def __init__(self) -> None:
        # 注意构造顺序：ModelService 依赖 FeatureService
        self.feature: FeatureService = FeatureService()
        self.model: ModelService = ModelService(self.feature)
        self.insight: InsightService = InsightService()
        self.dataset: DatasetService = DatasetService(self.model)

    def warm_up(self) -> dict:
        """预热耗时步骤，返回自检摘要（供启动日志使用）。"""
        summary = {
            "cold": self.model.forests["cold"].describe(),
            "full": self.model.forests["full"].describe(),
        }
        self.dataset.warm()
        return summary
