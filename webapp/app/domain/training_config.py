# -*- coding: utf-8 -*-
"""
领域层 · 训练侧配置桥接
=======================================================================
题材标签、爽点关键词、情感词表、随机种子与划分比例在训练侧定义于
src/config.py。服务侧**不复制**这些常量（复制必然随时间漂移），
而是运行时按文件路径动态加载，保证线上口径与训练口径永远同源。

这是保证「回测能复现出与论文一致的指标」的前提：
随机种子与测试集比例必须与训练时逐字一致，否则 hold-out 划分不同，
复算出的 AUC 自然对不上。
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from typing import Any

from config import Config

# 只搬运服务端真正用得到的常量，避免把整个训练配置暴露出去
_WANTED = (
    "GENRES", "PLOT_KEYWORDS", "POSITIVE_WORDS", "NEGATIVE_WORDS", "URGE_WORDS",
    "GENRE_MIN_COUNT", "RANDOM_SEED", "TEST_SIZE", "HOT_QUANTILE",
)


def _empty(reason: str) -> dict[str, Any]:
    """兜底：仅保证服务可启动，不保证与训练口径一致。"""
    base = {k: ([] if k.endswith("S") or k.endswith("WORDS") else {}) for k in _WANTED}
    base.update({
        "GENRES": [], "PLOT_KEYWORDS": {},
        "RANDOM_SEED": 42, "TEST_SIZE": 0.2, "HOT_QUANTILE": 0.8,
        "degraded": True, "reason": reason, "source": None,
    })
    return base


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """从 src/config.py 动态加载训练侧常量。"""
    src_config = Config.PROJECT_ROOT / "src" / "config.py"
    if not src_config.exists():
        return _empty(f"未找到训练侧配置 {src_config}")

    spec = importlib.util.spec_from_file_location("_train_config", src_config)
    if spec is None or spec.loader is None:
        return _empty("无法解析训练侧配置")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                                  # pragma: no cover
        return _empty(f"训练侧配置导入失败：{exc}")

    data = {name: getattr(module, name) for name in _WANTED
            if hasattr(module, name)}
    data.update({"degraded": False, "reason": None, "source": str(src_config)})
    return data


# ---------------------------------------------------------------------- #
# 便捷访问
# ---------------------------------------------------------------------- #
def genres() -> list[str]:
    return list(load().get("GENRES", []))


def plot_keywords() -> dict[str, list[str]]:
    return dict(load().get("PLOT_KEYWORDS", {}))


def random_seed() -> int:
    return int(load().get("RANDOM_SEED", 42))


def test_size() -> float:
    return float(load().get("TEST_SIZE", 0.2))
