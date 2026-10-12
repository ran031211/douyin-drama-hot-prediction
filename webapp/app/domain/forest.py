# -*- coding: utf-8 -*-
"""
领域层 · 梯度提升森林推理器
=======================================================================
职责：把导出的树结构还原成可推理的对象，提供 predict_proba()。

设计约束：
  1. **不依赖任何第三方库**（只用 base64 / array / struct / math），
     因此领域层可以被单独测试，也不受 xgboost / sklearn 版本影响；
  2. 推理语义必须与训练端 XGBoost 完全一致 —— 关键在于阈值比较要
     在 float32 下进行（XGBoost 内部以 float32 存特征与分裂点）。
     本实现的做法是：**加载时把分裂点一次性降为 float32**，
     调用时把输入向量降为 float32，避免逐节点做类型转换的开销。

树数组编码（由 src/export_webmodel.py 生成）：
    i : int16[] ，每 3 个一组 (特征下标, 左子, 右子)；特征下标 < 0 表示叶子
    d : int8[]  ，缺失值默认走向（1 = 走左）
    t : float64[]，分裂阈值
    v : float64[]，叶子取值
原始分数 = intercept + Σ 各树叶子值，再经 sigmoid 得到正类概率。
"""
from __future__ import annotations

import base64
import math
import struct
import sys
from array import array
from typing import Iterable, Sequence

__all__ = ["GradientBoostedForest", "to_float32"]

_F32 = struct.Struct("<f")
_IS_LITTLE = sys.byteorder == "little"


def to_float32(value: float) -> float:
    """把 Python 双精度浮点降为 float32 再升回双精度，等价于 JS 的 Math.fround。"""
    try:
        return _F32.unpack(_F32.pack(value))[0]
    except (OverflowError, struct.error):
        return value


def _decode(b64: str, typecode: str) -> array:
    """base64 → array。typecode 取 'h'(int16) / 'b'(int8) / 'd'(float64)。"""
    buf = array(typecode)
    buf.frombytes(base64.b64decode(b64))
    if not _IS_LITTLE and buf.itemsize > 1:
        buf.byteswap()
    return buf


class _Tree:
    """单棵决策树，预解析为扁平数组以贴近原始内存布局。"""

    __slots__ = ("_idx", "_miss", "_thr", "_val")

    def __init__(self, payload: dict) -> None:
        self._idx = _decode(payload["i"], "h")
        self._miss = _decode(payload["d"], "b")
        self._thr = _decode(payload["t"], "d")
        self._val = _decode(payload["v"], "d")
        # 阈值一次性降为 float32，与 XGBoost 的分裂判断对齐
        self._thr = array("d", (to_float32(x) for x in self._thr))

    def leaf_value(self, features: Sequence[float]) -> float:
        """沿树下行，返回落点叶子的取值。

        k 为**节点下标**，节点属性统一用 idx[k*3 + 0/1/2] 寻址；
        idx[k*3] < 0 表示叶子，此时 val[k] 即该叶取值。
        """
        idx, miss, thr, val = self._idx, self._miss, self._thr, self._val
        k = 0
        while True:
            base = k * 3
            feat = idx[base]
            if feat < 0:
                return val[k]
            x = features[feat]
            if x != x:                      # NaN：按缺失方向走
                go_left = miss[k]
            else:
                go_left = 1 if x < thr[k] else 0
            k = idx[base + 1] if go_left else idx[base + 2]


class GradientBoostedForest:
    """一个二分类梯度提升模型（XGBoost 的纯 Python 推理对应物）。"""

    def __init__(self, payload: dict) -> None:
        self.name: str = payload["name"]
        self.features: list[str] = list(payload["features"])
        self.feature_cn: dict[str, str] = dict(payload["feature_cn"])
        # feature_group 在产物里是**与 features 平行的列表**（而非字典），
        # feature_cn 则是字典 —— 两者格式不一致，这里统一归一为字典。
        raw_group = payload.get("feature_group") or {}
        if isinstance(raw_group, dict):
            self.feature_group: dict[str, str] = dict(raw_group)
        else:
            self.feature_group = dict(zip(self.features, raw_group))
        self.intercept: float = float(payload["intercept"])
        self.metrics: dict = dict(payload.get("metrics") or {})
        self.threshold: float = float(self.metrics.get("threshold", 0.5))
        self._trees = [_Tree(t) for t in payload["trees"]]
        self.n_trees = len(self._trees)
        self.n_nodes = int(payload.get("n_nodes") or 0)
        # SHAP 查找表：按特征分箱的均值贡献，用于局部归因
        shap = payload.get("shap") or {}
        self._shap_edges: list[list[float]] = shap.get("edges") or []
        self._shap_means: list[list[float]] = shap.get("means") or []

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def raw_score(self, features: Sequence[float]) -> float:
        """返回加性空间上的原始分数（sigmoid 之前）。"""
        score = self.intercept
        x = [to_float32(v) for v in features]
        for tree in self._trees:
            score += tree.leaf_value(x)
        return score

    def predict_proba(self, features: Sequence[float]) -> float:
        """返回爆款（正类）概率，取值 (0, 1)。"""
        if len(features) != len(self.features):
            raise ValueError(
                "特征维度不匹配：模型 %s 需要 %d 维，实际 %d 维"
                % (self.name, len(self.features), len(features))
            )
        z = self.raw_score(features)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    # ------------------------------------------------------------------ #
    # 归因
    # ------------------------------------------------------------------ #
    def local_contributions(
        self, features: Sequence[float], top_k: int = 10
    ) -> list[dict]:
        """SHAP 局部贡献近似：按取值所在分箱取该特征的均值贡献。

        这是把 SHAP 依赖图离散化为「分箱 → 平均贡献」查找表后的查表结果，
        量级与方向可用于解释单个样本，不等于逐样本精确 Shapley 值。
        """
        if not self._shap_edges:
            return []
        out: list[dict] = []
        for j, value in enumerate(features):
            edges = self._shap_edges[j] if j < len(self._shap_edges) else None
            means = self._shap_means[j] if j < len(self._shap_means) else None
            if not edges or not means or len(edges) < 2:
                continue
            bucket = 0
            while bucket < len(edges) - 2 and value >= edges[bucket + 1]:
                bucket += 1
            contrib = means[bucket] if bucket < len(means) else 0.0
            if abs(contrib) < 1e-6:
                continue
            name = self.features[j]
            out.append({
                "feature": name,
                "label": self.feature_cn.get(name, name),
                "group": self.feature_group.get(name, "其他"),
                "contribution": round(float(contrib), 4),
            })
        out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return out[:top_k]

    def describe(self) -> dict:
        """模型元信息（供接口层直接序列化）。"""
        return {
            "name": self.name,
            "n_features": len(self.features),
            "n_trees": self.n_trees,
            "n_nodes": self.n_nodes,
            "intercept": round(self.intercept, 6),
            "decision_threshold": round(self.threshold, 4),
            "metrics": {
                k: (round(v, 4) if isinstance(v, (int, float)) else v)
                for k, v in self.metrics.items()
            },
        }
