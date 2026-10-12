"""
=======================================================================
 模块四·附：部署模型导出（XGBoost 树结构编译为可移植 JSON）
-----------------------------------------------------------------------
 把训练好的 XGBoost 树模型"扁平化"成紧凑二进制，base64 内嵌进 HTML，
 让服务端在**不加载任何 .pkl 模型对象**的情况下对任意输入实时给出爆款概率。

 为什么要做两个部署模型？
   题目是"热度预测 + 影响因素分析"，而本研究的关键结论之一正是：
   仅靠内容属性（上线前可得）预测力有限，一旦加入观众反馈信号，
   判别能力显著提升。
   因此服务端并列部署两个模型，让使用者直观看到"追加反馈后的预测提升"：
     A  冷启动版   仅内容属性（40 维）
     B  全特征版   内容属性 + 观众反馈（50 维）

 XGBoost 与随机森林的导出差异：
   · 叶子值位于 margin（对数几率）空间，需累加后再过 sigmoid；
   · 分裂判据是 `x < threshold`（sklearn 是 `x <= threshold`）；
   · 存在 base_score 截距，本脚本用"margin − 叶子和"中位数标定，
     避免不同 XGBoost 版本对 base_score 变换口径不一致的坑。

 产出：outputs/dashboard/webmodel.json
 用法：python src/export_webmodel.py
=======================================================================
"""
from __future__ import annotations

import base64
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DASHBOARD, DATA_PROCESSED, RANDOM_SEED, TEST_SIZE  # noqa: E402
from preprocess import (FEAT_ALL, FEAT_CONTENT, FEAT_ENGAGE, FEAT_RATING,  # noqa: E402
                        FEAT_TEXT, FEATURE_CN, GROUP_OF)
from model import xgb_clf  # noqa: E402

# ---------------------------------------------------------------- 部署特征
# 消费方：webapp/app/services/feature_service.py（Flask 服务的特征构造）
# 与服务端实现 webapp/app/domain/forest.py 的走树逻辑字段一一对应
WEB_FEAT_FEEDBACK = ["hot_rating_mean", "hot_rating_good", "hot_rating_bad",
                     "log_hot_digg_sum", "hot_digg_mean",
                     "ep1_digg_mean", "ep1_hot_ratio",
                     "ep1_urge_ratio", "ep1_pos_ratio", "ep1_neg_ratio"]
WEB_FEAT_CONTENT = FEAT_CONTENT                      # 40 维：集数/题材/爽点/简介
WEB_FEAT_FULL = WEB_FEAT_CONTENT + WEB_FEAT_FEEDBACK  # 50 维

FEATURE_GROUP = {f: GROUP_OF[f] for f in WEB_FEAT_FULL}
FEATURE_GROUP.update({f: "内容属性" for f in WEB_FEAT_CONTENT})

CANDIDATES = [
    dict(n_estimators=120, max_depth=3, learning_rate=0.07),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05),
    dict(n_estimators=300, max_depth=4, learning_rate=0.04),
    dict(n_estimators=400, max_depth=5, learning_rate=0.035),
]
SIZE_BUDGET_MB = 1.5
AUC_TOL = 0.004
BYTES_PER_NODE = 30          # int16×3 + int8 + float64×2 ≈ 30B


# ---------------------------------------------------------------- 编码工具
def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 得分区间（百分比）。

    为何不用正态近似 p ± z·sqrt(p(1-p)/n)？
    题材是小样本，像"宫廷"只有 19 部，且 p 靠近 0 / 1 时正态近似会给出
    越界区间（甚至负数）。Wilson 区间在 n 小、p 极端时仍稳定，
    是本页"样本量警示"的依据。
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - half) * 100.0, min(1.0, ctr + half) * 100.0)


def _genre_tier(rate: float, ci_lo: float, base: float = 20.0) -> str:
    """题材分档（四条判据全部写在页面里，可复核）。

      首选  95% 置信区间下界 > 全站基准 —— 统计上显著更高的"真红利"
      优先  点估计 ≥ 基准 × 1.2        —— 明显偏高，但样本量尚未支撑显著
      中性  基准 × 0.8 ≤ 点估计 < 基准 × 1.2 —— 与基准无实质差异
      规避  点估计 < 基准 × 0.8        —— 明显偏低

    为何用「相对基准的比例」而不是固定的 ±5 个百分点？
    热度数值本身高度集中（变异系数仅 6%），题材爆款率整体挤在 13%~50% 之间，
    固定百分点带会把大量题材塞进同一档；按比例缩放能保留档间的可比性。
    """
    if ci_lo > base:
        return "首选"
    if rate >= base * 1.2:
        return "优先"
    if rate < base * 0.8:
        return "规避"
    return "中性"


def export_xgb_trees(model, feat_names) -> tuple[list[dict], int]:
    """把 XGBoost 森林导成紧凑二进制。

    每个节点：i = int16×3 [特征下标, 左子, 右子]（特征下标 -1 表示叶子）
              d = int8    缺失值默认走向（1=左，0=右）
              t = float64 分裂阈值（左子条件 x < t）
              v = float64 叶子值（margin 空间），非叶子填 0

    注意：XGBoost 在 DataFrame 上训练后，trees_to_dataframe() 的 Feature 列
    存的是**真实特征名**（如 "log_episodes"）而非 "f12"，必须显式映射到下标。
    """
    booster = model.get_booster()
    names = booster.feature_names if booster.feature_names else list(feat_names)
    fmap = {n: i for i, n in enumerate(names)}
    df = booster.trees_to_dataframe()
    trees, total = [], 0
    for _, sub in df.groupby("Tree", sort=True):
        sub = sub.reset_index(drop=True)
        id2i = {nid: i for i, nid in enumerate(sub["ID"])}
        n = len(sub)
        idx = np.full((n, 3), -1, dtype=np.int16)
        miss = np.zeros(n, dtype=np.int8)
        thr = np.zeros(n, dtype=np.float64)
        val = np.zeros(n, dtype=np.float64)
        for i, row in sub.iterrows():
            if str(row["Feature"]) == "Leaf":
                val[i] = float(row["Gain"])
            else:
                fname = str(row["Feature"])
                if fname not in fmap:
                    raise KeyError(f"树节点引用了未知特征：{fname}")
                idx[i, 0] = fmap[fname]
                idx[i, 1] = id2i[row["Yes"]]
                idx[i, 2] = id2i[row["No"]]
                miss[i] = 1 if row["Missing"] == row["Yes"] else 0
                thr[i] = float(row["Split"])
        trees.append({"n": int(n), "i": _b64(idx.ravel()), "d": _b64(miss),
                      "t": _b64(thr), "v": _b64(val)})
        total += n
    return trees, total


def decode(trees):
    out = []
    for t in trees:
        idx = np.frombuffer(base64.b64decode(t["i"]), dtype=np.int16).reshape(-1, 3)
        miss = np.frombuffer(base64.b64decode(t["d"]), dtype=np.int8)
        thr = np.frombuffer(base64.b64decode(t["t"]), dtype=np.float64)
        val = np.frombuffer(base64.b64decode(t["v"]), dtype=np.float64)
        out.append((idx, miss, thr, val))
    return out


def raw_margin(dec, row: np.ndarray) -> float:
    """复刻部署推理：x < threshold 走左子，缺失按默认方向。

    关键：XGBoost 在 **float32 域**比较。必须把特征值与分裂阈值都降到
    float32 再比，否则 `np.float32(x) < float64(t)` 会被 NumPy 提升为
    float64 比较，恰好贴着阈值的样本会走错分支——实测 offset 标准差
    从 8e-7 恶化到 1.5e-2，概率最大偏差 1.8e-2。
    服务端实现（Python）对应使用 `np.float32` 对两侧做同样的舍入，
    早期 JS 实现使用 `Math.fround`。
    """
    s = 0.0
    f32 = np.float32
    for idx, miss, thr, val in dec:
        k = 0
        while idx[k, 0] >= 0:
            x = row[idx[k, 0]]
            if np.isnan(x):
                k = idx[k, 1] if miss[k] else idx[k, 2]
            else:
                k = idx[k, 1] if f32(x) < f32(thr[k]) else idx[k, 2]
        s += val[k]
    return s


def fit_intercept(model, dec, X) -> float:
    """标定 base_score 截距：margin − 叶子和，取中位数。

    走 sklearn 包装器的 predict(output_margin=True)，它接受 DataFrame 并保留
    特征名，避免 DMatrix 因缺列名报错。
    """
    margin = np.asarray(model.predict(X, output_margin=True), dtype=float)
    arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
    leaf = np.array([raw_margin(dec, arr[i]) for i in range(len(arr))])
    return float(np.median(margin - leaf))


def export_curves(y_true, prob, n_points=120):
    fpr, tpr, _ = roc_curve(y_true, prob)
    pi, ri, _ = precision_recall_curve(y_true, prob)
    k1 = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    k2 = np.unique(np.linspace(0, len(pi) - 1, n_points).astype(int))
    return {"roc": {"x": [round(float(v), 4) for v in fpr[k1]],
                    "y": [round(float(v), 4) for v in tpr[k1]]},
            "pr": {"x": [round(float(v), 4) for v in ri[k2]],
                   "y": [round(float(v), 4) for v in pi[k2]]}}


def export_shap_lut(X: pd.DataFrame, prob_model, nbins: int = 8):
    """SHAP 查找表：连续特征分箱 → 箱内平均 SHAP 值（服务端查表即得贡献）"""
    try:
        import shap
    except ImportError:
        print("  （未安装 shap，跳过特征贡献查找表）")
        return None, None
    expl = shap.TreeExplainer(prob_model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]

    edges, means = [], []
    for j in range(X.shape[1]):
        v = X.iloc[:, j].values.astype(float)
        s = sv[:, j]
        qs = np.unique(np.quantile(v, np.linspace(0, 1, nbins + 1)))
        if len(qs) < 3:
            qs = np.array([v.min() - 1e-9, v.max() + 1e-9])
        b = np.clip(np.searchsorted(qs, v, side="right") - 1, 0, len(qs) - 2)
        edges.append([float(x) for x in qs])
        means.append([float(s[b == k].mean()) if (b == k).any() else 0.0
                      for k in range(len(qs) - 1)])
    gimp = np.abs(sv).mean(axis=0)
    tot = float(gimp.sum()) or 1.0
    importance = {X.columns[j]: round(float(gimp[j]) / tot * 100, 3)
                  for j in range(X.shape[1])}
    return {"edges": edges, "means": means}, importance


def pick_config(Xtr, ytr, Xte, yte, tag):
    cands = []
    print(f"  [{tag}] {'配置':<30} {'测试AUC':>8} {'节点数':>8} {'体积':>8}")
    for cfg in CANDIDATES:
        m = xgb_clf(random_state=RANDOM_SEED, **cfg)
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        nodes = int(m.get_booster().trees_to_dataframe().shape[0])
        mb = nodes * BYTES_PER_NODE / 1024 / 1024 * 1.34
        cands.append((auc, cfg, m, nodes, mb))
        print(f"       树{cfg['n_estimators']:>3} 深{cfg['max_depth']} 学习率{cfg['learning_rate']}"
              f"     {auc:>8.4f} {nodes:>8,} {mb:>7.2f}M"
              f"{'' if mb <= SIZE_BUDGET_MB else '  ✗超预算'}")
    best = max(c[0] for c in cands)
    elig = [c for c in cands if c[0] >= best - AUC_TOL and c[4] <= SIZE_BUDGET_MB]
    if not elig:
        raise RuntimeError("候选配置全部超出体积预算")
    elig.sort(key=lambda c: c[4])
    return elig[0]


def build_payload(name, feat_cols, cfg, model, Xtr, ytr, Xte, yte, with_shap=True):
    """打包单个部署模型：树 + 截距 + 指标 + 曲线 + SHAP 表"""
    trees, n_nodes = export_xgb_trees(model, feat_cols)
    dec = decode(trees)
    intercept = fit_intercept(model, dec, Xte)

    # 自检：解码后重算，必须与 XGBoost 原生预测一致
    sub = Xte.values.astype(float)[:300]
    got = expit(np.array([raw_margin(dec, r) for r in sub]) + intercept)
    ref = model.predict_proba(Xte.iloc[:300])[:, 1]
    dmax, dmean = float(np.abs(ref - got).max()), float(np.abs(ref - got).mean())

    oof = np.zeros(len(Xtr))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED).split(Xtr, ytr):
        mm = xgb_clf(random_state=RANDOM_SEED, **cfg)
        mm.fit(Xtr.iloc[tr], ytr.iloc[tr])
        oof[va] = mm.predict_proba(Xtr.iloc[va])[:, 1]
    prec, rec, grid = precision_recall_curve(ytr, oof)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    thr = float(grid[int(np.nanargmax(f1s[:-1]))])

    prob = model.predict_proba(Xte)[:, 1]
    metrics = {"auc": float(roc_auc_score(yte, prob)),
               "ap": float(average_precision_score(yte, prob)),
               "f1": float(f1_score(yte, (prob >= thr).astype(int))),
               "accuracy": float(accuracy_score(yte, (prob >= thr).astype(int))),
               "brier": float(brier_score_loss(yte, prob)),
               "threshold": round(thr, 4)}

    lut, gimp = (None, None)
    if with_shap:
        lut, gimp = export_shap_lut(Xtr, model)
    top = sorted(gimp.items(), key=lambda kv: -kv[1])[:12] if gimp else []

    return {
        "name": name,
        "features": feat_cols,
        "feature_cn": {f: FEATURE_CN.get(f, f) for f in feat_cols},
        "feature_group": [FEATURE_GROUP[f] for f in feat_cols],
        "n_trees": len(trees), "n_nodes": n_nodes,
        "trees": trees, "intercept": round(intercept, 6),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "curves": export_curves(yte.values, prob),
        "shap": lut,
        "importance": {"keys": [k for k, _ in top],
                       "labels": [FEATURE_CN.get(k, k) for k, _ in top],
                       "values": [v for _, v in top],
                       "groups": [FEATURE_GROUP.get(k, "其他") for k, _ in top]},
        "selftest": {"max_diff": dmax, "mean_diff": dmean},
    }


def main():
    print("=" * 70)
    print(" 部署模型导出：XGBoost 树结构 → 可移植 JSON")
    print("=" * 70)

    df = pd.read_parquet(DATA_PROCESSED / "ml_dataset.parquet")
    y = df["is_hot"].astype(int)
    Xc = df[WEB_FEAT_CONTENT].astype(float).fillna(0)
    Xf = df[WEB_FEAT_FULL].astype(float).fillna(0)
    print(f"  样本 {len(df):,} 部 | 冷启动特征 {len(WEB_FEAT_CONTENT)} 维 | "
          f"全特征 {len(WEB_FEAT_FULL)} 维")

    Xc_tr, Xc_te, y_tr, y_te = train_test_split(Xc, y, test_size=TEST_SIZE,
                                                stratify=y, random_state=RANDOM_SEED)
    Xf_tr = Xf.loc[Xc_tr.index]
    Xf_te = Xf.loc[Xc_te.index]

    print("\n  [1/4] 冷启动模型（仅内容属性）配置筛选 ...")
    auc_c, cfg_c, m_c, n_c, _ = pick_config(Xc_tr, y_tr, Xc_te, y_te, "冷启动版")
    print(f"        → 树{cfg_c['n_estimators']} 深{cfg_c['max_depth']}，"
          f"AUC {auc_c:.4f}，节点 {n_c:,}")

    print("\n  [2/4] 全特征模型（内容 + 观众反馈）配置筛选 ...")
    auc_f, cfg_f, m_f, n_f, _ = pick_config(Xf_tr, y_tr, Xf_te, y_te, "全特征版")
    print(f"        → 树{cfg_f['n_estimators']} 深{cfg_f['max_depth']}，"
          f"AUC {auc_f:.4f}，节点 {n_f:,}")
    print(f"        观众反馈带来的 AUC 增量：{auc_f - auc_c:+.4f}")

    print("\n  [3/4] 导出树结构与自检 ...")
    t0 = time.time()
    model_a = build_payload("冷启动版（仅内容属性）", WEB_FEAT_CONTENT, cfg_c, m_c,
                            Xc_tr, y_tr, Xc_te, y_te, with_shap=True)
    print(f"        冷启动版：{model_a['n_nodes']:,} 节点 | "
          f"自检最大偏差 {model_a['selftest']['max_diff']:.2e}")
    model_b = build_payload("全特征版（内容 + 观众反馈）", WEB_FEAT_FULL, cfg_f, m_f,
                            Xf_tr, y_tr, Xf_te, y_te, with_shap=True)
    print(f"        全特征版：{model_b['n_nodes']:,} 节点 | "
          f"自检最大偏差 {model_b['selftest']['max_diff']:.2e}（{time.time()-t0:.0f}s）")
    for m in (model_a, model_b):
        if m["selftest"]["max_diff"] >= 1e-6:
            raise RuntimeError(
                f"{m['name']} 的编解码链路自检未通过（{m['selftest']['max_diff']:.2e}），"
                f"部署概率将不可信")

    print("\n  [4/4] 打包数据集画像与题材先验 ...")
    d = pd.read_parquet(DATA_PROCESSED / "ml_dataset.parquet")
    kpi = {
        "dramas": int(len(d)),
        "hot_rate": round(float(d["is_hot"].mean()) * 100, 2),
        "median_heat_wan": int(d["heat_value"].median() / 1e4),
        "median_episodes": int(d["episodes"].median()),
        "median_rating": round(float(d["hot_rating_mean"].median()), 2),
        "median_ep1_digg": round(float(d["ep1_digg_mean"].median()), 1),
        "hot_threshold_wan": int(d["heat_threshold"].iloc[0] / 1e4),
        "genres": int(d["n_genre_tags"].sum()),
    }
    base_rate = kpi["hot_rate"]
    cat_stats = {}
    for g in [c[4:] for c in d.columns if c.startswith("tag_")]:
        m = d[f"tag_{g}"].astype(bool)
        if m.sum() < 8:
            continue
        ns, k = int(m.sum()), int(d.loc[m, "is_hot"].sum())
        lo, hi = _wilson(k, ns)
        cat_stats[g] = {
            "n": ns,
            "k": k,
            "hot_rate": round(k / ns * 100, 2),
            "ci": [round(lo, 1), round(hi, 1)],
            "tier": _genre_tier(k / ns * 100, lo, base_rate),
            "heat_wan": int(d.loc[m, "heat_value"].median() / 1e4),
            "episodes": float(d.loc[m, "episodes"].median()),
            "rating": round(float(d.loc[m, "hot_rating_mean"].median()), 2),
            "ep1_digg": float(d.loc[m, "ep1_digg_mean"].median()),
            "q": [float(d.loc[m, "heat_value"].quantile(i / 20)) for i in range(21)],
        }

    # 双标签组合：category 本身是多标签，组合往往是比单标签更强的前置信号
    tag_cols = [c for c in d.columns if c.startswith("tag_")]
    combos = []
    for i in range(len(tag_cols)):
        for j in range(i + 1, len(tag_cols)):
            a, b = tag_cols[i][4:], tag_cols[j][4:]
            m = d[tag_cols[i]].astype(bool) & d[tag_cols[j]].astype(bool)
            n = int(m.sum())
            if n < 15:
                continue
            k = int(d.loc[m, "is_hot"].sum())
            lo, _ = _wilson(k, n)
            combos.append({"a": a, "b": b, "n": n,
                           "hot_rate": round(k / n * 100, 1),
                           "ci_lo": round(lo, 1),
                           "tier": _genre_tier(k / n * 100, lo, base_rate)})
    combos.sort(key=lambda r: (-r["hot_rate"], -r["n"]))
    combo_stats = {
        "base": base_rate,
        "min_n": 15,
        "best": combos[:8],
        "worst": list(reversed(combos[-6:])),
        "n_pairs": len(combos),
    }

    tier_order = {"首选": 0, "优先": 1, "中性": 2, "规避": 3}
    genre_rank = sorted(cat_stats.items(), key=lambda kv: (-kv[1]["hot_rate"], -kv[1]["n"]))
    genre_tier = {t: [g for g, v in genre_rank if v["tier"] == t]
                  for t in sorted(tier_order, key=lambda x: tier_order[x])}

    plot_stats = {}
    for c in [c for c in d.columns if c.startswith("plot_")]:
        m = d[c].astype(float) > 0
        plot_stats[c[5:]] = {"n": int(m.sum()),
                             "hot_rate": round(float(d.loc[m, "is_hot"].mean()) * 100, 2)}
    nums = {}
    for c in WEB_FEAT_FULL:
        s = d[c].astype(float)
        nums[c] = {"p10": float(s.quantile(.10)), "p50": float(s.quantile(.50)),
                   "p90": float(s.quantile(.90)), "max": float(s.max())}

    payload = {
        "meta": {"title": "抖音微短剧热度预测工作台",
                 "topic": "基于 XGBoost 的抖音微短剧热度预测与影响因素分析",
                 "n_samples": int(len(d)), "test_size": TEST_SIZE,
                 "hot_quantile": 0.80},
        "models": {"cold": model_a, "full": model_b},
        "importance_overall": {
            "labels": model_b["importance"]["labels"],
            "values": model_b["importance"]["values"],
            "groups": model_b["importance"]["groups"]},
        "kpi": kpi,
        "cat_stats": cat_stats,
        "genre_tier": genre_tier,
        "combo_stats": combo_stats,
        "plot_stats": plot_stats,
        "num_stats": nums,
        "feature_group": FEATURE_GROUP,
    }
    out = DASHBOARD / "webmodel.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    size = out.stat().st_size / 1024 / 1024
    print(f"        → {out}  ({size:.2f} MB)")
    print(f"        题材分档："
          + " / ".join(f"{t} {'、'.join(v)}" for t, v in genre_tier.items() if v))
    print(f"        双标签组合 {combo_stats['n_pairs']} 对（样本 ≥{combo_stats['min_n']}）"
          f"｜最优 {combo_stats['best'][0]['a']}+{combo_stats['best'][0]['b']} "
          f"{combo_stats['best'][0]['hot_rate']}% ｜最差 "
          f"{combo_stats['worst'][0]['a']}+{combo_stats['worst'][0]['b']} "
          f"{combo_stats['worst'][0]['hot_rate']}%")
    print("=" * 70)
    print(f"  冷启动版 AUC {model_a['metrics']['auc']:.4f} → "
          f"全特征版 AUC {model_b['metrics']['auc']:.4f}")
    print(f"  推理链路：表单输入 → JS 计算特征 → 遍历 {model_a['n_trees']}+"
          f"{model_b['n_trees']} 棵树 → sigmoid → 爆款概率")
    print("=" * 70)
    return out


if __name__ == "__main__":
    main()
