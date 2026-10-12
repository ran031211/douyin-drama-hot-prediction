"""
=======================================================================
 模块三 · 大数据分析（EDA）
-----------------------------------------------------------------------
 分析主线：微短剧的"热度长什么样 → 什么在影响热度"。
   ①  热度分布形态      长尾 / 分位结构 / 爆款阈值
   ②  榜单结构          排名分档与热度、榜单标签
   ③  题材热度画像      各赛道的剧数、平均热度、爆款率
   ④  体量因素          集数分布与集数—热度关系
   ⑤  观众反馈信号      第一集评论互动 → 热度（分层对照）
   ⑥  口碑因素          评分分布与评分—热度关系
   ⑦  相关性全景        特征组与热度的 Spearman 相关
   ⑧  时间规律          评论的月度 / 小时分布

 输出：outputs/figures/fig*.png + outputs/eda_summary.json
=======================================================================
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, FIGURES, OUTPUTS, GENRES
import viz_style as vs
from preprocess import FEAT_ALL, GROUP_OF

P = vs.PALETTE
E = {}


def banner(t):
    print("\n" + "=" * 68)
    print(f" {t}")
    print("=" * 68)


def _sub(n=1, figsize=(13, 5.2)):
    fig, axes = plt.subplots(1, n, figsize=figsize)
    return fig, (axes if n > 1 else [axes])


# =====================================================================
# ① 热度分布形态
# =====================================================================
def fig_heat_distribution(df):
    banner("① 热度分布形态")
    h = df["heat_value"].astype(float) / 1e4
    thr = df["heat_threshold"].iloc[0] / 1e4
    fig, axes = _sub(2, (13.5, 5.0))

    ax = axes[0]
    ax.hist(h, bins=48, color=P["ai"], alpha=0.88, edgecolor="white", linewidth=0.6)
    ax.axvline(thr, color=P["accent"], linestyle="--", linewidth=1.6)
    ax.annotate(f"爆款阈值 {thr:,.0f} 万\n(Top20%)", xy=(thr, ax.get_ylim()[1] * 0.82),
                xytext=(thr + 260, ax.get_ylim()[1] * 0.82), color=P["accent"],
                fontsize=9.5, va="center")
    ax.set_xlabel("榜单热度（万）")
    ax.set_ylabel("剧集数")
    ax.set_title("热度分布：高峰集中 + 右侧长尾")
    ax.text(0.98, 0.92, f"中位 {h.median():,.0f} 万\nIQR "
                        f"{h.quantile(.25):,.0f}–{h.quantile(.75):,.0f} 万",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color=P["ink_soft"])

    ax = axes[1]
    qs = np.linspace(0, 1, 101)
    ax.plot(qs * 100, np.quantile(h, qs), color=P["human"], linewidth=2.0)
    ax.fill_between(qs * 100, np.quantile(h, qs), h.min(), color=P["human_light"],
                    alpha=0.22)
    ax.axhline(thr, color=P["accent"], linestyle="--", linewidth=1.4)
    ax.set_xlabel("累计分位（%）")
    ax.set_ylabel("热度（万）")
    ax.set_title("分位曲线：前 20% 拉开了主要差距")
    for q in (0.5, 0.8, 0.95):
        v = np.quantile(h, q)
        ax.scatter([q * 100], [v], color=P["accent"], zorder=5, s=26)
        ax.annotate(f"P{int(q*100)}  {v:,.0f}万", xy=(q * 100, v),
                    xytext=(q * 100 - 20, v + 120), fontsize=9, color=P["ink_soft"])

    fig.suptitle("热度分布诊断：榜单热度高度集中，头部与其余存在系统性差距",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    E["heat_distribution"] = {
        "min": float(h.min()), "p25": float(h.quantile(.25)),
        "median": float(h.median()), "p75": float(h.quantile(.75)),
        "p90": float(h.quantile(.90)), "max": float(h.max()),
        "threshold": float(thr), "cv_pct": round(float(h.std() / h.mean() * 100), 2)}
    print(f"  热度中位 {h.median():,.0f} 万 | 变异系数 {E['heat_distribution']['cv_pct']}%")
    return vs.finish(fig, FIGURES / "fig01_heat_distribution.png",
                     "口径：榜单热度数值（万）· 爆款 = 热度 Top20%")


# =====================================================================
# ② 榜单结构
# =====================================================================
def fig_rank_structure(df):
    banner("② 榜单结构：排名与热度")
    d = df.dropna(subset=["rank"]).copy()
    fig, axes = _sub(2, (13.5, 4.9))

    ax = axes[0]
    ax.scatter(d["rank"], d["heat_value"] / 1e4, s=11, alpha=0.45, color=P["blue"],
               edgecolors="none")
    bins = [0, 50, 100, 160, 300, 500, 800]
    mid = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    gm = [d[(d["rank"] > bins[i]) & (d["rank"] <= bins[i + 1])]["heat_value"].median() / 1e4
          for i in range(len(bins) - 1)]
    ax.plot(mid, gm, color=P["accent"], linewidth=2.2, marker="o", markersize=5,
            label="分档中位")
    ax.set_xlabel("榜单排名")
    ax.set_ylabel("热度（万）")
    ax.set_title("排名—热度：头部陡降，腰部趋平")
    ax.legend(loc="upper right")

    ax = axes[1]
    lab, cnt, hr = [], [], []
    for i in range(len(bins) - 1):
        seg = d[(d["rank"] > bins[i]) & (d["rank"] <= bins[i + 1])]
        lab.append(f"{bins[i]+1}–{bins[i+1]}")
        cnt.append(len(seg))
        hr.append(float(seg["is_hot"].mean()) * 100)
    x = np.arange(len(lab))
    ax.bar(x, cnt, color=P["gray"], width=0.6, label="剧集数")
    ax.set_ylabel("剧集数", color=P["ink_soft"])
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlabel("排名区间")
    ax2 = ax.twinx()
    ax2.plot(x, hr, color=P["accent"], linewidth=2.2, marker="o", markersize=5,
             label="爆款率")
    ax2.set_ylabel("爆款率（%）", color=P["accent"])
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    ax.set_title("榜单标签：前 99 名带「热播AI剧榜」标记")

    fig.suptitle("榜单结构分析：热度排名前 20% 与其余存在显著断层",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    tagged = int(df["rank_tag_present"].sum())
    E["rank_structure"] = {"tagged_dramas": tagged,
                           "top50_median_wan": round(gm[0], 0),
                           "tail_median_wan": round(gm[-1], 0),
                           "spearman_rank_heat": round(
                               float(d["rank"].corr(d["heat_value"], method="spearman")), 4)}
    print(f"  带榜单标签 {tagged} 部 | 前50中位 {gm[0]:,.0f} 万 vs 尾部 {gm[-1]:,.0f} 万")
    return vs.finish(fig, FIGURES / "fig02_rank_structure.png",
                     "排名由热度数值降序生成 · Spearman 秩相关 "
                     f"{E['rank_structure']['spearman_rank_heat']}")


# =====================================================================
# ③ 题材热度画像
# =====================================================================
def fig_genre_profile(df):
    banner("③ 题材热度画像")
    rows = []
    for g in GENRES:
        m = df[f"tag_{g}"].astype(bool)
        if m.sum() < 8:
            continue
        rows.append({"genre": g, "n": int(m.sum()),
                     "heat": float(df.loc[m, "heat_value"].median() / 1e4),
                     "hot": float(df.loc[m, "is_hot"].mean() * 100),
                     "rating": float(df.loc[m, "hot_rating_mean"].median())})
    g = pd.DataFrame(rows).sort_values("heat", ascending=True)
    fig, axes = _sub(2, (14, max(5.4, len(g) * 0.32)))

    ax = axes[0]
    bars = ax.barh(g["genre"], g["heat"], color=P["ai"], height=0.66)
    base = float(df["heat_value"].median() / 1e4)
    ax.axvline(base, color=P["accent"], linestyle="--", linewidth=1.4)
    ax.text(base, -0.9, f"全站中位 {base:,.0f}万", color=P["accent"], fontsize=9)
    vs.value_labels(ax, bars, "{:,.0f}", horizontal=True)
    ax.set_xlabel("热度中位（万）")
    ax.set_title("各题材热度中位（按升序）")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9.2)

    ax = axes[1]
    sc = ax.scatter(g["n"], g["hot"], s=g["heat"] * 2.2,
                    c=g["heat"], cmap="coolwarm", alpha=0.85, edgecolors="white")
    for _, r in g.iterrows():
        ax.annotate(r["genre"], (r["n"], r["hot"]), fontsize=8.6,
                    xytext=(4, 3), textcoords="offset points", color=P["ink_soft"])
    ax.axhline(float(df["is_hot"].mean() * 100), color=P["gray"], linestyle="--",
               linewidth=1.2)
    ax.set_xlabel("题材内剧集数")
    ax.set_ylabel("爆款率（%）")
    ax.set_title("题材规模 × 爆款率（气泡=热度中位）")

    fig.suptitle("题材画像：哪些赛道更容易跑出高热度短剧",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    E["genre_profile"] = g.sort_values("hot", ascending=False).round(2).to_dict("records")
    print("  爆款率 Top5 题材：" + "、".join(
        f"{r['genre']} {r['hot']:.0f}%" for _, r in
        g.sort_values("hot", ascending=False).head(5).iterrows()))
    return vs.finish(fig, FIGURES / "fig03_genre_profile.png",
                     "题材为「分类」字段的多标签拆解 · 仅展示剧集数 ≥ 8 的题材")


# =====================================================================
# ④ 体量因素：集数
# =====================================================================
def fig_episodes(df):
    banner("④ 体量因素：集数与热度")
    fig, axes = _sub(2, (13.5, 4.9))

    ax = axes[0]
    ax.hist(df["episodes"], bins=40, color=P["human"], alpha=0.88,
            edgecolor="white", linewidth=0.6)
    ax.axvline(df["episodes"].median(), color=P["accent"], linestyle="--", linewidth=1.5)
    ax.annotate(f"中位 {df['episodes'].median():.0f} 集",
                xy=(df["episodes"].median(), ax.get_ylim()[1] * 0.8),
                xytext=(df["episodes"].median() + 40, ax.get_ylim()[1] * 0.8),
                color=P["accent"], fontsize=9.5, va="center")
    ax.set_xlabel("总集数")
    ax.set_ylabel("剧集数")
    ax.set_title("集数分布")

    ax = axes[1]
    tmp = df[["episodes", "is_hot", "heat_value"]].copy()
    tmp["_bin"] = pd.qcut(tmp["episodes"], 6, duplicates="drop")
    gg = tmp.groupby("_bin", observed=True).agg(
        n=("is_hot", "size"), hot=("is_hot", "mean"),
        heat=("heat_value", "median")).reset_index()
    x = np.arange(len(gg))
    lab = [f"{int(iv.left)}–{int(iv.right)}" for iv in gg["_bin"]]
    ax.bar(x, gg["hot"] * 100, color=P["blue"], width=0.62)
    for i, v in enumerate(gg["hot"] * 100):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9.5, color=P["ink_soft"])
    ax.axhline(df["is_hot"].mean() * 100, color=P["gray"], linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=8.8)
    ax.set_xlabel("集数区间")
    ax.set_ylabel("爆款率（%）")
    ax.set_title("集数分箱 × 爆款率（虚线=全站均值）")

    fig.suptitle("体量因素：集数与爆款概率并非单调关系，中长剧更占优",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    E["episodes"] = {"median": float(df["episodes"].median()),
                     "spearman_heat": round(float(
                         df["episodes"].corr(df["heat_value"], method="spearman")), 4),
                     "bins": [{"range": l, "hot_rate_pct": round(float(r["hot"]) * 100, 2)}
                              for l, (_, r) in zip(lab, gg.iterrows())]}
    print(f"  集数中位 {E['episodes']['median']:.0f} | 与热度秩相关 "
          f"{E['episodes']['spearman_heat']}")
    return vs.finish(fig, FIGURES / "fig04_episodes.png",
                     "集数已做 3.5σ 截尾（上限 500 集）处理")


# =====================================================================
# ⑤ 观众反馈信号
# =====================================================================
def fig_engagement(df):
    banner("⑤ 观众反馈信号：评论互动 → 热度")
    fig, axes = _sub(3, (15.5, 4.8))

    # (a) 第一集评论平均点赞 分箱
    ax = axes[0]
    b = pd.qcut(df["ep1_digg_mean"].rank(method="first"), 8, labels=False)
    gg = df.groupby(b).agg(hot=("is_hot", "mean"), n=("is_hot", "size"),
                           heat=("heat_value", "median")).reset_index()
    x = np.arange(len(gg))
    ax.bar(x, gg["hot"] * 100, color=P["ai"], width=0.66)
    ax.axhline(df["is_hot"].mean() * 100, color=P["gray"], linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i+1}" for i in x], fontsize=9)
    ax.set_xlabel("第一集评论平均点赞 分位档（低→高）")
    ax.set_ylabel("爆款率（%）")
    ax.set_title("(a) 评论互动强度 → 爆款率")
    ax.grid(axis="x", visible=False)

    # (b) 百赞评论占比
    ax = axes[1]
    b = pd.qcut(df["ep1_hot_ratio"].rank(method="first"), 6, labels=False)
    gg2 = df.groupby(b).agg(hot=("is_hot", "mean"), r=("ep1_hot_ratio", "median")).reset_index()
    ax.plot(np.arange(len(gg2)), gg2["hot"] * 100, color=P["accent"], linewidth=2.3,
            marker="o", markersize=6)
    ax.fill_between(np.arange(len(gg2)), gg2["hot"] * 100, alpha=0.18, color=P["accent"])
    for i, r in gg2.iterrows():
        ax.annotate(f"{r['r']*100:.0f}%", (i, r["hot"] * 100), fontsize=8.8,
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    color=P["ink_soft"])
    ax.set_xticks(np.arange(len(gg2)))
    ax.set_xticklabels([f"D{i+1}" for i in np.arange(len(gg2))], fontsize=9)
    ax.set_xlabel("百赞评论占比 分位档（低→高，标注=档内中位值）")
    ax.set_ylabel("爆款率（%）")
    ax.set_title("(b) 高赞评论集中度 → 爆款率")

    # (c) 两组信号散点
    ax = axes[2]
    sc = ax.scatter(df["ep1_digg_mean"].clip(upper=200), df["hot_rating_mean"],
                    c=df["is_hot"], cmap="coolwarm", s=16, alpha=0.7, edgecolors="none")
    ax.set_xlabel("第一集评论平均点赞（截尾 200）")
    ax.set_ylabel("热评均分")
    ax.set_title("(c) 口碑 × 互动 的爆款分布")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=P["ai"],
                              label="爆款"), Line2D([], [], marker="o", linestyle="",
                                                    color="#C0392B", label="Top20%")],
              loc="lower right", fontsize=8.6)

    fig.suptitle("观众反馈：第一集评论互动强度与爆款概率呈明显单调关系",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    E["engagement"] = {
        "ep1_digg_mean_spearman": round(float(
            df["ep1_digg_mean"].corr(df["heat_value"], method="spearman")), 4),
        "ep1_hot_ratio_spearman": round(float(
            df["ep1_hot_ratio"].corr(df["heat_value"], method="spearman")), 4),
        "digg_d1_hot_pct": round(float(gg.iloc[0]["hot"]) * 100, 2),
        "digg_d8_hot_pct": round(float(gg.iloc[-1]["hot"]) * 100, 2)}
    print(f"  第一集评论点赞 最低档爆款率 {E['engagement']['digg_d1_hot_pct']}% → "
          f"最高档 {E['engagement']['digg_d8_hot_pct']}%")
    return vs.finish(fig, FIGURES / "fig05_engagement.png",
                     "分箱按分位切分，各档样本量均衡 · 爆款 = 热度 Top20%")


# =====================================================================
# ⑥ 口碑因素
# =====================================================================
def fig_rating(df):
    banner("⑥ 口碑因素：评分与热度")
    fig, axes = _sub(2, (13.5, 4.9))

    ax = axes[0]
    ax.hist(df["hot_rating_mean"], bins=26, color=P["teal"], alpha=0.88,
            edgecolor="white", linewidth=0.6)
    ax.axvline(df["hot_rating_mean"].median(), color=P["accent"], linestyle="--",
               linewidth=1.5)
    ax.set_xlabel("热评均分")
    ax.set_ylabel("剧集数")
    ax.set_title("评分分布：集中在 8–10 分区间")

    ax = axes[1]
    b = pd.qcut(df["hot_rating_mean"].rank(method="first"), 6, labels=False)
    gg = df.groupby(b).agg(hot=("is_hot", "mean"), r=("hot_rating_mean", "median"),
                           std=("hot_rating_std", "median")).reset_index()
    x = np.arange(len(gg))
    ax.bar(x, gg["hot"] * 100, color=P["human"], width=0.62)
    for i, r in gg.iterrows():
        ax.text(i, r["hot"] * 100 + 0.7, f"{r['hot']*100:.0f}%", ha="center",
                fontsize=9.5, color=P["ink_soft"])
    ax.axhline(df["is_hot"].mean() * 100, color=P["gray"], linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['r']:.1f}" for _, r in gg.iterrows()], fontsize=9)
    ax.set_xlabel("评分档（标注=档内中位分）")
    ax.set_ylabel("爆款率（%）")
    ax.set_title("评分档 × 爆款率：口碑的解释力有限")

    fig.suptitle("口碑因素：评分对热度的区分能力远弱于评论互动量",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    E["rating"] = {"mean": round(float(df["hot_rating_mean"].mean()), 3),
                   "median": round(float(df["hot_rating_mean"].median()), 3),
                   "spearman_heat": round(float(
                       df["hot_rating_mean"].corr(df["heat_value"], method="spearman")), 4)}
    print(f"  评分中位 {E['rating']['median']} | 与热度秩相关 "
          f"{E['rating']['spearman_heat']}（弱）")
    return vs.finish(fig, FIGURES / "fig06_rating.png",
                     "评分来自热评明细 · 8 部剧无明细者以榜单均分兜底")


# =====================================================================
# ⑦ 相关性全景
# =====================================================================
def fig_correlation(df):
    banner("⑦ 相关性全景：各特征组与热度的关联强度")
    rows = []
    for f in FEAT_ALL:
        if f not in df.columns:
            continue
        s = df[f].astype(float)
        if s.nunique() <= 1:
            continue
        rows.append({"feature": f, "group": GROUP_OF[f],
                     "rho": float(s.corr(df["heat_value"], method="spearman"))})
    r = pd.DataFrame(rows)
    r["abs"] = r["rho"].abs()
    r = r.sort_values("abs", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    top = r.head(20).iloc[::-1]
    ax.barh(top["feature"], top["rho"],
            color=[vs.GROUP_COLOR[g] for g in top["group"]], height=0.7)
    ax.axvline(0, color="#444441", linewidth=0.8)
    ax.set_xlabel("与热度的 Spearman 秩相关")
    ax.set_title("单特征相关性 Top20")
    ax.tick_params(axis="y", labelsize=8.8)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=g) for g, c in vs.GROUP_COLOR.items()],
              loc="lower right", fontsize=9)

    ax = axes[1]
    grp = (r.assign(a=r["abs"]).groupby("group")["abs"]
           .agg(["mean", "max", "count"]).sort_values("mean"))
    ax.barh(grp.index, grp["mean"] * 100,
            color=[vs.GROUP_COLOR[g] for g in grp.index], height=0.6)
    for i, (_, row) in enumerate(grp.iterrows()):
        ax.text(row["mean"] * 100 + 0.25, i, f"{row['mean']*100:.1f}",
                va="center", fontsize=10, color=P["ink_soft"])
    ax.set_xlabel("组内平均 |秩相关| ×100")
    ax.set_title("四类特征组的平均解释力")
    ax.grid(axis="y", visible=False)

    fig.suptitle("相关性全景：评论互动与文本态度是最强的单变量信号",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    E["correlation"] = {
        "top10": r.head(10)[["feature", "group", "rho"]].round(4).to_dict("records"),
        "group_mean_abs": {k: round(float(v) * 100, 2)
                           for k, v in grp["mean"].items()}}
    print("  相关性 Top5：" + "、".join(f"{x['feature']} {x['rho']:.3f}"
                                       for x in E["correlation"]["top10"][:5]))
    return vs.finish(fig, FIGURES / "fig07_correlation.png",
                     "Spearman 秩相关（对长尾与非线性单调关系稳健）")


# =====================================================================
# ⑧ 评论时间规律
# =====================================================================
def fig_time_pattern(df):
    banner("⑧ 评论时间规律")
    hot = pd.read_parquet(DATA_PROCESSED / "hot_comments_clean.parquet")
    ep1 = pd.read_parquet(DATA_PROCESSED / "ep1_comments_clean.parquet")
    allc = pd.concat([hot[["comment_time"]], ep1[["comment_time"]]], ignore_index=True)
    allc["comment_time"] = pd.to_datetime(allc["comment_time"])
    allc["month"] = allc["comment_time"].dt.to_period("M").astype(str)
    allc["hour"] = allc["comment_time"].dt.hour

    fig, axes = _sub(2, (13.5, 4.7))
    ax = axes[0]
    mc = allc["month"].value_counts().sort_index()
    ax.bar(mc.index, mc.values, color=P["violet"], width=0.58)
    for i, v in enumerate(mc.values):
        ax.text(i, v + max(mc.values) * 0.015, f"{v:,}", ha="center", fontsize=9,
                color=P["ink_soft"])
    ax.set_ylabel("评论数")
    ax.set_title("评论月度分布")
    ax.tick_params(axis="x", labelsize=9)

    ax = axes[1]
    hc = allc.groupby("hour").size().reindex(range(24), fill_value=0)
    ax.bar(hc.index, hc.values, color=P["amber"], width=0.68)
    ax.axvspan(19, 23, color=P["accent"], alpha=0.12)
    ax.text(21, hc.max() * 0.9, "晚间高峰", color=P["accent"], fontsize=9.5,
            ha="center")
    ax.set_xlabel("小时（0–23）")
    ax.set_ylabel("评论数")
    ax.set_title("评论时段分布")
    ax.set_xticks(range(0, 24, 3))

    fig.suptitle("时间规律：评论集中在上线当月，晚间为互动高峰",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    E["time_pattern"] = {
        "monthly": {k: int(v) for k, v in mc.items()},
        "peak_hour": int(hc.idxmax()),
        "evening_share_pct": round(float(hc.loc[18:23].sum() / hc.sum() * 100), 2)}
    print(f"  评论峰值时段 {E['time_pattern']['peak_hour']} 时 | "
          f"晚间(18–23)占比 {E['time_pattern']['evening_share_pct']}%")
    return vs.finish(fig, FIGURES / "fig08_time_pattern.png",
                     "基于热评 + 第一集评论的合并时间戳")


def main():
    print("=" * 68)
    print(" 模块三 · 大数据分析（EDA）")
    print("=" * 68)
    df = pd.read_parquet(DATA_PROCESSED / "ml_dataset.parquet")
    print(f"  样本 {len(df):,} 部微短剧 | 特征 {len(FEAT_ALL)} 维")

    fig_heat_distribution(df)
    fig_rank_structure(df)
    fig_genre_profile(df)
    fig_episodes(df)
    fig_engagement(df)
    fig_rating(df)
    fig_correlation(df)
    fig_time_pattern(df)

    with open(OUTPUTS / "eda_summary.json", "w", encoding="utf-8") as fp:
        json.dump(E, fp, ensure_ascii=False, indent=2)
    print(f"\n  EDA 结论 → {OUTPUTS / 'eda_summary.json'}")
    print(f"  图表 → {FIGURES}/fig01~fig08")
    print("=" * 68)
    return E


if __name__ == "__main__":
    main()
