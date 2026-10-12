"""
=======================================================================
 模块二 · 大数据预处理 + 特征工程（剧集级）
-----------------------------------------------------------------------
 流程：
   ①  原始层加载与质量审计
   ②  清洗：主键去重 / 文本规整 / 类型兜底 / 评价一致性校验
   ③  异常值治理：对数域 3σ 检测 + 分位截尾（保留样本、打标记）
   ④  评论侧聚合：把热评表、第一集评论表汇总为剧集级"观众反馈"特征
   ⑤  特征工程：内容属性 / 剧情爽点 / 观众口碑 / 评论互动 / 文本态度
   ⑥  标签构造：热度 Top20% 爆款二分类 + log 热度回归目标
   ⑦  输出 ml_dataset.{parquet,csv} 与数据质量报告

 分析单元：一部微短剧。特征严格区分「上线前内容属性」与
 「上线后观众反馈」两组，便于后续做增量贡献的消融实验。
=======================================================================
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA_RAW, DATA_PROCESSED, OUTPUTS, HOT_QUANTILE,
                    N_HEAT_TIERS, GENRES, GENRE_MIN_COUNT, PLOT_KEYWORDS,
                    POSITIVE_WORDS, NEGATIVE_WORDS, URGE_WORDS)

REPORT = {}

# ---------------------------------------------------------------- 特征分组
# 顺序即建模时的列顺序，供 model.py / export_webmodel.py 直接复用
TAG_FEATS = [f"tag_{g}" for g in GENRES]
PLOT_FEATS = [f"plot_{k}" for k in PLOT_KEYWORDS]

FEAT_CONTENT = ([
    "episodes", "log_episodes", "is_long_series", "n_genre_tags",
    "title_len", "title_has_season", "title_has_number",
    "summary_len", "summary_has_question", "summary_has_exclaim",
] + TAG_FEATS + PLOT_FEATS)

FEAT_RATING = ["hot_rating_avg", "hot_rating_mean", "hot_rating_std",
               "hot_rating_good", "hot_rating_bad"]

FEAT_ENGAGE = [
    "hot_cmt_cnt", "hot_digg_mean", "log_hot_digg_max", "log_hot_digg_sum",
    "hot_hot_ratio", "hot_user_div",
    "ep1_cmt_cnt", "ep1_digg_mean", "log_ep1_digg_max", "log_ep1_digg_sum",
    "ep1_hot_ratio", "ep1_super_ratio", "ep1_user_div",
]

FEAT_TEXT = ["hot_text_len", "ep1_text_len", "ep1_emoji_ratio", "ep1_urge_ratio",
             "ep1_pos_ratio", "ep1_neg_ratio"]

FEAT_ALL = FEAT_CONTENT + FEAT_RATING + FEAT_ENGAGE + FEAT_TEXT

FEATURE_CN = {
    "episodes": "总集数", "log_episodes": "集数(对数)", "is_long_series": "长剧(≥100集)",
    "n_genre_tags": "题材标签数", "title_len": "剧名长度", "title_has_season": "剧名含季数",
    "title_has_number": "剧名含数字", "summary_len": "简介字数",
    "summary_has_question": "简介含疑问", "summary_has_exclaim": "简介含感叹",
    "hot_rating_avg": "热评均分(榜单)", "hot_rating_mean": "热评均分(明细)",
    "hot_rating_std": "评分分歧度", "hot_rating_good": "好评率(≥9分)",
    "hot_rating_bad": "差评率(≤3分)",
    "hot_cmt_cnt": "热评条数", "hot_digg_mean": "热评平均点赞", "log_hot_digg_max": "最高热评点赞(对数)",
    "log_hot_digg_sum": "热评点赞合计(对数)", "hot_hot_ratio": "高热评占比",
    "hot_user_div": "热评用户多样性",
    "ep1_cmt_cnt": "第一集评论数", "ep1_digg_mean": "第一集评论平均点赞",
    "log_ep1_digg_max": "第一集最高评论点赞(对数)", "log_ep1_digg_sum": "第一集评论点赞合计(对数)",
    "ep1_hot_ratio": "第一集百赞评论占比", "ep1_super_ratio": "第一集千赞评论占比",
    "ep1_user_div": "第一集评论用户多样性",
    "hot_text_len": "热评平均字数", "ep1_text_len": "第一集评论平均字数",
    "ep1_emoji_ratio": "第一集表情占比", "ep1_urge_ratio": "催更评论占比",
    "ep1_pos_ratio": "正面情感占比", "ep1_neg_ratio": "负面情感占比",
}
FEATURE_CN.update({f"tag_{g}": f"题材·{g}" for g in GENRES})
FEATURE_CN.update({f"plot_{k}": f"爽点·{k}" for k in PLOT_KEYWORDS})

GROUP_OF = {}
for _f in FEAT_CONTENT:
    GROUP_OF[_f] = "内容属性"
for _f in FEAT_RATING:
    GROUP_OF[_f] = "观众口碑"
for _f in FEAT_ENGAGE:
    GROUP_OF[_f] = "评论互动"
for _f in FEAT_TEXT:
    GROUP_OF[_f] = "文本态度"


def banner(t):
    print("\n" + "=" * 68)
    print(f" {t}")
    print("=" * 68)


def _has_emoji(s: str) -> int:
    """判断文本是否含 emoji（纯码位判断，避免 pandas 正则引擎差异）"""
    for ch in str(s):
        o = ord(ch)
        if (0x1F300 <= o <= 0x1FAFF or 0x1F000 <= o <= 0x1F2FF or
                0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF or
                o == 0xFE0F or 0x2190 <= o <= 0x21FF):
            return 1
    return 0


def _kw_hit(s: str, words) -> int:
    s = str(s)
    return int(any(w in s for w in words))


def _kw_ratio(series: pd.Series, words) -> pd.Series:
    """一组关键词在每条文本中是否命中，返回 0/1 序列"""
    return series.astype(str).map(lambda s: _kw_hit(s, words))


def audit(df, name):
    info = {
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "dup_full": int(df.duplicated().sum()),
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "missing_rate_pct": round(float(df.isna().mean().mean()) * 100, 3),
    }
    print(f"  [{name}] {info['rows']:,} 行 × {info['cols']} 列 | "
          f"完全重复 {info['dup_full']} | 整体缺失率 {info['missing_rate_pct']}%")
    if info["missing"]:
        for c, v in list(info["missing"].items())[:6]:
            print(f"          缺失 {c}: {v} ({v/max(len(df),1)*100:.1f}%)")
    return info


# =====================================================================
# ①  加载与审计
# =====================================================================
def load_raw():
    banner("① 加载原始层并做质量审计")
    dfs = {
        "drama_list": pd.read_parquet(DATA_RAW / "drama_list.parquet"),
        "hot_comments": pd.read_parquet(DATA_RAW / "hot_comments.parquet"),
        "ep1_comments": pd.read_parquet(DATA_RAW / "ep1_comments.parquet"),
    }
    REPORT["raw_audit"] = {k: audit(v, k) for k, v in dfs.items()}
    return dfs


# =====================================================================
# ③  异常值治理
# =====================================================================
def handle_outliers(df, cols, name, k=3.0, hard_cap=None, clip=True):
    """
    对数域 3σ 检测 + 分位截尾。不删行——极端值往往正是"爆款"样本，
    删除会让模型失去最有价值的学习信号，故改为打标记。

    clip=False 用于「标签列」：热度数值的头部区间本身就是被预测的对象，
    截尾会把 Top 剧集压平成同一个值，既破坏回归目标也误导结论，
    因此对标签只标记、不改数值。
    """
    report = {}
    for c in cols:
        x = df[c].astype(float)
        lx = np.log1p(x.clip(lower=0))
        lo = lx.mean() - k * lx.std()
        hi = lx.mean() + k * lx.std()
        lo_v, hi_v = max(np.expm1(lo), 0.0), np.expm1(hi)
        if hard_cap is not None:
            hi_v = min(hi_v, hard_cap)
        if not np.isfinite(hi_v) or hi_v <= lo_v:
            report[c] = {"lower": 0.0, "upper": float(x.max()), "n_flag": 0,
                         "rate_pct": 0.0, "clipped": bool(clip)}
            continue
        flag = (x < lo_v) | (x > hi_v)
        df[f"{c}_outlier"] = flag.astype(int)
        if clip:
            df[c] = x.clip(lo_v, hi_v)
        report[c] = {"lower": round(float(lo_v), 2), "upper": round(float(hi_v), 2),
                     "n_flag": int(flag.sum()),
                     "rate_pct": round(float(flag.mean()) * 100, 3),
                     "clipped": bool(clip)}
    tot = int(df[[c for c in df.columns if c.endswith("_outlier")]].sum().sum())
    print(f"  [{name}] 标记异常 {tot:,} 个 | " + " ".join(
        f"{c}∈[{v['lower']:.0f},{v['upper']:.0f}]" for c, v in list(report.items())[:3]))
    REPORT[f"outlier_{name}"] = report
    return df


# =====================================================================
# ④  评论侧聚合 → 剧集级观众反馈特征
# =====================================================================
def aggregate_hot_comments(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("drama_id").agg(
        hot_cmt_cnt=("digg", "size"),
        hot_digg_mean=("digg", "mean"),
        hot_digg_max=("digg", "max"),
        hot_digg_sum=("digg", "sum"),
        hot_rating_mean=("rating", "mean"),
        hot_rating_std=("rating", "std"),
        hot_rating_good=("rating", lambda s: float((s >= 9).mean())),
        hot_rating_bad=("rating", lambda s: float((s <= 3).mean())),
        hot_text_len=("content_len", "mean"),
        hot_users=("user_nick", "nunique"),
    ).reset_index()
    g["hot_hot_ratio"] = df.assign(_h=df["digg"] >= 100).groupby("drama_id")["_h"].mean().values
    g["hot_user_div"] = g["hot_users"] / g["hot_cmt_cnt"].replace(0, 1)
    g["log_hot_digg_max"] = np.log1p(g["hot_digg_max"])
    g["log_hot_digg_sum"] = np.log1p(g["hot_digg_sum"])
    g = g.drop(columns=["hot_users"])
    REPORT["hot_comment_agg"] = {
        "dramas": int(len(g)),
        "avg_comments": round(float(g["hot_cmt_cnt"].mean()), 2),
        "avg_rating": round(float(g["hot_rating_mean"].mean()), 3),
        "avg_digg_mean": round(float(g["hot_digg_mean"].mean()), 2),
    }
    print(f"  [热评聚合] 覆盖 {len(g)} 部 | 平均 {g['hot_cmt_cnt'].mean():.1f} 条/部 | "
          f"平均评分 {g['hot_rating_mean'].mean():.2f}")
    return g


def aggregate_ep1_comments(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(
        _emoji=df["content"].map(_has_emoji),
        _urge=_kw_ratio(df["content"], URGE_WORDS),
        _pos=_kw_ratio(df["content"], POSITIVE_WORDS),
        _neg=_kw_ratio(df["content"], NEGATIVE_WORDS),
        _hot=(df["digg"] >= 100).astype(int),
        _super=(df["digg"] >= 1000).astype(int),
    )
    g = d.groupby("drama_id").agg(
        ep1_cmt_cnt=("digg", "size"),
        ep1_digg_mean=("digg", "mean"),
        ep1_digg_max=("digg", "max"),
        ep1_digg_sum=("digg", "sum"),
        ep1_hot_ratio=("_hot", "mean"),
        ep1_super_ratio=("_super", "mean"),
        ep1_text_len=("content_len", "mean"),
        ep1_emoji_ratio=("_emoji", "mean"),
        ep1_urge_ratio=("_urge", "mean"),
        ep1_pos_ratio=("_pos", "mean"),
        ep1_neg_ratio=("_neg", "mean"),
        ep1_users=("user_nick", "nunique"),
    ).reset_index()
    g["ep1_user_div"] = g["ep1_users"] / g["ep1_cmt_cnt"].replace(0, 1)
    g["log_ep1_digg_max"] = np.log1p(g["ep1_digg_max"])
    g["log_ep1_digg_sum"] = np.log1p(g["ep1_digg_sum"])
    g = g.drop(columns=["ep1_users"])
    REPORT["ep1_comment_agg"] = {
        "dramas": int(len(g)),
        "avg_comments": round(float(g["ep1_cmt_cnt"].mean()), 2),
        "avg_urge_ratio": round(float(g["ep1_urge_ratio"].mean()), 4),
        "avg_pos_ratio": round(float(g["ep1_pos_ratio"].mean()), 4),
        "avg_neg_ratio": round(float(g["ep1_neg_ratio"].mean()), 4),
    }
    print(f"  [第一集评论聚合] 覆盖 {len(g)} 部 | 平均 {g['ep1_cmt_cnt'].mean():.1f} 条/部 | "
          f"催更率 {g['ep1_urge_ratio'].mean()*100:.1f}% | "
          f"正面 {g['ep1_pos_ratio'].mean()*100:.1f}%")
    return g


# =====================================================================
# ⑤  特征工程
# =====================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    banner("⑤ 特征工程：内容属性 / 剧情爽点 / 观众口碑 / 评论互动 / 文本态度")
    f = pd.DataFrame(index=df.index)

    # ---------- 一、内容属性 ----------
    ep = df["episodes"].astype(float)
    f["episodes"] = ep
    f["log_episodes"] = np.log1p(ep)
    f["is_long_series"] = (ep >= 100).astype(int)

    tags = df["_tags"]
    f["n_genre_tags"] = tags.map(len)
    for g_ in GENRES:
        f[f"tag_{g_}"] = tags.map(lambda L, gg=g_: int(gg in L))

    name = df["drama_name"].astype(str)
    f["title_len"] = name.str.len()
    f["title_has_season"] = name.str.contains("第.{1,3}季|季").astype(int)
    f["title_has_number"] = name.str.contains(r"\d").astype(int)

    summ = df["summary"].astype(str)
    f["summary_len"] = summ.str.len()
    f["summary_has_question"] = summ.map(lambda s: int("?" in s or "？" in s))
    f["summary_has_exclaim"] = summ.map(lambda s: int("!" in s or "！" in s))
    for k, words in PLOT_KEYWORDS.items():
        f[f"plot_{k}"] = summ.map(lambda s, w=words: int(sum(s.count(x) for x in w)))

    # ---------- 二、观众口碑（评分） ----------
    f["hot_rating_avg"] = df["hot_rating_avg"].astype(float)
    for c in ["hot_rating_mean", "hot_rating_std", "hot_rating_good", "hot_rating_bad"]:
        f[c] = df[c].astype(float) if c in df.columns else 0.0
    # 8 部剧无热评明细，用榜单均分兜底
    f["hot_rating_mean"] = f["hot_rating_mean"].fillna(f["hot_rating_avg"])
    f["hot_rating_std"] = f["hot_rating_std"].fillna(0.0)
    f["hot_rating_good"] = f["hot_rating_good"].fillna(0.0)
    f["hot_rating_bad"] = f["hot_rating_bad"].fillna(0.0)

    # ---------- 三、评论互动 ----------
    for c in FEAT_ENGAGE:
        f[c] = df[c].astype(float) if c in df.columns else 0.0
    f["log_hot_digg_max"] = f["log_hot_digg_max"].fillna(0.0)
    f["log_hot_digg_sum"] = f["log_hot_digg_sum"].fillna(0.0)
    f["log_ep1_digg_max"] = f["log_ep1_digg_max"].fillna(0.0)
    f["log_ep1_digg_sum"] = f["log_ep1_digg_sum"].fillna(0.0)

    # ---------- 四、文本态度 ----------
    for c in FEAT_TEXT:
        f[c] = df[c].astype(float) if c in df.columns else 0.0
    for c in FEAT_TEXT:
        f[c] = f[c].fillna(f[c].median() if f[c].notna().any() else 0.0)

    f = f.fillna(0.0)

    # ---------- 标签构造 ----------
    heat = df["heat_value"].astype(float)
    thr = float(heat.quantile(HOT_QUANTILE))
    f["heat_value"] = heat
    f["log_heat"] = np.log10(heat)
    f["heat_threshold"] = thr
    f["is_hot"] = (heat >= thr).astype(int)
    f["heat_tier"] = pd.qcut(heat.rank(method="first"),
                             N_HEAT_TIERS, labels=False).astype(int)

    REPORT["label"] = {
        "quantile": HOT_QUANTILE, "heat_threshold": round(thr, 2),
        "positive_rate_pct": round(float(f["is_hot"].mean()) * 100, 2),
        "heat_min": int(heat.min()), "heat_max": int(heat.max()),
        "heat_median": int(heat.median()),
        "log_heat_std": round(float(np.log10(heat).std()), 4),
    }
    print(f"  特征 {f[FEAT_ALL].shape[1]} 列 | 爆款阈值热度 ≥ {thr/1e4:,.0f} 万 | "
          f"正类占比 {f['is_hot'].mean()*100:.2f}%")
    return f


# =====================================================================
# 主流程
# =====================================================================
def main():
    dfs = load_raw()

    banner("② 清洗与一致性校验")
    dramas = dfs["drama_list"].copy()
    dramas["_tags"] = dramas["category"].map(
        lambda s: [x.strip() for x in str(s).replace("，", "/").replace(",", "/").split("/")
                   if x.strip()])
    # 题材标签频次统计：只保留高频标签，长尾归入"其他"
    from collections import Counter
    cnt = Counter(t for L in dramas["_tags"] for t in L)
    kept = [g_ for g_, c in cnt.most_common() if c >= GENRE_MIN_COUNT]
    print(f"  题材标签 {len(cnt)} 种 → 频次 ≥{GENRE_MIN_COUNT} 的 {len(kept)} 种保留为特征")
    REPORT["genre_tags"] = {"total": len(cnt),
                            "kept": len(kept),
                            "kept_list": kept,
                            "long_tail_share_pct": round(
                                (1 - sum(cnt[t] for t in kept) / sum(cnt.values())) * 100, 2)}

    banner("③ 异常值治理（对数域 3σ 检测 + 标记；标签列只标记不截尾）")
    # 集数为长尾特征（个别"合集"上传达 1000 集），截尾至 500 并取整；
    # 热度数值是被预测对象，只标记不截尾，保留头部爆款间的真实差距
    dramas = handle_outliers(dramas, ["episodes"], "drama_episodes",
                             k=3.5, hard_cap=500, clip=True)
    dramas["episodes"] = dramas["episodes"].round().astype(int)
    dramas = handle_outliers(dramas, ["heat_value"], "drama_heat", k=3.0, clip=False)

    banner("④ 评论侧聚合")
    hot_agg = aggregate_hot_comments(dfs["hot_comments"])
    ep1_agg = aggregate_ep1_comments(dfs["ep1_comments"])

    banner("⑤ 特征工程与标签构造")
    df = (dramas
          .merge(hot_agg, on="drama_id", how="left")
          .merge(ep1_agg, on="drama_id", how="left"))
    # 无热评明细的 7 部剧：热度相关字段留空由特征层兜底
    features = engineer_features(df)

    meta = df[["drama_id", "drama_name", "category", "rank",
               "rank_tag", "rank_tag_present"]].reset_index(drop=True)
    features = features.reset_index(drop=True)
    out = pd.concat([meta, features], axis=1)

    banner("⑥ 输出")
    out.to_parquet(DATA_PROCESSED / "ml_dataset.parquet", index=False)
    out.to_csv(DATA_PROCESSED / "ml_dataset.csv", index=False, encoding="utf-8-sig")
    dfs["drama_list"].to_parquet(DATA_PROCESSED / "drama_clean.parquet", index=False)
    dfs["hot_comments"].to_parquet(DATA_PROCESSED / "hot_comments_clean.parquet", index=False)
    dfs["ep1_comments"].to_parquet(DATA_PROCESSED / "ep1_comments_clean.parquet", index=False)

    REPORT["dataset"] = {"rows": int(len(out)), "cols": int(out.shape[1]),
                         "n_features": len(FEAT_ALL),
                         "groups": {"内容属性": len(FEAT_CONTENT),
                                    "观众口碑": len(FEAT_RATING),
                                    "评论互动": len(FEAT_ENGAGE),
                                    "文本态度": len(FEAT_TEXT)}}
    with open(OUTPUTS / "data_quality_report.json", "w", encoding="utf-8") as fp:
        json.dump(REPORT, fp, ensure_ascii=False, indent=2)

    print(f"  建模数据集：{out.shape[0]:,} 行 × {out.shape[1]} 列"
          f"（其中特征 {len(FEAT_ALL)} 维）")
    print(f"  → {DATA_PROCESSED / 'ml_dataset.parquet'}")
    print(f"  → {OUTPUTS / 'data_quality_report.json'}")
    print("\n  关键分布：")
    print(f"    爆款率            {out['is_hot'].mean()*100:.2f}%")
    print(f"    集数中位          {out['episodes'].median():.0f} 集")
    print(f"    热评均分中位      {out['hot_rating_mean'].median():.2f}")
    print(f"    第一集评论点赞中位 {out['ep1_digg_mean'].median():.1f}")
    print(f"    催更评论占比中位   {out['ep1_urge_ratio'].median()*100:.1f}%")
    print("=" * 68)
    return out


if __name__ == "__main__":
    main()
