"""
=======================================================================
 模块一 · 数据采集与原始层构建
-----------------------------------------------------------------------
 输入：用户提供的三个真实数据表（抖音 AI 短剧榜采集）
        表1  表1_AI剧短剧列表.xlsx      —— 榜单剧集清单
        表2  表2_AI剧热评列表.xlsx      —— 每部剧 Top 热评
        表3  表3_AI剧第一集评论列表.xlsx —— 每部剧第一集评论

 处理：字段中英映射 → 类型规整 → 主键去重 → 文本清洗 → 时间解析
 输出：data/raw/{drama_list, hot_comments, ep1_comments}.{parquet,csv}
=======================================================================
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA_RAW, DATA_SOURCE, FALLBACK_DIR, OUTPUTS,
                    SRC_DRAMA_LIST, SRC_HOT_COMMENTS, SRC_EP1_COMMENTS)

AUDIT = {}


def banner(t):
    print("\n" + "=" * 68)
    print(f" {t}")
    print("=" * 68)


def resolve(p: Path) -> Path:
    """定位源文件：优先项目内 data/source，其次回退目录"""
    if p.exists():
        return p
    alt = FALLBACK_DIR / p.name
    if alt.exists():
        print(f"  [回退] 使用 {alt}")
        return alt
    raise FileNotFoundError(f"找不到源数据文件：{p}（也未在 {FALLBACK_DIR} 找到）")


def _clean_text(s: pd.Series) -> pd.Series:
    """评论文本清洗：去 URL、收敛空白、去首尾空格"""
    s = s.astype(str)
    s = s.str.replace(r"https?://\S+", "", regex=True)
    s = s.str.replace(r"\[[^\]]{1,12}\]", "", regex=True)   # 去掉 [飞吻] 这类表情占位
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.str.strip()


# =====================================================================
# ①  剧集列表（表1）
# =====================================================================
def load_drama_list() -> pd.DataFrame:
    src = resolve(SRC_DRAMA_LIST)
    df = pd.read_excel(src)
    n0 = len(df)
    print(f"  [表1] {src.name} | 原始 {n0:,} 行 × {df.shape[1]} 列")

    df = df.rename(columns={
        "排名": "rank", "页码": "page_no", "短剧id": "drama_id",
        "短剧名称": "drama_name", "分类": "category", "集数": "episodes",
        "热度": "heat_text", "热度数值": "heat_value",
        "评分(热评均分)": "hot_rating_avg", "点赞量": "like_count",
        "收藏量": "collect_count", "榜单标签": "rank_tag", "简介": "summary",
    })

    # 主键去重（短剧 id 唯一，保留后出现的记录）
    dup = int(df.duplicated(subset=["drama_id"]).sum())
    df = df.drop_duplicates(subset=["drama_id"], keep="last").copy()

    # 类型规整
    for c in ["rank", "page_no", "episodes", "heat_value", "like_count",
              "collect_count", "hot_rating_avg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 热度文本与数值一致性校验（如"6849万热度" ↔ 68491671）
    parsed = pd.to_numeric(
        df["heat_text"].astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce")
    unit_wan = df["heat_text"].astype(str).str.contains("万")
    parsed = np.where(unit_wan, parsed * 1e4, parsed)
    df["heat_parsed"] = parsed
    rel = ((df["heat_parsed"] - df["heat_value"]).abs() / df["heat_value"]).dropna()
    match_rate = float((rel < 0.01).mean()) if len(rel) else float("nan")

    # 缺失与文本兜底
    df["rank_tag"] = df["rank_tag"].fillna("").astype(str).str.strip()
    df["rank_tag_present"] = (df["rank_tag"] != "").astype(int)
    df["category"] = df["category"].fillna("未分类").astype(str).str.strip()
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()
    df["drama_name"] = df["drama_name"].astype(str).str.strip()

    # 异常值：集数为正、热度为正
    bad = (df["episodes"].isna() | (df["episodes"] <= 0) | df["heat_value"].isna())
    n_bad = int(bad.sum())
    df = df[~bad].copy()
    df["episodes"] = df["episodes"].clip(lower=1)

    # 重算排名（按热度降序），保证与热度口径一致
    df = df.sort_values("heat_value", ascending=False).reset_index(drop=True)
    df["rank_recomputed"] = np.arange(1, len(df) + 1)

    print(f"        去重 {dup} 行 | 非法记录 {n_bad} 行 → 保留 {len(df):,} 部")
    print(f"        热度文本↔数值一致率 {match_rate*100:.1f}% | "
          f"热度区间 {df['heat_value'].min()/1e4:.0f}~{df['heat_value'].max()/1e4:.0f} 万")
    AUDIT["drama_list"] = {"raw_rows": n0, "kept": int(len(df)),
                           "dropped_dup": dup, "dropped_invalid": n_bad,
                           "heat_text_match_rate_pct": round(match_rate * 100, 2)}
    return df


# =====================================================================
# ②  评论表（表2 / 表3）
# =====================================================================
def load_comments(src: Path, name: str, has_rating: bool) -> pd.DataFrame:
    df = pd.read_excel(src)
    n0 = len(df)
    print(f"  [{name}] {src.name} | 原始 {n0:,} 行 × {df.shape[1]} 列")

    ren = {"短剧名": "drama_name", "短剧id": "drama_id", "评论人昵称": "user_nick",
           "评论人id": "user_id", "点赞": "digg", "内容": "content",
           "评论时间": "comment_time"}
    if has_rating:
        ren["评分"] = "rating"
    df = df.rename(columns=ren)

    # 完全重复行 + 业务去重（同剧同人同内容视为重复）
    dup_full = int(df.duplicated().sum())
    df = df.drop_duplicates()
    dup_biz = int(df.duplicated(subset=["drama_id", "user_nick", "content"]).sum())
    df = df.drop_duplicates(subset=["drama_id", "user_nick", "content"], keep="first")

    # 评论时间解析
    df["comment_time"] = pd.to_datetime(df["comment_time"], errors="coerce")
    n_bad_time = int(df["comment_time"].isna().sum())
    df = df[df["comment_time"].notna()].copy()

    # 文本清洗 + 空评论剔除
    df["content_raw"] = df["content"].astype(str)
    df["content"] = _clean_text(df["content"])
    n_empty = int((df["content"].str.len() == 0).sum())
    df = df[df["content"].str.len() > 0].copy()

    df["digg"] = pd.to_numeric(df["digg"], errors="coerce").fillna(0).clip(lower=0)
    df["content_len"] = df["content"].str.len()
    if has_rating:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 10)

    # 主键：剧集 + 评论时间 + 昵称哈希
    df = df.sort_values(["drama_id", "comment_time"]).reset_index(drop=True)
    df["comment_id"] = (df["drama_id"].astype(str) + "_"
                        + df.groupby("drama_id").cumcount().astype(str))
    df["user_id"] = df["user_id"].fillna(df["user_nick"].astype(str))

    cols = ["comment_id", "drama_id", "drama_name", "user_nick", "user_id",
            "digg", "content", "content_len", "comment_time"]
    if has_rating:
        cols.insert(6, "rating")
    df = df[cols]

    print(f"        去重 {dup_full} 行(全同) + {dup_biz} 行(同人同文) | "
          f"时间缺失 {n_bad_time} | 空评论 {n_empty} → 保留 {len(df):,} 条")
    AUDIT[name] = {"raw_rows": n0, "kept": int(len(df)),
                   "dropped_full_dup": dup_full, "dropped_biz_dup": dup_biz,
                   "dropped_bad_time": n_bad_time, "dropped_empty": n_empty,
                   "dramas_covered": int(df["drama_id"].nunique()),
                   "time_range": [str(df["comment_time"].min()),
                                  str(df["comment_time"].max())]}
    return df


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 68)
    print(" 模块一 · 数据采集与原始层构建（真实数据）")
    print("=" * 68)

    banner("① 加载剧集列表")
    dramas = load_drama_list()

    banner("② 加载评论数据")
    hot = load_comments(resolve(SRC_HOT_COMMENTS), "hot_comments", has_rating=True)
    ep1 = load_comments(resolve(SRC_EP1_COMMENTS), "ep1_comments", has_rating=False)

    banner("③ 外键一致性校验")
    ids = set(dramas["drama_id"])
    hot_out = int((~hot["drama_id"].isin(ids)).sum())
    ep1_out = int((~ep1["drama_id"].isin(ids)).sum())
    print(f"        热评中游离记录 {hot_out} | 第一集评论中游离记录 {ep1_out}")
    cover_hot = hot["drama_id"].nunique()
    cover_ep1 = ep1["drama_id"].nunique()
    print(f"        热评覆盖 {cover_hot}/{len(dramas)} 部 "
          f"({cover_hot/len(dramas)*100:.1f}%) | "
          f"第一集评论覆盖 {cover_ep1}/{len(dramas)} 部 "
          f"({cover_ep1/len(dramas)*100:.1f}%)")
    AUDIT["fk_check"] = {"hot_orphan": hot_out, "ep1_orphan": ep1_out,
                         "hot_coverage_pct": round(cover_hot / len(dramas) * 100, 2),
                         "ep1_coverage_pct": round(cover_ep1 / len(dramas) * 100, 2)}

    banner("④ 写出原始层")
    tables = {"drama_list": dramas, "hot_comments": hot, "ep1_comments": ep1}
    for name, df in tables.items():
        df.to_parquet(DATA_RAW / f"{name}.parquet", index=False)
        df.to_csv(DATA_RAW / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"        → data/raw/{name}.parquet  ({len(df):,} 行)")
    with open(OUTPUTS / "raw_audit.json", "w", encoding="utf-8") as fp:
        json.dump(AUDIT, fp, ensure_ascii=False, indent=2)

    print("\n  原始层概览：")
    print(f"    剧集 {len(dramas):,} 部 | 热评 {len(hot):,} 条 | "
          f"第一集评论 {len(ep1):,} 条")
    print(f"    题材组合 {dramas['category'].nunique()} 种 | "
          f"集数中位 {dramas['episodes'].median():.0f} 集")
    print("=" * 68)
    return tables


if __name__ == "__main__":
    main()
