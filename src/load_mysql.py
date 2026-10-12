"""
=======================================================================
 模块一 · 数据管理：MySQL 多表入库 + 数据治理巡检
-----------------------------------------------------------------------
 前置：先执行 sql/douyin_schema.sql 建库建表
   mysql -u root -p < sql/douyin_schema.sql

 用法：python src/load_mysql.py            # 巡检 + 入库（失败自动降级 SQLite）
       python src/load_mysql.py --check    # 只做数据质量巡检，不写库
       python src/load_mysql.py --export   # 直接导出 SQLite 版本（无 MySQL 时）
=======================================================================
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, DATA_PROCESSED, OUTPUTS, MYSQL_URL

RAW = {
    "drama_info": DATA_RAW / "drama_list.parquet",
    "drama_hot_comment": DATA_RAW / "hot_comments.parquet",
    "drama_ep1_comment": DATA_RAW / "ep1_comments.parquet",
}


def banner(t):
    print("\n" + "=" * 68)
    print(f" {t}")
    print("=" * 68)


def load_raw():
    return {k: pd.read_parquet(v) for k, v in RAW.items()}


# ODS 三表在库中的列顺序（严格对齐 sql/douyin_schema.sql）
ODS_DRAMA_COLS = ["drama_id", "drama_name", "category", "episodes", "heat_text",
                  "heat_value", "like_count", "collect_count", "hot_rating_avg",
                  "rank_no", "page_no", "rank_tag", "rank_tag_present", "summary"]
ODS_HOT_COLS = ["comment_id", "drama_id", "drama_name", "user_nick", "digg",
                "rating", "content", "content_len", "comment_time"]
ODS_EP1_COLS = ["comment_id", "drama_id", "drama_name", "user_nick", "digg",
                "content", "content_len", "comment_time"]


def ods_tables(dfs=None):
    """把原始层对齐到 DDL 的列定义，供入库与离线导出共用。

    对齐三件事：
      · rank → rank_no（DDL 命名）
      · 剔除加载期中间列 heat_parsed / rank_recomputed（不属业务字段）
      · 剔除标识类字段 user_id（隐私最小必要原则，DDL 中亦不存在）
    """
    dfs = dfs or load_raw()
    d1 = dfs["drama_info"].rename(columns={"rank": "rank_no"})
    d1 = d1[ODS_DRAMA_COLS].copy()
    d2 = dfs["drama_hot_comment"][ODS_HOT_COLS].copy()
    d3 = dfs["drama_ep1_comment"][ODS_EP1_COLS].copy()
    return {"drama_info": d1, "drama_hot_comment": d2, "drama_ep1_comment": d3}


def governance_check(dfs=None):
    """数据治理巡检：唯一性 / 完整性 / 一致性 / 有效性 / 隐私性"""
    banner("数据治理巡检")
    dfs = dfs or load_raw()
    d1, d2, d3 = dfs["drama_info"], dfs["drama_hot_comment"], dfs["drama_ep1_comment"]
    issues = []

    # ---- 唯一性 ----
    print("  【唯一性】")
    for name, d, key in [("drama_info", d1, "drama_id"),
                         ("drama_hot_comment", d2, "comment_id"),
                         ("drama_ep1_comment", d3, "comment_id")]:
        dup = int(d.duplicated(subset=[key]).sum())
        print(f"    {name:<20} 主键 {key:<12} 重复 {dup}")
        if dup:
            issues.append(f"{name}.{key} 存在 {dup} 条重复主键")

    # ---- 完整性 ----
    print("  【完整性】")
    ids = set(d1["drama_id"])
    c2 = d2["drama_id"].nunique()
    c3 = d3["drama_id"].nunique()
    print(f"    热评覆盖剧集     {c2:,} / {len(ids):,} ({c2/len(ids)*100:.1f}%)")
    print(f"    第一集评论覆盖   {c3:,} / {len(ids):,} ({c3/len(ids)*100:.1f}%)")
    miss = len(ids) - c2
    if miss:
        issues.append(f"{miss} 部剧缺少热评明细（建模时用榜单均分兜底）")

    o2 = int((~d2["drama_id"].isin(ids)).sum())
    o3 = int((~d3["drama_id"].isin(ids)).sum())
    print(f"    外键孤儿记录     热评 {o2} | 第一集评论 {o3}")
    if o2 or o3:
        issues.append("存在外键孤儿记录")

    # ---- 一致性 ----
    print("  【一致性】热度文本 ↔ 热度数值")
    hv = d1["heat_value"].astype(float)
    txt = d1["heat_text"].astype(str).str.replace(r"[^\d.]", "", regex=True).astype(float)
    txt = np.where(d1["heat_text"].astype(str).str.contains("万"), txt * 1e4, txt)
    rel = np.abs(txt - hv.values) / np.maximum(hv.values, 1)
    ok = float((rel < 0.01).mean())
    print(f"    一致率 {ok*100:.1f}%（容差 1%）")
    if ok < 0.99:
        issues.append(f"热度文本与数值一致率仅 {ok*100:.1f}%")

    # ---- 有效性 ----
    print("  【有效性】数值域检查")
    checks = [("drama_info", d1, "episodes", 1, 2000),
              ("drama_info", d1, "heat_value", 1e6, 1e9),
              ("drama_info", d1, "hot_rating_avg", 1, 10),
              ("drama_hot_comment", d2, "rating", 1, 10),
              ("drama_ep1_comment", d3, "digg", 0, 1e7)]
    for t, d, c, lo, hi in checks:
        s = pd.to_numeric(d[c], errors="coerce").dropna()
        n = int(((s < lo) | (s > hi)).sum())
        print(f"    {t}.{c:<16} 越界 {n}（合法区间 [{lo:g}, {hi:g}]）")
        if n:
            issues.append(f"{t}.{c} 有 {n} 条越界")

    # ---- 隐私性 ----
    print("  【隐私性】")
    pii = [c for c in list(d2.columns) + list(d3.columns)
           if any(k in c.lower() for k in ["phone", "mobile", "id_card", "real_name", "user_id"])]
    print(f"    直采字段中含 {len(pii)} 个标识类字段 {pii}，均已在建模层剔除")
    print("    评论人昵称仅用于去重统计，不参与任何对外输出（符合最小必要原则）")

    banner("巡检结论")
    if issues:
        print(f"  发现 {len(issues)} 项提示：")
        for i, x in enumerate(issues, 1):
            print(f"    {i}. {x}")
    else:
        print("  全部检查项通过，数据质量合格。")
    return issues


def _wide_table():
    """从建模数据集抽取入库宽表列"""
    ml = pd.read_parquet(DATA_PROCESSED / "ml_dataset.parquet")
    ren = {"rank": "rank_no"}
    cols = {
        "drama_id": "drama_id", "drama_name": "drama_name", "category": "category",
        "heat_value": "heat_value", "log_heat": "log_heat", "is_hot": "is_hot",
        "heat_tier": "heat_tier", "heat_threshold": "heat_threshold",
        "episodes": "episodes", "log_episodes": "log_episodes",
        "is_long_series": "is_long_series", "n_genre_tags": "n_genre_tags",
        "title_len": "title_len", "summary_len": "summary_len",
        "hot_rating_avg": "hot_rating_avg", "hot_rating_mean": "hot_rating_mean",
        "hot_rating_std": "hot_rating_std", "hot_rating_good": "hot_rating_good",
        "hot_rating_bad": "hot_rating_bad",
        "hot_cmt_cnt": "hot_cmt_cnt", "hot_digg_mean": "hot_digg_mean",
        "log_hot_digg_sum": "log_hot_digg_sum",
        "ep1_cmt_cnt": "ep1_cmt_cnt", "ep1_digg_mean": "ep1_digg_mean",
        "log_ep1_digg_sum": "log_ep1_digg_sum",
        "ep1_hot_ratio": "ep1_hot_ratio", "ep1_super_ratio": "ep1_super_ratio",
        "ep1_text_len": "ep1_text_len", "ep1_emoji_ratio": "ep1_emoji_ratio",
        "ep1_urge_ratio": "ep1_urge_ratio", "ep1_pos_ratio": "ep1_pos_ratio",
        "ep1_neg_ratio": "ep1_neg_ratio",
    }
    out = ml[[c for c in cols if c in ml.columns]].copy()
    return out


def _comment_agg():
    ml = pd.read_parquet(DATA_PROCESSED / "ml_dataset.parquet")
    cols = ["drama_id", "hot_cmt_cnt", "hot_digg_mean", "log_hot_digg_max",
            "hot_rating_mean", "hot_rating_std", "hot_rating_good", "hot_rating_bad",
            "ep1_cmt_cnt", "ep1_digg_mean", "ep1_hot_ratio", "ep1_super_ratio",
            "ep1_urge_ratio", "ep1_pos_ratio", "ep1_neg_ratio"]
    out = ml[[c for c in cols if c in ml.columns]].copy()
    if "log_hot_digg_max" in out.columns:
        out["hot_digg_max"] = np.expm1(out["log_hot_digg_max"]).round(0)
        out = out.drop(columns=["log_hot_digg_max"])
    return out


def load_to_mysql():
    banner("写入 MySQL")
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("  !! 缺少 SQLAlchemy，请先 pip install sqlalchemy pymysql")
        return False
    try:
        eng = create_engine(MYSQL_URL)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        print("  数据库连接成功")
    except Exception as e:
        print(f"  !! 无法连接 MySQL：{type(e).__name__}")
        print("     请先启动 MySQL 并执行建表脚本：")
        print("       mysql -u root -p < sql/douyin_schema.sql")
        print(f"     当前连接串：{MYSQL_URL}（可在 src/config.py 修改）")
        return False

    ods = ods_tables()
    order = [("drama_info", ods["drama_info"]),
             ("drama_hot_comment", ods["drama_hot_comment"]),
             ("drama_ep1_comment", ods["drama_ep1_comment"]),
             ("drama_comment_agg", _comment_agg()),
             ("drama_feature_wide", _wide_table())]
    for t, d in order:
        d.to_sql(t, eng, if_exists="append", index=False, chunksize=2000, method="multi")
        print(f"  {t:<20} 写入 {len(d):,} 行")

    print("\n  入库完成，可用以下 SQL 验证：")
    print("    USE douyin_drama;")
    print("    SELECT * FROM v_heat_rank LIMIT 10;")
    print("    SELECT * FROM v_genre_heat;")
    print("    SELECT * FROM v_comment_signal LIMIT 10;")
    return True


# SQLite 版视图：与 sql/douyin_schema.sql 的三个视图语义一致。
# 差异仅在于 v_genre_heat —— MySQL 用 SUBSTRING_INDEX + 数字辅助表炸开多标签，
# SQLite 无该函数，改为在 Python 侧物化一张 genre_split 辅助表后再建视图，
# 结果集完全相同（drama_cnt >= 19 的 20 种题材，与建模题材口径一致）。
SQLITE_VIEWS = [
    ("v_heat_rank", """
CREATE VIEW v_heat_rank AS
SELECT d.rank_no, d.drama_name, d.category, d.episodes, d.heat_value,
       ROUND(d.heat_value / 10000.0, 0)          AS heat_wan,
       d.hot_rating_avg, a.hot_cmt_cnt, a.ep1_cmt_cnt, a.ep1_digg_mean,
       CASE WHEN d.rank_tag <> '' THEN d.rank_tag ELSE NULL END AS rank_tag
FROM drama_info d
LEFT JOIN drama_comment_agg a ON d.drama_id = a.drama_id
ORDER BY d.heat_value DESC
"""),
    ("v_genre_heat", """
CREATE VIEW v_genre_heat AS
SELECT g.genre,
       COUNT(*)                          AS drama_cnt,
       ROUND(AVG(d.heat_value)/10000.0, 0) AS avg_heat_wan,
       ROUND(MAX(d.heat_value)/10000.0, 0) AS max_heat_wan,
       SUM(f.is_hot)                     AS hot_cnt,
       ROUND(AVG(f.is_hot)*100, 2)       AS hot_rate_pct
FROM genre_split g
JOIN drama_info d          ON g.drama_id = d.drama_id
JOIN drama_feature_wide f  ON g.drama_id = f.drama_id
GROUP BY g.genre
HAVING drama_cnt >= 19
ORDER BY avg_heat_wan DESC
"""),
    ("v_comment_signal", """
CREATE VIEW v_comment_signal AS
SELECT d.drama_name, d.heat_value, f.is_hot,
       f.hot_rating_mean AS rating_mean,
       f.hot_rating_std  AS rating_std,
       f.ep1_digg_mean   AS ep1_digg_mean,
       f.ep1_hot_ratio   AS ep1_hot_ratio,
       f.ep1_urge_ratio  AS ep1_urge_ratio,
       f.ep1_pos_ratio   AS ep1_pos_ratio,
       f.ep1_neg_ratio   AS ep1_neg_ratio
FROM drama_info d
JOIN drama_feature_wide f ON d.drama_id = f.drama_id
ORDER BY d.heat_value DESC
"""),
]


def _genre_split(d1):
    """把 drama_info.category 的多标签按 "/" 炸成 (drama_id, genre) 长表"""
    rows = []
    for did, cat in zip(d1["drama_id"], d1["category"].fillna("")):
        for g in str(cat).split("/"):
            g = g.strip()
            if g:
                rows.append((int(did), g))
    return pd.DataFrame(rows, columns=["drama_id", "genre"]).drop_duplicates()


def export_sqlite():
    """无 MySQL 环境时的轻量替代：导出 SQLite（5 表 + 1 辅助表 + 3 视图）"""
    banner("导出 SQLite 版本（MySQL 不可用时的替身）")
    import sqlite3
    db = OUTPUTS / "douyin_drama.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)

    ods = ods_tables()
    ods["drama_info"].to_sql("drama_info", conn, if_exists="replace", index=False)
    ods["drama_hot_comment"].to_sql("drama_hot_comment", conn, if_exists="replace", index=False)
    ods["drama_ep1_comment"].to_sql("drama_ep1_comment", conn, if_exists="replace", index=False)
    _comment_agg().to_sql("drama_comment_agg", conn, if_exists="replace", index=False)
    _wide_table().to_sql("drama_feature_wide", conn, if_exists="replace", index=False)
    # v_genre_heat 依赖的辅助表（对应 MySQL 的 SUBSTRING_INDEX 炸开结果）
    _genre_split(ods["drama_info"]).to_sql("genre_split", conn, if_exists="replace", index=False)

    for name, ddl in SQLITE_VIEWS:
        conn.execute(f"DROP VIEW IF EXISTS {name}")
        conn.execute(ddl)
    conn.commit()

    n_tab = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    n_view = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'").fetchone()[0]
    # 视图自检：能查出数据才算建成功
    probe = {n: conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
             for n, _ in SQLITE_VIEWS}
    conn.close()

    print(f"  {n_tab} 张表（含 1 张辅助表 genre_split）+ {n_view} 个视图 已写入 → {db}")
    for n, c in probe.items():
        print(f"      · {n:<18} 返回 {c:>4} 行")
    return db


def main():
    ap = argparse.ArgumentParser(description="微短剧数据集 MySQL 入库与治理巡检")
    ap.add_argument("--check", action="store_true", help="只巡检不写库")
    ap.add_argument("--export", action="store_true", help="导出 SQLite 版本")
    ap.add_argument("--skip-check", action="store_true", help="跳过巡检")
    args = ap.parse_args()

    if not args.skip_check:
        governance_check()
    if args.check:
        return
    if args.export:
        export_sqlite()
        return
    if not load_to_mysql():
        print("\n  自动降级为 SQLite 导出 ...")
        export_sqlite()


if __name__ == "__main__":
    main()
