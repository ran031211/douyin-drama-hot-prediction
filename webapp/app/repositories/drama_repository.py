# -*- coding: utf-8 -*-
"""
仓储层 · 剧集原始库（SQLite）
=======================================================================
职责：从 outputs/douyin_drama.sqlite 读取剧集清单与简介文本，支撑
      「数据集浏览」页面；提供分页、关键词检索与排序。

为什么用 SQLite 而不是重复读 CSV：
  · 建模数据集 (ml_dataset.csv) 只保留特征列，**不含剧名与简介原文**；
  · 浏览页需要剧名/简介/排名等展示字段，走原始库更直接；
  · SQLite 是单文件、无服务进程，适合随项目打包演示。
"""
from __future__ import annotations

import sqlite3
import threading
from typing import Any, Iterable

from config import Config

_local = threading.local()

# 允许的排序字段白名单（防止 SQL 注入）
_SORT_FIELDS = {
    "heat": "heat_value",
    "episodes": "episodes",
    "rating": "hot_rating_avg",
    "rank": "rank_no",
}


def _conn() -> sqlite3.Connection:
    """每线程一个只读连接。"""
    conn = getattr(_local, "conn", None)
    if conn is None:
        if not Config.SQLITE_FILE.exists():
            raise FileNotFoundError(f"缺少剧集库：{Config.SQLITE_FILE}")
        conn = sqlite3.connect(
            f"file:{Config.SQLITE_FILE.as_posix()}?mode=ro", uri=True
        )
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


class DramaRepository:
    """剧集清单的只读访问。"""

    @staticmethod
    def list_dramas(
        limit: int = 20,
        offset: int = 0,
        keyword: str = "",
        sort: str = "heat",
        descending: bool = True,
    ) -> dict:
        """分页列出剧集。keyword 在剧名与简介上做模糊匹配。"""
        col = _SORT_FIELDS.get(sort, "heat_value")
        direction = "DESC" if descending else "ASC"

        where, params = "", []
        if keyword:
            where = "WHERE drama_name LIKE ? OR summary LIKE ?"
            like = f"%{keyword}%"
            params = [like, like]

        conn = _conn()
        total = conn.execute(
            f"SELECT COUNT(*) FROM drama_info {where}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"""SELECT drama_id, drama_name, category, episodes,
                       heat_value, hot_rating_avg, rank_no, rank_tag, summary
                FROM drama_info {where}
                ORDER BY {col} {direction} NULLS LAST, drama_id
                LIMIT ? OFFSET ?""",
            [*params, max(1, min(limit, 100)), max(0, offset)],
        ).fetchall()

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [DramaRepository._row_to_dict(r) for r in rows],
        }

    @staticmethod
    def get_drama(drama_id: str) -> dict | None:
        """取单部剧的展示信息 + 题材标签。"""
        conn = _conn()
        row = conn.execute(
            """SELECT drama_id, drama_name, category, episodes, heat_value,
                      heat_text, hot_rating_avg, rank_no, rank_tag, summary
               FROM drama_info WHERE drama_id = ?""",
            [str(drama_id)],
        ).fetchone()
        if row is None:
            return None
        item = DramaRepository._row_to_dict(row)
        item["heat_text"] = row["heat_text"]
        item["genres"] = [
            r["genre"] for r in conn.execute(
                "SELECT genre FROM genre_split WHERE drama_id = ? ORDER BY genre",
                [str(drama_id)],
            ).fetchall()
        ]
        return item

    @staticmethod
    def count() -> int:
        return _conn().execute("SELECT COUNT(*) FROM drama_info").fetchone()[0]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        summary = row["summary"] or ""
        return {
            "drama_id": row["drama_id"],
            "drama_name": row["drama_name"],
            "category": row["category"] or "",
            "episodes": int(row["episodes"]) if row["episodes"] else None,
            "heat_wan": round((row["heat_value"] or 0) / 1e4, 1),
            "rating": round(row["hot_rating_avg"], 2) if row["hot_rating_avg"] else None,
            "rank_no": row["rank_no"],
            "rank_tag": row["rank_tag"] or "",
            "summary": summary,
            "summary_short": summary[:60] + ("…" if len(summary) > 60 else ""),
        }
