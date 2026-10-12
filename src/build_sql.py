"""
=======================================================================
 模块一 · 附：SQL 三层建模（ODS / DWS / ADS）+ 数据治理巡检
-----------------------------------------------------------------------
 说明：load_mysql.py 是面向交互式 MySQL 入库的入口（需要真实 MySQL 实例，
       默认会先尝试连接）。本模块是它在**一键全流程**里的离线替身：
       只做两件确定能跑完的事 ——
         ① 六维数据治理巡检（唯一性/完整性/一致性/有效性/隐私性/分布）
         ② 落地 SQLite 版本（outputs/douyin_drama.sqlite，5 张表）

       若要写入真实 MySQL：先 `mysql < sql/douyin_schema.sql` 建表，
       再执行 `python src/load_mysql.py`。

 用法：python src/build_sql.py
=======================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_mysql


def main():
    print("=" * 68)
    print(" 模块一 · 附：SQL 三层建模 + 治理巡检（离线通道）")
    print("=" * 68)
    load_mysql.governance_check()
    load_mysql.export_sqlite()
    print("\n  SQL 层构建完成：ODS(drama_info / drama_hot_comment / drama_ep1_comment)")
    print("                  DWS(drama_comment_agg)")
    print("                  ADS(drama_feature_wide) + 3 视图")
    return 0


if __name__ == "__main__":
    sys.exit(main())
