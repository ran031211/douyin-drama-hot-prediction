"""
一键运行全流程：真实数据加载 → 预处理 → EDA → 建模 → 部署导出 → 验证
用法：python src/run_all.py

可视化由 webapp/ 的 Flask 服务承担（见 webapp/run.py），不再产出单文件大屏。
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGES = [
    ("模块一 · 数据采集（真实数据加载与原始层构建）", "load_real_data"),
    ("模块二 · 大数据预处理 + 特征工程", "preprocess"),
    ("模块一·附 · SQL 三层建模（ODS/DWS/ADS）+ 治理巡检", "build_sql"),
    ("模块三 · 大数据分析（EDA）", "eda"),
    ("模块四 · 大数据挖掘（XGBoost 建模）", "model"),
    ("模块四·附 · 部署模型导出（树结构编译为 JSON）", "export_webmodel"),
    ("附加 · 端到端验证（部署模型结构一致性）", "verify_webmodel"),
]


def main():
    t_all = time.time()
    results = []
    for i, (title, mod) in enumerate(STAGES, 1):
        print("\n" + "#" * 68)
        print(f"# 阶段 {i}/{len(STAGES)}  {title}")
        print("#" * 68)
        t0 = time.time()
        try:
            m = __import__(mod)
            rc = m.main()
            # 只有「真正的整数退出码」才代表失败信号；
            # 多数模块的 main() 返回审计/汇总 dict（正常），不得误判。
            if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
                raise RuntimeError(f"{mod}.main() 返回非零退出码 {rc}（校验未全通过）")
            results.append((title, "成功", round(time.time() - t0, 1), None))
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"{type(e).__name__}: {e}"
            print(f"\n  !! 阶段失败：{msg[:600]}")
            print(tb[-1200:])
            results.append((title, "失败", round(time.time() - t0, 1), msg[:300]))

    print("\n" + "=" * 68)
    print(" 全流程执行汇总")
    print("=" * 68)
    for title, st, sec, err in results:
        mark = "✓" if st == "成功" else "✗"
        print(f"  {mark} {title:<40} {sec:>6.1f}s  {st}")
        if err:
            print(f"      └─ {err}")
    print("-" * 68)
    print(f"  总耗时 {time.time() - t_all:.1f}s")
    ok = sum(1 for r in results if r[1] == "成功")
    print(f"  成功 {ok}/{len(results)}")
    print("=" * 68)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
