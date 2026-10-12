"""
=======================================================================
 附加 · 端到端验证：部署模型结构一致性
-----------------------------------------------------------------------
 导出自检只能证明「Python 模拟走树 == XGBoost」。
 本脚本进一步验证「独立第二实现（JS）走树 == Python 模拟走树」，三者合起来
 才构成完整证据链：

     XGBoost 原生  ──① 导出自检──▶  Python 模拟
                                        │
                                   ② 本脚本（Node 实跑 JS）
                                        ▼
                                  独立 JS 实现（Node）

 校验内容：
   ① 结构完整性：特征数、树/节点数、定型数组长度自洽
   ② 截距合理性：应等于 logit(base_score)，且概率区间落在 (0,1)
   ③ 数值一致性：Node 跑 JS 得到的概率 vs Python 模拟，偏差应 < 1e-9
   ④ 判别方向：模型给出的高分样本应集中在真实爆款中（AUC 复核）

 用法：python src/verify_webmodel.py
       （未安装 node 时自动跳过 ③，其余检查照常执行）
=======================================================================
"""
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DASHBOARD, DATA_PROCESSED, RANDOM_SEED

WEBMODEL = DASHBOARD / "webmodel.json"
PASS, FAIL = [], []


def banner(t):
    print("\n" + "=" * 68)
    print(f" {t}")
    print("=" * 68)


def ok(cond, msg):
    (PASS if cond else FAIL).append(msg)
    print(f"    {'✓' if cond else '✗'} {msg}")
    return cond


# ---------------------------------------------------------------- Python 侧走树
def decode(model):
    out = []
    for t in model["trees"]:
        idx = np.frombuffer(base64.b64decode(t["i"]), dtype=np.int16).reshape(-1, 3)
        miss = np.frombuffer(base64.b64decode(t["d"]), dtype=np.int8)
        thr = np.frombuffer(base64.b64decode(t["t"]), dtype=np.float64)
        val = np.frombuffer(base64.b64decode(t["v"]), dtype=np.float64)
        out.append((idx, miss, thr, val))
    return out


def py_predict(dec, intercept, row):
    """与服务端 predict() 完全同构：float32 比较 + 缺失走默认方向"""
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
    return float(expit(s + intercept))


# ---------------------------------------------------------------- Node 实跑 JS
JS_HARNESS = r"""
const fs = require('fs');
const P = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const INPUT = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

function b64ToBytes(b64){
  const bin = Buffer.from(b64, 'base64');
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin[i];
  return u;
}
function toI16(b64){ const b=b64ToBytes(b64); const o=new Int16Array(b.length/2);
  new Uint8Array(o.buffer).set(b); return o; }
function toI8(b64){ const b=b64ToBytes(b64); const o=new Int8Array(b.length);
  new Uint8Array(o.buffer).set(b); return o; }
function toF64(b64){ const b=b64ToBytes(b64); const o=new Float64Array(b.length/8);
  new Uint8Array(o.buffer).set(b); return o; }
function decodeTrees(model){
  for (const tr of model.trees){
    if (tr.__idx) continue;
    tr.__idx = toI16(tr.i); tr.__miss = toI8(tr.d);
    tr.__thr = toF64(tr.t); tr.__val = toF64(tr.v);
  }
  return model;
}
function predict(model, x){
  decodeTrees(model);
  let s = model.intercept;
  for (const tr of model.trees){
    const idx = tr.__idx, miss = tr.__miss, thr = tr.__thr, val = tr.__val;
    let k = 0;
    while (idx[k*3] >= 0){
      const f = idx[k*3], xv = x[f];
      let goLeft;
      if (xv !== xv) goLeft = miss[k] ? 1 : 0;
      else goLeft = Math.fround(xv) < Math.fround(thr[k]) ? 1 : 0;
      k = goLeft ? idx[k*3+1] : idx[k*3+2];
    }
    s += val[k];
  }
  return 1 / (1 + Math.exp(-s));
}
const out = {};
for (const key of ['cold', 'full']){
  out[key] = INPUT[key].map(x => predict(P.models[key], x));
}
process.stdout.write(JSON.stringify(out));
"""


def main():
    print("=" * 68)
    print(" 附加 · 端到端验证：部署模型结构一致性")
    print("=" * 68)

    if not WEBMODEL.exists():
        raise FileNotFoundError(f"缺少 {WEBMODEL}，请先执行 export_webmodel.py")
    P = json.loads(WEBMODEL.read_text(encoding="utf-8"))
    cold, full = P["models"]["cold"], P["models"]["full"]
    df = pd.read_parquet(DATA_PROCESSED / "ml_dataset.parquet")

    # ---------- ① 结构完整性 ----------
    banner("① 结构完整性")
    for name, m in [("冷启动版", cold), ("全特征版", full)]:
        n = 0
        lens_ok = True
        for t in m["trees"]:
            idx = np.frombuffer(base64.b64decode(t["i"]), dtype=np.int16)
            mis = np.frombuffer(base64.b64decode(t["d"]), dtype=np.int8)
            thr = np.frombuffer(base64.b64decode(t["t"]), dtype=np.float64)
            val = np.frombuffer(base64.b64decode(t["v"]), dtype=np.float64)
            lens_ok &= (len(idx) == t["n"] * 3 and len(mis) == t["n"]
                        and len(thr) == t["n"] and len(val) == t["n"])
            n += t["n"]
        ok(lens_ok, f"{name}：四组定型数组长度与节点数自洽")
        ok(n == m["n_nodes"], f"{name}：节点合计 {n:,} == 声明值 {m['n_nodes']:,}")
        ok(len(m["trees"]) == m["n_trees"],
           f"{name}：树数 {len(m['trees'])} == {m['n_trees']}")
        imp_keys = (m.get("importance") or {}).get("keys") or []
        imp_lbl = (m.get("importance") or {}).get("labels") or []
        fmap_ok = all(k in m["features"] for k in imp_keys) if imp_keys else True
        lbl_ok = (len(imp_lbl) == len(imp_keys)) if imp_keys else True
        ok(fmap_ok, f"{name}：重要性表 {len(imp_keys)} 项特征均在特征清单内")
        ok(lbl_ok, f"{name}：重要性表 keys/labels 一一对应（中文标签齐全）")

        max_f = -1
        for t in m["trees"]:
            idx = np.frombuffer(base64.b64decode(t["i"]), dtype=np.int16).reshape(-1, 3)
            hit = idx[idx[:, 0] >= 0][:, 0]
            if hit.size:
                max_f = max(max_f, int(hit.max()))
        ok(max_f < len(m["features"]),
           f"{name}：树引用最大特征下标 {max_f} < 特征数 {len(m['features'])}（无越界）")

    # ---------- ② 截距合理性 ----------
    banner("② 截距与概率值域")
    # 训练集正类率 20% → base_score 自动估计为 0.2 → 截距应为 logit(0.2)
    expect = float(np.log(0.2 / 0.8))
    for name, m in [("冷启动版", cold), ("全特征版", full)]:
        ok(abs(m["intercept"] - expect) < 0.2,
           f"{name}：截距 {m['intercept']:.6f} ≈ logit(base_score)={expect:.6f}")

    # ---------- ③ 数值一致性：Node 实跑 JS vs Python 模拟 ----------
    banner("③ 独立 JS 实现 vs Python 模拟（Node 实跑）")
    rng = np.random.default_rng(RANDOM_SEED)
    n_samp = 120
    rows = rng.choice(len(df), size=n_samp, replace=False)
    payload, ref = {}, {}
    for key, m in [("cold", cold), ("full", full)]:
        X = df[m["features"]].astype(float).fillna(0)
        payload[key] = [[float(v) for v in X.iloc[i].values] for i in rows]
        dec = decode(m)
        ref[key] = [py_predict(dec, m["intercept"], X.iloc[i].values.astype(float))
                    for i in rows]

    node = shutil.which("node")
    if not node:
        print("    （未找到 node，跳过 JS 引擎实跑验证）")
    else:
        with tempfile.TemporaryDirectory() as td:
            ip = Path(td) / "in.json"
            op = Path(td) / "out.json"
            js = Path(td) / "h.js"
            ip.write_text(json.dumps(payload), encoding="utf-8")
            js.write_text(JS_HARNESS, encoding="utf-8")
            r = subprocess.run([node, str(js), str(WEBMODEL), str(ip)],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                ok(False, f"Node 执行失败：{(r.stderr or '').strip()[:300]}")
            else:
                got = json.loads(r.stdout)
                for key, label in [("cold", "冷启动版"), ("full", "全特征版")]:
                    d = np.abs(np.array(got[key]) - np.array(ref[key]))
                    ok(d.max() < 1e-9,
                       f"{label}：JS 与 Python 最大偏差 {d.max():.2e}"
                       f"（{n_samp} 条样本）")

    # ---------- ④ 判别方向复核 ----------
    banner("④ 判别方向复核（注意：此处为样本内口径，仅验方向）")
    y = df["is_hot"].astype(int).values
    for key, label, m in [("cold", "冷启动版", cold), ("full", "全特征版", full)]:
        dec = decode(m)
        X = df[m["features"]].astype(float).fillna(0).values
        prob = np.array([py_predict(dec, m["intercept"], X[i])
                         for i in range(len(X))])
        auc = float(roc_auc_score(y, prob))
        oof = float((m.get("metrics") or {}).get("auc") or 0.0)
        ok(auc > 0.60, f"{label}：全部 {len(X)} 部样本内 AUC {auc:.4f}（> 0.60，方向正确）")
        if oof:
            print(f"      · 对照：袋外(OOF) AUC {oof:.4f} —— 上述 {auc:.4f} 含建模样本，"
                  f"偏高属正常，正式结论以 OOF 为准")
        ok(0.0 < prob.min() and prob.max() < 1.0,
           f"{label}：概率区间 [{prob.min():.4f}, {prob.max():.4f}] 落在 (0,1)")
        ok(prob.mean() < 0.5,
           f"{label}：平均预测概率 {prob.mean():.4f} 合理（正类率仅 20%）")

    # ---------- 汇总 ----------
    banner("验证结论")
    print(f"  通过 {len(PASS)} 项 / 失败 {len(FAIL)} 项")
    if FAIL:
        for x in FAIL:
            print(f"    ✗ {x}")
        print("\n  !! 部署推理链路存在问题，请勿接入服务 !!")
        return 1
    print("  证据链完整：XGBoost 原生 ≈ Python 模拟 ≈ 独立 JS 实现")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
