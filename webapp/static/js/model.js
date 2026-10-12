/* =====================================================================
   model.js · 模型评估页
   ---------------------------------------------------------------------
   区分两组指标，避免混淆：
     「论文口径」—— src/ 流水线用 5 折交叉与 hold-out 得到，对应论文表格；
     「在线回测」—— 部署模型在服务端复现同一 hold-out 划分后算出的结果。
   ===================================================================== */
(function () {
  "use strict";
  const { C, AXIS, GROUP_COLOR } = Charts;

  /* ---------------- ROC / PR ---------------- */
  function renderCurves(curves) {
    const palette = { cold: C.accent, full: C.top };
    const names = Object.keys(curves);

    function lineData(curve) {
      const out = [];
      curve.x.forEach(function (v, i) { out.push([v, curve.y[i]]); });
      return out;
    }

    Charts.make("chart-roc", {
      grid: { left: 46, right: 14, top: 34, bottom: 30 },
      legend: {
        data: names.map(function (k) { return curves[k].label; }),
        right: 0, top: 0, itemWidth: 14, itemHeight: 8,
        textStyle: { fontSize: 11, color: C.text2 }
      },
      tooltip: { trigger: "axis", valueFormatter: function (v) { return U.num(v, 3); } },
      xAxis: Object.assign({ type: "value", min: 0, max: 1, name: "FPR",
        nameTextStyle: { color: C.text3, fontSize: 11 } }, AXIS),
      yAxis: Object.assign({ type: "value", min: 0, max: 1, name: "TPR",
        nameTextStyle: { color: C.text3, fontSize: 11 } }, AXIS),
      series: names.map(function (k) {
        return {
          name: curves[k].label, type: "line", showSymbol: false, smooth: true,
          data: lineData(curves[k].roc),
          lineStyle: { width: 2, color: palette[k] || C.mid }
        };
      }).concat([{
        name: "随机基线", type: "line", data: [[0, 0], [1, 1]], showSymbol: false,
        lineStyle: { width: 1, type: "dashed", color: C.axis }, silent: true
      }])
    });

    Charts.make("chart-pr", {
      grid: { left: 46, right: 14, top: 34, bottom: 30 },
      legend: {
        data: names.map(function (k) { return curves[k].label; }),
        right: 0, top: 0, itemWidth: 14, itemHeight: 8,
        textStyle: { fontSize: 11, color: C.text2 }
      },
      tooltip: { trigger: "axis", valueFormatter: function (v) { return U.num(v, 3); } },
      xAxis: Object.assign({ type: "value", min: 0, max: 1, name: "Recall",
        nameTextStyle: { color: C.text3, fontSize: 11 } }, AXIS),
      yAxis: Object.assign({ type: "value", min: 0, max: 1, name: "Precision",
        nameTextStyle: { color: C.text3, fontSize: 11 } }, AXIS),
      series: names.map(function (k) {
        return {
          name: curves[k].label, type: "line", showSymbol: false, smooth: true,
          data: lineData(curves[k].pr),
          lineStyle: { width: 2, color: palette[k] || C.mid }
        };
      })
    });
  }

  /* ---------------- 消融 ---------------- */
  function renderAblation(rows) {
    Charts.make("chart-ablation", {
      grid: { left: 8, right: 46, top: 16, bottom: 26, containLabel: true },
      tooltip: {
        trigger: "item",
        formatter: function (p) {
          const d = rows[p.dataIndex];
          return d.feature_set + "<br>AUC " + U.num(d.auc_mean, 4)
            + "<br>维度 " + d.n_features
            + (d.auc_gain !== null && d.auc_gain !== undefined
              ? "<br>边际增量 " + (d.auc_gain >= 0 ? "+" : "") + U.num(d.auc_gain, 4) : "");
        }
      },
      xAxis: Object.assign({ type: "value", min: 0.55, max: 0.78,
        axisLabel: { show: false }, splitLine: { show: false } }, AXIS),
      yAxis: Object.assign({
        type: "category",
        data: rows.map(function (d) { return d.feature_set; }),
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { color: C.text2, fontSize: 11 }
      }, AXIS),
      series: [{
        type: "bar", barWidth: "52%",
        data: rows.map(function (d, i) {
          return {
            value: d.auc_mean,
            itemStyle: {
              color: i === rows.length - 1 ? C.accent : C.accentSoft,
              borderRadius: [0, 3, 3, 0]
            }
          };
        }),
        label: {
          show: true, position: "right", fontSize: 11, color: C.text2,
          formatter: function (p) {
            const d = rows[p.dataIndex];
            return U.num(p.value, 4) + (d.auc_gain ? "  (+" + U.num(d.auc_gain, 4) + ")" : "");
          }
        }
      }]
    });
  }

  /* ---------------- 特征组占比 ---------------- */
  function renderGroups(groups) {
    const data = Object.keys(groups).map(function (k) {
      return { name: k, value: groups[k], itemStyle: { color: GROUP_COLOR[k] || C.mid } };
    });
    Charts.make("chart-groups", {
      tooltip: { trigger: "item", formatter: "{b}：{c}%（{d}%）" },
      legend: { bottom: 0, itemWidth: 10, itemHeight: 8, textStyle: { fontSize: 11, color: C.text2 } },
      series: [{
        type: "pie", radius: ["46%", "70%"], center: ["50%", "44%"],
        avoidLabelOverlap: true, itemStyle: { borderColor: C.surface, borderWidth: 2 },
        label: { formatter: "{b}\n{c}%", fontSize: 11, color: C.text2 },
        data: data
      }]
    });
  }

  /* ---------------- 表格 ---------------- */
  function renderClfTable(rows) {
    U.fill(U.el("tbl-clf").querySelector("tbody"), rows, function (d) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", null, d.model));
      tr.appendChild(U.h("td", "is-num", U.num(d.auc, 4)));
      tr.appendChild(U.h("td", "is-num", U.num(d.cv_auc, 4)));
      tr.appendChild(U.h("td", "is-num", U.num(d.f1, 4)));
      tr.appendChild(U.h("td", "is-num", U.num(d.ap, 4)));
      return tr;
    });
  }

  function renderRegTable(rows) {
    U.fill(U.el("tbl-reg").querySelector("tbody"), rows, function (d) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", null, d.model));
      tr.appendChild(U.h("td", "is-num", U.num(d.R2, 4)));
      tr.appendChild(U.h("td", "is-num", U.num(d.MAE, 4)));
      tr.appendChild(U.h("td", "is-num", U.num(d.RMSE, 4)));
      tr.appendChild(U.h("td", "is-num", U.num(d.MAE_wan, 1)));
      return tr;
    });
  }

  function renderShapTable(top, direction) {
    const rows = Object.keys(top).map(function (k) { return [k, top[k], direction[k] || "—"]; });
    U.fill(U.el("tbl-shap").querySelector("tbody"), rows, function (r, i) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", "mono", i + 1));
      tr.appendChild(U.h("td", null, r[0]));
      tr.appendChild(U.h("td", "is-num", U.num(r[1], 4)));
      tr.appendChild(U.h("td", null, r[2]));
      return tr;
    });
  }

  /* ---------------- 稳健性 ---------------- */
  function renderRobust(rob) {
    const box = U.el("robust-body");
    const rows = [
      ["重复次数", rob.n_repeats],
      ["AUC 均值", U.num(rob.auc_mean, 4)],
      ["AUC 标准差", U.num(rob.auc_std, 4)],
      ["AUC 区间", U.num(rob.auc_min, 4) + " ~ " + U.num(rob.auc_max, 4)],
      ["F1 均值", U.num(rob.f1_mean, 4)]
    ];
    const tbl = U.h("table", "table");
    const tb = document.createElement("tbody");
    rows.forEach(function (r) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", "muted", r[0]));
      tr.appendChild(U.h("td", "is-num", String(r[1])));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    box.replaceChildren(tbl);
  }

  /* ---------------- 在线回测 ---------------- */
  function renderBacktest(bt) {
    const box = U.el("backtest-body");
    box.replaceChildren();
    const h = bt.holdout;
    if (!h) {
      box.appendChild(U.h("div", "notice notice--warn",
        "全样本拟合度（含训练集）：冷启动 AUC " + U.num(bt.full_sample_fit.auc_cold, 4)
        + "，全特征 " + U.num(bt.full_sample_fit.auc_full, 4) + "。该值非泛化指标。"));
      return;
    }
    const rows = [
      ["测试集样本", U.int(h.n_test)],
      ["随机种子", h.random_seed],
      ["冷启动 AUC", U.num(h.auc_cold, 4) + "  （记录 " + U.num(h.auc_cold_recorded, 4) + "）"],
      ["全特征 AUC", U.num(h.auc_full, 4) + "  （记录 " + U.num(h.auc_full_recorded, 4) + "）"],
      ["反馈特征增量", "+" + U.num(h.auc_gain, 4)],
      ["判定阈值", U.num(h.threshold, 4)],
      ["精确率 / 召回率", U.num(h.precision, 4) + " / " + U.num(h.recall, 4)]
    ];
    const tbl = U.h("table", "table");
    const tb = document.createElement("tbody");
    rows.forEach(function (r) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", "muted", r[0]));
      tr.appendChild(U.h("td", "is-num", String(r[1])));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    box.appendChild(tbl);

    const ok = h.auc_cold_match && h.auc_full_match;
    const line = U.h("div", "row");
    line.style.marginTop = "var(--sp-3)";
    line.appendChild(U.h("span", "badge " + (ok ? "badge--ok" : "badge--fail"),
      ok ? "复现一致" : "复现不一致"));
    box.appendChild(line);
  }

  function renderTop(rows) {
    U.fill(U.el("tbl-top"), rows, function (d) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", null, d.drama_name));
      tr.appendChild(U.h("td", "muted", d.category));
      tr.appendChild(U.h("td", "is-num", U.num(d.prob, 1) + "%"));
      tr.appendChild(U.h("td", "is-num", U.int(d.heat_wan)));
      const td = document.createElement("td");
      td.appendChild(U.h("span", "badge " + (d.is_hot ? "badge--ok" : ""),
        d.is_hot ? "爆款" : "非爆款"));
      tr.appendChild(td);
      return tr;
    });
  }

  /* ---------------- 启动 ---------------- */
  /* 缓存接口数据，主题切换后图表用同一数据按新配色重建 */
  const chartCache = { res: null };

  Charts.onThemeChange(function () {
    if (!chartCache.res) return;
    const res = chartCache.res;
    renderCurves(res.curves);
    renderAblation(res.paper_metrics.ablation || []);
    renderGroups(res.paper_metrics.importance_group || {});
  });

  (async function init() {
    try {
      const res = await API.get("/api/insights/model");
      chartCache.res = res;
      renderCurves(res.curves);
      renderAblation(res.paper_metrics.ablation || []);
      renderGroups(res.paper_metrics.importance_group || {});
      renderClfTable(res.paper_metrics.classification || []);
      renderRegTable(res.paper_metrics.regression || []);
      renderShapTable(res.paper_metrics.shap_top || {}, res.paper_metrics.shap_direction || {});
      renderRobust(res.paper_metrics.robustness || {});
    } catch (err) {
      U.showError(U.el("tbl-clf").parentElement, err);
    }

    try {
      const bt = await API.get("/api/backtest");
      renderBacktest(bt);
      renderTop(bt.top_predicted || []);
    } catch (err) {
      U.showError(U.el("backtest-body"), err);
    }
  })();
})();
