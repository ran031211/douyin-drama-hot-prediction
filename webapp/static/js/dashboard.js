/* =====================================================================
   dashboard.js · 总览页
   ===================================================================== */
(function () {
  "use strict";
  const { C, AXIS, GROUP_COLOR, TIER_COLOR } = Charts;

  /* ---------------- KPI ---------------- */
  function renderKpis(kpi) {
    const box = U.el("kpis");
    const items = [
      [U.int(kpi.dramas), "", "剧集样本"],
      [U.num(kpi.hot_rate, 1), "%", "爆款占比"],
      [U.int(kpi.hot_threshold_wan), "万", "爆款热度阈值"],
      [U.int(kpi.median_heat_wan), "万", "热度中位数"],
      [U.int(kpi.median_episodes), "集", "集数中位数"],
      [U.num(kpi.median_rating, 2), "", "热评均分中位数"]
    ];
    box.replaceChildren();
    items.forEach(function (it) {
      const card = U.h("div", "kpi");
      const v = U.h("div", "kpi__value");
      v.textContent = it[0];
      if (it[1]) {
        const s = document.createElement("small");
        s.textContent = it[1];
        v.appendChild(s);
      }
      card.appendChild(v);
      card.appendChild(U.h("div", "kpi__label", it[2]));
      box.appendChild(card);
    });
    U.el("head-scale").textContent = U.int(kpi.dramas);
  }

  /* ---------------- 模型表 ---------------- */
  function renderModels(cold, full) {
    const tbody = U.el("model-table").querySelector("tbody");
    [[cold, "冷启动版"], [full, "全特征版"]].forEach(function (pair) {
      const m = pair[0];
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", null, pair[1]));
      tr.appendChild(U.h("td", "mono", m.n_features + " 维"));
      tr.appendChild(U.h("td", "mono", U.int(m.n_trees)));
      tr.appendChild(U.h("td", "is-num", U.num(m.metrics.auc, 4)));
      tbody.appendChild(tr);
    });
  }

  /* ---------------- 回测自检 ---------------- */
  function renderBacktest(bt) {
    const box = U.el("backtest-body");
    const h = bt.holdout;
    U.el("backtest-state").textContent = h ? "hold-out 划分已复现" : "降级运行";
    box.replaceChildren();

    if (!h) {
      box.appendChild(U.h("div", "notice notice--warn", "未安装 scikit-learn，无法复现 hold-out 划分。"));
      return;
    }

    const rows = [
      ["测试集样本", U.int(h.n_test) + " 部（" + U.num(h.test_size * 100, 0) + "%）"],
      ["随机种子", String(h.random_seed)],
      ["冷启动 AUC", U.num(h.auc_cold, 4)],
      ["全特征 AUC", U.num(h.auc_full, 4)]
    ];
    const tbl = U.h("table", "table");
    const tb = document.createElement("tbody");
    rows.forEach(function (r) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", "muted", r[0]));
      tr.appendChild(U.h("td", "is-num", r[1]));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    box.appendChild(tbl);

    const ok = h.auc_cold_match && h.auc_full_match;
    const tag = U.h("div", "row");
    tag.style.marginTop = "var(--sp-3)";
    const badge = U.h("span", "badge " + (ok ? "badge--ok" : "badge--fail"),
      ok ? "与产物记录一致" : "与产物记录不一致");
    tag.appendChild(badge);
    tag.appendChild(U.h("span", "muted", "产物记录 " + U.num(h.auc_cold_recorded, 4)
      + " / " + U.num(h.auc_full_recorded, 4)));
    box.appendChild(tag);
  }

  /* ---------------- 图表：题材爆款率 ---------------- */
  function renderGenreChart(ranking) {
    const items = ranking.items;
    const base = ranking.baseline;
    Charts.make("chart-genre", {
      grid: { left: 8, right: 66, top: 16, bottom: 28, containLabel: true },
      tooltip: {
        trigger: "item",
        formatter: function (p) {
          const d = items[p.dataIndex];
          return "<b>" + d.genre + "</b>（" + d.tier + "）<br>"
            + "爆款率 " + U.num(d.hot_rate, 1) + "%<br>"
            + "95% 区间 " + U.num(d.ci_low, 1) + "% ~ " + U.num(d.ci_high, 1) + "%<br>"
            + "样本 " + d.n + " 部";
        }
      },
      xAxis: Object.assign({
        type: "value", min: 0, max: 72,
        axisLabel: { show: false }, splitLine: { show: false }
      }, AXIS),
      yAxis: Object.assign({
        type: "category",
        data: items.map(function (d) { return d.genre; }),
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: {
          color: C.text2, fontSize: 11,
          formatter: function (v, i) { return v + "  " + items[i].n + "部"; }
        }
      }, AXIS),
      series: [
        {
          type: "bar", barWidth: "55%", silent: true,
          data: items.map(function (d) {
            return {
              value: d.hot_rate,
              itemStyle: { color: TIER_COLOR[d.tier] || C.mid, borderRadius: [0, 3, 3, 0] }
            };
          }),
          label: {
            show: true, position: "right", fontSize: 11, color: C.text2,
            formatter: function (p) { return U.num(p.value, 1) + "%"; }
          },
          markLine: {
            silent: true, symbol: "none",
            lineStyle: { color: C.text3, type: "dashed", width: 1 },
            data: [{
              xAxis: base,
              label: {
                formatter: "基准 " + U.num(base, 0) + "%",
                fontSize: 11, color: C.text3, position: "end"
              }
            }]
          }
        },
        {
          type: "scatter", symbol: "rect", z: 5, silent: true,
          data: items.map(function (d) {
            return {
              value: [(d.ci_low + d.ci_high) / 2, d.genre],
              symbolSize: [Math.max(2, d.ci_high - d.ci_low), 2],
              itemStyle: { color: C.mid, opacity: .55 }
            };
          })
        }
      ]
    });
  }

  /* ---------------- 图表：全局重要性 ---------------- */
  function renderImportanceChart(imp) {
    const labels = imp.labels.slice().reverse();
    const values = imp.values.slice().reverse();
    const groups = imp.groups.slice().reverse();
    Charts.make("chart-importance", {
      grid: { left: 8, right: 52, top: 10, bottom: 10, containLabel: true },
      tooltip: {
        trigger: "item",
        formatter: function (p) {
          return p.name + "：" + U.num(p.value, 2) + "%";
        }
      },
      xAxis: Object.assign({ type: "value", axisLabel: { show: false }, splitLine: { show: false } }, AXIS),
      yAxis: Object.assign({
        type: "category", data: labels,
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { color: C.text2, fontSize: 11 }
      }, AXIS),
      series: [{
        type: "bar", barWidth: "60%",
        data: values.map(function (v, i) {
          const col = GROUP_COLOR[groups[i]] || C.mid;
          return { value: v, itemStyle: { color: col, borderRadius: [0, 3, 3, 0] } };
        }),
        label: {
          show: true, position: "right", fontSize: 11, color: C.text3,
          formatter: function (p) { return U.num(p.value, 1) + "%"; }
        }
      }]
    });
  }

  /* ---------------- 选题分档 ---------------- */
  function renderTiers(tiers) {
    const box = U.el("tiers");
    U.el("tier-baseline").textContent = "全站基准 " + U.num(tiers.baseline, 0) + "%";
    box.replaceChildren();
    tiers.groups.forEach(function (g) {
      const wrap = U.h("div", "tier");
      const head = U.h("div", "tier__head");
      const dot = document.createElement("i");
      dot.style.background = TIER_COLOR[g.tier] || C.mid;
      head.appendChild(dot);
      head.appendChild(document.createTextNode(g.tier));
      head.appendChild(U.h("span", "tier__count", g.count + " 个题材 · " + g.rule));
      wrap.appendChild(head);

      const items = U.h("div", "tier__items");
      g.members.forEach(function (m) {
        const chip = U.h("span", "tier__item");
        chip.appendChild(U.h("b", null, m.genre));
        chip.appendChild(document.createTextNode(U.num(m.hot_rate, 1) + "%"));
        chip.appendChild(U.h("em", null, "n=" + m.n));
        items.appendChild(chip);
      });
      wrap.appendChild(items);
      box.appendChild(wrap);
    });
  }

  /* ---------------- 双标签组合 ---------------- */
  function renderCombos(cb) {
    U.el("combo-min").textContent = cb.min_samples;
    U.el("combo-n").textContent = cb.n_pairs;
    function fill(id, rows) {
      U.fill(U.el(id), rows, function (r) {
        const tr = document.createElement("tr");
        tr.appendChild(U.h("td", null, r.a + " + " + r.b));
        tr.appendChild(U.h("td", "is-num", U.num(r.hot_rate, 1) + "%"));
        tr.appendChild(U.h("td", "is-num", r.n));
        return tr;
      });
    }
    fill("combo-best", cb.best || []);
    fill("combo-worst", cb.worst || []);
  }

  /* ---------------- 启动 ---------------- */
  /* 缓存接口数据，主题切换后图表用同一数据按新配色重建 */
  const chartCache = { genres: null, model: null };

  Charts.onThemeChange(function () {
    if (chartCache.genres) renderGenreChart(chartCache.genres);
    if (chartCache.model) renderImportanceChart(chartCache.model);
  });

  (async function init() {
    try {
      const meta = await API.get("/api/meta");
      renderKpis(meta.overview.kpi);
      renderModels(meta.models, meta.models_full);

      const genres = await API.get("/api/insights/genres");
      chartCache.genres = genres.ranking;
      renderGenreChart(genres.ranking);
      renderTiers(genres.tiers);

      const model = await API.get("/api/insights/model");
      chartCache.model = model.importance;
      renderImportanceChart(model.importance);

      const combos = await API.get("/api/insights/combos");
      renderCombos(combos);
    } catch (err) {
      U.showError(U.el("kpis"), err);
    }

    try {
      renderBacktest(await API.get("/api/backtest"));
    } catch (err) {
      U.showError(U.el("backtest-body"), err);
    }
  })();
})();
