/* =====================================================================
   dataset.js · 数据集浏览页
   ===================================================================== */
(function () {
  "use strict";
  const { C } = Charts;

  const state = { offset: 0, limit: 25, total: 0 };

  function renderKpis(stats) {
    const k = stats.kpi;
    const box = U.el("ds-kpis");
    const items = [
      [U.int(k.dramas), "部", "剧集样本"],
      [U.int(k.genres), "条", "题材标签记录"],
      [U.int(k.median_episodes), "集", "集数中位数"],
      [U.int(k.hot_threshold_wan), "万", "爆款热度阈值"],
      [U.num(k.hot_rate, 1) + "%", "", "爆款占比"],
      [U.int(stats.n_features), "维", "入模特征"]
    ];
    box.replaceChildren();
    items.forEach(function (it) {
      const card = U.h("div", "kpi");
      const v = U.h("div", "kpi__value");
      v.textContent = it[0];
      if (it[1]) v.appendChild(Object.assign(document.createElement("small"), { textContent: it[1] }));
      card.appendChild(v);
      card.appendChild(U.h("div", "kpi__label", it[2]));
      box.appendChild(card);
    });
  }

  function renderRows(page) {
    U.fill(U.el("tbl-dramas").querySelector("tbody"), page.items, function (d) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", null, d.drama_name));
      tr.appendChild(U.h("td", "muted", d.category));

      const ep = U.h("td", "is-num", d.episodes === null ? "—" : d.episodes);
      tr.appendChild(ep);

      tr.appendChild(U.h("td", "is-num", U.num(d.heat_wan, 0)));
      tr.appendChild(U.h("td", "is-num", d.rating === null ? "—" : U.num(d.rating, 2)));

      const rk = U.h("td", "is-num");
      if (d.rank_no !== null && d.rank_no !== undefined) {
        const b = U.h("span", "badge", "#" + d.rank_no);
        rk.appendChild(b);
      } else {
        rk.textContent = "—";
      }
      tr.appendChild(rk);

      const td = document.createElement("td");
      const dot = document.createElement("i");
      dot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;background:"
        + (Charts.TIER_COLOR[d.tier] || C.mid);
      td.appendChild(dot);
      td.appendChild(document.createTextNode(d.tier));
      tr.appendChild(td);

      const sum = U.h("td", "muted", d.summary_short || "—");
      sum.title = d.summary || "";
      tr.appendChild(sum);
      return tr;
    });

    const from = page.total ? page.offset + 1 : 0;
    const to = Math.min(page.offset + page.limit, page.total);
    U.el("page-info").textContent = "第 " + from + " – " + to + " 条，共 "
      + U.int(page.total) + " 条";
    U.el("ds-total").textContent = "共 " + U.int(page.total) + " 部";
    U.el("btn-prev").disabled = page.offset <= 0;
    U.el("btn-next").disabled = to >= page.total;
  }

  async function load() {
    const params = new URLSearchParams({
      limit: state.limit,
      offset: state.offset,
      keyword: U.el("q-keyword").value.trim(),
      sort: U.el("q-sort").value,
      order: U.el("q-order").value
    });
    try {
      const page = await API.get("/api/dramas?" + params.toString());
      state.total = page.total;
      renderRows(page);
    } catch (err) {
      U.showError(U.el("tbl-dramas").parentElement, err);
    }
  }

  (async function init() {
    try {
      renderKpis(await API.get("/api/dataset/stats"));
    } catch (err) {
      U.showError(U.el("ds-kpis"), err);
    }

    U.el("btn-query").addEventListener("click", function () {
      state.offset = 0;
      load();
    });
    U.el("q-keyword").addEventListener("keydown", function (e) {
      if (e.key === "Enter") { state.offset = 0; load(); }
    });
    U.el("q-sort").addEventListener("change", function () { state.offset = 0; load(); });
    U.el("q-order").addEventListener("change", function () { state.offset = 0; load(); });
    U.el("btn-prev").addEventListener("click", function () {
      state.offset = Math.max(0, state.offset - state.limit);
      load();
    });
    U.el("btn-next").addEventListener("click", function () {
      state.offset += state.limit;
      load();
    });

    load();
  })();
})();
