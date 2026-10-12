/* =====================================================================
   predict.js · 新剧实时预测
   ---------------------------------------------------------------------
   输入侧只收集「上线前可得」的信息：剧名、简介、题材、集数。
   爽点标签由服务端词典自动抽取（前端不维护会漂移的关键词表）。
   观众反馈为可选项，勾选后追加一次全特征模型计算做对比。
   ===================================================================== */
(function () {
  "use strict";
  const { C } = Charts;
  const { AXIS } = Charts;

  const state = { genres: [], picked: [], levelClass: {} };
  const LEVEL_CLASS = {
    "high": "verdict--high",
    "mid-high": "verdict--mid-high",
    "baseline": "verdict--baseline",
    "low": "verdict--low"
  };

  /* ---------------- 题材选择器 ---------------- */
  /* div 承载的 chip 补充 role/tabindex/键盘事件，保证纯键盘可操作 */
  function renderGenreChips(list) {
    const box = U.el("f-genres");
    box.replaceChildren();
    list.forEach(function (g) {
      const chip = U.h("div", "chip");
      chip.appendChild(document.createTextNode(g.genre));
      chip.appendChild(U.h("span", "chip__rate", U.num(g.hot_rate, 0) + "%"));
      chip.dataset.genre = g.genre;
      chip.setAttribute("role", "checkbox");
      chip.setAttribute("aria-checked", "false");
      chip.tabIndex = 0;
      chip.addEventListener("click", function () { toggleGenre(g.genre); });
      chip.addEventListener("keydown", function (e) {
        if (e.key === " " || e.key === "Enter") {
          e.preventDefault();
          toggleGenre(g.genre);
        }
      });
      box.appendChild(chip);
    });
  }

  function toggleGenre(name) {
    const i = state.picked.indexOf(name);
    if (i >= 0) {
      state.picked.splice(i, 1);
    } else {
      if (state.picked.length >= window.MAX_GENRES) state.picked.shift();
      state.picked.push(name);
    }
    syncChips();
  }

  function syncChips() {
    U.el("f-genres").querySelectorAll(".chip").forEach(function (c) {
      const on = state.picked.indexOf(c.dataset.genre) >= 0;
      c.classList.toggle("is-on", on);
      c.setAttribute("aria-checked", on ? "true" : "false");
    });
  }

  /* ---------------- 简介即时抽取 ---------------- */
  let extractTimer = null;
  function scheduleExtract() {
    clearTimeout(extractTimer);
    extractTimer = setTimeout(runExtract, 400);
  }

  async function runExtract() {
    const title = U.el("f-title").value.trim();
    const synopsis = U.el("f-synopsis").value.trim();
    const badge = U.el("auto-plots");
    if (!title && !synopsis) {
      badge.textContent = "尚未识别到爽点标签";
      return;
    }
    try {
      const res = await API.post("/api/extract", { title, synopsis });
      const tags = res.plot_tags || [];
      badge.textContent = tags.length
        ? "爽点标签：" + tags.join("、")
        : "尚未识别到爽点标签";
    } catch (e) {
      badge.textContent = "抽取失败";
    }
  }

  /* ---------------- 结果渲染 ---------------- */
  function probBlock(title, pack, primary) {
    const box = U.h("div", "prob" + (primary ? " prob--primary" : ""));
    box.appendChild(U.h("div", "prob__label", title));

    const val = U.h("div", "prob__value");
    val.textContent = U.num(pack.probability_pct, 1);
    const unit = document.createElement("span");
    unit.textContent = "%";
    val.appendChild(unit);
    box.appendChild(val);

    const meta = U.h("div", "prob__meta");
    const v = U.h("span", "verdict " + (LEVEL_CLASS[pack.level] || ""), pack.verdict);
    meta.appendChild(v);
    meta.appendChild(U.h("span", null, "基线 " + U.num(pack.baseline_pct, 0) + "% 的 "
      + U.num(pack.lift, 2) + " 倍"));
    meta.style.marginTop = "var(--sp-2)";
    box.appendChild(meta);

    const bar = U.h("div", "prob__bar");
    const fill = document.createElement("i");
    fill.style.width = Math.min(pack.probability_pct, 100) + "%";
    bar.appendChild(fill);
    box.appendChild(bar);

    const foot = U.h("div", "prob__meta");
    foot.style.marginTop = "var(--sp-2)";
    foot.appendChild(U.h("span", null, pack.label));
    foot.appendChild(U.h("span", null, "AUC " + U.num(pack.auc, 4)));
    box.appendChild(foot);
    return box;
  }

  function renderResult(res) {
    const box = U.el("result-body");
    box.replaceChildren();

    const grid = U.h("div", "result-grid");
    grid.appendChild(probBlock("冷启动版爆款概率", res.cold, true));
    if (res.full) {
      const g = U.h("div", "prob");
      g.appendChild(U.h("div", "prob__label", "全特征版爆款概率"));
      const val = U.h("div", "prob__value");
      val.textContent = U.num(res.full.probability_pct, 1);
      val.appendChild(Object.assign(document.createElement("span"), { textContent: "%" }));
      g.appendChild(val);
      const meta = U.h("div", "prob__meta");
      meta.style.marginTop = "var(--sp-2)";
      meta.appendChild(U.h("span", "verdict " + (LEVEL_CLASS[res.full.level] || ""), res.full.verdict));
      meta.appendChild(U.h("span", null, "较冷启动 " + (res.delta.probability_gain_pp >= 0 ? "+" : "")
        + U.num(res.delta.probability_gain_pp, 1) + " pp"));
      g.appendChild(meta);
      grid.appendChild(g);
    }
    box.appendChild(grid);

    const note = U.h("div", "notice");
    note.style.marginTop = "var(--sp-4)";
    note.textContent = res.cold.advice
      + (res.full ? "" : "　未提供观众反馈，全特征模型未参与计算。");
    box.appendChild(note);

    U.el("result-note").textContent = res.full ? "双模型已计算" : "冷启动模型";
    renderFactors(res.factors || []);
    renderPrior(res.genre_prior || []);
  }

  function renderFactors(list) {
    const card = U.el("card-factors");
    const box = U.el("factors");
    box.replaceChildren();
    if (!list.length) { card.hidden = true; return; }
    card.hidden = false;

    const max = Math.max.apply(null, list.map(function (f) { return Math.abs(f.contribution); }));
    list.forEach(function (f) {
      const row = U.h("div", "factor");
      const dot = U.h("span", "factor__dot");
      dot.style.background = Charts.GROUP_COLOR[f.group] || C.mid;
      row.appendChild(dot);
      row.appendChild(U.h("span", "factor__name", f.label));

      const track = U.h("span", "factor__track");
      const bar = document.createElement("i");
      const w = Math.abs(f.contribution) / max * 50;
      bar.style.width = w + "%";
      bar.style.background = f.contribution > 0 ? C.top : C.low;
      if (f.contribution > 0) bar.style.left = "50%";
      else bar.style.right = "50%";
      track.appendChild(bar);
      row.appendChild(track);

      const sign = f.contribution > 0 ? "+" : "";
      row.appendChild(U.h("span", "factor__value", sign + U.num(f.contribution, 3)));
      box.appendChild(row);
    });
  }

  function renderPrior(list) {
    const card = U.el("card-prior");
    if (!list.length) { card.hidden = true; return; }
    card.hidden = false;
    U.fill(U.el("prior-body"), list, function (d) {
      const tr = document.createElement("tr");
      tr.appendChild(U.h("td", null, d.genre));
      tr.appendChild(U.h("td", "is-num", d.n));
      tr.appendChild(U.h("td", "is-num", U.num(d.hot_rate, 1) + "%"));
      tr.appendChild(U.h("td", "is-num", U.num(d.ci_low, 1) + "% ~ " + U.num(d.ci_high, 1) + "%"));
      const td = document.createElement("td");
      const dot = document.createElement("i");
      dot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;background:"
        + (Charts.TIER_COLOR[d.tier] || C.mid);
      td.appendChild(dot);
      td.appendChild(document.createTextNode(d.tier));
      tr.appendChild(td);
      return tr;
    });
  }

  /* ---------------- 提交 ---------------- */
  function collect() {
    const body = {
      title: U.el("f-title").value.trim(),
      synopsis: U.el("f-synopsis").value.trim(),
      genres: state.picked.slice(),
      episodes: Number(U.el("f-episodes").value) || 0
    };
    if (U.el("e-enable").checked) {
      body.hot_rating_mean = Number(U.el("e-rating").value);
      body.hot_digg_mean = Number(U.el("e-hotdigg").value);
      body.ep1_digg_mean = Number(U.el("e-ep1digg").value);
      body.ep1_pos_ratio = Number(U.el("e-pos").value);
      body.ep1_urge_ratio = Number(U.el("e-urge").value);
    }
    return body;
  }

  async function submit() {
    const btn = U.el("btn-predict");
    btn.disabled = true;
    btn.classList.add("is-loading");
    btn.textContent = "计算中…";
    try {
      const res = await API.post("/api/predict", collect());
      renderResult(res);
    } catch (err) {
      U.showError(U.el("result-body"), err);
      U.el("card-factors").hidden = true;
      U.el("card-prior").hidden = true;
    } finally {
      btn.disabled = false;
      btn.classList.remove("is-loading");
      btn.textContent = "开始预测";
    }
  }

  /* ---------------- 滑块 ---------------- */
  const SLIDER_IDS = ["e-rating", "e-hotdigg", "e-ep1digg", "e-pos", "e-urge"];

  /* 未勾选「一并计算全特征模型」时把滑块禁用：可拖但不生效会造成误解。 */
  const SliderRows = {
    enabled: function (on) {
      SLIDER_IDS.forEach(function (id) {
        U.el(id).disabled = !on;
        U.el(id + "-v").classList.toggle("is-off", !on);
      });
    }
  };

  function bindSlider(id, fmt) {
    const input = U.el(id);
    const label = U.el(id + "-v");
    const sync = function () { label.textContent = fmt(Number(input.value)); };
    input.addEventListener("input", sync);
    sync();
  }

  /* ---------------- 示例 ---------------- */
  const EXAMPLE = {
    title: "重生八零：我带全家奔小康",
    synopsis: "林晓意外重生回到1980年，靠着一手金手指带着一家人从农村走进城里，"
      + "在打脸仇人的同时，也重新赢回了亲情。",
    genres: ["重生", "年代", "乡村"],
    episodes: 120
  };

  function fillExample() {
    U.el("f-title").value = EXAMPLE.title;
    U.el("f-synopsis").value = EXAMPLE.synopsis;
    U.el("f-episodes").value = EXAMPLE.episodes;
    state.picked = EXAMPLE.genres.slice();
    syncChips();
    updateSynopsisLen();
    syncWaybackCard();
    runExtract();
  }

  function clearAll() {
    U.el("f-title").value = "";
    U.el("f-synopsis").value = "";
    U.el("f-episodes").value = 100;
    state.picked = [];
    syncChips();
    updateSynopsisLen();
    U.el("auto-plots").textContent = "尚未识别到爽点标签";
    U.el("result-body").replaceChildren(
      U.h("div", "empty", "填写左侧信息后点击「开始预测」"));
    U.el("card-factors").hidden = true;
    U.el("card-prior").hidden = true;
    U.el("card-wayback").hidden = true;
    U.el("e-enable").checked = false;
    SliderRows.enabled(false);
  }

  function updateSynopsisLen() {
    U.el("f-synopsis-len").textContent = U.el("f-synopsis").value.length;
  }

  /* 基本信息填全后才展开「追加观众反馈」卡片：空表单时不占版面。
     该卡片是通往全特征模型的唯一入口，缺了它整条链路在界面上不可达。 */
  function syncWaybackCard() {
    const ready = U.el("f-title").value.trim() !== ""
      && U.el("f-synopsis").value.trim() !== "";
    U.el("card-wayback").hidden = !ready;
  }

  /* ---------------- 启动 ---------------- */
  (async function init() {
    U.el("f-synopsis").addEventListener("input", function () {
      updateSynopsisLen();
      scheduleExtract();
      syncWaybackCard();
    });
    U.el("f-title").addEventListener("input", function () {
      scheduleExtract();
      syncWaybackCard();
    });
    U.el("e-enable").addEventListener("change", function () {
      // 未勾选时滑块置灰，避免「调了却没用」的歧义
      SliderRows.enabled(this.checked);
    });
    U.el("btn-predict").addEventListener("click", submit);
    U.el("btn-example").addEventListener("click", fillExample);
    U.el("btn-clear").addEventListener("click", clearAll);
    bindSlider("e-rating", function (v) { return v.toFixed(1); });
    bindSlider("e-hotdigg", function (v) { return v.toFixed(0); });
    bindSlider("e-ep1digg", function (v) { return v.toFixed(0); });
    bindSlider("e-pos", function (v) { return (v * 100).toFixed(0) + "%"; });
    bindSlider("e-urge", function (v) { return (v * 100).toFixed(0) + "%"; });

    SliderRows.enabled(U.el("e-enable").checked);
    syncWaybackCard();

    try {
      const meta = await API.get("/api/meta");
      state.genres = meta.genres;
      window.MAX_GENRES = meta.app.max_genres;
      renderGenreChips(meta.genres);
    } catch (err) {
      U.showError(U.el("f-genres"), err);
    }
  })();
})();
