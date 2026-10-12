/* =====================================================================
   charts.js · 图表公共配置
   ---------------------------------------------------------------------
   配色不在此硬编码：每次 make() 前从 CSS 自定义属性解析当前主题
   的 token，保证图表随浅色/暗色切换自动换色。
   主题切换后广播的「themechange」事件会触发注册的重建回调。
   ===================================================================== */
(function (global) {
  "use strict";

  /* 兜底值 = 浅色主题（CSS 加载失败时图表仍可读） */
  const FALLBACK = {
    text: "#1e2430", text2: "#5a6472", text3: "#8b94a3",
    axis: "#cdd2da", split: "#eef0f3",
    accent: "#2f5fd0", accentSoft: "#9bb0e0",
    top: "#c0392b", high: "#d98324", mid: "#7b8494", low: "#2f7f6a",
    surface: "#ffffff"
  };

  const C = Object.assign({}, FALLBACK);

  /* 特征组配色：与 CSS 变量保持一致 */
  const GROUP_COLOR = {
    "内容属性": C.accent,
    "观众口碑": C.low,
    "评论互动": "#7a5cc4",
    "文本态度": C.high
  };

  /* 题材分档配色 */
  const TIER_COLOR = {
    "首选": C.top,
    "优先": C.high,
    "中性": C.mid,
    "规避": C.low
  };

  const AXIS = {
    axisLine: { lineStyle: { color: C.axis } },
    axisTick: { lineStyle: { color: C.axis } },
    axisLabel: { color: C.text2, fontSize: 11 },
    splitLine: { lineStyle: { color: C.split } }
  };

  const TOOLTIP = {
    backgroundColor: C.surface,
    borderColor: C.axis,
    borderWidth: 1,
    padding: [8, 10],
    textStyle: { color: C.text, fontSize: 12 },
    extraCssText: "box-shadow:0 2px 8px rgba(0,0,0,.18);border-radius:4px;"
  };

  function cssVar(name) {
    const v = getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
    return v || null;
  }

  /* 从主题 token 解析当前配色（每次建图前调用，属性原地更新，
     页面里 const {C, AXIS} = Charts 的解构引用不受影响） */
  function resolve() {
    C.text   = cssVar("--c-text")        || FALLBACK.text;
    C.text2  = cssVar("--c-text-2")      || FALLBACK.text2;
    C.text3  = cssVar("--c-text-3")      || FALLBACK.text3;
    C.axis   = cssVar("--c-border-strong") || FALLBACK.axis;
    C.split  = cssVar("--c-border")      || FALLBACK.split;
    C.accent = cssVar("--c-accent")      || FALLBACK.accent;
    C.accentSoft = cssVar("--c-accent-soft") || FALLBACK.accentSoft;
    C.top    = cssVar("--c-tier-top")    || FALLBACK.top;
    C.high   = cssVar("--c-tier-high")   || FALLBACK.high;
    C.mid    = cssVar("--c-tier-mid")    || FALLBACK.mid;
    C.low    = cssVar("--c-tier-low")    || FALLBACK.low;
    C.surface = cssVar("--c-surface")    || FALLBACK.surface;

    GROUP_COLOR["内容属性"] = C.accent;
    GROUP_COLOR["观众口碑"] = C.low;
    GROUP_COLOR["评论互动"] = cssVar("--c-g-engage") || "#7a5cc4";
    GROUP_COLOR["文本态度"] = C.high;

    TIER_COLOR["首选"] = C.top;
    TIER_COLOR["优先"] = C.high;
    TIER_COLOR["中性"] = C.mid;
    TIER_COLOR["规避"] = C.low;

    AXIS.axisLine.lineStyle.color = C.axis;
    AXIS.axisTick.lineStyle.color = C.axis;
    AXIS.axisLabel.color = C.text2;
    AXIS.splitLine.lineStyle.color = C.split;

    TOOLTIP.backgroundColor = C.surface;
    TOOLTIP.borderColor = C.axis;
    TOOLTIP.textStyle.color = C.text;
  }

  /* ---------------- 实例管理与主题重建 ---------------- */
  const instances = {};   // dom id -> chart
  const builders = [];    // themechange 重建回调（页面注册，闭包缓存数据）

  function onThemeChange(fn) { builders.push(fn); }

  global.addEventListener("themechange", function () {
    Object.keys(instances).forEach(function (id) {
      instances[id].dispose();
      delete instances[id];
    });
    builders.forEach(function (fn) {
      try { fn(); } catch (e) { /* 单个图表重建失败不影响其余 */ }
    });
  });

  /* ---------------- 初始化图表 ---------------- */
  function make(id, option) {
    const dom = document.getElementById(id);
    if (!dom || typeof echarts === "undefined") return null;
    resolve(); /* 主题可能已切换，建图前重取配色 */
    if (instances[id]) { instances[id].dispose(); }
    const chart = echarts.init(dom);
    option.tooltip = Object.assign({}, TOOLTIP, option.tooltip || {});
    option.textStyle = Object.assign({
      fontFamily: '"PingFang SC","Microsoft YaHei",system-ui,sans-serif'
    }, option.textStyle || {});
    chart.setOption(option);
    instances[id] = chart;
    return chart;
  }

  let resizeTimer = null;
  global.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      Object.keys(instances).forEach(function (id) { instances[id].resize(); });
    }, 120);
  });

  resolve(); /* 模块加载即解析一次，页面直接引用 C.XXX 也有正确值 */

  global.Charts = {
    make: make, C: C, AXIS: AXIS,
    GROUP_COLOR: GROUP_COLOR, TIER_COLOR: TIER_COLOR,
    onThemeChange: onThemeChange
  };
})(window);
