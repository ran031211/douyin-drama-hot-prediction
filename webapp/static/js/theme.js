/* =====================================================================
   theme.js · 主题偏好（浅色 / 暗色 / 跟随系统）
   ---------------------------------------------------------------------
   - 初始渲染由 base.html 内联脚本完成（防 FOUC），本文件负责交互。
   - 手动选择写入 localStorage；「跟随系统」移除 data-theme 属性，
     由 CSS 的 prefers-color-scheme 媒体查询接管。
   - 切换后广播 window「themechange」，图表等 JS 配色据此重建。
   ===================================================================== */
(function () {
  "use strict";

  var KEY = "theme";
  var root = document.documentElement;
  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  var details = null, btn = null, radios = [];

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }

  /* 当前生效模式：URL 参数 > localStorage > 跟随系统 */
  function mode() {
    try {
      var q = new URLSearchParams(location.search).get("theme");
      if (q === "dark" || q === "light") return q;
    } catch (e) { /* ignore */ }
    var s = stored();
    return (s === "dark" || s === "light") ? s : "system";
  }

  /* 实际呈现的主题（system 时看系统偏好） */
  function effective() {
    var m = mode();
    if (m === "system") return mq.matches ? "dark" : "light";
    return m;
  }

  function apply(m, persist) {
    if (persist) {
      try {
        if (m === "system") localStorage.removeItem(KEY);
        else localStorage.setItem(KEY, m);
        /* 显式选择后清掉 URL 预览参数，否则 ?theme= 会一直覆盖用户选择，
           菜单回显和下次加载都会回到预览值 */
        var u = new URL(location.href);
        if (u.searchParams.has("theme")) {
          u.searchParams.delete("theme");
          history.replaceState(null, "", u);
        }
      } catch (e) { /* 隐私模式等场景：本次会话内仍生效 */ }
    }
    if (m === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", m);
    sync();
    window.dispatchEvent(new CustomEvent("themechange"));
  }

  function sync() {
    var m = mode();
    if (btn) btn.setAttribute("data-mode", effective());
    radios.forEach(function (r) { r.checked = (r.value === m); });
  }

  function init() {
    details = document.getElementById("theme-menu");
    btn = document.getElementById("theme-menu-btn");
    if (!details || !btn) return;
    radios = Array.prototype.slice.call(
      details.querySelectorAll('input[name="theme-pref"]'));

    radios.forEach(function (r) {
      r.addEventListener("change", function () {
        if (r.checked) {
          apply(r.value, true);
          details.removeAttribute("open"); /* 选中后收起，减少一步操作 */
        }
      });
    });

    /* Esc 关闭菜单（details 原生支持展开，收起需补） */
    details.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && details.open) {
        details.removeAttribute("open");
        btn.focus();
      }
    });

    /* 系统主题变化：仅当处于「跟随系统」时需要通知图表重建，
       手动模式下 CSS 属性不变、无需重建。 */
    var onMedia = function () { if (mode() === "system") { sync(); window.dispatchEvent(new CustomEvent("themechange")); } };
    if (mq.addEventListener) mq.addEventListener("change", onMedia);
    else mq.addListener(onMedia); /* 旧浏览器回退 */

    sync();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
