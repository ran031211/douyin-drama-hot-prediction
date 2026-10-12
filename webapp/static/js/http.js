/* =====================================================================
   http.js · 接口访问与错误处理
   ---------------------------------------------------------------------
   全站唯一的网络出口。后端错误契约：
       {"error": {"code": "...", "message": "...", "detail": {...}}}
   这里统一转成 Error（带 code / detail），调用方只处理成功分支。
   ===================================================================== */
(function (global) {
  "use strict";

  async function request(path, options) {
    const res = await fetch(path, Object.assign({
      headers: { "Content-Type": "application/json" }
    }, options || {}));

    let body = null;
    try { body = await res.json(); } catch (e) { body = null; }

    if (!res.ok) {
      const err = (body && body.error) || {};
      const message = err.message || ("请求失败（HTTP " + res.status + "）");
      const detail = err.detail && Object.keys(err.detail).length
        ? Object.entries(err.detail).map(([k, v]) => k + "：" + v).join("；")
        : "";
      const e = new Error(detail ? message + " —— " + detail : message);
      e.code = err.code || "http_error";
      e.status = res.status;
      throw e;
    }
    return body;
  }

  const API = {
    get: (path) => request(path),
    post: (path, data) => request(path, {
      method: "POST",
      body: JSON.stringify(data || {})
    })
  };

  /* ---------------- 通用小工具 ---------------- */
  const util = {
    num(v, digits) {
      if (v === null || v === undefined || v === "" || Number.isNaN(v)) return "—";
      return Number(v).toFixed(digits === undefined ? 0 : digits);
    },
    pct(v, digits) {
      if (v === null || v === undefined) return "—";
      return Number(v).toFixed(digits === undefined ? 1 : digits) + "%";
    },
    int(v) {
      if (v === null || v === undefined) return "—";
      return Number(v).toLocaleString("zh-CN");
    },
    el(id) { return document.getElementById(id); },
    /** 创建元素并填充，避免各页面重复 innerHTML 拼接 */
    h(tag, cls, text) {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text !== undefined && text !== null) node.textContent = text;
      return node;
    },
    fill(tbody, rows, render) {
      tbody.replaceChildren();
      if (!rows.length) {
        const tr = document.createElement("tr");
        const td = util.h("td", "muted", "无数据");
        td.colSpan = 12;
        tr.appendChild(td);
        tbody.appendChild(tr);
        return;
      }
      rows.forEach((row, i) => tbody.appendChild(render(row, i)));
    },
    /** 统一的错误提示位置：写进目标容器，不弹窗打断 */
    showError(container, err) {
      if (!container) return;
      container.replaceChildren();
      container.appendChild(util.h("div", "notice notice--error", err.message || String(err)));
    }
  };

  global.API = API;
  global.U = util;
})(window);
