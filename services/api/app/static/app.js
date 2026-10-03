/* Digit Lab - vanilla JS front-end for the model service. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
// Escape any server-provided text before it goes into innerHTML.
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : res.statusText);
  return body;
}

/* ---------- 8x8 digit rendering ---------- */
function drawDigit(canvas, pixels) {
  canvas.width = 8; canvas.height = 8;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(8, 8);
  pixels.forEach((v, i) => {
    const g = 255 - Math.round((Math.max(0, Math.min(16, v)) / 16) * 255);
    img.data.set([g, g, g, 255], i * 4);
  });
  ctx.putImageData(img, 0, 0);
  canvas.classList.add("pixelated");
}

/* ---------- tabs ---------- */
$$(".tab[data-tab]").forEach((btn) => btn.addEventListener("click", () => {
  $$(".tab[data-tab]").forEach((b) => { b.classList.toggle("active", b === btn); b.setAttribute("aria-selected", b === btn); });
  $$(".panel").forEach((p) => { const on = p.id === `tab-${btn.dataset.tab}`; p.hidden = !on; p.classList.toggle("active", on); });
  if (btn.dataset.tab === "label") loadLabelBatch();
  if (btn.dataset.tab === "monitor") loadMonitoring();
}));

/* ---------- health ---------- */
async function refreshHealth() {
  const el = $("#status");
  try {
    const h = await api("/health");
    const v = h.model_versions || {};
    el.className = "status " + (h.status === "ok" ? "ok" : "bad");
    $("#status-text").textContent = h.status === "ok"
      ? `@${h.alias}: classifier v${v.supervised} · semi v${v.semi_supervised} · detector v${v.anomaly}`
      : "no models deployed yet - run the pipeline";
  } catch (e) {
    el.className = "status bad";
    $("#status-text").textContent = "service unreachable";
  }
}

/* ---------- drawing pad -> 8x8 (same preprocessing as the UCI digits) ---------- */
const pad = $("#pad");
const pctx = pad.getContext("2d");
let drawing = false, hasInk = false;
function resetPad() { pctx.clearRect(0, 0, pad.width, pad.height); hasInk = false; drawDigit($("#preview"), new Array(64).fill(0)); }
function pos(e) { const r = pad.getBoundingClientRect(); return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height]; }
pad.addEventListener("pointerdown", (e) => { drawing = true; pad.setPointerCapture(e.pointerId); const [x, y] = pos(e); pctx.beginPath(); pctx.moveTo(x, y); pctx.lineTo(x + 0.1, y + 0.1); stroke(); });
pad.addEventListener("pointermove", (e) => { if (!drawing) return; const [x, y] = pos(e); pctx.lineTo(x, y); stroke(); });
pad.addEventListener("pointerup", () => { drawing = false; const px = padTo8x8(); if (px) drawDigit($("#preview"), px); });
function stroke() { pctx.lineWidth = 22; pctx.lineCap = "round"; pctx.lineJoin = "round"; pctx.strokeStyle = "#111"; pctx.stroke(); hasInk = true; }

function padTo8x8() {
  if (!hasInk) return null;
  const { width: w, height: h } = pad;
  const data = pctx.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (data[(y * w + x) * 4 + 3] > 40) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  if (x1 < 0) return null;
  // Digits in the dataset span the full 32px height of their bitmap and are centred horizontally.
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const scale = Math.min(32 / bh, 32 / bw);
  const dw = Math.max(1, Math.round(bw * scale)), dh = Math.max(1, Math.round(bh * scale));
  const off = document.createElement("canvas"); off.width = 32; off.height = 32;
  const octx = off.getContext("2d");
  octx.drawImage(pad, x0, y0, bw, bh, Math.round((32 - dw) / 2), Math.round((32 - dh) / 2), dw, dh);
  const od = octx.getImageData(0, 0, 32, 32).data;
  const px = new Array(64).fill(0);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
    if (od[(y * 32 + x) * 4 + 3] > 100) px[Math.floor(y / 4) * 8 + Math.floor(x / 4)] += 1; // count "on" pixels per 4x4 block
  }
  return px;
}

$("#clear").addEventListener("click", resetPad);
$("#predict-draw").addEventListener("click", async () => {
  const px = padTo8x8();
  if (!px) { showError("Draw a digit first."); return; }
  await runPrediction(px, "canvas", null);
});
$$(".chip[data-kind]").forEach((chip) => chip.addEventListener("click", async () => {
  try {
    const s = await api(`/samples?kind=${chip.dataset.kind}`);
    resetPad();
    drawDigit($("#preview"), s.pixels);
    await runPrediction(s.pixels, `sample-${s.kind}`, s.kind === "clean" ? s.true_label : null, s.kind);
  } catch (e) { showError(e.message); }
}));

function renderBars(el, probs, top, alt) {
  el.className = "bars" + (alt ? " alt" : "");
  el.innerHTML = probs.map((p, d) =>
    `<div class="bar ${d === top ? "top" : ""}" title="digit ${d}: ${(100 * p).toFixed(1)}%"><i style="height:${Math.max(2, p * 100)}%"></i><span>${d}</span></div>`).join("");
}

function ordinal(n) {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function showError(msg) {
  $("#result-empty").hidden = false; $("#result").hidden = true;
  $("#result-empty").textContent = msg;
}

async function runPrediction(pixels, source, trueLabel, kind) {
  try {
    const r = await api("/predict", { method: "POST", body: JSON.stringify({ pixels, source }) });
    $("#result-empty").hidden = true; $("#result").hidden = false;
    const a = r.anomaly;
    const v = $("#verdict");
    v.className = "verdict " + (a.is_anomaly ? "anom" : "ok");
    const pct = Math.max(0, Math.min(100, a.density_percentile));
    v.innerHTML = `<span class="icon">${a.is_anomaly ? "⚠" : "✓"}</span><div style="flex:1">
      <strong>${a.is_anomaly ? "Unusual input - flagged by the GMM anomaly detector" : "Typical input"}</strong>
      <small>${esc(a.message)} Log-density ${a.log_density.toFixed(1)} vs threshold ${a.threshold.toFixed(1)}
      (${pct < 1 ? "below the 1st percentile" : `about the ${ordinal(Math.round(pct))} percentile`} of normal digits).</small>
      <div class="density" title="density percentile among normal digits"><i style="width:${pct}%"></i><b style="left:4%"></b></div></div>`;
    $("#sup-digit").textContent = r.supervised.digit;
    $("#sup-conf").textContent = `${(100 * r.supervised.confidence).toFixed(1)}% confident`;
    $("#semi-digit").textContent = r.semi_supervised.digit;
    $("#semi-conf").textContent = `${(100 * r.semi_supervised.confidence).toFixed(1)}% confident`;
    renderBars($("#sup-bars"), r.supervised.probabilities, r.supervised.digit, false);
    renderBars($("#semi-bars"), r.semi_supervised.probabilities, r.semi_supervised.digit, true);
    const truth = trueLabel === null || trueLabel === undefined ? "" : ` · true label: ${trueLabel}`;
    const what = kind && kind !== "clean" ? ` · corrupted test digit (${kind})` : "";
    $("#meta").textContent = `prediction #${r.prediction_id ?? "-"} · ${r.latency_ms} ms · versions ${JSON.stringify(r.model_versions)}${truth}${what}`;
  } catch (e) { showError(e.message); }
}

/* ---------- labelling ---------- */
let labelBatch = null;
async function loadLabelBatch() {
  const grid = $("#label-grid");
  try {
    labelBatch = await api("/labeling/batch");
  } catch (e) { grid.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  grid.innerHTML = "";
  labelBatch.items.forEach((it, i) => {
    const el = document.createElement("div");
    el.className = "label-item" + (it.human_label !== null ? " done" : "");
    el.innerHTML = `<canvas title="training image #${Number(it.train_index)}"></canvas>
      <input inputmode="numeric" maxlength="1" aria-label="label for image ${i + 1}" data-idx="${Number(it.train_index)}" value="${it.human_label ?? ""}">`;
    drawDigit($("canvas", el), it.pixels);
    const input = $("input", el);
    input.addEventListener("input", () => {
      input.value = input.value.replace(/[^0-9]/g, "").slice(0, 1);
      el.classList.toggle("done", input.value !== "");
      updateLabelProgress();
      if (input.value !== "") { const next = $$("#label-grid input")[i + 1]; if (next) next.focus(); }
    });
    grid.appendChild(el);
  });
  updateLabelProgress();
}
function updateLabelProgress() {
  const inputs = $$("#label-grid input");
  const done = inputs.filter((x) => x.value !== "").length;
  $("#label-count").textContent = `${done} / ${inputs.length} labelled`;
  $("#label-progress").style.width = `${inputs.length ? (100 * done) / inputs.length : 0}%`;
}
$("#save-labels").addEventListener("click", async () => {
  const labels = $$("#label-grid input").filter((x) => x.value !== "").map((x) => ({ train_index: Number(x.dataset.idx), label: Number(x.value) }));
  if (!labels.length) { $("#label-msg").textContent = "Type at least one label."; return; }
  const btn = $("#save-labels"); btn.disabled = true;
  try {
    const r = await api("/labeling/labels", { method: "POST", body: JSON.stringify({ labels, labeler: $("#labeler").value || "anonymous" }) });
    $("#label-msg").textContent = `Saved ${r.saved} labels (${r.labelled}/${r.total} done). Agreement with the dataset's labels: ${r.agreement_with_dataset_labels}.`;
  } catch (e) { $("#label-msg").textContent = e.message; }
  btn.disabled = false;
});

/* ---------- monitoring ---------- */
async function loadMonitoring() {
  let m;
  try { m = await api("/monitoring/summary?hours=168"); } catch (e) { $("#recent").innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  $("#mon-window").textContent = `· last ${Math.round(m.window_hours / 24)} days`;
  $("#k-n").textContent = m.n_predictions.toLocaleString();
  $("#k-anom").textContent = m.anomaly_rate_pct === null ? "-" : `${m.anomaly_rate_pct.toFixed(1)}%`;
  const sub = $("#k-anom-sub");
  const high = m.anomaly_rate_pct !== null && m.anomaly_rate_pct > 3 * m.expected_anomaly_rate_pct;
  sub.textContent = high ? `⚠ above 3x the expected ${m.expected_anomaly_rate_pct}%: inputs may have drifted` : `expected about ${m.expected_anomaly_rate_pct}% on normal digits`;
  sub.className = "kpi-sub" + (high ? " alert" : "");
  $("#k-agree").textContent = m.models_agree_pct === null ? "-" : `${m.models_agree_pct}%`;
  $("#k-lat").textContent = m.mean_latency_ms === null ? "-" : `${m.mean_latency_ms} ms`;

  const counts = Object.values(m.class_counts);
  const max = Math.max(1, ...counts);
  $("#class-chart").innerHTML = counts.map((c, d) =>
    `<div class="col" title="digit ${d}: ${c} predictions"><em>${c || ""}</em><i style="height:${(100 * c) / max}%"></i><span>${d}</span></div>`).join("");

  $("#source-table tbody").innerHTML = Object.entries(m.by_source).map(([src, s]) =>
    `<tr><td>${esc(src)}</td><td class="num">${s.n}</td><td class="num">${s.anomalies}</td><td class="num">${((100 * s.anomalies) / s.n).toFixed(0)}%</td></tr>`).join("")
    || `<tr><td colspan="4" class="muted">no requests yet</td></tr>`;

  const rec = $("#recent");
  rec.innerHTML = "";
  if (!m.recent.length) { rec.innerHTML = `<div class="empty">No predictions logged yet.</div>`; return; }
  m.recent.forEach((r) => {
    const el = document.createElement("div");
    el.className = "rec";
    el.innerHTML = `<canvas></canvas><div><b>${Number(r.supervised_digit)}</b> / ${Number(r.semi_supervised_digit)} ${r.is_anomaly ? '<span class="flag">⚠ flagged</span>' : ""}
      <small>${esc(r.source)} · #${Number(r.id)}</small><small>${esc((r.created_at || "").replace("T", " ").slice(5, 16))}</small></div>`;
    drawDigit($("canvas", el), r.pixels);
    rec.appendChild(el);
  });
}
$("#refresh-mon").addEventListener("click", loadMonitoring);

resetPad();
refreshHealth();
setInterval(refreshHealth, 15000);
