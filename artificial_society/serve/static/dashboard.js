// Dashboard client: polls the API ~1s and renders cards + tiny canvas charts.
// No third-party libraries — the line chart is a self-contained canvas helper,
// so the dashboard works fully offline.
"use strict";

const POLL_MS = 1000;

// Metric cards shown from /api/status -> stats. [key, label, formatter]
const CARDS = [
  ["tick", "tick", (v) => v],
  ["population", "population", (v) => v],
  ["avg_energy", "avg energy", fmt1],
  ["tribes", "tribes", (v) => v],
  ["technologies", "technologies", (v) => v],
  ["knowledge", "knowledge sites", (v) => v],
  ["cooperation", "cooperation", fmt2],
  ["avg_sick", "avg sick", fmt2],
  ["avg_reward", "avg reward", fmt2],
  ["n_child", "children", (v) => v],
  ["n_adult", "adults", (v) => v],
  ["n_elder", "elders", (v) => v],
];

// History series to chart: [historyKey, label, color]
const SERIES = [
  ["population_history", "population", "#5b9dff"],
  ["food_history", "world food", "#46c46e"],
  ["energy_history", "avg energy", "#e0a13f"],
  ["knowledge_history", "knowledge", "#b07bff"],
  ["cooperation_history", "cooperation", "#3fd0e0"],
];

function fmt1(v) { return Number(v).toFixed(1); }
function fmt2(v) { return Number(v).toFixed(2); }

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + " -> " + r.status);
  return r.json();
}

// --- tiny canvas line chart -------------------------------------------------
function drawChart(canvas, points, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  if (!points || points.length < 2) return;

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const pad = 6;
  const sx = (x) => pad + ((x - xMin) / (xMax - xMin || 1)) * (w - 2 * pad);
  const sy = (y) => h - pad - ((y - yMin) / (yMax - yMin || 1)) * (h - 2 * pad);

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((p, i) => {
    const X = sx(p[0]), Y = sy(p[1]);
    i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
  });
  ctx.stroke();
}

// --- rendering --------------------------------------------------------------
function renderCards(stats) {
  const el = document.getElementById("cards");
  el.innerHTML = CARDS.map(([k, label, f]) => {
    const v = stats && stats[k] != null ? f(stats[k]) : "–";
    return `<div class="card"><div class="k">${label}</div><div class="v">${v}</div></div>`;
  }).join("");
}

function ensureChartBoxes() {
  const wrap = document.getElementById("charts");
  if (wrap.childElementCount) return;
  wrap.innerHTML = SERIES.map(([key, label]) =>
    `<div class="chart-box"><div class="title">${label}</div>` +
    `<canvas id="c-${key}"></canvas></div>`
  ).join("");
}

function renderCharts(history) {
  ensureChartBoxes();
  SERIES.forEach(([key, , color]) => {
    drawChart(document.getElementById("c-" + key), history[key], color);
  });
}

function renderDevice(device) {
  const el = document.getElementById("device");
  const type = (device && device.type) || "unknown";
  el.className = "badge badge-" + type;
  el.textContent = "device: " + type + " — " + ((device && device.name) || "?");
}

function renderStatus(snap) {
  const s = document.getElementById("status-line");
  s.textContent = "status: " + snap.status + (snap.error ? " — see console" : "");
  if (snap.error) console.error(snap.error);
  document.getElementById("start-btn").disabled = snap.status === "running";
  document.getElementById("stop-btn").disabled = snap.status !== "running";
}

// --- polling loop -----------------------------------------------------------
async function tick() {
  try {
    const snap = await getJSON("/api/status");
    renderDevice(snap.device);
    renderStatus(snap);
    renderCards(snap.stats);
    const hist = await getJSON("/api/history");
    renderCharts(hist);
    if ((hist.population_history || []).length) {
      document.getElementById("ecology").src = "/api/graph.png?t=" + Date.now();
    }
  } catch (e) {
    console.error(e);
  }
}

// --- controls ---------------------------------------------------------------
document.getElementById("run-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const data = new FormData(ev.target);
  const body = {};
  for (const k of ["seed", "ticks", "grid_w", "grid_h", "pop"]) {
    const v = data.get(k);
    body[k] = v === "" || v == null ? null : Number(v);
  }
  const r = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (r.status === 409) alert("A run is already active — stop it first.");
  tick();
});

document.getElementById("stop-btn").addEventListener("click", async () => {
  await fetch("/api/stop", { method: "POST" });
  tick();
});

tick();
setInterval(tick, POLL_MS);
