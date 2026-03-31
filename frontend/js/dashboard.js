/**
 * dashboard.js — AnomalyIQ Dashboard Section
 *
 * Top metric bar (4 stat cards) + model comparison table
 * + risk leaderboard + live detection feed.
 */

import { getMetrics, getLeaderboard, simulate } from "./api.js";
import { getSession } from "./session.js";
import { showToast } from "./app.js";
import { renderRocCurve, renderPrCurve, renderConfusionMatrix, renderFeatureImportance } from "./charts.js";

let _initialized = false;
let _currentView = "auditor"; // "auditor" | "technical"
const _liveItems = [];

// ── Init ──────────────────────────────────────────────────────────────────────

export async function init() {
  if (_initialized) return;
  _initialized = true;

  _setupViewToggle();
  await _load();
}

export async function refresh() {
  _initialized = false;
  await init();
}

// ── Load Data ─────────────────────────────────────────────────────────────────

async function _load() {
  const sessionId = getSession();
  if (!sessionId) return;

  try {
    const [metrics, leaderboard] = await Promise.all([
      getMetrics(sessionId),
      getLeaderboard(sessionId),
    ]);

    _renderStatCards(metrics);
    _renderModelTable(metrics);
    _renderLeaderboard(leaderboard.leaderboard || []);
    _renderTechnicalCharts(sessionId, metrics);

  } catch (err) {
    showToast("Dashboard error", err.message, "error");
  }
}

// ── Stat Cards ────────────────────────────────────────────────────────────────

function _renderStatCards(metrics) {
  const models = Object.values(metrics);

  // Best AUC
  const bestAuc = Math.max(...models.map(m => m.auc || 0));
  _setCard("stat-best-auc", (bestAuc * 100).toFixed(1) + "%", "Best AUC  across 4 models");

  // Fraud count (TP+FN across models — use the best model's CM)
  const sortedByAuc = models.slice().sort((a, b) => (b.auc || 0) - (a.auc || 0));
  const bestModel = sortedByAuc[0];
  if (bestModel?.confusion_matrix) {
    const { tp = 0, fn = 0, fp = 0, tn = 0 } = bestModel.confusion_matrix;
    _setCard("stat-fraud-count", tp + fn, "Fraud transactions detected");
    _setCard("stat-normal-count", tn + fp, "Normal transactions");
  }

  // Models trained
  _setCard("stat-models-count", models.length + " / 4", "Models trained");
}

function _setCard(id, value, subtitle) {
  const el = document.getElementById(id);
  if (!el) return;
  const val = el.querySelector(".stat-value");
  const sub = el.querySelector(".stat-sub");
  if (val) val.textContent = value;
  if (sub) sub.textContent = subtitle;
  el.classList.add("animate-in");
}

// ── Model Comparison Table ─────────────────────────────────────────────────────

const MODEL_NAME_MAP = {
  logistic_regression: "Logistic Regression",
  svm:                 "Support Vector Machine",
  decision_tree:       "Decision Tree",
  neural_network:      "Neural Network",
};

function _renderModelTable(metrics) {
  const tbody = document.getElementById("dashboard-model-tbody");
  if (!tbody) return;

  // Sort by AUC desc
  const sorted = Object.entries(metrics)
    .sort((a, b) => (b[1].auc || 0) - (a[1].auc || 0));

  const bestAuc = sorted[0]?.[1]?.auc || 0;

  tbody.innerHTML = sorted.map(([name, m], i) => `
    <tr class="${m.auc === bestAuc ? "best-row" : ""}">
      <td class="model-name">
        ${MODEL_NAME_MAP[name] || name}
        ${m.auc === bestAuc ? '<span class="badge badge-accent" style="margin-left:6px;font-size:9px">BEST</span>' : ""}
      </td>
      <td class="mono-val">${_pct(m.auc)}</td>
      <td class="mono-val">${_pct(m.accuracy)}</td>
      <td class="mono-val">${_pct(m.precision)}</td>
      <td class="mono-val">${_pct(m.recall)}</td>
      <td class="mono-val">${_pct(m.f1)}</td>
      <td class="mono-val">${m.train_time_ms ? m.train_time_ms.toFixed(0) + "ms" : "—"}</td>
    </tr>
  `).join("");
}

// ── Risk Leaderboard ──────────────────────────────────────────────────────────

function _renderLeaderboard(items) {
  const container = document.getElementById("dashboard-leaderboard");
  if (!container) return;

  if (!items.length) {
    container.innerHTML = `<div class="empty-state"><p>No data yet</p></div>`;
    return;
  }

  container.innerHTML = items.map((item, i) => {
    const score = item.anomaly_score || 0;
    const barWidth = (score * 100).toFixed(1);
    const rankNum = i + 1;
    const rankClass = rankNum <= 3 ? `rank-${rankNum}` : "";

    return `
      <div class="flex items-center gap-3" style="padding:8px 0;border-bottom:1px solid var(--border)">
        <div class="rank-badge ${rankClass}">${rankNum}</div>
        <div style="flex:1;min-width:0">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span class="mono" style="font-size:11px;color:var(--txt2)">TX #${item.index}</span>
            <span class="badge ${_verdictClass(item.verdict)}">${item.verdict?.replace(/_/g,' ') || '—'}</span>
          </div>
          <div style="height:5px;background:var(--border);border-radius:99px;overflow:hidden">
            <div class="leaderboard-score-bar" style="width:${barWidth}%"></div>
          </div>
        </div>
        <div class="mono" style="font-size:12px;color:var(--anomaly);min-width:40px;text-align:right">${(score * 100).toFixed(0)}%</div>
      </div>
    `;
  }).join("");
}

// ── Live Feed ─────────────────────────────────────────────────────────────────

export function addFeedItem(mlResult, trueLabel) {
  const feed = document.getElementById("dashboard-live-feed");
  if (!feed) return;

  const score = mlResult?.ensemble_score || 0;
  const isFraud = score >= 0.5;
  const verdict = mlResult?.verdict || "LIKELY_NORMAL";
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  const item = document.createElement("div");
  item.className = "feed-item feed-enter";
  item.innerHTML = `
    <div class="feed-dot ${isFraud ? "fraud" : "normal"}"></div>
    <span style="font-size:12px;color:var(--txt2);flex:1">${verdict.replace(/_/g, " ")}</span>
    <span class="badge ${_verdictClass(verdict)}" style="font-size:10px">${(score * 100).toFixed(0)}%</span>
    <span class="feed-time">${time}</span>
  `;

  // Insert at top
  feed.insertBefore(item, feed.firstChild);

  // Keep last 10 items
  while (feed.children.length > 10) {
    feed.removeChild(feed.lastChild);
  }
}

// ── Technical Charts ──────────────────────────────────────────────────────────

async function _renderTechnicalCharts(sessionId, metrics) {
  try {
    const { getRocCurve, getPrCurve, getFeatureImportance } = await import("./api.js");

    const [rocData, prData, fiData] = await Promise.all([
      getRocCurve(sessionId),
      getPrCurve(sessionId),
      getFeatureImportance(sessionId),
    ]);

    renderRocCurve("dashboard-roc-canvas", rocData);
    renderPrCurve("dashboard-pr-canvas", prData);

    const importances = fiData.feature_importances || fiData.coefficients || {};
    renderFeatureImportance("dashboard-fi-canvas", importances);

    // Confusion matrix for best model
    const bestModel = Object.entries(metrics).sort((a, b) => (b[1].auc || 0) - (a[1].auc || 0))[0];
    if (bestModel) {
      renderConfusionMatrix("dashboard-cm-container", bestModel[1].confusion_matrix);
    }
  } catch (_) {}
}

// ── View Toggle ───────────────────────────────────────────────────────────────

function _setupViewToggle() {
  const toggle = document.getElementById("dashboard-view-toggle");
  if (!toggle) return;

  toggle.addEventListener("click", e => {
    const btn = e.target.closest(".view-toggle-btn");
    if (!btn) return;

    _currentView = btn.dataset.view;
    toggle.querySelectorAll(".view-toggle-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.view === _currentView);
    });

    document.getElementById("dashboard-auditor-view")?.classList.toggle("hidden", _currentView !== "auditor");
    document.getElementById("dashboard-technical-view")?.classList.toggle("hidden", _currentView !== "technical");
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _pct(val) {
  if (val == null) return "—";
  return (val * 100).toFixed(1) + "%";
}

function _verdictClass(verdict) {
  const map = {
    HIGH_RISK: "badge-high",
    LIKELY_ANOMALY: "badge-warn",
    AMBIGUOUS: "badge-accent",
    LIKELY_NORMAL: "badge-normal",
  };
  return map[verdict] || "badge-muted";
}
