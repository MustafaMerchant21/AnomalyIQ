/**
 * simulator.js — AnomalyIQ Sample Simulation
 *
 * Random draw button, auto-run toggle (every 2s),
 * transaction card with feature list + 4 model probability bars,
 * history strip of last 20 results + rolling accuracy tracker.
 */

import { simulate } from "./api.js";
import { getSession } from "./session.js";
import { showToast } from "./app.js";
import { renderProbabilityBars } from "./charts.js";
import { addFeedItem } from "./dashboard.js";

let _initialized = false;
let _autoRunTimer = null;
let _autoRunActive = false;
let _history = [];   // {fraud: bool, score: number, correct: bool}
let _totalRuns = 0;
let _correctPreds = 0;

// ── Init ──────────────────────────────────────────────────────────────────────

export function init() {
  if (_initialized) return;
  _initialized = true;

  document.getElementById("sim-draw-btn")?.addEventListener("click", _drawSample);
  document.getElementById("sim-autorun-btn")?.addEventListener("click", _toggleAutoRun);
}

export function reset() {
  _initialized = false;
  _history = [];
  _totalRuns = 0;
  _correctPreds = 0;
  if (_autoRunTimer) {
    clearInterval(_autoRunTimer);
    _autoRunTimer = null;
    _autoRunActive = false;
  }
}

// ── Draw Sample ───────────────────────────────────────────────────────────────

async function _drawSample() {
  const sessionId = getSession();
  if (!sessionId) {
    showToast("No session", "Train a model first.", "error");
    return;
  }

  _setLoading(true);

  try {
    const result = await simulate(sessionId);
    _renderTransactionCard(result);
    _updateHistory(result);
    addFeedItem(result.ml, result.true_label);
  } catch (err) {
    showToast("Simulation failed", err.message, "error");
  } finally {
    _setLoading(false);
  }
}

// ── Auto-run Toggle ───────────────────────────────────────────────────────────

function _toggleAutoRun() {
  const btn = document.getElementById("sim-autorun-btn");
  _autoRunActive = !_autoRunActive;

  if (_autoRunActive) {
    btn.classList.add("active");
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
      </svg>
      Stop Auto-run
    `;
    _autoRunTimer = setInterval(_drawSample, 2000);
    _drawSample(); // Immediate first draw
  } else {
    btn.classList.remove("active");
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="6,3 20,12 6,21"/>
      </svg>
      Auto-run (every 2s)
    `;
    if (_autoRunTimer) {
      clearInterval(_autoRunTimer);
      _autoRunTimer = null;
    }
  }
}

// ── Transaction Card ──────────────────────────────────────────────────────────

function _renderTransactionCard(result) {
  const { features, true_label, ml } = result;
  const verdict = ml?.verdict || "LIKELY_NORMAL";
  const ensembleScore = ml?.ensemble_score || 0;

  // True label badge
  const isFraud = int(true_label) === 1;
  const trueLabelBadge = isFraud
    ? `<span class="badge badge-high">Actual: FRAUD</span>`
    : `<span class="badge badge-normal">Actual: NORMAL</span>`;

  // Feature table
  const featureRows = Object.entries(features || {})
    .filter(([k]) => !k.startsWith("_"))
    .map(([k, v]) => `
      <tr>
        <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--accent);padding:5px 10px">${k}</td>
        <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--txt);padding:5px 10px;text-align:right">${typeof v === "number" ? v.toFixed(4) : v}</td>
      </tr>
    `).join("");

  const card = document.getElementById("sim-transaction-card");
  if (!card) return;

  card.innerHTML = `
    <div class="card animate-in">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--sp-4)">
        <div>
          <h4 style="color:var(--txt);margin-bottom:4px">Transaction Sample</h4>
          <div style="display:flex;gap:8px;align-items:center">
            ${trueLabelBadge}
            <span class="badge ${_verdictClass(verdict)}">${verdict.replace(/_/g, " ")}</span>
          </div>
        </div>
        <div style="text-align:right">
          <div class="mono" style="font-size:22px;color:${ensembleScore >= 0.5 ? "var(--anomaly)" : "var(--normal)"}">${(ensembleScore * 100).toFixed(1)}%</div>
          <div class="caption">Ensemble Score</div>
        </div>
      </div>

      <div class="table-wrapper" style="margin-bottom:var(--sp-5)">
        <table><tbody>${featureRows}</tbody></table>
      </div>

      <div style="margin-bottom:var(--sp-2)">
        <div class="section-title">Model Probabilities</div>
        <div id="sim-prob-bars"></div>
      </div>

      <div style="display:flex;align-items:center;gap:8px;margin-top:var(--sp-3)">
        <div class="mono" style="font-size:12px;color:var(--txt3)">Flagged by ${ml?.flags || 0}/4 models</div>
        <div style="flex:1;height:1px;background:var(--border)"></div>
        <div class="mono" style="font-size:11px;color:var(--txt3)">Disagreement: ${ml?.disagreement_index?.toFixed(1) || "0.0"}</div>
      </div>
    </div>
  `;

  // Render probability bars
  if (ml?.probabilities) {
    renderProbabilityBars("sim-prob-bars", ml.probabilities);
  }
}

function int(val) {
  return parseInt(val, 10);
}

// ── History Strip ─────────────────────────────────────────────────────────────

function _updateHistory(result) {
  const ml = result.ml || {};
  const trueLabel = int(result.true_label || 0);
  const predicted = ml?.ensemble_score >= 0.5 ? 1 : 0;
  const correct = trueLabel === predicted;
  const score = ml?.ensemble_score || 0;
  const isFraud = trueLabel === 1;

  _history.unshift({ isFraud, score, correct });
  if (_history.length > 20) _history = _history.slice(0, 20);

  _totalRuns++;
  if (correct) _correctPreds++;

  _renderHistoryStrip();
  _renderAccuracy();
}

function _renderHistoryStrip() {
  const strip = document.getElementById("sim-history-strip");
  if (!strip) return;

  strip.innerHTML = _history.map((h, i) => `
    <div
      class="history-circle ${h.isFraud ? "fraud" : "normal"}"
      title="${h.isFraud ? "FRAUD" : "NORMAL"} — Score: ${(h.score * 100).toFixed(0)}% ${h.correct ? "✓" : "✗"}"
      style="${!h.correct ? "opacity:0.5;outline:2px solid var(--warn)" : ""}"
    ></div>
  `).join("");
}

function _renderAccuracy() {
  const el = document.getElementById("sim-accuracy");
  if (!el) return;

  const pct = _totalRuns > 0 ? (_correctPreds / _totalRuns * 100).toFixed(1) : "—";
  el.innerHTML = `
    <span style="font-family:'JetBrains Mono',monospace;font-size:16px;color:var(--ok)">${pct}%</span>
    <span style="font-size:11px;color:var(--txt3);margin-left:6px">rolling accuracy (${_totalRuns} runs)</span>
  `;
}

// ── Loading State ─────────────────────────────────────────────────────────────

function _setLoading(loading) {
  const btn = document.getElementById("sim-draw-btn");
  if (!btn) return;

  if (loading) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;display:inline-block;margin-right:6px"></span>Drawing...`;
  } else {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
      </svg>
      Draw Random Transaction
    `;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _verdictClass(verdict) {
  const map = {
    HIGH_RISK: "badge-high",
    LIKELY_ANOMALY: "badge-warn",
    AMBIGUOUS: "badge-accent",
    LIKELY_NORMAL: "badge-normal",
  };
  return map[verdict] || "badge-muted";
}
