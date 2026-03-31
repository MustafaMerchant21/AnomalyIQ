/**
 * predictor.js — AnomalyIQ Custom Predictor Section
 *
 * Feature sliders + risk meter SVG arc speedometer + verdict panel
 * + score breakdown bars + model agreement + reason codes + LLM narrative
 * + What-if hint
 */

import { getColumnConfig, getRiskScore, explain } from "./api.js";
import { getSession } from "./session.js";
import { showToast } from "./app.js";
import { renderRocCurve, renderPrCurve, renderConfusionMatrix, renderFeatureImportance } from "./charts.js";

let _initialized = false;
let _colConfig = null;
let _currentFeatures = {};
let _lastResult = null;
let _currentView = "auditor";

// ── Init ──────────────────────────────────────────────────────────────────────

export async function init() {
  if (_initialized) return;
  _initialized = true;

  _setupViewToggle();

  const sessionId = getSession();
  if (!sessionId) return;

  try {
    _colConfig = await getColumnConfig(sessionId);
    _buildSliders(_colConfig);
    _setupAnalyzeButton();
    _setupPresetButtons();
  } catch (err) {
    showToast("Predictor error", err.message, "error");
  }
}

export async function refresh() {
  _initialized = false;
  _colConfig = null;
  await init();
}

// ── Slider Builder ────────────────────────────────────────────────────────────

function _buildSliders(cfg) {
  const container = document.getElementById("predictor-sliders");
  if (!container) return;

  const colStats = cfg.col_stats || {};
  const featureCols = cfg.feature_cols || [];

  container.innerHTML = featureCols.map(feat => {
    const stats = colStats[feat] || { min: 0, max: 1, mean: 0.5, std: 0.2 };
    const min = parseFloat(stats.min.toFixed(4));
    const max = parseFloat(stats.max.toFixed(4));
    const defaultVal = parseFloat(stats.mean.toFixed(4));
    const step = max - min > 100 ? 1 : max - min > 10 ? 0.1 : 0.01;

    _currentFeatures[feat] = defaultVal;

    return `
      <div class="slider-row" id="slider-row-${_safeId(feat)}">
        <div class="slider-header">
          <span class="slider-label">${_humanize(feat)}</span>
          <span class="slider-value mono" id="slider-val-${_safeId(feat)}">${defaultVal}</span>
        </div>
        <input
          type="range"
          id="slider-${_safeId(feat)}"
          data-feature="${feat}"
          min="${min}"
          max="${max}"
          step="${step}"
          value="${defaultVal}"
          style="width:100%"
        />
        <div style="display:flex;justify-content:space-between;margin-top:2px">
          <span style="font-size:10px;color:var(--txt3)">${min}</span>
          <span style="font-size:10px;color:var(--txt3)">${max}</span>
        </div>
      </div>
    `;
  }).join("");

  // Attach listeners
  container.querySelectorAll("input[type=range]").forEach(slider => {
    slider.addEventListener("input", () => {
      const feat = slider.dataset.feature;
      const val = parseFloat(slider.value);
      _currentFeatures[feat] = val;

      const display = document.getElementById(`slider-val-${_safeId(feat)}`);
      if (display) display.textContent = val;
    });
  });
}

// ── Preset Buttons ─────────────────────────────────────────────────────────────

function _setupPresetButtons() {
  document.getElementById("preset-fraud")?.addEventListener("click", () => _applyPreset("fraud"));
  document.getElementById("preset-normal")?.addEventListener("click", () => _applyPreset("normal"));
  document.getElementById("preset-mean")?.addEventListener("click", () => _applyPreset("mean"));
}

function _applyPreset(type) {
  if (!_colConfig) return;

  const classMeans = _colConfig.class_means || {};
  const colStats = _colConfig.col_stats || {};

  let values;
  if (type === "fraud") {
    values = classMeans["1"] || {};
  } else if (type === "normal") {
    values = classMeans["0"] || {};
  } else {
    // Mean from col_stats
    values = Object.fromEntries(
      Object.entries(colStats).map(([feat, stats]) => [feat, stats.mean])
    );
  }

  for (const feat of Object.keys(_currentFeatures)) {
    if (feat in values) {
      const val = parseFloat(values[feat]);
      _currentFeatures[feat] = val;

      const slider = document.getElementById(`slider-${_safeId(feat)}`);
      if (slider) slider.value = val;

      const display = document.getElementById(`slider-val-${_safeId(feat)}`);
      if (display) display.textContent = val.toFixed(4);
    }
  }
}

// ── Analyze Button ─────────────────────────────────────────────────────────────

function _setupAnalyzeButton() {
  const btn = document.getElementById("predictor-analyze-btn");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    // Validate features
    const features = { ..._currentFeatures };
    const hasNaN = Object.values(features).some(v => isNaN(v) || v === null || v === undefined);
    if (hasNaN) {
      showToast("Invalid input", "All feature values must be valid numbers.", "error");
      return;
    }

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;display:inline-block;margin-right:6px"></span>Analyzing...`;

    try {
      const sessionId = getSession();
      const result = await getRiskScore(sessionId, features);
      _lastResult = result;
      _renderVerdict(result);

      // Get explanation
      const explainResult = await explain(
        sessionId,
        features,
        result.ml,
        result.rules,
        result.graph,
      );
      _renderExplanation(explainResult);

      // What-if hint
      _renderWhatIf(features, result);

    } catch (err) {
      showToast("Analysis failed", err.message, "error");
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> Analyze Transaction →`;
    }
  });
}

// ── Verdict Panel ─────────────────────────────────────────────────────────────

function _renderVerdict(result) {
  const { risk, ml, rules, graph } = result;
  const score = risk?.final_score || 0;
  const scorePct = risk?.final_score_pct || 0;
  const band = risk?.risk_band || "LOW";
  const verdict = ml?.verdict || "LIKELY_NORMAL";
  const confidence = risk?.confidence || "MEDIUM";

  // Verdict badge
  const verdictEl = document.getElementById("predictor-verdict-badge");
  if (verdictEl) {
    verdictEl.textContent = verdict.replace(/_/g, " ");
    verdictEl.className = `badge ${_verdictClass(verdict)} verdict-badge-appear`;

    // Re-trigger animation
    verdictEl.style.animation = "none";
    verdictEl.offsetHeight; // reflow
    verdictEl.style.animation = "";
  }

  // Risk meter
  _animateRiskMeter(scorePct, band);

  // Score breakdown bars
  _renderScoreBars(
    ml?.ensemble_score || 0,
    rules?.rule_score || 0,
    graph?.graph_score || 0,
  );

  // Model agreement dots
  const flags = ml?.flags || 0;
  _renderModelDots(flags, ml?.probabilities || {});

  // Confidence badge
  const confEl = document.getElementById("predictor-confidence");
  if (confEl) {
    confEl.textContent = confidence + " CONFIDENCE";
    confEl.className = `badge ${confidence === "HIGH" ? "badge-ok" : confidence === "MEDIUM" ? "badge-warn" : "badge-accent"}`;
  }

  // Triggered rules
  _renderTriggeredRules(rules?.triggered_rules || []);

  // Show result panel
  document.getElementById("predictor-result-panel")?.classList.remove("hidden");
}

function _animateRiskMeter(scorePct, band) {
  const svg = document.getElementById("predictor-risk-svg");
  const scoreDisplay = document.getElementById("predictor-risk-score");
  const bandDisplay = document.getElementById("predictor-risk-band");

  if (scoreDisplay) scoreDisplay.textContent = Math.round(scorePct);
  if (bandDisplay) {
    bandDisplay.textContent = band + " RISK";
    bandDisplay.style.color = band === "HIGH" ? "var(--anomaly)" : band === "MEDIUM" ? "var(--warn)" : "var(--ok)";
  }

  // Animate needle via inline SVG
  if (svg) {
    const needle = svg.querySelector(".risk-needle");
    if (needle) {
      // Arc spans -120° to +120° (240° total)
      const targetDeg = -120 + (scorePct / 100) * 240;
      needle.style.transform = `rotate(${targetDeg}deg)`;
    }
  }
}

function _renderScoreBars(mlScore, ruleScore, graphScore) {
  const bars = [
    { id: "score-bar-ml",    val: mlScore,    cls: "ml" },
    { id: "score-bar-rule",  val: ruleScore,  cls: "rule" },
    { id: "score-bar-graph", val: graphScore, cls: "graph" },
  ];

  for (const { id, val, cls } of bars) {
    const fill = document.getElementById(id);
    if (!fill) continue;
    fill.className = `score-bar-fill ${cls}`;
    fill.style.width = (val * 100).toFixed(1) + "%";

    // Update value label
    const valEl = document.getElementById(id + "-val");
    if (valEl) valEl.textContent = (val * 100).toFixed(0) + "%";
  }
}

function _renderModelDots(flags, probabilities) {
  const dotsEl = document.getElementById("predictor-model-dots");
  const agreementEl = document.getElementById("predictor-model-agreement");

  if (!dotsEl) return;

  const models = ["logistic_regression", "svm", "decision_tree", "neural_network"];
  dotsEl.innerHTML = models.map(name => {
    const prob = probabilities[name];
    const flagged = prob !== null && prob !== undefined && prob >= 0.5;
    return `<div class="model-dot ${flagged ? "flagged" : "safe"}" title="${name}: ${prob !== null ? (prob * 100).toFixed(1) + "%" : "N/A"}"></div>`;
  }).join("");

  if (agreementEl) {
    agreementEl.textContent = `${flags} of 4 models flagged as fraud`;
  }
}

function _renderTriggeredRules(rules) {
  const container = document.getElementById("predictor-triggered-rules");
  if (!container) return;

  if (!rules.length) {
    container.innerHTML = `<p class="caption" style="color:var(--txt3)">No policy rules triggered</p>`;
    return;
  }

  container.innerHTML = rules.map(rule => `
    <div style="display:flex;align-items:flex-start;gap:8px;padding:8px 12px;background:rgba(255,64,129,0.06);border:1px solid rgba(255,64,129,0.2);border-radius:var(--r-md);margin-bottom:5px">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--anomaly)" stroke-width="2" style="margin-top:1px;flex-shrink:0">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z"/>
      </svg>
      <div>
        <div style="font-size:12px;font-weight:600;color:var(--anomaly)">${rule.name}</div>
        <div style="font-size:11px;color:var(--txt3);margin-top:2px">${rule.description}</div>
      </div>
      <span style="margin-left:auto;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--warn)">${(rule.weight * 100).toFixed(0)}%</span>
    </div>
  `).join("");
}

// ── Explanation ───────────────────────────────────────────────────────────────

function _renderExplanation(explainResult) {
  const { reason_codes = [], narrative = "", narrative_source = "local_fallback" } = explainResult;

  // Reason codes
  const rcContainer = document.getElementById("predictor-reason-codes");
  if (rcContainer) {
    rcContainer.innerHTML = reason_codes.map((rc, i) => `
      <div class="reason-code-item animate-in" style="animation-delay:${i * 60}ms">
        <div class="reason-code-num">${i + 1}.</div>
        <div class="reason-code-text">${rc.description}</div>
        <div class="reason-code-badge badge ${rc.severity === "high" ? "badge-high" : rc.severity === "medium" ? "badge-warn" : "badge-normal"}">${rc.source?.replace(/_/g, " ") || "signal"}</div>
      </div>
    `).join("");
  }

  // LLM narrative
  const narrativeEl = document.getElementById("predictor-narrative");
  if (narrativeEl) narrativeEl.textContent = narrative;

  const sourceEl = document.getElementById("predictor-narrative-source");
  if (sourceEl) {
    sourceEl.textContent = narrative_source === "openrouter" ? "Generated by OpenRouter · nemotron" : "Local analysis";
    sourceEl.className = `source-badge ${narrative_source === "openrouter" ? "badge-accent" : "badge-muted"}`;
  }
}

// ── What-if ───────────────────────────────────────────────────────────────────

function _renderWhatIf(features, result) {
  const container = document.getElementById("predictor-whatif");
  if (!container || !_colConfig) return;

  const colStats = _colConfig.col_stats || {};
  const scorePct = result.risk?.final_score_pct || 0;

  // Find the feature with highest Z-score
  let topFeat = null;
  let topZ = 0;

  for (const [feat, val] of Object.entries(features)) {
    const stats = colStats[feat];
    if (!stats) continue;
    const z = (val - stats.mean) / (stats.std || 1);
    if (z > topZ) {
      topZ = z;
      topFeat = feat;
    }
  }

  if (!topFeat || topZ < 0.5) {
    container.innerHTML = `<div class="whatif-box"><span class="text-muted">No significant reduction factor found for this transaction.</span></div>`;
    return;
  }

  const stats = colStats[topFeat];
  const reducedVal = Math.max(stats.min, features[topFeat] - stats.std);
  const reduction = ((features[topFeat] - reducedVal) / Math.max(features[topFeat], 0.01) * 100).toFixed(0);

  container.innerHTML = `
    <div class="whatif-box">
      <strong>What-if hint:</strong>
      Reducing <strong class="feature-name" style="font-family:'JetBrains Mono',monospace;color:var(--accent)">${_humanize(topFeat)}</strong>
      by <strong>${stats.std.toFixed(2)}</strong> (1 standard deviation) from
      <strong>${features[topFeat].toFixed(2)}</strong> → <strong>${reducedVal.toFixed(2)}</strong>
      would reduce anomaly signals significantly (↓${reduction}% on this feature).
      Current risk score: <strong style="color:var(--warn)">${scorePct.toFixed(0)}%</strong>.
    </div>
  `;
}

// ── Technical View Charts ─────────────────────────────────────────────────────

export async function renderTechnical(sessionId, metrics) {
  try {
    const { getRocCurve, getPrCurve, getFeatureImportance } = await import("./api.js");
    const [roc, pr, fi] = await Promise.all([
      getRocCurve(sessionId),
      getPrCurve(sessionId),
      getFeatureImportance(sessionId),
    ]);
    renderRocCurve("predictor-roc-canvas", roc);
    renderPrCurve("predictor-pr-canvas", pr);
    const imps = fi.feature_importances || fi.coefficients || {};
    renderFeatureImportance("predictor-fi-canvas", imps);

    const bestModel = Object.entries(metrics).sort((a, b) => (b[1].auc || 0) - (a[1].auc || 0))[0];
    if (bestModel) {
      renderConfusionMatrix("predictor-cm-container", bestModel[1].confusion_matrix);
    }
  } catch (_) {}
}

// ── View Toggle ───────────────────────────────────────────────────────────────

function _setupViewToggle() {
  const toggle = document.getElementById("predictor-view-toggle");
  if (!toggle) return;

  toggle.addEventListener("click", e => {
    const btn = e.target.closest(".view-toggle-btn");
    if (!btn) return;
    _currentView = btn.dataset.view;
    toggle.querySelectorAll(".view-toggle-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.view === _currentView);
    });
    document.getElementById("predictor-auditor-view")?.classList.toggle("hidden", _currentView !== "auditor");
    document.getElementById("predictor-technical-view")?.classList.toggle("hidden", _currentView !== "technical");
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _humanize(feat) {
  return feat.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function _safeId(feat) {
  return feat.replace(/[^a-zA-Z0-9]/g, "_");
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
