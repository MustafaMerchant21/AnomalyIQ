/**
 * testdataset.js — AnomalyIQ Test Dataset (Unlabelled Prediction System)
 *
 * Allows the user to upload an unlabelled CSV, runs inference via the
 * already-trained session model, and displays results as interactive
 * transaction cards split into Normal / Fraud categories.
 *
 * Exports: init()
 */

import { testDataset } from "./api.js";
import { getSession, hasSession } from "./session.js";
import { showToast } from "./app.js";

// ── State ─────────────────────────────────────────────────────────────────────

let _initialized = false;
let _lastResults = null;  // { normal[], fraud[], feature_cols[], model_used }

// ── Init ──────────────────────────────────────────────────────────────────────

export function init() {
  if (_initialized) return;
  _initialized = true;

  _bindDropZone();
  _bindFileInput();
  _bindModalClose();
  _checkSession();
}

export function refresh() {
  _checkSession();
}

// ── Session guard ─────────────────────────────────────────────────────────────

function _checkSession() {
  const guard = document.getElementById("td-session-guard");
  const body  = document.getElementById("td-body");
  if (!hasSession()) {
    if (guard) guard.classList.remove("hidden");
    if (body)  body.classList.add("hidden");
  } else {
    if (guard) guard.classList.add("hidden");
    if (body)  body.classList.remove("hidden");
  }
}

// ── Drop zone wiring ──────────────────────────────────────────────────────────

function _bindDropZone() {
  const zone  = document.getElementById("td-drop-zone");
  const input = document.getElementById("td-file-input");
  if (!zone || !input) return;

  zone.addEventListener("click", () => input.click());

  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") input.click();
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) _handleFile(file);
  });
}

function _bindFileInput() {
  const input = document.getElementById("td-file-input");
  if (!input) return;
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (file) _handleFile(file);
    input.value = ""; // allow re-selecting the same file
  });
}

// ── File handling & prediction ────────────────────────────────────────────────

async function _handleFile(file) {
  if (!file.name.endsWith(".csv")) {
    showToast("Invalid File", "Only .csv files are accepted.", "error");
    return;
  }

  const sessionId = getSession();
  if (!sessionId) {
    showToast("No Session", "Please train a model first (Upload Dataset).", "error");
    return;
  }

  // Update drop zone label
  const label = document.getElementById("td-drop-label");
  if (label) label.textContent = file.name;

  _showSpinner(true);
  _clearResults();

  try {
    const result = await testDataset(sessionId, file);
    _lastResults = result;
    _renderResults(result);
    showToast(
      "Prediction Complete",
      `${result.total} transactions processed — ${result.fraud_count} fraud, ${result.normal_count} normal.`,
      "success"
    );
  } catch (err) {
    showToast("Prediction Failed", err.message, "error");
    const errEl = document.getElementById("td-error");
    if (errEl) {
      errEl.textContent = err.message;
      errEl.classList.remove("hidden");
    }
  } finally {
    _showSpinner(false);
  }
}

// ── Spinner ───────────────────────────────────────────────────────────────────

function _showSpinner(show) {
  const spinner  = document.getElementById("td-spinner-overlay");
  const results  = document.getElementById("td-results");
  if (spinner) spinner.classList.toggle("hidden", !show);
  if (results && !show) results.classList.remove("hidden");
}

function _clearResults() {
  const errEl = document.getElementById("td-error");
  if (errEl) errEl.classList.add("hidden");

  const results = document.getElementById("td-results");
  if (results) results.classList.add("hidden");

  document.getElementById("td-fraud-grid")?.replaceChildren();
}

// ── Render results ────────────────────────────────────────────────────────────

function _renderResults(result) {
  const { normal, fraud, total, fraud_count, normal_count, model_used } = result;

  // Summary stats
  _setText("td-stat-total",  total);
  _setText("td-stat-fraud",  fraud_count);
  _setText("td-stat-normal", normal_count);
  _setText("td-stat-model",  _humanModelName(model_used));

  // Fraud rate badge
  const rate = total > 0 ? ((fraud_count / total) * 100).toFixed(1) : "0.0";
  _setText("td-stat-rate", `${rate}%`);

  // Render cards
  _renderCardGrid("td-fraud-grid",  fraud,  "fraud");

  // Count badges on section headers
  _setText("td-fraud-count-badge",  fraud_count);

  // Show empty states if needed
  _toggleEmpty("td-fraud-empty",  fraud_count  === 0);

  document.getElementById("td-results")?.classList.remove("hidden");
}

function _renderCardGrid(gridId, items, type) {
  const grid = document.getElementById(gridId);
  if (!grid) return;

  grid.replaceChildren();

  const CHUNK = 50; // render first 50 immediately, rest on scroll-demand
  const visible = items.slice(0, CHUNK);

  visible.forEach((item, i) => {
    const card = _buildCard(item, type, i);
    grid.appendChild(card);
  });

  // Lazy-load remaining cards (intersection observer)
  if (items.length > CHUNK) {
    const sentinel = document.createElement("div");
    sentinel.className = "td-sentinel";
    grid.appendChild(sentinel);
    let idx = CHUNK;
    const obs = new IntersectionObserver((entries) => {
      if (!entries[0].isIntersecting) return;
      const batch = items.slice(idx, idx + CHUNK);
      sentinel.remove();
      obs.disconnect();
      batch.forEach((item, bi) => {
        const card = _buildCard(item, type, idx + bi);
        grid.appendChild(card);
      });
      idx += CHUNK;
    });
    obs.observe(sentinel);
  }
}

function _buildCard(item, type, idx) {
  const isFraud = type === "fraud";
  const prob    = (item.probability * 100).toFixed(1);
  const conf    = (item.confidence  * 100).toFixed(1);

  const card = document.createElement("div");
  card.className   = `td-card td-card-${type}`;
  card.tabIndex    = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-label", `${item.transaction_id} — ${item.verdict}`);

  card.innerHTML = `
    <div class="td-card-top">
      <span class="td-card-id">${item.transaction_id}</span>
      <span class="td-card-badge td-badge-${type}">${item.verdict}</span>
    </div>
    <div class="td-card-prob-bar">
      <div class="td-card-prob-fill td-fill-${type}" style="width:${prob}%"></div>
    </div>
    <div class="td-card-meta">
      <span class="td-card-prob-label">Risk: <strong>${prob}%</strong></span>
      <span class="td-card-conf-label">Confidence: <strong>${conf}%</strong></span>
    </div>
  `;

  card.addEventListener("click",  () => _openModal(item, type));
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") _openModal(item, type);
  });

  // Stagger animation
  card.style.animationDelay = `${Math.min(idx * 30, 500)}ms`;

  return card;
}

// ── Modal ─────────────────────────────────────────────────────────────────────

function _openModal(item, type) {
  const modal   = document.getElementById("td-modal");
  const overlay = document.getElementById("td-modal-overlay");
  if (!modal || !overlay) return;

  const isFraud = type === "fraud";
  const prob    = (item.probability * 100).toFixed(1);
  const conf    = (item.confidence  * 100).toFixed(1);

  // Header
  document.getElementById("td-modal-title")?.setAttribute(
    "data-verdict", item.verdict
  );
  _setText("td-modal-txn-id",  item.transaction_id);
  _setText("td-modal-verdict", item.verdict);
  _setText("td-modal-prob",    `${prob}%`);
  _setText("td-modal-conf",    `${conf}%`);

  const badge = document.getElementById("td-modal-verdict-badge");
  if (badge) {
    badge.textContent = item.verdict;
    badge.className   = `badge td-badge-${type}`;
  }

  // Probability bar
  const bar = document.getElementById("td-modal-prob-bar");
  if (bar) {
    bar.style.width = `${prob}%`;
    bar.className   = `td-modal-prob-bar-fill td-fill-${type}`;
  }

  // Feature table
  const tbody = document.getElementById("td-modal-features-tbody");
  if (tbody) {
    tbody.replaceChildren();
    const features = item.features || {};
    Object.entries(features).forEach(([key, val]) => {
      const tr = document.createElement("tr");
      const formatted = typeof val === "number"
        ? (Number.isInteger(val) ? val : val.toFixed(4))
        : (val ?? "—");
      tr.innerHTML = `
        <td class="td-feat-key">${key}</td>
        <td class="td-feat-val mono">${formatted}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Show
  overlay.classList.remove("hidden");
  modal.classList.remove("hidden");
  modal.classList.add("td-modal-in");
  document.body.style.overflow = "hidden";
}

function _closeModal() {
  const modal   = document.getElementById("td-modal");
  const overlay = document.getElementById("td-modal-overlay");
  if (!modal || !overlay) return;

  modal.classList.remove("td-modal-in");
  modal.classList.add("td-modal-out");

  setTimeout(() => {
    overlay.classList.add("hidden");
    modal.classList.add("hidden");
    modal.classList.remove("td-modal-out");
    document.body.style.overflow = "";
  }, 250);
}

function _bindModalClose() {
  document.getElementById("td-modal-close")?.addEventListener("click", _closeModal);
  document.getElementById("td-modal-overlay")?.addEventListener("click", _closeModal);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") _closeModal();
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? "—";
}

function _toggleEmpty(id, show) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("hidden", !show);
}

function _humanModelName(name) {
  const map = {
    logistic_regression: "Logistic Regression",
    svm:                 "Support Vector Machine",
    decision_tree:       "Decision Tree",
    neural_network:      "Neural Network",
  };
  return map[name] || name;
}
