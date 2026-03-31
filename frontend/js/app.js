/**
 * app.js — AnomalyIQ Router, Navigation, Init
 *
 * Hash-based/data-attribute router with navigateTo(sectionId).
 * Health check every 30s.
 * Imports and calls init() on each section module on first navigation.
 */

import { loadSession, setSession, hasSession, getSession, clearSession } from "./session.js";
import { getHealth } from "./api.js";
import { init as initUpload, resetUploadSection } from "./upload.js";
import { init as initDashboard, refresh as refreshDashboard } from "./dashboard.js";
import { init as initTestDataset, refresh as refreshTestDataset } from "./testdataset.js";
import { init as initPredictor } from "./predictor.js";
import { init as initSimulator } from "./simulator.js";
import { init as initGlobe } from "./globe.js";
import { init as initInsight } from "./insight.js";
import { init as initAudit } from "./audit.js";
import { init as initInvestigation } from "./investigation.js";
import { init as initBlacklist } from "./blacklist.js";

// ── Section registry ──────────────────────────────────────────────────────────

const SECTIONS = {
  upload:      { id: "section-upload",       init: null, title: "Upload Dataset" },
  testdataset: { id: "section-testdataset",  init: null, title: "Test Dataset" },
  dashboard:   { id: "section-dashboard",   init: null, title: "Dashboard" },
  predictor:   { id: "section-predictor",   init: null, title: "Predictor" },
  simulator:   { id: "section-simulator",   init: null, title: "Simulator" },
  globe:       { id: "section-globe",       init: null, title: "3D Globe" },
  insight:     { id: "section-insight",     init: null, title: "AI Insight" },
  audit:       { id: "section-audit",       init: null, title: "Audit Trail" },
  investigation: { id: "section-investigation", init: null, title: "Investigation Queue" },
  blacklist:   { id: "section-blacklist",   init: null, title: "Blacklist" },
};

let _currentSection = null;
let _healthTimer = null;
let _sectionInitFlags = new Set();

// ── App Bootstrap ─────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  _initNav();
  _startHealthCheck();

  // Restore session from localStorage
  const sessionId = loadSession();

  // Register section inits
  SECTIONS.upload.init      = () => initUpload(_onTrainingComplete);
  SECTIONS.testdataset.init = initTestDataset;
  SECTIONS.dashboard.init   = initDashboard;
  SECTIONS.predictor.init   = initPredictor;
  SECTIONS.simulator.init   = initSimulator;
  SECTIONS.globe.init       = initGlobe;
  SECTIONS.insight.init     = initInsight;
  SECTIONS.audit.init       = initAudit;
  SECTIONS.investigation.init = initInvestigation;
  SECTIONS.blacklist.init   = initBlacklist;

  // Route to appropriate section
  if (sessionId) {
    navigateTo("dashboard");
  } else {
    navigateTo("upload");
  }
});

// ── Callback: training complete → navigate to dashboard ───────────────────────

function _onTrainingComplete(sessionId) {
  setSession(sessionId);
  navigateTo("dashboard");
  // Refresh dashboard with new data
  refreshDashboard?.();
  refreshTestDataset?.();
}

// ── Router ────────────────────────────────────────────────────────────────────

export function navigateTo(sectionName) {
  const section = SECTIONS[sectionName];
  if (!section) return;

  // Hide all sections
  Object.values(SECTIONS).forEach(s => {
    const el = document.getElementById(s.id);
    if (el) el.classList.remove("active");
  });

  // Show target section
  const target = document.getElementById(section.id);
  if (target) target.classList.add("active");

  // Update nav
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.toggle("active", item.dataset.section === sectionName);
  });

  // Update topbar title
  const topbarTitle = document.getElementById("topbar-section-title");
  if (topbarTitle) topbarTitle.textContent = section.title;

  _currentSection = sectionName;

  // Init section on first visit
  if (!_sectionInitFlags.has(sectionName) && section.init) {
    _sectionInitFlags.add(sectionName);
    section.init();
  } else if (sectionName === "testdataset") {
    refreshTestDataset?.();
  }
}

// ── Navigation setup ──────────────────────────────────────────────────────────

function _initNav() {
  document.querySelectorAll(".nav-item[data-section]").forEach(item => {
    item.addEventListener("click", () => {
      const target = item.dataset.section;
      navigateTo(target);
    });
  });
}

// ── Health Check ──────────────────────────────────────────────────────────────

function _startHealthCheck() {
  _checkHealth();
  _healthTimer = setInterval(_checkHealth, 30000);
}

async function _checkHealth() {
  const dot = document.getElementById("connection-dot");
  try {
    const result = await getHealth();
    if (result.status === "ok") {
      dot?.classList.add("connected");
      dot?.classList.remove("disconnected");
    } else {
      _setDisconnected(dot);
    }
  } catch (_) {
    _setDisconnected(dot);
  }
}

function _setDisconnected(dot) {
  dot?.classList.remove("connected");
  dot?.classList.add("disconnected");
}

// ── Toast System ──────────────────────────────────────────────────────────────

export function showToast(title, message, type = "error") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type === "success" ? "success" : type === "warning" ? "warning" : ""}`;

  const iconSvg = type === "success"
    ? `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="var(--ok)" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`
    : type === "warning"
    ? `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="var(--warn)" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z"/></svg>`
    : `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="var(--anomaly)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`;

  toast.innerHTML = `
    ${iconSvg}
    <div class="toast-content">
      <div class="toast-title">${title}</div>
      ${message ? `<div class="toast-message">${message}</div>` : ""}
    </div>
  `;

  container.appendChild(toast);

  // Auto-dismiss after 4 seconds
  setTimeout(() => {
    toast.style.animation = "fade-out 0.3s ease forwards";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Export navigateTo for use in other modules
window.navigateTo = navigateTo;
