/**
 * audit.js — AnomalyIQ Case Management & Audit Trail
 *
 * Case list with filter tabs (Flag/Allow/Escalate/Note),
 * case detail panel with transaction features + ML scores + narrative,
 * actions (Flag/Allow/Escalate/Add Note), localStorage persistence,
 * CSV export + PDF export (jsPDF).
 */

import { getCaseList, saveCase, exportCsv } from "./api.js";
import { getSession } from "./session.js";
import { showToast } from "./app.js";

let _initialized = false;
let _cases = [];
let _selectedCase = null;
let _activeFilter = "all";

// ── Init ──────────────────────────────────────────────────────────────────────

export async function init() {
  if (_initialized) return;
  _initialized = true;

  _setupFilters();
  _setupExportButtons();
  await _loadCases();
}

export async function refresh() {
  _initialized = false;
  _cases = [];
  _selectedCase = null;
  await init();
}

// ── Load Cases ────────────────────────────────────────────────────────────────

async function _loadCases() {
  const sessionId = getSession();

  // Load from backend (persist across sessions)
  let backendCases = [];
  if (sessionId) {
    try {
      const result = await getCaseList(sessionId);
      backendCases = result.cases || [];
    } catch (_) {}
  }

  // Merge with localStorage cases
  const localStr = localStorage.getItem("anomalyiq_cases") || "[]";
  let localCases = [];
  try { localCases = JSON.parse(localStr); } catch (_) {}

  // Deduplicate by ID
  const allById = {};
  for (const c of [...backendCases, ...localCases]) {
    if (!allById[c.id]) allById[c.id] = c;
  }
  _cases = Object.values(allById).sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

  _renderCaseList();
}

// ── Filter Tabs ───────────────────────────────────────────────────────────────

function _setupFilters() {
  const tabs = document.querySelectorAll(".audit-filter-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      _activeFilter = tab.dataset.filter || "all";
      tabs.forEach(t => t.classList.toggle("active", t.dataset.filter === _activeFilter));
      _renderCaseList();
    });
  });
}

// ── Case List ─────────────────────────────────────────────────────────────────

function _renderCaseList() {
  const container = document.getElementById("audit-case-list");
  if (!container) return;

  const filtered = _activeFilter === "all"
    ? _cases
    : _cases.filter(c => c.decision === _activeFilter);

  if (!filtered.length) {
    container.innerHTML = `
      <div class="empty-state" style="padding:32px 16px">
        <p style="font-size:13px;color:var(--txt3)">No ${_activeFilter === "all" ? "" : _activeFilter + " "}cases yet</p>
        <p class="caption" style="margin-top:4px">Simulate transactions and save decisions to build your audit trail.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = filtered.map(c => `
    <div class="case-item ${_selectedCase?.id === c.id ? "selected" : ""}"
         data-case-id="${c.id}"
         onclick="window._auditSelectCase('${c.id}')">
      <div class="case-item-header">
        <span class="badge ${_decisionClass(c.decision)}">${c.decision?.toUpperCase() || "UNKNOWN"}</span>
        <span class="badge ${_verdictClass(c.verdict)}" style="font-size:9px">${c.verdict?.replace(/_/g, " ") || "—"}</span>
      </div>
      <div style="font-size:12px;color:var(--txt2);margin-bottom:3px">
        TX ${c.transaction_id?.substring(0, 8) || c.id?.substring(0, 8)}
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="mono" style="font-size:11px;color:${c.risk_score >= 0.7 ? "var(--anomaly)" : c.risk_score >= 0.4 ? "var(--warn)" : "var(--ok)"}">
          ${c.risk_score != null ? (c.risk_score * 100).toFixed(0) + "%" : "—"}
        </span>
        <span class="case-timestamp">${_formatTime(c.timestamp)}</span>
      </div>
    </div>
  `).join("");
}

// Expose select function globally for onclick
window._auditSelectCase = (caseId) => {
  _selectedCase = _cases.find(c => c.id === caseId) || null;
  _renderCaseList();
  _renderCaseDetail(_selectedCase);
};

// ── Case Detail Panel ─────────────────────────────────────────────────────────

function _renderCaseDetail(c) {
  const container = document.getElementById("audit-case-detail");
  if (!container) return;

  if (!c) {
    container.innerHTML = `
      <div class="empty-state">
        <p>Select a case from the list to review details.</p>
      </div>
    `;
    return;
  }

  const features = c.transaction_data?.features || {};
  const featureRows = Object.entries(features).length > 0
    ? Object.entries(features).map(([k, v]) => `
        <tr>
          <td style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--accent)">${k}</td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--txt);text-align:right">${typeof v === "number" ? v.toFixed(4) : v}</td>
        </tr>
      `).join("")
    : `<tr><td colspan="2" style="color:var(--txt3);font-size:12px">No feature data recorded</td></tr>`;

  const narrative = c.transaction_data?.narrative || c.narrative || "No narrative recorded.";
  const reasonCodes = c.transaction_data?.reason_codes || [];

  container.innerHTML = `
    <div class="case-detail animate-in">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:var(--sp-5)">
        <div>
          <h3 style="color:var(--txt);margin-bottom:4px">Case Review</h3>
          <div style="display:flex;gap:6px">
            <span class="badge ${_decisionClass(c.decision)}">${c.decision?.toUpperCase()}</span>
            <span class="badge ${_verdictClass(c.verdict)}">${c.verdict?.replace(/_/g, " ") || "—"}</span>
            ${c.risk_score != null ? `<span class="mono" style="font-size:12px;color:var(--txt2)">${(c.risk_score * 100).toFixed(0)}% risk</span>` : ""}
          </div>
        </div>
        <div class="caption">${_formatTime(c.timestamp, true)}</div>
      </div>

      ${featureRows ? `
        <div style="margin-bottom:var(--sp-5)">
          <div class="section-title">Transaction Features</div>
          <div class="table-wrapper">
            <table><tbody>${featureRows}</tbody></table>
          </div>
        </div>
      ` : ""}

      ${reasonCodes.length ? `
        <div style="margin-bottom:var(--sp-5)">
          <div class="section-title">Reason Codes</div>
          ${reasonCodes.map((rc, i) => `
            <div class="reason-code-item">
              <div class="reason-code-num">${i + 1}.</div>
              <div class="reason-code-text">${rc.description || rc}</div>
            </div>
          `).join("")}
        </div>
      ` : ""}

      <div style="margin-bottom:var(--sp-5)">
        <div class="section-title">AI Narrative</div>
        <div class="narrative-block">${narrative}</div>
      </div>

      ${c.note ? `
        <div style="margin-bottom:var(--sp-4);padding:12px;background:var(--surface-up);border-radius:var(--r-md);border:1px solid var(--border)">
          <div class="section-title">Investigator Note</div>
          <p style="font-size:13px;color:var(--txt2);margin-top:4px">${c.note}</p>
        </div>
      ` : ""}

      <div class="divider"></div>

      <div>
        <div class="section-title">Actions</div>
        <div class="audit-actions">
          <button class="btn btn-danger" onclick="window._auditAction('flag', '${c.id}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>
            Flag
          </button>
          <button class="btn btn-success" onclick="window._auditAction('allow', '${c.id}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>
            Allow
          </button>
          <button class="btn btn-warn" onclick="window._auditAction('escalate', '${c.id}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>
            Escalate
          </button>
        </div>

        <div style="display:flex;gap:8px;margin-top:8px">
          <input type="text" id="audit-note-input" placeholder="Add a note..." style="flex:1">
          <button class="btn btn-ghost" onclick="window._auditAddNote('${c.id}')">Add Note</button>
        </div>
      </div>
    </div>
  `;
}

window._auditAction = async (action, caseId) => {
  const c = _cases.find(c => c.id === caseId);
  if (!c) return;

  c.decision = action;
  c.timestamp = Date.now();

  await _persistCase(c);
  _renderCaseList();
  _renderCaseDetail(c);
  showToast("Case updated", `Decision: ${action}`, "success");
};

window._auditAddNote = async (caseId) => {
  const input = document.getElementById("audit-note-input");
  const note = input?.value?.trim();
  if (!note) return;

  const c = _cases.find(c => c.id === caseId);
  if (!c) return;

  c.note = note;
  c.decision = c.decision || "note";
  c.timestamp = Date.now();

  await _persistCase(c);
  if (input) input.value = "";
  _renderCaseList();
  _renderCaseDetail(c);
  showToast("Note saved", "", "success");
};

// ── Persist Case ──────────────────────────────────────────────────────────────

async function _persistCase(caseObj) {
  // Update in-memory list
  const idx = _cases.findIndex(c => c.id === caseObj.id);
  if (idx >= 0) _cases[idx] = caseObj;
  else _cases.unshift(caseObj);

  // Persist to localStorage
  localStorage.setItem("anomalyiq_cases", JSON.stringify(_cases.slice(0, 100)));

  // Sync to backend
  const sessionId = getSession();
  if (sessionId) {
    try {
      await saveCase(
        sessionId,
        caseObj.transaction_id || caseObj.id,
        caseObj.decision || "note",
        caseObj.note || null,
        caseObj.transaction_data || null,
        caseObj.risk_score || null,
        caseObj.verdict || null,
      );
    } catch (_) {}
  }
}

// ── Export ────────────────────────────────────────────────────────────────────

function _setupExportButtons() {
  document.getElementById("audit-export-csv")?.addEventListener("click", _exportCsvHandler);
  document.getElementById("audit-export-pdf")?.addEventListener("click", _exportPdf);
}

function _exportCsvHandler() {
  const sessionId = getSession();
  if (!sessionId) {
    showToast("No session", "Train a model first.", "error");
    return;
  }
  const url = exportCsv(sessionId);
  const a = document.createElement("a");
  a.href = url;
  a.download = `anomalyiq_export.csv`;
  a.click();
  showToast("Exporting", "CSV download started", "success");
}

function _exportPdf() {
  if (!_selectedCase) {
    showToast("No case selected", "Select a case to export.", "error");
    return;
  }

  if (typeof jspdf === "undefined" && typeof jsPDF === "undefined") {
    showToast("PDF unavailable", "jsPDF library not loaded.", "error");
    return;
  }

  const JsPDF = window.jspdf?.jsPDF || window.jsPDF;
  if (!JsPDF) {
    showToast("PDF unavailable", "jsPDF not loaded.", "error");
    return;
  }

  const c = _selectedCase;
  const doc = new JsPDF({ unit: "pt", format: "a4" });

  let y = 60;

  // Header
  doc.setFontSize(20);
  doc.setTextColor(156, 111, 255);
  doc.text("AnomalyIQ — Case Report", 60, y);
  y += 30;

  doc.setFontSize(10);
  doc.setTextColor(107, 100, 146);
  doc.text(`Generated: ${new Date().toLocaleString()}`, 60, y);
  y += 10;
  doc.text(`Session: ${getSession()?.substring(0, 16) || "—"}`, 60, y);
  y += 10;
  doc.text(`Case ID: ${c.id}`, 60, y);
  y += 24;

  // Decision + Verdict
  doc.setFontSize(12);
  doc.setTextColor(237, 233, 255);
  doc.text(`Decision: ${(c.decision || "—").toUpperCase()}`, 60, y);
  y += 16;
  doc.text(`Verdict: ${c.verdict?.replace(/_/g, " ") || "—"}`, 60, y);
  y += 16;
  if (c.risk_score != null) {
    doc.text(`Risk Score: ${(c.risk_score * 100).toFixed(1)}%`, 60, y);
    y += 16;
  }
  y += 8;

  // Reason codes
  const reasonCodes = c.transaction_data?.reason_codes || [];
  if (reasonCodes.length) {
    doc.setFontSize(12);
    doc.setTextColor(156, 111, 255);
    doc.text("Reason Codes:", 60, y);
    y += 16;
    doc.setFontSize(10);
    doc.setTextColor(168, 158, 214);
    for (const rc of reasonCodes) {
      const lines = doc.splitTextToSize(`• ${rc.description || rc}`, 480);
      doc.text(lines, 70, y);
      y += lines.length * 13;
    }
    y += 8;
  }

  // Narrative
  const narrative = c.transaction_data?.narrative || c.narrative || "No narrative.";
  doc.setFontSize(12);
  doc.setTextColor(156, 111, 255);
  doc.text("AI Narrative:", 60, y);
  y += 16;
  doc.setFontSize(10);
  doc.setTextColor(237, 233, 255);
  const narrativeLines = doc.splitTextToSize(narrative, 480);
  doc.text(narrativeLines, 60, y);
  y += narrativeLines.length * 13 + 8;

  // Note
  if (c.note) {
    doc.setFontSize(10);
    doc.setTextColor(107, 100, 146);
    doc.text(`Investigator note: ${c.note}`, 60, y);
    y += 16;
  }

  // Timestamp
  doc.setFontSize(9);
  doc.setTextColor(107, 100, 146);
  doc.text(`Decision timestamp: ${_formatTime(c.timestamp, true)}`, 60, y);

  doc.save(`anomalyiq_case_${c.id?.substring(0, 8)}.pdf`);
  showToast("PDF exported", "Case report downloaded", "success");
}

// ── Public: Save a case from simulator/predictor ───────────────────────────────

export async function saveCurrentCase(transactionData) {
  const { features, verdict, riskScore, mlResult, narrative, reasonCodes } = transactionData;

  const caseObj = {
    id: crypto.randomUUID(),
    transaction_id: `TX_${Date.now()}`,
    decision: "note",
    note: null,
    transaction_data: {
      features: features || {},
      verdict,
      narrative: narrative || "",
      reason_codes: reasonCodes || [],
      ml_probs: mlResult?.probabilities || {},
    },
    risk_score: riskScore || 0,
    verdict: verdict || "LIKELY_NORMAL",
    timestamp: Date.now(),
    timestamp_iso: new Date().toISOString(),
  };

  _cases.unshift(caseObj);
  await _persistCase(caseObj);
  showToast("Case saved", "Added to audit trail", "success");
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function _decisionClass(decision) {
  const map = {
    flag: "badge-high",
    allow: "badge-ok",
    escalate: "badge-warn",
    note: "badge-accent",
  };
  return map[decision] || "badge-muted";
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

function _formatTime(ts, full = false) {
  if (!ts) return "—";
  const d = new Date(ts * 1000 || ts);
  if (full) return d.toLocaleString();
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
