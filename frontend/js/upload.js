/**
 * upload.js — AnomalyIQ CSV Upload + Column Mapper + Training Progress
 *
 * Drop zone with file picking, demo dataset button,
 * column selector/mapper, 9-step progress stepper.
 * Polls /api/training_status every 1200ms.
 */

import { uploadCsv, uploadDemo, getTrainingStatus } from "./api.js";
import { setSession } from "./session.js";
import { showToast } from "./app.js";

let _pollTimer = null;
let _currentSessionId = null;
let _parsedHeaders = [];
let _onTrainingComplete = null;

// ── Init ──────────────────────────────────────────────────────────────────────

export function init(onComplete) {
  _onTrainingComplete = onComplete;
  _setupDropZone();
  _setupDemoButton();
  _setupTrainButton();
}

// ── Drop Zone ─────────────────────────────────────────────────────────────────

function _setupDropZone() {
  const zone = document.getElementById("upload-drop-zone");
  const fileInput = document.getElementById("upload-file-input");

  if (!zone || !fileInput) return;

  zone.addEventListener("click", () => fileInput.click());

  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("drag-over");
  });

  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer?.files?.[0];
    if (file) _handleFile(file);
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    if (file) _handleFile(file);
  });
}

function _handleFile(file) {
  if (!file.name.endsWith(".csv")) {
    showToast("Invalid file", "Please upload a CSV file.", "error");
    return;
  }

  const zone = document.getElementById("upload-drop-zone");
  zone.innerHTML = `
    <svg class="cloud-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path stroke-linecap="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
    </svg>
    <p class="drop-zone-title" style="color:var(--ok)">${file.name}</p>
    <p class="drop-zone-sub">${(file.size / 1024).toFixed(1)} KB — Ready to configure</p>
  `;

  _readCsvHeaders(file);
}

function _readCsvHeaders(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const text = e.target.result;
    const firstLine = text.split("\n")[0].trim();
    _parsedHeaders = firstLine.split(",").map(h => h.trim().replace(/^"|"$/g, ""));
    _buildColumnMapper(_parsedHeaders);
    document.getElementById("upload-mapper-section").classList.remove("hidden");
    document.getElementById("upload-train-btn").classList.remove("hidden");
  };
  reader.readAsText(file.slice(0, 4096)); // Read just the first 4KB for headers
}

function _buildColumnMapper(headers) {
  const featureList = document.getElementById("upload-feature-list");
  const targetSelect = document.getElementById("upload-target-select");

  if (!featureList || !targetSelect) return;

  // Feature checkboxes
  featureList.innerHTML = headers.map((h, i) => `
    <label class="checkbox-label" style="margin-bottom:6px">
      <input type="checkbox" class="feature-checkbox" value="${_esc(h)}" ${i < headers.length - 1 ? "checked" : ""}>
      <span class="feature-chip" style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--accent)">${_esc(h)}</span>
    </label>
  `).join("");

  // Target dropdown
  targetSelect.innerHTML = headers.map((h, i) => `
    <option value="${_esc(h)}" ${i === headers.length - 1 ? "selected" : ""}>${_esc(h)}</option>
  `).join("");

  // Dataset context fields (rendered once; safe to re-render on new file)
  const contextSection = document.getElementById("upload-context-section");
  if (contextSection) {
    contextSection.innerHTML = `
      <div class="context-field-group">
        <div class="context-field">
          <label class="context-label" for="upload-dataset-name">Dataset Name <span style="opacity:0.5;font-size:11px">(optional)</span></label>
          <input
            id="upload-dataset-name"
            type="text"
            class="context-input"
            placeholder="e.g. Kaggle Fraud Detection 2020"
            maxlength="120"
          />
        </div>
        <div class="context-field">
          <label class="context-label" for="upload-domain-desc">Domain / Context <span style="opacity:0.5;font-size:11px">(optional — helps the AI write better summaries)</span></label>
          <input
            id="upload-domain-desc"
            type="text"
            class="context-input"
            placeholder="e.g. Credit card fraud detection for US retail banking customers"
            maxlength="250"
          />
        </div>
      </div>
    `;
  }
}

// ── Demo Dataset Button ───────────────────────────────────────────────────────

function _setupDemoButton() {
  const btn = document.getElementById("upload-demo-btn");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;display:inline-block;margin-right:6px"></span>Starting demo...`;

    try {
      const result = await uploadDemo();
      _currentSessionId = result.session_id;
      setSession(_currentSessionId);
      _showProgressSection();
      _startPolling(_currentSessionId);
    } catch (err) {
      showToast("Demo failed", err.message, "error");
      btn.disabled = false;
      btn.innerHTML = `Try with demo dataset`;
    }
  });
}

// ── Train Button ──────────────────────────────────────────────────────────────

function _setupTrainButton() {
  const btn = document.getElementById("upload-train-btn");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    const fileInput = document.getElementById("upload-file-input");
    const file = fileInput?.files?.[0];
    if (!file) {
      showToast("No file", "Please select a CSV file first.", "error");
      return;
    }

    const featureCheckboxes = document.querySelectorAll(".feature-checkbox:checked");
    const featureCols = Array.from(featureCheckboxes).map(cb => cb.value);
    const targetCol = document.getElementById("upload-target-select")?.value;

    if (featureCols.length === 0) {
      showToast("No features", "Select at least one feature column.", "error");
      return;
    }
    if (!targetCol) {
      showToast("No target", "Select a target column.", "error");
      return;
    }
    if (featureCols.includes(targetCol)) {
      showToast("Column conflict", "Target column cannot also be a feature.", "error");
      return;
    }

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;display:inline-block;margin-right:6px"></span>Uploading...`;

    try {
      const datasetName = document.getElementById("upload-dataset-name")?.value || "";
      const domainDesc = document.getElementById("upload-domain-desc")?.value || "";
      const result = await uploadCsv(file, featureCols, targetCol, datasetName, domainDesc);
      _currentSessionId = result.session_id;
      setSession(_currentSessionId);
      _showProgressSection();
      _startPolling(_currentSessionId);
    } catch (err) {
      showToast("Upload failed", err.message, "error");
      btn.disabled = false;
      btn.innerHTML = `Start Training →`;
    }
  });
}

// ── Progress Stepper ──────────────────────────────────────────────────────────

const STEP_LABELS = [
  "",                               // 0 — unused
  "Uploading and validating data",  // 1
  "Preprocessing features",         // 2
  "Applying SMOTE balancing",       // 3
  "Training Logistic Regression",   // 4
  "Training Support Vector Machine",// 5
  "Training Decision Tree",         // 6
  "Training Neural Network",        // 7
  "Evaluating all 4 models",        // 8
  "✓ All models ready",             // 9
];

function _showProgressSection() {
  document.getElementById("upload-drop-zone").closest(".upload-drop-area")?.classList.add("hidden");
  document.getElementById("upload-mapper-section")?.classList.add("hidden");
  document.getElementById("upload-demo-btn")?.classList.add("hidden");
  document.getElementById("upload-train-btn")?.classList.add("hidden");

  const progressSection = document.getElementById("upload-progress-section");
  progressSection.classList.remove("hidden");

  // Build stepper
  const stepper = document.getElementById("upload-stepper");
  stepper.innerHTML = STEP_LABELS.slice(1).map((label, i) => `
    <div class="step-item pending" id="step-item-${i + 1}">
      <div class="step-icon">${i + 1}</div>
      <div class="step-message">${label}</div>
    </div>
  `).join("");
}

function _startPolling(sessionId) {
  if (_pollTimer) clearInterval(_pollTimer);

  _pollTimer = setInterval(async () => {
    try {
      const status = await getTrainingStatus(sessionId);
      _updateStepper(status);

      if (status.done) {
        clearInterval(_pollTimer);
        _pollTimer = null;

        if (status.error) {
          showToast("Training error", status.error, "error");
        } else {
          showToast("Training complete", "All 4 models are ready!", "success");
          setTimeout(() => {
            if (_onTrainingComplete) _onTrainingComplete(sessionId);
          }, 800);
        }
      }
    } catch (err) {
      // Polling failure is non-fatal — keep trying
    }
  }, 1200); // Poll every 1200ms per spec
}

function _updateStepper(status) {
  const { step, message, done, error } = status;

  for (let i = 1; i <= 9; i++) {
    const el = document.getElementById(`step-item-${i}`);
    if (!el) continue;

    el.className = "step-item";
    const icon = el.querySelector(".step-icon");
    const msg = el.querySelector(".step-message");

    if (i < step) {
      el.classList.add("done");
      icon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>`;
    } else if (i === step) {
      if (error) {
        el.classList.add("error");
        icon.innerHTML = `✗`;
        msg.textContent = message || "Error occurred";
      } else if (done && i === 9) {
        el.classList.add("done");
        icon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/></svg>`;
        msg.textContent = message;
      } else {
        el.classList.add("active");
        icon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>`;
        if (message) msg.textContent = message;
      }
    } else {
      el.classList.add("pending");
    }
  }
}

function _esc(str) {
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export function stopPolling() {
  if (_pollTimer) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}

export function resetUploadSection() {
  stopPolling();

  // Reset drop zone
  const zone = document.getElementById("upload-drop-zone");
  if (zone) {
    zone.innerHTML = `
      <svg class="cloud-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round"
          d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.338-2.32 5.75 5.75 0 011.548 11.095"/>
      </svg>
      <p class="drop-zone-title">Drop your CSV file here</p>
      <p class="drop-zone-sub">or click to browse — supports .csv files up to 50MB</p>
    `;
  }

  document.getElementById("upload-drop-zone")?.closest(".upload-drop-area")?.classList.remove("hidden");
  document.getElementById("upload-mapper-section")?.classList.add("hidden");
  document.getElementById("upload-train-btn")?.classList.add("hidden");
  document.getElementById("upload-demo-btn")?.classList.remove("hidden");
  document.getElementById("upload-progress-section")?.classList.add("hidden");

  const trainBtn = document.getElementById("upload-train-btn");
  if (trainBtn) {
    trainBtn.disabled = false;
    trainBtn.innerHTML = "Start Training →";
  }
  const demoBtn = document.getElementById("upload-demo-btn");
  if (demoBtn) {
    demoBtn.disabled = false;
    demoBtn.innerHTML = "Try with demo dataset";
  }

  const fileInput = document.getElementById("upload-file-input");
  if (fileInput) fileInput.value = "";

  _parsedHeaders = [];
}
