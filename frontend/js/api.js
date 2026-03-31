/**
 * api.js — AnomalyIQ API Client
 *
 * Thin fetch() wrappers for all backend endpoints.
 * All functions are async and return parsed JSON.
 * exportCsv returns a URL string for direct download.
 */

export const API_BASE = "http://localhost:8000";

// ── Core fetch helper ─────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let errMsg = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      errMsg = body.detail || body.message || errMsg;
    } catch (_) {}
    throw new Error(errMsg);
  }

  // Some endpoints return non-JSON (CSV download)
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return res.json();
  }
  return res;
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function getHealth() {
  return apiFetch("/api/health");
}

// ── Upload ────────────────────────────────────────────────────────────────────

export async function uploadCsv(file, featureCols, targetCol, datasetName = "", domainDescription = "") {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("feature_cols", featureCols.join(","));
  formData.append("target_col", targetCol);
  formData.append("dataset_name", datasetName.trim() || "Unnamed Dataset");
  formData.append("domain_description", domainDescription.trim() || "Anomaly detection");

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let err = `HTTP ${res.status}`;
    try {
      const b = await res.json();
      err = b.detail || err;
    } catch (_) {}
    throw new Error(err);
  }
  return res.json();
}

export async function uploadDemo() {
  return apiFetch("/api/upload_demo", { method: "POST" });
}

// ── Training Status ────────────────────────────────────────────────────────────

export async function getTrainingStatus(sessionId) {
  return apiFetch(`/api/training_status?session_id=${encodeURIComponent(sessionId)}`);
}

// ── Metrics & Charts ─────────────────────────────────────────────────────────

export async function getMetrics(sessionId) {
  return apiFetch(`/api/metrics?session_id=${encodeURIComponent(sessionId)}`);
}

export async function getConfusionMatrix(sessionId, model = "logistic_regression") {
  return apiFetch(
    `/api/confusion_matrix?session_id=${encodeURIComponent(sessionId)}&model=${encodeURIComponent(model)}`
  );
}

export async function getRocCurve(sessionId) {
  return apiFetch(`/api/roc_curve?session_id=${encodeURIComponent(sessionId)}`);
}

export async function getPrCurve(sessionId) {
  return apiFetch(`/api/pr_curve?session_id=${encodeURIComponent(sessionId)}`);
}

// ── Column Config & Feature Info ──────────────────────────────────────────────

export async function getColumnConfig(sessionId) {
  return apiFetch(`/api/column_config?session_id=${encodeURIComponent(sessionId)}`);
}

export async function getFeatureImportance(sessionId) {
  return apiFetch(`/api/feature_importance?session_id=${encodeURIComponent(sessionId)}`);
}

// ── Insight Report ─────────────────────────────────────────────────────────────

export async function getInsightReport(sessionId) {
  return apiFetch(`/api/insight_report?session_id=${encodeURIComponent(sessionId)}`);
}

// ── Prediction & Scoring ──────────────────────────────────────────────────────

export async function predict(sessionId, features) {
  return apiFetch("/api/predict", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, features }),
  });
}

export async function getRiskScore(sessionId, features) {
  return apiFetch("/api/risk_score", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, features }),
  });
}

export async function explain(sessionId, features, mlResult, ruleResult, graphResult) {
  return apiFetch("/api/explain", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      features,
      ml_result: mlResult || null,
      rule_result: ruleResult || null,
      graph_result: graphResult || null,
    }),
  });
}

// ── Simulation ────────────────────────────────────────────────────────────────

export async function simulate(sessionId) {
  return apiFetch(`/api/simulate?session_id=${encodeURIComponent(sessionId)}`);
}

// ── Globe ─────────────────────────────────────────────────────────────────────

export async function getGlobeData(sessionId) {
  return apiFetch(`/api/globe_data?session_id=${encodeURIComponent(sessionId)}`);
}

export async function injectPoint(sessionId, features) {
  return apiFetch("/api/inject_point", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, features }),
  });
}

// ── Leaderboard ───────────────────────────────────────────────────────────────

export async function getLeaderboard(sessionId) {
  return apiFetch(`/api/leaderboard?session_id=${encodeURIComponent(sessionId)}`);
}

// ── Cases ─────────────────────────────────────────────────────────────────────

export async function getCaseList(sessionId) {
  return apiFetch(`/api/case_list?session_id=${encodeURIComponent(sessionId)}`);
}

export async function saveCase(sessionId, transactionId, decision, note, transactionData, riskScore, verdict) {
  return apiFetch("/api/case_save", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      transaction_id: transactionId,
      decision,
      note: note || null,
      transaction_data: transactionData || null,
      risk_score: riskScore || null,
      verdict: verdict || null,
    }),
  });
}

// ── Test Dataset (Unlabelled Prediction) ─────────────────────────────────────

/**
 * testDataset — upload an unlabelled CSV and get back predictions.
 * @param {string} sessionId
 * @param {File} file  — .csv file without the target column
 * @returns {Promise<{normal: [], fraud: [], total: number, fraud_count: number, normal_count: number, feature_cols: string[], model_used: string}>}
 */
export async function testDataset(sessionId, file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);

  const res = await fetch(`${API_BASE}/api/test_dataset`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let err = `HTTP ${res.status}`;
    try {
      const b = await res.json();
      err = b.detail || err;
    } catch (_) {}
    throw new Error(err);
  }
  return res.json();
}

// ── Export ────────────────────────────────────────────────────────────────────

/**
 * exportCsv returns a URL string for direct download (not a fetch call).
 * Use with an anchor tag or window.open().
 */
export function exportCsv(sessionId) {
  return `${API_BASE}/api/export_csv?session_id=${encodeURIComponent(sessionId)}`;
}


// ── Investigation & Blacklist ──────────────────────────────────────────────────

export async function getFraudRings(sessionId) {
  return apiFetch(`/api/fraud_rings?session_id=${encodeURIComponent(sessionId)}`);
}

export async function addToBlacklist(sessionId, accounts) {
  return apiFetch("/api/blacklist", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, accounts }),
  });
}

export async function getBlacklist(sessionId) {
  return apiFetch(`/api/blacklist?session_id=${encodeURIComponent(sessionId)}`);
}
