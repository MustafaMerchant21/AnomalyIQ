/**
 * charts.js — AnomalyIQ Chart.js Chart Rendering
 *
 * ROC Curve, PR Curve, Feature Importance, Class Distribution Doughnut.
 * Confusion matrix is rendered as HTML grid (not a Chart.js chart).
 * All charts destroyed before re-render to prevent canvas conflicts.
 */

// Registry of active chart instances to prevent memory leaks
const _charts = {};

// ── Shared base config ─────────────────────────────────────────────────────────

function baseChartOptions(extraOptions = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600, easing: "easeInOutQuart" },
    plugins: {
      legend: {
        labels: {
          color: "#A89ED6",
          font: { family: "'Inter', sans-serif", size: 11 },
          padding: 16,
          usePointStyle: true,
          pointStyleWidth: 10,
        },
      },
      tooltip: {
        backgroundColor: "#1C1930",
        borderColor: "#3D3660",
        borderWidth: 1,
        titleColor: "#EDE9FF",
        bodyColor: "#A89ED6",
        padding: 10,
        cornerRadius: 8,
        displayColors: true,
      },
    },
    scales: {
      x: {
        grid: { color: "#2A2540", drawBorder: false },
        ticks: {
          color: "#6B6492",
          font: { family: "'Inter', sans-serif", size: 11 },
        },
      },
      y: {
        grid: { color: "#2A2540", drawBorder: false },
        ticks: {
          color: "#6B6492",
          font: { family: "'Inter', sans-serif", size: 11 },
        },
      },
    },
    ...extraOptions,
  };
}

// Model colors per spec
const MODEL_COLORS = {
  logistic_regression: "#9C6FFF",
  svm:                 "#69FCFF",
  decision_tree:       "#F5A623",
  neural_network:      "#FF4081",
};

const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  svm:                 "SVM",
  decision_tree:       "Decision Tree",
  neural_network:      "Neural Network",
};

function _destroyChart(id) {
  if (_charts[id]) {
    _charts[id].destroy();
    delete _charts[id];
  }
}

// ── ROC Curve ─────────────────────────────────────────────────────────────────

export function renderRocCurve(canvasId, rocData) {
  _destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const datasets = [];

  // Diagonal reference line
  datasets.push({
    label: "Random Classifier",
    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
    borderColor: "rgba(107,100,146,0.4)",
    borderDash: [5, 5],
    borderWidth: 1,
    pointRadius: 0,
    fill: false,
    tension: 0,
  });

  for (const [modelName, curves] of Object.entries(rocData)) {
    const fpr = curves.fpr || [];
    const tpr = curves.tpr || [];
    if (!fpr.length) continue;

    datasets.push({
      label: MODEL_LABELS[modelName] || modelName,
      data: fpr.map((x, i) => ({ x, y: tpr[i] })),
      borderColor: MODEL_COLORS[modelName] || "#9C6FFF",
      backgroundColor: "transparent",
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.1,
    });
  }

  const opts = baseChartOptions({
    plugins: {
      ...baseChartOptions().plugins,
      legend: { ...baseChartOptions().plugins.legend, position: "top" },
    },
    scales: {
      x: {
        type: "linear",
        ...baseChartOptions().scales.x,
        title: {
          display: true,
          text: "False Positive Rate",
          color: "#6B6492",
          font: { size: 11 },
        },
        min: 0, max: 1,
      },
      y: {
        ...baseChartOptions().scales.y,
        title: {
          display: true,
          text: "True Positive Rate",
          color: "#6B6492",
          font: { size: 11 },
        },
        min: 0, max: 1,
      },
    },
  });

  _charts[canvasId] = new Chart(canvas, {
    type: "line",
    data: { datasets },
    options: opts,
  });
}

// ── PR Curve ──────────────────────────────────────────────────────────────────

export function renderPrCurve(canvasId, prData) {
  _destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const datasets = [];

  for (const [modelName, curves] of Object.entries(prData)) {
    const precision = curves.precision || [];
    const recall = curves.recall || [];
    if (!precision.length) continue;

    datasets.push({
      label: MODEL_LABELS[modelName] || modelName,
      data: recall.map((r, i) => ({ x: r, y: precision[i] })),
      borderColor: MODEL_COLORS[modelName] || "#9C6FFF",
      backgroundColor: "transparent",
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.1,
    });
  }

  const opts = baseChartOptions({
    plugins: {
      ...baseChartOptions().plugins,
      legend: { ...baseChartOptions().plugins.legend, position: "top" },
    },
    scales: {
      x: {
        type: "linear",
        ...baseChartOptions().scales.x,
        title: {
          display: true,
          text: "Recall",
          color: "#6B6492",
          font: { size: 11 },
        },
        min: 0, max: 1,
      },
      y: {
        ...baseChartOptions().scales.y,
        title: {
          display: true,
          text: "Precision",
          color: "#6B6492",
          font: { size: 11 },
        },
        min: 0, max: 1,
      },
    },
  });

  _charts[canvasId] = new Chart(canvas, {
    type: "line",
    data: { datasets },
    options: opts,
  });
}

// ── Feature Importance ─────────────────────────────────────────────────────────

export function renderFeatureImportance(canvasId, importances, maxFeatures = 12) {
  _destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  if (!importances || Object.keys(importances).length === 0) return;

  // Sort descending, take top N
  const sorted = Object.entries(importances)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxFeatures);

  const labels = sorted.map(([k]) => k);
  const values = sorted.map(([, v]) => v);

  _charts[canvasId] = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Importance",
        data: values,
        backgroundColor: values.map((_, i) => {
          const ratio = i / Math.max(values.length - 1, 1);
          return `rgba(${Math.round(107 + ratio * 49)}, ${Math.round(77 + ratio * 113)}, ${Math.round(184 + ratio * 71)}, 0.8)`;
        }),
        borderColor: "rgba(156,111,255,0.5)",
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      ...baseChartOptions(),
      indexAxis: "y",
      plugins: {
        ...baseChartOptions().plugins,
        legend: { display: false },
      },
      scales: {
        x: {
          ...baseChartOptions().scales.x,
          min: 0,
          ticks: {
            ...baseChartOptions().scales.x.ticks,
            callback: v => v.toFixed(3),
          },
        },
        y: {
          ...baseChartOptions().scales.y,
          ticks: {
            ...baseChartOptions().scales.y.ticks,
            font: {
              family: "'JetBrains Mono', monospace",
              size: 10,
            },
            color: "#9C6FFF",
          },
        },
      },
    },
  });
}

// ── Class Distribution Doughnut ────────────────────────────────────────────────

export function renderClassDoughnut(canvasId, class0Count, class1Count) {
  _destroyChart(canvasId);
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const total = class0Count + class1Count;
  const pct0 = total > 0 ? ((class0Count / total) * 100).toFixed(1) : 0;
  const pct1 = total > 0 ? ((class1Count / total) * 100).toFixed(1) : 0;

  _charts[canvasId] = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: [`Normal (${pct0}%)`, `Fraud (${pct1}%)`],
      datasets: [{
        data: [class0Count, class1Count],
        backgroundColor: [
          "rgba(105,252,255,0.7)",
          "rgba(255,64,129,0.7)",
        ],
        borderColor: [
          "rgba(105,252,255,1)",
          "rgba(255,64,129,1)",
        ],
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "70%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#A89ED6",
            font: { family: "'Inter', sans-serif", size: 11 },
            padding: 16,
            usePointStyle: true,
          },
        },
        tooltip: {
          backgroundColor: "#1C1930",
          borderColor: "#3D3660",
          borderWidth: 1,
          titleColor: "#EDE9FF",
          bodyColor: "#A89ED6",
          padding: 10,
          cornerRadius: 8,
        },
      },
    },
  });
}

// ── Confusion Matrix HTML Grid ────────────────────────────────────────────────
// NOT a Chart.js chart — rendered as styled HTML

export function renderConfusionMatrix(containerId, cm) {
  const container = document.getElementById(containerId);
  if (!container || !cm) return;

  const { tn = 0, fp = 0, fn = 0, tp = 0 } = cm;

  container.innerHTML = `
    <div class="confusion-matrix">
      <div class="cm-cell cm-tp">
        <div class="cm-cell-label">True Positive</div>
        <div class="cm-cell-val">${tp}</div>
      </div>
      <div class="cm-cell cm-fp">
        <div class="cm-cell-label">False Positive</div>
        <div class="cm-cell-val">${fp}</div>
      </div>
      <div class="cm-cell cm-fn">
        <div class="cm-cell-label">False Negative</div>
        <div class="cm-cell-val">${fn}</div>
      </div>
      <div class="cm-cell cm-tn">
        <div class="cm-cell-label">True Negative</div>
        <div class="cm-cell-val">${tn}</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:8px;font-size:10px;color:var(--txt3);text-align:center">
      <span>Predicted FRAUD</span>
      <span>Predicted NORMAL</span>
    </div>
  `;
}

// ── Probability bars (for simulator/predictor) ─────────────────────────────────

export function renderProbabilityBars(containerId, probabilities) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const modelOrder = ["logistic_regression", "svm", "decision_tree", "neural_network"];

  container.innerHTML = modelOrder.map(name => {
    const prob = probabilities[name];
    if (prob === null || prob === undefined) return "";
    const pct = (prob * 100).toFixed(1);
    const color = prob >= 0.5 ? "var(--anomaly)" : "var(--normal)";
    const label = MODEL_LABELS[name] || name;

    return `
      <div style="margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <span style="font-size:11px;color:var(--txt2)">${label}</span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:${color}">${pct}%</span>
        </div>
        <div class="progress-bar-wrapper">
          <div class="progress-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
      </div>
    `;
  }).join("");
}

export function destroyAll() {
  Object.keys(_charts).forEach(id => {
    if (_charts[id]) _charts[id].destroy();
  });
  Object.keys(_charts).forEach(k => delete _charts[k]);
}
