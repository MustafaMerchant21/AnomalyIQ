/**
 * globe.js — AnomalyIQ Three.js 3D Scatter Globe
 *
 * THREE.WebGLRenderer, PerspectiveCamera, OrbitControls
 * Reference wireframe sphere, anomaly points (red glow), normal points (cyan),
 * injected points (purple glow), auto-rotate, show-only-anomalies toggle,
 * inject point panel.
 */

import { getGlobeData, injectPoint, getColumnConfig } from "./api.js";
import { getSession } from "./session.js";
import { showToast } from "./app.js";

let _initialized = false;
let _scene, _camera, _renderer, _controls;
let _animationId = null;
let _isDragging = false;
let _normalGroup, _anomalyGroup, _injectedGroup;
let _showOnlyAnomalies = false;
let _globeData = null;
let _colConfig = null;
let _labelRenderer = null;

const ANOMALY_COLOR  = 0xFF4081;
const NORMAL_COLOR   = 0x69FCFF;
const INJECTED_COLOR = 0x9C6FFF;
const BG_COLOR       = 0x0C0A14;

// ── Init ──────────────────────────────────────────────────────────────────────

export async function init() {
  if (_initialized) return;
  _initialized = true;

  _setupThreeJs();
  _setupControls();
  _setupButtons();

  await _loadData();
}

export function reset() {
  if (_animationId) {
    cancelAnimationFrame(_animationId);
    _animationId = null;
  }
  if (_renderer) {
    _renderer.dispose();
    const canvas = document.getElementById("globe-canvas");
    if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
  }
  _initialized = false;
  _scene = _camera = _renderer = _controls = null;
  _normalGroup = _anomalyGroup = _injectedGroup = null;
  _globeData = null;
}

// ── Three.js Bootstrap ────────────────────────────────────────────────────────

function _setupThreeJs() {
  const container = document.getElementById("globe-container");
  if (!container) return;

  // Renderer
  _renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
  });
  _renderer.setPixelRatio(window.devicePixelRatio);
  _renderer.setClearColor(BG_COLOR, 1);
  _renderer.setSize(container.clientWidth, container.clientHeight);

  // Create canvas
  const canvas = _renderer.domElement;
  canvas.id = "globe-canvas";
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  container.appendChild(canvas);

  // Camera
  _camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
  _camera.position.set(0, 0, 12);

  // Scene
  _scene = new THREE.Scene();

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.4);
  _scene.add(ambient);

  const dirLight = new THREE.DirectionalLight(0x9C6FFF, 0.8);
  dirLight.position.set(5, 5, 5);
  _scene.add(dirLight);

  // Reference wireframe sphere
  const sphereGeo = new THREE.SphereGeometry(5, 24, 18);
  const sphereMat = new THREE.MeshBasicMaterial({
    color: 0x3D3660,
    wireframe: true,
    transparent: true,
    opacity: 0.18,
  });
  const sphere = new THREE.Mesh(sphereGeo, sphereMat);
  _scene.add(sphere);

  // Point groups
  _normalGroup   = new THREE.Group();
  _anomalyGroup  = new THREE.Group();
  _injectedGroup = new THREE.Group();
  _scene.add(_normalGroup, _anomalyGroup, _injectedGroup);

  // Animation loop
  _animate();

  // Resize handler
  window.addEventListener("resize", _onResize);
}

function _animate() {
  _animationId = requestAnimationFrame(_animate);

  // Auto-rotate y when not dragging
  if (!_isDragging && _scene) {
    _scene.rotation.y += 0.002;
  }

  if (_controls) _controls.update();
  if (_renderer && _scene && _camera) {
    _renderer.render(_scene, _camera);
  }
}

function _onResize() {
  const container = document.getElementById("globe-container");
  if (!container || !_camera || !_renderer) return;

  _camera.aspect = container.clientWidth / container.clientHeight;
  _camera.updateProjectionMatrix();
  _renderer.setSize(container.clientWidth, container.clientHeight);
}

// ── OrbitControls ──────────────────────────────────────────────────────────────

function _setupControls() {
  setTimeout(() => {
    if (typeof THREE.OrbitControls === "undefined") return;

    const canvas = _renderer?.domElement;
    if (!canvas) return;

    _controls = new THREE.OrbitControls(_camera, canvas);
    _controls.enableDamping = true;
    _controls.dampingFactor = 0.05;
    _controls.rotateSpeed = 0.8;
    _controls.zoomSpeed = 0.8;
    _controls.minDistance = 6;
    _controls.maxDistance = 25;

    _controls.addEventListener("start", () => { _isDragging = true; });
    _controls.addEventListener("end",   () => { _isDragging = false; });
  }, 200);
}

// ── Load Globe Data ────────────────────────────────────────────────────────────

async function _loadData() {
  const sessionId = getSession();
  if (!sessionId) return;

  _setLoading(true);

  try {
    [_globeData, _colConfig] = await Promise.all([
      getGlobeData(sessionId),
      getColumnConfig(sessionId).catch(() => null),
    ]);

    _renderPoints(_globeData.points || []);
    _updateInfoOverlay(_globeData);
  } catch (err) {
    showToast("Globe error", err.message, "error");
  } finally {
    _setLoading(false);
  }
}

// ── Render Points ─────────────────────────────────────────────────────────────

function _renderPoints(points) {
  // Clear existing
  [_normalGroup, _anomalyGroup].forEach(g => {
    while (g.children.length) g.remove(g.children[0]);
  });

  const normalMat = new THREE.MeshStandardMaterial({
    color: NORMAL_COLOR,
    emissive: 0x003B3B,
    emissiveIntensity: 0.3,
    roughness: 0.4,
    metalness: 0.2,
  });

  const anomalyMat = new THREE.MeshStandardMaterial({
    color: ANOMALY_COLOR,
    emissive: 0x440020,
    emissiveIntensity: 0.5,
    roughness: 0.3,
    metalness: 0.3,
  });

  const sphereGeo = new THREE.SphereGeometry(0.08, 8, 6);

  for (const pt of points) {
    const isAnomaly = pt.true_label === 1;
    const mesh = new THREE.Mesh(sphereGeo, isAnomaly ? anomalyMat : normalMat);
    mesh.position.set(pt.x, pt.y, pt.z);
    mesh.userData = pt;

    if (isAnomaly) {
      _anomalyGroup.add(mesh);
    } else {
      _normalGroup.add(mesh);
    }
  }
}

// ── Toggle Anomalies Only ─────────────────────────────────────────────────────

function _toggleAnomaliesOnly() {
  _showOnlyAnomalies = !_showOnlyAnomalies;
  if (_normalGroup) {
    _normalGroup.visible = !_showOnlyAnomalies;
  }

  const btn = document.getElementById("globe-toggle-anomalies");
  if (btn) {
    btn.classList.toggle("active", _showOnlyAnomalies);
    btn.innerHTML = _showOnlyAnomalies
      ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg> Show All`
      : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29"/></svg> Anomalies Only`;
  }
}

// ── Inject Point ──────────────────────────────────────────────────────────────

async function _handleInjectPoint() {
  const sessionId = getSession();
  if (!sessionId || !_colConfig) {
    showToast("No session", "Train a model first.", "error");
    return;
  }

  // Read values from inject form if available, else use midpoints
  const features = {};
  const featureCols = _colConfig.feature_cols || [];
  const colStats = _colConfig.col_stats || {};

  for (const feat of featureCols) {
    const input = document.getElementById(`inject-${feat.replace(/[^a-zA-Z0-9]/g, "_")}`);
    const stats = colStats[feat] || { mean: 0.5 };
    features[feat] = input ? parseFloat(input.value) || stats.mean : stats.mean;
  }

  try {
    const result = await injectPoint(sessionId, features);
    _addInjectedPoint(result);

    showToast(
      "Point injected",
      `Verdict: ${result.verdict?.replace(/_/g, " ")} (${(result.anomaly_score * 100).toFixed(0)}%)`,
      result.verdict === "HIGH_RISK" ? "error" : "success"
    );

    // Close panel
    document.getElementById("globe-inject-panel")?.classList.add("hidden");
  } catch (err) {
    showToast("Inject failed", err.message, "error");
  }
}

function _addInjectedPoint(result) {
  const { x, y, z, verdict, anomaly_score } = result;

  const geo = new THREE.SphereGeometry(0.18, 10, 8);
  const mat = new THREE.MeshStandardMaterial({
    color: INJECTED_COLOR,
    emissive: 0x3D1A90,
    emissiveIntensity: 0.8 + Math.sin(Date.now() * 0.005) * 0.2,
    roughness: 0.2,
    metalness: 0.5,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, y, z);
  mesh.userData = { verdict, anomaly_score, injected: true };

  _injectedGroup.add(mesh);

  // Pulse animation
  let t = 0;
  const animatePulse = () => {
    t += 0.05;
    const s = 1 + Math.sin(t) * 0.15;
    mesh.scale.setScalar(s);
    if (_injectedGroup.children.includes(mesh)) {
      requestAnimationFrame(animatePulse);
    }
  };
  animatePulse();
}

// ── Buttons ───────────────────────────────────────────────────────────────────

function _setupButtons() {
  // Toggle anomalies only
  document.getElementById("globe-toggle-anomalies")?.addEventListener("click", _toggleAnomaliesOnly);

  // Reset camera
  document.getElementById("globe-reset-camera")?.addEventListener("click", () => {
    if (_camera) _camera.position.set(0, 0, 12);
    if (_controls) _controls.reset();
  });

  // Open inject panel
  document.getElementById("globe-inject-btn")?.addEventListener("click", () => {
    _buildInjectForm();
    document.getElementById("globe-inject-panel")?.classList.remove("hidden");
  });

  // Close inject panel
  document.getElementById("globe-inject-close")?.addEventListener("click", () => {
    document.getElementById("globe-inject-panel")?.classList.add("hidden");
  });

  // Submit inject
  document.getElementById("globe-inject-submit")?.addEventListener("click", _handleInjectPoint);

  // Clear injected points
  document.getElementById("globe-clear-injected")?.addEventListener("click", () => {
    while (_injectedGroup?.children.length) {
      _injectedGroup.remove(_injectedGroup.children[0]);
    }
  });
}

function _buildInjectForm() {
  const container = document.getElementById("globe-inject-form");
  if (!container || !_colConfig) return;

  const featureCols = _colConfig.feature_cols || [];
  const colStats = _colConfig.col_stats || {};

  container.innerHTML = featureCols.map(feat => {
    const stats = colStats[feat] || { min: 0, max: 1, mean: 0.5 };
    const safeId = feat.replace(/[^a-zA-Z0-9]/g, "_");
    return `
      <div class="form-group" style="margin-bottom:8px">
        <label style="font-size:10px;color:var(--txt3)">${feat}</label>
        <input type="number" id="inject-${safeId}" step="any"
          value="${stats.mean.toFixed(4)}"
          min="${stats.min}" max="${stats.max}"
          style="padding:5px 8px;font-size:12px">
      </div>
    `;
  }).join("");
}

// ── Info Overlay ──────────────────────────────────────────────────────────────

function _updateInfoOverlay(data) {
  const el = document.getElementById("globe-info-overlay");
  if (!el || !data) return;

  const variance = data.variance_explained || [];
  const totalVar = data.total_variance || 0;
  const nPoints = data.n_points || 0;

  el.innerHTML = `
    <div style="font-size:11px;color:var(--txt2);margin-bottom:6px">
      <strong style="color:var(--txt)">${nPoints}</strong> transactions
    </div>
    <div style="font-size:10px;color:var(--txt3)">
      PC1: ${((variance[0] || 0) * 100).toFixed(1)}% variance<br>
      PC2: ${((variance[1] || 0) * 100).toFixed(1)}% variance<br>
      PC3: ${((variance[2] || 0) * 100).toFixed(1)}% variance<br>
      <strong style="color:var(--accent)">Total: ${(totalVar * 100).toFixed(1)}%</strong>
    </div>
  `;
}

// ── Loading ───────────────────────────────────────────────────────────────────

function _setLoading(loading) {
  const overlay = document.getElementById("globe-loading");
  if (overlay) overlay.classList.toggle("hidden", !loading);
}

// Error catch for WebGL context loss
window.addEventListener("webglcontextlost", e => {
  e.preventDefault();
  const container = document.getElementById("globe-container");
  if (container) {
    container.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:center;height:100%;background:var(--bg)">
        <div style="text-align:center;color:var(--txt3)">
          <div style="font-size:32px;margin-bottom:16px">⚠</div>
          <div>WebGL context lost. Please refresh the page.</div>
        </div>
      </div>
    `;
  }
});
