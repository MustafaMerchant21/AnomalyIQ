import { getSession } from "./session.js";
import { getFraudRings, addToBlacklist } from "./api.js";
import { showToast } from "./app.js";

let _network = null;
let _currentRings = [];
let _activeRingId = null;
let _pollTimer = null;

export async function init() {
  document.getElementById("investigation-refresh-btn")?.addEventListener("click", _loadRings);
  document.getElementById("investigation-dismiss-btn")?.addEventListener("click", _handleDismiss);
  document.getElementById("investigation-block-btn")?.addEventListener("click", _handleBlock);
  
  // Modal listeners
  document.getElementById("inv-modal-close")?.addEventListener("click", _closeModal);
  document.getElementById("inv-modal-overlay")?.addEventListener("click", _closeModal);
  
  // Attempt initial load
  await _loadRings();
}

async function _loadRings() {
  const sessionId = getSession();
  if (!sessionId) return;
  
  const listEl = document.getElementById("investigation-ring-list");
  const emptyEl = document.getElementById("investigation-list-empty");
  if (!listEl) return;
  
  listEl.innerHTML = "";
  if (emptyEl) listEl.appendChild(emptyEl);
  
  // Stop any existing poll
  if (_pollTimer) clearInterval(_pollTimer);
  
  try {
    const data = await getFraudRings(sessionId);
    _currentRings = data.rings || [];
    
    if (_currentRings.length === 0) {
      if (emptyEl) emptyEl.style.display = "block";
      return;
    }
    
    if (emptyEl) emptyEl.style.display = "none";
    
    _currentRings.forEach(ring => {
      const el = document.createElement("div");
      el.className = "card";
      el.style.marginBottom = "var(--sp-3)";
      el.style.cursor = "pointer";
      el.style.transition = "border-color 0.2s, background 0.2s";
      el.dataset.id = ring.cluster_id;
      
      const hubLabel = ring.hub_count > 0 ? `<span style="color:var(--warn);margin-left:8px">★ Hubs</span>` : '';
      
      el.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center">
          <h4 style="margin:0;font-size:14px">${ring.cluster_id}</h4>
          <span class="badge ${ring.fraud_count > 0 ? 'badge-anomaly' : 'badge-muted'}">${ring.fraud_count}/${ring.transaction_count} Fraud</span>
        </div>
        <div style="font-size:12px;color:var(--txt3);margin-top:8px">
          ${ring.transactions.length} Accounts ${hubLabel}
        </div>
      `;
      
      el.addEventListener("click", () => _selectRing(ring.cluster_id));
      listEl.appendChild(el);
    });
    
    // Start polling if any clusters still have a pending AI summary
    _startSummaryPoll(sessionId);
    
  } catch (err) {
    showToast("Error loading fraud rings", err.message, "error");
  }
}

/** Poll every 8s until all visible ring summaries are resolved. */
async function _startSummaryPoll(sessionId) {
  const hasPending = () => _currentRings.some(r => !r.summary || r.summary === "Pending AI analysis...");
  
  if (!hasPending()) return; // Nothing to poll
  
  _pollTimer = setInterval(async () => {
    try {
      const data = await getFraudRings(sessionId);
      const fresh = data.rings || [];
      
      fresh.forEach(freshRing => {
        const existing = _currentRings.find(r => r.cluster_id === freshRing.cluster_id);
        if (!existing) return;
        
        // If summary just arrived, update in memory and refresh active panel summary
        if (freshRing.summary && freshRing.summary !== "Pending AI analysis..." &&
            (existing.summary === "Pending AI analysis..." || !existing.summary)) {
          existing.summary = freshRing.summary;
          
          // If this ring is currently selected, update the visible summary text
          if (freshRing.cluster_id === _activeRingId) {
            const summaryEl = document.getElementById("investigation-ai-summary");
            if (summaryEl) summaryEl.innerHTML = freshRing.summary;
          }
        }
      });
      
      // Stop polling once everything is resolved
      if (!hasPending()) {
        clearInterval(_pollTimer);
        _pollTimer = null;
      }
    } catch (_) {
      // Ignore network errors during background poll
    }
  }, 8000); // 8 seconds: respects the 8 req/min rate limit
}

function _selectRing(clusterId) {
  _activeRingId = clusterId;
  
  const listEl = document.getElementById("investigation-ring-list");
  Array.from(listEl.children).forEach(el => {
    if (el.dataset.id === clusterId) {
      el.style.borderColor = "var(--accent)";
      el.style.background = "var(--surface-hover)";
    } else if (el.classList.contains("card")) {
      el.style.borderColor = "";
      el.style.background = "";
    }
  });
  
  const ring = _currentRings.find(r => r.cluster_id === clusterId);
  if (!ring) return;
  
  document.getElementById("investigation-empty-detail")?.classList.add("hidden");
  document.getElementById("investigation-detail-panel")?.classList.remove("hidden");
  
  const summaryEl = document.getElementById("investigation-ai-summary");
  if (summaryEl) summaryEl.innerHTML = ring.summary;
  
  _renderGraph(ring);
  _renderAccountsTable(ring);
}

function _renderAccountsTable(ring) {
  const tbody = document.getElementById("investigation-accounts-tbody");
  const countEl = document.getElementById("investigation-accounts-count");
  if (!tbody) return;
  
  if (countEl) countEl.innerText = ring.transactions.length;
  
  tbody.innerHTML = "";
  ring.transactions.forEach(tx => {
    const tr = document.createElement("tr");
    
    const roleBadge = tx.is_hub ? `<span class="badge badge-warn">Hub</span>` : `<span class="badge badge-muted">Node</span>`;
    const labelBadge = tx.label === 1 
      ? `<span class="badge badge-anomaly">Fraud</span>` 
      : `<span class="badge badge-ok">Normal</span>`;
      
    // Create a safe stringified version of features
    const safeFeatures = encodeURIComponent(JSON.stringify(tx.features || {}));
    const nameStr = (tx.name && tx.name.trim() !== "") ? tx.name : "Unknown Sender";
    
    tr.innerHTML = `
      <td class="mono" style="color:var(--accent)">${tx.cc_num}</td>
      <td style="font-weight:500">${nameStr}</td>
      <td>${roleBadge}</td>
      <td>${labelBadge}</td>
      <td><button class="btn btn-sm" onclick="window.showInvestigationAccountDetails('${tx.cc_num}', '${tx.label}', '${safeFeatures}', ${tx.is_hub})">View Details</button></td>
    `;
    tbody.appendChild(tr);
  });
}

// Global exposure for the inline onclick handler
window.showInvestigationAccountDetails = function(ccNum, label, featuresSafeStr, isHub) {
  let features = {};
  try {
     features = JSON.parse(decodeURIComponent(featuresSafeStr));
  } catch(e) {}
  
  const modal = document.getElementById("inv-modal");
  const overlay = document.getElementById("inv-modal-overlay");
  if (!modal || !overlay) return;
  
  document.getElementById("inv-modal-cc").innerText = ccNum;
  
  const badge = document.getElementById("inv-modal-badge");
  badge.className = "badge";
  if (label == 1) {
    badge.classList.add("badge-anomaly");
    badge.innerText = isHub ? "Fraud (Hub)" : "Fraud";
  } else {
    badge.classList.add("badge-ok");
    badge.innerText = isHub ? "Normal (Hub)" : "Normal";
  }
  
  const tbody = document.getElementById("inv-modal-features-tbody");
  tbody.innerHTML = "";
  
  if (Object.keys(features).length === 0) {
     tbody.innerHTML = `<tr><td colspan="2" style="text-align:center;color:var(--txt3)">No raw features available.</td></tr>`;
  } else {
     Object.keys(features).sort().forEach(k => {
       const tr = document.createElement("tr");
       tr.innerHTML = `
         <td style="font-weight:600;color:var(--txt2)">${k}</td>
         <td class="mono">${features[k]}</td>
       `;
       tbody.appendChild(tr);
     });
  }
  
  modal.classList.remove("hidden");
  overlay.classList.remove("hidden");
};

function _closeModal() {
  document.getElementById("inv-modal")?.classList.add("hidden");
  document.getElementById("inv-modal-overlay")?.classList.add("hidden");
}

function _renderGraph(ring) {
  const container = document.getElementById("investigation-graph-container");
  if (!container || !window.vis) return;
  
  const nodes = [];
  const edges = [];
  
  ring.transactions.forEach(tx => {
    nodes.push({
      id: tx.id,
      label: tx.name.substring(0, 15) || tx.cc_num || tx.id,
      shape: tx.is_hub ? "star" : "dot",
      size: tx.is_hub ? 30 : (tx.label === 1 ? 16 : 10),
      color: tx.label === 1 ? (tx.is_hub ? "#D50000" : "#FF4081") : "#3DDC97",
      font: { color: "#EDE9FF", size: 11 },
      title: `cc_num: ${tx.cc_num}<br>is_hub: ${tx.is_hub}`
    });
  });
  
  ring.shared_entities.forEach(ent => {
    nodes.push({
      id: ent.id,
      label: ent.feature + "\\n" + ent.bucket,
      shape: "box",
      color: "#3D3660",
      font: { color: "#EDE9FF", size: 10 },
      title: `Shared Feature: ${ent.feature}`
    });
    
    // Connect to transactions
    ring.transactions.forEach(tx => {
       edges.push({ 
           from: tx.id, 
           to: ent.id, 
           color: { color: "#6B6492", opacity: 0.4 }, 
           physics: true 
       });
    });
  });

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    nodes: { borderWidth: 2, shadow: true },
    edges: { smooth: { type: "continuous" } },
    physics: {
      barnesHut: { 
        gravitationalConstant: -2000, 
        centralGravity: 0.3, 
        springLength: 95,
        springConstant: 0.04,
        damping: 0.09,
        avoidOverlap: 0.1
      },
      stabilization: {
        enabled: true,
        iterations: 150,
      }
    },
    interaction: { hover: true }
  };
  
  if (_network) _network.destroy();
  _network = new vis.Network(container, data, options);
  
  // Disable physics once stabilization is complete to prevent continuous moving/jittering
  _network.once("stabilizationIterationsDone", function() {
    _network.setOptions({ physics: false });
  });
}

async function _handleBlock() {
  if (!_activeRingId) return;
  const ring = _currentRings.find(r => r.cluster_id === _activeRingId);
  if (!ring) return;
  
  const sessionId = getSession();
  
  const accountsToBlock = ring.transactions.map(tx => ({
    cc_num: tx.cc_num,
    name: tx.name,
    reason: ring.summary
  })).filter(acc => acc.cc_num !== "Unknown Account");
  
  if (accountsToBlock.length === 0) {
     showToast("No Accounts Extracted", "No valid CC numbers found in this cluster.", "warning");
     return;
  }
  
  try {
    await addToBlacklist(sessionId, accountsToBlock);
    showToast("Cluster Blocked", `Moved ${accountsToBlock.length} accounts to the Blacklist.`, "success");
    _removeActiveRingLocally(ring.cluster_id);
    
  } catch (err) {
    showToast("Block failed", err.message, "error");
  }
}

function _handleDismiss() {
  if (!_activeRingId) return;
  showToast("Dismissed", `Cluster ${_activeRingId} ignored.`, "success");
  _removeActiveRingLocally(_activeRingId);
}

function _removeActiveRingLocally(clusterId) {
    _currentRings = _currentRings.filter(r => r.cluster_id !== clusterId);
    _activeRingId = null;
    
    document.getElementById("investigation-detail-panel")?.classList.add("hidden");
    document.getElementById("investigation-empty-detail")?.classList.remove("hidden");
    
    const listEl = document.getElementById("investigation-ring-list");
    Array.from(listEl.children).forEach(el => {
      if (el.dataset.id === clusterId) el.remove();
    });
    
    if (_currentRings.length === 0) {
        document.getElementById("investigation-list-empty").style.display = "block";
    }
}
