import { getSession } from "./session.js";
import { getBlacklist } from "./api.js";
import { showToast } from "./app.js";

export async function init() {
  const navBtn = document.getElementById("nav-blacklist");
  if (navBtn) {
    navBtn.addEventListener("click", _loadBlacklist);
  }
  await _loadBlacklist();
}

async function _loadBlacklist() {
  const sessionId = getSession();
  if (!sessionId) return;
  
  try {
    const data = await getBlacklist(sessionId);
    const blacklist = data.blacklist || [];
    
    const tbody = document.getElementById("blacklist-tbody");
    if (!tbody) return;
    
    tbody.innerHTML = "";
    
    if (blacklist.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;padding:24px;color:var(--txt3)">No blacklisted accounts.</td></tr>`;
      return;
    }
    
    // Sort so newest are first
    blacklist.sort((a,b) => new Date(b.added_at) - new Date(a.added_at));
    
    blacklist.forEach(acc => {
      const tr = document.createElement("tr");
      const dateStr = acc.added_at ? new Date(acc.added_at).toLocaleString() : "Unknown";
      
      const safeReason = (acc.reason || "No reason specified").replace(/'/g, "\\'").replace(/"/g, '&quot;');
      
      tr.innerHTML = `
        <td class="mono" style="color:var(--accent)">${acc.cc_num}</td>
        <td style="font-weight:600">${acc.name}</td>
        <td style="font-size:12px;color:var(--txt3)">${dateStr}</td>
        <td>
           <button class="btn btn-sm" style="font-size:11px" onclick="alert('${safeReason}')">View Details</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
    
  } catch (err) {
    showToast("Error loading blacklist", err.message, "error");
  }
}
