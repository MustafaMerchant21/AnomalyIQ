/**
 * session.js — AnomalyIQ Session Lifecycle Management
 *
 * Persists session_id to localStorage — survives page refresh.
 * Updates sidebar badge and topbar chip on session change.
 */

const SESSION_KEY = "anomalyiq_session_id";

let _currentSessionId = null;

// ── Public API ─────────────────────────────────────────────────────────────────

export function getSession() {
  return _currentSessionId;
}

export function setSession(id) {
  _currentSessionId = id;
  localStorage.setItem(SESSION_KEY, id);
  _updateUI(id);
}

export function clearSession() {
  _currentSessionId = null;
  localStorage.removeItem(SESSION_KEY);
  _updateUI(null);
}

/**
 * Load session from localStorage on page load.
 * Returns the session ID if one exists, null otherwise.
 */
export function loadSession() {
  const stored = localStorage.getItem(SESSION_KEY);
  if (stored) {
    _currentSessionId = stored;
    _updateUI(stored);
    return stored;
  }
  return null;
}

export function hasSession() {
  return !!_currentSessionId;
}

// ── UI Updates ─────────────────────────────────────────────────────────────────

function _updateUI(sessionId) {
  // Sidebar session badge
  const sidebarBadge = document.getElementById("sb-session-id");
  if (sidebarBadge) {
    sidebarBadge.textContent = sessionId
      ? sessionId.substring(0, 8) + "..."
      : "No session";
    sidebarBadge.style.color = sessionId
      ? "var(--accent)"
      : "var(--txt3)";
  }

  // Topbar session chip
  const topbarChip = document.getElementById("topbar-session-id");
  if (topbarChip) {
    if (sessionId) {
      topbarChip.textContent = sessionId.substring(0, 8);
      topbarChip.parentElement?.classList.remove("hidden");
    } else {
      topbarChip.textContent = "—";
    }
  }

  // Nav items that require a session — disable if no session
  const sessionNavItems = document.querySelectorAll(".nav-item[data-requires-session]");
  sessionNavItems.forEach(item => {
    if (sessionId) {
      item.style.opacity = "1";
      item.style.pointerEvents = "auto";
    } else {
      item.style.opacity = "0.4";
      item.style.pointerEvents = "none";
    }
  });
}
