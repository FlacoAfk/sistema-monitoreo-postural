/**
 * PostureMonitor PWA — Application Logic
 *
 * Responsibilities:
 *   - QR code scanning via jsQR library
 *   - WebSocket connection with auto-reconnect (exponential backoff)
 *   - Real-time alert display with visual cards + vibration
 *   - Connection state management
 *
 * Protocol:
 *   QR payload:  {"sid": "<uuid>", "ws": "ws://<host>:<port>"}
 *   PWA → Server: {"type": "pair", "sid": "<uuid>"}
 *   Server → PWA: {"type": "paired", "sid": "<uuid>"}
 *   Server → PWA: {"type": "ping"}  (heartbeat)
 *   PWA → Server: {"type": "pong"}  (heartbeat reply)
 *   Server → PWA: {"type": "alert", "person_id": N, "status_code": "crit", ...}
 *   Server → PWA: {"type": "resolution", "person_id": N}
 *
 * Universidad Surcolombiana, 2026
 */

/* global jsQR */

// ── Application State ──────────────────────────────────────────
const AppState = {
  ws: null,
  sid: null,
  wsUrl: null,
  paired: false,
  scanning: false,
  stream: null,
  reconnectAttempts: 0,
  maxReconnectDelay: 30000,
  reconnectTimer: null,
  heartbeatTimer: null,
  alertCount: 0,
};

// ── DOM References ─────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const video = $('qr-video');
const canvas = $('qr-canvas');
const scanContainer = $('scan-container');
const scanBtn = $('scan-btn');
const rescanBtn = $('rescan-btn');
const statusEl = $('pairing-status');
const statusDot = $('status-dot');
const statusText = $('status-text');
const reconnectNotice = $('reconnect-notice');
const alertsContainer = $('alerts-container');
const emptyState = $('empty-state');
const sessionIdEl = $('session-id');

// ── Status Helpers ─────────────────────────────────────────────
function setStatus(mode, text) {
  const modeMap = {
    disconnected: 'status-disconnected',
    paired: 'status-paired',
    error: 'status-error',
    connecting: 'status-connecting',
  };
  const dotMap = {
    disconnected: 'dot-disconnected',
    paired: 'dot-paired',
    error: 'dot-error',
    connecting: 'dot-connecting',
  };

  statusEl.className = 'pairing-status ' + (modeMap[mode] || 'status-disconnected');
  statusDot.className = 'status-dot ' + (dotMap[mode] || 'dot-disconnected');
  statusText.textContent = text;
}

function showReconnecting(visible, msg) {
  reconnectNotice.classList.toggle('visible', visible);
  if (msg) reconnectNotice.textContent = msg;
}

// ── QR Scanner ─────────────────────────────────────────────────
async function startScanner() {
  if (AppState.scanning) return;

  // Hide empty state when scanning
  emptyState.classList.add('hidden');

  try {
    const constraints = {
      video: {
        facingMode: 'environment',
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    };

    AppState.stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = AppState.stream;
    video.setAttribute('playsinline', '');
    video.setAttribute('autoplay', '');
    video.setAttribute('muted', '');

    await video.play();

    AppState.scanning = true;
    scanBtn.disabled = true;
    scanBtn.textContent = '🔍 Escaneando...';

    // Configure canvas matching video dimensions
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Start scan loop
    requestAnimationFrame(scanFrame);
  } catch (err) {
    handleScanError(err);
  }
}

function stopScanner() {
  AppState.scanning = false;

  if (AppState.stream) {
    AppState.stream.getTracks().forEach(function (track) {
      track.stop();
    });
    AppState.stream = null;
  }

  if (video) {
    video.srcObject = null;
  }

  scanBtn.disabled = false;
  scanBtn.textContent = '📷 Escanear QR';
}

function scanFrame() {
  if (!AppState.scanning) return;

  if (video.readyState < video.HAVE_ENOUGH_DATA) {
    requestAnimationFrame(scanFrame);
    return;
  }

  var vw = video.videoWidth;
  var vh = video.videoHeight;

  if (vw === 0 || vh === 0) {
    requestAnimationFrame(scanFrame);
    return;
  }

  canvas.width = vw;
  canvas.height = vh;

  var ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, vw, vh);

  var imageData;
  try {
    imageData = ctx.getImageData(0, 0, vw, vh);
  } catch (e) {
    // Canvas may be tainted by cross-origin video
    requestAnimationFrame(scanFrame);
    return;
  }

  var code = jsQR(imageData.data, imageData.width, imageData.height, {
    inversionAttempts: 'dontInvert',
  });

  if (code && code.data) {
    try {
      var payload = JSON.parse(code.data);
      if (payload && payload.sid && payload.ws) {
        handleQrScanned(payload.sid, payload.ws);
        return;
      }
    } catch (e) {
      // QR data was not valid JSON — keep scanning
    }
  }

  requestAnimationFrame(scanFrame);
}

function handleQrScanned(sid, wsUrl) {
  stopScanner();

  setStatus('connecting', 'Conectando...');

  // Show session ID
  sessionIdEl.textContent = sid.substring(0, 8) + '...';
  sessionIdEl.parentElement.style.display = 'flex';

  connect(sid, wsUrl);
}

function handleScanError(err) {
  AppState.scanning = false;
  scanBtn.disabled = false;
  scanBtn.textContent = '📷 Escanear QR';

  var msg;
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    msg = '⚠ Permiso de cámara denegado. Ajusta los permisos en la configuración de tu navegador.';
    setStatus('error', 'Permiso denegado');
  } else if (err.name === 'NotFoundError') {
    msg = '⚠ No se detectó una cámara en este dispositivo.';
    setStatus('error', 'Sin cámara');
  } else {
    msg = '⚠ Error al acceder a la cámara: ' + (err.message || 'desconocido');
    setStatus('error', 'Error de cámara');
  }

  var errEl = $('scanner-hint');
  if (errEl) {
    errEl.textContent = msg;
    errEl.style.color = 'var(--pm-danger)';
  }
}

// ── WebSocket Connection ───────────────────────────────────────
function connect(sid, wsUrl) {
  // Clean up any existing connection or reconnect timer
  disconnect();

  AppState.sid = sid;
  AppState.wsUrl = wsUrl;

  try {
    AppState.ws = new WebSocket(wsUrl);
  } catch (err) {
    setStatus('error', 'Error de conexión');
    showReconnecting(true, 'Error: URL de WebSocket inválida. Re-escanee el QR.');
    return;
  }

  // Timeout: if not paired within 10s, show error
  var pairTimeout = setTimeout(function () {
    if (!AppState.paired && AppState.ws) {
      AppState.ws.close();
      setStatus('error', 'Tiempo de espera agotado');
      showReconnecting(true, 'No se recibió confirmación del servidor. Reintentando...');
      scheduleReconnect();
    }
  }, 10000);

  AppState.ws.onopen = function () {
    // Send pair request
    try {
      AppState.ws.send(JSON.stringify({ type: 'pair', sid: AppState.sid }));
    } catch (e) {
      // Connection may have closed already
    }
  };

  AppState.ws.onmessage = function (event) {
    try {
      var msg = JSON.parse(event.data);
    } catch (e) {
      return;
    }

    switch (msg.type) {
      case 'paired':
        clearTimeout(pairTimeout);
        AppState.reconnectAttempts = 0;
        AppState.paired = true;
        showReconnecting(false);
        setStatus('paired', '✓ Vinculado — Recibiendo alertas');
        startHeartbeat();
        break;

      case 'alert':
        handleAlert(msg);
        break;

      case 'resolution':
        handleResolution(msg);
        break;

      case 'ping':
        // Respond with pong
        try {
          AppState.ws.send(JSON.stringify({ type: 'pong' }));
        } catch (e) {
          // Connection may be closed
        }
        break;

      case 'error':
        setStatus('error', 'Error del servidor: ' + (msg.message || 'desconocido'));
        break;
    }
  };

  AppState.ws.onclose = function () {
    clearTimeout(pairTimeout);
    stopHeartbeat();

    if (AppState.paired) {
      // Was previously paired — unexpected disconnect
      setStatus('connecting', 'Reconectando...');
      showReconnecting(true, 'Conexión perdida. Reconectando...');
      scheduleReconnect();
    } else if (AppState.sid && AppState.wsUrl) {
      // Was connecting but didn't pair yet
      setStatus('connecting', 'Reconectando...');
      showReconnecting(true, 'Conectando al servidor...');
      scheduleReconnect();
    }
  };

  AppState.ws.onerror = function () {
    // onclose will fire after this, so we handle it there
  };
}

function disconnect() {
  AppState.paired = false;
  stopHeartbeat();

  if (AppState.reconnectTimer) {
    clearTimeout(AppState.reconnectTimer);
    AppState.reconnectTimer = null;
  }

  if (AppState.ws) {
    try {
      AppState.ws.onclose = null; // Prevent reconnect loop on manual close
      AppState.ws.close();
    } catch (e) {
      // Ignore
    }
    AppState.ws = null;
  }
}

function scheduleReconnect() {
  if (AppState.reconnectTimer) {
    clearTimeout(AppState.reconnectTimer);
  }

  var delay = Math.min(
    1000 * Math.pow(2, AppState.reconnectAttempts),
    AppState.maxReconnectDelay
  );
  AppState.reconnectAttempts++;

  var delaySec = Math.round(delay / 1000);
  showReconnecting(true, 'Reconectando en ' + delaySec + 's... (intento ' + AppState.reconnectAttempts + ')');

  AppState.reconnectTimer = setTimeout(function () {
    if (AppState.sid && AppState.wsUrl) {
      setStatus('connecting', 'Reconectando...');
      connect(AppState.sid, AppState.wsUrl);
    }
  }, delay);
}

// ── Heartbeat Monitor ──────────────────────────────────────────
function startHeartbeat() {
  stopHeartbeat();
  // If we don't receive a ping within 60s, consider the connection dead
  AppState.heartbeatTimer = setInterval(function () {
    if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
      // Connection still appears open — send a ping to verify
      try {
        AppState.ws.send(JSON.stringify({ type: 'ping' }));
      } catch (e) {
        // Will be caught by onclose
      }
    }
  }, 45000);
}

function stopHeartbeat() {
  if (AppState.heartbeatTimer) {
    clearInterval(AppState.heartbeatTimer);
    AppState.heartbeatTimer = null;
  }
}

// ── Alert Display ──────────────────────────────────────────────
function handleAlert(data) {
  AppState.alertCount++;
  emptyState.classList.add('hidden');

  var card = document.createElement('div');
  card.className = 'alert-card alert-' + (data.status_code || 'warn');

  // Map status codes
  var statusLabel = {
    ok: '✓ CORRECTO',
    warn: '⚠ ALERTA LEVE',
    crit: '✕ ALERTA CRÍTICA',
  };

  var label = statusLabel[data.status_code] || '⚠ ALERTA';

  var badTime = data.bad_posture_accumulated_s || 0;
  var cpi = data.cpi !== undefined ? data.cpi : 0;
  var lumbar = data.lumbar_angle_deg !== undefined ? data.lumbar_angle_deg + '°' : '—';
  var curv = data.curvature_pct !== undefined ? data.curvature_pct.toFixed(1) + '%' : '—';
  var personId = data.person_id !== undefined ? data.person_id : 1;
  var ts = data.timestamp
    ? new Date(data.timestamp * 1000).toLocaleTimeString()
    : new Date().toLocaleTimeString();

  card.innerHTML =
    '<div class="alert-header">' + label + ' — Persona #' + personId + '</div>' +
    '<div class="alert-body">' +
      '<div class="alert-field">' +
        '<div class="label">CPI</div>' +
        '<div class="value ' + (data.status_code === 'crit' ? 'crit-color' : data.status_code === 'warn' ? 'warn-color' : 'ok-color') + '">' + cpi.toFixed(1) + '</div>' +
      '</div>' +
      '<div class="alert-field">' +
        '<div class="label">Tiempo acum.</div>' +
        '<div class="value ' + (data.status_code === 'crit' ? 'crit-color' : 'warn-color') + '">' + badTime.toFixed(0) + 's</div>' +
      '</div>' +
      '<div class="alert-field">' +
        '<div class="label">Lumbar</div>' +
        '<div class="value">' + lumbar + '</div>' +
      '</div>' +
      '<div class="alert-field">' +
        '<div class="label">Curvatura</div>' +
        '<div class="value">' + curv + '</div>' +
      '</div>' +
    '</div>' +
    '<div class="alert-timestamp">' + ts + '</div>';

  alertsContainer.insertBefore(card, alertsContainer.firstChild);

  // Vibrate on critical/mild alerts
  if (data.status_code === 'crit') {
    if (navigator.vibrate) {
      navigator.vibrate([200, 100, 200, 100, 400]);
    }
  } else if (data.status_code === 'warn') {
    if (navigator.vibrate) {
      navigator.vibrate([150, 75, 150]);
    }
  }

  // Limit visible cards to 50 — remove oldest
  while (alertsContainer.children.length > 50) {
    alertsContainer.removeChild(alertsContainer.lastChild);
  }
}

function handleResolution(data) {
  // Person lost — show resolution card
  var personId = data.person_id !== undefined ? data.person_id : 1;
  var ts = new Date().toLocaleTimeString();

  var card = document.createElement('div');
  card.className = 'alert-card alert-resolution';
  card.innerHTML =
    '<div class="alert-header">✓ Persona #' + personId + ' — Sin alerta</div>' +
    '<div class="alert-body">' +
      '<div class="alert-field" style="grid-column: 1 / -1">' +
        '<div class="label">Estado</div>' +
        '<div class="value ok-color">Postura recuperada o persona ya no detectada</div>' +
      '</div>' +
    '</div>' +
    '<div class="alert-timestamp">' + ts + '</div>';

  alertsContainer.insertBefore(card, alertsContainer.firstChild);

  // Limit visible cards
  while (alertsContainer.children.length > 50) {
    alertsContainer.removeChild(alertsContainer.lastChild);
  }
}

// ── Re-scan ────────────────────────────────────────────────────
function resetAndRescan() {
  disconnect();
  stopScanner();
  startScanner();
}

// ── Initialize ─────────────────────────────────────────────────
(function init() {
  // Set initial status
  setStatus('disconnected', '○ No vinculado — Escanea el QR');

  // Register service worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker
      .register('/pwa/sw.js')
      .then(function () {
        // Service worker registered
      })
      .catch(function (err) {
        console.warn('[PWA] Service worker registration failed:', err);
      });
  }

  // Event handlers
  scanBtn.addEventListener('click', startScanner);
  rescanBtn.addEventListener('click', resetAndRescan);
})();
