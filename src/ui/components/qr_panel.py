"""
QR Pairing Panel — Gradio UI component for mobile device pairing.

Generates a QR code containing a JSON payload:
    {"sid": "<uuid>", "ws": "ws://<host>:<port>"}

The user scans the QR with the PWA, which extracts the session ID and
WebSocket URL, then connects to the WS server.

Usage:
    panel = QRPairingPanel()
    html = panel.render()  # Returns Gradio-compatible HTML string
    status = panel.get_status_html()  # Current pairing status

The panel is embedded in the dashboard sidebar via app.py.

Universidad Surcolombiana, 2026
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from typing import Optional

import qrcode
from qrcode.image.pil import PilImage

from src.ws.manager import WSManager

# Default WS port (must match server.py)
_WS_PORT: int = 8765


def _get_default_ws_url() -> str:
    """Resolve the WebSocket URL for QR code generation.

    Strategy:
        1. Use POSTURE_WS_URL env var if set (overrides everything).
        2. Use POSTURE_WS_HOST + POSTURE_WS_PORT if set.
        3. Auto-detect host IP using multiple methods (prefer LAN address).
        4. Use localhost as last resort.

    Returns:
        ws://<host>:<port>
    """
    override = os.environ.get("POSTURE_WS_URL", "")
    if override:
        return override

    host = os.environ.get("POSTURE_WS_HOST", "")
    port = int(os.environ.get("POSTURE_WS_PORT", str(_WS_PORT)))

    if not host:
        # Method 1: Try to get LAN IP via UDP socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 1))  # Doesn't actually connect
            host = s.getsockname()[0]
            s.close()
            if host and not host.startswith("127."):
                return f"ws://{host}:{port}"
        except Exception:
            pass

        # Method 2: Use socket.gethostbyname with hostname
        try:
            hostname = socket.gethostname()
            host = socket.gethostbyname(hostname)
            if host and not host.startswith("127."):
                return f"ws://{host}:{port}"
        except Exception:
            pass

        # Method 3: Use getaddrinfo to find non-loopback addresses
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None):
                addr = info[4][0]
                if not addr.startswith("127.") and addr != "::1":
                    host = addr
                    return f"ws://{host}:{port}"
        except Exception:
            pass

        # Fallback: localhost (will NOT work from phone)
        host = "127.0.0.1"
        print(f"[QR] WARNING: Could not detect LAN IP, using {host}. "
              f"Phone will NOT be able to connect. Set POSTURE_WS_HOST env var.")

    return f"ws://{host}:{port}"


class QRPairingPanel:
    """Generates a QR code and displays pairing status for the dashboard.

    Each instance creates a new session UUID on construction. The QR code
    embeds the SID and WS URL.

    The paired status is polled from WSManager.is_paired() every time
    the status HTML is rendered.
    """

    def __init__(
        self,
        sid: Optional[str] = None,
        ws_url: Optional[str] = None,
    ) -> None:
        self.sid: str = sid or str(uuid.uuid4())
        self.ws_url: str = ws_url or _get_default_ws_url()
        self._qr_image_b64: Optional[str] = None
        self._generated_at: float = time.time()

    # ── QR generation ───────────────────────────────────────────────────────

    @property
    def payload(self) -> dict:
        """The JSON payload encoded in the QR code.

        For deployment: the QR contains only the session ID.
        The mobile app constructs the WS URL from the server URL.

        For local dev: includes the full ws:// URL as fallback.
        """
        payload = {"sid": self.sid}
        # Include ws URL for local dev / backward compat
        payload["ws"] = self.ws_url
        return payload

    def _generate_qr_base64(self) -> str:
        """Generate QR code and return as base64-encoded PNG."""
        import base64 as _b64
        import io as _io

        import json as _json

        payload_str = _json.dumps(self.payload, ensure_ascii=False)

        qr = qrcode.QRCode(
            version=2,  # 25×25 matrix — enough for ~100 chars
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6,
            border=2,
        )
        qr.add_data(payload_str)
        qr.make(fit=True)

        img: PilImage = qr.make_image(image_factory=PilImage, fill_color="#1e293b", back_color="white")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return _b64.b64encode(buf.read()).decode("ascii")

    # ── Rendering ───────────────────────────────────────────────────────────

    def get_qr_html(self, lang: str = "es") -> str:
        """Render the QR code image as an inline base64 HTML <img> tag.

        The QR is cached for the lifetime of the QRPairingPanel instance.
        """
        if self._qr_image_b64 is None:
            self._qr_image_b64 = self._generate_qr_base64()

        badges = {
            "es": {
                "title": "Vincular dispositivo móvil",
                "subtitle": "Escanea el código QR con la app PostureMonitor",
                "connected": "✓ Vinculado",
                "disconnected": "○ No vinculado",
                "sid_label": "Sesión:",
                "hint": "Abre la app en tu móvil y escanea este código QR",
            },
            "en": {
                "title": "Pair mobile device",
                "subtitle": "Scan the QR code with the PostureMonitor app",
                "connected": "✓ Connected",
                "disconnected": "○ Disconnected",
                "sid_label": "Session:",
                "hint": "Open the app on your phone and scan this QR code",
            },
            "pt": {
                "title": "Vincular dispositivo móvel",
                "subtitle": "Escaneie o código QR com o app PostureMonitor",
                "connected": "✓ Vinculado",
                "disconnected": "○ Desvinculado",
                "sid_label": "Sessão:",
                "hint": "Abra o app no seu celular e escaneie este código QR",
            },
        }

        t = badges.get(lang, badges["es"])
        sid_short = f"{self.sid[:8]}..."

        # Show the WS URL prominently so user can verify it's not 127.0.0.1
        ws_display = self.ws_url.replace("ws://", "").replace("wss://", "")
        is_localhost = self.ws_url.startswith("ws://127.0.0.1") or self.ws_url.startswith("ws://localhost")
        url_color = "#ef4444" if is_localhost else "#22c55e"
        url_warning = ""
        if is_localhost:
            url_warning = """<div style="margin-top:8px;padding:6px 10px;background:#fef2f2;border:1px solid #fecaca;border-radius:6px;font-size:10px;color:#991b1b">
                ⚠ <strong>IP incorrecta:</strong> El QR apunta a localhost. Tu celular NO podrá conectar.
                <br>Ejecutá: <code>set POSTURE_WS_HOST=TU_IP</code> antes de iniciar la app.
                <br>Para ver tu IP: <code>ipconfig</code> → buscá "IPv4" en tu adaptador Wi-Fi.
            </div>"""
        else:
            url_warning = f"""<div style="margin-top:6px;font-size:10px;color:var(--pm-text-3)">
                📡 WebSocket: <code style="color:{url_color}">{self.ws_url}</code>
            </div>"""

        return f"""<div class="pm-card" style="text-align:center">
  <div class="pm-section-title">{t['title']}</div>
  <div style="margin:12px auto;width:180px;height:180px;background:white;border-radius:12px;display:flex;align-items:center;justify-content:center;overflow:hidden;border:2px solid var(--pm-border)">
    <img src="data:image/png;base64,{self._qr_image_b64}"
         alt="QR Code" style="width:168px;height:168px;display:block"
         id="pm-qr-img" />
  </div>
  <div style="font-size:10px;color:var(--pm-text-3);margin-bottom:6px">
    {t['sid_label']} <code style="font-family:'JetBrains Mono',monospace;color:var(--pm-accent-cyan)">{sid_short}</code>
  </div>
  {url_warning}
  <div id="pm-pairing-status" style="font-size:11px;font-weight:700;margin-bottom:8px;color:var(--pm-text-muted);transition:color 0.3s ease">
    {t['disconnected']}
  </div>
  <div style="font-size:10px;color:var(--pm-text-muted);line-height:1.4">{t['hint']}</div>
</div>"""

    def get_status_html(self) -> str:
        """Return a JSON carrier div that the JS polling loop reads to
        update the pairing status indicator in real time.

        The JS (in METRICS_JS) reads <div id='pm-pairing-data'> every
        100ms, parses the JSON payload, and updates the visible badge
        (<div id='pm-pairing-status'>) with the current pair state.

        This method is called periodically via a gr.Timer in app.py,
        so the carrier content always reflects the live pair state.
        """
        manager = WSManager()
        is_paired = manager.is_paired(self.sid)
        status_text = "✓ Vinculado" if is_paired else "○ No vinculado"

        # Return a JSON carrier for JS polling (same pattern as metrics data)
        import json as _json

        payload = _json.dumps(
            {
                "paired": is_paired,
                "text": status_text,
                "sid": self.sid[:8],
                "ws_url": self.ws_url,
            }
        )
        return f'<div id="pm-pairing-data" style="display:none">{payload}</div>'


