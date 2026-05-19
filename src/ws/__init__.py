# src/ws/ — WebSocket layer for mobile QR notifications
#
# Provides:
#   src/ws/manager.py     — WSManager (connection registry, pairing, broadcast)
#   src/ws/alert_router.py — AlertRouter (multi-person debounced alerts)
#   src/ws/server.py       — Async WS server running parallel to Gradio
#
# Universidad Surcolombiana, 2026

from src.ws.manager import WSManager, SessionContext, PersonAlertState
from src.ws.alert_router import AlertRouter
from src.ws.server import start_ws_server, stop_ws_server

__all__ = [
    "WSManager",
    "SessionContext",
    "PersonAlertState",
    "AlertRouter",
    "start_ws_server",
    "stop_ws_server",
]
