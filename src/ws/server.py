"""
Async WebSocket Server — runs parallel to Gradio on port 8765.

Manages:
    - Client connections (pair/unpair)
    - Heartbeat (server-side ping every 30s)
    - Stale session cleanup (120s idle timeout)
    - Thread-safe dispatch bridge (Gradio threads → asyncio)

Architecture:
    A daemon thread runs an asyncio event loop. The websockets server
    listens on port 8765. The WSManager singleton is shared between the
    Gradio main thread and the WS handler via thread-safe locks.

    The WS server event loop is stored in WSManager.set_loop() so that
    WSManager.broadcast() can dispatch send coroutines via
    run_coroutine_threadsafe().

Startup:
    Call start_ws_server() from the Gradio main thread after app.launch().
    It spawns a daemon thread and returns immediately.

Shutdown:
    Call stop_ws_server() to gracefully close all connections.

Usage:
    from src.ws.server import start_ws_server, stop_ws_server

    start_ws_server(host="0.0.0.0", port=8765)
    ...
    stop_ws_server()

Universidad Surcolombiana, 2026
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import threading
import time
from typing import Any, Optional

import websockets

from src.ws.manager import WSManager

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("ws-server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("[WS] %(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(_ch)

# ── Module-level state ────────────────────────────────────────────────────────
_ws_server: Optional[Any] = None  # The running websockets server
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_shutdown_event: Optional[asyncio.Event] = None

# Constants
WS_HOST: str = "0.0.0.0"
WS_PORT: int = 8765
HEARTBEAT_INTERVAL_S: float = 30.0  # Server → Client ping every 30s
STALE_CLEANUP_INTERVAL_S: float = 60.0  # Cleanup dead sessions every 60s
MAX_IDLE_S: float = 120.0  # Disconnect idle clients after 2 min


# ── Connection handler ────────────────────────────────────────────────────────
async def _handler(ws: websockets.WebSocketServerProtocol) -> None:
    """Handle a single WebSocket connection lifecycle.

    Protocol:
        Client → Server:  {"type": "pair", "sid": "<uuid>"}
        Server → Client:  {"type": "paired", "sid": "<uuid>"}
        Server → Client:  {"type": "ping"}  (heartbeat, every 30s)
        Client → Server:  {"type": "pong"}  (heartbeat reply)
        Server → Client:  {"type": "alert", ...}  (posture alert)
        Server → Client:  {"type": "resolution", ...}  (person lost)
    """
    manager = WSManager()
    paired_sid: Optional[str] = None

    try:
        async for raw in ws:
            # ── Parse message ───────────────────────────────────────────────
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                await ws.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = msg.get("type", "")
            # ── Pairing ─────────────────────────────────────────────────────
            if msg_type == "pair":
                sid = msg.get("sid", "")
                if not sid or len(sid) < 8:
                    await ws.send(
                        json.dumps({"type": "error", "message": "Invalid SID"})
                    )
                    continue

                success = await manager.pair(sid, ws)
                if success:
                    paired_sid = sid
                    await ws.send(
                        json.dumps({"type": "paired", "sid": sid})
                    )
                    logger.info(f"Client paired: sid={sid[:8]}...")
                else:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": "SID already paired with another connection",
                            }
                        )
                    )

            # ── Heartbeat reply ─────────────────────────────────────────────
            elif msg_type == "pong":
                if paired_sid:
                    ctx = manager.get_session(paired_sid)
                    if ctx:
                        ctx.touch()

            # ── Unpair ──────────────────────────────────────────────────────
            elif msg_type == "unpair":
                if paired_sid:
                    await manager.unpair(paired_sid)
                    logger.info(f"Client unpaired: sid={paired_sid[:8]}...")
                    paired_sid = None
                    await ws.send(
                        json.dumps({"type": "unpaired"})
                    )
                # Stay connected (can re-pair with a new SID)

            else:
                await ws.send(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Unknown message type: {msg_type}",
                        }
                    )
                )

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed: sid={paired_sid[:8] if paired_sid else 'unknown'}... code={e.code} reason={e.reason!r}")
    except Exception:
        logger.exception("WS handler error")
    finally:
        # Clean up on disconnect
        if paired_sid:
            await manager.unpair(paired_sid)
            logger.info(f"Client disconnected: sid={paired_sid[:8]}...")


# ── Heartbeat / cleanup background task ───────────────────────────────────────
async def _heartbeat_loop() -> None:
    """Periodic heartbeat ping + stale session cleanup."""
    while not _shutdown_event or not _shutdown_event.is_set():
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)

            manager = WSManager()
            sessions = manager.get_all_sessions()

            # Send ping to all active sessions
            for sid, ctx in sessions.items():
                if ctx.is_alive:
                    try:
                        await ctx.ws.send(json.dumps({"type": "ping"}))
                    except Exception:
                        pass

            # Cleanup stale sessions
            stale_sids = manager.cleanup_stale(max_idle_seconds=MAX_IDLE_S)
            for sid in stale_sids:
                logger.info(f"Cleaned stale session: sid={sid[:8]}...")

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Heartbeat loop error")


# ── Startup / Shutdown ────────────────────────────────────────────────────────
async def _run_server(host: str, port: int) -> None:
    """Start the websockets server and run until shutdown."""
    global _ws_server, _shutdown_event

    _shutdown_event = asyncio.Event()

    # Store the event loop reference in WSManager for thread-safe broadcasts
    loop = asyncio.get_running_loop()
    WSManager().set_loop(loop)

    # Configure the server
    _ws_server = await websockets.serve(
        _handler,
        host,
        port,
        ping_interval=20,  # Send protocol-level pings every 20s (OkHttp responds automatically)
        ping_timeout=10,   # Close connection if no pong within 10s
        max_size=2**16,  # 64KB max message
    )

    logger.info(f"WebSocket server listening on ws://{host}:{port}")

    # Start heartbeat + cleanup task
    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    # Wait for shutdown signal
    try:
        await _shutdown_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        if _ws_server:
            _ws_server.close()
            await _ws_server.wait_closed()
            _ws_server = None

        logger.info("WebSocket server stopped.")


def start_ws_server(host: str = WS_HOST, port: int = WS_PORT) -> None:
    """Start the WebSocket server in a background daemon thread.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _thread, _event_loop

    if _thread is not None and _thread.is_alive():
        logger.warning("WS server already running — ignoring start request.")
        return

    async def _runner() -> None:
        global _event_loop
        _event_loop = asyncio.get_running_loop()
        await _run_server(host, port)

    def _thread_target() -> None:
        asyncio.run(_runner())

    _thread = threading.Thread(
        target=_thread_target, name="ws-server-thread", daemon=True
    )
    _thread.start()
    logger.info(f"WS server starting on ws://{host}:{port} (background thread)")


def stop_ws_server() -> None:
    """Gracefully shut down the WebSocket server and close all connections."""
    global _ws_server, _shutdown_event, _thread

    if _shutdown_event is not None and not _shutdown_event.is_set():
        _shutdown_event.set()

    # Cleanup all sessions
    WSManager().cleanup_all()

    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=5.0)
        _thread = None

    _ws_server = None
    logger.info("WS server shut down complete.")
