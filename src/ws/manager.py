"""
WebSocket Connection Manager — WSManager

Singleton that manages all WebSocket connections and session registries
for the mobile QR notification system.

Provides:
    WSManager          — Singleton: pair/unpair/broadcast/is_paired
    SessionContext     — Per-session state (connection, timestamps, person states)
    PersonAlertState — Per-person alert interval tracking

Architecture:
    The Gradio main thread calls process_frame() which feeds results to
    AlertRouter. AlertRouter decides whether to emit; if yes, it calls
    WSManager.broadcast(sid, payload). WSManager writes to the async
    WebSocket connection (via a threadsafe queue → asyncio event loop).

Thread safety:
    All public methods use a threading.Lock to protect the sessions dict.
    The actual ws.send() calls are dispatched to the asyncio event loop
    via a threadsafe call_soon_threadsafe pattern (managed in server.py).

Universidad Surcolombiana, 2026
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

from src.core.posture_analyzer import PostureStatus


# ── Forward reference ─────────────────────────────────────────────────────────
# Avoid circular import; server.py bridges the gap at runtime.
_WS_CONNECTION = Any  # websockets.WebSocketServerProtocol (runtime-only)


@dataclass
class PersonAlertState:
    """Tracks alert state for a single detected person within a session.

    Used by AlertRouter to decide whether a new WebSocket message should
    be emitted (fixed interval while in bad posture + transition events).
    """

    person_id: int
    last_status: PostureStatus = PostureStatus.NO_DETECTADO
    last_sent_at: float = 0.0  # Timestamp of last emission (interval check)
    bad_posture_armed_at: Optional[float] = None  # Timestamp when bad posture gate was armed


@dataclass
class SessionContext:
    """All state held for a single paired session (one QR → one PWA)."""

    sid: str
    ws: _WS_CONNECTION  # websockets.WebSocketServerProtocol
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    person_states: dict[int, PersonAlertState] = field(default_factory=dict)

    @property
    def is_alive(self) -> bool:
        """Check if the underlying WebSocket connection is considered open.

        Supports both websockets v16+ (uses ``state`` attribute) and older
        versions (uses ``open`` attribute).
        """
        try:
            # websockets v16+ uses a State enum (OPEN=1, CLOSING=2, CLOSED=3)
            state = getattr(self.ws, "state", None)
            if state is not None:
                from websockets.protocol import State as _WSState
                return state is _WSState.OPEN
            # Fallback for older versions
            return getattr(self.ws, "open", False)
        except Exception:
            return False

    def touch(self) -> None:
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def get_or_create_person_state(self, person_id: int) -> PersonAlertState:
        """Get or create a PersonAlertState for the given person_id."""
        if person_id not in self.person_states:
            self.person_states[person_id] = PersonAlertState(person_id=person_id)
        return self.person_states[person_id]


class WSManager:
    """Singleton WebSocket connection manager.

    Manages session registry, pairing, broadcast, heartbeat, and cleanup.

    Thread safety:
        _lock guards all access to the `sessions` dict.
        The actual async send is dispatched via an external loop reference
        set by server.py at startup (see server.py for details).
    """

    _instance: Optional["WSManager"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "WSManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.sessions: dict[str, SessionContext] = {}
                    cls._instance._loop = None  # Set by server.py
        return cls._instance

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def set_loop(self, loop: Any) -> None:
        """Store the asyncio event loop reference for thread-safe dispatch."""
        self._loop = loop

    @property
    def loop(self) -> Any:
        """Return the stored asyncio event loop."""
        return self._loop

    # ── Pairing ─────────────────────────────────────────────────────────────

    async def pair(self, sid: str, ws: _WS_CONNECTION) -> bool:
        """Register a WebSocket connection under the given session ID.

        Returns True if pairing succeeds, False if the SID is already paired
        with a *different* connection (we refuse to overwrite — the old
        session must unpair first or the client should use a new QR).
        """
        with self._lock:
            existing = self.sessions.get(sid)
            if existing is not None and existing.ws is not ws:
                # Already paired with a different connection — reject
                return False

            ctx = SessionContext(sid=sid, ws=ws)
            ctx.touch()
            self.sessions[sid] = ctx
            return True

    async def unpair(self, sid: str) -> None:
        """Remove a session from the registry.

        The caller (server.py) is responsible for closing the WS connection.
        """
        with self._lock:
            self.sessions.pop(sid, None)

    def is_paired(self, sid: str) -> bool:
        """Check whether a session ID is currently registered."""
        with self._lock:
            return sid in self.sessions

    def get_session(self, sid: str) -> Optional[SessionContext]:
        """Retrieve a SessionContext by ID (or None if not found)."""
        with self._lock:
            return self.sessions.get(sid)

    def get_all_sessions(self) -> dict[str, SessionContext]:
        """Return a snapshot of all sessions (copy for iteration safety)."""
        with self._lock:
            return dict(self.sessions)

    # ── Broadcast ───────────────────────────────────────────────────────────

    async def broadcast(self, sid: str, payload: dict[str, Any]) -> bool:
        """Send a JSON payload to a specific paired session.

        Returns True on success, False if the session does not exist or the
        connection is dead. On failure the caller should unpair the session.

        This method is called from a non-async context (Gradio threads) via
        AlertRouter, NOT from the async WS handler. The actual send is
        dispatched to the asyncio event loop thread.

        If called from an async context (e.g., heartbeat), the send happens
        directly.
        """
        ctx = self.get_session(sid)
        if ctx is None:
            import logging as _logging
            _logging.getLogger("ws-server").warning(f"broadcast: sid={sid[:8]}... not found")
            return False
        if not ctx.is_alive:
            import logging as _logging
            _logging.getLogger("ws-server").warning(f"broadcast: sid={sid[:8]}... connection not alive")
            return False

        try:
            import json as _json
            import logging as _logging

            data = _json.dumps(payload, ensure_ascii=False)
            _logging.getLogger("ws-server").info(
                f"Broadcast to sid={sid[:8]}... type={payload.get('type', '?')} "
                f"person={payload.get('person_id', '?')} size={len(data)}b"
            )

            # If we are already in the async event loop, send directly.
            # Otherwise, dispatch via call_soon_threadsafe.
            loop = self._loop
            if loop is not None and not loop.is_closed():
                # Check if we're already inside the loop
                import asyncio as _asyncio

                try:
                    running = _asyncio.get_running_loop()
                    if running is loop:
                        await ctx.ws.send(data)
                        _logging.getLogger("ws-server").info(f"Broadcast OK (direct): sid={sid[:8]}...")
                    else:
                        # Dispatch to the WS event loop
                        fut = _asyncio.run_coroutine_threadsafe(
                            ctx.ws.send(data), loop
                        )
                        fut.result(timeout=5.0)
                        _logging.getLogger("ws-server").info(f"Broadcast OK (threadsafe): sid={sid[:8]}...")
                except RuntimeError:
                    # No running loop — dispatch
                    fut = _asyncio.run_coroutine_threadsafe(
                        ctx.ws.send(data), loop
                    )
                    fut.result(timeout=5.0)
                    _logging.getLogger("ws-server").info(f"Broadcast OK (runtimeerror): sid={sid[:8]}...")
            else:
                # No loop reference — try direct send (may fail)
                _logging.getLogger("ws-server").warning(f"Broadcast: no event loop, trying direct send: sid={sid[:8]}...")
                await ctx.ws.send(data)

            ctx.touch()
            return True
        except Exception as e:
            _logging.getLogger("ws-server").error(f"Broadcast FAILED: sid={sid[:8]}... error={e}")
            return False

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def cleanup_stale(self, max_idle_seconds: float = 120.0) -> list[str]:
        """Remove sessions that have been idle beyond the threshold.

        Returns a list of removed SIDs. The caller (server.py) should close
        the corresponding WS connections.
        """
        now = time.time()
        stale: list[str] = []
        with self._lock:
            for sid, ctx in list(self.sessions.items()):
                if now - ctx.last_heartbeat > max_idle_seconds or not ctx.is_alive:
                    stale.append(sid)
                    del self.sessions[sid]
        return stale

    def cleanup_all(self) -> list[str]:
        """Remove all sessions. Returns removed SIDs."""
        with self._lock:
            sids = list(self.sessions.keys())
            self.sessions.clear()
        return sids
