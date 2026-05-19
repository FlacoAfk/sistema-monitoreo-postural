"""
Multi-Person Alert Router — AlertRouter

Emits WebSocket alert messages at a configurable fixed interval while a
person remains in bad posture (ALERTA_LEVE / ALERTA_CRITICA).

Design:
- On first entry into an alert state: emit IMMEDIATELY (no delay).
- While the person stays in ALERTA_LEVE or ALERTA_CRITICA: emit every
  ``alert_interval_s`` seconds (default 30s), configurable from the
  Gradio frontend.
- On transition back to CORRECTO or NO_DETECTADO: emit a resolution
  payload once, then go silent.

All state lives in PersonAlertState (per-session, per-person).
AlertRouter itself is stateless and thread-safe.

Universidad Surcolombiana, 2026
"""

from __future__ import annotations

import time
from typing import Optional

from src.core.posture_analyzer import PostureResult, PostureStatus
from src.ws.manager import PersonAlertState


# ── Status code map: PostureStatus enum → normalized WebSocket string ──
_STATUS_MAP: dict[PostureStatus, str] = {
    PostureStatus.CORRECTO: "ok",
    PostureStatus.ALERTA_LEVE: "warn",
    PostureStatus.ALERTA_CRITICA: "crit",
    PostureStatus.NO_DETECTADO: "nd",
}


class AlertRouter:
    """Fixed-interval alert dispatcher for mobile push notifications.

    Emits an alert every ``alert_interval_s`` seconds while a person
    remains in bad posture.  On first entry into an alert state the
    alert is sent immediately (zero delay).

    Class constants can be overridden per-instance for testing.
    """

    # Seconds between consecutive WebSocket emissions per person while
    # in bad posture.  Overridable from the Gradio UI slider.
    alert_interval_s: float = 30.0

    # Seconds of continuous bad posture required before the first alarm.
    # Overridable from the Gradio UI slider.
    alert_threshold_s: float = 30.0

    # ── Public API ──────────────────────────────────────────────────────

    def evaluate(
        self,
        person_state: PersonAlertState,
        result: PostureResult,
        now: Optional[float] = None,
    ) -> Optional[dict]:
        """Decide whether to emit an alert for a person.

        Implements a threshold gate: bad posture must persist for
        ``alert_threshold_s`` seconds before the first alarm fires.
        Once the threshold is reached, subsequent alerts repeat every
        ``alert_interval_s`` seconds.

        Args:
            person_state: Mutable state for this person (updated in-place).
            result: Latest posture analysis result.
            now: Current timestamp (defaults to time.time()).

        Returns:
            Alert payload dict if an alert should be emitted, None otherwise.
        """
        if now is None:
            now = time.time()

        status = result.status

        # Track last status for debugging / transition detection
        person_state.last_status = status

        # ── Bad posture (ALERTA_LEVE / ALERTA_CRITICA) ──────────────
        if status in (PostureStatus.ALERTA_LEVE, PostureStatus.ALERTA_CRITICA):
            if person_state.bad_posture_armed_at is None:
                # Arm the gate — start timing
                person_state.bad_posture_armed_at = now
                return None  # No alert yet

            elapsed = now - person_state.bad_posture_armed_at
            if elapsed < self.alert_threshold_s:
                return None  # Threshold not reached yet

            # Threshold reached — check repetition interval (existing logic)
            if now - person_state.last_sent_at >= self.alert_interval_s:
                person_state.last_sent_at = now
                return self.build_payload(person_state.person_id, result)
            return None

        # ── CORRECTO — disarm + emit resolution if previously alerted ──
        if status is PostureStatus.CORRECTO:
            person_state.bad_posture_armed_at = None  # Disarm
            # Emit resolution only if we previously sent an alert
            if person_state.last_sent_at > 0:
                person_state.last_sent_at = 0
                return self._build_resolution_payload(person_state.person_id, result)
            return None

        # ── NO_DETECTADO / NO_INICIADO — disarm, no alert ────────────
        person_state.bad_posture_armed_at = None  # Disarm
        return None

    # ── Payload builders ────────────────────────────────────────────────

    @staticmethod
    def build_payload(person_id: int, result: PostureResult) -> dict:
        """Build the WebSocket alert payload from a PostureResult.

        Field names include both canonical names (for web frontend) and
        short aliases (for mobile app compatibility).
        """
        return {
            "type": "alert",
            "person_id": person_id,
            "status_code": _STATUS_MAP.get(result.status, "nd"),
            "status_label": result.status.value,
            "cpi": round(result.cpi, 1),
            # Canonical names (web frontend)
            "lumbar_angle_deg": round(result.lumbar_angle_deg, 1),
            "curvature_pct": round(result.curvature_pct, 2),
            "bad_posture_accumulated_s": round(result.bad_posture_accumulated_s, 1),
            # Short aliases (mobile app compatibility)
            "lumbar": round(result.lumbar_angle_deg, 1),
            "curvature": round(result.curvature_pct, 2),
            "bad_time": round(result.bad_posture_accumulated_s, 1),
            "confidence": round(result.confidence, 3),
            "timestamp": result.timestamp,
            "frame_id": result.frame_id,
        }

    @staticmethod
    def _build_resolution_payload(person_id: int, result: PostureResult) -> dict:
        """Build a resolution payload (person no longer in bad posture)."""
        return {
            "type": "resolution",
            "person_id": person_id,
            "timestamp": result.timestamp,
            "message": "Person no longer in bad posture — alert resolved.",
        }
