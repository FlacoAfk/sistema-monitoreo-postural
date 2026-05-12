"""
Component 2 — Backend Matemático (posture_analyzer.py)

Calcula el Combined Posture Index (CPI) usando 5 keypoints de la cadena
posterior de la espalda: K0, K1, K8, K3, K4.

Fórmula CPI (validada 2026-05-08 con 6 imágenes × 4 modelos):
  CPI = déficit_lumbar × 2 + curvatura_escapular_normalizada × 100

donde:
  déficit_lumbar = max(0, 180° - ∠K8-K3-K4)
  curvatura_escapular_normalizada = dist_⊥(K8, línea K1→K4) / |K1→K4|

Mapeo Roboflow verificado (2026-05-08):
| K (YOLO) | Roboflow ID | Nombre |
|:--------:|:-----------:|--------|
| K0 | 0  | Head-back (Occipital) |
| K1 | 1  | Neck-back (Cervical C7) |
| K2 | 2  | Shoulder-top (Acromion) |
| K3 | 6  | Back-backedge (Espalda media) |
| K4 | 7  | Hips-backedge (Cadera) |
| K5 | 10 | Neck-middle (Cervical media) |
| K6 | 13 | Jaw (Mandíbula) |
| K7 | 14 | Chin (Mentón) |
| K8 | 18 | Shoulder-back (Escápula) |

Clasificación (umbrales calibrados por el usuario):
  CPI ≤ 35           → CORRECTO (verde)
  35 < CPI ≤ 50      → ALERTA LEVE (amarillo)
  CPI > 50           → ALERTA CRÍTICA (rojo)

Autor: Sistema de Monitoreo Postural — Universidad Surcolombiana 2026
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PostureStatus(Enum):
    """Estados de clasificación postural según umbrales ergonómicos."""

    CORRECTO = "CORRECTO"
    ALERTA_LEVE = "ALERTA LEVE"
    ALERTA_CRITICA = "ALERTA CRÍTICA"
    NO_DETECTADO = "NO DETECTADO"

    @property
    def color_hex(self) -> str:
        return {
            PostureStatus.CORRECTO: "#00FF00",
            PostureStatus.ALERTA_LEVE: "#FFD700",
            PostureStatus.ALERTA_CRITICA: "#FF0000",
            PostureStatus.NO_DETECTADO: "#808080",
        }[self]

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        return {
            PostureStatus.CORRECTO: (0, 255, 0),
            PostureStatus.ALERTA_LEVE: (0, 215, 255),
            PostureStatus.ALERTA_CRITICA: (0, 0, 255),
            PostureStatus.NO_DETECTADO: (128, 128, 128),
        }[self]


@dataclass
class PostureResult:
    """Resultado del análisis postural para un frame."""

    timestamp: float
    frame_id: int
    status: PostureStatus = PostureStatus.NO_DETECTADO

    # CPI — Combined Posture Index (métrica principal)
    cpi: float = 0.0

    # Componentes del CPI
    lumbar_angle_deg: float = 0.0       # ∠K8-K3-K4
    curvature_pct: float = 0.0           # curvatura escapular normalizada (%)
    spine_length_px: float = 0.0         # |K1→K4|

    # Ángulos auxiliares (solo informativos)
    cervicodorsal_deg: float = 0.0      # ∠K0-K1-K8
    angle_deg: float = 0.0              # alias para compatibilidad

    confidence: float = 0.0
    bad_posture_accumulated_s: float = 0.0

    @property
    def needs_alert(self) -> bool:
        return self.bad_posture_accumulated_s > 30.0 and self.status in (
            PostureStatus.ALERTA_LEVE,
            PostureStatus.ALERTA_CRITICA,
        )


class PostureAnalyzer:
    """
    Analizador postural — Combined Posture Index (CPI) multivectorial.

    Usa 5 keypoints de la cadena posterior (K0, K1, K8, K3, K4) para
    calcular curvatura escapular + ángulo lumbar → CPI.

    CPI bajo → espalda recta; CPI alto → encorvado.
    """

    # Umbrales CPI calibrados por el usuario
    CPI_LEVE: float = 35.0      # CPI > 35 → ALERTA LEVE
    CPI_CRITICO: float = 50.0   # CPI > 50 → ALERTA CRÍTICA
    ALERT_CONTINUOUS_SECONDS: float = 30.0

    def __init__(self) -> None:
        self._bad_posture_start: Optional[float] = None
        self._last_status: PostureStatus = PostureStatus.NO_DETECTADO
        self._last_update: float = time.time()

    # ── Helpers estáticos ──────────────────────────────────────────────────

    @staticmethod
    def _vec_len(ax: float, ay: float, bx: float, by: float) -> float:
        return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

    @staticmethod
    def _angle_at_vertex(ax: float, ay: float,
                          vx: float, vy: float,
                          bx: float, by: float) -> float:
        """Ángulo en el vértice V: ∠(A-V-B). Retorna grados o -1 si inválido."""
        ux, uy = ax - vx, ay - vy
        wx, wy = bx - vx, by - vy
        nu = math.sqrt(ux * ux + uy * uy)
        nw = math.sqrt(wx * wx + wy * wy)
        if nu < 1.0 or nw < 1.0:
            return -1.0
        cos_a = (ux * wx + uy * wy) / (nu * nw)
        cos_a = max(-1.0, min(1.0, cos_a))
        return math.degrees(math.acos(cos_a))

    @staticmethod
    def _point_line_distance(px: float, py: float,
                              ax: float, ay: float,
                              bx: float, by: float) -> float:
        """Distancia perpendicular de P a la línea AB. Retorna px o -1."""
        dx = bx - ax
        dy = by - ay
        line_len = math.sqrt(dx * dx + dy * dy)
        if line_len < 1.0:
            return -1.0
        return abs(dx * (ay - py) - (ax - px) * dy) / line_len

    # ── Análisis principal ─────────────────────────────────────────────────

    def analyze(
        self,
        keypoints: list[list[float]],
        detected: bool,
        timestamp: Optional[float] = None,
        frame_id: int = 0,
    ) -> PostureResult:
        """
        Calcula el CPI a partir de 5 keypoints posteriores.

        Args:
            keypoints: Lista de 9 keypoints [[x, y, conf], ...].
            detected: Si hubo detección.
            timestamp: Timestamp del frame.
            frame_id: ID secuencial.

        Returns:
            PostureResult con CPI, ángulos, curvatura y estado.
        """
        if timestamp is None:
            timestamp = time.time()

        if not detected or len(keypoints) < 9:
            return self._no_detection(timestamp, frame_id)

        # ── Extraer los 5 keypoints de la cadena posterior ──────────────────
        k0 = keypoints[0]  # Head-back / Occipital
        k1 = keypoints[1]  # C7
        k3 = keypoints[3]  # Back-backedge / Espalda media
        k4 = keypoints[4]  # Hips-backedge / Cadera
        k8 = keypoints[8]  # Shoulder-back / Escápula

        MIN_CONF = 0.1
        if any(kp[2] < MIN_CONF for kp in (k1, k3, k4, k8)):
            return self._no_detection(timestamp, frame_id)

        # ── 1. Ángulo lumbar ∠K8-K3-K4 ──────────────────────────────────────
        lumbar_deg = self._angle_at_vertex(
            k8[0], k8[1], k3[0], k3[1], k4[0], k4[1]
        )
        if lumbar_deg < 0:
            return self._no_detection(timestamp, frame_id)
        lumbar_deficit = max(0.0, 180.0 - lumbar_deg)

        # ── 2. Curvatura escapular: dist_⊥(K8, línea K1→K4) ────────────────
        curv_px = self._point_line_distance(
            k8[0], k8[1], k1[0], k1[1], k4[0], k4[1]
        )
        spine_len = self._vec_len(k1[0], k1[1], k4[0], k4[1])
        if curv_px < 0 or spine_len < 10:
            return self._no_detection(timestamp, frame_id)
        curvature_pct = (curv_px / spine_len) * 100.0

        # ── 3. CPI — Combined Posture Index ─────────────────────────────────
        cpi = lumbar_deficit * 2.0 + curvature_pct

        # ── 4. Ángulos auxiliares ───────────────────────────────────────────
        cerv_deg = self._angle_at_vertex(
            k0[0], k0[1], k1[0], k1[1], k8[0], k8[1]
        ) if k0[2] >= MIN_CONF else -1.0

        # ── 5. Confianza promedio en los 5 keypoints usados ──────────────────
        conf_vals = [k0[2], k1[2], k3[2], k4[2], k8[2]]
        conf_avg = sum(conf_vals) / len(conf_vals)

        # ── 6. Clasificación ─────────────────────────────────────────────────
        if cpi <= self.CPI_LEVE:
            status = PostureStatus.CORRECTO
        elif cpi <= self.CPI_CRITICO:
            status = PostureStatus.ALERTA_LEVE
        else:
            status = PostureStatus.ALERTA_CRITICA

        # ── 7. Contador de tiempo en mala postura ────────────────────────────
        now = timestamp
        if status in (PostureStatus.ALERTA_LEVE, PostureStatus.ALERTA_CRITICA):
            if self._bad_posture_start is None:
                self._bad_posture_start = now
                bad_accum = 0.0
            else:
                bad_accum = now - self._bad_posture_start
        else:
            self._bad_posture_start = None
            bad_accum = 0.0

        self._last_status = status
        self._last_update = now

        return PostureResult(
            timestamp=timestamp,
            frame_id=frame_id,
            status=status,
            cpi=round(cpi, 1),
            lumbar_angle_deg=round(lumbar_deg, 1),
            curvature_pct=round(curvature_pct, 1),
            spine_length_px=round(spine_len, 1),
            cervicodorsal_deg=round(cerv_deg, 1) if cerv_deg > 0 else 0.0,
            angle_deg=round(lumbar_deg, 1),  # compatibilidad: mostrar lumbar
            confidence=round(conf_avg, 4),
            bad_posture_accumulated_s=round(bad_accum, 1),
        )

    def _no_detection(self, timestamp: float, frame_id: int) -> PostureResult:
        elapsed = timestamp - self._last_update
        if elapsed > 2.0:
            self._bad_posture_start = None
        self._last_update = timestamp
        self._last_status = PostureStatus.NO_DETECTADO
        return PostureResult(
            timestamp=timestamp,
            frame_id=frame_id,
            status=PostureStatus.NO_DETECTADO,
        )

    def reset_counters(self) -> None:
        self._bad_posture_start = None
        self._last_status = PostureStatus.NO_DETECTADO
        self._last_update = time.time()
