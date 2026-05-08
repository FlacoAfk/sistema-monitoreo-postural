"""
Component 2 — Backend Matemático (posture_analyzer.py)

Recibe coordenadas de los 9 keypoints del torso, calcula el ángulo de flexión
cervicodorsal θ mediante trigonometría vectorial (producto punto) y clasifica
la postura según umbrales ergonómicos.

Mapeo REAL verificado visualmente (2026-05-08):
    K0 = Head-back / Occipital (parte posterior de la cabeza)
    K1 = Neck-back / Cervical posterior C7  ← PIVOTE ⚠
    K2 = Shoulder-top / Acromion
    K3 = Back-backedge / Borde dorsal (espalda)
    K4 = Hips-backedge / Cadera
    K5 = Neck-middle / Cervical media
    K6 = Jaw / Mandíbula
    K7 = Chin / Mentón
    K8 = Shoulder-back / Zona escapular

Fórmula:
    u = K0_Occipital − K1_C7          (vector cefálico)
    v = K3_BordeDorsal − K1_C7        (vector dorsolumbar)
    θ = ∠(K1→K0, K1→K3)
    cos(θ) = (u·v) / (|u| × |v|)
    θ = arccos(cos(θ))  en radianes
    α = 180° − θ  (ángulo de flexión cervicodorsal)
    α bajo → head forward posture (protrusión cefálica)

Clasificación:
    α ≤ 15°  → CORRECTO (verde)
    15° < α ≤ 25° → ALERTA LEVE (amarillo)
    α > 25°  → ALERTA CRÍTICA (rojo)

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
        """Color hexadecimal asociado al estado."""
        return {
            PostureStatus.CORRECTO: "#00FF00",
            PostureStatus.ALERTA_LEVE: "#FFD700",
            PostureStatus.ALERTA_CRITICA: "#FF0000",
            PostureStatus.NO_DETECTADO: "#808080",
        }[self]

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        """Color BGR (OpenCV) asociado al estado."""
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
    angle_deg: float = 0.0
    angle_rad: float = 0.0
    confidence: float = 0.0  # Confianza promedio de los keypoints críticos (K0, K1, K3)
    # Tiempo acumulado en postura inadecuada (ALERTA LEVE o CRÍTICA) en segundos
    bad_posture_accumulated_s: float = 0.0
    # Detalles de vectores para debug
    u_vector: tuple[float, float] = (0.0, 0.0)
    v_vector: tuple[float, float] = (0.0, 0.0)

    @property
    def needs_alert(self) -> bool:
        """Indica si se debe emitir alerta (>30s continuos en mala postura)."""
        return self.bad_posture_accumulated_s > 30.0 and self.status in (
            PostureStatus.ALERTA_LEVE,
            PostureStatus.ALERTA_CRITICA,
        )


class PostureAnalyzer:
    """
    Analizador postural determinista — sin ML, pura trigonometría vectorial.

    Calcula el ángulo de flexión cervicodorsal a partir de 3 keypoints críticos:
    K0 (Occipital), K1 (Cervical posterior C7, pivote), K3 (Borde dorsal).

    Fórmula: θ = ∠(K1→K0, K1→K3)

    Mantiene un contador de tiempo acumulado en postura inadecuada para
    el sistema de alertas (>30s continuos dispara notificación).
    """

    # Umbrales ergonómicos (en grados)
    THRESHOLD_LEVE: float = 15.0   # α > 15° → ALERTA LEVE
    THRESHOLD_CRITICO: float = 25.0  # α > 25° → ALERTA CRÍTICA
    ALERT_CONTINUOUS_SECONDS: float = 30.0  # Tiempo continuo para disparar alerta

    def __init__(self) -> None:
        """Inicializa el analizador con contadores en cero."""
        self._bad_posture_start: Optional[float] = None  # Timestamp cuando empezó mala postura
        self._last_status: PostureStatus = PostureStatus.NO_DETECTADO
        self._last_update: float = time.time()

    def analyze(
        self,
        keypoints: list[list[float]],
        detected: bool,
        timestamp: Optional[float] = None,
        frame_id: int = 0,
    ) -> PostureResult:
        """
        Analiza una detección de pose y calcula el ángulo cervicodorsal.

        Args:
            keypoints: Lista de 9 keypoints [[x, y, conf], ...].
            detected: Si hubo detección de persona.
            timestamp: Timestamp del frame (usa time.time() si es None).
            frame_id: ID secuencial del frame.

        Returns:
            PostureResult con ángulo, estado y tiempo acumulado.
        """
        if timestamp is None:
            timestamp = time.time()

        # Verificar que tenemos keypoints válidos
        if not detected or len(keypoints) < 9:
            return self._no_detection(timestamp, frame_id)

        # Extraer keypoints críticos (topología verificada 2026-05-08)
        k0_occ = keypoints[0]     # K0: Occipital (parte posterior cabeza): [x, y, conf]
        k1_cerv = keypoints[1]    # K1: Cervical posterior C7 (PIVOTE): [x, y, conf]
        k3_dorsal = keypoints[3]  # K3: Borde dorsal / Espalda: [x, y, conf]

        # Verificar confianza mínima en los 3 keypoints críticos
        MIN_CONF = 0.1
        if k0_occ[2] < MIN_CONF or k1_cerv[2] < MIN_CONF or k3_dorsal[2] < MIN_CONF:
            return self._no_detection(timestamp, frame_id)

        # ── Cálculo del ángulo cervicodorsal ─────────────────────────────────
        # Vector u = K0 − K1  (C7 → Occipital = cefálico)
        ux = k0_occ[0] - k1_cerv[0]
        uy = k0_occ[1] - k1_cerv[1]

        # Vector v = K3 − K1  (C7 → BordeDorsal = dorsolumbar)
        vx = k3_dorsal[0] - k1_cerv[0]
        vy = k3_dorsal[1] - k1_cerv[1]

        # Magnitudes
        mag_u = math.sqrt(ux * ux + uy * uy)
        mag_v = math.sqrt(vx * vx + vy * vy)

        if mag_u < 1e-6 or mag_v < 1e-6:
            return self._no_detection(timestamp, frame_id)

        # Producto punto y ángulo
        dot = ux * vx + uy * vy
        cos_theta = dot / (mag_u * mag_v)
        # Clampear por errores de punto flotante
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)

        # Ángulo de flexión cervicodorsal α = 180° − θ
        alpha_flexion_deg = 180.0 - theta_deg

        # Confianza promedio de los keypoints críticos
        conf_critical = (k0_occ[2] + k1_cerv[2] + k3_dorsal[2]) / 3.0

        # ── Clasificación ─────────────────────────────────────────────────────
        if alpha_flexion_deg <= self.THRESHOLD_LEVE:
            status = PostureStatus.CORRECTO
        elif alpha_flexion_deg <= self.THRESHOLD_CRITICO:
            status = PostureStatus.ALERTA_LEVE
        else:
            status = PostureStatus.ALERTA_CRITICA

        # ── Contador de tiempo en mala postura ────────────────────────────────
        bad_accumulated = 0.0
        now = timestamp

        if status in (PostureStatus.ALERTA_LEVE, PostureStatus.ALERTA_CRITICA):
            if self._bad_posture_start is None:
                # Inicia nuevo período de mala postura
                self._bad_posture_start = now
                bad_accumulated = 0.0
            else:
                # Ya estaba en mala postura → acumular tiempo
                bad_accumulated = now - self._bad_posture_start
        else:
            # Postura correcta → resetear contador
            self._bad_posture_start = None
            bad_accumulated = 0.0

        self._last_status = status
        self._last_update = now

        return PostureResult(
            timestamp=timestamp,
            frame_id=frame_id,
            status=status,
            angle_deg=round(alpha_flexion_deg, 2),
            angle_rad=round(theta_rad, 4),
            confidence=round(conf_critical, 4),
            bad_posture_accumulated_s=round(bad_accumulated, 1),
            u_vector=(round(ux, 2), round(uy, 2)),
            v_vector=(round(vx, 2), round(vy, 2)),
        )

    def _no_detection(self, timestamp: float, frame_id: int) -> PostureResult:
        """Construye resultado para frame sin detección válida."""
        # NO reseteamos el contador en una pérdida momentánea (1-2 frames)
        # Solo si pasan más de 2 segundos sin detección
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
        """Reinicia todos los contadores internos."""
        self._bad_posture_start = None
        self._last_status = PostureStatus.NO_DETECTADO
        self._last_update = time.time()
