"""
Component 3 — Dashboard Interactivo (app.py)

Interfaz gráfica con Gradio para el Sistema de Monitoreo Postural en Tiempo Real.

Funcionalidades:
- Visualización en tiempo real del video con overlay de 9 keypoints + esqueleto
- Nombres de keypoints visibles sobre cada punto
- Panel de métricas: ángulo cervicodorsal, estado postural, tiempo acumulado
- Alertas visuales (código de color) y sonoras (>30s en mala postura)
- Selector para cambiar entre los 4 mejores modelos

Arquitectura simplificada (streaming directo por frame):
  Webcam frame → YOLO inference → draw overlay → PostureAnalyzer → display

Autor: Sistema de Monitoreo Postural — Universidad Surcolombiana 2026
"""

from __future__ import annotations

import time
import winsound
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

# ── Detección automática de GPU ──────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Dispositivo de inferencia detectado: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from inference_engine import (
    KEYPOINT_NAMES,
    CRITICAL_KEYPOINT_INDICES,
    SKELETON_CONNECTIONS,
    COLORS_BGR,
    COLOR_SKELETON,
    COLOR_ANGLE_LINE,
    KeypointResult,
    draw_pose_overlay,
)
from posture_analyzer import PostureAnalyzer, PostureStatus

# ── Rutas de modelos ─────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\elkaw\Desktop\Modelos entrenados")
MODEL_CONFIGS = [
    {
        "name": "yolov8n_pose_b16_lr05 🚀 (Más rápido — 22ms)",
        "path": str(BASE_DIR / "yolov8n_pose_b16_lr05" / "weights" / "best.pt"),
        "key": "yolov8n",
    },
    {
        "name": "yolov5n_pose_b16_lr05 🎯 (Mejor detección — 95.2%)",
        "path": str(BASE_DIR / "yolov5n_pose_b16_lr05" / "weights" / "best.pt"),
        "key": "yolov5n",
    },
    {
        "name": "yolov26n_pose_b128_lr05 ⚖️ (Balanceado)",
        "path": str(BASE_DIR / "yolov26n_pose_b128_lr05" / "weights" / "best.pt"),
        "key": "yolov26n",
    },
    {
        "name": "yolov11n_pose_b16_lr01 ⭐ (Excel OKS)",
        "path": str(BASE_DIR / "yolov11n_pose_b16_lr01" / "weights" / "best.pt"),
        "key": "yolov11n",
    },
]

# Lookup rápido O(1) para evitar loop cada frame
MODEL_LOOKUP: dict[str, dict[str, str]] = {c["name"]: c for c in MODEL_CONFIGS}

# ── Estado global ────────────────────────────────────────────────────────────
class AppState:
    """Estado persistente de la aplicación (entre frames)."""

    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.model_key: str = "yolov8n"
        self.analyzer: PostureAnalyzer = PostureAnalyzer()
        self.bad_posture_start: Optional[float] = None
        self.last_alert_beep: float = 0.0
        self.frame_count: int = 0
        self._fps_times: list[float] = []  # Para medir FPS real
        self._current_fps: float = 0.0      # Último FPS medido
        self._cached_metrics: str = ""  # Cache HTML métricas
        self._cached_status: str = ""  # Cache HTML estado
        self._last_angle: float = -1.0
        self._last_status: str = ""
        self._last_bad_time: float = -1.0
        self._last_alert: bool = False

    def load_model(self, model_path: str) -> None:
        """Carga o recarga el modelo YOLO en GPU/CPU."""
        if self.model is None or model_path != getattr(self, "_loaded_path", None):
            print(f"[INFO] Cargando modelo: {model_path}")
            print(f"[INFO] Dispositivo: {DEVICE.upper()}")
            self.model = YOLO(model_path)
            self.model.to(DEVICE)  # Mover a GPU (ultralytics 8.x)
            self._loaded_path = model_path
            # Warmup con tamaño real para compilar kernels CUDA
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, imgsz=640)
            print(f"[INFO] Modelo cargado en {next(self.model.model.parameters()).device} ✓")
            print(f"[INFO] VRAM usada: {torch.cuda.memory_allocated()/1024**2:.0f} MB")


state = AppState()


# ── Función principal: procesar un frame de webcam ───────────────────────────
def process_frame(frame: np.ndarray, model_choice: str) -> tuple[np.ndarray, str, str]:
    """
    Procesa un frame de la webcam: YOLO inference + overlay + análisis postural.

    Args:
        frame: Imagen RGB desde la webcam (numpy array H×W×3).
        model_choice: Nombre del modelo seleccionado.

    Returns:
        (frame_con_overlay, HTML_métricas, HTML_estado)
    """
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank, _build_metrics_html(0, "NO DETECTADO", 0), _build_status_html("NO DETECTADO", 0, False)

    # Buscar modelo seleccionado (O(1) con dict precomputado)
    cfg = MODEL_LOOKUP.get(model_choice)
    if cfg is None:
        return frame, _build_metrics_html(0, "ERROR MODELO", 0), _build_status_html("ERROR", 0, False)
    model_path = cfg["path"]
    state.model_key = cfg["key"]

    # Cargar modelo si es necesario
    try:
        state.load_model(model_path)
    except Exception as e:
        return frame, _build_metrics_html(0, f"ERROR: {e}", 0), _build_status_html("ERROR", 0, False)

    # Convertir RGB → BGR para YOLO/OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── YOLO inference ──────────────────────────────────────────────────
    try:
        t_inf = time.time()
        preds = state.model(frame_bgr, verbose=False, conf=0.3, imgsz=640)
        inference_ms = (time.time() - t_inf) * 1000
    except Exception as e:
        return frame, _build_metrics_html(0, f"INFERENCIA ERROR: {e}", 0), _build_status_html("ERROR", 0, False)

    # ── Extraer keypoints ──────────────────────────────────────────────────
    if not preds or preds[0].keypoints is None:
        posture = state.analyzer.analyze(
            keypoints=[], detected=False, timestamp=time.time(), frame_id=state.frame_count
        )
        state.frame_count += 1
        # Dibujar "No detectado" sobre el frame
        out = frame_bgr.copy()
        cv2.putText(out, "Sin deteccion — Situate frente a la camara",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out_rgb, _build_metrics_html(0, "NO DETECTADO", 0), _build_status_html("NO DETECTADO", 0, False)

    kp_data = preds[0].keypoints.data.cpu().numpy()  # [N_personas, 9_kp, 3]

    if kp_data.shape[0] == 0:
        posture = state.analyzer.analyze(
            keypoints=[], detected=False, timestamp=time.time(), frame_id=state.frame_count
        )
        state.frame_count += 1
        out = frame_bgr.copy()
        cv2.putText(out, "Sin deteccion — Situate frente a la camara",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out_rgb, _build_metrics_html(0, "NO DETECTADO", 0), _build_status_html("NO DETECTADO", 0, False)

    # Seleccionar persona con mayor confianza
    confidences = kp_data[:, :, 2]
    best_idx = int(np.argmax(confidences.mean(axis=1)))
    raw_kps = kp_data[best_idx]  # [9, 3]

    # Construir lista de keypoints
    keypoints: list[list[float]] = []
    for i in range(min(9, len(raw_kps))):
        x, y, c = raw_kps[i]
        keypoints.append([float(x), float(y), float(c)])

    # Rellenar con ceros si menos de 9
    while len(keypoints) < 9:
        keypoints.append([0.0, 0.0, 0.0])

    # ── Análisis postural ──────────────────────────────────────────────────
    timestamp = time.time()
    state.frame_count += 1

    posture = state.analyzer.analyze(
        keypoints=keypoints,
        detected=True,
        timestamp=timestamp,
        frame_id=state.frame_count,
    )

    # Variables locales (usadas en banner y cache HTML)
    angle = posture.angle_deg
    bad = posture.bad_posture_accumulated_s
    stat_val = posture.status.value
    is_alert = posture.needs_alert

    # ── Dibujar overlay con keypoints + nombres ────────────────────────────
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Dibujar esqueleto
    for conn in SKELETON_CONNECTIONS:
        i_a, i_b = conn
        if i_a >= len(keypoints) or i_b >= len(keypoints):
            continue
        kp_a = keypoints[i_a]
        kp_b = keypoints[i_b]
        if kp_a[2] > 0.1 and kp_b[2] > 0.1:
            pt_a = (int(kp_a[0]), int(kp_a[1]))
            pt_b = (int(kp_b[0]), int(kp_b[1]))
            cv2.line(out, pt_a, pt_b, COLOR_SKELETON, 2, cv2.LINE_AA)

    # Dibujar keypoints con NOMBRES visibles
    for i, kp in enumerate(keypoints):
        if kp[2] <= 0.1:
            continue
        cx, cy = int(kp[0]), int(kp[1])
        color = COLORS_BGR[i] if i < len(COLORS_BGR) else (0, 255, 0)

        # Círculo del keypoint — más grande para K0, K6, K7
        radius = 7 if i in CRITICAL_KEYPOINT_INDICES else 4
        cv2.circle(out, (cx, cy), radius, color, -1, cv2.LINE_AA)

        # Contorno blanco para visibilidad
        cv2.circle(out, (cx, cy), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Nombre del keypoint — SIEMPRE visible
        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"K{i}"
        conf_text = f" {kp[2]:.0%}"

        # Posición del texto: arriba del punto, ajustado para no salir del frame
        text_x = cx + 10
        text_y = cy - 10
        if text_y < 15:
            text_y = cy + 20
        if text_x + 120 > w:
            text_x = cx - 130

        # Fondo oscuro para legibilidad
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45 if i in CRITICAL_KEYPOINT_INDICES else 0.35
        thickness = 1
        text = name + conf_text
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(out, (text_x - 2, text_y - th - 2), (text_x + tw + 2, text_y + 4), (0, 0, 0), -1)
        cv2.putText(out, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    # ── Dibujar líneas del ángulo ──────────────────────────────────────
    if posture.status != PostureStatus.NO_DETECTADO and posture.angle_deg > 0:
        k0 = keypoints[0]  # Occipital
        k1 = keypoints[1]  # Cervical C7 (pivote)
        k3 = keypoints[3]  # Borde dorsal

        if k0[2] > 0.1 and k1[2] > 0.1 and k3[2] > 0.1:
            p1 = (int(k1[0]), int(k1[1]))
            p0 = (int(k0[0]), int(k0[1]))
            p3 = (int(k3[0]), int(k3[1]))

            # Líneas del ángulo (naranja grueso)
            cv2.line(out, p1, p0, COLOR_ANGLE_LINE, 3, cv2.LINE_AA)
            cv2.line(out, p1, p3, COLOR_ANGLE_LINE, 3, cv2.LINE_AA)

            # Arco del ángulo
            angle_text = f"alpha = {posture.angle_deg:.1f} deg"
            cx_angle = int((p0[0] + p1[0] + p3[0]) / 3) - 60
            cy_angle = int((p0[1] + p1[1] + p3[1]) / 3)
            cv2.putText(out, angle_text, (cx_angle, cy_angle),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ANGLE_LINE, 2, cv2.LINE_AA)

    # ── Banner inferior de estado ──────────────────────────────────────────
    status_colors_bgr = {
        PostureStatus.CORRECTO: (0, 200, 0),
        PostureStatus.ALERTA_LEVE: (0, 215, 255),
        PostureStatus.ALERTA_CRITICA: (0, 0, 255),
        PostureStatus.NO_DETECTADO: (128, 128, 128),
    }
    banner_color = status_colors_bgr.get(posture.status, (128, 128, 128))

    # Banner inferior — multiplicación in-place (sin copia del frame)
    roi = out[h - 45:h, 0:w]
    np.multiply(roi, 0.55, out=roi, casting='unsafe')

    # ── Texto del banner inferior ────────────────────────────────────────
    fps_str = f"FPS: {state._current_fps:.0f}" if state._current_fps > 0 else ""
    status_text = f"Estado: {stat_val}"
    if angle > 0:
        status_text += f"  |  Angulo: {angle:.1f} deg"
    if bad > 0:
        status_text += f"  |  Mala postura: {bad:.0f}s"
    # FPS a la derecha del banner
    if fps_str:
        (tw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(out, fps_str, (w - tw - 10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(out, status_text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2, cv2.LINE_AA)

    # ── Alerta sonora (>30s mala postura, cada 5s) ────────────────────────
    if posture.needs_alert:
        now = time.time()
        if now - state.last_alert_beep > 5.0:
            try:
                winsound.Beep(1000, 300)  # 1000Hz, 300ms
            except Exception:
                pass
            state.last_alert_beep = now

    # Convertir BGR → RGB para Gradio
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # ── Log de FPS cada 30 frames ─────────────────────────────────────────
    state._fps_times.append(time.time())
    if len(state._fps_times) > 30:
        state._fps_times.pop(0)
    if len(state._fps_times) >= 2:
        elapsed = state._fps_times[-1] - state._fps_times[0]
        state._current_fps = (len(state._fps_times) - 1) / elapsed if elapsed > 0 else 0
    if state.frame_count % 30 == 0 and state._current_fps > 0:
        print(f"[FPS] Frame {state.frame_count}: {state._current_fps:.1f} fps | "
              f"inferencia: {inference_ms:.1f}ms GPU | "
              f"angulo: {angle:.1f}deg | {stat_val}")

    # ── Construir HTML (cacheado — solo regenera si cambió) ──────────────
    if angle != state._last_angle or stat_val != state._last_status or abs(bad - state._last_bad_time) > 0.5:
        state._cached_metrics = _build_metrics_html(angle, stat_val, bad)
        state._last_angle = angle
        state._last_bad_time = bad

    # Status HTML — solo cambia al cambiar estado o alerta
    if stat_val != state._last_status or is_alert != state._last_alert or abs(bad - state._last_bad_time) > 0.5:
        state._cached_status = _build_status_html(stat_val, bad, is_alert)
        state._last_status = stat_val
        state._last_alert = is_alert

    return (
        out_rgb,
        state._cached_metrics,
        state._cached_status,
    )


# ── HTML builders ────────────────────────────────────────────────────────────
def _build_metrics_html(angle: float, status: str, bad_time: float) -> str:
    """Construye HTML del panel de métricas con anillo SVG animado."""
    palette = {
        "CORRECTO":       ("#10b981", "badge-ok"),
        "ALERTA LEVE":    ("#f59e0b", "badge-warn"),
        "ALERTA CRÍTICA": ("#ef4444", "badge-crit"),
        "NO DETECTADO":   ("#94a3b8", "badge-nd"),
        "NO INICIADO":    ("#94a3b8", "badge-nd"),
    }
    color, badge_cls = palette.get(status, ("#94a3b8", "badge-nd"))

    # Anillo SVG: radio 52, circunferencia ~326.73
    r = 52
    c = 2 * 3.1416 * r
    pct = min(max(angle, 0), 100) / 100
    dash = c * pct

    return f"""
    <div class="pm-card" style="text-align:center;">
        <div class="pm-gauge-wrap">
            <svg width="130" height="130" viewBox="0 0 130 130">
                <circle class="pm-gauge-track" cx="65" cy="65" r="{r}"/>
                <circle class="pm-gauge-fill" cx="65" cy="65" r="{r}"
                    stroke="{color}"
                    stroke-dasharray="{c}"
                    stroke-dashoffset="{c - dash}"
                    style="color:{color}"/>
            </svg>
            <div class="pm-gauge-value" style="color:{color}">{angle:.1f}°</div>
        </div>
        <div class="pm-metric-label">Ángulo Mentoniano &nbsp;∠K2‑K1‑K6</div>
        <div class="pm-metric-sub">
            <span class="pm-badge {badge_cls}">{status}</span>
            &nbsp;&nbsp; Acumulado: <strong>{bad_time:.0f}s</strong> &nbsp;|&nbsp; Umbral: 30s
        </div>
    </div>
    """


def _build_status_html(status: str, bad_time: float, alert: bool) -> str:
    """Construye HTML del panel de estado y alertas con glow dinámico."""
    if alert:
        cls = "pm-status pm-status-crit pulse"
        icon = '<span class="pm-live-dot" style="background:#ef4444;box-shadow:0 0 8px #ef4444"></span>ALERTA CRÍTICA'
        detail = f"Mala postura acumulada: {bad_time:.0f}s · Corregí la posición de tu cabeza"
    elif status == "ALERTA CRÍTICA":
        cls = "pm-status pm-status-crit"
        icon = "ALERTA CRÍTICA"
        detail = f"Protrusión cefálica severa detectada · {bad_time:.0f}s acumulados"
    elif status == "ALERTA LEVE":
        cls = "pm-status pm-status-warn"
        icon = "ALERTA LEVE"
        detail = f"Cabeza ligeramente adelantada · {bad_time:.0f}s acumulados"
    elif status in ("NO DETECTADO", "NO INICIADO"):
        cls = "pm-status pm-status-nd"
        icon = status
        detail = "Posicionate frente a la cámara para iniciar el monitoreo"
    else:
        cls = "pm-status pm-status-ok"
        icon = '<span class="pm-live-dot"></span>POSTURA CORRECTA'
        detail = "Alineación cervical dentro de parámetros ergonómicos"

    return (
        f'<div class="{cls}" onmousemove="this.style.setProperty(\'--rx\',event.offsetX+\'px\');'
        f'this.style.setProperty(\'--ry\',event.offsetY+\'px\')">'
        f'<div style="font-size:15px;font-weight:700;position:relative;z-index:2">{icon}</div>'
        f'<div style="font-size:12px;margin-top:8px;opacity:.88;position:relative;z-index:2">{detail}</div></div>'
    )


# ── CSS y tema ──────────────────────────────────────────────────────────────
CSS = """
/* ═══════════════════════════════════════════════════════════════
   PALETA & VARIABLES
   ═══════════════════════════════════════════════════════════════ */
:root {
    --bg-deep:    #0b0f19;
    --bg-card:    rgba(17,24,39,0.65);
    --bg-elevated:#151d2e;
    --border:     rgba(148,163,184,0.10);
    --text-main:  #e2e8f0;
    --text-muted: #94a3b8;
    --accent-cyan:#06b6d4;
    --accent-teal:#14b8a6;
    --ok:         #10b981;
    --warn:       #f59e0b;
    --critical:   #ef4444;
    --radius:     16px;
    --glow-ok:    0 0 18px rgba(16,185,129,0.15);
    --glow-warn:  0 0 18px rgba(245,158,11,0.15);
    --glow-crit:  0 0 22px rgba(239,68,68,0.20);
}

/* ═══════════════════════════════════════════════════════════════
   GRADIENT MESH DE FONDO (sutil, no distrae)
   ═══════════════════════════════════════════════════════════════ */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    background:
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(6,182,212,0.035) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 60%, rgba(20,184,166,0.03) 0%, transparent 70%),
        var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
    animation: bg-drift 20s ease-in-out infinite alternate;
}
@keyframes bg-drift {
    0%   { background-position: 0% 0%, 100% 100%, 0 0; }
    100% { background-position: 3% 2%, 97% 98%, 0 0; }
}

footer, .gradio-footer { display: none !important; }

/* ═══════════════════════════════════════════════════════════════
   HEADER — Glassmorphism + aurora
   ═══════════════════════════════════════════════════════════════ */
.pm-header {
    position: relative;
    overflow: hidden;
    border-radius: var(--radius);
    padding: 32px 36px;
    margin-bottom: 24px;
    background: rgba(15,23,42,0.55);
    backdrop-filter: blur(16px) saturate(1.2);
    -webkit-backdrop-filter: blur(16px) saturate(1.2);
    border: 1px solid rgba(148,163,184,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04);
    animation: fade-in-up 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal), var(--ok));
    opacity: 0.8;
}
.pm-header::after {
    content: "";
    position: absolute;
    top: -60%; right: -10%; width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
    pointer-events: none;
    animation: aurora-float 8s ease-in-out infinite alternate;
}
@keyframes aurora-float {
    0%   { transform: translate(0,0) scale(1); opacity: 0.5; }
    100% { transform: translate(20px,10px) scale(1.08); opacity: 0.7; }
}
.pm-header h1 {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
    color: #f8fafc;
    position: relative;
    z-index: 2;
}
.pm-header p {
    margin: 0;
    font-size: 13.5px;
    color: var(--text-muted);
    line-height: 1.55;
    position: relative;
    z-index: 2;
}
.pm-header .brand-line {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 12px;
    padding: 4px 12px;
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.18);
    border-radius: 20px;
    font-size: 11px;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 0.7px;
    position: relative;
    z-index: 2;
    animation: fade-in-up 1s 0.15s cubic-bezier(0.16,1,0.3,1) both;
}

/* ═══════════════════════════════════════════════════════════════
   CARDS — Glass + glow sutil
   ═══════════════════════════════════════════════════════════════ */
.pm-card {
    background: var(--bg-card);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 26px;
    transition: transform 0.35s cubic-bezier(0.16,1,0.3,1),
                box-shadow 0.35s ease,
                border-color 0.35s ease;
    position: relative;
    overflow: hidden;
    animation: fade-in-up 0.7s 0.1s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: linear-gradient(160deg, rgba(255,255,255,0.06), transparent 40%, transparent 60%, rgba(255,255,255,0.03));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.4s ease;
}
.pm-card:hover::before { opacity: 1; }
.pm-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.35);
    border-color: rgba(148,163,184,0.18);
}

/* ═══════════════════════════════════════════════════════════════
   RING GAUGE — SVG circular animado
   ═══════════════════════════════════════════════════════════════ */
.pm-gauge-wrap {
    position: relative;
    width: 130px; height: 130px;
    margin: 0 auto 14px;
}
.pm-gauge-wrap svg {
    transform: rotate(-90deg);
    overflow: visible;
}
.pm-gauge-track {
    fill: none;
    stroke: rgba(148,163,184,0.12);
    stroke-width: 6;
    stroke-linecap: round;
}
.pm-gauge-fill {
    fill: none;
    stroke-width: 6;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.6s cubic-bezier(0.16,1,0.3,1), stroke 0.4s ease;
    filter: drop-shadow(0 0 6px currentColor);
}
.pm-gauge-value {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1px;
    font-variant-numeric: tabular-nums;
}

/* ═══════════════════════════════════════════════════════════════
   MÉTRICAS
   ═══════════════════════════════════════════════════════════════ */
.pm-metric-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 10px;
}
.pm-metric-sub {
    font-size: 12.5px;
    color: var(--text-muted);
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
}

/* ═══════════════════════════════════════════════════════════════
   STATUS PANEL — Glow dinámico según estado
   ═══════════════════════════════════════════════════════════════ */
.pm-status {
    border-radius: var(--radius);
    padding: 20px 24px;
    text-align: center;
    font-weight: 600;
    border: 1px solid transparent;
    transition: all 0.5s cubic-bezier(0.16,1,0.3,1);
    position: relative;
    overflow: hidden;
    animation: fade-in-up 0.7s 0.2s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-status-ok    {
    background: rgba(16,185,129,0.06);
    color: var(--ok);
    border-color: rgba(16,185,129,0.20);
    box-shadow: var(--glow-ok), inset 0 1px 0 rgba(255,255,255,0.03);
}
.pm-status-warn  {
    background: rgba(245,158,11,0.06);
    color: var(--warn);
    border-color: rgba(245,158,11,0.22);
    box-shadow: var(--glow-warn), inset 0 1px 0 rgba(255,255,255,0.03);
}
.pm-status-crit  {
    background: rgba(239,68,68,0.06);
    color: var(--critical);
    border-color: rgba(239,68,68,0.28);
    box-shadow: var(--glow-crit), inset 0 1px 0 rgba(255,255,255,0.03);
}
.pm-status-nd    {
    background: rgba(21,29,46,0.7);
    color: var(--text-muted);
    border-color: var(--border);
}

/* Ripple sutil en hover de status */
.pm-status::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at var(--rx,50%) var(--ry,50%), rgba(255,255,255,0.06) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
}
.pm-status:hover::after { opacity: 1; }

/* Pulse crítico — glow respirando */
@keyframes crit-breathe {
    0%, 100% { box-shadow: 0 0 18px rgba(239,68,68,0.12), 0 0 40px rgba(239,68,68,0.05); }
    50%      { box-shadow: 0 0 28px rgba(239,68,68,0.22), 0 0 55px rgba(239,68,68,0.10); }
}
.pm-status-crit.pulse { animation: crit-breathe 2.2s ease-in-out infinite; }

/* Scan-line sutil en alerta */
.pm-status-crit.pulse::before {
    content: "";
    position: absolute;
    top: 0; left: -100%; width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
    animation: scan-slide 2.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes scan-slide {
    0%   { left: -60%; }
    100% { left: 140%; }
}

/* ═══════════════════════════════════════════════════════════════
   LIVE DOT
   ═══════════════════════════════════════════════════════════════ */
.pm-live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--ok);
    box-shadow: 0 0 8px var(--ok);
    margin-right: 6px;
    animation: live-pulse 1.8s ease-in-out infinite;
    vertical-align: middle;
}
@keyframes live-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.7); }
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR INFO
   ═══════════════════════════════════════════════════════════════ */
.pm-sidebar-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #94a3b8;
    margin: 26px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    animation: fade-in-up 0.6s 0.3s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 12px;
    animation: fade-in-up 0.6s 0.35s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-table th {
    text-align: left;
    color: var(--text-muted);
    font-weight: 600;
    padding: 8px 6px;
    border-bottom: 1px solid var(--border);
}
.pm-table td {
    padding: 9px 6px;
    color: var(--text-main);
    border-bottom: 1px solid rgba(148,163,184,0.06);
    transition: background 0.25s ease;
}
.pm-table tr:hover td {
    background: rgba(148,163,184,0.04);
    border-radius: 6px;
}
.pm-table tr:last-child td { border-bottom: none; }

.pm-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
    transition: transform 0.2s ease;
}
.pm-badge:hover { transform: scale(1.04); }
.badge-ok   { background: rgba(16,185,129,0.14); color: var(--ok);   box-shadow: 0 0 10px rgba(16,185,129,0.08); }
.badge-warn { background: rgba(245,158,11,0.14); color: var(--warn); box-shadow: 0 0 10px rgba(245,158,11,0.08); }
.badge-crit { background: rgba(239,68,68,0.14);  color: var(--critical); box-shadow: 0 0 10px rgba(239,68,68,0.08); }

.pm-kp-list {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: 12.5px;
    line-height: 2.1;
    color: var(--text-muted);
    animation: fade-in-up 0.6s 0.4s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-kp-list li strong { color: var(--text-main); }
.pm-kp-k1 { color: var(--warn); font-weight: 700; }

.pm-note {
    font-size: 11.5px;
    color: var(--text-muted);
    padding: 12px 16px;
    background: rgba(21,29,46,0.6);
    border-radius: 10px;
    border-left: 3px solid var(--accent-cyan);
    margin-top: 14px;
    position: relative;
    overflow: hidden;
    animation: fade-in-up 0.6s 0.45s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-note::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent-cyan), var(--accent-teal));
}

/* ═══════════════════════════════════════════════════════════════
   ENTRADAS ANIMADAS
   ═══════════════════════════════════════════════════════════════ */
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Scrollbar elegante */
* { scrollbar-width: thin; scrollbar-color: rgba(148,163,184,0.15) transparent; }
*::-webkit-scrollbar { width: 5px; }
*::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.15); border-radius: 10px; }
*::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.25); }
"""

THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="*neutral_950",
    body_text_color="*neutral_200",
    body_text_color_subdued="*neutral_400",
    background_fill_primary="*neutral_900",
    background_fill_secondary="*neutral_800",
    border_color_accent="*neutral_700",
    border_color_primary="*neutral_800",
    color_accent="*primary_500",
    color_accent_soft="*primary_400",
    block_background_fill="*neutral_900",
    block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_800",
    block_shadow="none",
    block_title_text_color="*neutral_200",
    input_background_fill="*neutral_800",
    input_border_color="*neutral_700",
    input_border_color_focus="*primary_500",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_text_color="white",
    button_secondary_background_fill="*neutral_800",
    button_secondary_background_fill_hover="*neutral_700",
    stat_background_fill="*neutral_900",
)


# ── Construir UI ─────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    """Construye la interfaz Gradio completa."""

    with gr.Blocks(
        title="Monitoreo Postural — USCO 2026",
    ) as app:
        gr.HTML("""
        <div class="pm-header">
            <h1>Sistema de Monitoreo Postural en Tiempo Real</h1>
            <p>
                Estimación del ángulo mentoniano (∠K2‑K1‑K6) vía YOLO‑Pose y trigonometría vectorial.<br>
                Universidad Surcolombiana &nbsp;·&nbsp; Castañeda Guzmán &amp; Idarraga Plazas, 2026
            </p>
            <span class="brand-line">
                <span class="pm-live-dot"></span>
                Biomecánica Computacional — Procesamiento de Video
            </span>
        </div>
        """)

        with gr.Row():
            # ── Columna izquierda: Video ──────────────────────────────
            with gr.Column(scale=3):
                webcam = gr.Image(
                    sources=["webcam"],
                    label="Camara en Vivo",
                    height=480,
                    width=640,
                    streaming=True,
                )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=[c["name"] for c in MODEL_CONFIGS],
                        value=MODEL_CONFIGS[0]["name"],
                        label="Modelo YOLO-Pose",
                        info="Selecciona el modelo para inferencia",
                        interactive=True,
                    )

                model_info = gr.Markdown(
                    "**Modelo actual:** YOLOv8n — Mas rapido (22ms, SCORE 0.9189)"
                )

            # ── Columna derecha: Métricas ─────────────────────────────
            with gr.Column(scale=1):
                angle_display = gr.HTML(_build_metrics_html(0, "NO INICIADO", 0))
                status_display = gr.HTML(_build_status_html("NO INICIADO", 0, False))

                gr.HTML('<div class="pm-sidebar-title">Umbrales Ergonómicos</div>')
                gr.HTML("""
                <table class="pm-table">
                    <tr><th>Ángulo α</th><th>Estado</th><th>Significado clínico</th></tr>
                    <tr><td>α ≥ 80°</td><td><span class="pm-badge badge-ok">Correcto</span></td><td>Cabeza alineada, lordosis cervical neutral</td></tr>
                    <tr><td>70° ≤ α < 80°</td><td><span class="pm-badge badge-warn">Alerta leve</span></td><td>Cabeza adelantada, flexión cervical incipiente</td></tr>
                    <tr><td>α < 70°</td><td><span class="pm-badge badge-crit">Alerta crítica</span></td><td>Protrusión cefálica severa, encorvamiento marcado</td></tr>
                </table>
                """)

                gr.HTML('<div class="pm-sidebar-title">Keypoints del Ángulo</div>')
                gr.HTML("""
                <ul class="pm-kp-list">
                    <li><strong>K2</strong> — Occipital (protuberancia posterior de la cabeza)</li>
                    <li><strong class="pm-kp-k1">K1</strong> — Mentón · vértice del ángulo α</li>
                    <li><strong>K6</strong> — Apófisis espinos C7 (pivote cervical posterior)</li>
                </ul>
                """)

                gr.HTML('<div class="pm-sidebar-title">Alerta Sonora</div>')
                gr.HTML('<div class="pm-note">Se emite un beep cada 5 s cuando la mala postura supera 30 s de acumulación continua.</div>')

        # ── Evento streaming: cada frame de webcam → procesar ─────────
        webcam.stream(
            fn=process_frame,
            inputs=[webcam, model_dropdown],
            outputs=[webcam, angle_display, status_display],
        )

        model_dropdown.change(
            fn=lambda m: f"**Modelo seleccionado:** {m}",
            inputs=[model_dropdown],
            outputs=[model_info],
        )

        return app


# ── Limpieza de cache ────────────────────────────────────────────────────────
def _clear_gradio_cache() -> None:
    """Limpia cache residual de Gradio y libera memoria GPU."""
    # Limpiar memoria GPU de ejecuciones anteriores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("[Cache] GPU memory liberada — arranque limpio.")


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _clear_gradio_cache()

    # Precargar modelo default ANTES de arrancar servidor
    # Así el primer frame de webcam no tiene que esperar la carga + warmup
    print("[INIT] Precargando modelo default para arranque instantáneo...")
    state.load_model(MODEL_CONFIGS[0]["path"])
    print("[INIT] Modelo listo. Iniciando servidor Gradio...\n")

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        prevent_thread_lock=True,
        css=CSS,
        theme=THEME,
    )

    # Mantener el proceso vivo mientras el servidor corre
    import time as _time
    try:
        while True:
            _time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        app.close()
