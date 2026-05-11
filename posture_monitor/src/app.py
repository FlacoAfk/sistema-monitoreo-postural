"""
Component 3 — Dashboard Interactivo (app.py)

Interfaz gráfica con Gradio para el Sistema de Monitoreo Postural en Tiempo Real.

Funcionalidades:
- Visualización en tiempo real del video con overlay limpio: keypoints (K# IDs) + esqueleto + líneas CPI
- Banner de video minimalista: solo FPS (estado/métricas en panel lateral)
- Panel de métricas lateral: CPI gauge, ángulo cervicodorsal, estado postural, tiempo acumulado
- Tabla de keypoints completa (9 pts) con ID + nombre anatómico en sidebar
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

# Skip ratio: process every Nth frame through YOLO, skip remaining
SKIP_RATIO = 3

# ── Suavizado EMA para keypoints (anti-flicker) ───────────────────────
EMA_ALPHA: float = 0.35          # Factor de suavizado (0=sin cambio, 1=sin suavizar)
KP_GRACE_FRAMES: int = 8         # Frames de gracia antes de descartar un keypoint perdido
ALERT_POPUP_DURATION_S: float = 4.0  # Duración del popup de alerta (segundos)
MAX_PERSONS: int = 6             # Detección máxima de personas

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
        self._skip_counter: int = 0
        self._last_overlay_bgr: Optional[np.ndarray] = None
        self._last_posture_result: Optional[tuple] = None
        self._smoothed_kps: Optional[np.ndarray] = None  # [9,3] EMA-smoothed keypoints
        self._kp_missing_count: np.ndarray = np.zeros(9, dtype=int)  # Grace counter per kp
        self._alert_popup_until: float = 0.0  # Timestamp until popup is visible

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

    # ── Frame skipping: skip every Nth frame (save YOLO inference) ────
    state._skip_counter += 1
    if state._skip_counter % SKIP_RATIO != 0 and state._last_overlay_bgr is not None:
        # Skip path: reuse overlay from last inference frame
        out = state._last_overlay_bgr.copy()
        h, w = out.shape[:2]
        cpi_s, stat_s, lumbar_s, curv_s, bad_s, alert_s = state._last_posture_result or (0, "NO DETECTADO", 0, 0, 0, False)

        # Banner inferior — solo FPS (skip path)
        roi = out[h - 32:h, 0:w]
        np.multiply(roi, 0.5, out=roi, casting='unsafe')

        fps_str = f"FPS: {state._current_fps:.0f}" if state._current_fps > 0 else ""
        if fps_str:
            (tw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(out, fps_str, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        # FPS tracking: include skip frames for accurate display rate
        state._fps_times.append(time.time())
        if len(state._fps_times) > 30:
            state._fps_times.pop(0)
        if len(state._fps_times) >= 2:
            elapsed = state._fps_times[-1] - state._fps_times[0]
            state._current_fps = (len(state._fps_times) - 1) / elapsed if elapsed > 0 else 0

        return out_rgb, _build_metrics_html(cpi_s, stat_s, bad_s, cpi_s, lumbar_s, curv_s), _build_status_html(stat_s, bad_s, alert_s)

    # Convertir RGB → BGR para YOLO/OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── YOLO inference ──────────────────────────────────────────────────
    try:
        t_inf = time.time()
        preds = state.model(frame_bgr, verbose=False, conf=0.25, imgsz=416, max_det=MAX_PERSONS)
        inference_ms = (time.time() - t_inf) * 1000
    except Exception as e:
        return frame, _build_metrics_html(0, f"INFERENCIA ERROR: {e}", 0), _build_status_html("ERROR", 0, False)

    # ── Extraer keypoints (multi-persona) + EMA smoothing ────────────
    if not preds or preds[0].keypoints is None:
        posture = state.analyzer.analyze(
            keypoints=[], detected=False, timestamp=time.time(), frame_id=state.frame_count
        )
        state.frame_count += 1
        out = frame_bgr
        cv2.putText(out, "Sin deteccion — Situate frente a la camara",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        state._last_overlay_bgr = out.copy()
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out_rgb, _build_metrics_html(0, "NO DETECTADO", 0), _build_status_html("NO DETECTADO", 0, False)

    kp_data = preds[0].keypoints.data.cpu().numpy()  # [N_personas, 9_kp, 3]

    if kp_data.shape[0] == 0:
        posture = state.analyzer.analyze(
            keypoints=[], detected=False, timestamp=time.time(), frame_id=state.frame_count
        )
        state.frame_count += 1
        out = frame_bgr
        cv2.putText(out, "Sin deteccion — Situate frente a la camara",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        state._last_overlay_bgr = out.copy()
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out_rgb, _build_metrics_html(0, "NO DETECTADO", 0), _build_status_html("NO DETECTADO", 0, False)

    # ── Seleccionar persona principal (mayor confianza promedio) ────────
    confidences = kp_data[:, :, 2]
    best_idx = int(np.argmax(confidences.mean(axis=1)))
    raw_kps = kp_data[best_idx]  # [9, 3]

    # ── Construir keypoints con EMA smoothing (anti-flicker) ──────────
    keypoints: list[list[float]] = []
    for i in range(min(9, len(raw_kps))):
        x_new, y_new, c_new = float(raw_kps[i][0]), float(raw_kps[i][1]), float(raw_kps[i][2])
        if c_new < 0.1:
            # Keypoint no detectado — usar último suavizado si está en grace period
            if state._smoothed_kps is not None and state._kp_missing_count[i] < KP_GRACE_FRAMES:
                x_s, y_s, c_s = state._smoothed_kps[i]
                x_new, y_new, c_new = x_s, y_s, c_s * 0.85  # Decaer confianza gradualmente
            else:
                x_new, y_new, c_new = 0.0, 0.0, 0.0
            state._kp_missing_count[i] += 1
        else:
            state._kp_missing_count[i] = 0  # Reset grace counter

        # EMA: mezcla entre valor nuevo y anterior suavizado
        if state._smoothed_kps is not None and state._smoothed_kps[i][2] > 0.1:
            x_prev, y_prev, c_prev = state._smoothed_kps[i]
            alpha = EMA_ALPHA
            x_new = alpha * x_new + (1 - alpha) * x_prev
            y_new = alpha * y_new + (1 - alpha) * y_prev
            c_new = max(c_new, alpha * c_new + (1 - alpha) * c_prev)

        keypoints.append([x_new, y_new, c_new])

    while len(keypoints) < 9:
        keypoints.append([0.0, 0.0, 0.0])

    # Actualizar estado EMA
    state._smoothed_kps = np.array(keypoints, dtype=np.float32)

    # ── Preparar datos de todas las personas para overlay multi-persona ─
    all_persons_kps = []  # Lista de (keypoints_list, is_primary)
    for p_idx in range(kp_data.shape[0]):
        is_primary = (p_idx == best_idx)
        person_kps = []
        for k in range(min(9, kp_data.shape[1])):
            person_kps.append([float(kp_data[p_idx][k][0]), float(kp_data[p_idx][k][1]), float(kp_data[p_idx][k][2])])
        while len(person_kps) < 9:
            person_kps.append([0.0, 0.0, 0.0])
        all_persons_kps.append((person_kps, is_primary))

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
    cpi = posture.cpi
    lumbar = posture.lumbar_angle_deg
    curv = posture.curvature_pct
    bad = posture.bad_posture_accumulated_s
    stat_val = posture.status.value
    is_alert = posture.needs_alert

    # ── Dibujar overlay para todas las personas ─────────────────────────
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for person_kps, is_primary in all_persons_kps:
        # ── Esqueleto ──
        for conn in SKELETON_CONNECTIONS:
            i_a, i_b = conn
            if i_a >= len(person_kps) or i_b >= len(person_kps):
                continue
            kp_a = person_kps[i_a]
            kp_b = person_kps[i_b]
            if kp_a[2] > 0.1 and kp_b[2] > 0.1:
                pt_a = (int(kp_a[0]), int(kp_a[1]))
                pt_b = (int(kp_b[0]), int(kp_b[1]))
                cv2.line(out, pt_a, pt_b, COLOR_SKELETON, 2, cv2.LINE_AA)

        # ── Keypoints — solo ID sutil ──
        for i, kp in enumerate(person_kps):
            if kp[2] <= 0.1:
                continue
            cx, cy = int(kp[0]), int(kp[1])
            color = COLORS_BGR[i] if i < len(COLORS_BGR) else (0, 255, 0)
            radius = 6 if i in CRITICAL_KEYPOINT_INDICES else 3
            if not is_primary:
                radius = max(radius - 1, 2)  # Personas secundarias: más chicos
            cv2.circle(out, (cx, cy), radius, color, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
            label = f"K{i}"
            if is_primary:
                cv2.putText(out, label, (cx + 8, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Líneas del ángulo lumbar (solo persona principal) ──
        if is_primary and posture.status != PostureStatus.NO_DETECTADO and posture.lumbar_angle_deg > 0:
            k8_scapula = keypoints[8]
            k3_back = keypoints[3]
            k4_hips = keypoints[4]
            if k8_scapula[2] > 0.1 and k3_back[2] > 0.1 and k4_hips[2] > 0.1:
                p_scap = (int(k8_scapula[0]), int(k8_scapula[1]))
                p_mid = (int(k3_back[0]), int(k3_back[1]))
                p_hip = (int(k4_hips[0]), int(k4_hips[1]))
                cv2.line(out, p_mid, p_scap, COLOR_ANGLE_LINE, 2, cv2.LINE_AA)
                cv2.line(out, p_mid, p_hip, COLOR_ANGLE_LINE, 2, cv2.LINE_AA)

    # ── Banner inferior — solo FPS ──────────────────────────────────────
    roi = out[h - 32:h, 0:w]
    np.multiply(roi, 0.5, out=roi, casting='unsafe')

    fps_str = f"FPS: {state._current_fps:.0f}" if state._current_fps > 0 else ""
    if fps_str:
        (tw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(out, fps_str, (w - tw - 10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Alerta sonora + popup visual (>30s mala postura) ──────────────
    if posture.needs_alert:
        now = time.time()
        if now - state.last_alert_beep > 5.0:
            try:
                winsound.Beep(1000, 300)  # 1000Hz, 300ms
            except Exception:
                pass
            state.last_alert_beep = now
        # Activar popup visual por ALERT_POPUP_DURATION_S segundos
        state._alert_popup_until = now + ALERT_POPUP_DURATION_S

    # ── Dibujar popup de alerta si está activo ─────────────────────────
    if time.time() < state._alert_popup_until:
        # Popup semi-transparente en esquina superior derecha
        popup_w, popup_h = min(300, w - 20), 70
        popup_x = w - popup_w - 15
        popup_y = 15
        overlay_bg = out[popup_y:popup_y+popup_h, popup_x:popup_x+popup_w].copy()
        cv2.rectangle(overlay_bg, (0, 0), (popup_w, popup_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay_bg, 0.75, out[popup_y:popup_y+popup_h, popup_x:popup_x+popup_w], 0.25, 0,
            out[popup_y:popup_y+popup_h, popup_x:popup_x+popup_w])
        cv2.rectangle(out, (popup_x, popup_y), (popup_x+popup_w, popup_y+popup_h), (0, 0, 255), 2, cv2.LINE_AA)
        bad_s = posture.bad_posture_accumulated_s
        cv2.putText(out, f"ALERTA: Mala postura {bad_s:.0f}s",
            (popup_x + 10, popup_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, "Corregi tu posicion",
            (popup_x + 10, popup_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Store overlay + posture for frame skip re-use
    state._last_overlay_bgr = out.copy()
    state._last_posture_result = (cpi, stat_val, lumbar, curv, bad, is_alert)

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
              f"CPI: {cpi:.0f} | {stat_val}")

    # Build HTML — always fresh with data-* attrs; JS handles DOM updates
    metrics_html = _build_metrics_html(cpi, stat_val, bad, cpi, lumbar, curv)
    status_html = _build_status_html(stat_val, bad, is_alert)

    return (
        out_rgb,
        metrics_html,
        status_html,
    )


# ── HTML builders ────────────────────────────────────────────────────────────
def _build_metrics_html(angle: float, status: str, bad_time: float,
                         cpi: float = 0, lumbar: float = 0, curv: float = 0) -> str:
    """Construye HTML del panel de métricas — estructura estable + JS inject."""
    palette = {
        "CORRECTO":       ("#10b981", "badge-ok"),
        "ALERTA LEVE":    ("#f59e0b", "badge-warn"),
        "ALERTA CRÍTICA": ("#ef4444", "badge-crit"),
        "NO DETECTADO":   ("#94a3b8", "badge-nd"),
        "NO INICIADO":    ("#94a3b8", "badge-nd"),
    }
    color, badge_cls = palette.get(status, ("#94a3b8", "badge-nd"))

    r = 52
    circumference = 2 * 3.1416 * r
    pct = min(max(cpi, 0), 100) / 100
    offset = circumference - circumference * pct

    return f"""<div class="pm-card" style="text-align:center;" data-cpi="{cpi}" data-color="{color}" data-lumbar="{lumbar}" data-curv="{curv}" data-status="{status}" data-bad-time="{bad_time}" data-badge="{badge_cls}">
    <div class="pm-gauge-wrap">
      <svg width="130" height="130" viewBox="0 0 130 130">
        <circle class="pm-gauge-track" cx="65" cy="65" r="52"/>
        <circle class="pm-gauge-fill" cx="65" cy="65" r="52" id="pm-gauge-arc"
          stroke="{color}" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" style="color:{color}"/>
      </svg>
      <div class="pm-gauge-value" id="pm-gauge-num" style="color:{color}">{cpi:.1f}</div>
    </div>
    <div class="pm-metric-label">CPI — Combined Posture Index</div>
    <div class="pm-metric-sub" id="pm-metric-sub">
      <span class="pm-badge {badge_cls}" id="pm-badge">{status}</span>
      &nbsp;|&nbsp; Lumbar: <strong id="pm-lumbar">{lumbar:.0f}°</strong>
      &nbsp;|&nbsp; Curv: <strong id="pm-curv">{curv:.1f}%</strong>
      &nbsp;|&nbsp; Acum: <strong id="pm-bad-time">{bad_time:.0f}s</strong>
    </div>
  </div>
  <script>
  (function(){{
    var el = document.currentScript.previousElementSibling;
    var cpi = parseFloat(el.dataset.cpi);
    var color = el.dataset.color;
    var lumbar = el.dataset.lumbar;
    var curv = el.dataset.curv;
    var status = el.dataset.status;
    var badge = el.dataset.badge;
    var badTime = el.dataset.badTime;
    var r = 52, c = 2 * 3.1416 * r;
    var pct = Math.min(Math.max(cpi, 0), 100) / 100;
    var dash = c * pct;
    var arc = el.querySelector('#pm-gauge-arc');
    var num = el.querySelector('#pm-gauge-num');
    var badgeEl = el.querySelector('#pm-badge');
    var lumbarEl = el.querySelector('#pm-lumbar');
    var curvEl = el.querySelector('#pm-curv');
    var badTimeEl = el.querySelector('#pm-bad-time');
    if (arc) {{ arc.setAttribute('stroke-dashoffset', String(c - dash)); arc.setAttribute('stroke', color); }}
    if (num) {{ num.textContent = cpi.toFixed(1); num.style.color = color; }}
    if (badgeEl) {{ badgeEl.textContent = status; badgeEl.className = 'pm-badge ' + badge; }}
    if (lumbarEl) lumbarEl.textContent = lumbar + String.fromCharCode(176);
    if (curvEl) curvEl.textContent = parseFloat(curv).toFixed(1) + '%';
    if (badTimeEl) badTimeEl.textContent = badTime + 's';
  }})();
  </script>"""


def _build_status_html(status: str, bad_time: float, alert: bool) -> str:
    """Construye HTML del panel de estado — estructura estable + JS inject."""
    cls_map_safe = {"CORRECTO":"ok","ALERTA LEVE":"warn","ALERTA CRÍTICA":"crit","NO DETECTADO":"nd","NO INICIADO":"nd"}
    cls = cls_map_safe.get(status, "nd")
    return f"""<div class="pm-status pm-status-{cls}" data-status="{status}" data-bad-time="{bad_time}" data-alert="{str(alert)}" onmousemove="this.style.setProperty('--rx',event.offsetX+'px');this.style.setProperty('--ry',event.offsetY+'px')">
    <div style="font-size:15px;font-weight:700;position:relative;z-index:2" id="pm-status-icon">{status}</div>
    <div style="font-size:12px;margin-top:8px;opacity:.88;position:relative;z-index:2" id="pm-status-detail">...</div>
  </div>
  <script>
  (function(){{
    var el = document.currentScript.previousElementSibling;
    var status = el.dataset.status;
    var badTime = el.dataset.badTime;
    var isAlert = el.dataset.alert === 'True';
    var clsMap = {{'CORRECTO':'pm-status-ok','ALERTA LEVE':'pm-status-warn','ALERTA CRÍTICA':'pm-status-crit','NO DETECTADO':'pm-status-nd','NO INICIADO':'pm-status-nd'}};
    var cls = clsMap[status] || 'pm-status-nd';
    if (isAlert && (status === 'ALERTA CRÍTICA')) cls += ' pulse';
    el.className = 'pm-status ' + cls;
    var iconEl = el.querySelector('#pm-status-icon');
    var detailEl = el.querySelector('#pm-status-detail');
    if (iconEl && detailEl) {{
      if (isAlert) {{
        iconEl.innerHTML = '<span class="pm-live-dot" style="background:#ef4444;box-shadow:0 0 8px #ef4444"></span>ALERTA CRÍTICA';
        detailEl.textContent = 'Mala postura acumulada: ' + badTime + 's · Corregí la posición de tu cabeza';
      }} else if (status === 'ALERTA CRÍTICA') {{
        iconEl.textContent = 'ALERTA CRÍTICA';
        detailEl.textContent = 'Protrusión cefálica severa detectada · ' + badTime + 's acumulados';
      }} else if (status === 'ALERTA LEVE') {{
        iconEl.textContent = 'ALERTA LEVE';
        detailEl.textContent = 'Cabeza ligeramente adelantada · ' + badTime + 's acumulados';
      }} else if (status === 'NO DETECTADO' || status === 'NO INICIADO') {{
        iconEl.textContent = status;
        detailEl.textContent = 'Posicionate frente a la cámara para iniciar el monitoreo';
      }} else {{
        iconEl.innerHTML = '<span class="pm-live-dot"></span>POSTURA CORRECTA';
        detailEl.textContent = 'Alineación cervical dentro de parámetros ergonómicos';
      }}
    }}
  }})();
  </script>"""


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
}
.pm-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 12px;
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
                Estimación del Combined Posture Index (CPI) — curvatura escapular + ángulo lumbar.<br>
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
                angle_display = gr.HTML(_build_metrics_html(0, "NO INICIADO", 0, 0, 0, 0))
                status_display = gr.HTML(_build_status_html("NO INICIADO", 0, False))

                gr.HTML('<div class="pm-sidebar-title">Umbrales CPI</div>')
                gr.HTML("""
                <table class="pm-table">
                    <tr><th>CPI</th><th>Estado</th><th>Significado</th></tr>
                    <tr><td>CPI ≤ 35</td><td><span class="pm-badge badge-ok">Correcto</span></td><td>Columna alineada, postura recta</td></tr>
                    <tr><td>35 < CPI ≤ 50</td><td><span class="pm-badge badge-warn">Alerta leve</span></td><td>Curvatura dorsal leve</td></tr>
                    <tr><td>CPI > 50</td><td><span class="pm-badge badge-crit">Alerta critica</span></td><td>Cifosis / hombros caidos</td></tr>
                </table>
                """)

                gr.HTML('<div class="pm-sidebar-title">Keypoints del CPI (5 pts)</div>')
                gr.HTML("""
                <ul class="pm-kp-list">
                    <li><strong>K0</strong> — Head-back / Occipital</li>
                    <li><strong>K1</strong> — C7 / Neck-back</li>
                    <li><strong>K8</strong> — Shoulder-back / Escapula</li>
                    <li><strong>K3</strong> — Back-backedge / Espalda media</li>
                    <li><strong>K4</strong> — Hips-backedge / Cadera</li>
                </ul>
                """)

                gr.HTML('<div class="pm-sidebar-title">Mapa de Keypoints (9 pts)</div>')
                gr.HTML("""
                    <table class="pm-table">
                    <tr><th>ID</th><th>Nombre</th><th>Ubicacion</th></tr>
                    <tr><td><strong>K0</strong></td><td>Head-back</td><td>Occipital</td></tr>
                    <tr><td><strong>K1</strong></td><td>Neck-back</td><td>C7 cervical</td></tr>
                    <tr><td><strong>K2</strong></td><td>Shoulder-top</td><td>Acromion</td></tr>
                    <tr><td><strong>K3</strong></td><td>Back-borde</td><td>Espalda media</td></tr>
                    <tr><td><strong>K4</strong></td><td>Hips-backedge</td><td>Cadera</td></tr>
                    <tr><td><strong>K5</strong></td><td>Neck-middle</td><td>Cervical media</td></tr>
                    <tr><td><strong>K6</strong></td><td>Jaw</td><td>Mandibula</td></tr>
                    <tr><td><strong>K7</strong></td><td>Chin</td><td>Menton</td></tr>
                    <tr><td><strong>K8</strong></td><td>Shoulder-back</td><td>Escapula</td></tr>
                    </table>
                """)

                gr.HTML('<div class="pm-sidebar-title">Alerta Sonora</div>')
                gr.HTML('<div class="pm-note">Se emite un beep cada 5 s cuando la mala postura supera 30 s de acumulación continua.</div>')

        # ── Evento streaming: cada frame de webcam → procesar ─────────
        webcam.stream(
            fn=process_frame,
            inputs=[webcam, model_dropdown],
            outputs=[webcam, angle_display, status_display],
            stream_every=0.05,
            time_limit=None,
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
