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

import csv
import tempfile
import time
import winsound
from datetime import datetime
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
MAX_PERSONS: int = 6 # Detección máxima de personas

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
        self._kp_missing_count: np.ndarray = np.zeros(9, dtype=int) # Grace counter per kp

        # ── Session recording ────────────────────────────────────────
        self.session_data: list[dict] = []
        self.session_active: bool = False
        self.session_start_time: Optional[float] = None
        self.session_frame_counter: int = 0

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




def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calcula IoU entre dos cajas [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_persons(boxes: np.ndarray, kp_data: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """NMS sobre detecciones de persona: elimina boxes con IoU > iou_thresh contra la mejor.

    Args:
        boxes: [N, 4] cajas xyxy por persona
        kp_data: [N, 9, 3] keypoints por persona
        iou_thresh: umbral IoU para suprimir duplicados

    Returns:
        kp_data filtrado [M, 9, 3] con M <= N
    """
    if boxes.shape[0] <= 1:
        return kp_data

    # Ordenar por confianza promedio de keypoints (mayor primero)
    conf_mean = kp_data[:, :, 2].mean(axis=1)
    order = np.argsort(-conf_mean)

    keep = []
    suppressed = set()

    for i_idx in range(len(order)):
        i = order[i_idx]
        if i in suppressed:
            continue
        keep.append(i)
        # Suprimir todos los que se superponen significativamente con este
        for j_idx in range(i_idx + 1, len(order)):
            j = order[j_idx]
            if j in suppressed:
                continue
            if _iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed.add(j)

    keep.sort()
    return kp_data[np.array(keep)]

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

        # Skip path: frame limpio sin overlay de FPS (FPS va al panel HTML)
        pass
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        # FPS tracking: include skip frames for accurate display rate
        state._fps_times.append(time.time())
        if len(state._fps_times) > 30:
            state._fps_times.pop(0)
        if len(state._fps_times) >= 2:
            elapsed = state._fps_times[-1] - state._fps_times[0]
            state._current_fps = (len(state._fps_times) - 1) / elapsed if elapsed > 0 else 0

        return out_rgb, _build_metrics_html(cpi_s, stat_s, bad_s, cpi_s, lumbar_s, curv_s, fps=state._current_fps), _build_status_html(stat_s, bad_s, alert_s)

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
        cv2.putText(out, "Sin deteccion — Colocate frente a la camara",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        state._last_overlay_bgr = out.copy()
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return out_rgb, _build_metrics_html(0, "NO DETECTADO", 0), _build_status_html("NO DETECTADO", 0, False)

    kp_data = preds[0].keypoints.data.cpu().numpy() # [N_personas, 9_kp, 3]

    # ── NMS: filtrar detecciones duplicadas de la misma persona ──────
    if kp_data.shape[0] > 1 and preds[0].boxes is not None and len(preds[0].boxes) > 0:
        boxes_xyxy = preds[0].boxes.xyxy.cpu().numpy()  # [N, 4]
        if boxes_xyxy.shape[0] == kp_data.shape[0]:
            kp_data = _nms_persons(boxes_xyxy, kp_data, iou_thresh=0.5)

    if kp_data.shape[0] == 0:
        posture = state.analyzer.analyze(
            keypoints=[], detected=False, timestamp=time.time(), frame_id=state.frame_count
        )
        state.frame_count += 1
        out = frame_bgr
        cv2.putText(out, "Sin deteccion — Colocate frente a la camara",
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

    # ── Session recording: append frame data ─────────────────────────
    if state.session_active and stat_val != "NO DETECTADO":
        state.session_frame_counter += 1
        avg_conf = float(np.mean([kp[2] for kp in keypoints if kp[2] > 0.1])) if keypoints else 0.0
        state.session_data.append({
            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "frame_id": state.session_frame_counter,
            "cpi": round(cpi, 2),
            "lumbar_angle_deg": round(lumbar, 1),
            "curvature_pct": round(curv, 2),
            "status": stat_val,
            "bad_posture_accumulated_s": round(bad, 1),
            "avg_confidence": round(avg_conf, 3),
        })
        # Advertencia si el buffer supera 10,000 filas (~1h a 3fps)
        if len(state.session_data) == 10000:
            print("[WARNING] Session buffer reached 10,000 rows (~1h). Consider exporting.")

    # ── Dibujar overlay para todas las personas ─────────────────────────
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for person_kps, is_primary in all_persons_kps:
        # Usar coordenadas EMA para la persona principal (evita overlap)
        draw_kps = keypoints if is_primary else person_kps
        # ── Esqueleto ──
        for conn in SKELETON_CONNECTIONS:
            i_a, i_b = conn
            if i_a >= len(draw_kps) or i_b >= len(draw_kps):
                continue
            kp_a = draw_kps[i_a]
            kp_b = draw_kps[i_b]
            if kp_a[2] > 0.1 and kp_b[2] > 0.1:
                pt_a = (int(kp_a[0]), int(kp_a[1]))
                pt_b = (int(kp_b[0]), int(kp_b[1]))
                cv2.line(out, pt_a, pt_b, COLOR_SKELETON, 2, cv2.LINE_AA)

        # ── Keypoints — solo ID sutil ──
        for i, kp in enumerate(draw_kps):
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

    # FPS va al panel HTML — frame limpio sin overlay de texto

    # ── Alerta sonora + popup visual (>30s mala postura) ──────────────
    if posture.needs_alert:
        now = time.time()
        if now - state.last_alert_beep > 5.0:
            try:
                winsound.Beep(1000, 300)  # 1000Hz, 300ms
            except Exception:
                pass
            state.last_alert_beep = now
        # (Popup visual ahora manejado por frontend vía data-alert)

    # (Popup de alerta movido al frontend — ya no se dibuja sobre el video)

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
    metrics_html = _build_metrics_html(cpi, stat_val, bad, cpi, lumbar, curv, fps=state._current_fps)
    status_html = _build_status_html(stat_val, bad, is_alert)

    return (
        out_rgb,
        metrics_html,
        status_html,
    )


# ── HTML builders ────────────────────────────────────────────────────────────
def _build_metrics_html(angle: float, status: str, bad_time: float,
                         cpi: float = 0, lumbar: float = 0, curv: float = 0,
                         fps: float = 0) -> str:
    """Construye HTML del panel de métricas — prev-value trick para CSS transitions."""
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
    new_offset = circumference - circumference * pct

    return f"""<div class="pm-card" style="text-align:center;"
      data-cpi="{cpi:.2f}" data-color="{color}" data-lumbar="{lumbar:.1f}"
      data-curv="{curv:.2f}" data-status="{status}" data-bad-time="{bad_time:.1f}"
      data-badge="{badge_cls}" data-fps="{fps:.1f}" data-offset="{new_offset:.2f}"
      data-circ="{circumference:.2f}">
    <div class="pm-gauge-wrap">
      <svg width="160" height="160" viewBox="0 0 160 160">
        <circle class="pm-gauge-track" cx="80" cy="80" r="52"/>
        <circle class="pm-gauge-fill" cx="80" cy="80" r="52" id="pm-gauge-arc"
          stroke="{color}"
          stroke-dasharray="{circumference:.2f}"
          stroke-dashoffset="{circumference:.2f}"
          style="color:{color}"/>
      </svg>
      <div class="pm-gauge-value" id="pm-gauge-num" style="color:{color}">{cpi:.1f}</div>
    </div>
    <div class="pm-metric-label">CPI — Combined Posture Index</div>
    <div class="pm-metric-sub" id="pm-metric-sub">
      <span class="pm-badge {badge_cls}" id="pm-badge">{status}</span>
      &nbsp;|&nbsp; Lumbar: <strong id="pm-lumbar">{lumbar:.0f}°</strong>
      &nbsp;|&nbsp; Curv: <strong id="pm-curv">{curv:.1f}%</strong>
      &nbsp;|&nbsp; Acum: <strong id="pm-bad-time">{bad_time:.0f}s</strong>
      &nbsp;|&nbsp; <span style="color:var(--accent-cyan);font-weight:700" id="pm-fps-val">{fps:.0f} fps</span>
    </div>
  </div>
  <script>
  (function(){{
    var el = document.currentScript.previousElementSibling;
    var cpi    = parseFloat(el.dataset.cpi);
    var color  = el.dataset.color;
    var lumbar = el.dataset.lumbar;
    var curv   = el.dataset.curv;
    var status = el.dataset.status;
    var badge  = el.dataset.badge;
    var badTime= el.dataset.badTime;
    var fps    = el.dataset.fps;
    var newOff = parseFloat(el.dataset.offset);
    var circ   = parseFloat(el.dataset.circ);

    // ── Gauge: prev-value trick → CSS transition dispara ──
    var arc = el.querySelector('#pm-gauge-arc');
    var num = el.querySelector('#pm-gauge-num');
    if (arc) {{
      // Arrancar desde el offset anterior (guardado en window global)
      var prevOff = (typeof window._pmPrevOffset !== 'undefined') ? window._pmPrevOffset : circ;
      arc.setAttribute('stroke-dashoffset', String(prevOff));
      arc.setAttribute('stroke', color);
      arc.style.color = color;
      // En el próximo frame de render, setear el nuevo valor → dispara transition
      requestAnimationFrame(function() {{
        arc.setAttribute('stroke-dashoffset', String(newOff));
        window._pmPrevOffset = newOff;
      }});
    }}
    if (num) {{ num.textContent = cpi.toFixed(1); num.style.color = color; }}

    // ── Textos ──
    var badgeEl   = el.querySelector('#pm-badge');
    var lumbarEl  = el.querySelector('#pm-lumbar');
    var curvEl    = el.querySelector('#pm-curv');
    var badTimeEl = el.querySelector('#pm-bad-time');
    var fpsEl     = el.querySelector('#pm-fps-val');
    if (badgeEl)   {{ badgeEl.textContent = status; badgeEl.className = 'pm-badge ' + badge; }}
    if (lumbarEl)  lumbarEl.textContent = lumbar + String.fromCharCode(176);
    if (curvEl)    curvEl.textContent = parseFloat(curv).toFixed(1) + '%';
    if (badTimeEl) badTimeEl.textContent = badTime + 's';
    if (fpsEl)     fpsEl.textContent = parseFloat(fps).toFixed(0) + ' fps';
  }})();
  </script>"""


def _build_status_html(status: str, bad_time: float, alert: bool) -> str:
    """Construye HTML del panel de estado — animation-class trick anti-flicker."""
    cls_map = {"CORRECTO":"ok","ALERTA LEVE":"warn","ALERTA CRÍTICA":"crit","NO DETECTADO":"nd","NO INICIADO":"nd"}
    cls = cls_map.get(status, "nd")
    return f"""<div class="pm-status pm-status-{cls}" data-status="{status}" data-bad-time="{bad_time:.1f}" data-alert="{str(alert)}" onmousemove="this.style.setProperty('--rx',event.offsetX+'px');this.style.setProperty('--ry',event.offsetY+'px')">
    <div style="font-size:15px;font-weight:700;position:relative;z-index:2" id="pm-status-icon">{status}</div>
    <div style="font-size:12px;margin-top:8px;opacity:.88;position:relative;z-index:2" id="pm-status-detail">...</div>
  </div>
  <script>
  (function(){{
    var el = document.currentScript.previousElementSibling;
    var status  = el.dataset.status;
    var badTime = el.dataset.badTime;
    var isAlert = el.dataset.alert === 'True';
    var clsMap  = {{'CORRECTO':'pm-status-ok','ALERTA LEVE':'pm-status-warn','ALERTA CRÍTICA':'pm-status-crit','NO DETECTADO':'pm-status-nd','NO INICIADO':'pm-status-nd'}};
    var newCls  = 'pm-status ' + (clsMap[status] || 'pm-status-nd');
    if (isAlert && status === 'ALERTA CRÍTICA') newCls += ' pulse';

    // ── Animation trick: solo cambiar clase si el status cambió ──
    // Esto evita que crit-breathe se reinicie cada 50ms
    var prevStatus = window._pmPrevStatus;
    if (prevStatus !== status || isAlert !== window._pmPrevAlert) {{
      el.className = newCls;
      window._pmPrevStatus = status;
      window._pmPrevAlert  = isAlert;
    }}
    // else: no-op — preservar className actual para que la animación continúe

    var iconEl   = el.querySelector('#pm-status-icon');
    var detailEl = el.querySelector('#pm-status-detail');
    if (iconEl && detailEl) {{
      if (isAlert) {{
        iconEl.innerHTML = '<span class="pm-live-dot" style="background:#ef4444;box-shadow:0 0 8px #ef4444"></span>ALERTA CRÍTICA';
        detailEl.textContent = 'Mala postura acumulada: ' + badTime + 's · Corrija la posicion de su cabeza';
      }} else if (status === 'ALERTA CRÍTICA') {{
        iconEl.textContent = 'ALERTA CRÍTICA';
        detailEl.textContent = 'Protrusión cefálica severa detectada · ' + badTime + 's acumulados';
      }} else if (status === 'ALERTA LEVE') {{
        iconEl.textContent = 'ALERTA LEVE';
        detailEl.textContent = 'Cabeza ligeramente adelantada · ' + badTime + 's acumulados';
      }} else if (status === 'NO DETECTADO' || status === 'NO INICIADO') {{
        iconEl.textContent = status;
        detailEl.textContent = 'Coloquese frente a la camara para iniciar el monitoreo';
      }} else {{
        iconEl.innerHTML = '<span class="pm-live-dot"></span>POSTURA CORRECTA';
        detailEl.textContent = 'Alineación cervical dentro de parámetros ergonómicos';
      }}
    }}

    // ── Frontend alert popup ──
    var popupId = 'pm-alert-popup';
    var popup   = document.getElementById(popupId);
    if (isAlert) {{
      if (!popup) {{
        popup = document.createElement('div');
        popup.id = popupId;
        document.body.appendChild(popup);
      }}
      popup.className = 'pm-alert-popup';
      popup.innerHTML = '<h3>⚠ ALERTA: Mala postura ' + badTime + 's</h3><p>Corrija su posicion</p>';
      popup.style.display = 'block';
      clearTimeout(window._pmAlertTimer);
      window._pmAlertTimer = setTimeout(function() {{
        popup.className = 'pm-alert-popup fade-out';
        setTimeout(function() {{ popup.style.display = 'none'; }}, 400);
      }}, 4000);
    }} else {{
      if (popup && popup.style.display !== 'none') {{
        clearTimeout(window._pmAlertTimer);
        popup.className = 'pm-alert-popup fade-out';
        setTimeout(function() {{ popup.style.display = 'none'; }}, 400);
      }}
    }}
  }})();
  </script>"""


def _compute_summary(session_data: list[dict]) -> Optional[dict]:
    """Calcula estadísticas agregadas de la sesión."""
    if not session_data:
        return None
    n = len(session_data)
    cpis = [r["cpi"] for r in session_data]
    statuses = [r["status"] for r in session_data]
    n_ok   = sum(1 for s in statuses if s == "CORRECTO")
    n_leve = sum(1 for s in statuses if s == "ALERTA LEVE")
    n_crit = sum(1 for s in statuses if s == "ALERTA CRÍTICA")
    duration = session_data[-1]["bad_posture_accumulated_s"] if session_data else 0
    return {
        "total_frames": n,
        "pct_correcto": round(n_ok / n * 100, 1),
        "pct_leve":     round(n_leve / n * 100, 1),
        "pct_critico":  round(n_crit / n * 100, 1),
        "avg_cpi":      round(sum(cpis) / n, 1),
        "max_cpi":      round(max(cpis), 1),
        "min_cpi":      round(min(cpis), 1),
        "total_bad_posture_s": round(duration, 1),
    }


def _export_csv_file() -> tuple[Optional[str], str]:
    """Genera archivo CSV de la sesión actual. Retorna (ruta_tmp, mensaje)."""
    if not state.session_data:
        return None, "⚠ No hay datos en la sesión actual. Inicia el monitoreo primero."
    if len(state.session_data) > 10000:
        return None, f"⚠ Buffer muy grande ({len(state.session_data)} filas). Exportando de todas formas..."
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"sesion_postural_{ts}.csv"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=fname + "_",
        delete=False, encoding="utf-8", newline=""
    )
    fieldnames = ["timestamp","frame_id","cpi","lumbar_angle_deg","curvature_pct",
                  "status","bad_posture_accumulated_s","avg_confidence"]
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(state.session_data)
    tmp.close()
    return tmp.name, f"✓ CSV exportado: {fname} ({len(state.session_data)} filas)"


def _build_summary_html(summary: Optional[dict]) -> str:
    """Construye HTML de la tarjeta de resumen de sesión."""
    if summary is None:
        return ""
    return f"""<div class="pm-card" style="margin-top:12px">
  <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--pm-text-3);margin-bottom:14px">
    Resumen de Sesión
  </div>
  <table class="pm-table">
    <tr><td>Frames analizados</td><td><strong>{summary['total_frames']}</strong></td></tr>
    <tr><td>Postura correcta</td><td><strong style="color:var(--pm-green)">{summary['pct_correcto']}%</strong></td></tr>
    <tr><td>Alerta leve</td><td><strong style="color:var(--pm-amber)">{summary['pct_leve']}%</strong></td></tr>
    <tr><td>Alerta crítica</td><td><strong style="color:var(--pm-red)">{summary['pct_critico']}%</strong></td></tr>
    <tr><td>CPI promedio</td><td><strong>{summary['avg_cpi']}</strong></td></tr>
    <tr><td>CPI máximo</td><td><strong>{summary['max_cpi']}</strong></td></tr>
    <tr><td>CPI mínimo</td><td><strong>{summary['min_cpi']}</strong></td></tr>
    <tr><td>Tiempo mala postura</td><td><strong>{summary['total_bad_posture_s']}s</strong></td></tr>
  </table>
</div>"""


# ── CSS y tema ──────────────────────────────────────────────────────────────
CSS = """
/* ═══════════════════════════════════════════════════════════════
   SISTEMA DE DISEÑO — Variables globales
   ═══════════════════════════════════════════════════════════════ */
:root {
    --pm-bg:          #0a0a0f;
    --pm-surface:     #12121a;
    --pm-surface-2:   #1a1a26;
    --pm-border:      rgba(255,255,255,0.07);
    --pm-border-2:    rgba(255,255,255,0.12);
    --pm-blue:        #3b82f6;
    --pm-green:       #22c55e;
    --pm-amber:       #f59e0b;
    --pm-red:         #ef4444;
    --pm-cyan:        #06b6d4;
    --pm-teal:        #14b8a6;
    --pm-text-1:      #f1f5f9;
    --pm-text-2:      #94a3b8;
    --pm-text-3:      #475569;
    --pm-radius:      14px;
    --pm-radius-sm:   8px;
    --pm-shadow:      0 4px 24px rgba(0,0,0,0.4);
    --pm-shadow-lg:   0 8px 40px rgba(0,0,0,0.5);
    --ok:             var(--pm-green);
    --warn:           var(--pm-amber);
    --critical:       var(--pm-red);
    --accent-cyan:    var(--pm-cyan);
    --accent-teal:    var(--pm-teal);
    --bg-deep:        var(--pm-bg);
    --bg-card:        rgba(18,18,26,0.8);
    --bg-elevated:    var(--pm-surface-2);
    --border:         var(--pm-border);
    --text-main:      var(--pm-text-1);
    --text-muted:     var(--pm-text-2);
    --radius:         var(--pm-radius);
    --glow-ok:        0 0 20px rgba(34,197,94,0.12);
    --glow-warn:      0 0 20px rgba(245,158,11,0.12);
    --glow-crit:      0 0 24px rgba(239,68,68,0.16);
}

/* ═══════════════════════════════════════════════════════════════
   BASE
   ═══════════════════════════════════════════════════════════════ */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    background: var(--pm-bg) !important;
    color: var(--pm-text-1) !important;
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}
footer, .gradio-footer { display: none !important; }

/* ═══════════════════════════════════════════════════════════════
   HEADER
   ═══════════════════════════════════════════════════════════════ */
.pm-header {
    position: relative;
    overflow: hidden;
    border-radius: var(--pm-radius);
    padding: 28px 32px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, rgba(18,18,26,0.95) 0%, rgba(26,26,38,0.9) 100%);
    border: 1px solid var(--pm-border-2);
    box-shadow: var(--pm-shadow-lg), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: fade-in-up 0.6s cubic-bezier(0.16,1,0.3,1) both;
}
.pm-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--pm-blue), var(--pm-cyan), var(--pm-teal));
    opacity: 0.9;
}
.pm-header::after {
    content: "";
    position: absolute;
    top: -40%; right: -5%; width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%);
    pointer-events: none;
    animation: aurora-float 10s ease-in-out infinite alternate;
}
@keyframes aurora-float {
    0%   { transform: translate(0,0) scale(1); }
    100% { transform: translate(15px,8px) scale(1.06); }
}
.pm-header h1 {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.4px;
    margin: 0 0 6px 0;
    color: var(--pm-text-1);
    position: relative; z-index: 2;
}
.pm-header p {
    margin: 0;
    font-size: 13px;
    color: var(--pm-text-2);
    line-height: 1.6;
    position: relative; z-index: 2;
}
.pm-header .brand-line {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 10px;
    padding: 3px 10px;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 20px;
    font-size: 10px;
    color: var(--pm-cyan);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    position: relative; z-index: 2;
}

/* ═══════════════════════════════════════════════════════════════
   CARDS
   ═══════════════════════════════════════════════════════════════ */
.pm-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--pm-border);
    border-radius: var(--pm-radius);
    padding: 24px;
    box-shadow: var(--pm-shadow);
    transition: transform 0.3s cubic-bezier(0.16,1,0.3,1),
                box-shadow 0.3s ease,
                border-color 0.3s ease;
    position: relative;
    overflow: hidden;
}
.pm-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--pm-shadow-lg);
    border-color: var(--pm-border-2);
}

/* ═══════════════════════════════════════════════════════════════
   RING GAUGE
   ═══════════════════════════════════════════════════════════════ */
.pm-gauge-wrap {
    position: relative;
    width: 160px; height: 160px;
    margin: 0 auto 12px;
}
.pm-gauge-wrap svg {
    transform: rotate(-90deg);
    overflow: visible;
}
.pm-gauge-track {
    fill: none;
    stroke: rgba(255,255,255,0.06);
    stroke-width: 7;
    stroke-linecap: round;
}
.pm-gauge-fill {
    fill: none;
    stroke-width: 7;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.55s cubic-bezier(0.16,1,0.3,1), stroke 0.35s ease;
    filter: drop-shadow(0 0 5px currentColor);
}
.pm-gauge-value {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1.5px;
    font-variant-numeric: tabular-nums;
}

/* ═══════════════════════════════════════════════════════════════
   MÉTRICAS
   ═══════════════════════════════════════════════════════════════ */
.pm-metric-label {
    font-size: 10px;
    color: var(--pm-text-3);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 8px;
    font-weight: 600;
}
.pm-metric-sub {
    font-size: 12px;
    color: var(--pm-text-2);
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--pm-border);
    line-height: 1.8;
}

/* ═══════════════════════════════════════════════════════════════
   STATUS PANEL
   ═══════════════════════════════════════════════════════════════ */
.pm-status {
    border-radius: var(--pm-radius);
    padding: 18px 20px;
    text-align: center;
    font-weight: 600;
    border: 1px solid transparent;
    transition: background 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease;
    position: relative;
    overflow: hidden;
}
.pm-status-ok   { background: rgba(34,197,94,0.05);  color: var(--pm-green); border-color: rgba(34,197,94,0.18);  box-shadow: var(--glow-ok); }
.pm-status-warn { background: rgba(245,158,11,0.05); color: var(--pm-amber); border-color: rgba(245,158,11,0.20); box-shadow: var(--glow-warn); }
.pm-status-crit { background: rgba(239,68,68,0.05);  color: var(--pm-red);   border-color: rgba(239,68,68,0.24);  box-shadow: var(--glow-crit); }
.pm-status-nd   { background: rgba(18,18,26,0.7);    color: var(--pm-text-2); border-color: var(--pm-border); }

.pm-status::after {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(circle at var(--rx,50%) var(--ry,50%), rgba(255,255,255,0.05) 0%, transparent 55%);
    opacity: 0; transition: opacity 0.3s ease; pointer-events: none;
}
.pm-status:hover::after { opacity: 1; }

@keyframes crit-breathe {
    0%, 100% { box-shadow: 0 0 16px rgba(239,68,68,0.10), 0 0 36px rgba(239,68,68,0.04); }
    50%      { box-shadow: 0 0 26px rgba(239,68,68,0.20), 0 0 52px rgba(239,68,68,0.08); }
}
.pm-status-crit.pulse { animation: crit-breathe 2.2s ease-in-out infinite; }

.pm-status-crit.pulse::before {
    content: "";
    position: absolute;
    top: 0; left: -100%; width: 55%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.03), transparent);
    animation: scan-slide 2.8s ease-in-out infinite;
    pointer-events: none;
}
@keyframes scan-slide {
    0%   { left: -55%; }
    100% { left: 145%; }
}

/* ═══════════════════════════════════════════════════════════════
   LIVE DOT
   ═══════════════════════════════════════════════════════════════ */
.pm-live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--pm-green);
    box-shadow: 0 0 7px var(--pm-green);
    margin-right: 6px;
    animation: live-pulse 1.8s ease-in-out infinite;
    vertical-align: middle;
}
@keyframes live-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.45; transform: scale(0.65); }
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════ */
.pm-sidebar-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: var(--pm-text-3);
    margin: 22px 0 10px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--pm-border);
}
.pm-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 11.5px;
}
.pm-table th {
    text-align: left;
    color: var(--pm-text-3);
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 6px 8px;
    border-bottom: 1px solid var(--pm-border);
}
.pm-table td {
    padding: 8px 8px;
    color: var(--pm-text-1);
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 11.5px;
}
.pm-table tr:nth-child(even) td { background: rgba(255,255,255,0.015); }
.pm-table tr:hover td { background: rgba(255,255,255,0.04); }
.pm-table tr:last-child td { border-bottom: none; }

.pm-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: var(--pm-radius-sm);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.badge-ok   { background: rgba(34,197,94,0.12);  color: var(--pm-green); }
.badge-warn { background: rgba(245,158,11,0.12); color: var(--pm-amber); }
.badge-crit { background: rgba(239,68,68,0.12);  color: var(--pm-red); }
.badge-nd   { background: rgba(148,163,184,0.10); color: var(--pm-text-2); }

.pm-kp-list {
    list-style: none;
    padding: 0; margin: 0;
    font-size: 11.5px;
    line-height: 2;
    color: var(--pm-text-2);
}
.pm-kp-list li strong { color: var(--pm-text-1); }

.pm-note {
    font-size: 11px;
    color: var(--pm-text-2);
    padding: 10px 14px;
    background: rgba(18,18,26,0.7);
    border-radius: var(--pm-radius-sm);
    border-left: 2px solid var(--pm-cyan);
    margin-top: 10px;
    line-height: 1.55;
}

/* ═══════════════════════════════════════════════════════════════
   ALERT POPUP
   ═══════════════════════════════════════════════════════════════ */
.pm-alert-popup {
    position: fixed;
    top: 20px; right: 20px;
    z-index: 9999;
    background: rgba(239,68,68,0.94);
    border: 1px solid rgba(239,68,68,0.6);
    border-radius: var(--pm-radius);
    padding: 14px 20px;
    color: #fff;
    font-family: "Inter", system-ui, sans-serif;
    box-shadow: 0 8px 32px rgba(239,68,68,0.35);
    animation: pm-alert-in 0.25s cubic-bezier(0.16,1,0.3,1);
    pointer-events: none;
    max-width: 300px;
    backdrop-filter: blur(8px);
}
.pm-alert-popup h3 { margin: 0 0 4px 0; font-size: 14px; font-weight: 700; }
.pm-alert-popup p  { margin: 0; font-size: 12px; opacity: 0.88; }
@keyframes pm-alert-in {
    from { opacity: 0; transform: translateY(-10px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.pm-alert-popup.fade-out { animation: pm-alert-out 0.35s ease-in forwards; }
@keyframes pm-alert-out {
    from { opacity: 1; transform: translateY(0); }
    to   { opacity: 0; transform: translateY(-10px); }
}

/* ═══════════════════════════════════════════════════════════════
   ANIMACIONES
   ═══════════════════════════════════════════════════════════════ */
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════════════════════ */
* { scrollbar-width: thin; scrollbar-color: rgba(148,163,184,0.12) transparent; }
*::-webkit-scrollbar { width: 4px; }
*::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.12); border-radius: 10px; }
*::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.22); }
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


# ── Callbacks de sesión ──────────────────────────────────────────────────────
def _start_session(is_active: bool) -> tuple[bool, str, str, object, object]:
    """Inicia grabación de sesión. Retorna nuevo estado."""
    if is_active:
        # Ya activa — no hacer nada
        return True, "⏹ Detener sesión", "🔴 Sesión activa — grabando datos...", gr.update(visible=False, value=""), gr.update(visible=False)
    state.session_data = []
    state.session_frame_counter = 0
    state.session_active = True
    state.session_start_time = time.time()
    return True, "⏹ Detener sesión", "🔴 Sesión activa — grabando datos...", gr.update(visible=False, value=""), gr.update(visible=False)


def _stop_session(is_active: bool) -> tuple[bool, str, str, object, object]:
    """Detiene grabación y muestra resumen."""
    if not is_active:
        # Ya detenida — no hacer nada
        return False, "▶ Iniciar sesión", "_Sin sesión activa_", gr.update(visible=False), gr.update(visible=False)
    state.session_active = False
    summary = _compute_summary(state.session_data)
    summary_html = _build_summary_html(summary)
    n = len(state.session_data)
    msg = f"✓ Sesión detenida — {n} frames grabados. Exporta el CSV para analizar los datos."
    return False, "▶ Iniciar sesión", msg, gr.update(visible=bool(summary_html), value=summary_html if summary_html else ""), gr.update(visible=n > 0)


def _toggle_session(is_active: bool) -> tuple[bool, str, str, object, object]:
    """Toggle start/stop sesión usando gr.State (no el label del botón)."""
    if is_active:
        return _stop_session(is_active)
    else:
        return _start_session(is_active)


def _do_export() -> tuple[object, str]:
    """Exporta CSV y retorna el archivo."""
    path, msg = _export_csv_file()
    if path:
        return gr.update(value=path, visible=True), msg
    return gr.update(visible=False), msg


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

        session_state = gr.State(False)  # False = no activa, True = activa

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
                gr.HTML('<div class="pm-note">Se emite un beep cada 5 s cuando la mala postura supera 30 s de acumulación continua.</div>')

                gr.HTML('<div class="pm-sidebar-title">Grabación de Sesión</div>')
                session_btn = gr.Button("▶ Iniciar sesión", variant="primary", size="sm")
                session_status = gr.Markdown("_Sin sesión activa_")
                export_btn = gr.Button("⬇ Exportar CSV", variant="secondary", size="sm", visible=False)
                export_file = gr.File(label="Archivo CSV", visible=False, interactive=False)
                export_msg = gr.Markdown("")
                summary_display = gr.HTML("", visible=False)

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

        session_btn.click(
            fn=_toggle_session,
            inputs=[session_state],
            outputs=[session_state, session_btn, session_status, summary_display, export_btn],
        )

        export_btn.click(
            fn=_do_export,
            inputs=[],
            outputs=[export_file, export_msg],
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
