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

        # ── CPI history para sparkline ────────────────────────────────
        self._cpi_history: list[tuple[float, float]] = []  # (timestamp, cpi)

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


def _nms_persons(boxes: np.ndarray, kp_data: np.ndarray, iou_thresh: float = 0.3) -> np.ndarray:
    """NMS + distance filter para eliminar detecciones duplicadas de la misma persona."""
    if boxes.shape[0] <= 1:
        return kp_data

    conf_mean = kp_data[:, :, 2].mean(axis=1)
    order = np.argsort(-conf_mean)

    keep = []
    suppressed = set()

    for i_idx in range(len(order)):
        i = order[i_idx]
        if i in suppressed:
            continue
        keep.append(i)
        cx_i = (boxes[i][0] + boxes[i][2]) / 2
        cy_i = (boxes[i][1] + boxes[i][3]) / 2
        for j_idx in range(i_idx + 1, len(order)):
            j = order[j_idx]
            if j in suppressed:
                continue
            # Suprimir por IoU
            if _iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed.add(j)
                continue
            # Suprimir por distancia entre centros (< 80px = misma persona)
            cx_j = (boxes[j][0] + boxes[j][2]) / 2
            cy_j = (boxes[j][1] + boxes[j][3]) / 2
            dist = ((cx_i - cx_j)**2 + (cy_i - cy_j)**2) ** 0.5
            if dist < 80:
                suppressed.add(j)

    keep.sort()
    return kp_data[np.array(keep)]

# ── Función principal: procesar un frame de webcam ───────────────────────────
def process_frame(frame: np.ndarray, model_choice: str) -> tuple[np.ndarray, str]:
    """
    Procesa un frame de la webcam: YOLO inference + overlay + análisis postural.

    Args:
        frame: Imagen RGB desde la webcam (numpy array H×W×3).
        model_choice: Nombre del modelo seleccionado.

    Returns:
        (frame_con_overlay, metrics_json)
    """
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        return blank, _build_metrics_json(history=[])

    # Buscar modelo seleccionado (O(1) con dict precomputado)
    cfg = MODEL_LOOKUP.get(model_choice)
    if cfg is None:
        return frame, _build_metrics_json(0, "NO DETECTADO", 0, 0, 0, 0, 0.0, False, history=[])
    model_path = cfg["path"]
    state.model_key = cfg["key"]

    # Cargar modelo si es necesario
    try:
        state.load_model(model_path)
    except Exception as e:
        return frame, _build_metrics_json(0, "NO DETECTADO", 0, 0, 0, 0, 0.0, False, history=[])

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

        return out_rgb, _build_metrics_json(cpi_s, stat_s, bad_s, lumbar_s, curv_s, state._current_fps, 0.0, alert_s, history=state._cpi_history)

    # Convertir RGB → BGR para YOLO/OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── YOLO inference ──────────────────────────────────────────────────
    try:
        t_inf = time.time()
        preds = state.model(frame_bgr, verbose=False, conf=0.25, imgsz=320, max_det=MAX_PERSONS)
        inference_ms = (time.time() - t_inf) * 1000
    except Exception as e:
        return frame, _build_metrics_json(0, "NO DETECTADO", 0, 0, 0, 0, 0.0, False), _build_sparkline_html([])

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
        return out_rgb, _build_metrics_json(0, "NO DETECTADO", 0, 0, 0, 0, 0.0, False, history=[])

    kp_data = preds[0].keypoints.data.cpu().numpy() # [N_personas, 9_kp, 3]

    # ── NMS: filtrar detecciones duplicadas de la misma persona ──────
    if kp_data.shape[0] > 1 and preds[0].boxes is not None and len(preds[0].boxes) > 0:
        boxes_xyxy = preds[0].boxes.xyxy.cpu().numpy()  # [N, 4]
        if boxes_xyxy.shape[0] == kp_data.shape[0]:
            kp_data = _nms_persons(boxes_xyxy, kp_data, iou_thresh=0.3)

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
        return out_rgb, _build_metrics_json(0, "NO DETECTADO", 0, 0, 0, 0, 0.0, False, history=[])

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

    # ── Confianza de detección (5 keypoints críticos del CPI) ─────────
    CRITICAL_KP_IDX = [0, 1, 3, 4, 8]
    conf_vals = [keypoints[i][2] for i in CRITICAL_KP_IDX if i < len(keypoints) and keypoints[i][2] > 0.1]
    avg_confidence = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0

    # ── CPI history: append + prune (sparkline) ───────────────────────
    state._cpi_history.append((timestamp, cpi))
    cutoff = timestamp - 60.0
    state._cpi_history = [(t, v) for t, v in state._cpi_history if t >= cutoff]
    if len(state._cpi_history) > 180:
        state._cpi_history = state._cpi_history[-180:]

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

    # Build JSON — always fresh; JS polling loop handles DOM updates
    metrics_json = _build_metrics_json(cpi, stat_val, bad, lumbar, curv, state._current_fps, avg_confidence, is_alert, history=state._cpi_history)

    return (
        out_rgb,
        metrics_json,
    )


# ── JSON builder (Static HTML + Hidden Textbox pattern) ──────────────────────
def _build_metrics_json(cpi: float = 0, status: str = "NO DETECTADO",
                         bad_time: float = 0, lumbar: float = 0,
                         curv: float = 0, fps: float = 0,
                         conf: float = 0.0, alert: bool = False,
                         history: list = None) -> str:
    """Serializa métricas a JSON para el panel estático."""
    import json as _json
    palette = {
        "CORRECTO":       "#22c55e",
        "ALERTA LEVE":    "#f59e0b",
        "ALERTA CRÍTICA": "#ef4444",
        "NO DETECTADO":   "#94a3b8",
        "NO INICIADO":    "#94a3b8",
    }
    # history is list of (timestamp, cpi) tuples — send only cpi values to JS
    history_values = [round(v, 1) for _, v in history] if history else []
    payload = _json.dumps({
        "cpi": round(cpi, 1),
        "status": status,
        "bad_time": round(bad_time, 1),
        "lumbar": round(lumbar, 1),
        "curv": round(curv, 2),
        "fps": round(fps, 1),
        "conf": round(conf, 3),
        "alert": alert,
        "color": palette.get(status, "#94a3b8"),
        "history": history_values,
    })
    return f'<div id="pm-metrics-data-inner" style="display:none">{payload}</div>'


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

/* ── Confidence bar ── */
.pm-conf-bar-wrap { height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
.pm-conf-bar { height: 100%; border-radius: 2px; transition: width 0.4s ease, background 0.4s ease; min-width: 2px; }

/* ── CPI Sparkline ── */
.pm-sparkline-wrap { margin-top: 4px; }
.pm-sparkline-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--pm-text-3);
    margin-top: 4px;
    padding: 0 2px;
}
.pm-sparkline-placeholder {
    font-size: 11px;
    color: var(--pm-text-3);
    text-align: center;
    padding: 18px 0;
    font-style: italic;
}
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


# ── Threshold helpers ────────────────────────────────────────────────────────
def _build_threshold_table(leve: float = 35, critico: float = 50) -> str:
    """Genera HTML de la tabla de umbrales CPI con valores actuales."""
    return f"""<table class="pm-table">
    <tr><th>CPI</th><th>Estado</th><th>Significado</th></tr>
    <tr><td>CPI ≤ {leve:.0f}</td><td><span class="pm-badge badge-ok">Correcto</span></td><td>Columna alineada, postura recta</td></tr>
    <tr><td>{leve:.0f} &lt; CPI ≤ {critico:.0f}</td><td><span class="pm-badge badge-warn">Alerta leve</span></td><td>Curvatura dorsal leve</td></tr>
    <tr><td>CPI &gt; {critico:.0f}</td><td><span class="pm-badge badge-crit">Alerta critica</span></td><td>Cifosis / hombros caidos</td></tr>
</table>"""


def _update_thresholds(leve: float, critico: float) -> tuple[str, str]:
    """Actualiza umbrales CPI en el analizador. Retorna (tabla_html, mensaje)."""
    if leve >= critico:
        return (
            _build_threshold_table(state.analyzer.CPI_LEVE, state.analyzer.CPI_CRITICO),
            f"⚠ Umbral leve ({leve:.0f}) debe ser menor que crítico ({critico:.0f})"
        )
    state.analyzer.CPI_LEVE = float(leve)
    state.analyzer.CPI_CRITICO = float(critico)
    return (
        _build_threshold_table(leve, critico),
        f"✓ Umbrales actualizados — Leve: {leve:.0f} | Crítico: {critico:.0f}"
    )


METRICS_JS = """
() => {
  var CIRC = 326.73;
  var prevStatus = '';
  var alertTimer = null;

  function animateValue(el, newText) {
    if (!el || el.textContent === newText) return;
    el.style.transition = 'opacity 0.15s ease';
    el.style.opacity = '0.2';
    setTimeout(function() {
      el.textContent = newText;
      el.style.opacity = '1';
    }, 150);
  }

  function drawSparkline(history) {
    if (!history || history.length < 2) return;
    var W = 280, H = 64, MAX = 100;
    var n = history.length;
    var pts = history.map(function(v, i) {
      return [i / (n - 1) * W, H - (Math.min(Math.max(v, 0), MAX) / MAX) * (H - 4) - 2];
    });
    var d = pts.map(function(p, i) {
      return (i === 0 ? 'M' : 'L') + p[0].toFixed(1) + ',' + p[1].toFixed(1);
    }).join(' ');
    var lineEl = document.getElementById('spark-line');
    if (lineEl) lineEl.setAttribute('d', d);
    var area = d + ' L' + pts[pts.length-1][0].toFixed(1) + ',' + H + ' L0,' + H + ' Z';
    var areaEl = document.getElementById('spark-area');
    if (areaEl) areaEl.setAttribute('d', area);
    var last = pts[pts.length - 1];
    var dotEl = document.getElementById('spark-dot');
    if (dotEl) { dotEl.setAttribute('cx', last[0].toFixed(1)); dotEl.setAttribute('cy', last[1].toFixed(1)); }
  }

  function updateMetrics(data) {
    var cpi     = data.cpi     !== undefined ? data.cpi     : 0;
    var status  = data.status  || 'NO DETECTADO';
    var badTime = data.bad_time !== undefined ? data.bad_time : 0;
    var lumbar  = data.lumbar  !== undefined ? data.lumbar  : 0;
    var curv    = data.curv    !== undefined ? data.curv    : 0;
    var fps     = data.fps     !== undefined ? data.fps     : 0;
    var conf    = data.conf    !== undefined ? data.conf    : 0;
    var alert   = data.alert   || false;
    var color   = data.color   || '#94a3b8';
    var history = data.history || [];

    // Gauge arc
    var pct = Math.min(Math.max(cpi, 0), 100) / 100;
    var offset = CIRC - CIRC * pct;
    var arc = document.getElementById('pm-gauge-arc');
    if (arc) { arc.style.strokeDashoffset = offset; arc.style.stroke = color; }

    // Gauge number
    var num = document.getElementById('pm-gauge-num');
    if (num) { animateValue(num, cpi.toFixed(1)); num.style.color = color; }

    // Badge
    var badgeEl = document.getElementById('pm-badge');
    var badgeMap = {
      'CORRECTO':       ['badge-ok',   'CORRECTO'],
      'ALERTA LEVE':    ['badge-warn', 'ALERTA LEVE'],
      'ALERTA CRÍTICA': ['badge-crit', 'ALERTA CRÍTICA'],
      'NO DETECTADO':   ['badge-nd',   'NO DETECTADO'],
      'NO INICIADO':    ['badge-nd',   'NO INICIADO'],
    };
    if (badgeEl) {
      var b = badgeMap[status] || ['badge-nd', status];
      if (badgeEl.textContent !== b[1]) { badgeEl.className = 'pm-badge ' + b[0]; badgeEl.textContent = b[1]; }
    }

    // Metrics text
    animateValue(document.getElementById('pm-lumbar'),   lumbar.toFixed(0) + '°');
    animateValue(document.getElementById('pm-curv'),     curv.toFixed(1) + '%');
    animateValue(document.getElementById('pm-bad-time'), badTime.toFixed(0) + 's');
    animateValue(document.getElementById('pm-fps-val'),  fps.toFixed(0) + ' fps');

    // Confidence
    var confPct = Math.round(conf * 100);
    var confColor = conf >= 0.7 ? '#22c55e' : conf >= 0.4 ? '#f59e0b' : '#ef4444';
    var confBar = document.getElementById('pm-conf-bar');
    var confVal = document.getElementById('pm-conf-val');
    var confBadge = document.getElementById('pm-conf-badge');
    if (confBar) { confBar.style.width = confPct + '%'; confBar.style.background = confColor; confBar.style.transition = 'width 0.3s ease, background 0.3s ease'; }
    if (confVal) { animateValue(confVal, confPct + '%'); confVal.style.color = confColor; }
    if (confBadge) confBadge.style.display = conf < 0.4 ? 'inline-block' : 'none';

    // Status card — solo cambiar si el status cambió
    var card = document.getElementById('pm-status-card');
    var iconEl = document.getElementById('pm-status-icon');
    var detailEl = document.getElementById('pm-status-detail');
    if (card && prevStatus !== status) {
      var clsMap = {
        'CORRECTO':       'pm-status pm-status-ok',
        'ALERTA LEVE':    'pm-status pm-status-warn',
        'ALERTA CRÍTICA': 'pm-status pm-status-crit',
        'NO DETECTADO':   'pm-status pm-status-nd',
        'NO INICIADO':    'pm-status pm-status-nd',
      };
      card.className = (clsMap[status] || 'pm-status pm-status-nd') + (alert ? ' pulse' : '');
      prevStatus = status;
    }
    if (iconEl) {
      var icons = { 'CORRECTO': '✓ POSTURA CORRECTA', 'ALERTA LEVE': '⚠ ALERTA LEVE', 'ALERTA CRÍTICA': '✕ ALERTA CRÍTICA', 'NO DETECTADO': '— NO DETECTADO', 'NO INICIADO': '— NO INICIADO' };
      iconEl.textContent = icons[status] || status;
    }
    if (detailEl) {
      if (status === 'CORRECTO') detailEl.textContent = 'Alineación cervical dentro de parámetros ergonómicos';
      else if (status === 'ALERTA LEVE') detailEl.textContent = 'Cabeza ligeramente adelantada · ' + badTime.toFixed(0) + 's acumulados';
      else if (status === 'ALERTA CRÍTICA') detailEl.textContent = 'Protrusión cefálica severa · ' + badTime.toFixed(0) + 's acumulados';
      else detailEl.textContent = 'Colóquese frente a la cámara para iniciar';
    }

    // Alert popup
    var popup = document.getElementById('pm-alert-popup');
    if (popup && alert) {
      var t = document.getElementById('pm-alert-title');
      if (t) t.textContent = '⚠ Mala postura: ' + badTime.toFixed(0) + 's';
      popup.style.display = 'block'; popup.style.opacity = '1';
      clearTimeout(alertTimer);
      alertTimer = setTimeout(function() {
        popup.style.opacity = '0';
        setTimeout(function() { popup.style.display = 'none'; }, 400);
      }, 4000);
    }

    // Sparkline
    drawSparkline(history);
  }

  // Polling loop — lee el div carrier cada 100ms
  setInterval(function() {
    var el = document.getElementById('pm-metrics-data-inner');
    if (!el) return;
    var raw = (el.textContent || el.innerText || '').trim();
    if (!raw || raw === '{}') return;
    try { updateMetrics(JSON.parse(raw)); } catch(e) {}
  }, 100);
}
"""


# ── Construir UI ─────────────────────────────────────────────────────────────
def _build_sparkline_html(history: list[tuple[float, float]]) -> str:
    """Genera SVG sparkline de CPI (últimos 60s)."""
    W, H, PAD = 280, 64, 4
    if len(history) < 2:
        return '<div class="pm-sparkline-placeholder">Recopilando datos...</div>'

    times  = [t for t, _ in history]
    values = [v for _, v in history]
    t_min, t_max = times[0], times[-1]
    t_range = max(t_max - t_min, 1.0)

    def _x(t: float) -> float:
        return PAD + (t - t_min) / t_range * (W - 2 * PAD)

    def _y(v: float) -> float:
        return H - PAD - min(max(v, 0), 100) / 100.0 * (H - 2 * PAD)

    points = " ".join(f"{_x(t):.1f},{_y(v):.1f}" for t, v in history)
    last_cpi = values[-1]
    line_color = "#22c55e" if last_cpi <= 35 else "#f59e0b" if last_cpi <= 50 else "#ef4444"

    # Zone band Y coords
    y_0   = _y(0)
    y_35  = _y(35)
    y_50  = _y(50)
    y_100 = _y(100)
    x0, xw = PAD, W - 2 * PAD

    return f"""<div class="pm-sparkline-wrap">
  <svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="display:block;width:100%">
    <rect x="{x0}" y="{y_100:.1f}" width="{xw}" height="{y_35 - y_100:.1f}" fill="rgba(34,197,94,0.07)" rx="2"/>
    <rect x="{x0}" y="{y_35:.1f}"  width="{xw}" height="{y_50 - y_35:.1f}"  fill="rgba(245,158,11,0.07)" rx="2"/>
    <rect x="{x0}" y="{y_50:.1f}"  width="{xw}" height="{y_0 - y_50:.1f}"   fill="rgba(239,68,68,0.07)"  rx="2"/>
    <polyline points="{points}" fill="none" stroke="{line_color}" stroke-width="1.8"
      stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>
    <circle cx="{_x(times[-1]):.1f}" cy="{_y(last_cpi):.1f}" r="3.5" fill="{line_color}"/>
  </svg>
  <div class="pm-sparkline-labels">
    <span>60s</span>
    <span style="color:{line_color};font-weight:700">CPI {last_cpi:.0f}</span>
    <span>ahora</span>
  </div>
</div>"""


def _build_static_metrics_panel() -> str:
    """Panel de métricas estático — se renderiza UNA VEZ. Sin scripts (Gradio 6 los elimina).
    El JS se inyecta via app.load(js=METRICS_JS)."""
    return """
<style>
  #pm-metrics-root { font-family: 'Inter', sans-serif; color: #e2e8f0; }

  .pm-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
  }

  .pm-section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 10px;
  }

  /* Gauge */
  .pm-gauge-wrap { position: relative; width: 140px; height: 140px; margin: 0 auto 8px; }
  .pm-gauge-track { fill: none; stroke: rgba(99,102,241,0.15); stroke-width: 10; }
  .pm-gauge-fill  { fill: none; stroke-width: 10; stroke-linecap: round;
    transform: rotate(-90deg); transform-origin: 50% 50%;
    stroke-dasharray: 326.73; stroke-dashoffset: 326.73;
    transition: stroke-dashoffset 0.5s ease, stroke 0.4s ease; }
  .pm-gauge-value {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
    font-size: 28px; font-weight: 800; transition: color 0.4s ease;
  }
  .pm-gauge-label { text-align: center; font-size: 10px; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; }

  /* Badges */
  .pm-badge { display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }
  .badge-ok   { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
  .badge-warn { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }
  .badge-crit { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
  .badge-nd   { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }

  /* Metrics grid */
  .pm-metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
  .pm-metric-item { background: rgba(99,102,241,0.06); border-radius: 8px; padding: 8px 10px; }
  .pm-metric-item .label { font-size: 9px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
  .pm-metric-item .value { font-size: 18px; font-weight: 700; color: #e2e8f0; }

  /* Confidence bar */
  .pm-conf-track { background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; overflow: hidden; margin: 6px 0 4px; }
  .pm-conf-fill  { height: 100%; border-radius: 4px; width: 0%; transition: width 0.3s ease, background 0.3s ease; }

  /* Status card */
  .pm-status {
    border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;
    border: 1px solid transparent; transition: background 0.4s ease, border-color 0.4s ease;
  }
  .pm-status-nd   { background: rgba(148,163,184,0.08); border-color: rgba(148,163,184,0.2); }
  .pm-status-ok   { background: rgba(34,197,94,0.08);   border-color: rgba(34,197,94,0.3); }
  .pm-status-warn { background: rgba(245,158,11,0.08);  border-color: rgba(245,158,11,0.3); }
  .pm-status-crit { background: rgba(239,68,68,0.08);   border-color: rgba(239,68,68,0.3); }
  .pm-status.pulse { animation: pm-pulse 1.8s ease-in-out infinite; }
  @keyframes pm-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); } 50% { box-shadow: 0 0 0 8px rgba(239,68,68,0.2); } }
  .pm-status-icon   { font-size: 13px; font-weight: 700; }
  .pm-status-detail { font-size: 11px; color: #94a3b8; margin-top: 4px; }

  /* Sparkline */
  #pm-sparkline-svg { display: block; width: 100%; }

  /* Alert popup */
  #pm-alert-popup {
    display: none; opacity: 0;
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: rgba(239,68,68,0.95); color: #fff;
    border-radius: 10px; padding: 12px 18px;
    font-size: 13px; font-weight: 700;
    box-shadow: 0 8px 32px rgba(239,68,68,0.4);
    transition: opacity 0.3s ease;
  }
</style>

<div id="pm-metrics-root">

  <!-- ── GAUGE + ESTADO ── -->
  <div class="pm-card">
    <div class="pm-section-title">CPI — Combined Posture Index</div>
    <div class="pm-gauge-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle class="pm-gauge-track" cx="70" cy="70" r="52"/>
        <circle class="pm-gauge-fill" id="pm-gauge-arc" cx="70" cy="70" r="52" stroke="#94a3b8"/>
      </svg>
      <div class="pm-gauge-value" id="pm-gauge-num" style="color:#94a3b8">0.0</div>
    </div>
    <div class="pm-gauge-label">
      <span class="pm-badge badge-nd" id="pm-badge">NO INICIADO</span>
    </div>

    <div class="pm-metrics-grid" style="margin-top:12px">
      <div class="pm-metric-item">
        <div class="label">Lumbar</div>
        <div class="value" id="pm-lumbar">0°</div>
      </div>
      <div class="pm-metric-item">
        <div class="label">Curvatura</div>
        <div class="value" id="pm-curv">0.0%</div>
      </div>
      <div class="pm-metric-item">
        <div class="label">Mala postura</div>
        <div class="value" id="pm-bad-time">0s</div>
      </div>
      <div class="pm-metric-item">
        <div class="label">FPS</div>
        <div class="value" id="pm-fps-val" style="color:#6366f1">0</div>
      </div>
    </div>
  </div>

  <!-- ── ESTADO POSTURAL ── -->
  <div class="pm-status pm-status-nd" id="pm-status-card">
    <div class="pm-status-icon" id="pm-status-icon">— NO INICIADO</div>
    <div class="pm-status-detail" id="pm-status-detail">Colóquese frente a la cámara para iniciar</div>
  </div>

  <!-- ── CONFIANZA ── -->
  <div class="pm-card">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span class="pm-section-title" style="margin-bottom:0">Confianza detección</span>
      <span id="pm-conf-val" style="font-size:12px;font-weight:700;color:#94a3b8">0%</span>
    </div>
    <div class="pm-conf-track">
      <div class="pm-conf-fill" id="pm-conf-bar"></div>
    </div>
    <span id="pm-conf-badge" style="display:none;font-size:10px;font-weight:700;color:#ef4444">⚠ Detección débil — datos no confiables</span>
  </div>

  <!-- ── SPARKLINE ── -->
  <div class="pm-card">
    <div class="pm-section-title">Historial CPI — últimos 60s</div>
    <svg id="pm-sparkline-svg" height="56" viewBox="0 0 280 56" preserveAspectRatio="none">
      <rect x="0" y="0" width="280" height="56" fill="rgba(99,102,241,0.03)" rx="4"/>
      <path id="spark-area" fill="rgba(99,102,241,0.12)" d=""/>
      <path id="spark-line" fill="none" stroke="#6366f1" stroke-width="1.5" stroke-linejoin="round" d=""/>
      <circle id="spark-dot" r="3.5" fill="#6366f1" cx="280" cy="28"/>
    </svg>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:#475569;margin-top:3px">
      <span>60s atrás</span><span>ahora</span>
    </div>
  </div>

</div>

<div id="pm-alert-popup">
  <div id="pm-alert-title">⚠ Alerta postural</div>
</div>
"""


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

            # ── Columna derecha: Métricas + controles ─────────────────
            with gr.Column(scale=1):
                metrics_panel = gr.HTML(_build_static_metrics_panel())
                metrics_data = gr.HTML(
                    value='<div id="pm-metrics-data-inner" style="display:none">{}</div>',
                    elem_id="pm-metrics-data",
                )

                with gr.Accordion("Calibrar umbrales CPI", open=False):
                    threshold_table = gr.HTML(_build_threshold_table())
                    leve_slider = gr.Slider(
                        minimum=10, maximum=80, value=35, step=1,
                        label="Umbral Leve", interactive=True
                    )
                    critico_slider = gr.Slider(
                        minimum=20, maximum=100, value=50, step=1,
                        label="Umbral Crítico", interactive=True
                    )
                    threshold_msg = gr.Markdown("_Ajusta los sliders para calibrar_")

                with gr.Accordion("Referencia de keypoints", open=False):
                    gr.HTML("""
                    <table style="width:100%;font-size:11px;border-collapse:collapse">
                      <tr style="color:#6366f1"><th style="padding:4px 6px;text-align:left">ID</th><th style="padding:4px 6px;text-align:left">Nombre</th><th style="padding:4px 6px;text-align:left">Ubicación</th></tr>
                      <tr><td style="padding:3px 6px"><b>K0</b></td><td>Head-back</td><td>Occipital</td></tr>
                      <tr><td style="padding:3px 6px"><b>K1</b></td><td>Neck-back</td><td>C7 cervical</td></tr>
                      <tr><td style="padding:3px 6px"><b>K2</b></td><td>Shoulder-top</td><td>Acromion</td></tr>
                      <tr><td style="padding:3px 6px"><b>K3</b></td><td>Back-borde</td><td>Espalda media</td></tr>
                      <tr><td style="padding:3px 6px"><b>K4</b></td><td>Hips-backedge</td><td>Cadera</td></tr>
                      <tr><td style="padding:3px 6px"><b>K5</b></td><td>Neck-middle</td><td>Cervical media</td></tr>
                      <tr><td style="padding:3px 6px"><b>K6</b></td><td>Jaw</td><td>Mandíbula</td></tr>
                      <tr><td style="padding:3px 6px"><b>K7</b></td><td>Chin</td><td>Mentón</td></tr>
                      <tr><td style="padding:3px 6px"><b>K8</b></td><td>Shoulder-back</td><td>Escápula</td></tr>
                    </table>
                    """)

                with gr.Accordion("Grabación de sesión", open=True):
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
            outputs=[webcam, metrics_data],
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

        leve_slider.change(
            fn=_update_thresholds,
            inputs=[leve_slider, critico_slider],
            outputs=[threshold_table, threshold_msg],
        )
        critico_slider.change(
            fn=_update_thresholds,
            inputs=[leve_slider, critico_slider],
            outputs=[threshold_table, threshold_msg],
        )

        # Inyectar JS de métricas — Gradio 6 elimina <script> en gr.HTML,
        # app.load(js=...) es la única forma confiable de ejecutar JS al cargar
        app.load(fn=None, js=METRICS_JS)

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
