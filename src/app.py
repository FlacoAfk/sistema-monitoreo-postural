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
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO

# winsound solo existe en Windows — importación condicional
try:
    import winsound as _winsound
    def _beep(freq: int, duration: int) -> None:
        _winsound.Beep(freq, duration)
except ImportError:
    def _beep(freq: int, duration: int) -> None:
        pass  # No-op en Linux/Mac

# ── Detección automática de GPU + capacidades ────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16: bool = False  # Half-precision (FP16) — activado si GPU lo soporta
INFER_IMGSZ: int = 256  # Tamaño de inferencia (adaptativo según hardware)

print(f"[INFO] Dispositivo de inferencia detectado: {DEVICE.upper()}")
if DEVICE == "cuda":
    _gpu_props = torch.cuda.get_device_properties(0)
    _gpu_name = _gpu_props.name
    _gpu_vram_gb = _gpu_props.total_memory / 1024**3
    _gpu_compute = f"{_gpu_props.major}.{_gpu_props.minor}"
    print(f"[INFO] GPU: {_gpu_name}")
    print(f"[INFO] VRAM: {_gpu_vram_gb:.1f} GB | Compute Capability: {_gpu_compute}")
    # cuDNN benchmark: auto-tuning de kernels para tamaño de input fijo → +10-15% FPS
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] ✓ cuDNN benchmark activado (auto-tuning kernels)")
    # FP16 soportado en compute capability >= 5.3 (Maxwell+) con buen rendimiento en >= 7.0 (Volta+)
    # Pascal (6.x) lo soporta con throughput reducido pero sigue siendo más rápido que FP32
    if _gpu_props.major >= 6:
        USE_FP16 = True
        print(f"[INFO] ✓ FP16 (half-precision) ACTIVADO — inferencia acelerada")
    elif _gpu_props.major == 5 and _gpu_props.minor >= 3:
        USE_FP16 = True
        print(f"[INFO] ✓ FP16 activado (soporte básico, compute {_gpu_compute})")
    else:
        print(f"[INFO] ✗ FP16 no disponible (compute {_gpu_compute} < 5.3) — usando FP32")
    INFER_IMGSZ = 256  # GPU: tamaño estándar
else:
    # ── Optimizaciones CPU ────────────────────────────────────────────────
    import os
    _cpu_count = os.cpu_count() or 4
    # Limitar threads de PyTorch para evitar over-subscription en CPUs modestas
    _optimal_threads = max(2, min(_cpu_count, 8))
    torch.set_num_threads(_optimal_threads)
    torch.set_num_interop_threads(max(1, _optimal_threads // 2))
    INFER_IMGSZ = 192  # CPU: tamaño reducido para inferencia más rápida
    print(f"[INFO] CPU: {_cpu_count} cores · PyTorch threads: {_optimal_threads}")
    print(f"[INFO] Tamaño de inferencia reducido: {INFER_IMGSZ}px (optimización CPU)")
    print("[INFO] Sin GPU CUDA — inferencia en CPU (FP32)")

# Skip ratio: process every Nth frame through YOLO, skip remaining
# GPU+FP16: skip 1/2 | GPU sin FP16: skip 1/3 | CPU: skip 1/4 (más agresivo)
if DEVICE == "cuda":
    SKIP_RATIO = 2 if USE_FP16 else 3
else:
    SKIP_RATIO = 4  # CPU necesita más skip para mantener fluidez

# ── Info string para UI (adaptativo) ─────────────────────────────────────────
if DEVICE == "cuda":
    _GPU_STATUS = f"🟢 GPU: {_gpu_name} ({_gpu_vram_gb:.1f}GB) · FP16: {'✓' if USE_FP16 else '✗'} · Skip: 1/{SKIP_RATIO}"
else:
    _GPU_STATUS = f"🟡 CPU ({_cpu_count} cores) · FP32 · img:{INFER_IMGSZ}px · Skip: 1/{SKIP_RATIO}"
LANGS: dict[str, dict[str, str]] = {
    "es": {
        "title":          "Sistema de Monitoreo Postural en Tiempo Real",
        "subtitle":       "Estimación del Combined Posture Index (CPI) — curvatura escapular + ángulo lumbar.<br>Universidad Surcolombiana &nbsp;·&nbsp; Castañeda Guzmán &amp; Idarraga Plazas, 2026",
        "brand":          "Biomecánica Computacional — Procesamiento de Video",
        "model_label":    "Modelo YOLO-Pose",
        "model_info_def": "**Modelo actual:** YOLOv8n — Más rápido (22ms, SCORE 0.9189)",
        "calib_title":    "Calibrar umbrales CPI",
        "kp_title":       "Referencia de keypoints",
        "session_title":  "Grabación de sesión",
        "btn_start":      "▶ Iniciar sesión",
        "btn_stop":       "⏹ Detener sesión",
        "session_idle":   "_Sin sesión activa_",
        "session_active": "🔴 Sesión activa — grabando datos...",
        "session_done":   "✓ Sesión detenida — {n} frames grabados. Exporta el CSV para analizar los datos.",
        "export_btn":     "⬇ Exportar CSV",
        "export_file":    "Archivo CSV",
        "thresh_hint":    "_Ajusta los sliders para calibrar_",
        "thresh_leve":    "Umbral Leve",
        "thresh_crit":    "Umbral Crítico",
        "thresh_ok":      "✓ Umbrales actualizados — Leve: {leve:.0f} | Crítico: {crit:.0f}",
        "thresh_err":     "⚠ Umbral leve ({leve:.0f}) debe ser menor que crítico ({crit:.0f})",
        "lang_label":     "Idioma / Language",
        # JS strings (embedded in METRICS_JS)
        "js_cpi_title":   "CPI — Combined Posture Index",
        "js_lumbar":      "Lumbar",
        "js_curv":        "Curvatura",
        "js_bad_time":    "Mala postura",
        "js_fps":         "FPS",
        "js_conf_title":  "Confianza detección",
        "js_spark_title": "Historial CPI — últimos 60s",
        "js_spark_ago":   "60s atrás",
        "js_spark_now":   "ahora",
        "js_ok":          "✓ POSTURA CORRECTA",
        "js_warn":        "⚠ ALERTA LEVE",
        "js_crit":        "✕ ALERTA CRÍTICA",
        "js_nd":          "— NO DETECTADO",
        "js_ni":          "— NO INICIADO",
        "js_badge_ok":    "CORRECTO",
        "js_badge_warn":  "ALERTA LEVE",
        "js_badge_crit":  "ALERTA CRÍTICA",
        "js_badge_nd":    "NO DETECTADO",
        "js_badge_ni":    "NO INICIADO",
        "js_detail_ok":   "Alineación cervical dentro de parámetros ergonómicos",
        "js_detail_warn": "Cabeza ligeramente adelantada · {t}s acumulados",
        "js_detail_crit": "Protrusión cefálica severa · {t}s acumulados",
        "js_detail_nd":   "Colóquese frente a la cámara para iniciar",
        "js_alert_title": "⚠ Mala postura: {t}s",
        "js_weak_det":    "⚠ Detección débil — datos no confiables",
        "js_no_data":     "Recopilando datos...",
        "js_no_person":   "Sin deteccion — Colocate frente a la camara",
        "js_thresh_ok":   "Correcto",
        "js_thresh_warn": "Alerta leve",
        "js_thresh_crit": "Alerta critica",
        "js_thresh_ok_d": "Columna alineada, postura recta",
        "js_thresh_warn_d": "Curvatura dorsal leve",
        "js_thresh_crit_d": "Cifosis / hombros caidos",
        "js_summary":     "Resumen de Sesión",
        "js_frames":      "Frames analizados",
        "js_pct_ok":      "Postura correcta",
        "js_pct_warn":    "Alerta leve",
        "js_pct_crit":    "Alerta crítica",
        "js_avg_cpi":     "CPI promedio",
        "js_max_cpi":     "CPI máximo",
        "js_min_cpi":     "CPI mínimo",
        "js_bad_total":   "Tiempo mala postura",
        # Python-side strings
        "webcam_label":   "Cámara en Vivo",
        "model_info_sel": "Selecciona el modelo para inferencia",
        "export_no_data": "⚠ No hay datos en la sesión actual. Inicia el monitoreo primero.",
        "export_buf_warn":"⚠ Buffer muy grande ({n} filas). Exportando de todas formas...",
        "export_success": "✓ CSV exportado: {fname} ({n} filas)",
        "kp_col_name":    "Nombre",
        "kp_col_loc":     "Ubicación",
        "thresh_col_status":  "Estado",
        "thresh_col_meaning": "Significado",
        "kp_locations": ["Occipital", "C7 cervical", "Acromion", "Espalda media",
                         "Cadera", "Cervical media", "Mandíbula", "Mentón", "Escápula"],
    },
    "en": {
        "title":          "Real-Time Postural Monitoring System",
        "subtitle":       "Combined Posture Index (CPI) estimation — scapular curvature + lumbar angle.<br>Universidad Surcolombiana &nbsp;·&nbsp; Castañeda Guzmán &amp; Idarraga Plazas, 2026",
        "brand":          "Computational Biomechanics — Video Processing",
        "model_label":    "YOLO-Pose Model",
        "model_info_def": "**Current model:** YOLOv8n — Fastest (22ms, SCORE 0.9189)",
        "calib_title":    "Calibrate CPI thresholds",
        "kp_title":       "Keypoint reference",
        "session_title":  "Session recording",
        "btn_start":      "▶ Start session",
        "btn_stop":       "⏹ Stop session",
        "session_idle":   "_No active session_",
        "session_active": "🔴 Session active — recording data...",
        "session_done":   "✓ Session stopped — {n} frames recorded. Export CSV to analyze.",
        "export_btn":     "⬇ Export CSV",
        "export_file":    "CSV File",
        "thresh_hint":    "_Adjust sliders to calibrate_",
        "thresh_leve":    "Mild threshold",
        "thresh_crit":    "Critical threshold",
        "thresh_ok":      "✓ Thresholds updated — Mild: {leve:.0f} | Critical: {crit:.0f}",
        "thresh_err":     "⚠ Mild threshold ({leve:.0f}) must be less than critical ({crit:.0f})",
        "lang_label":     "Idioma / Language",
        "js_cpi_title":   "CPI — Combined Posture Index",
        "js_lumbar":      "Lumbar",
        "js_curv":        "Curvature",
        "js_bad_time":    "Bad posture",
        "js_fps":         "FPS",
        "js_conf_title":  "Detection confidence",
        "js_spark_title": "CPI history — last 60s",
        "js_spark_ago":   "60s ago",
        "js_spark_now":   "now",
        "js_ok":          "✓ CORRECT POSTURE",
        "js_warn":        "⚠ MILD ALERT",
        "js_crit":        "✕ CRITICAL ALERT",
        "js_nd":          "— NOT DETECTED",
        "js_ni":          "— NOT STARTED",
        "js_badge_ok":    "CORRECT",
        "js_badge_warn":  "MILD ALERT",
        "js_badge_crit":  "CRITICAL ALERT",
        "js_badge_nd":    "NOT DETECTED",
        "js_badge_ni":    "NOT STARTED",
        "js_detail_ok":   "Cervical alignment within ergonomic parameters",
        "js_detail_warn": "Head slightly forward · {t}s accumulated",
        "js_detail_crit": "Severe cephalic protrusion · {t}s accumulated",
        "js_detail_nd":   "Position yourself in front of the camera",
        "js_alert_title": "⚠ Bad posture: {t}s",
        "js_weak_det":    "⚠ Weak detection — unreliable data",
        "js_no_data":     "Collecting data...",
        "js_no_person":   "No detection — stand in front of the camera",
        "js_thresh_ok":   "Correct",
        "js_thresh_warn": "Mild alert",
        "js_thresh_crit": "Critical alert",
        "js_thresh_ok_d": "Aligned spine, straight posture",
        "js_thresh_warn_d": "Mild dorsal curvature",
        "js_thresh_crit_d": "Kyphosis / dropped shoulders",
        "js_summary":     "Session Summary",
        "js_frames":      "Analyzed frames",
        "js_pct_ok":      "Correct posture",
        "js_pct_warn":    "Mild alert",
        "js_pct_crit":    "Critical alert",
        "js_avg_cpi":     "Average CPI",
        "js_max_cpi":     "Max CPI",
        "js_min_cpi":     "Min CPI",
        "js_bad_total":   "Bad posture time",
        # Python-side strings
        "webcam_label":   "Live Camera",
        "model_info_sel": "Select model for inference",
        "export_no_data": "⚠ No session data. Start monitoring first.",
        "export_buf_warn":"⚠ Large buffer ({n} rows). Exporting anyway...",
        "export_success": "✓ CSV exported: {fname} ({n} rows)",
        "kp_col_name":    "Name",
        "kp_col_loc":     "Location",
        "thresh_col_status":  "Status",
        "thresh_col_meaning": "Meaning",
        "kp_locations": ["Occipital", "C7 cervical", "Acromion", "Mid-back",
                         "Hip", "Mid-cervical", "Jaw", "Chin", "Scapula"],
    },
    "pt": {
        "title":          "Sistema de Monitoramento Postural em Tempo Real",
        "subtitle":       "Estimativa do Combined Posture Index (CPI) — curvatura escapular + ângulo lombar.<br>Universidad Surcolombiana &nbsp;·&nbsp; Castañeda Guzmán &amp; Idarraga Plazas, 2026",
        "brand":          "Biomecânica Computacional — Processamento de Vídeo",
        "model_label":    "Modelo YOLO-Pose",
        "model_info_def": "**Modelo atual:** YOLOv8n — Mais rápido (22ms, SCORE 0.9189)",
        "calib_title":    "Calibrar limiares CPI",
        "kp_title":       "Referência de keypoints",
        "session_title":  "Gravação de sessão",
        "btn_start":      "▶ Iniciar sessão",
        "btn_stop":       "⏹ Parar sessão",
        "session_idle":   "_Sem sessão ativa_",
        "session_active": "🔴 Sessão ativa — gravando dados...",
        "session_done":   "✓ Sessão encerrada — {n} frames gravados. Exporte o CSV para analisar.",
        "export_btn":     "⬇ Exportar CSV",
        "export_file":    "Arquivo CSV",
        "thresh_hint":    "_Ajuste os sliders para calibrar_",
        "thresh_leve":    "Limiar Leve",
        "thresh_crit":    "Limiar Crítico",
        "thresh_ok":      "✓ Limiares atualizados — Leve: {leve:.0f} | Crítico: {crit:.0f}",
        "thresh_err":     "⚠ Limiar leve ({leve:.0f}) deve ser menor que crítico ({crit:.0f})",
        "lang_label":     "Idioma / Language",
        "js_cpi_title":   "CPI — Combined Posture Index",
        "js_lumbar":      "Lombar",
        "js_curv":        "Curvatura",
        "js_bad_time":    "Má postura",
        "js_fps":         "FPS",
        "js_conf_title":  "Confiança de detecção",
        "js_spark_title": "Histórico CPI — últimos 60s",
        "js_spark_ago":   "60s atrás",
        "js_spark_now":   "agora",
        "js_ok":          "✓ POSTURA CORRETA",
        "js_warn":        "⚠ ALERTA LEVE",
        "js_crit":        "✕ ALERTA CRÍTICA",
        "js_nd":          "— NÃO DETECTADO",
        "js_ni":          "— NÃO INICIADO",
        "js_badge_ok":    "CORRETO",
        "js_badge_warn":  "ALERTA LEVE",
        "js_badge_crit":  "ALERTA CRÍTICA",
        "js_badge_nd":    "NÃO DETECTADO",
        "js_badge_ni":    "NÃO INICIADO",
        "js_detail_ok":   "Alinhamento cervical dentro dos parâmetros ergonômicos",
        "js_detail_warn": "Cabeça levemente avançada · {t}s acumulados",
        "js_detail_crit": "Protrusão cefálica severa · {t}s acumulados",
        "js_detail_nd":   "Posicione-se em frente à câmera para iniciar",
        "js_alert_title": "⚠ Má postura: {t}s",
        "js_weak_det":    "⚠ Detecção fraca — dados não confiáveis",
        "js_no_data":     "Coletando dados...",
        "js_no_person":   "Sem detecção — posicione-se em frente à câmera",
        "js_thresh_ok":   "Correto",
        "js_thresh_warn": "Alerta leve",
        "js_thresh_crit": "Alerta crítica",
        "js_thresh_ok_d": "Coluna alinhada, postura ereta",
        "js_thresh_warn_d": "Curvatura dorsal leve",
        "js_thresh_crit_d": "Cifose / ombros caídos",
        "js_summary":     "Resumo da Sessão",
        "js_frames":      "Frames analisados",
        "js_pct_ok":      "Postura correta",
        "js_pct_warn":    "Alerta leve",
        "js_pct_crit":    "Alerta crítica",
        "js_avg_cpi":     "CPI médio",
        "js_max_cpi":     "CPI máximo",
        "js_min_cpi":     "CPI mínimo",
        "js_bad_total":   "Tempo de má postura",
        # Python-side strings
        "webcam_label":   "Câmera ao Vivo",
        "model_info_sel": "Selecione o modelo para inferência",
        "export_no_data": "⚠ Sem dados na sessão atual. Inicie o monitoramento primeiro.",
        "export_buf_warn":"⚠ Buffer muito grande ({n} linhas). Exportando mesmo assim...",
        "export_success": "✓ CSV exportado: {fname} ({n} linhas)",
        "kp_col_name":    "Nome",
        "kp_col_loc":     "Localização",
        "thresh_col_status":  "Estado",
        "thresh_col_meaning": "Significado",
        "kp_locations": ["Occipital", "C7 cervical", "Acrômio", "Meio das costas",
                         "Quadril", "Cervical média", "Mandíbula", "Queixo", "Escápula"],
    },
}

DEFAULT_LANG = "es"

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
# MODELS_DIR: por defecto <repo_root>/models/
# Se puede sobreescribir con la variable de entorno POSTURE_MODELS_DIR.
import os as _os
MODELS_DIR = Path(_os.environ.get("POSTURE_MODELS_DIR", "") or Path(__file__).resolve().parent.parent / "models")
MODEL_CONFIGS = [
    {
        "name": "yolov8n 🚀 (Más rápido — 22ms)",
        "path": str(MODELS_DIR / "yolov8n.pt"),
        "key": "yolov8n",
    },
    {
        "name": "yolov5n 🎯 (Mejor detección — 95.2%)",
        "path": str(MODELS_DIR / "yolov5n.pt"),
        "key": "yolov5n",
    },
    {
        "name": "yolov26n ⚖️ (Balanceado)",
        "path": str(MODELS_DIR / "yolov26n.pt"),
        "key": "yolov26n",
    },
    {
        "name": "yolov11n ⭐ (Excel OKS)",
        "path": str(MODELS_DIR / "yolov11n.pt"),
        "key": "yolov11n",
    },
]

# Lookup rápido O(1) para evitar loop cada frame
MODEL_LOOKUP: dict[str, dict[str, str]] = {c["name"]: c for c in MODEL_CONFIGS}

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
        self._last_conf: float = 0.0
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
        """Carga o recarga el modelo YOLO en GPU/CPU con optimización FP16."""
        if self.model is None or model_path != getattr(self, "_loaded_path", None):
            print(f"[INFO] Cargando modelo: {model_path}")
            print(f"[INFO] Dispositivo: {DEVICE.upper()} | FP16: {'SÍ' if USE_FP16 else 'NO'}")
            self.model = YOLO(model_path)
            self.model.to(DEVICE)  # Mover a GPU (ultralytics 8.x)
            self._loaded_path = model_path
            # Warmup con tamaño de inferencia real para compilar kernels
            dummy = np.zeros((INFER_IMGSZ, INFER_IMGSZ, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, imgsz=INFER_IMGSZ, half=USE_FP16)
            print(f"[INFO] Modelo cargado en {next(self.model.model.parameters()).device} ✓")
            if DEVICE == "cuda":
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
        # Skip path: reuse overlay from last inference frame (sin copia — cvtColor no muta src)
        cpi_s, stat_s, lumbar_s, curv_s, bad_s, alert_s = state._last_posture_result or (0, "NO DETECTADO", 0, 0, 0, False)
        out_rgb = cv2.cvtColor(state._last_overlay_bgr, cv2.COLOR_BGR2RGB)

        # FPS tracking: include skip frames for accurate display rate
        state._fps_times.append(time.time())
        if len(state._fps_times) > 30:
            state._fps_times.pop(0)
        if len(state._fps_times) >= 2:
            elapsed = state._fps_times[-1] - state._fps_times[0]
            state._current_fps = (len(state._fps_times) - 1) / elapsed if elapsed > 0 else 0

        return out_rgb, _build_metrics_json(cpi_s, stat_s, bad_s, lumbar_s, curv_s, state._current_fps, state._last_conf, alert_s, history=state._cpi_history)

    # Convertir RGB → BGR para YOLO/OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ── YOLO inference (FP16 si GPU lo soporta) ───────────────────────────
    try:
        t_inf = time.time()
        preds = state.model(frame_bgr, verbose=False, conf=0.25, imgsz=INFER_IMGSZ, max_det=MAX_PERSONS, half=USE_FP16)
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
        _no_person_msg = LANGS.get(_current_lang, LANGS["es"])["js_no_person"]
        cv2.putText(out, _no_person_msg,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        state._last_overlay_bgr = out.copy()
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        state._last_conf = 0.0
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
        _no_person_msg = LANGS.get(_current_lang, LANGS["es"])["js_no_person"]
        cv2.putText(out, _no_person_msg,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        state._last_overlay_bgr = out.copy()
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        state._last_conf = 0.0
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
    state._last_conf = avg_confidence

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
                _beep(1000, 300)  # 1000Hz, 300ms
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
_STATUS_TO_CODE: dict[str, str] = {
    "CORRECTO":       "ok",
    "ALERTA LEVE":    "warn",
    "ALERTA CRÍTICA": "crit",
    "NO DETECTADO":   "nd",
    "NO INICIADO":    "ni",
}
_CODE_COLOR: dict[str, str] = {
    "ok":   "#22c55e",
    "warn": "#f59e0b",
    "crit": "#ef4444",
    "nd":   "#94a3b8",
    "ni":   "#94a3b8",
}

def _build_metrics_json(cpi: float = 0, status: str = "NO DETECTADO",
                         bad_time: float = 0, lumbar: float = 0,
                         curv: float = 0, fps: float = 0,
                         conf: float = 0.0, alert: bool = False,
                         history: list = None) -> str:
    """Serializa métricas a JSON para el panel estático.
    Envía status_code (ok/warn/crit/nd/ni) — el JS traduce según idioma activo."""
    import json as _json
    code = _STATUS_TO_CODE.get(status, "nd")
    history_values = [round(v, 1) for _, v in history] if history else []
    payload = _json.dumps({
        "cpi":         round(cpi, 1),
        "status_code": code,
        "bad_time":    round(bad_time, 1),
        "lumbar":      round(lumbar, 1),
        "curv":        round(curv, 2),
        "fps":         round(fps, 1),
        "conf":        round(conf, 3),
        "alert":       alert,
        "color":       _CODE_COLOR.get(code, "#94a3b8"),
        "history":     history_values,
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
    t = LANGS.get(_current_lang, LANGS["es"])
    if not state.session_data:
        return None, t["export_no_data"]
    if len(state.session_data) > 10000:
        return None, t["export_buf_warn"].format(n=len(state.session_data))
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
    return tmp.name, t["export_success"].format(fname=fname, n=len(state.session_data))


def _build_summary_html(summary: Optional[dict]) -> str:
    """Construye HTML de la tarjeta de resumen de sesión."""
    if summary is None:
        return ""
    t = LANGS.get(_current_lang, LANGS["es"])
    return f"""<div class="pm-card" style="margin-top:12px">
  <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;color:var(--pm-text-3);margin-bottom:14px">
    {t['js_summary']}
  </div>
  <table class="pm-table">
    <tr><td>{t['js_frames']}</td><td><strong>{summary['total_frames']}</strong></td></tr>
    <tr><td>{t['js_pct_ok']}</td><td><strong style="color:var(--pm-green)">{summary['pct_correcto']}%</strong></td></tr>
    <tr><td>{t['js_pct_warn']}</td><td><strong style="color:var(--pm-amber)">{summary['pct_leve']}%</strong></td></tr>
    <tr><td>{t['js_pct_crit']}</td><td><strong style="color:var(--pm-red)">{summary['pct_critico']}%</strong></td></tr>
    <tr><td>{t['js_avg_cpi']}</td><td><strong>{summary['avg_cpi']}</strong></td></tr>
    <tr><td>{t['js_max_cpi']}</td><td><strong>{summary['max_cpi']}</strong></td></tr>
    <tr><td>{t['js_min_cpi']}</td><td><strong>{summary['min_cpi']}</strong></td></tr>
    <tr><td>{t['js_bad_total']}</td><td><strong>{summary['total_bad_posture_s']}s</strong></td></tr>
  </table>
</div>"""


# ── CSS y tema ──────────────────────────────────────────────────────────────
CSS = """
:root {
    --pm-bg: #0a0e1a;
    --pm-surface: rgba(15, 23, 42, 0.7);
    --pm-surface-2: rgba(30, 41, 59, 0.5);
    --pm-border: rgba(99, 102, 241, 0.18);
    --pm-border-strong: rgba(99, 102, 241, 0.35);
    --pm-text-1: #f1f5f9;
    --pm-text-2: #cbd5e1;
    --pm-text-3: #94a3b8;
    --pm-text-muted: #64748b;
    --pm-accent: #6366f1;
    --pm-accent-cyan: #06b6d4;
    --pm-success: #22c55e;
    --pm-warn: #f59e0b;
    --pm-danger: #ef4444;
    --pm-radius: 14px;
    --pm-radius-sm: 8px;
    --pm-space-1: 6px;
    --pm-space-2: 10px;
    --pm-space-3: 16px;
    --pm-space-4: 24px;
    --pm-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

:root[data-pm-theme="light"] {
    --pm-bg: #f1f5f9;
    --pm-surface: rgba(255, 255, 255, 0.95);
    --pm-surface-2: rgba(241, 245, 249, 0.85);
    --pm-border: rgba(99, 102, 241, 0.2);
    --pm-border-strong: rgba(99, 102, 241, 0.4);
    --pm-text-1: #0f172a;
    --pm-text-2: #1e293b;
    --pm-text-3: #475569;
    --pm-text-muted: #64748b;
    --pm-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    /* Override Gradio internal CSS variables for light mode */
    --body-text-color: #0f172a !important;
    --body-text-color-subdued: #475569 !important;
    --neutral-200: #1e293b !important;
    --neutral-400: #475569 !important;
    --block-title-text-color: #0f172a !important;
    --color-text-body: #0f172a !important;
}

/* ── Gear popup (bottom-right) ── */
.gear-root {
    position: fixed !important;
    bottom: 24px !important;
    right: 24px !important;
    z-index: 9999 !important;
    width: auto !important;
    height: auto !important;
    background: transparent !important;
    border: none !important;
    border-bottom: none !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
.gear-root > div,
.gear-root > * {
    border: none !important;
    border-bottom: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}
#pm-gear-wrap { position: relative; }
#pm-gear-icon {
    font-size: 26px;
    opacity: 0.4;
    cursor: pointer;
    transition: opacity 0.2s;
    display: block;
    text-align: center;
    line-height: 1;
    user-select: none;
}
#pm-gear-icon:hover { opacity: 1; }
#pm-gear-popup {
    display: none;
    position: absolute;
    bottom: 40px;
    right: 0;
    flex-direction: column;
    gap: 4px;
    background: var(--pm-surface);
    border: 1px solid var(--pm-border);
    border-radius: 10px;
    padding: 6px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    min-width: 110px;
}
.gear-root:hover #pm-gear-popup,
#pm-gear-popup[style*="flex"] {
    display: flex !important;
}
.pm-gear-lang {
    background: none !important;
    border: none !important;
    color: var(--pm-text-2) !important;
    cursor: pointer !important;
    font-size: 13px !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    text-align: left !important;
    width: 100% !important;
    transition: none !important;
}
.pm-gear-lang:hover {
    background: var(--pm-surface-2) !important;
    color: var(--pm-text-1) !important;
}
.pm-gear-divider {
    height: 1px;
    background: var(--pm-border);
    margin: 2px 6px;
}
#pm-theme-toggle {
    background: none !important;
    border: none !important;
    color: var(--pm-text-2) !important;
    cursor: pointer !important;
    font-size: 15px !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    width: 100% !important;
    text-align: left !important;
    transition: none !important;
}
#pm-theme-toggle:hover {
    background: var(--pm-surface-2) !important;
    color: var(--pm-text-1) !important;
}
:root[data-pm-theme="light"] .pm-gear-lang,
:root[data-pm-theme="light"] #pm-theme-toggle {
    color: #475569 !important;
}
:root[data-pm-theme="light"] .pm-gear-lang:hover,
:root[data-pm-theme="light"] #pm-theme-toggle:hover {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}
:root[data-pm-theme="light"] #pm-gear-popup {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.1) !important;
}
:root[data-pm-theme="light"] .pm-gear-divider {
    background: #e2e8f0 !important;
}
:root[data-pm-theme="light"] #pm-gear-icon {
    opacity: 0.55 !important;
}
/* Hidden dropdown (kept for backend callback) */
.pm-lang-hidden {
    visibility: hidden !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}

* { box-sizing: border-box; }

body, html, .gradio-container {
    background: var(--pm-bg) !important;
    background-color: var(--pm-bg) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--pm-text-1) !important;
}

.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: var(--pm-space-3) !important; }
/* ── Header ── */
.pm-header {
    position: relative;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(6, 182, 212, 0.08));
    border: 1px solid var(--pm-border);
    border-radius: var(--pm-radius);
    padding: var(--pm-space-4);
    margin-bottom: var(--pm-space-4);
    position: relative;
    overflow: hidden;
}
.pm-header::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.08), transparent 40%);
    pointer-events: none;
}
.pm-header h1 {
    font-size: 22px;
    font-weight: 700;
    color: var(--pm-text-1);
    margin: 0 0 6px;
    letter-spacing: -0.3px;
}
.pm-header p {
    color: var(--pm-text-3);
    font-size: 12px;
    line-height: 1.6;
    margin: 0 0 10px;
}
.pm-header .brand-line {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    font-weight: 600;
    color: var(--pm-accent-cyan);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.pm-live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--pm-success);
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.5);
    animation: pm-live-pulse 2s ease-in-out infinite;
}
@keyframes pm-live-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.5); }
    50% { box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
}

/* ── Panel metrics root ── */
#pm-metrics-root { font-family: 'Inter', sans-serif; color: var(--pm-text-1); }

.pm-card {
    background: var(--pm-surface);
    border: 1px solid var(--pm-border);
    border-radius: var(--pm-radius);
    padding: var(--pm-space-3);
    margin-bottom: var(--pm-space-2);
    backdrop-filter: blur(10px);
}

.pm-section-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--pm-accent);
    margin-bottom: var(--pm-space-2);
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Gauge ── */
.pm-gauge-wrap {
    position: relative;
    width: 140px; height: 140px;
    margin: 0 auto 8px;
}
.pm-gauge-track { fill: none; stroke: rgba(99, 102, 241, 0.12); stroke-width: 11; }
.pm-gauge-fill {
    fill: none; stroke-width: 11; stroke-linecap: round;
    transform: rotate(-90deg); transform-origin: 50% 50%;
    stroke-dasharray: 326.73; stroke-dashoffset: 326.73;
    transition: stroke-dashoffset 0.6s cubic-bezier(0.22, 1, 0.36, 1), stroke 0.4s ease;
    filter: drop-shadow(0 0 6px currentColor);
}
.pm-gauge-value {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 32px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    transition: color 0.4s ease;
    letter-spacing: -1px;
}
.pm-gauge-sublabel {
    text-align: center;
    font-size: 10px;
    font-weight: 600;
    color: var(--pm-text-3);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Badges ── */
.pm-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    transition: all 0.3s ease;
}
.badge-ok   { background: rgba(34, 197, 94, 0.12);  color: var(--pm-success); border: 1px solid rgba(34, 197, 94, 0.3); }
.badge-warn { background: rgba(245, 158, 11, 0.12); color: var(--pm-warn);    border: 1px solid rgba(245, 158, 11, 0.3); }
.badge-crit { background: rgba(239, 68, 68, 0.12);  color: var(--pm-danger);  border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-nd   { background: rgba(148, 163, 184, 0.08); color: var(--pm-text-3); border: 1px solid rgba(148, 163, 184, 0.2); }

/* ── Metrics grid ── */
.pm-metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: var(--pm-space-3);
}
.pm-metric-item {
    background: var(--pm-surface-2);
    border: 1px solid var(--pm-border);
    border-radius: var(--pm-radius-sm);
    padding: 10px 12px;
    transition: border-color 0.3s ease;
}
.pm-metric-item:hover { border-color: var(--pm-border-strong); }
.pm-metric-item .label {
    font-size: 9px;
    color: var(--pm-text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
    font-weight: 600;
}
.pm-metric-item .value {
    font-size: 20px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--pm-text-1);
    transition: opacity 0.15s ease, color 0.3s ease;
    letter-spacing: -0.5px;
}

/* ── Confidence bar ── */
.pm-conf-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0 4px;
    border: 1px solid rgba(255, 255, 255, 0.04);
}
.pm-conf-fill {
    height: 100%;
    border-radius: 6px;
    width: 0%;
    transition: width 0.4s cubic-bezier(0.22, 1, 0.36, 1), background 0.3s ease;
    box-shadow: 0 0 8px currentColor;
}

/* ── Status card ── */
.pm-status {
    border-radius: var(--pm-radius);
    padding: 14px 16px;
    margin-bottom: var(--pm-space-2);
    border: 1px solid transparent;
    transition: background 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease;
}
.pm-status-nd   { background: rgba(148, 163, 184, 0.06); border-color: rgba(148, 163, 184, 0.18); }
.pm-status-ok   { background: rgba(34, 197, 94, 0.08);   border-color: rgba(34, 197, 94, 0.3); box-shadow: 0 0 20px rgba(34, 197, 94, 0.1); }
.pm-status-warn { background: rgba(245, 158, 11, 0.08);  border-color: rgba(245, 158, 11, 0.3); box-shadow: 0 0 20px rgba(245, 158, 11, 0.1); }
.pm-status-crit { background: rgba(239, 68, 68, 0.1);    border-color: rgba(239, 68, 68, 0.35); box-shadow: 0 0 20px rgba(239, 68, 68, 0.15); }
.pm-status.pulse { animation: pm-pulse 1.8s ease-in-out infinite; }
@keyframes pm-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.15); }
    50% { box-shadow: 0 0 32px rgba(239, 68, 68, 0.35); }
}
.pm-status-icon {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.pm-status-detail {
    font-size: 11px;
    color: var(--pm-text-3);
    margin-top: 6px;
    line-height: 1.5;
}

/* ── Sparkline ── */
#pm-sparkline-svg { display: block; width: 100%; }

/* ── Alert popup ── */
#pm-alert-popup {
    display: none; opacity: 0;
    position: fixed; bottom: 24px; right: 24px; z-index: 9999;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: #fff;
    border-radius: 12px;
    padding: 14px 20px;
    font-size: 13px;
    font-weight: 700;
    box-shadow: 0 8px 40px rgba(239, 68, 68, 0.5);
    transition: opacity 0.3s ease, transform 0.3s ease;
    transform: translateY(10px);
}

/* ── Gradio overrides ── */
.gradio-html, .gr-block, .block { background: transparent !important; border: none !important; }
.gr-block > label, .gr-label label, span.label, .block-title { color: var(--pm-accent) !important; font-size: 12px !important; font-weight: 700 !important; background: var(--pm-surface); padding: 4px 8px; border-radius: 4px; }
.gr-button {
    background: linear-gradient(135deg, var(--pm-accent), #4f46e5) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 12px !important;
    transition: all 0.2s ease !important;
}
.gr-button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important; }
.gr-button-secondary {
    background: var(--pm-surface-2) !important;
    border: 1px solid var(--pm-border-strong) !important;
    color: var(--pm-text-2) !important;
}

/* Accordion */
.gr-accordion {
    background: var(--pm-surface-2) !important;
    border: 1px solid var(--pm-border-strong) !important;
    border-radius: var(--pm-radius-sm) !important;
    margin-bottom: var(--pm-space-2) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}
.gr-accordion > button {
    color: var(--pm-text-1) !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 10px 14px !important;
    background: var(--pm-surface) !important;
    border-radius: var(--pm-radius-sm) !important;
}

/* Dropdown / Slider */
.gr-dropdown, .gr-slider { background: var(--pm-surface) !important; border-radius: 8px !important; }
.gr-dropdown select { background: var(--pm-surface-2) !important; color: var(--pm-text-1) !important; border: 1px solid var(--pm-border) !important; }

/* Image container */
.gr-image {
    background: #000 !important;
    border: 1px solid var(--pm-border) !important;
    border-radius: var(--pm-radius) !important;
    overflow: hidden !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--pm-border-strong); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--pm-accent); }

/* Sidebar width lock */
.pm-sidebar { max-width: 380px !important; }
.pm-sidebar .gr-accordion,
.pm-sidebar .gr-slider,
.pm-sidebar .gr-button,
.pm-sidebar .gr-file,
.pm-sidebar > * { width: 100% !important; min-width: 0 !important; }
.pm-sidebar .gr-accordion > div { width: 100% !important; }

/* Left column expansion */
.pm-leftcol { width: 100% !important; }
.pm-leftcol > * { width: 100% !important; }

/* ── Light mode: Gradio native element overrides ── */
:root[data-pm-theme="light"] .gradio-container,
:root[data-pm-theme="light"] html,
:root[data-pm-theme="light"] body { background: var(--pm-bg) !important; background-color: var(--pm-bg) !important; color: var(--pm-text-1) !important; }
:root[data-pm-theme="light"] .block { background: var(--pm-surface) !important; border-color: var(--pm-border) !important; }
:root[data-pm-theme="light"] input, :root[data-pm-theme="light"] textarea,
:root[data-pm-theme="light"] select, :root[data-pm-theme="light"] .wrap {
    background: #ffffff !important; color: #0f172a !important; border-color: #cbd5e1 !important;
}
:root[data-pm-theme="light"] .label-wrap, :root[data-pm-theme="light"] label { color: var(--pm-text-2) !important; }
:root[data-pm-theme="light"] .accordion { background: var(--pm-surface) !important; border-color: var(--pm-border) !important; }
:root[data-pm-theme="light"] .gr-button.secondary { background: #e2e8f0 !important; color: #0f172a !important; }
:root[data-pm-theme="light"] .gr-button.secondary:hover { background: #cbd5e1 !important; }
:root[data-pm-theme="light"] .gr-input-label, :root[data-pm-theme="light"] .gr-check-radio { color: var(--pm-text-1) !important; }
:root[data-pm-theme="light"] .markdown { color: var(--pm-text-2) !important; }

/* ── Light mode: Custom metrics panel overrides ── */
/* Gradio wraps gr.HTML in .prose which sets color via --body-text-color */
/* Must use .prose ancestor + #id for specificity to win over Svelte scoped styles */
:root[data-pm-theme="light"] .prose #pm-metrics-root,
:root[data-pm-theme="light"] .prose #pm-metrics-root div,
:root[data-pm-theme="light"] .prose #pm-metrics-root span,
:root[data-pm-theme="light"] .prose #pm-metrics-root strong,
:root[data-pm-theme="light"] .prose #pm-metrics-root td,
:root[data-pm-theme="light"] .prose #pm-metrics-root th,
:root[data-pm-theme="light"] .prose #pm-metrics-root p,
:root[data-pm-theme="light"] .prose #pm-metrics-root table { color: #1e293b !important; }
:root[data-pm-theme="light"] .prose #pm-metrics-root .label,
:root[data-pm-theme="light"] .prose #pm-metrics-root .pm-gauge-sublabel,
:root[data-pm-theme="light"] .prose #pm-metrics-root .pm-status-detail { color: #475569 !important; }
:root[data-pm-theme="light"] .prose #pm-metrics-root .pm-section-title { color: #4f46e5 !important; }

/* Light mode: tables */
:root[data-pm-theme="light"] .pm-table { color: #1e293b !important; }
:root[data-pm-theme="light"] .pm-table td { color: #1e293b !important; border-color: rgba(0, 0, 0, 0.08) !important; }
:root[data-pm-theme="light"] .pm-table strong { color: #0f172a !important; }

/* Light mode: badges */
:root[data-pm-theme="light"] .badge-ok { background: rgba(34, 197, 94, 0.15) !important; color: #15803d !important; border-color: rgba(34, 197, 94, 0.4) !important; }
:root[data-pm-theme="light"] .badge-warn { background: rgba(245, 158, 11, 0.15) !important; color: #b45309 !important; border-color: rgba(245, 158, 11, 0.4) !important; }
:root[data-pm-theme="light"] .badge-crit { background: rgba(239, 68, 68, 0.15) !important; color: #dc2626 !important; border-color: rgba(239, 68, 68, 0.4) !important; }
:root[data-pm-theme="light"] .badge-nd { background: rgba(100, 116, 139, 0.1) !important; color: #475569 !important; border-color: rgba(100, 116, 139, 0.3) !important; }

/* Light mode: sparkline */
:root[data-pm-theme="light"] .pm-sparkline-wrap { background: rgba(241, 245, 249, 0.9) !important; }
:root[data-pm-theme="light"] #pm-sparkline-svg text { fill: #475569 !important; }

/* Light mode: header */
:root[data-pm-theme="light"] .pm-header {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(6, 182, 212, 0.05)) !important;
    border-color: rgba(99, 102, 241, 0.2) !important;
}
:root[data-pm-theme="light"] .pm-header h1 { color: #0f172a !important; }
:root[data-pm-theme="light"] .pm-header p { color: #334155 !important; }
:root[data-pm-theme="light"] .pm-header .brand-line { color: #475569 !important; }

/* Light mode: accordion & buttons */
:root[data-pm-theme="light"] .gr-accordion { background: rgba(255, 255, 255, 0.9) !important; border-color: rgba(99, 102, 241, 0.2) !important; }
:root[data-pm-theme="light"] .gr-accordion > button { color: #1e293b !important; }
:root[data-pm-theme="light"] .gr-dropdown select { background: #ffffff !important; color: #0f172a !important; border-color: #cbd5e1 !important; }
:root[data-pm-theme="light"] .gr-button-secondary { background: #e2e8f0 !important; color: #1e293b !important; border-color: #cbd5e1 !important; }

/* Light mode: video container — prevent flicker on theme switch */
:root[data-pm-theme="light"] .pm-leftcol .block { background: var(--pm-surface) !important; border-color: var(--pm-border) !important; color: var(--pm-text-1) !important; }
:root[data-pm-theme="light"] .pm-leftcol .block video,
:root[data-pm-theme="light"] .pm-leftcol .block img,
:root[data-pm-theme="light"] .pm-leftcol .block .image-container,
:root[data-pm-theme="light"] .pm-leftcol .block .upload-container { transition: none !important; }

/* Light mode: session panel */
:root[data-pm-theme="light"] .gr-file { background: rgba(241, 245, 249, 0.9) !important; border-color: #cbd5e1 !important; color: #1e293b !important; }
:root[data-pm-theme="light"] .gr-markdown { color: #334155 !important; }


/* ── Light mode: Forzar reset en contenedores nativos ── */
:root[data-pm-theme="light"] .pm-leftcol .block,
:root[data-pm-theme="light"] .pm-leftcol .gr-block,
:root[data-pm-theme="light"] .pm-leftcol .gr-box,
:root[data-pm-theme="light"] .pm-leftcol .gr-panel,
:root[data-pm-theme="light"] .gr-accordion,
:root[data-pm-theme="light"] .gr-dropdown {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
}

:root[data-pm-theme="light"] .gr-dropdown select,
:root[data-pm-theme="light"] .gr-accordion > button {
    background: #f8fafc !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

:root[data-pm-theme="light"] .gr-block > label, 
:root[data-pm-theme="light"] .gr-label label, 
:root[data-pm-theme="light"] span.label, 
:root[data-pm-theme="light"] .block-title { 
    color: #4f46e5 !important;
    background: #f1f5f9 !important;
}

/* ── Light mode: Forzar textos legibles ── */
:root[data-pm-theme="light"] .pm-leftcol,
:root[data-pm-theme="light"] .pm-leftcol .markdown, 
:root[data-pm-theme="light"] .pm-leftcol .gr-markdown, 
:root[data-pm-theme="light"] .pm-leftcol .prose, 
:root[data-pm-theme="light"] .pm-leftcol p,
:root[data-pm-theme="light"] .pm-leftcol td,
:root[data-pm-theme="light"] .pm-leftcol th,
:root[data-pm-theme="light"] .pm-leftcol span,
:root[data-pm-theme="light"] .gr-accordion,
:root[data-pm-theme="light"] .gr-accordion .markdown,
:root[data-pm-theme="light"] .gr-accordion p,
:root[data-pm-theme="light"] .gr-accordion td,
:root[data-pm-theme="light"] .gr-accordion th,
:root[data-pm-theme="light"] .gr-form-info,
:root[data-pm-theme="light"] .gr-text-sm,
:root[data-pm-theme="light"] span[data-testid="block-info"] {
    color: #0f172a !important;
}

:root[data-pm-theme="light"] .pm-leftcol .gr-form-info,
:root[data-pm-theme="light"] .pm-leftcol span.text-sm,
:root[data-pm-theme="light"] .gr-input-label span {
    color: #334155 !important;
}

/* ── Dropdown List Light Mode Fix ── */
:root[data-pm-theme="light"] .gr-dropdown-list,
:root[data-pm-theme="light"] ul.options,
:root[data-pm-theme="light"] .options {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}
:root[data-pm-theme="light"] .gr-dropdown-list li,
:root[data-pm-theme="light"] ul.options li,
:root[data-pm-theme="light"] .options li {
    color: #0f172a !important;
}
:root[data-pm-theme="light"] .gr-dropdown-list li:hover,
:root[data-pm-theme="light"] ul.options li:hover,
:root[data-pm-theme="light"] .options li:hover,
:root[data-pm-theme="light"] .gr-dropdown-list li.selected,
:root[data-pm-theme="light"] ul.options li.selected {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}

/* ── Prevent Video Flicker ── */
.pm-leftcol .image-frame,
.pm-leftcol .image-container,
.pm-leftcol img,
.pm-leftcol video,
.pm-leftcol [data-testid="image"] {
    background-color: transparent !important;
    transition: none !important;
    animation: none !important;
}

/* ── Text Alignment ── */
.center-text, .center-text p, .center-text .prose, .center-text .gr-markdown {
    text-align: center !important;
}

/* ── Gradio Settings Dialog Light Mode ── */
:root[data-pm-theme="light"] dialog[open] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.12) !important;
    color: #0f172a !important;
}
:root[data-pm-theme="light"] dialog[open] h2,
:root[data-pm-theme="light"] dialog[open] h3,
:root[data-pm-theme="light"] dialog[open] label,
:root[data-pm-theme="light"] dialog[open] .block-title {
    color: #0f172a !important;
}
:root[data-pm-theme="light"] dialog[open] select,
:root[data-pm-theme="light"] dialog[open] input,
:root[data-pm-theme="light"] dialog[open] textarea {
    background: #f8fafc !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
:root[data-pm-theme="light"] dialog[open] p,
:root[data-pm-theme="light"] dialog[open] .text-sm,
:root[data-pm-theme="light"] dialog[open] .gr-text-sm,
:root[data-pm-theme="light"] dialog[open] .description {
    color: #475569 !important;
}
:root[data-pm-theme="light"] dialog[open] button:not(.gr-button-lg) {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
}
:root[data-pm-theme="light"] dialog[open] button:hover {
    background: #e2e8f0 !important;
}
:root[data-pm-theme="light"] dialog[open] input[type="checkbox"] {
    accent-color: #6366f1 !important;
}
:root[data-pm-theme="light"] dialog[open] .close-button,
:root[data-pm-theme="light"] dialog[open] button[aria-label="Close"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
}
:root[data-pm-theme="light"] dialog[open] .close-button:hover,
:root[data-pm-theme="light"] dialog[open] button[aria-label="Close"]:hover {
    color: #0f172a !important;
    background: #f1f5f9 !important;
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


def _do_export() -> tuple[object, str]:
    """Exporta CSV y retorna el archivo."""
    path, msg = _export_csv_file()
    if path:
        return gr.update(value=path, visible=True), msg
    return gr.update(visible=False), msg


# ── Threshold helpers ────────────────────────────────────────────────────────
def _build_threshold_table(leve: float = 35, critico: float = 50, lang: str = "es") -> str:
    """Genera HTML de la tabla de umbrales CPI con valores actuales."""
    t = LANGS.get(lang, LANGS["es"])
    return f"""<table class="pm-table">
    <tr><th>CPI</th><th>{t['thresh_col_status']}</th><th>{t['thresh_col_meaning']}</th></tr>
    <tr><td>CPI ≤ {leve:.0f}</td><td><span class="pm-badge badge-ok">{t['js_thresh_ok']}</span></td><td>{t['js_thresh_ok_d']}</td></tr>
    <tr><td>{leve:.0f} &lt; CPI ≤ {critico:.0f}</td><td><span class="pm-badge badge-warn">{t['js_thresh_warn']}</span></td><td>{t['js_thresh_warn_d']}</td></tr>
    <tr><td>CPI &gt; {critico:.0f}</td><td><span class="pm-badge badge-crit">{t['js_thresh_crit']}</span></td><td>{t['js_thresh_crit_d']}</td></tr>
</table>"""


def _update_thresholds(leve: float, critico: float, lang: str = "es") -> tuple[str, str]:
    """Actualiza umbrales CPI en el analizador. Retorna (tabla_html, mensaje)."""
    t = LANGS.get(lang, LANGS["es"])
    if leve >= critico:
        return (
            _build_threshold_table(state.analyzer.CPI_LEVE, state.analyzer.CPI_CRITICO, lang),
            t["thresh_err"].format(leve=leve, crit=critico)
        )
    state.analyzer.CPI_LEVE = float(leve)
    state.analyzer.CPI_CRITICO = float(critico)
    return (
        _build_threshold_table(leve, critico, lang),
        t["thresh_ok"].format(leve=leve, crit=critico)
    )


def _build_metrics_js() -> str:
    """Genera METRICS_JS con el dict I18N de los 3 idiomas embebido."""
    # Serializar los 3 dicts de strings JS
    import json as _json
    i18n_js = {lang: {k: v for k, v in d.items() if k.startswith("js_")}
               for lang, d in LANGS.items()}
    i18n_json = _json.dumps(i18n_js, ensure_ascii=False)
    return f"""
() => {{
  var CIRC = 326.73;
  var prevCode = '';
  var alertTimer = null;
  var I18N = {i18n_json};

  function getLang() {{
    var el = document.getElementById('pm-lang-code');
    return (el ? (el.textContent || el.innerText || '').trim() : '') || 'es';
  }}

  function t(key) {{
    var lang = getLang();
    var d = I18N[lang] || I18N['es'];
    return d[key] || (I18N['es'][key] || key);
  }}

  function animateValue(el, newText) {{
    if (!el || el.textContent === newText) return;
    el.style.transition = 'opacity 0.15s ease';
    el.style.opacity = '0.2';
    setTimeout(function() {{
      el.textContent = newText;
      el.style.opacity = '1';
    }}, 150);
  }}

  function drawSparkline(history) {{
    if (!history || history.length < 2) return;
    var W = 280, H = 64, MAX = 100;
    var n = history.length;
    var pts = history.map(function(v, i) {{
      return [i / (n - 1) * W, H - (Math.min(Math.max(v, 0), MAX) / MAX) * (H - 4) - 2];
    }});
    var d = pts.map(function(p, i) {{
      return (i === 0 ? 'M' : 'L') + p[0].toFixed(1) + ',' + p[1].toFixed(1);
    }}).join(' ');
    var lineEl = document.getElementById('spark-line');
    if (lineEl) lineEl.setAttribute('d', d);
    var area = d + ' L' + pts[pts.length-1][0].toFixed(1) + ',' + H + ' L0,' + H + ' Z';
    var areaEl = document.getElementById('spark-area');
    if (areaEl) areaEl.setAttribute('d', area);
    var last = pts[pts.length - 1];
    var dotEl = document.getElementById('spark-dot');
    if (dotEl) {{ dotEl.setAttribute('cx', last[0].toFixed(1)); dotEl.setAttribute('cy', last[1].toFixed(1)); }}
  }}

  function updateMetrics(data) {{
    var cpi     = data.cpi        !== undefined ? data.cpi        : 0;
    var code    = data.status_code || 'nd';
    var badTime = data.bad_time   !== undefined ? data.bad_time   : 0;
    var lumbar  = data.lumbar     !== undefined ? data.lumbar     : 0;
    var curv    = data.curv       !== undefined ? data.curv       : 0;
    var fps     = data.fps        !== undefined ? data.fps        : 0;
    var conf    = data.conf       !== undefined ? data.conf       : 0;
    var alert   = data.alert      || false;
    var color   = data.color      || '#94a3b8';
    var history = data.history    || [];

    // Gauge arc
    var pct = Math.min(Math.max(cpi, 0), 100) / 100;
    var offset = CIRC - CIRC * pct;
    var arc = document.getElementById('pm-gauge-arc');
    if (arc) {{ arc.style.strokeDashoffset = offset; arc.style.stroke = color; }}

    // Gauge number
    var num = document.getElementById('pm-gauge-num');
    if (num) {{ animateValue(num, cpi.toFixed(1)); num.style.color = color; }}

    // Badge
    var badgeEl = document.getElementById('pm-badge');
    var badgeCls = {{ ok: 'badge-ok', warn: 'badge-warn', crit: 'badge-crit', nd: 'badge-nd', ni: 'badge-nd' }};
    var badgeKey = {{ ok: 'js_badge_ok', warn: 'js_badge_warn', crit: 'js_badge_crit', nd: 'js_badge_nd', ni: 'js_badge_ni' }};
    if (badgeEl) {{
      var cls = badgeCls[code] || 'badge-nd';
      var label = t(badgeKey[code] || 'js_badge_nd');
      if (badgeEl.textContent !== label) {{ badgeEl.className = 'pm-badge ' + cls; badgeEl.textContent = label; }}
    }}

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
    if (confBar) {{ confBar.style.width = confPct + '%'; confBar.style.background = confColor; confBar.style.transition = 'width 0.3s ease, background 0.3s ease'; }}
    if (confVal) {{ animateValue(confVal, confPct + '%'); confVal.style.color = confColor; }}
    if (confBadge) {{
      confBadge.style.visibility = conf < 0.4 ? 'visible' : 'hidden';
      confBadge.textContent = t('js_weak_det');
    }}

    // Status card — solo cambiar si el code cambió
    var card = document.getElementById('pm-status-card');
    var iconEl = document.getElementById('pm-status-icon');
    var detailEl = document.getElementById('pm-status-detail');
    if (card && prevCode !== code) {{
      var clsMap = {{ ok: 'pm-status pm-status-ok', warn: 'pm-status pm-status-warn', crit: 'pm-status pm-status-crit', nd: 'pm-status pm-status-nd', ni: 'pm-status pm-status-nd' }};
      card.className = (clsMap[code] || 'pm-status pm-status-nd') + (alert ? ' pulse' : '');
      prevCode = code;
    }}
    if (iconEl) {{
      var iconKey = {{ ok: 'js_ok', warn: 'js_warn', crit: 'js_crit', nd: 'js_nd', ni: 'js_ni' }};
      iconEl.textContent = t(iconKey[code] || 'js_nd');
    }}
    if (detailEl) {{
      if (code === 'ok')   detailEl.textContent = t('js_detail_ok');
      else if (code === 'warn') detailEl.textContent = t('js_detail_warn').replace('{{t}}', badTime.toFixed(0));
      else if (code === 'crit') detailEl.textContent = t('js_detail_crit').replace('{{t}}', badTime.toFixed(0));
      else detailEl.textContent = t('js_detail_nd');
    }}

    // Alert popup
    var popup = document.getElementById('pm-alert-popup');
    if (popup && alert) {{
      var titleEl = document.getElementById('pm-alert-title');
      if (titleEl) titleEl.textContent = t('js_alert_title').replace('{{t}}', badTime.toFixed(0));
      popup.style.display = 'block'; popup.style.opacity = '1';
      clearTimeout(alertTimer);
      alertTimer = setTimeout(function() {{
        popup.style.opacity = '0';
        setTimeout(function() {{ popup.style.display = 'none'; }}, 400);
      }}, 4000);
    }}

    // Sparkline
    drawSparkline(history);
  }}

  // Polling loop — lee el div carrier cada 100ms
  var __lightCssInjected = false;
  setInterval(function() {{
    // Inject light mode CSS once (must be done after Gradio finishes DOM setup)
    if (!__lightCssInjected) {{
      var s = document.createElement('style');
      s.textContent = ':root[data-pm-theme="light"] #pm-metrics-root,' +
        ':root[data-pm-theme="light"] #pm-metrics-root div,' +
        ':root[data-pm-theme="light"] #pm-metrics-root span,' +
        ':root[data-pm-theme="light"] #pm-metrics-root strong,' +
        ':root[data-pm-theme="light"] #pm-metrics-root td,' +
        ':root[data-pm-theme="light"] #pm-metrics-root th,' +
        ':root[data-pm-theme="light"] #pm-metrics-root p,' +
        ':root[data-pm-theme="light"] #pm-metrics-root table {{ color: #1e293b !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .label,' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-gauge-sublabel,' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-status-detail {{ color: #475569 !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-section-title {{ color: #4f46e5 !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-conf-track {{ background: rgba(0,0,0,0.06) !important; border-color: rgba(0,0,0,0.08) !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-card {{ background: rgba(255,255,255,0.92) !important; border-color: rgba(99,102,241,0.25) !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-metric-item {{ background: rgba(241,245,249,0.9) !important; border-color: rgba(99,102,241,0.2) !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-status-nd {{ background: rgba(148,163,184,0.1) !important; border-color: rgba(100,116,139,0.3) !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-status-ok {{ background: rgba(34,197,94,0.1) !important; border-color: rgba(34,197,94,0.4) !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-status-warn {{ background: rgba(245,158,11,0.1) !important; border-color: rgba(245,158,11,0.4) !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .pm-status-crit {{ background: rgba(239,68,68,0.1) !important; border-color: rgba(239,68,68,0.4) !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .badge-ok {{ background: rgba(34,197,94,0.15) !important; color: #15803d !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .badge-warn {{ background: rgba(245,158,11,0.15) !important; color: #b45309 !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .badge-crit {{ background: rgba(239,68,68,0.15) !important; color: #dc2626 !important; }}' +
        ':root[data-pm-theme="light"] #pm-metrics-root .badge-nd {{ background: rgba(100,116,139,0.1) !important; color: #475569 !important; }}';
      document.body.appendChild(s);
      __lightCssInjected = true;
    }}
    var el = document.getElementById('pm-metrics-data-inner');
    if (!el) return;
    var raw = (el.textContent || el.innerText || '').trim();
    if (!raw || raw === '{{}}') return;
    try {{ updateMetrics(JSON.parse(raw)); }} catch(e) {{}}
  }}, 100);
}}
"""

METRICS_JS = _build_metrics_js()


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


def _build_static_metrics_panel(lang: str = "es") -> str:
    """Panel de métricas estático — se renderiza UNA VEZ por cambio de idioma.
    Incluye el carrier pm-lang-code que el JS lee para traducir en tiempo real."""
    t = LANGS.get(lang, LANGS["es"])
    return f"""
<style>
  #pm-metrics-root {{ font-family: 'Inter', sans-serif; color: #e2e8f0; }}
  .pm-card {{ background: rgba(15,23,42,0.7); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 16px; margin-bottom: 10px; }}
  .pm-section-title {{ font-size: 10px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #6366f1; margin-bottom: 10px; }}
  .pm-gauge-wrap {{ position: relative; width: 140px; height: 140px; margin: 0 auto 8px; }}
  .pm-gauge-track {{ fill: none; stroke: rgba(99,102,241,0.15); stroke-width: 10; }}
  .pm-gauge-fill  {{ fill: none; stroke-width: 10; stroke-linecap: round; transform: rotate(-90deg); transform-origin: 50% 50%; stroke-dasharray: 326.73; stroke-dashoffset: 326.73; transition: stroke-dashoffset 0.5s ease, stroke 0.4s ease; }}
  .pm-gauge-value {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 28px; font-weight: 800; transition: color 0.4s ease; }}
  .pm-gauge-label {{ text-align: center; font-size: 10px; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; }}
  .pm-badge {{ display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }}
  .badge-ok   {{ background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }}
  .badge-warn {{ background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }}
  .badge-crit {{ background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }}
  .badge-nd   {{ background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }}
  .pm-metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }}
  .pm-metric-item {{ background: rgba(99,102,241,0.06); border-radius: 8px; padding: 8px 10px; }}
  .pm-metric-item .label {{ font-size: 9px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }}
  .pm-metric-item .value {{ font-size: 18px; font-weight: 700; color: #e2e8f0; }}
  .pm-conf-track {{ background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; overflow: hidden; margin: 6px 0 4px; }}
  .pm-conf-fill  {{ height: 100%; border-radius: 4px; width: 0%; transition: width 0.3s ease, background 0.3s ease; }}
  .pm-status {{ border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid transparent; transition: background 0.4s ease, border-color 0.4s ease; }}
  .pm-status-nd   {{ background: rgba(148,163,184,0.08); border-color: rgba(148,163,184,0.2); }}
  .pm-status-ok   {{ background: rgba(34,197,94,0.08);   border-color: rgba(34,197,94,0.3); }}
  .pm-status-warn {{ background: rgba(245,158,11,0.08);  border-color: rgba(245,158,11,0.3); }}
  .pm-status-crit {{ background: rgba(239,68,68,0.08);   border-color: rgba(239,68,68,0.3); }}
  .pm-status.pulse {{ animation: pm-pulse 1.8s ease-in-out infinite; }}
  @keyframes pm-pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(239,68,68,0); }} 50% {{ box-shadow: 0 0 0 8px rgba(239,68,68,0.2); }} }}
  .pm-status-icon   {{ font-size: 13px; font-weight: 700; }}
  .pm-status-detail {{ font-size: 11px; color: #94a3b8; margin-top: 4px; }}
  #pm-sparkline-svg {{ display: block; width: 100%; }}
  #pm-alert-popup {{ display: none; opacity: 0; position: fixed; bottom: 24px; right: 24px; z-index: 9999; background: rgba(239,68,68,0.95); color: #fff; border-radius: 10px; padding: 12px 18px; font-size: 13px; font-weight: 700; box-shadow: 0 8px 32px rgba(239,68,68,0.4); transition: opacity 0.3s ease; }}
</style>

<!-- Carrier de idioma: el JS lee este div para saber qué idioma mostrar -->
<div id="pm-lang-code" style="display:none">{lang}</div>

<div id="pm-metrics-root">

  <!-- ── GAUGE + ESTADO ── -->
  <div class="pm-card">
    <div class="pm-section-title">{t['js_cpi_title']}</div>
    <div class="pm-gauge-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle class="pm-gauge-track" cx="70" cy="70" r="52"/>
        <circle class="pm-gauge-fill" id="pm-gauge-arc" cx="70" cy="70" r="52" stroke="#94a3b8"/>
      </svg>
      <div class="pm-gauge-value" id="pm-gauge-num" style="color:#94a3b8">0.0</div>
    </div>
    <div class="pm-gauge-label">
      <span class="pm-badge badge-nd" id="pm-badge">{t['js_badge_ni']}</span>
    </div>

    <div class="pm-metrics-grid" style="margin-top:12px">
      <div class="pm-metric-item">
        <div class="label">{t['js_lumbar']}</div>
        <div class="value" id="pm-lumbar">0°</div>
      </div>
      <div class="pm-metric-item">
        <div class="label">{t['js_curv']}</div>
        <div class="value" id="pm-curv">0.0%</div>
      </div>
      <div class="pm-metric-item">
        <div class="label">{t['js_bad_time']}</div>
        <div class="value" id="pm-bad-time">0s</div>
      </div>
      <div class="pm-metric-item">
        <div class="label">{t['js_fps']}</div>
        <div class="value" id="pm-fps-val" style="color:#6366f1">0</div>
      </div>
    </div>
  </div>

  <!-- ── ESTADO POSTURAL ── -->
  <div class="pm-status pm-status-nd" id="pm-status-card">
    <div class="pm-status-icon" id="pm-status-icon">{t['js_ni']}</div>
    <div class="pm-status-detail" id="pm-status-detail">{t['js_detail_nd']}</div>
  </div>

  <!-- ── CONFIANZA ── -->
  <div class="pm-card" style="min-height:86px">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span class="pm-section-title" style="margin-bottom:0">{t['js_conf_title']}</span>
      <span id="pm-conf-val" style="font-size:12px;font-weight:700;color:#94a3b8;font-family:'JetBrains Mono',monospace">0%</span>
    </div>
    <div class="pm-conf-track">
      <div class="pm-conf-fill" id="pm-conf-bar"></div>
    </div>
    <div id="pm-conf-badge-slot" style="min-height:16px;margin-top:6px">
      <span id="pm-conf-badge" style="visibility:hidden;font-size:10px;font-weight:700;color:#ef4444;letter-spacing:0.3px">{t['js_weak_det']}</span>
    </div>
  </div>

  <!-- ── SPARKLINE ── -->
  <div class="pm-card">
    <div class="pm-section-title">{t['js_spark_title']}</div>
    <svg id="pm-sparkline-svg" height="56" viewBox="0 0 280 56" preserveAspectRatio="none">
      <rect x="0" y="0" width="280" height="56" fill="rgba(99,102,241,0.03)" rx="4"/>
      <path id="spark-area" fill="rgba(99,102,241,0.12)" d=""/>
      <path id="spark-line" fill="none" stroke="#6366f1" stroke-width="1.5" stroke-linejoin="round" d=""/>
      <circle id="spark-dot" r="3.5" fill="#6366f1" cx="280" cy="28"/>
    </svg>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:#475569;margin-top:3px">
      <span>{t['js_spark_ago']}</span><span>{t['js_spark_now']}</span>
    </div>
  </div>

</div>

<div id="pm-alert-popup">
  <div id="pm-alert-title">{t['js_alert_title'].replace('{{t}}', '0')}</div>
</div>
"""


def _build_keypoints_table_html(lang: str = "es") -> str:
    """Genera tabla HTML de referencia de keypoints traducida."""
    t = LANGS.get(lang, LANGS["es"])
    locs = t["kp_locations"]
    kp_names = ["Head-back", "Neck-back", "Shoulder-top", "Back-backedge",
                "Hips-backedge", "Neck-middle", "Jaw", "Chin", "Shoulder-back"]
    rows = "".join(
        f'<tr><td style="padding:3px 6px"><b>K{i}</b></td>'
        f'<td>{kp_names[i]}</td>'
        f'<td>{locs[i]}</td></tr>'
        for i in range(9)
    )
    return f"""<table style="width:100%;font-size:11px;border-collapse:collapse;color:#cbd5e1">
  <tr style="color:#6366f1">
    <th style="padding:4px 6px;text-align:left">ID</th>
    <th style="padding:4px 6px;text-align:left">{t['kp_col_name']}</th>
    <th style="padding:4px 6px;text-align:left">{t['kp_col_loc']}</th>
  </tr>
  {rows}
</table>"""


def _build_header_html(lang: str = "es") -> str:
    """Genera el HTML del header traducido."""
    t = LANGS.get(lang, LANGS["es"])
    return f"""
    <div class="pm-header">
        <h1>{t['title']}</h1>
        <p>{t['subtitle']}</p>
        <span class="brand-line">
            <span class="pm-live-dot"></span>
            {t['brand']}
        </span>
        <span class="brand-line" style="font-size:0.75rem; opacity:0.8; margin-top:2px;">
            {_GPU_STATUS}
        </span>
    </div>
    """


# ── Idioma activo (módulo-level, leído por callbacks de sesión) ───────────────
_current_lang: str = DEFAULT_LANG


def _on_lang_change(lang: str, leve: float, critico: float, is_active: bool) -> tuple:
    """Reconstruye todos los componentes traducibles al cambiar idioma."""
    global _current_lang
    _current_lang = lang
    t = LANGS.get(lang, LANGS["es"])
    btn_label = t["btn_stop"] if is_active else t["btn_start"]
    session_msg = t["session_active"] if is_active else t["session_idle"]
    return (
        gr.update(value=_build_header_html(lang)),                    # header_html
        gr.update(value=_build_static_metrics_panel(lang)),           # metrics_panel
        gr.update(value=_build_threshold_table(leve, critico, lang)), # threshold_table
        gr.update(label=t["thresh_leve"]),                            # leve_slider
        gr.update(label=t["thresh_crit"]),                            # critico_slider
        gr.update(value=btn_label),                                   # session_btn
        gr.update(value=session_msg),                                 # session_status
        gr.update(value=t["thresh_hint"]),                            # threshold_msg
        gr.update(label=t["export_file"]),                            # export_file
        gr.update(value=t["export_btn"]),                             # export_btn
        gr.update(label=t["model_label"]),                            # model_dropdown
        gr.update(label=t["webcam_label"]),                           # webcam
        gr.update(value=_build_keypoints_table_html(lang)),           # keypoints_table
        gr.update(label=t["calib_title"]),                            # calib_accordion
        gr.update(label=t["kp_title"]),                               # kp_accordion
        gr.update(label=t["session_title"]),                          # session_accordion
        gr.update(value=t["model_info_def"]),                         # model_info
    )


def _toggle_session(is_active: bool) -> tuple[bool, str, str, object, object]:
    """Toggle start/stop sesión usando gr.State (no el label del botón)."""
    t = LANGS.get(_current_lang, LANGS["es"])
    if is_active:
        # Detener
        state.session_active = False
        summary = _compute_summary(state.session_data)
        summary_html = _build_summary_html(summary)
        n = len(state.session_data)
        msg = t["session_done"].format(n=n)
        return (
            False,
            t["btn_start"],
            msg,
            gr.update(visible=bool(summary_html), value=summary_html if summary_html else ""),
            gr.update(visible=True),   # siempre mostrar export_btn al detener
        )
    else:
        # Iniciar
        state.session_data = []
        state.session_frame_counter = 0
        state.session_active = True
        state.session_start_time = time.time()
        return True, t["btn_stop"], t["session_active"], gr.update(visible=False, value=""), gr.update(visible=False)


def build_ui() -> gr.Blocks:
    """Construye la interfaz Gradio completa."""
    t0 = LANGS[DEFAULT_LANG]
    theme_js = """
    (function(){
        // ── Suprimir transiciones con CSS injectado ──────────────────────────
        // Un <style> cubre TODOS los elementos, incluso los que Gradio crea después
        var __suppressStyle = null;

        function suppressAllTransitions() {
            if (__suppressStyle && __suppressStyle.parentNode) return;
            __suppressStyle = document.createElement('style');
            __suppressStyle.textContent = '*, *::before, *::after { transition: none !important; animation: none !important; }';
            document.head.appendChild(__suppressStyle);
        }

        function restoreTransitions() {
            if (__suppressStyle && __suppressStyle.parentNode) {
                __suppressStyle.remove();
                __suppressStyle = null;
            }
        }

        // ── Inicialización del theme ─────────────────────────────────────────
        suppressAllTransitions();
        var saved = localStorage.getItem('pm-theme') || 'dark';
        document.documentElement.setAttribute('data-pm-theme', saved);

        function applyTextColors(theme) {
            var root = document.getElementById('pm-metrics-root');
            if (!root) return;
            var isDark = theme === 'dark';
            var color = isDark ? '#f1f5f9' : '#1e293b';
            var muted = isDark ? '#94a3b8' : '#475569';
            var accent = isDark ? '#818cf8' : '#4f46e5';
            var els = root.querySelectorAll('div, span, strong, td, th, p, table');
            for (var i = 0; i < els.length; i++) {
                els[i].style.setProperty('color', color, 'important');
            }
            var labels = root.querySelectorAll('.label, .pm-gauge-sublabel, .pm-status-detail');
            for (var j = 0; j < labels.length; j++) {
                labels[j].style.setProperty('color', muted, 'important');
            }
            var titles = root.querySelectorAll('.pm-section-title');
            for (var k = 0; k < titles.length; k++) {
                titles[k].style.setProperty('color', accent, 'important');
            }
            root.style.setProperty('color', color, 'important');
        }

        // ── Apply on load after Gradio renders ───────────────────────────────
        function tryApply() {
            var theme = localStorage.getItem('pm-theme') || 'dark';
            applyTextColors(theme);
            restoreTransitions();
        }
        // Primero restauramos transiciones después del primer repaint
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                restoreTransitions();
            });
        });
        setTimeout(tryApply, 500);
        setTimeout(tryApply, 1500);
        setTimeout(tryApply, 3000);

        window.__pmToggleTheme = function() {
            var cur = document.documentElement.getAttribute('data-pm-theme') || 'dark';
            var next = cur === 'dark' ? 'light' : 'dark';
            suppressAllTransitions();
            document.documentElement.setAttribute('data-pm-theme', next);
            localStorage.setItem('pm-theme', next);
            var btn = document.getElementById('pm-theme-toggle');
            if (btn) btn.textContent = next === 'dark' ? '\\u2600\\uFE0F' : '\\uD83C\\uDF19';
            applyTextColors(next);
            requestAnimationFrame(function() {
                requestAnimationFrame(function() {
                    restoreTransitions();
                });
            });
        };
        // ── Gear popup: toggle + lang ──
        window.__pmTogglePopup = function() {
            var p = document.getElementById('pm-gear-popup');
            if (!p) return;
            p.style.display = p.style.display === 'flex' ? 'none' : 'flex';
        };
        window.__pmSetLang = function(lang) {
            var sel = document.querySelector('.pm-lang-hidden select');
            if (!sel) return;
            sel.value = lang;
            sel.dispatchEvent(new Event('input', { bubbles: true }));
            // Close popup
            document.getElementById('pm-gear-popup').style.display = 'none';
        };
        // Close popup on click outside
        document.addEventListener('click', function(e) {
            var w = document.getElementById('pm-gear-wrap');
            var p = document.getElementById('pm-gear-popup');
            if (p && p.style.display === 'flex' && w && !w.contains(e.target)) {
                p.style.display = 'none';
            }
        });
    })();
    """
    # Light override CSS is injected via JS above (Gradio strips <style> from head param)
    head_script = f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap"><script>{theme_js}({METRICS_JS})();</script>'
    with gr.Blocks(
        title="Monitoreo Postural — USCO 2026",
    ) as app:
        session_state = gr.State(False)

        header_html = gr.HTML(_build_header_html(DEFAULT_LANG))

        # ── Floating gear popup (bottom-right) ──
        with gr.Row(elem_classes=["gear-root"]):
            gear_icon = gr.HTML('''
<div id="pm-gear-wrap">
    <span id="pm-gear-icon" onclick="__pmTogglePopup()">⚙️</span>
    <div id="pm-gear-popup">
        <button class="pm-gear-lang" onclick="__pmSetLang('es')">🇪🇸 ES</button>
        <button class="pm-gear-lang" onclick="__pmSetLang('en')">🇬🇧 EN</button>
        <button class="pm-gear-lang" onclick="__pmSetLang('pt')">🇧🇷 PT</button>
        <div class="pm-gear-divider"></div>
        <button id="pm-theme-toggle" onclick="window.__pmToggleTheme()">☀️</button>
    </div>
</div>
            ''')
            lang_dropdown = gr.Dropdown(
                choices=[("🇪🇸 Español", "es"), ("🇬🇧 English", "en"), ("🇧🇷 Português", "pt")],
                value=DEFAULT_LANG,
                label=t0["lang_label"],
                interactive=True,
                container=True, show_label=False,
                elem_classes=["pm-lang-hidden"],
            )

        with gr.Row():
            # ── IZQUIERDA: Video + historial + calibración + referencia ──
            with gr.Column(scale=2, elem_classes=["pm-leftcol"]):
                webcam = gr.Image(
                    sources=["webcam"],
                    label=t0["webcam_label"],
                    streaming=True,
                )

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=[c["name"] for c in MODEL_CONFIGS],
                        value=MODEL_CONFIGS[0]["name"],
                        label=t0["model_label"],
                        info=t0["model_info_sel"],
                        interactive=True,
                    )

                model_info = gr.Markdown(t0["model_info_def"], elem_classes=["center-text"])

                calib_accordion = gr.Accordion(t0["calib_title"], open=False)
                with calib_accordion:
                    threshold_table = gr.HTML(_build_threshold_table(lang=DEFAULT_LANG))
                    leve_slider = gr.Slider(
                        minimum=10, maximum=80, value=35, step=1,
                        label=t0["thresh_leve"], interactive=True
                    )
                    critico_slider = gr.Slider(
                        minimum=20, maximum=100, value=50, step=1,
                        label=t0["thresh_crit"], interactive=True
                    )
                    threshold_msg = gr.Markdown(t0["thresh_hint"])

                kp_accordion = gr.Accordion(t0["kp_title"], open=False)
                with kp_accordion:
                    keypoints_table = gr.HTML(_build_keypoints_table_html(DEFAULT_LANG))

            # ── DERECHA: solo métricas vivas + sesión ──
            with gr.Column(scale=1, min_width=340, elem_classes=["pm-sidebar"]):
                metrics_panel = gr.HTML(_build_static_metrics_panel(DEFAULT_LANG))
                metrics_data = gr.HTML(
                    value='<div id="pm-metrics-data-inner" style="display:none">{}</div>',
                    elem_id="pm-metrics-data",
                )

                session_accordion = gr.Accordion(t0["session_title"], open=True)
                with session_accordion:
                    session_btn = gr.Button(t0["btn_start"], variant="primary", size="sm")
                    session_status = gr.Markdown(t0["session_idle"])
                    export_btn = gr.Button(t0["export_btn"], variant="secondary", size="sm", visible=False)
                    export_file = gr.File(label=t0["export_file"], visible=False, interactive=False)
                    export_msg = gr.Markdown("")
                    summary_display = gr.HTML("", visible=False)

        # ── Eventos ──────────────────────────────────────────────────────────
        webcam.stream(
            fn=process_frame,
            inputs=[webcam, model_dropdown],
            outputs=[webcam, metrics_data],
            stream_every=0.05,
            time_limit=None,
        )

        model_dropdown.change(
            fn=lambda m: f"**{LANGS[_current_lang]['model_label']}:** {m}",
            inputs=[model_dropdown],
            outputs=[model_info],
        )

        lang_dropdown.change(
            fn=_on_lang_change,
            inputs=[lang_dropdown, leve_slider, critico_slider, session_state],
            outputs=[
                header_html, metrics_panel, threshold_table, leve_slider, critico_slider,
                session_btn, session_status, threshold_msg, export_file,
                export_btn, model_dropdown, webcam, keypoints_table,
                calib_accordion, kp_accordion, session_accordion, model_info,
            ],
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
            fn=lambda leve, critico: _update_thresholds(leve, critico, _current_lang),
            inputs=[leve_slider, critico_slider],
            outputs=[threshold_table, threshold_msg],
        )
        critico_slider.change(
            fn=lambda leve, critico: _update_thresholds(leve, critico, _current_lang),
            inputs=[leve_slider, critico_slider],
            outputs=[threshold_table, threshold_msg],
        )

        return app, head_script


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

    app, head_script = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        prevent_thread_lock=True,
        css=CSS,
        theme=THEME,
        head=head_script,
    )

    # Mantener el proceso vivo mientras el servidor corre
    import time as _time
    try:
        while True:
            _time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        app.close()
