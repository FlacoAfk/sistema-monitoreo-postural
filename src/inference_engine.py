"""
Component 1 — Motor de Inferencia (inference_engine.py)

Carga modelos YOLO-Pose, captura frames de cámara web en hilo independiente
(pipeline asíncrono), ejecuta inferencia y devuelve coordenadas de keypoints
en formato JSON por frame.

NO toma decisiones clasificatorias — eso es responsabilidad exclusiva del
backend matemático (posture_analyzer.py).

Arquitectura: productor-consumidor con cola thread-safe.

Autor: Sistema de Monitoreo Postural — Universidad Surcolombiana 2026
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Keypoint names — Roboflow mapping (verified 2026-05-08) ──
# | K (YOLO) | Roboflow ID | Nombre anatómico |
# |:--------:|:-----------:|------------------|
# | K0       | 0           | Head-back (Occipital) |
# | K1       | 1           | Neck-back (Cervical C7) ← PIVOTE anatómico |
# | K2       | 2           | Shoulder-top (Acromion) |
# | K3       | 6           | Back-backedge (Espalda media) |
# | K4       | 7           | Hips-backedge (Cadera lumbosacra) |
# | K5       | 10          | Neck-middle (Cervical media) |
# | K6       | 13          | Jaw (Mandíbula) |
# | K7       | 14          | Chin (Mentón) ← VÉRTICE del ángulo mentoniano ⚠ |
# | K8       | 18          | Shoulder-back (Escápula) |
KEYPOINT_NAMES: list[str] = [
    "K0_HeadBack",      # 0: Occipital / Head-back
    "K1_NeckBack",      # 1: Cervical C7 / Neck-back
    "K2_ShoulderTop",   # 2: Acromion / Shoulder-top
    "K3_BackBorde",     # 3: Espalda media / Back-backedge
    "K4_HipsBack",      # 4: Cadera lumbosacra / Hips-backedge
    "K5_NeckMid",       # 5: Cervical media / Neck-middle
    "K6_Jaw",           # 6: Mandíbula / Jaw
    "K7_Chin",          # 7: Mentón / Chin ← VÉRTICE ⚠
    "K8_ShoulderBack",  # 8: Escápula / Shoulder-back
]

# Keypoints críticos para el ángulo cervicodorsal α = ∠(K0-K1-K8) en C7:
# K0 (Head-back / Occipital) → extremo craneal del vector
# K1 (Neck-back / Cervical C7) → VÉRTICE del ángulo ⚠
# K8 (Shoulder-back / Escápula) → extremo dorsal del vector
CRITICAL_KEYPOINT_INDICES: list[int] = [0, 1, 8]

# Conexiones anatómicas — Cadena posterior (espalda)
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),  # Head-back → C7 (columna cervical alta)
    (1, 8),  # C7 → Escápula (columna cervical baja)
    (8, 3),  # Escápula → Espalda media (columna torácica)
    (3, 4),  # Espalda media → Cadera (columna lumbar)
]

# Colores BGR para visualización
COLOR_KEYPOINT = (0, 255, 0)       # Verde
COLOR_SKELETON = (255, 200, 0)     # Cyan/amarillo
COLOR_ANGLE_LINE = (0, 165, 255)   # Naranja (líneas del ángulo)


@dataclass
class KeypointResult:
    """Resultado de inferencia de pose para un frame."""

    timestamp: float
    frame_id: int
    detected: bool
    num_people: int = 0
    # Lista de 9 keypoints [x, y, confidence] por persona detectada
    # Forma: [[x0,y0,c0], [x1,y1,c1], ...]  o [] si no hay detección
    keypoints: list[list[float]] = field(default_factory=list)
    # Frame original (None si no se solicita)
    frame: Optional[np.ndarray] = None

    @property
    def has_valid_pose(self) -> bool:
        """Verifica que los 9 keypoints tengan confianza > 0."""
        return self.detected and len(self.keypoints) == 9 and all(k[2] > 0 for k in self.keypoints)

    def get_kp_coords(self, index: int) -> tuple[float, float, float]:
        """Retorna (x, y, conf) para un keypoint específico (0-8)."""
        if 0 <= index < len(self.keypoints):
            kp = self.keypoints[index]
            return (kp[0], kp[1], kp[2])
        return (0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serializa a diccionario JSON-compatible."""
        return {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "detected": self.detected,
            "num_people": self.num_people,
            "keypoints": [
                {"name": KEYPOINT_NAMES[i], "x": kp[0], "y": kp[1], "confidence": kp[2]}
                for i, kp in enumerate(self.keypoints)
            ] if self.keypoints else [],
        }

    def to_json(self) -> str:
        """Serializa a string JSON."""
        return json.dumps(self.to_dict())


# ── Colores BGR para visualización de keypoints ────────────────────────────
COLORS_BGR: list[tuple[int, int, int]] = [
    (255, 0, 0),     # K0: azul (Head-back / Occipital — crítico, extremo craneal)
    (0, 165, 255),   # K1: naranja (Neck-back / C7 — crítico, pivote)
    (0, 255, 0),     # K2: verde (Shoulder-top / Acromion)
    (0, 200, 200),   # K3: cyan claro (Back-backedge / Espalda media)
    (128, 128, 128), # K4: gris (Hips-backedge / Cadera)
    (200, 200, 0),   # K5: cyan oscuro (Neck-middle / Cervical media)
    (200, 0, 200),   # K6: magenta (Jaw / Mandíbula)
    (0, 0, 255),     # K7: rojo (Chin / Mentón — VÉRTICE del ángulo) ⚠
    (128, 0, 128),   # K8: púrpura (Shoulder-back / Escápula)
]


class InferenceEngine:
    """
    Motor de inferencia YOLO-Pose con pipeline asíncrono.

    Arquitectura productor-consumidor:
        Hilo productor → captura frames de webcam → cola thread-safe
        Hilo consumidor → ejecuta YOLO → cola de resultados

    Atributos:
        model_path: Ruta al archivo .pt del modelo YOLO-Pose.
        camera_id: Índice de la cámara (0 = default).
        confidence_threshold: Umbral mínimo de confianza para keypoints.
        img_size: Tamaño de redimensionamiento para inferencia (None = original).
    """

    def __init__(
        self,
        model_path: str | Path,
        camera_id: int = 0,
        confidence_threshold: float = 0.3,
        img_size: int | None = 640,
        device: str | None = None,
    ) -> None:
        """
        Inicializa el motor de inferencia.

        Args:
            model_path: Ruta al archivo .pt del modelo YOLO-Pose entrenado.
            camera_id: Índice de cámara OpenCV (0 = cámara por defecto).
            confidence_threshold: Confianza mínima para considerar un keypoint válido.
            img_size: Tamaño de imagen para YOLO (None = tamaño nativo).
            device: Dispositivo de inferencia ('cpu', 'cuda:0', etc.). Auto-detecta GPU si es None.
        """
        # Auto-detectar GPU si no se especifica dispositivo
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[INFO] InferenceEngine auto-detectó dispositivo: {device.upper()}")
            if device == "cuda":
                _props = torch.cuda.get_device_properties(0)
                print(f"[INFO] GPU: {_props.name}")
                print(f"[INFO] VRAM: {_props.total_memory / 1024**3:.1f} GB | Compute: {_props.major}.{_props.minor}")
                torch.backends.cudnn.benchmark = True
        self._use_fp16 = (device == "cuda" and torch.cuda.get_device_properties(0).major >= 6)
        self.model_path = Path(model_path)
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        # Cargar modelo YOLO y mover a GPU/CPU
        self.model = YOLO(str(self.model_path))
        self.model.to(device)

        # Configuración de cámara
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_width: int = 640
        self._frame_height: int = 480
        self._fps: float = 30.0

        # Estado y sincronización
        self._running: bool = False
        self._capture_thread: Optional[threading.Thread] = None
        self._inference_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._result_queue: queue.Queue = queue.Queue(maxsize=10)
        self._frame_counter: int = 0
        self._lock = threading.Lock()

        # Callback opcional (se ejecuta en el hilo de inferencia)
        self._on_result: Optional[Callable[[KeypointResult], None]] = None

        # Warmup — primera inferencia siempre es más lenta
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._run_inference(dummy, frame_id=-1)
        if self._use_fp16:
            print(f"[INFO] InferenceEngine: FP16 activado ✓")

    # ── Propiedades ──────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """Indica si el pipeline de captura+inferencia está activo."""
        return self._running

    @property
    def fps(self) -> float:
        """FPS actual de la cámara."""
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        """Dimensiones del frame (width, height)."""
        return (self._frame_width, self._frame_height)

    # ── Pipeline público ─────────────────────────────────────────────────────

    def start(
        self,
        on_result: Optional[Callable[[KeypointResult], None]] = None,
    ) -> None:
        """
        Inicia el pipeline asíncrono: captura de cámara + inferencia YOLO.

        Args:
            on_result: Callback llamado con cada KeypointResult (en hilo de inferencia).
        """
        if self._running:
            return  # Ya está corriendo

        self._on_result = on_result

        # Abrir cámara
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara (ID={self.camera_id})")

        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0

        self._running = True

        # Hilo de captura de frames
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="capture-thread", daemon=True
        )
        self._capture_thread.start()

        # Hilo de inferencia
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name="inference-thread", daemon=True
        )
        self._inference_thread.start()

    def stop(self) -> None:
        """Detiene el pipeline de forma segura."""
        self._running = False

        # Desbloquear colas
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass

        if self._cap:
            self._cap.release()
            self._cap = None

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)

    def get_result(self, timeout: float = 1.0) -> Optional[KeypointResult]:
        """
        Obtiene el siguiente resultado de inferencia (bloqueante).

        Útil para modo síncrono (sin callbacks).

        Args:
            timeout: Tiempo máximo de espera en segundos.

        Returns:
            KeypointResult o None si timeout.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def process_single_frame(self, frame: np.ndarray) -> KeypointResult:
        """
        Procesa un frame individual de forma síncrona (sin pipeline).

        Útil para benchmark y procesamiento por lote.

        Args:
            frame: Imagen en formato BGR (numpy array, shape H×W×3).

        Returns:
            KeypointResult con los keypoints detectados.
        """
        return self._run_inference(frame, frame_id=self._frame_counter)

    def switch_model(self, model_path: str | Path) -> None:
        """
        Cambia el modelo en caliente (thread-safe).

        Args:
            model_path: Ruta al nuevo archivo .pt.
        """
        new_path = Path(model_path)
        if not new_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {new_path}")

        with self._lock:
            # Cerrar modelo anterior y cargar nuevo
            del self.model
            self.model = YOLO(str(new_path))
            self.model_path = new_path

    # ── Hilos internos ───────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Hilo productor: captura frames de webcam continuamente."""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            try:
                self._frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                # Descartar frame si la cola está llena (el consumidor está atrás)
                pass

    def _inference_loop(self) -> None:
        """Hilo consumidor: ejecuta YOLO sobre frames en cola."""
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None:
                break

            self._frame_counter += 1
            result = self._run_inference(frame, frame_id=self._frame_counter)

            try:
                self._result_queue.put(result, timeout=0.1)
            except queue.Full:
                pass

            if self._on_result:
                self._on_result(result)

    def _run_inference(self, frame: np.ndarray, frame_id: int) -> KeypointResult:
        """
        Ejecuta YOLO-Pose sobre un frame y construye KeypointResult.

        NO toma decisiones clasificatorias — solo extrae coordenadas.

        Args:
            frame: Imagen BGR (numpy array).
            frame_id: Identificador secuencial del frame.

        Returns:
            KeypointResult con coordenadas de los 9 keypoints.
        """
        timestamp = time.time()

        # Preprocesar frame si es necesario
        if self.img_size and (frame.shape[0] != self.img_size or frame.shape[1] != self.img_size):
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            # Mantener relación de aspecto: resize manteniendo ratio
            # h, w = frame.shape[:2]
            # scale = self.img_size / max(h, w)
            # frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Ejecutar YOLO — solo predicción, sin clasificación postural
        preds = self.model(frame, verbose=False, half=self._use_fp16)

        if not preds or preds[0].keypoints is None:
            return KeypointResult(
                timestamp=timestamp,
                frame_id=frame_id,
                detected=False,
            )

        kp = preds[0].keypoints
        data = kp.data.cpu().numpy()  # [N_personas, 9_kp, 3_xyz]

        num_people = data.shape[0]
        if num_people == 0:
            return KeypointResult(
                timestamp=timestamp,
                frame_id=frame_id,
                detected=False,
            )

        # Seleccionar la persona con mayor confianza promedio en keypoints
        confidences = data[:, :, 2]  # [N, 9]
        avg_conf = confidences.mean(axis=1)
        best_idx = int(np.argmax(avg_conf))

        # Filtrar keypoints bajo umbral de confianza
        raw_kps = data[best_idx]  # [9, 3]
        keypoints: list[list[float]] = []
        for i in range(min(9, len(raw_kps))):
            x, y, c = raw_kps[i]
            if c < self.confidence_threshold:
                keypoints.append([float(x), float(y), 0.0])  # Marcar como no detectado
            else:
                keypoints.append([float(x), float(y), float(c)])

        return KeypointResult(
            timestamp=timestamp,
            frame_id=frame_id,
            detected=True,
            num_people=num_people,
            keypoints=keypoints,
            frame=frame.copy(),
        )


def draw_pose_overlay(
    frame: np.ndarray,
    result: KeypointResult,
    angle_deg: Optional[float] = None,
    posture_status: str = "CORRECTO",
) -> np.ndarray:
    """
    Dibuja overlay de keypoints, esqueleto y ángulo sobre el frame.

    Args:
        frame: Imagen BGR original.
        result: Resultado de inferencia con keypoints.
        angle_deg: Ángulo cervicodorsal calculado (opcional).
        posture_status: Estado postural ("CORRECTO", "ALERTA LEVE", "ALERTA CRÍTICA").

    Returns:
        Frame con overlay dibujado.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    if not result.detected or not result.keypoints:
        cv2.putText(out, "No detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    # Dibujar esqueleto (conexiones anatómicas)
    for conn in SKELETON_CONNECTIONS:
        i_a, i_b = conn
        if i_a >= len(result.keypoints) or i_b >= len(result.keypoints):
            continue
        kp_a = result.keypoints[i_a]
        kp_b = result.keypoints[i_b]
        if kp_a[2] > 0 and kp_b[2] > 0:  # Ambos detectados
            pt_a = (int(kp_a[0]), int(kp_a[1]))
            pt_b = (int(kp_b[0]), int(kp_b[1]))
            cv2.line(out, pt_a, pt_b, COLOR_SKELETON, 2, cv2.LINE_AA)

    # Dibujar keypoints
    for i, kp in enumerate(result.keypoints):
        if kp[2] <= 0:
            continue  # Keypoint no detectado
        cx, cy = int(kp[0]), int(kp[1])
        color = COLORS_BGR[i] if i < len(COLORS_BGR) else COLOR_KEYPOINT
        cv2.circle(out, (cx, cy), 4, color, -1, cv2.LINE_AA)
        # Etiqueta para K0, K1, K8 (críticos de la cadena posterior)
        if i in (0, 1, 8):
            cv2.putText(out, KEYPOINT_NAMES[i], (cx + 8, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Dibujar líneas del ángulo lumbar ∠K8-K3-K4 + referencia K1→K4
    if angle_deg is not None:
        k8_scapula = result.get_kp_coords(8)
        k3_back    = result.get_kp_coords(3)
        k4_hips    = result.get_kp_coords(4)
        k1_c7      = result.get_kp_coords(1)

        if k8_scapula[2] > 0 and k3_back[2] > 0 and k4_hips[2] > 0:
            p_scap = (int(k8_scapula[0]), int(k8_scapula[1]))
            p_mid  = (int(k3_back[0]), int(k3_back[1]))    # Espalda media (vértice)
            p_hip  = (int(k4_hips[0]), int(k4_hips[1]))

            # Vector K3→K8 (torácico superior)
            cv2.line(out, p_mid, p_scap, COLOR_ANGLE_LINE, 2, cv2.LINE_AA)
            # Vector K3→K4 (lumbar inferior)
            cv2.line(out, p_mid, p_hip, COLOR_ANGLE_LINE, 2, cv2.LINE_AA)

            # Línea de referencia K1→K4 (spine teórico)
            if k1_c7[2] > 0:
                p_c7 = (int(k1_c7[0]), int(k1_c7[1]))
                cv2.line(out, p_c7, p_hip, (180, 180, 180), 1, cv2.LINE_AA)

            # Mostrar ángulo lumbar cerca del vértice (K3)
            cx_angle = p_mid[0] + 15
            cy_angle = p_mid[1] - 10
            cv2.putText(out, f"L={angle_deg:.1f}°", (cx_angle, cy_angle),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ANGLE_LINE, 2, cv2.LINE_AA)

    # Banner de estado
    status_colors: dict[str, tuple[int, int, int]] = {
        "CORRECTO": (0, 255, 0),
        "ALERTA LEVE": (0, 255, 255),
        "ALERTA CRÍTICA": (0, 0, 255),
    }
    banner_color = status_colors.get(posture_status, (128, 128, 128))

    # Barra inferior
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

    cv2.putText(out, f"Estado: {posture_status}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2, cv2.LINE_AA)

    if angle_deg is not None:
        cv2.putText(out, f"Angulo: {angle_deg:.1f}°", (w - 220, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out
