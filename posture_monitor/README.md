# Sistema de Monitoreo Postural en Tiempo Real

**Universidad Surcolombiana — Facultad de Ingeniería**  
Castañeda Guzmán & Idarraga Plazas, 2026

---

## Resumen

Sistema de detección de posturas corporales inadecuadas en trabajadores de oficina,
basado en visión artificial con redes YOLO-Pose y el Combined Posture Index (CPI),
un índice multivectorial que integra curvatura escapular y ángulo lumbar.

### Arquitectura

```
Webcam → YOLO-Pose (9 keypoints) → CPI (5 keypoints posteriores) → Alertas
```

El CPI usa 5 keypoints de la cadena posterior de la espalda (K0, K1, K8, K3, K4)
para calcular simultáneamente la curvatura escapular y el déficit angular lumbar,
proporcionando una separación 12.7× mayor entre posturas que los enfoques monoangulares.

Documentación completa del modelo matemático: [`MODELO_MATEMATICO_CPI.md`](MODELO_MATEMATICO_CPI.md)

---

## Keypoints — Mapeo Roboflow → YOLO

| Índice | Roboflow ID | Nombre anatómico | Descripción |
|:---:|:---:|---|---|
| K0 | 0 | Head-back | Occipital / Parte posterior de la cabeza |
| K1 | 1 | Neck-back | Cervical C7 / Base posterior del cuello |
| K2 | 2 | Shoulder-top | Acromion / Parte superior del hombro |
| K3 | 6 | Back-backedge | Borde posterior dorsal / Espalda media |
| K4 | 7 | Hips-backedge | Borde posterior de cadera / Lumbosacra |
| K5 | 10 | Neck-middle | Cervical media |
| K6 | 13 | Jaw | Mandíbula |
| K7 | 14 | Chin | Mentón |
| K8 | 18 | Shoulder-back | Zona escapular / Escápula posterior |

**Keypoints del CPI (5):** K0, K1, K3, K4, K8

---

## Fórmula CPI

```
CPI = (180° − ∠K8-K3-K4) × 2  +  curvatura_escapular_normalizada × 100

donde:
  curvatura_escapular_normalizada = dist_⊥(K8, línea K1→K4) / |K1→K4|
```

### Clasificación

| CPI | Estado | Significado |
|-----|--------|-------------|
| ≤ 35 | CORRECTO | Columna alineada, postura recta |
| 35–50 | ALERTA LEVE | Curvatura dorsal leve |
| > 50 | ALERTA CRÍTICA | Cifosis / hombros caídos |

---

## Modelos

De 108 submodelos evaluados, 4 seleccionados por score compuesto (mAP50-95 + detección + velocidad):

| Modelo | SCORE | Latencia | Detección |
|--------|-------|----------|-----------|
| YOLOv5n 🎯 | 0.9109 | 30.6ms | 95.2% |
| YOLOv8n 🚀 | 0.9189 | **22.4ms** | 90.5% |
| YOLOv26n ⚖️ | 0.9050 | 27.3ms | 90.5% |
| YOLO11n ⭐ | 0.8990 | 30.7ms | 90.5% |

---

## Instalación

### Requisitos
- Python 3.10+
- Cámara web funcional
- (Opcional) GPU NVIDIA con CUDA

### Pasos

```bash
cd posture_monitor

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Verificar modelos en:
#   ..\Modelos entrenados\yolov8n_pose_b16_lr05\weights\best.pt
#   ..\Modelos entrenados\yolov5n_pose_b16_lr05\weights\best.pt
#   ..\Modelos entrenados\yolov26n_pose_b128_lr05\weights\best.pt
#   ..\Modelos entrenados\yolov11n_pose_b16_lr01\weights\best.pt
```

---

## Uso

### Dashboard en tiempo real

```bash
cd src
python app.py
```
Abre http://127.0.0.1:7860

- **Selector de modelo**: cambio en caliente entre 4 modelos
- **Video en vivo**: overlay con esqueleto posterior + líneas del ángulo lumbar + referencia espinal
- **Panel CPI**: gauge circular + ángulo lumbar + curvatura escapular
- **Alertas**: sonora (>30s en mala postura, beep cada 5s)

### Benchmark

```bash
cd src
python model_benchmark.py
```

### Validación de keypoints

```bash
cd posture_monitor
python validate_keypoints.py
```
Genera overlays y JSON en `keypoint_output/`.

---

## Estructura del Proyecto

```
posture_monitor/
├── src/
│   ├── inference_engine.py    # Motor YOLO-Pose (carga, inferencia, webcam)
│   ├── posture_analyzer.py    # CPI — Combined Posture Index (5 keypoints)
│   ├── app.py                 # Dashboard Gradio (UI tiempo real)
│   ├── model_benchmark.py     # Comparador de modelos
│   └── test_system.py         # Tests del sistema
├── docs/
│   └── INFORME_MODELO_MATEMATICO.md
├── keypoint_output/           # Resultados de validación (overlays + JSON)
├── requirements.txt
├── README.md
└── MODELO_MATEMATICO_CPI.md   # Documentación completa del CPI
```

---

## Componentes

### 1. `inference_engine.py` — Motor de Inferencia
- Carga modelos YOLO-Pose (.pt) con pipeline asíncrono (hilo captura + hilo inferencia)
- Devuelve `KeypointResult` con 9 keypoints (x, y, confianza)
- Esqueleto visual: K0→K1→K8→K3→K4 (cadena posterior)
- NO toma decisiones clasificatorias

### 2. `posture_analyzer.py` — Backend Matemático (CPI)
- Extrae 5 keypoints (K0, K1, K8, K3, K4)
- Calcula ángulo lumbar `∠K8-K3-K4` + curvatura escapular normalizada
- CPI = déficit_lumbar × 2 + curvatura% × 100
- Clasifica: ≤35 CORRECTO, 35–50 ALERTA LEVE, >50 ALERTA CRÍTICA
- Contador de tiempo acumulado + alertas

### 3. `app.py` — Dashboard Interactivo
- Gradio con streaming de webcam
- Overlay: esqueleto azul + líneas naranjas del ángulo lumbar + referencia gris K1→K4
- Gauge CPI con anillo SVG + métricas en tiempo real
- Selector de modelo en caliente

### 4. `model_benchmark.py` — Comparador de Modelos
- Evalúa 4 modelos sobre imágenes de prueba
- JSON con métricas detalladas + gráfica comparativa

---

## Paper de Referencia

Castañeda Guzmán, J. & Idarraga Plazas, L. (2026). *Sistema de Monitoreo
Postural en Tiempo Real Mediante Técnicas de Visión Artificial.*
Universidad Surcolombiana, Neiva, Colombia.
