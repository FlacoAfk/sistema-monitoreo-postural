# Sistema de Monitoreo Postural en Tiempo Real

**Universidad Surcolombiana — Facultad de Ingeniería**  
Castañeda Guzmán & Idarraga Plazas, 2026

---

## Resumen

Sistema de detección de posturas corporales inadecuadas en trabajadores de oficina,
basado en visión artificial con redes YOLO-Pose y el **Combined Posture Index (CPI)**,
un índice multivectorial que integra curvatura escapular y ángulo lumbar.

```
Webcam → YOLO-Pose (9 keypoints) → CPI (5 keypoints posteriores) → Alertas
```

El CPI usa 5 keypoints de la cadena posterior de la espalda (K0, K1, K8, K3, K4)
para calcular simultáneamente la curvatura escapular y el déficit angular lumbar,
proporcionando una separación 12.7× mayor entre posturas que los enfoques monoangulares.

Documentación completa del modelo matemático: [`MODELO_MATEMATICO_CPI.md`](MODELO_MATEMATICO_CPI.md)

---

## Instalación

### Requisitos del sistema

- Python 3.10 o superior
- Cámara web funcional
- (Opcional) GPU NVIDIA con CUDA 12.x para inferencia acelerada

### 1 — Clonar el repositorio

```bash
git clone https://github.com/FlacoAfk/sistema-monitoreo-postural.git
cd sistema-monitoreo-postural
```

### 2 — Crear entorno virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3 — Instalar PyTorch

**CPU (cualquier sistema operativo):**
```bash
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU NVIDIA — CUDA 12.6 (recomendado para mejor rendimiento):**
```bash
pip install torch==2.11.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4 — Instalar el resto de dependencias

```bash
pip install -r posture_monitor/requirements.txt
```

### 5 — Ejecutar

Todos los modelos entrenados ya están incluidos en el repositorio bajo `models/`. No necesitás descargar nada extra.

```
sistema-monitoreo-postural/
├── models/
│   ├── yolov8n.pt    ✓ nano  — usado por el dashboard
│   ├── yolov5n.pt    ✓ nano  — usado por el dashboard
│   ├── yolov26n.pt   ✓ nano  — usado por el dashboard
│   ├── yolov11n.pt   ✓ nano  — usado por el dashboard
│   ├── yolov8s.pt    ✓ small — evidencia de entrenamiento
│   ├── yolov11s.pt   ✓ small — evidencia de entrenamiento
│   ├── yolov26s.pt   ✓ small — evidencia de entrenamiento
│   ├── yolov5m.pt    ✓ medium — evidencia de entrenamiento
│   ├── yolov26m.pt   ✓ medium — evidencia de entrenamiento
│   └── yolov11m.pt   ✓ medium — evidencia de entrenamiento (Git LFS)
├── src/
│   └── app.py
├── requirements.txt
└── README.md
```

> Si por alguna razón los pesos están en otra ubicación, definí la variable de entorno antes de correr:
>
> ```bash
> # Windows
> set POSTURE_MODELS_DIR=C:\ruta\a\tus\modelos
>
> # Linux / Mac
> export POSTURE_MODELS_DIR=/ruta/a/tus/modelos
> ```

---

## Uso

### Dashboard en tiempo real

```bash
cd posture_monitor/src
python app.py
```

Abrí el navegador en **http://127.0.0.1:7860**

#### Funcionalidades del dashboard

| Feature | Descripción |
|---------|-------------|
| **Selector de modelo** | Cambio en caliente entre 4 modelos YOLO-Pose |
| **Gauge CPI** | Anillo SVG animado con valor en tiempo real |
| **Sparkline** | Historial de CPI de los últimos 60 segundos |
| **Indicador de confianza** | Barra de progreso + badge de detección débil |
| **Estado postural** | Card con color reactivo (verde / amarillo / rojo) |
| **Alertas** | Popup visual + beep sonoro tras 30s de mala postura |
| **Umbrales configurables** | Sliders para ajustar CPI leve y crítico |
| **Grabación de sesión** | Exporta CSV con métricas frame a frame |
| **Idiomas** | Español 🇪🇸 / English 🇬🇧 / Português 🇧🇷 |

### Benchmark de modelos

```bash
cd posture_monitor/src
python model_benchmark.py
```

### Validación de keypoints

```bash
cd posture_monitor
python validate_keypoints.py
```

Genera overlays y JSON en `keypoint_output/`.

---

## Estructura del proyecto

```
sistema-monitoreo-postural/
├── posture_monitor/
│   ├── src/
│   │   ├── app.py                 # Dashboard Gradio (UI tiempo real)
│   │   ├── inference_engine.py    # Motor YOLO-Pose (carga, inferencia, webcam)
│   │   ├── posture_analyzer.py    # CPI — Combined Posture Index (5 keypoints)
│   │   ├── model_benchmark.py     # Comparador de modelos
│   │   └── test_system.py         # Tests del sistema
│   ├── docs/
│   │   └── INFORME_MODELO_MATEMATICO.md
│   ├── requirements.txt
│   ├── README.md
│   └── MODELO_MATEMATICO_CPI.md
├── yolov8n_pose_b16_lr05/         # Pesos del modelo (no incluidos en git)
├── yolov5n_pose_b16_lr05/
├── yolov26n_pose_b128_lr05/
├── yolov11n_pose_b16_lr01/
└── .gitignore
```

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

Los umbrales son configurables desde el dashboard sin reiniciar.

---

## Modelos evaluados

De 108 submodelos evaluados, 4 seleccionados por score compuesto (mAP50-95 + detección + velocidad):

| Modelo | SCORE | Latencia | Detección |
|--------|-------|----------|-----------|
| YOLOv5n 🎯 | 0.9109 | 30.6ms | 95.2% |
| YOLOv8n 🚀 | 0.9189 | **22.4ms** | 90.5% |
| YOLOv26n ⚖️ | 0.9050 | 27.3ms | 90.5% |
| YOLO11n ⭐ | 0.8990 | 30.7ms | 90.5% |

---

## Componentes

### `inference_engine.py`
- Carga modelos YOLO-Pose (.pt) con pipeline asíncrono
- Devuelve `KeypointResult` con 9 keypoints (x, y, confianza)
- Esqueleto visual: K0→K1→K8→K3→K4 (cadena posterior)

### `posture_analyzer.py`
- Extrae 5 keypoints (K0, K1, K8, K3, K4)
- Calcula ángulo lumbar `∠K8-K3-K4` + curvatura escapular normalizada
- Clasifica: ≤35 CORRECTO, 35–50 ALERTA LEVE, >50 ALERTA CRÍTICA
- Contador de tiempo acumulado en mala postura

### `app.py`
- Dashboard Gradio con streaming de webcam
- Panel de métricas estático actualizado por JS (sin flicker)
- Gauge CPI animado, sparkline, indicador de confianza
- Soporte i18n ES/EN/PT con dropdown de idioma
- Exportación de sesión a CSV

### `model_benchmark.py`
- Evalúa los 4 modelos sobre imágenes de prueba
- Genera JSON con métricas detalladas + gráfica comparativa

---

## Paper de Referencia

Castañeda Guzmán, J. & Idarraga Plazas, L. (2026). *Sistema de Monitoreo
Postural en Tiempo Real Mediante Técnicas de Visión Artificial.*
Universidad Surcolombiana, Neiva, Colombia.
