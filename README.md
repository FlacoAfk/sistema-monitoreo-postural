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
pip install -r requirements.txt
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
│   └── yolov11m.pt   ✗ 106MB — supera límite GitHub; pedirlo al equipo
├── src/
│   ├── core/             # Lógica de dominio — PostureAnalyzer, CPI math
│   │   ├── __init__.py
│   │   └── posture_analyzer.py
│   ├── inference/        # Runtime ML — YOLO inference, keypoints
│   │   ├── __init__.py
│   │   └── inference_engine.py
│   ├── ui/               # Presentación — Gradio dashboard
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── audio_alert.py
│   ├── tools/            # Utilidades — benchmark
│   │   ├── __init__.py
│   │   └── model_benchmark.py
│   ├── tests/            # Tests del sistema
│   │   ├── __init__.py
│   │   └── test_system.py
│   └── __init__.py
├── Dockerfile            # CPU — despliegue contenerizado
├── Dockerfile.gpu        # GPU — requiere NVIDIA Container Toolkit
├── docker-compose.yml    # Orquestación con perfil CPU/GPU
├── .dockerignore
├── requirements.txt
└── README.md
```

> `yolov11m.pt` (106MB) no está en el repositorio por superar el límite de GitHub.
> Si lo necesitás, pedíselo al equipo o descargalo por separado y colocalo en `models/`.
>
> Si tus pesos están en otra ubicación, definí la variable de entorno antes de correr:
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
cd sistema-monitoreo-postural
python -m src.ui.app
```

Abrí el navegador en **http://127.0.0.1:7860**

**Con Docker (CPU, recomendado para despliegue):**
```bash
docker compose up --build
```

**Con Docker (GPU):**
```bash
docker compose --profile gpu up --build
```

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
| **Tema claro/oscuro** | Toggle con persistencia en localStorage |
| **Optimización adaptativa** | Auto-detecta GPU/CPU y ajusta rendimiento |

### Benchmark de modelos

```bash
cd sistema-monitoreo-postural
python -m src.tools.model_benchmark
```

### Validación de keypoints (deprecated)

La herramienta `validate_keypoints.py` fue removida del repositorio. La funcionalidad fue reemplazada por pruebas automatizadas en `src/tests/test_system.py` y el benchmark tool `src/tools/model_benchmark.py`.

---

## Rendimiento — Optimización adaptativa

El sistema detecta automáticamente el hardware disponible y ajusta la configuración:

| Hardware | FP16 | imgsz | Skip | FPS esperado |
|----------|------|-------|------|--------------|
| GPU NVIDIA (compute ≥ 6.0) | ✓ | 256px | 1/2 | 60-160 FPS |
| GPU NVIDIA (compute < 6.0) | ✗ | 256px | 1/3 | 30-60 FPS |
| CPU (cualquier) | ✗ | 192px | 1/4 | 8-20 FPS |

**Optimizaciones GPU:**
- FP16 half-precision (reduce cómputo ~40%)
- cuDNN benchmark (auto-tuning de kernels CUDA)
- Frame skipping inteligente con reutilización de overlay

**Optimizaciones CPU:**
- Tamaño de inferencia reducido (192px vs 256px)
- Threads de PyTorch ajustados al número de cores
- Skip más agresivo (1/4) para mantener fluidez

El header del dashboard muestra el estado de hardware detectado:
- `🟢 GPU: ... · FP16: ✓ · Skip: 1/2` — GPU con FP16 activo
- `🟡 CPU (N cores) · FP32 · img:192px · Skip: 1/4` — modo CPU

---

## Estructura del proyecto

```
sistema-monitoreo-postural/
├── src/
│   ├── core/                  # Lógica de dominio
│   │   ├── __init__.py
│   │   └── posture_analyzer.py    # CPI — Combined Posture Index
│   ├── inference/             # Runtime ML
│   │   ├── __init__.py
│   │   └── inference_engine.py    # YOLO-Pose, keypoints, overlay
│   ├── ui/                    # Presentación
│   │   ├── __init__.py
│   │   ├── app.py                 # Dashboard Gradio
│   │   └── audio_alert.py         # Alerta sonora cross-platform
│   ├── tools/                 # Utilidades
│   │   ├── __init__.py
│   │   └── model_benchmark.py     # Comparador de modelos
│   ├── tests/                 # Tests del sistema
│   │   ├── __init__.py
│   │   └── test_system.py
│   └── __init__.py
├── models/                    # Pesos entrenados (.pt)
├── docs/
│   └── INFORME_MODELO_MATEMATICO.md
├── Dockerfile                 # CPU — despliegue contenerizado
├── Dockerfile.gpu             # GPU — NVIDIA Container Toolkit
├── docker-compose.yml         # Orquestación multi-perfil
├── .dockerignore
├── requirements.txt
├── MODELO_MATEMATICO_CPI.md
├── README.md
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

### `src/inference/inference_engine.py`
- Carga modelos YOLO-Pose (.pt) con pipeline asíncrono
- Devuelve `KeypointResult` con 9 keypoints (x, y, confianza)
- Esqueleto visual: K0→K1→K8→K3→K4 (cadena posterior)

### `src/core/posture_analyzer.py`
- Extrae 5 keypoints (K0, K1, K8, K3, K4)
- Calcula ángulo lumbar `∠K8-K3-K4` + curvatura escapular normalizada
- Clasifica: ≤35 CORRECTO, 35–50 ALERTA LEVE, >50 ALERTA CRÍTICA
- Contador de tiempo acumulado en mala postura

### `src/ui/app.py`
- Dashboard Gradio con streaming de webcam
- Panel de métricas estático actualizado por JS (sin flicker)
- Gauge CPI animado, sparkline, indicador de confianza
- Soporte i18n ES/EN/PT con dropdown de idioma
- Exportación de sesión a CSV

### `src/tools/model_benchmark.py`
- Evalúa los 4 modelos sobre imágenes de prueba
- Genera JSON con métricas detalladas + gráfica comparativa

### `Dockerfile` / `Dockerfile.gpu`
- `Dockerfile`: imagen CPU basada en `python:3.11-slim` (~2 GB)
- `Dockerfile.gpu`: imagen GPU basada en `nvidia/cuda:12.6-runtime` (~6 GB, requiere NVIDIA Container Toolkit)
- `docker-compose.yml`: perfil CPU por defecto; `--profile gpu` para GPU
- Volumen `./models:/app/models:ro` para pesos de modelos

---

## Paper de Referencia

Castañeda Guzmán, J. & Idarraga Plazas, L. (2026). *Sistema de Monitoreo
Postural en Tiempo Real Mediante Técnicas de Visión Artificial.*
Universidad Surcolombiana, Neiva, Colombia.
