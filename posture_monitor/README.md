# Sistema de Monitoreo Postural en Tiempo Real

**Universidad Surcolombiana — Facultad de Ingeniería**  
Castañeda Guzmán & Idarraga Plazas, 2026

---

## Resumen

Sistema de detección de posturas corporales inadecuadas en trabajadores de oficina,
basado en visión artificial con redes YOLO-Pose y trigonometría vectorial.

### Arquitectura
```
Webcam → YOLO-Pose (9 keypoints del torso) → Trigonometría (ángulo cervicodorsal) → Alertas
```

El ángulo de flexión cervicodorsal θ = ∠(K6→K0, K6→K7) se calcula a partir de 3 keypoints
(mapeo Roboflow → YOLO, confirmado 2026-05-07):
- **K0** (Roboflow 0) — Cabeza / Coronilla → extremo cefálico
- **K6** (Roboflow 13) — Cervical posterior C7 ← **PIVOTE** ⚠
- **K7** (Roboflow 14) — Borde dorsal / Escápula → extremo dorsal

---

## Modelos Seleccionados

De 108 submodelos evaluados (4 familias × 3 variantes × 9 checkpoints),
los 10 modelos entrenados fueron evaluados con benchmark intensivo (confianza K0+K6+K7, detección, velocidad):

| # | Modelo | SCORE | K6 (C7 pivote) | Latencia | Detección |
|---|--------|-------|-----------------|----------|-----------|
| 1 | YOLOv5n 🎯 | 0.9109 | 0.9998 | 30.6ms | **95.2%** |
| 2 | YOLOv8n 🚀 | 0.9189 | 0.9995 | **22.4ms** | 90.5% |
| 3 | YOLOv26n ⚖️ | 0.9050 | 0.9988 | 27.3ms | 90.5% |
| 4 | YOLO11n ⭐ | 0.8990 | 0.9996 | 30.7ms | 90.5% |

**Criterios de selección:** Mapeo Roboflow→YOLO corregido (K0=cabeza, K6=C7 pivote, K7=escápula),
score compuesto (confianza 50% + detección 25% + velocidad 25%), diversidad arquitectónica.

---

## Instalación

### Requisitos
- Python 3.10+
- Cámara web funcional
- (Opcional) GPU NVIDIA con CUDA para mejor rendimiento

### Pasos
```bash
# 1. Clonar o copiar el proyecto
cd posture_monitor

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar que los modelos están en la carpeta correcta
# Los archivos .pt deben estar en:
#   ..\Modelos entrenados\yolov8n_pose_b16_lr05\weights\best.pt
#   ..\Modelos entrenados\yolov5n_pose_b16_lr05\weights\best.pt
#   ..\Modelos entrenados\yolov26n_pose_b128_lr05\weights\best.pt
#   ..\Modelos entrenados\yolov11n_pose_b16_lr01\weights\best.pt
#   ..\Modelos entrenados\yolov11n_pose_b16_lr01\weights\best.pt
```

---

## Uso

### Interfaz Gráfica (Dashboard)
```bash
cd src
python app.py
```
Abre http://127.0.0.1:7860 en el navegador.

- **Selector de modelo**: cambiá entre los 4 modelos en caliente
- **Video en vivo**: overlay con los 9 keypoints + esqueleto anatómico
- **Panel de métricas**: ángulo cervicodorsal actual + estado postural
- **Alertas**: visuales (código de color) cuando α > 15° por más de 30s

### Benchmark de Modelos
```bash
cd src
python model_benchmark.py
```
Genera `outputs/model_benchmark.json` y `outputs/model_comparison.png`.

---

## Estructura del Proyecto

```
posture_monitor/
├── src/
│   ├── __init__.py
│   ├── inference_engine.py   # Motor YOLO-Pose (carga modelo, inferencia, webcam)
│   ├── posture_analyzer.py   # Backend matemático (trigonometría vectorial)
│   ├── app.py                # Dashboard Gradio (UI en tiempo real)
│   └── model_benchmark.py    # Comparador de modelos (métricas + gráficas)
├── models/                   # (opcional) copias locales de los .pt
├── outputs/                  # Resultados de benchmark
├── requirements.txt
└── README.md
```

---

## Componentes

### 1. `inference_engine.py` — Motor de Inferencia
- Carga modelos YOLO-Pose (.pt)
- Pipeline asíncrono: hilo de captura webcam + hilo de inferencia
- Devuelve `KeypointResult` con coordenadas (X,Y) y confidence de 9 keypoints
- NO toma decisiones clasificatorias

### 2. `posture_analyzer.py` — Backend Matemático
- Recibe keypoints → construye vectores u=K0−K6, v=K7−K6
- Calcula ángulo θ vía producto punto → α = 180° − θ
- Clasifica: α≤15° CORRECTO, 15<α≤25° ALERTA LEVE, α>25° CRÍTICA
- Contador de tiempo acumulado en mala postura

### 3. `app.py` — Dashboard
- Interfaz Gradio con video en tiempo real
- Overlay de keypoints + esqueleto anatómico + líneas del ángulo
- Panel con ángulo, estado, y tiempo acumulado
- Selector de modelo en caliente

### 4. `model_benchmark.py` — Comparador
- Evalúa los 4 modelos sobre imágenes de prueba
- Genera JSON con métricas detalladas por imagen
- Gráfica de barras: confianza K2/K4/K7 + latencia

---

## Paper de Referencia

Castañeda Guzmán, J. & Idarraga Plazas, L. (2026). *Sistema de Monitoreo
Postural en Tiempo Real Mediante Técnicas de Visión Artificial.*
Universidad Surcolombiana, Neiva, Colombia.
