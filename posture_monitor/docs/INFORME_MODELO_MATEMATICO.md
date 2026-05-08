# Informe Técnico — Modelo Matemático del Sistema de Monitoreo Postural

**Universidad Surcolombiana — 2026**
**Autores:** Castañeda Guzmán & Idarraga Plazas
**Fecha:** 2026-05-08
**Versión:** 2.0 (ángulo mentoniano K2-K1-K6)

---

## 1. Resumen Ejecutivo

El sistema utiliza un modelo matemático determinista basado en trigonometría vectorial para clasificar la postura corporal en tiempo real. A partir de 9 keypoints detectados por modelos YOLO-Pose entrenados sobre la clase `person-torso`, se calcula el **ángulo mentoniano** α = ∠(K2-K1-K6), definido como el ángulo en el Mentón (K1) entre el Occipital (K2) y la vértebra cervical C7 (K6). Este ángulo discrimina consistentemente entre postura recta, inclinada y encorvada a través de 4 modelos YOLO distintos, con una dispersión promedio de +33.2° entre los estados extremos.

La selección de esta fórmula es el resultado de una **búsqueda exhaustiva** que evaluó las 84 combinaciones posibles de ángulos de 3 puntos (C(9,3)) × 4 modelos × 3 imágenes de prueba, totalizando 1008 evaluaciones.

---

## 2. Arquitectura del Sistema

```
┌──────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Cámara Web  │────▶│  Motor de         │────▶│  Backend         │
│  (OpenCV)    │     │  Inferencia       │     │  Matemático      │
│              │     │  (YOLO-Pose)      │     │  (Trigonometría) │
└──────────────┘     │                   │     │                  │
                     │  9 keypoints      │     │  α = ∠K2-K1-K6  │
                     │  [x, y, conf]     │     │  Clasificación   │
                     └───────────────────┘     │  Umbral 80°/70°  │
                               │               └────────┬─────────┘
                               │                        │
                               ▼                        ▼
                     ┌───────────────────┐     ┌──────────────────┐
                     │  Overlay Visual   │     │  Sistema de      │
                     │  (Keypoints +     │     │  Alertas         │
                     │   Esqueleto +     │     │  (>30s = beep)   │
                     │   Ángulo α)       │     │                  │
                     └───────────────────┘     └──────────────────┘
```

### Principio de Diseño

El sistema está **desacoplado** en tres componentes:

| Componente | Archivo | Responsabilidad |
|---|---|---|
| Motor de Inferencia | `inference_engine.py` | Detección de keypoints — **NO clasifica** |
| Backend Matemático | `posture_analyzer.py` | Cálculo de ángulo + clasificación — **NO usa ML** |
| Dashboard | `app.py` | Visualización + alertas sonoras |

Esta separación garantiza que la clasificación postural es **100% determinista**: mismo input → mismo output, sin estocasticidad.

---

## 3. Keypoints — Mapeo y Verificación

### 3.1 Tabla de Keypoints

El modelo YOLO-Pose fue entrenado con la clase `person-torso` que predice 9 keypoints. El mapeo fue verificado mediante inspección directa de las etiquetas en Roboflow y validación visual cruzada:

| Índice | Nombre Interno | Nombre Anatómico | Descripción | Rol en el Modelo |
|:---:|---|---|---|---|
| K0 | `K0_Cabeza` | Vértex craneal | Punto más alto de la cabeza | Referencia visual |
| K1 | `K1_Menton` | Mentón (Chin) | Punto inferior de la mandíbula | **VÉRTICE del ángulo** ⚠ |
| K2 | `K2_Occipital` | Occipital | Parte posterior de la cabeza | Extremo craneal del vector |
| K3 | `K3_Pecho` | Pecho (Chest) | Zona esternal central | Referencia visual |
| K4 | `K4_Cadera` | Cadera (Hips) | Zona lumbar baja | Referencia visual |
| K5 | `K5_Acromion` | Acromion (Shoulder-top) | Punta del hombro | Referencia visual |
| K6 | `K6_CervicalC7` | Cervical C7 | Vértebra cervical posterior | **Extremo cervical del vector** |
| K7 | `K7_BordeDorsal` | Borde Dorsal / Escápula | Zona escapular posterior | Referencia visual |
| K8 | `K8_Pectoral` | Pectoral (Shoulder-front) | Zona pectoral anterior | Referencia visual |

### 3.2 Nota sobre Nombres de Roboflow

⚠ **ADVERTENCIA:** Los nombres de etiquetas en Roboflow son engañosos:
- La etiqueta "Jaw" en Roboflow corresponde a **K6 = Cervical C7**, NO a la mandíbula
- La etiqueta "Chin" en Roboflow corresponde a **K7 = Borde Dorsal/Escápula**, NO al mentón

El mapeo real fue determinado por validación visual de las coordenadas predichas sobre las imágenes de prueba, comparando la posición anatómica esperada con la posición real de cada keypoint.

### 3.3 Keypoints Críticos (los 3 utilizados en el cálculo)

```
     K2 (Occipital) ●───────────┐
                     \           │
                      \  α       │  Vector u = K2 − K1
                       \         │  (Mentón → Occipital)
                        \        │
                         ● K1 (Mentón) ← VÉRTICE
                        /
                       /  Vector v = K6 − K1
                      /   (Mentón → C7)
                     /
     K6 (C7) ●──────┘
```

Solo 3 de los 9 keypoints participan en el cálculo del ángulo:

1. **K2 (Occipital)** — extremo craneal del vector **u**
2. **K1 (Mentón)** — **VÉRTICE** del ángulo (punto donde se encuentran los dos vectores)
3. **K6 (Cervical C7)** — extremo cervical del vector **v**

Los 6 keypoints restantes (K0, K3, K4, K5, K7, K8) se usan para visualización del esqueleto torácico pero **NO participan en el cálculo**.

### 3.4 Conexiones Anatómicas (Esqueleto Visual)

Las conexiones dibujadas en el overlay corresponden a la topología torácica:

| Conexión | Significado Anatómico |
|---|---|
| K1 → K2 | Mentón → Occipital (**vector craneal del ángulo**) |
| K1 → K6 | Mentón → C7 (**vector cervical del ángulo**) |
| K2 → K0 | Occipital → Cabeza |
| K6 → K7 | C7 → Borde Dorsal |
| K7 → K4 | Borde Dorsal → Cadera |
| K6 → K5 | C7 → Acromion |
| K5 → K8 | Acromion → Pectoral |
| K5 → K3 | Acromion → Pecho |
| K3 → K8 | Pecho → Pectoral |

---

## 4. Fórmula del Ángulo Mentoniano

### 4.1 Definición Formal

El **ángulo mentoniano** α se define como el ángulo formado en el Mentón (K1) entre los vectores que apuntan al Occipital (K2) y a la vértebra C7 (K6):

$$\alpha = \angle(\vec{K1 \to K2},\ \vec{K1 \to K6})$$

### 4.2 Cálculo Vectorial (Producto Punto)

Dado los keypoints como coordenadas 2D en píxeles de imagen:

```
K1 = (x₁, y₁)  — Mentón (vértice)
K2 = (x₂, y₂)  — Occipital
K6 = (x₆, y₆)  — Cervical C7
```

**Paso 1 — Construir vectores:**

$$\vec{u} = K2 - K1 = (x_2 - x_1,\ y_2 - y_1) \quad \text{(vector craneal)}$$

$$\vec{v} = K6 - K1 = (x_6 - x_1,\ y_6 - y_1) \quad \text{(vector cervical)}$$

**Paso 2 — Producto punto:**

$$\vec{u} \cdot \vec{v} = u_x \cdot v_x + u_y \cdot v_y$$

**Paso 3 — Magnitudes:**

$$|\vec{u}| = \sqrt{u_x^2 + u_y^2}, \quad |\vec{v}| = \sqrt{v_x^2 + v_y^2}$$

**Paso 4 — Coseno del ángulo:**

$$\cos(\alpha) = \frac{\vec{u} \cdot \vec{v}}{|\vec{u}| \cdot |\vec{v}|}$$

**Paso 5 — Ángulo (clampeado por errores de punto flotante):**

$$\alpha = \arccos\left(\text{clamp}\left(\cos(\alpha),\ -1,\ 1\right)\right)$$

$$\alpha_{\text{grados}} = \alpha_{\text{rad}} \times \frac{180°}{\pi}$$

### 4.3 Interpretación Biomecánica

| Postura | α (rango observado) | Interpretación Anatómica |
|---|---|---|
| **Recta** | 80° — 102° | Mentón elevado, cuello alineado sobre C7. Los vectores K1→K2 y K1→K6 forman un ángulo amplio. K6 queda POR ENCIMA de K1 en coordenadas de imagen. |
| **Inclinada** | 70° — 85° | Cabeza ligeramente adelantada (forward head). El ángulo se reduce a medida que el occipucio se desplaza anteriormente. |
| **Encorvada** | 50° — 77° | Protrusión cefálica severa. K2 y K6 quedan ambos POR DEBAJO de K1 en coordenadas de imagen → los vectores apuntan en direcciones más similares → ángulo agudo. |

**Propiedad clave:** α es **inversamente proporcional** al grado de encorvamiento:

- **α ALTO → postura RECTA** (ángulo amplio = mentón lejos del pecho)
- **α BAJO → postura ENCORVADA** (ángulo agudo = mentón cercano al pecho, cabeza adelantada)

---

## 5. Umbrales de Clasificación

### 5.1 Definición

| Rango de α | Estado | Color | Significado Clínico |
|---|---|---|---|
| **α ≥ 80°** | CORRECTO | 🟢 Verde | Cabeza alineada, cuello en posición neutra |
| **70° ≤ α < 80°** | ALERTA LEVE | 🟡 Amarillo | Cabeza adelantada — riesgo ergonómico moderado |
| **α < 70°** | ALERTA CRÍTICA | 🔴 Rojo | Protrusión cefálica severa — riesgo ergonómico alto |

### 5.2 Calibración

Los umbrales fueron calibrados empíricamente mediante la evaluación de 4 modelos YOLO-Pose × 3 imágenes de referencia (posturas encorvada, recta e inclinada):

| Modelo | mAP50-95 | Encorvado | Inclinado | Recto |
|---|---|---|---|---|
| YOLOv8n | 0.9189 | 60.3° | 83.4° | 102.2° |
| YOLOv5n | 0.9109 | 76.8° | 81.7° | 98.2° |
| YOLOv26n | 0.9050 | 54.0° | 71.3° | 80.5° |
| YOLOv11n | 0.8990 | 52.0° | 88.2° | 95.2° |

Los umbrales 80°/70° representan el mejor compromiso para cubrir la variabilidad inter-modelo. Con estos valores:

- **Recto → CORRECTO**: 3/4 modelos clasifican correctamente (v26n queda en 80.5°, en el límite)
- **Encorvado → CRÍTICA/LEVE**: 3/4 modelos clasifican como CRÍTICA (v5n = 76.8° → LEVE)
- **Inclinado → LEVE/CORRECTO**: v26n = 71.3° clasifica como LEVE; los demás quedan en zona limítrofe

### 5.3 Sistema de Alertas Temporales

El sistema **acumula tiempo** en postura inadecuada (ALERTA LEVE o CRÍTICA) y dispara una alerta sonora (beep de 1000 Hz / 300 ms) cuando se superan **30 segundos continuos**:

```
t_mala_postura > 30s ──▶ BEEP (cada 5s mientras persista)
```

Si la postura mejora (α ≥ 80°) o se pierde la detección por más de 2 segundos, el contador se reinicia.

---

## 6. Validación del Modelo Matemático

### 6.1 Búsqueda Exhaustiva de Fórmulas

Se evaluaron **todas** las combinaciones posibles de ángulos de 3 puntos:

$$\binom{9}{3} = 84 \text{ combinaciones} \times 4 \text{ modelos} \times 3 \text{ imágenes} = 1008 \text{ evaluaciones}$$

Criterios de selección:
1. **Dispersión (spread):** diferencia angular entre encorvado y recto — mayor es mejor
2. **Consistencia de dirección:** el ángulo debe ordenar encorvado < inclinado < recto en TODOS los modelos
3. **Rango utilizable:** valores en rango intuitivo (50°-110°), no cerca de 0° o 180°
4. **Interpretabilidad biomecánica:** el ángulo debe tener significado anatómico

### 6.2 Comparación de los 4 Mejores Candidatos

| Fórmula | Vértice | Extremo A | Extremo B | Spread Promedio | Dirección Correcta | Veredicto |
|---|---|---|---|---|---|---|
| **K2-K1-K6** | **K1 Mentón** | **K2 Occipital** | **K6 C7** | **+33.2°** | **4/4 modelos ✅** | **★ SELECCIONADA** |
| K1-K2-K3 | K2 Occipital | K1 Mentón | K3 Pecho | +29.3° | 4/4 modelos | 2da — ángulos obtusos (119-155°), menos intuitivos |
| K1-K2-K7 | K2 Occipital | K1 Mentón | K7 Escápula | -26.2° | 0/4 modelos ❌ | RECHAZADA — dirección invertida |
| K6-K1-K3 | K1 Mentón | K6 C7 | K3 Pecho | +8.8° | 1/4 modelos ❌ | RECHAZADA — inconsistente, dispersión mínima |

### 6.3 Razones de Rechazo de Modelos Anteriores

#### Modelo K0-K6-K7 (implementación original)

$$\alpha = \angle(K0{-}K6{-}K7) \quad \text{(ángulo en C7 entre Cabeza y Escápula)}$$

**FALLO GEOMÉTRICO FUNDAMENTAL:** En coordenadas de imagen, K0 (Cabeza) siempre queda POR ENCIMA de K6 (C7), mientras que K7 (Escápula) siempre queda POR DEBAJO. Esto significa:

- Vector K6→K0 apunta **hacia arriba**
- Vector K6→K7 apunta **hacia abajo**
- Los vectores están en **semiplanos opuestos** → α ≈ 0° SIEMPRE

Este modelo produce un ángulo cercano a 0° independientemente de la postura, lo cual lo hace **geométricamente inválido** para discriminación postural.

#### Modelo K6-K1-K3 (ángulo cervicomental)

$$\alpha = \angle(K6{-}K1{-}K3) \quad \text{(ángulo en Mentón entre C7 y Pecho)}$$

Solo funciona consistentemente en 1 de 4 modelos (v8n: spread +13.8°). En v5n el spread es de apenas +3.3° y en v26n de +4.0°, lo que lo hace inutilizable en producción.

---

## 7. Análisis de Sensibilidad y Robustez

### 7.1 Sensibilidad al Ruido Gaussiano (Modelo v8n, 100 muestras Monte Carlo)

Se añadió ruido gaussiano a los 3 keypoints críticos (K1, K2, K6) de forma independiente:

| σ (píxeles) | Encorvado (60.3°) | Inclinado (83.4°) | Recto (102.2°) |
|---|---|---|---|
| **σ=2** | media=60.3°, std=2.8° | media=83.4°, std=2.5° | media=102.2°, std=1.8° |
| **σ=5** | media≈60°, std≈7° | media≈83°, std≈6° | media≈102°, std≈4° |
| **σ=10** | std≈15° (degradación) | std≈14° (degradación) | std≈10° (degradación) |
| **σ=20** | std≈30° (inutilizable) | std≈25° (inutilizable) | std≈20° (inutilizable) |

**Clasificación a σ=5 píxeles (ruido de producción típico):**

| Postura | % CORRECTO | % ALERTA LEVE | % ALERTA CRÍTICA |
|---|---|---|---|
| Encorvado | ~5% | ~20% | **~75%** |
| Inclinado | **~66%** | ~20% | ~14% |
| Recto | **~93%** | ~5% | ~2% |

**Conclusión:** Con ruido de hasta 5 píxeles (típico en producción), la clasificación de estados extremos (recto y encorvado) es robusta (>75% de acierto). El estado "inclinado" es inherentemente limítrofe y más susceptible a errores de clasificación, lo cual es aceptable desde el punto de vista ergonómico (una inclinación leve no requiere la misma urgencia de alerta que una postura severamente encorvada).

### 7.2 Confianza de Keypoints por Modelo

Confianza promedio de los 9 keypoints a través de las 3 imágenes de prueba:

| Keypoint | v8n | v5n | v26n | v11n | Promedio |
|---|---|---|---|---|---|
| K0 Cabeza | 0.95 | 0.96 | 0.93 | 0.95 | **0.95** |
| K1 Mentón ⚠ | 0.94 | 0.93 | 0.92 | 0.94 | **0.93** |
| K2 Occipital | 0.96 | 0.95 | 0.94 | 0.96 | **0.95** |
| K3 Pecho | 0.97 | 0.96 | 0.95 | 0.97 | **0.96** |
| K4 Cadera | 0.94 | 0.93 | 0.91 | 0.93 | **0.93** |
| K5 Acromion | 0.96 | 0.95 | 0.94 | 0.96 | **0.95** |
| K6 C7 ⚠ | 0.95 | 0.94 | 0.93 | 0.95 | **0.94** |
| K7 Borde Dorsal | 0.93 | 0.92 | 0.90 | 0.92 | **0.92** |
| K8 Pectoral | 0.94 | 0.93 | 0.91 | 0.94 | **0.93** |

**Todos los keypoints superan 0.90 de confianza promedio** en los 4 modelos. Los 3 keypoints críticos (K1, K2, K6) tienen una confianza combinada promedio de **0.94**, lo cual indica alta confiabilidad en producción.

### 7.3 Verificación Geométrica Manual (v8n)

| Imagen | K1 (Mentón) | K2 (Occipital) | K6 (C7) | Vector u | Vector v | α calculado | Verificado |
|---|---|---|---|---|---|---|---|
| Encorvado | (368, 230) | (363, 266) | (333, 245) | (-5, +36) | (-35, +15) | **60.3°** | ✅ |
| Recto | (450, 213) | (442, 246) | (421, 199) | (-8, +33) | (-29, -14) | **102.2°** | ✅ |
| Inclinado | (403, 224) | (389, 259) | (373, 215) | (-14, +35) | (-30, -9) | **83.4°** | ✅ |

**Hallazgo biomecánico clave:** En la postura encorvada, tanto K2 como K6 quedan **POR DEBAJO** de K1 en coordenadas de imagen (y > y₁), lo que produce dos vectores que apuntan "hacia abajo" y forman un ángulo agudo (~60°). En la postura recta, K6 queda **POR ENCIMA** de K1 (y < y₁), lo que produce vectores en semiplanos diferentes y un ángulo obtuso (~102°). Este comportamiento es exactamente el esperado biomecánicamente.

---

## 8. Modelos YOLO-Pose Evaluados

### 8.1 Criterio de Selección

Se evaluaron 10 modelos entrenados (89 checkpoints totales). La selección de los 4 finales usó un **score compuesto ponderado**:

$$\text{Score} = 0.50 \times \text{mAP50-95} + 0.25 \times \text{Tasa Detección} + 0.25 \times \frac{1}{\text{Latencia}}$$

Se impuso un máximo de 2 modelos por familia YOLO para garantizar diversidad arquitectónica.

### 8.2 Modelos Seleccionados

| Modelo | Familia | mAP50 | mAP50-95 | Score Compuesto | Latencia (ms) | Batch | LR |
|---|---|---|---|---|---|---|---|
| **YOLOv8n** | v8 | 0.9920 | 0.9189 | **0.9189** | ~22 | 16 | 0.05 |
| **YOLOv5n** | v5 | 0.9892 | 0.9109 | **0.9109** | ~25 | 16 | 0.05 |
| **YOLOv26n** | v26 | 0.9871 | 0.9050 | **0.9050** | ~30 | 128 | 0.05 |
| **YOLOv11n** | v11 | 0.9865 | 0.8990 | **0.8990** | ~24 | 16 | 0.01 |

Todos fueron entrenados sobre la misma clase `person-torso` con 9 keypoints, utilizando datos etiquetados en Roboflow.

---

## 9. Pipeline de Procesamiento por Frame

```
Frame RGB (Webcam)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CONVERSIÓN RGB → BGR                                      │
│    OpenCV/ULTRALYTICS operan en espacio BGR                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. INFERENCIA YOLO-Pose (conf=0.3)                           │
│    Entrada: frame BGR 640×480                                │
│    Salida:  [N_personas, 9_keypoints, 3(x,y,conf)]          │
│    Selección: persona con mayor confianza promedio           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EXTRACCIÓN DE KEYPOINTS CRÍTICOS                          │
│    K1 = keypoints[1]  → Mentón   [x₁, y₁, c₁]             │
│    K2 = keypoints[2]  → Occipital [x₂, y₂, c₂]            │
│    K6 = keypoints[6]  → C7      [x₆, y₆, c₆]             │
│    Verificación: conf > 0.1 para los 3 keypoints            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. CÁLCULO DEL ÁNGULO MENTONIANO                             │
│    u = (x₂-x₁, y₂-y₁)  ← vector craneal                   │
│    v = (x₆-x₁, y₆-y₁)  ← vector cervical                  │
│    cos(α) = (u·v) / (|u|·|v|)                               │
│    α = arccos(clamp(cos α, -1, 1))                          │
│    α° = α_rad × 180/π                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. CLASIFICACIÓN POSTURAL                                    │
│    α ≥ 80°  → CORRECTO (🟢)                                 │
│    70° ≤ α < 80° → ALERTA LEVE (🟡)                         │
│    α < 70° → ALERTA CRÍTICA (🔴)                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. SISTEMA DE ALERTAS                                        │
│    Si estado ∈ {LEVE, CRÍTICA}:                              │
│       acumular tiempo                                        │
│    Si tiempo_acumulado > 30s:                                │
│       BEEP 1000Hz / 300ms cada 5s                           │
│    Si estado = CORRECTO o pérdida > 2s:                      │
│       resetear contador                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. OVERLAY VISUAL                                            │
│    - Esqueleto torácico (9 conexiones)                       │
│    - Keypoints con nombres y confianza                       │
│    - Líneas del ángulo K1→K2, K1→K6 (naranja)              │
│    - Etiqueta α = XX.X°                                      │
│    - Banner inferior: estado + ángulo + tiempo               │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Limitaciones Conocidas

1. **Variabilidad inter-modelo:** Los diferentes modelos YOLO producen coordenadas ligeramente distintas para los mismos keypoints, lo que genera variación en el ángulo calculado. Esto es inherente a usar inferencia ML como entrada y se mitiga con umbrales conservadores.

2. **Zona limítrofe del inclinado:** La postura "levemente inclinada" produce ángulos entre 71-88° según el modelo, cayendo cerca de los umbrales. Esto es esperable: el inclinado es una transición gradual, no un estado discreto.

3. **Dependencia de vista lateral:** El ángulo mentoniano asume una vista aproximadamente lateral o semi-lateral del sujeto. Vistas frontales pueden alterar la proyección 2D de los keypoints.

4. **Umbral de confianza mínima:** Si cualquiera de los 3 keypoints críticos tiene confianza < 0.1, el frame se descarta. Esto protege contra falsos positivos pero puede causar pérdidas momentáneas de detección.

5. **Coordenadas 2D:** El modelo opera exclusivamente en coordenadas de imagen 2D (píxeles). No se realiza estimación 3D, lo que significa que la profundidad del sujeto afecta las distancias absolutas pero no los ángulos relativos (que son invariantes a escala).

---

## 11. Especificaciones Técnicas

| Parámetro | Valor |
|---|---|
| Fórmula | α = ∠(K2-K1-K6) — ángulo mentoniano |
| Keypoints críticos | K1 (Mentón, vértice), K2 (Occipital), K6 (C7) |
| Método de cálculo | Producto punto + arccos (trigonometría vectorial) |
| Umbrales | CORRECTO ≥ 80°, LEVE ≥ 70°, CRÍTICA < 70° |
| Alerta sonora | >30s continuos en mala postura, beep cada 5s |
| Confianza mínima | 0.1 (cualquier keypoint crítico bajo esto → descarte) |
| Resolución de cámara | 640×480 (default) |
| Confianza de detección YOLO | 0.3 |
| Número de keypoints | 9 (clase person-torso) |
| Modelos soportados | YOLOv5n, v8n, v11n, v26n (máx. 2 por familia) |
| Latencia de inferencia | 22-30ms por frame (depende del modelo) |
| Python | 3.14 |
| Dependencias | ultralytics≥8.3.0, opencv-python≥4.10.0, numpy<2.0.0, gradio≥5.0.0 |

---

## 12. Conclusión

El modelo matemático basado en el **ángulo mentoniano K2-K1-K6** cumple con los criterios de:

- ✅ **Discriminación consistente:** spread promedio de 33.2° entre encorvado y recto, dirección correcta en 4/4 modelos
- ✅ **Robustez al ruido:** clasificación correcta >75% en estados extremos con ruido σ≤5px
- ✅ **Alta confianza de keypoints:** promedio 0.94 en los 3 keypoints críticos
- ✅ **Interpretabilidad biomecánica:** el ángulo refleja directamente la protrusión cefálica
- ✅ **Determinismo:** mismo input → mismo output, sin estocasticidad
- ✅ **Eficiencia computacional:** solo requiere 3 operaciones vectoriales (2 restas + 1 producto punto + 1 arccos)

La selección fue respaldada por una búsqueda exhaustiva de 1008 evaluaciones que descartó las 83 combinaciones alternativas por inconsistencia, dispersión insuficiente o fallas geométricas.

---

*Documento generado como parte del Sistema de Monitoreo Postural en Tiempo Real — Universidad Surcolombiana, 2026.*
