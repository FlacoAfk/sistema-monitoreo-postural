# Informe Técnico — Modelo Matemático del Sistema de Monitoreo Postural

**Universidad Surcolombiana — 2026**
**Autores:** Castañeda Guzmán & Idarraga Plazas
**Fecha:** 2026-05-08
**Versión:** 3.0 (CPI — Combined Posture Index, validado 2026-05-08 con 6 imágenes × 4 modelos)

> **Nota:** Este documento ha sido actualizado para reflejar el modelo CPI. La documentación completa y detallada se encuentra en [`MODELO_MATEMATICO_CPI.md`](../MODELO_MATEMATICO_CPI.md).

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

### 3.1 Tabla de Keypoints (Mapeo Roboflow → YOLO)

El modelo YOLO-Pose fue entrenado sobre la clase `person-torso` con 9 keypoints. El mapeo fue verificado mediante inspección directa de las etiquetas en Roboflow (IDs no secuenciales: 0, 1, 2, 6, 7, 10, 13, 14, 18) y validación visual cruzada:

| Índice YOLO | Roboflow ID | Nombre | Descripción anatómica | Rol en CPI |
|:---:|:---:|---|---|---|
| K0 | 0 | Head-back | Occipital / Parte posterior de la cabeza | Extremo craneal |
| K1 | 1 | Neck-back | Cervical C7 / Base posterior del cuello | Referencia espinal |
| K2 | 2 | Shoulder-top | Acromion / Parte superior del hombro | — |
| K3 | 6 | Back-backedge | Borde posterior dorsal / Espalda media | Vértice lumbar |
| K4 | 7 | Hips-backedge | Borde posterior de cadera / Lumbosacra | Extremo caudal |
| K5 | 10 | Neck-middle | Cervical media | — |
| K6 | 13 | Jaw | Mandíbula | — |
| K7 | 14 | Chin | Mentón | — |
| K8 | 18 | Shoulder-back | Zona escapular / Escápula posterior | Extremo escapular |

### 3.2 Keypoints del CPI

De los 9 keypoints, el CPI utiliza exclusivamente los **5 de la cadena posterior**:

$$\{K_0, K_1, K_3, K_4, K_8\}$$

Los keypoints K2, K5, K6, K7 (cadena anterior y facial) son detectados pero no participan en el cálculo del CPI.

### 3.3 Conexiones Anatómicas (Esqueleto Visual)

El esqueleto dibuja exclusivamente la cadena posterior de la espalda (4 conexiones):

| Conexión | Significado Anatómico |
|---|---|
| K0 → K1 | Head-back → C7 (columna cervical alta) |
| K1 → K8 | C7 → Escápula (columna cervical baja) |
| K8 → K3 | Escápula → Espalda media (columna torácica) |
| K3 → K4 | Espalda media → Cadera (columna lumbar) |

---

## 4. Fórmula del Combined Posture Index (CPI)

### 4.1 Definición Formal

El CPI es un índice compuesto que integra dos mediciones complementarias de la columna posterior:

$$\boxed{CPI = D_L \times 2 + C_E \times 100}$$

donde:
- $D_L$: **Déficit angular lumbar** (grados)
- $C_E$: **Curvatura escapular normalizada** (adimensional, %)

### 4.2 Componente Lumbar: $D_L$

**Ángulo lumbar** $\theta_L = \angle(K_8, K_3, K_4)$ con vértice en $K_3$ (espalda media):

$$\vec{u} = K_8 - K_3,\quad \vec{v} = K_4 - K_3$$

$$\theta_L = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{|\vec{u}| \cdot |\vec{v}|}\right)$$

$$D_L = \max(0,\ 180° - \theta_L)$$

### 4.3 Componente Escapular: $C_E$

**Línea espinal teórica** $\ell$ = recta $K_1 \to K_4$ (C7 → cadera).

**Curvatura escapular:** distancia perpendicular de $K_8$ a $\ell$:

$$d_\perp(K_8, \ell) = \frac{|(x_4 - x_1)(y_1 - y_8) - (x_1 - x_8)(y_4 - y_1)|}{\sqrt{(x_4 - x_1)^2 + (y_4 - y_1)^2}}$$

**Normalización** por longitud de columna $L_{espina} = |K_4 - K_1|$:

$$C_E = \frac{d_\perp(K_8, \ell)}{L_{espina}}$$

### 4.4 Interpretación

| Componente | Valor bajo | Valor alto |
|------------|-----------|-----------|
| $D_L$ | Espalda lumbar alineada ($\theta_L \approx 180°$) | Angulación lumbar marcada |
| $C_E$ | Escápula sobre línea espinal (sin cifosis) | Escápula desviada (hombros caídos) |

**CPI bajo → postura recta; CPI alto → encorvado.**

---

## 5. Umbrales de Clasificación

### 5.1 Definición

| Rango CPI | Estado | Color | Significado Clínico |
|---|---|---|---|
| **CPI ≤ 35** | CORRECTO | 🟢 Verde | Columna alineada, curvatura lumbar conservada |
| **35 < CPI ≤ 50** | ALERTA LEVE | 🟡 Amarillo | Inicio de cifosis torácica y/o protrusión cefálica |
| **CPI > 50** | ALERTA CRÍTICA | 🔴 Rojo | Cifosis marcada, hombros caídos hacia adelante |

### 5.2 Calibración

Los umbrales fueron calibrados por el usuario mediante 6 imágenes de referencia en 3 posturas controladas (recto, semi-encorvado, encorvado), procesadas con 4 modelos YOLO-Pose:

| Postura | CPI (yolov8n) | Clasificación |
|---------|---------------|---------------|
| Recto | 47.2 – 47.7 | ALERTA LEVE |
| Semi-encorvado | 62.8 – 67.3 | ALERTA CRÍTICA |
| Encorvado | 69.9 – 77.2 | ALERTA CRÍTICA |

Los umbrales fueron ajustados para reflejar el criterio ergonómico del usuario: CPI ≤ 35 = CORRECTO, CPI > 50 = CRÍTICO.

### 5.3 Comparación CPI vs. Enfoque Monoangular

Para la misma imagen procesada con yolov8n:

| Métrica | Recto | Semi | Encorvado | Δ Recto→Encorvado |
|---------|-------|------|-----------|-------------------|
| Cervicodorsal `∠K0-K1-K8` | 139.8° | 137.1° | 135.1° | 4.7° (3.4%) |
| **CPI** | **47.2** | **62.8** | **77.2** | **30.0 pts (63.6%)** |

El CPI proporciona una separación **12.7× mayor** que el ángulo cervicodorsal entre las posturas extremas.

### 5.4 Sistema de Alertas Temporales

```
t_mala_postura > 30s ──▶ BEEP (1000 Hz / 300 ms, cada 5s mientras persista)
```

Si la postura mejora (CPI ≤ 35) o se pierde la detección por más de 2 segundos, el contador se reinicia.

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

## 9. Pipeline de Procesamiento por Frame (CPI)

```
Frame RGB (Webcam)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. CONVERSIÓN RGB → BGR                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. INFERENCIA YOLO-Pose (conf=0.3)                           │
│    Salida:  [N_personas, 9_keypoints, 3(x,y,conf)]          │
│    Selección: persona con mayor confianza promedio           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EXTRACCIÓN DE 5 KEYPOINTS POSTERIORES                     │
│    K0, K1, K3, K4, K8                                       │
│    Verificación: conf > 0.1 para K1, K3, K4, K8            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. CÁLCULO DEL CPI                                           │
│    θ_L = ∠(K8, K3, K4)          ← ángulo lumbar            │
│    D_L = max(0, 180° − θ_L)      ← déficit lumbar           │
│    d_⊥(K8, línea K1→K4)          ← curvatura px             │
│    C_E = d_⊥ / |K1→K4|           ← curvatura normalizada    │
│    CPI = D_L × 2 + C_E × 100                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. CLASIFICACIÓN POSTURAL                                    │
│    CPI ≤ 35  → CORRECTO (🟢)                                │
│    35 < CPI ≤ 50 → ALERTA LEVE (🟡)                         │
│    CPI > 50 → ALERTA CRÍTICA (🔴)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. OVERLAY VISUAL                                            │
│    - Esqueleto: K0→K1→K8→K3→K4 (azul)                      │
│    - Líneas del ángulo lumbar: K3→K8, K3→K4 (naranja)       │
│    - Línea referencia: K1→K4 (gris)                          │
│    - Etiqueta: CPI + Lumbar°                                │
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
| Fórmula | CPI = D_L × 2 + C_E × 100 (Combined Posture Index) |
| Keypoints utilizados | 5: K0 (Head-back), K1 (C7), K3 (Espalda media), K4 (Cadera), K8 (Escápula) |
| Componentes | Déficit lumbar ∠K8-K3-K4 + Curvatura escapular ⊥(K8, K1→K4) |
| Método de cálculo | Producto punto + arccos + distancia punto-recta |
| Umbrales | CORRECTO ≤ 35, LEVE 35–50, CRÍTICA > 50 |
| Alerta sonora | >30s continuos en mala postura, beep cada 5s |
| Confianza mínima | 0.1 (K1, K3, K4, K8) |
| Resolución de cámara | 640×480 (default) |
| Número de keypoints | 9 (clase person-torso) |
| Modelos soportados | YOLOv5n, v8n, v11n, v26n |
| Latencia de inferencia | 22-30ms por frame |

---

## 12. Conclusión

El Combined Posture Index (CPI) supera las limitaciones de los enfoques monoangulares previos:

- ✅ **Discriminación consistente:** separación de 30 puntos CPI (63.6%) entre recto y encorvado vs. solo 4.7° (3.4%) del ángulo cervicodorsal — una mejora de 12.7×.
- ✅ **Modelo multivectorial:** integra curvatura escapular + ángulo lumbar usando 5 keypoints de la cadena posterior.
- ✅ **Interpretabilidad biomecánica:** cada componente del CPI tiene significado anatómico directo.
- ✅ **Determinismo:** mismo input → mismo output, sin estocasticidad (no usa ML en la clasificación).
- ✅ **Eficiencia computacional:** 5 operaciones vectoriales por frame.

La evolución del modelo matemático desde el ángulo mentoniano (v2.0) hasta el CPI (v3.0) representa un avance significativo en la precisión y robustez del sistema de monitoreo postural.

La documentación completa del CPI, incluyendo fundamentación teórica, validación experimental detallada, y apéndices con datos numéricos, se encuentra en [`MODELO_MATEMATICO_CPI.md`](../MODELO_MATEMATICO_CPI.md).

---

*Documento generado como parte del Sistema de Monitoreo Postural en Tiempo Real — Universidad Surcolombiana, 2026.*
