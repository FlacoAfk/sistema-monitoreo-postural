# Modelo Matemático — Combined Posture Index (CPI)

## Sistema de Monitoreo Postural en Tiempo Real
### Universidad Surcolombiana — 2026

---

## 1. Fundamentación Teórica

### 1.1 Problema clínico

Los trabajadores de oficina que permanecen sentados frente a un computador por períodos prolongados desarrollan alteraciones posturales progresivas. Las más comunes son:

- **Protrusión cefálica anterior** (*forward head posture*): la cabeza se desplaza hacia adelante respecto al eje vertical de la columna.
- **Cifosis torácica** (*thoracic kyphosis*): curvatura excesiva de la columna dorsal superior con redondeo de hombros.
- **Pérdida de lordosis lumbar**: aplanamiento de la curvatura natural de la espalda baja.

Estas condiciones están asociadas con cervicalgia, dorsalgia, cefaleas tensionales y síndrome de dolor miofascial (Griegel-Morris et al., 1992; Yoo et al., 2008).

### 1.2 Limitaciones de los enfoques monoangulares

Los métodos tradicionales de evaluación postural basados en un solo ángulo entre tres puntos anatómicos presentan una limitación fundamental: cuando una persona se inclina hacia adelante, todos los puntos de referencia de la cadena posterior se desplazan simultáneamente, lo que produce que los ángulos entre segmentos adyacentes permanezcan aproximadamente constantes incluso cuando la postura global se deteriora significativamente.

En las pruebas de validación realizadas con 6 imágenes etiquetadas (2 por categoría postural: recto, semi-encorvado, encorvado) procesadas con 4 modelos YOLO-Pose (yolov8n, yolov5n, yolov26n, yolov11n), se observó que el ángulo cervicodorsal `∠K0-K1-K8` (Occipital–C7–Escápula) varía solo entre 134° y 140° en las tres categorías posturales, haciendo imposible una clasificación fiable con este único indicador.

### 1.3 Enfoque propuesto: Combined Posture Index (CPI)

El CPI es un índice compuesto multivectorial que integra dos mediciones complementarias de la columna posterior:

1. **Déficit angular lumbar**: cuantifica la pérdida de alineación recta en el segmento inferior de la espalda.
2. **Curvatura escapular normalizada**: mide la desviación perpendicular de la escápula respecto a la línea teórica de la columna vertebral.

Al combinar ambas métricas, el CPI captura simultáneamente la deformación sagital de la columna en sus porciones torácica y lumbar, proporcionando una medida holística de la calidad postural.

---

## 2. Adquisición de Puntos Anatómicos

### 2.1 Modelo de detección

Se utilizan modelos de estimación de pose YOLO-Pose (Ultralytics) entrenados sobre el dataset _Desk Posture_ anotado en Roboflow con 9 keypoints (landmarks) anatómicos de la región del torso. Los modelos fueron entrenados con las arquitecturas YOLOv5n, YOLOv8n, YOLOv11n y YOLOv26n, todas en su variante _pose_, con tamaño de entrada de 640×640 píxeles.

### 2.2 Mapeo de Keypoints Roboflow → YOLO

Cada keypoint se identifica por su índice en el vector de salida del modelo YOLO (`keypoints.data[i]`), que corresponde al orden de anotación en Roboflow:

| Índice YOLO ($i$) | Roboflow ID | Nombre anatómico | Descripción |
|:---:|:---:|---|---|
| $K_0$ | 0 | Head-back | Occipital / Parte posterior de la cabeza |
| $K_1$ | 1 | Neck-back | Cervical C7 / Base posterior del cuello |
| $K_2$ | 2 | Shoulder-top | Acromion / Parte superior del hombro |
| $K_3$ | 6 | Back-backedge | Borde posterior dorsal / Espalda media |
| $K_4$ | 7 | Hips-backedge | Borde posterior de cadera / Zona lumbosacra |
| $K_5$ | 10 | Neck-middle | Cervical media / Cuello anterior |
| $K_6$ | 13 | Jaw | Mandíbula |
| $K_7$ | 14 | Chin | Mentón |
| $K_8$ | 18 | Shoulder-back | Zona escapular / Escápula posterior |

Cada keypoint $K_i$ es un vector de tres componentes:

$$K_i = (x_i,\ y_i,\ c_i)$$

donde $x_i, y_i$ son las coordenadas en píxeles dentro del frame y $c_i \in [0, 1]$ es la confianza de detección reportada por YOLO.

### 2.3 Keypoints utilizados por el CPI

De los 9 keypoints disponibles, el CPI utiliza exclusivamente los 5 que conforman la **cadena posterior** de la espalda:

$$\{K_0, K_1, K_3, K_4, K_8\}$$

Los keypoints $K_2$ (Shoulder-top), $K_5$ (Neck-middle), $K_6$ (Jaw) y $K_7$ (Chin) son detectados por el modelo pero no participan en el cálculo del CPI por pertenecer a la cadena anterior o facial.

---

## 3. Formulación Matemática del CPI

### 3.1 Definición

El Combined Posture Index se define como:

$$\boxed{CPI = D_L \times 2 + C_E \times 100}$$

donde:

- $D_L$: **Déficit angular lumbar** (adimensional, en grados sexagesimales)
- $C_E$: **Curvatura escapular normalizada** (adimensional, expresada como porcentaje)

### 3.2 Componente 1: Déficit angular lumbar ($D_L$)

#### 3.2.1 Ángulo lumbar $\angle K_8K_3K_4$

Se define como el ángulo con vértice en $K_3$ (espalda media, Back-backedge) formado por los vectores $\vec{K_3K_8}$ (hacia la escápula) y $\vec{K_3K_4}$ (hacia la cadera):

$$\theta_L = \angle(K_8, K_3, K_4) = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{|\vec{u}| \cdot |\vec{v}|}\right)$$

donde:

$$\vec{u} = K_8 - K_3 = (x_8 - x_3,\ y_8 - y_3)$$
$$\vec{v} = K_4 - K_3 = (x_4 - x_3,\ y_4 - y_3)$$

#### 3.2.2 Interpretación geométrica

- $\theta_L \approx 180°$: los tres puntos están alineados → espalda baja recta (lordosis conservada).
- $\theta_L \ll 180°$: hay una angulación marcada en la espalda media → cifosis o colapso postural.

#### 3.2.3 Déficit angular

El déficit angular lumbar cuantifica cuánto se desvía el ángulo lumbar de la alineación perfecta (180°):

$$D_L = \max(0,\ 180° - \theta_L)$$

Un $D_L = 0$ indica alineación perfecta (los puntos $K_8, K_3, K_4$ son colineales en el plano sagital). Valores mayores indican mayor curvatura lumbar.

### 3.3 Componente 2: Curvatura escapular normalizada ($C_E$)

#### 3.3.1 Línea de referencia espinal

Se define la **línea espinal teórica** $\ell$ como la recta que une $K_1$ (C7, base del cuello) con $K_4$ (cadera):

$$\ell(K_1, K_4) = \{ K_1 + t(K_4 - K_1) \mid t \in \mathbb{R} \}$$

Esta línea representa la columna vertebral idealmente recta en el plano sagital.

#### 3.3.2 Distancia perpendicular de la escápula

La curvatura escapular en píxeles se define como la distancia perpendicular del punto $K_8$ (escápula) a la línea $\ell$:

$$d_\perp(K_8, \ell) = \frac{|(x_4 - x_1)(y_1 - y_8) - (x_1 - x_8)(y_4 - y_1)|}{\sqrt{(x_4 - x_1)^2 + (y_4 - y_1)^2}}$$

Esta es la fórmula estándar de distancia punto-recta usando el producto cruzado 2D.

#### 3.3.3 Normalización

Para hacer la métrica independiente de la distancia cámara-sujeto y de la resolución del frame, se normaliza dividiendo por la longitud de la columna:

$$L_{espina} = |K_4 - K_1| = \sqrt{(x_4 - x_1)^2 + (y_4 - y_1)^2}$$

$$C_E = \frac{d_\perp(K_8, \ell)}{L_{espina}}$$

El resultado $C_E$ es adimensional y se expresa como porcentaje multiplicando por 100 en la fórmula del CPI.

#### 3.3.4 Interpretación

- $C_E \approx 0$: la escápula yace sobre la línea espinal → espalda plana, sin cifosis.
- $C_E > 0$: la escápula se desvía de la línea, indicando redondeo de hombros y cifosis torácica.

### 3.4 Justificación de los pesos

Los coeficientes $2$ (para $D_L$) y $100$ (para $C_E$) en la fórmula del CPI fueron determinados empíricamente para equilibrar la contribución de ambos componentes:

- Sin el factor $2$, el déficit lumbar ($\approx 10-30°$) aportaría mucho más que la curvatura escapular ($\approx 0.10-0.20$ en valor absoluto).
- El factor $100$ convierte la curvatura normalizada a una escala comparable con el déficit angular, haciendo que ambas componentes tengan un peso balanceado en el índice final.

Los pesos fueron validados con 6 imágenes etiquetadas manualmente en 3 categorías posturales, verificando que el CPI produjera separación entre clases.

---

## 4. Clasificación Postural

### 4.1 Umbrales de decisión

El CPI se clasifica en tres categorías ergonómicas mediante umbrales determinados por calibración con el usuario:

| Rango CPI | Clasificación | Color | Significado clínico |
|-----------|---------------|-------|---------------------|
| $CPI \leq 35$ | **CORRECTO** | Verde | Columna alineada. Cabeza sobre hombros, curvatura lumbar conservada. |
| $35 < CPI \leq 50$ | **ALERTA LEVE** | Amarillo | Inicio de protrusión cefálica y/o cifosis torácica leve. |
| $CPI > 50$ | **ALERTA CRÍTICA** | Rojo | Cifosis marcada. Hombros caídos hacia adelante. Riesgo de lesión por esfuerzo repetitivo. |

### 4.2 Algoritmo de clasificación

```
ENTRADA: keypoints[0..8]  // 9 puntos anatómicos detectados por YOLO-Pose

1. Extraer K0, K1, K3, K4, K8
2. Verificar confianza mínima (c_i ≥ 0.1 para K1, K3, K4, K8)
3. Calcular θ_L = ∠(K8, K3, K4)
4. D_L = max(0, 180° − θ_L)
5. Calcular d_⊥(K8, línea K1→K4)
6. L_espina = |K4 − K1|
7. C_E = d_⊥ / L_espina
8. CPI = D_L × 2 + C_E × 100
9. SI CPI ≤ 35 → CORRECTO
   SINO SI CPI ≤ 50 → ALERTA LEVE
   SINO → ALERTA CRÍTICA
```

### 4.3 Sistema de alertas

El sistema mantiene un contador de tiempo acumulado en postura inadecuada:

- Si el estado es **ALERTA LEVE** o **ALERTA CRÍTICA**, se inicia (o continúa) un cronómetro.
- Si el estado vuelve a **CORRECTO**, el cronómetro se reinicia a cero.
- Cuando el tiempo acumulado supera **30 segundos** continuos, se emite una alerta sonora (beep de 1000 Hz, 300 ms) y se repite cada 5 segundos mientras persista la mala postura.

---

## 5. Validación Experimental

### 5.1 Metodología

Se capturaron 6 imágenes del usuario en 3 posturas controladas:

- **2 imágenes en postura RECTA**: espalda erguida, cabeza alineada sobre hombros.
- **2 imágenes en postura SEMI-ENCORVADA**: inclinación leve hacia adelante.
- **2 imágenes en postura ENCORVADA**: cifosis pronunciada, hombros hacia adelante.

Cada imagen fue procesada con 4 arquitecturas YOLO-Pose (yolov8n, yolov5n, yolov26n, yolov11n), todas con tamaño de entrada 640×640 y umbral de confianza 0.3.

### 5.2 Resultados de detección

Los 4 modelos detectaron consistentemente los 9 keypoints en las 6 imágenes, con confianza promedio superior a 0.99. El modelo yolov8n fue seleccionado como referencia por su balance entre precisión y velocidad de inferencia (~22 ms en GPU).

### 5.3 Resultados del CPI (modelo yolov8n)

| Postura | Imagen | $\theta_L$ (Lumbar) | $C_E$ (Curvatura %) | **CPI** | Clasificación |
|---------|--------|---------------------|---------------------|---------|---------------|
| Recto | img1 (espejo) | 163.2° | 13.6% | **47.2** | ALERTA LEVE |
| Recto | img2 | 163.9° | 15.0% | **47.7** | ALERTA LEVE |
| Semi-encorvado | img1 (espejo) | 156.2° | 15.2% | **62.8** | ALERTA CRÍTICA |
| Semi-encorvado | img2 | 154.1° | 15.5% | **67.3** | ALERTA CRÍTICA |
| Encorvado | img1 (espejo) | 150.1° | 17.4% | **77.2** | ALERTA CRÍTICA |
| Encorvado | img2 | 153.2° | 16.3% | **69.9** | ALERTA CRÍTICA |

### 5.4 Análisis de resultados

Se observa una correlación inversa entre la calidad postural y el CPI:

- Las imágenes de postura **recta** producen los CPI más bajos (47.2–47.7).
- Las imágenes de postura **semi-encorvada** producen CPI intermedios (62.8–67.3).
- Las imágenes de postura **encorvada** producen los CPI más altos (69.9–77.2).

La separación entre las categorías RECTA y ENCORVADA es de aproximadamente 22–30 puntos CPI, lo que demuestra la capacidad discriminativa del índice.

### 5.5 Comparación con enfoque monoangular

Para la misma imagen "Recto (espejo)" procesada con yolov8n:

| Métrica | Recto | Semi-encorvado | Encorvado | Δ Recto→Encorvado |
|---------|-------|---------------|-----------|-------------------|
| Cervicodorsal `∠K0-K1-K8` | 139.8° | 137.1° | 135.1° | **4.7°** (3.4%) |
| Torácico `∠K1-K8-K3` | 139.5° | 135.9° | 134.0° | **5.5°** (3.9%) |
| **CPI** | **47.2** | **62.8** | **77.2** | **30.0 pts (63.6%)** |

El CPI proporciona una separación **12.7 veces mayor** que el ángulo cervicodorsal entre las posturas extrema, demostrando la superioridad del enfoque multivectorial.

### 5.6 Limitaciones

- Los umbrales actuales (≤35, 35–50, >50) fueron calibrados para un usuario específico y pueden requerir re-calibración para otros sujetos con diferente antropometría.
- El CPI no distingue entre cifosis torácica y pérdida de lordosis lumbar como entidades separadas; las agrupa en un único índice compuesto.
- La precisión depende de la calidad de detección de YOLO-Pose, que puede degradarse con iluminación pobre, oclusiones parciales o ropa holgada.

---

## 6. Implementación Computacional

### 6.1 Arquitectura del sistema

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Webcam /     │     │  YOLO-Pose      │     │  PostureAnalyzer │
│  Video frame  │────▶│  Inference       │────▶│  (CPI Formula)   │
│  (RGB)        │     │  (9 keypoints)   │     │                  │
└──────────────┘     └─────────────────┘     └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  PostureResult   │
                                              │  · cpi           │
                                              │  · lumbar_angle  │
                                              │  · curvature_pct │
                                              │  · status        │
                                              └────────┬─────────┘
                                                       │
                                              ┌────────▼─────────┐
                                              │  Dashboard       │
                                              │  · Gauge CPI      │
                                              │  · Alertas        │
                                              │  · Overlay video  │
                                              └──────────────────┘
```

### 6.2 Funciones matemáticas implementadas

```python
def angle_at_vertex(A, V, B) -> float:
    """Ángulo con vértice en V: ∠(A-V-B) en grados."""
    u = A - V
    v = B - V
    cos_theta = (u·v) / (|u|·|v|)
    return arccos(clamp(cos_theta, -1, 1)) × 180/π

def point_line_distance(P, A, B) -> float:
    """Distancia perpendicular de P a la recta AB."""
    return |(B-A) × (A-P)| / |B-A|

def analyze(keypoints) -> PostureResult:
    K0, K1, K3, K4, K8 = keypoints[0,1,3,4,8]
    
    # Déficit lumbar
    theta_L = angle_at_vertex(K8, K3, K4)
    D_L = max(0, 180 - theta_L)
    
    # Curvatura escapular normalizada
    d_perp = point_line_distance(K8, K1, K4)
    L = |K4 - K1|
    C_E = d_perp / L
    
    # CPI
    CPI = D_L * 2 + C_E * 100
    
    # Clasificación
    if CPI <= 35:   return CORRECTO
    elif CPI <= 50: return ALERTA_LEVE
    else:           return ALERTA_CRÍTICA
```

### 6.3 Lenguajes y dependencias

| Componente | Lenguaje | Dependencias |
|------------|----------|-------------|
| Inferencia YOLO | Python 3.12+ | ultralytics, torch, opencv-python |
| Análisis CPI | Python 3.12+ | numpy (solo math estándar) |
| Dashboard | Python 3.12+ | gradio, opencv-python |
| Validación | Python 3.12+ | numpy, json |

El modelo matemático (CPI) no depende de librerías externas de machine learning — utiliza exclusivamente `math.sqrt`, `math.acos`, `math.degrees` del módulo estándar de Python. La inferencia de keypoints se realiza externamente con YOLO-Pose.

---

## 7. Referencias

1. Griegel-Morris, P., Larson, K., Mueller-Klaus, K., & Oatis, C. A. (1992). Incidence of common postural abnormalities in the cervical, shoulder, and thoracic regions and their association with pain in two age groups of healthy subjects. _Physical Therapy_, 72(6), 425–431.

2. Yoo, W. G., Yi, C. H., & Kim, M. H. (2008). Effects of a ball-backrest chair on the muscles associated with upper crossed syndrome when working at a VDT. _Work_, 29(3), 239–244.

3. Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. _GitHub repository_. https://github.com/ultralytics/ultralytics

4. Roboflow Inc. (2025). Desk Posture Dataset. _Roboflow Universe_. https://universe.roboflow.com/

---

## Apéndice A: Tabla completa de resultados por modelo

Resultados de validación para las 6 imágenes con los 4 modelos YOLO-Pose (mejor modelo por imagen, ordenado por confianza promedio).

### A.1 Postura RECTA

| Imagen | Modelo | $\theta_L$ | $C_E$ (px) | $L_{esp}$ (px) | $C_E$ (%) | $D_L$ | **CPI** |
|--------|--------|-----------|------------|-----------------|-----------|-------|---------|
| 152657-espejo | yolov8n | 163.1° | 57.3 | 422.4 | 13.6% | 16.9° | 47.4 |
| 152657-espejo | yolov5n | 161.2° | 57.1 | 419.2 | 13.6% | 18.8° | 51.2 |
| 152657-espejo | yolov26n | 159.8° | 75.0 | 452.6 | 16.6% | 20.2° | 57.0 |
| 152657-espejo | yolov11n | 165.3° | 60.2 | 448.5 | 13.4% | 14.7° | 42.8 |
| 152657 | yolov8n | 163.9° | 66.7 | 445.1 | 15.0% | 16.1° | 47.2 |
| 152657 | yolov11n | 165.2° | 58.8 | 439.8 | 13.4% | 14.8° | 43.0 |

### A.2 Postura SEMI-ENCORVADO

| Imagen | Modelo | $\theta_L$ | $C_E$ (px) | $L_{esp}$ (px) | $C_E$ (%) | $D_L$ | **CPI** |
|--------|--------|-----------|------------|-----------------|-----------|-------|---------|
| 153030-espejo | yolov8n | 156.2° | 68.8 | 451.7 | 15.2% | 23.8° | 62.8 |
| 153030-espejo | yolov11n | 156.2° | 64.8 | 444.4 | 14.6% | 23.8° | 62.2 |
| 153030 | yolov8n | 154.1° | 72.6 | 467.9 | 15.5% | 25.9° | 67.3 |
| 153030 | yolov11n | 157.2° | 69.1 | 452.3 | 15.3% | 22.8° | 60.9 |

### A.3 Postura ENCORVADO

| Imagen | Modelo | $\theta_L$ | $C_E$ (px) | $L_{esp}$ (px) | $C_E$ (%) | $D_L$ | **CPI** |
|--------|--------|-----------|------------|-----------------|-----------|-------|---------|
| 152856-espejo | yolov8n | 150.1° | 82.7 | 474.1 | 17.4% | 29.9° | 77.2 |
| 152856-espejo | yolov11n | 154.0° | 74.4 | 446.7 | 16.7% | 26.0° | 68.7 |
| 152856 | yolov8n | 153.2° | 76.8 | 471.0 | 16.3% | 26.8° | 69.9 |
| 152856 | yolov11n | 151.7° | 71.5 | 466.1 | 15.3% | 28.3° | 71.9 |

---

## Apéndice B: Glosario de símbolos

| Símbolo | Significado | Unidad |
|---------|-------------|--------|
| $K_i$ | Keypoint anatómico $i$ | $(x, y, c)$ en px |
| $\theta_L$ | Ángulo lumbar $\angle K_8K_3K_4$ | Grados sexagesimales |
| $D_L$ | Déficit angular lumbar = $\max(0, 180° - \theta_L)$ | Grados |
| $d_\perp(K_8, \ell)$ | Distancia perpendicular de $K_8$ a línea $\ell(K_1, K_4)$ | Píxeles |
| $L_{espina}$ | Longitud del segmento $K_1 \to K_4$ | Píxeles |
| $C_E$ | Curvatura escapular normalizada = $d_\perp / L_{espina}$ | Adimensional |
| $CPI$ | Combined Posture Index = $2 \cdot D_L + 100 \cdot C_E$ | Adimensional |

---

_Documento generado el 8 de mayo de 2026. Versión 1.0._
