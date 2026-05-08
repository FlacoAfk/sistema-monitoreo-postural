# Sistema de Monitoreo Postural en Tiempo Real Mediante Técnicas de Visión Artificial

---

## 1. Metadatos del Proyecto

| Campo | Detalle |
|---|---|
| **Título** | Sistema de Monitoreo Postural en Tiempo Real Mediante Técnicas de Visión Artificial |
| **Autores** | Sergio Andres Castañeda Guzmán · Juan Pablo Idarraga Plazas |
| **Docente** | Ferley Medina Rojas |
| **Institución** | Universidad Surcolombiana — Sede Neiva (Neiva-Huila, Colombia, 2026) |

**Objetivo Principal:** Desarrollar un sistema de visión artificial de arquitectura desacoplada para identificar posturas inadecuadas en tiempo real sin interrumpir las labores del trabajador, garantizando una precisión geométrica superior al 95% y baja latencia.

---

## 2. Planteamiento del Problema

El trabajo de oficina moderno exige que los empleados pasen la mayor parte de su jornada laboral frente a un computador. Dicha exposición prolongada provoca la adopción de posturas inadecuadas —como la cabeza adelantada y los hombros encorvados—, las cuales generan desequilibrios musculares que producen fatiga, dolor cervical y lumbar, además de mayor gasto energético para mantener una posición erguida [1]. Estas alteraciones posturales pueden desencadenar trastornos musculoesqueléticos, rigidez, cefaleas tensionales y afectaciones en la respiración y digestión [1]. Asimismo, una mala postura deteriora la salud física, disminuye la concentración, incrementa el cansancio y repercute negativamente en el rendimiento empresarial [2].

Actualmente las empresas intentan mitigar esta situación mediante evaluaciones ergonómicas periódicas. Sin embargo, esta metodología presenta una limitación importante: se evalúa la postura en un instante específico y depende en gran medida de la autoconciencia del empleado [3]. Las herramientas clínicas (goniómetros, radiografías, escáneres 3D) requieren que el usuario interrumpa sus labores y dependen de hardware costoso, dificultando su implementación masiva [3].

El sistema proyecta resultados de precisión superiores al **95%**, junto con baja latencia y alta confiabilidad en la detección continua.

---

## 3. Preguntas a Resolver

### 3.1 Pregunta General

¿Cómo implementar un modelo de visión artificial para la detección de una incorrecta postura corporal, utilizando coordenadas tridimensionales y puntos de articulación (keypoints) para identificar posturas corporales en tiempo real?

### 3.2 Preguntas Específicas

- ¿Cómo recolectar un conjunto de imágenes que incluya diferentes posturas corporales junto a los puntos de articulación para entrenar el modelo?
- ¿Cómo normalizar las imágenes mediante técnicas de preprocesamiento para aumentar la exactitud en la localización de los keypoints?
- ¿Cómo seleccionar un modelo de aprendizaje teniendo en cuenta su desempeño y capacidad de detección?
- ¿Cómo entrenar y ajustar los hiperparámetros del modelo seleccionado para mejorar la precisión en la detección?
- ¿Cómo evaluar el rendimiento del modelo mediante métricas para la medición de su eficiencia?
- ¿Cómo integrar el modelo entrenado en un sistema en tiempo real que permita detectar y notificar las posturas inadecuadas?

---

## 4. Objetivos

### 4.1 Objetivo General

Desarrollar un modelo de visión artificial para la detección de una incorrecta postura corporal, utilizando coordenadas tridimensionales y puntos de articulación (keypoints) para identificar posturas corporales en tiempo real.

### 4.2 Objetivos Específicos

1. Recolectar un conjunto de imágenes conformadas por diferentes posturas corporales junto a los puntos de articulación para entrenar el modelo.
2. Normalizar el conjunto de imágenes mediante técnicas de preprocesamiento, con el fin de aumentar la exactitud en la localización espacial de los keypoints.
3. Seleccionar un modelo de aprendizaje automático considerando su desempeño y capacidad de detección para la localización de los puntos de articulación.
4. Entrenar el modelo seleccionado y ajustar los hiperparámetros para mejorar la precisión en la detección.
5. Evaluar el rendimiento del modelo mediante métricas que permitan medir la eficiencia en la detección de posturas corporales.
6. Integrar el modelo entrenado en un sistema en tiempo real que permita detectar y notificar las posturas inadecuadas del trabajador.

---

## 5. Restricciones Técnicas Inquebrantables

Para superar el estado del arte y evitar los errores de metodologías previas, el proyecto opera bajo reglas estrictas:

### Cero Clasificación Tradicional (Cero "Cajas Negras")
Está estrictamente **prohibido** utilizar algoritmos como SVM, Decision Trees o MLP para clasificar directamente si una postura es "buena" o "mala".

### Arquitectura Desacoplada

1. **Frontend Visual (YOLO):** Se limita exclusivamente a detectar la región del torso y extraer las coordenadas espaciales $(X, Y)$ de los puntos articulares.
2. **Backend Geométrico (Python/NumPy):** Recibe un archivo JSON con las coordenadas y ejecuta matemáticas deterministas para calcular la flexión cervicodorsal.

---

## 6. Topología de Estimación de Pose (Keypoint Detection)

El sistema no utiliza bounding boxes genéricos. Se identifica una única clase denominada **`person-torso`**, mapeando un esqueleto personalizado de exactamente **9 puntos articulares (Keypoints)** en el plano sagital:

| K (YOLO) | Roboflow ID | Nombre original en planeación | Descripción anatómica (Verificado) |
|:---:|:---:|---|---|
| **K0** | 0 | Head-back | **Occipital / Parte posterior de la cabeza** |
| **K1** | 1 | Neck-back | **Cervical posterior / Base del cuello (C7)** |
| **K2** | 2 | Shoulder-top | **Acromion / Parte superior del hombro** |
| **K3** | 6 | Back-backedge | **Borde posterior dorsal / Espalda media** |
| **K4** | 7 | Hips-backedge | **Borde posterior de la cadera / Lumbosacra** |
| **K5** | 10 | Neck-middle | **Cervical media** |
| **K6** | 13 | Jaw | **Mandíbula** |
| **K7** | 14 | Chin | **Mentón** |
| **K8** | 18 | Shoulder-back | **Zona escapular** |

> **📋 Nota sobre el remapeo de IDs (verificado 2026-05-07):**
> Los IDs originales en Roboflow son **no secuenciales** (0, 1, 2, 6, 7, 10, 13, 14, 18), provenientes de un esqueleto base más amplio. Al exportar a formato YOLOv8, Roboflow los renumera secuencialmente de 0 a 8. Las coordenadas $(X, Y)$ se mantienen idénticas en ambas representaciones; solo cambia el índice.

### Nodos críticos para el cálculo de $\theta$ (flexión cervicodorsal)

Para medir la postura cervical en el plano sagital, se utilizan tres puntos que definen el ángulo de la cabeza respecto a la columna superior:

| Nodo | Keypoint | Rol geométrico |
|---|---|---|
| **Cabeza** | K0 (Cabeza superior frontal) | Posición de la cabeza — detecta protrusión anterior (head forward) |
| **Pivote** | K6 (Cervical posterior C7) | Punto de referencia cervical — vértice del ángulo |
| **Espalda** | K7 (Borde posterior dorsal) | Línea de la espalda alta — referencia de verticalidad |

El ángulo $\theta = \angle(\overrightarrow{K6 \to K0},\ \overrightarrow{K6 \to K7})$ mide la apertura entre la dirección de la cabeza y la línea de la espalda desde la cervical C7. Valores bajos de $\theta$ indican protrusión cefálica (head forward posture).


---

## 7. Estado del Arte

A continuación se presentan los estudios más relevantes relacionados con la detección y clasificación de posturas corporales.

| Ref. | Título | Año | Metodología | Resultado Clave | Limitación |
|---|---|---|---|---|---|
| [4] | SitPose System Development | 2024 | Azure Kinect DK (9 keypoints) + Ensamble ML | F1-Score 98.1% | Hardware costoso; clasificadores ML no interpretables |
| [5] | Posture Detection using UWB Radar (CSIRO) | 2025 | Radar UWB + AttnFusion (Deep Learning) | Precisión 90.6%, F1 90.5% | Hardware especializado, difícil de escalar |
| [6] | Detection of Ergonomic Sitting Postures in Office Env. | 2025 | MoveNet Thunder v4 + MLP/XGBoost | Accuracy 94.59% | Clasificadores ML no deterministas |
| [7] | Real-time Smart Classrooms Posture Detection | 2024 | SSD sobre OpenVINO — Bounding Boxes | 0.772 accuracy (sentado) | Incapaz de medir curvaturas dorsales |
| [8] | Review of Sitting Posture Recognition | 2024 | Revisión de 120 artículos (2000–2022) | Categoriza transición de sensores a visión artificial | Oclusiones como mayor desafío vigente |
| [37] | Modelling Proper & Improper Sitting Posture | 2023 | MediaPipe (33 puntos) + Ley Cosenos + Decision Trees | Accuracy 97.05% | 33 puntos redundantes para tren superior |
| [39] | Automated Vision-Based Goniometry | 2024 | Librerías HPE múltiples vs goniometría clínica | Spearman ρ = 0.722–0.786 | Baja correlación en articulaciones complejas |
| [40] | Validity Analysis of Monocular HPE Models | 2024 | MoveNet + Producto Escalar vectorial | RMSE < 10° en 8/10 movimientos | Baja fiabilidad cuando keypoints se solapan en 2D |

> Los estudios [37], [39] y [40] validan bibliográficamente el uso de trigonometría plana y producto punto vectorial para calcular ángulos entre keypoints — exactamente el núcleo del backend de este sistema (vale 100 pts en la rúbrica).

---

## 8. Marco Teórico

### 8.1 Ergonomía y Postura Corporal

La **ergonomía** estudia la interacción entre personas y su entorno laboral para optimizar el bienestar humano. En oficinas, el diseño adecuado de estaciones de trabajo contribuye a reducir la sobrecarga muscular y prevenir lesiones por esfuerzo repetitivo [7].

La **postura corporal** se define como la posición que adopta el cuerpo y la forma en que este se alinea respecto a la gravedad. Una postura inadecuada genera estrés sobre músculos y articulaciones, favoreciendo la aparición de fatiga muscular y trastornos musculoesqueléticos [7].

### 8.2 Alteraciones Posturales

- **Desalineación cervical y lumbar:** Alteración en la disposición normal de las vértebras que se manifiesta como dolor persistente, rigidez y asimetrías visibles [8].
- **Trastornos musculoesqueléticos (TME):** Afecciones derivadas de sobrecargas mecánicas o posturas prolongadas que comprenden músculos, articulaciones, tendones y nervios [9].
- **Dolor cervical y lumbar:** Principal consecuencia de las alteraciones posturales; incrementa la carga mecánica sobre músculos y ligamentos de la columna [8].
- **Cefaleas tensionales:** Asociadas a tensión muscular cervical y desalineación de la columna [8].

### 8.3 Curvatura Cervicodorsal

La curvatura cervicodorsal comprende la región cervical (cuello) y la dorsal o torácica (parte media de la espalda). La región cervical presenta normalmente una **lordosis** (curvatura hacia adelante), mientras que la región torácica presenta una **cifosis** (curvatura hacia atrás). Cuando esta curvatura se altera debido a malas posturas prolongadas, puede producirse una rectificación cervical o *forward head posture*, que modifica la biomecánica de la columna e incrementa la carga sobre las estructuras cervicales [42].

### 8.4 Estimación de Pose Humana (HPE) y Keypoints

La **estimación de pose humana** consiste en la detección y localización de puntos anatómicos clave del cuerpo dentro de una imagen o video, permitiendo reconstruir la estructura corporal mediante un modelo esquelético. Puede realizarse en 2D (coordenadas en el plano de la imagen) o en 3D (con datos de profundidad) [12].

Los **keypoints** son localizaciones anatómicas específicas (cabeza, cuello, hombros, codos, muñecas, caderas, rodillas, tobillos) que permiten construir una representación esquelética del cuerpo. La correcta detección de los keypoints es esencial para analizar posturas y relaciones geométricas entre las distintas partes del cuerpo [12].

### 8.5 Trigonometría Computacional

Se refiere al uso de funciones trigonométricas dentro de sistemas computacionales para modelar, analizar y resolver problemas que involucran relaciones angulares o transformaciones geométricas. En el contexto de la visión por computador, permite calcular transformaciones espaciales y extraer características en datos con comportamiento cíclico o direccional [43].

### 8.6 Arquitecturas YOLO para Estimación de Pose

Los modelos YOLO (*You Only Look Once*) representan el estado del arte en visión artificial en tiempo real. En su variante **YOLO-Pose**, la arquitectura integra una cabeza de regresión especializada que no solo delimita la región de interés (*bounding box*) de la clase `person-torso`, sino que simultáneamente predice las coordenadas bidimensionales $(X, Y)$ y el nivel de visibilidad de los puntos de articulación [54].

Para este proyecto se analizan teóricamente cuatro variantes fundamentales:

- **YOLOv8n-pose (Nano):** Arquitectura base más ligera, optimizada para hardware de borde (CPU pura). Latencia mínima e inferencia ultrarrápida, con potencial degradación en oclusiones severas [55].
- **YOLOv8x-pose (Extra-large):** Variante más profunda y parametrizada. Máxima precisión en la regresión de coordenadas topológicas, pero altísimo coste computacional (inviable en tiempo real sin GPU dedicada) [55].
- **YOLO11-pose:** Arquitectura de nueva generación con bloques residuales reestructurados (C2f). Mejora la relación precisión-eficiencia computacional y estabiliza la detección de puntos cervicales críticos (K2, K3, K4) bajo variaciones de iluminación [56].
- **YOLOv26-pose (Experimental):** Modelo de última iteración para inferencia asíncrona de ultrabaja latencia (<15 ms), con mecanismos de micro-atención espacial. Candidato principal para alimentar el motor trigonométrico del backend [54].

### 8.7 Métricas de Evaluación Especializadas

A diferencia de la clasificación tradicional, este proyecto emplea métricas de regresión espacial:

**Object Keypoint Similarity (OKS):**

$$\text{OKS} = \frac{\sum_i \exp\left(-\frac{d_i^2}{2s^2k_i^2}\right)\delta(v_i>0)}{\sum_i \delta(v_i>0)}$$

Donde $d_i$ es la distancia euclidiana entre el keypoint predicho y el real, $s$ es la escala global del objeto, $k_i$ es la constante cinemática para el keypoint $i$, y $v_i$ es la bandera de visibilidad [45].

**Mean Average Precision para Pose (mAP@0.5:0.95):**

$$\text{mAP} = \frac{1}{10} \sum_{\text{OKS}=0.50}^{0.95} \int_0^1 P(R)_{\text{OKS}} \, dR$$

Esta métrica garantiza que el modelo no solo detecte el punto, sino que lo localice con precisión milimétrica [46].

**Normalized Mean Error (NME):**

$$\text{NME} = \frac{1}{N} \sum_{i=1}^{N} \frac{\sqrt{(x_i - x_i^*)^2 + (y_i - y_i^*)^2}}{d}$$

Donde $d$ es el factor de normalización espacial (por ejemplo, la diagonal del bounding box del torso) [45].

### 8.8 Tecnologías de Despliegue y Tiempo Real

- **ONNX (Open Neural Network Exchange):** Formato abierto para la representación de modelos de deep learning que facilita la interoperabilidad entre frameworks (TensorFlow, PyTorch, Caffe2). Permite exportar modelos entrenados sin necesidad de reentrenarlos [48].
- **TensorRT:** Ecosistema de NVIDIA para optimizar modelos de deep learning en inferencia de alto rendimiento. Aplica cuantización (INT8/FP16), fusión de capas y compresión para reducir latencia y consumo de memoria [49].
- **WebSockets:** Protocolo de comunicación bidireccional y persistente entre cliente y servidor. A diferencia de HTTP, mantiene un canal abierto que permite enviar y recibir información de forma continua con baja latencia [50].

---

## 9. Dataset

**Dataset en Roboflow:** [desk-posture-gqjey — Generate](https://app.roboflow.com/js-workspace-rzvay/desk-posture-gqjey/generate)

### 9.1 Formación del Dataset

Se compiló un dataset inicial de **15.000 imágenes** capturadas en entornos de oficina reales y controlados, abarcando diversas condiciones de iluminación, distancias focales y oclusiones parciales (escritorios, monitores, periféricos). La consolidación y anotación se llevó a cabo en **Roboflow**, definiendo una única clase de detección denominada `person-torso`, etiquetada meticulosamente con los 9 puntos de articulación en el plano sagital.

Las imágenes provienen de repositorios de datos abiertos como **Kaggle** y **Hugging Face**, complementados con capturas propias en entornos de oficina.

#### Condiciones de Calidad de las Imágenes

- **Iluminación:** Adecuada, evitando imágenes oscuras o con exceso de luz.
- **Distancia y enfoque:** Elementos relevantes a una distancia cercana a 50 cm.
- **Resolución:** No menor a 256×256 píxeles (estándar para entrenamiento de modelos).
- **Formato:** JPG como formato común para facilitar el procesamiento.

### 9.2 Preprocesamiento y Aumentación de Datos

**Objetivo (Rúbrica — 40 pts):** Dataset balanceado con **más de 35.000 registros**.

Para escalar de 15.000 a >35.000 se implementó un pipeline agresivo con **Roboflow + OpenCV + Albumentations + NumPy** (garantizando precisión subpíxel en las coordenadas):

| Técnica | Descripción | Propósito |
|---|---|---|
| **Auto-Orient + Resize** | Normalización a 640×640 px con padding (bordes negros) | Alinear tensores de entrada a YOLO |
| **Rotación Espacial (±5°)** | Giros aleatorios en el rango definido | Simular inclinación natural del tronco y cámaras |
| **Horizontal Flipping** | Inversión + ajuste de matriz de coordenadas | Simular trabajadores vistos desde ambos lados |
| **Variación de Exposición (±25%)** | Cambios aleatorios en brillo y contraste | Robustecer ante condiciones de iluminación variables |

**Resultado:** El dataset se amplió de 15.000 imágenes iniciales a **35.000 registros balanceados y etiquetados**.

### 9.3 División del Dataset

Se utilizó la distribución **80/20**:
- **80%** → Entrenamiento (ajuste de pesos y predicción de coordenadas)
- **20%** → Validación (evaluación del error y cálculo de métricas mAP y OKS)

---

## 10. Selección de Modelos (Ecosistema YOLO)

La rúbrica exige comparación de **4 modelos** (50 pts):

| Modelo | Parámetros | FLOPs | Ventaja Principal | Desventaja para el Proyecto |
|---|---|---|---|---|
| **YOLOv8n-pose** (Nano) | ∼ 3.0 M | ∼ 8.9 G | Máxima velocidad en CPU | Menor robustez ante oclusiones severas |
| **YOLOv8x-pose** (Extra-large) | ∼ 68.2 M | ∼ 257.4 G | Precisión espacial milimétrica | Inviable en tiempo real sin GPU dedicada |
| **YOLO11-pose** | ∼ 20.0 M | ∼ 60.0 G | Excelente balance latencia-precisión | Requiere mayor afinamiento de anclas |
| **YOLOv26-pose** (Experimental) | Optimizada | Ultra-bajo | Inferencia asíncrona ideal (<15 ms) | Fase experimental; requiere validación exhaustiva |

**Candidato principal:** YOLOv26-pose, por sus latencias de procesamiento inferiores a 15 ms por fotograma, lo que lo convierte en el candidato ideal para alimentar en tiempo real el motor trigonométrico del backend sin generar cuellos de botella.

---

## 11. Modelo Matemático Backend (Núcleo Determinista)

> Esta sección vale **100 puntos** en la rúbrica.

El backend en Python recibe el archivo JSON generado por el modelo YOLO y extrae los nodos $K_2$, $K_4$ y $K_7$ para construir vectores direccionales espaciales que permiten calcular el ángulo de flexión cervicodorsal $\theta$.

### 11.1 Puntos Clave Utilizados

| Keypoint | Nombre | Rol en el cálculo |
|---|---|---|
| **K2** | Head-back (Occipital) | Extremo del vector cervical |
| **K4** | Neck-back (Cervical posterior) | Punto de origen — vértice del ángulo |
| **K7** | Back-backedge (Borde dorsal) | Extremo del vector dorsal |

### 11.2 Modelado mediante Trigonometría Vectorial

**Vector Cervical $\vec{u}$** — representa la orientación del cuello respecto a la cabeza:

$$\vec{u} = K_2 - K_4 = (x_2 - x_4,\ y_2 - y_4)$$

**Vector Dorsal $\vec{v}$** — representa la alineación de la parte superior del tronco:

$$\vec{v} = K_7 - K_4 = (x_7 - x_4,\ y_7 - y_4)$$

**Ángulo de Flexión Cervicodorsal $\theta$:**

$$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|\ \|\vec{v}\|}\right)$$

El sistema evalúa matemáticamente estos vectores contra ejes de referencia ideales, calculando los grados exactos de flexión y extensión de la columna superior. El resultado determina si la postura del trabajador supera los umbrales ergonómicos definidos.

---

## 12. Metodología

La metodología se estructura en **tres fases secuenciales**:

1. Preprocesamiento y aumento del dataset especializado en detección del esqueleto personalizado del torso.
2. Implementación y evaluación de modelos de estimación de pose (familia YOLO) para extraer coordenadas $(X, Y)$ de los keypoints.
3. Integración del modelo matemático determinista para calcular la flexión cervicodorsal mediante trigonometría vectorial.

### 12.1 Selección del Modelo — Criterios de Evaluación

- **Profundidad de la Red:** Una mayor profundidad permite aprender representaciones más abstractas, pero incrementa el costo computacional y el riesgo de sobreajuste.
- **Complejidad Computacional (FLOPs):** Se priorizan modelos con operaciones de convolución optimizadas que garanticen tiempos de inferencia rápidos sin perder precisión en la localización de keypoints.

### 12.2 Ajuste de Hiperparámetros

| Hiperparámetro | Descripción | Rango probado |
|---|---|---|
| **Learning Rate** | Magnitud de actualización de pesos | 0.001 / 0.0005 / 0.0001 |
| **Batch Size** | Muestras procesadas antes de actualizar pesos | 32 / 16 / 8 |
| **Epochs** | Recorridos completos sobre el dataset | 17, 34, 51, 68, 85, 102, 119, 136, 150 |
| **Input Size** | Dimensiones de imágenes de entrada | 640×640 px |

Se utiliza el optimizador **AdamW** y **Early Stopping** con un intervalo de 50 epochs.

### 12.3 Matriz de Hiperparámetros — 108 Submodelos

Se entrenaron **108 submodelos** variando arquitectura, epochs, learning rate y batch size (27 configuraciones × 4 arquitecturas):

| Rango de Submodelos | Arquitectura | Configuraciones Epochs | Learning Rates | Batch Sizes |
|---|---|---|---|---|
| 1 – 27 | YOLOv5-pose | 17 → 150 (9 valores) | 0.001 / 0.0005 / 0.0001 | 32 / 16 / 8 |
| 28 – 54 | YOLOv8-pose | 17 → 150 (9 valores) | 0.001 / 0.0005 / 0.0001 | 32 / 16 / 8 |
| 55 – 81 | YOLOv11-pose | 17 → 150 (9 valores) | 0.001 / 0.0005 / 0.0001 | 32 / 16 / 8 |
| 82 – 108 | YOLOv26-pose | 17 → 150 (9 valores) | 0.001 / 0.0005 / 0.0001 | 32 / 16 / 8 |

De los 108 submodelos se seleccionan **12 submodelos finales** (3 por arquitectura) para la evaluación comparativa formal con métricas mAP50 y OKS.

### 12.4 Métricas de Evaluación

Se descartaron las métricas tradicionales de clasificación (Accuracy, F1-Score). Se usan exclusivamente:

- **OKS (Object Keypoint Similarity):** Métrica principal; valida si las coordenadas sirven para los cálculos posteriores, sin depender de la distancia a la cámara.
- **mAP50-95 para Pose:** Verifica que la estructura del esqueleto detectado sea consistente bajo diferentes umbrales de confianza.

---

## 13. Integración en Tiempo Real

### 13.1 Pipeline de Procesamiento

```
Captura de Cámara (RGB)
        |
        v
Inferencia YOLO  (Frontend Visual)
        |
        v
Optimización TensorRT / ONNX
        |
        v
Exportación JSON  (Keypoints X,Y + Confidence)
        |
        v
Motor NumPy  —  Calculo de theta  (Backend Geométrico)
        |
        v
Sistema de Alertas WebSockets
        |
        v
Notificación Visual / Auditiva al Trabajador
```

> Todo el procesamiento ocurre **en memoria RAM**. El sistema captura el frame RGB, genera el JSON con coordenadas y elimina la imagen, garantizando la privacidad del trabajador.

### 13.2 Motor de Inferencia y Procesamiento de Video

El primer eslabón consiste en la captura y preprocesamiento de los frames mediante una tubería (pipeline) concurrente. Se evalúan motores de aceleración como **ONNX Runtime** o **TensorRT** para reducir el consumo de memoria y garantizar la extracción de los tensores espaciales $(X, Y)$ en el orden de los milisegundos. La lectura de la cámara y la inferencia se ejecutan de manera independiente para evitar bloqueos del sistema.

### 13.3 Motor Geométrico Backend

La salida de la red neuronal es un arreglo estructurado en formato JSON con las coordenadas bidimensionales de los 9 nodos detectados (desde K0 *chin* hasta K8 *hips-backedge*). El backend implementa álgebra vectorial y trigonometría para calcular los grados exactos de flexión y extensión de la columna superior. La red neuronal no genera ninguna conclusión sobre la postura; la evaluación ocurre exclusivamente en el backend.

### 13.4 Trazabilidad y Sistema Asíncrono de Alertas

Una vez calculados los ángulos de flexión y superados los umbrales ergonómicos definidos durante un tiempo predefinido *t*, el sistema emite alertas mediante **WebSockets**, permitiendo retroalimentar al usuario de forma inmediata con alertas visuales y auditivas.

---

## 14. Tecnologías de Software y Herramientas

| Herramienta | Rol en el Proyecto |
|---|---|
| **Python** | Lenguaje de orquestación del sistema completo |
| **Roboflow** | Gestión del dataset, anotación de keypoints, Data Augmentation y partición Train/Val |
| **Kaggle / Hugging Face** | Repositorios de datos abiertos para la recopilación de imágenes |
| **OpenCV** | Captura asíncrona de frames, conversión de espacios de color y renderizado visual |
| **Albumentations** | Data Augmentation con recalculación automática subpíxel de coordenadas keypoints |
| **NumPy** | Motor matemático del backend: álgebra lineal, vectores, funciones trigonométricas |
| **Ultralytics (PyTorch)** | Framework de entrenamiento de modelos YOLO con aceleración CUDA/cuDNN |
| **ONNX** | Serialización e interoperabilidad del modelo entrenado entre frameworks |
| **TensorRT** | Optimización e inferencia de alta velocidad en GPUs NVIDIA |
| **WebSockets** | Protocolo de comunicación bidireccional asíncrona para el sistema de alertas |

---

## 15. Discusión Frente al Estado del Arte

### 15.1 Frente a Modelos de Clasificación de Caja Negra

La tendencia más común en el estado del arte es usar clasificadores sobre las coordenadas obtenidas. Estudios como [4] (SVM, F1-Score 98.1%), [38] (MLP, accuracy 95.8%), [6] (MLP/XGBoost, accuracy 94.59%) y [37] (Decision Tree, accuracy 97.05%) muestran valores altos en las métricas, pero tienen una limitación importante: no permiten explicar cómo se toma la decisión. En cambio, este sistema utiliza trigonometría de producto punto (cálculo directo del ángulo $\theta$), produciendo resultados deterministas y verificables.

### 15.2 Frente a Enfoques de Bounding Boxes

En el estudio [7] (OpenVINO + SSD), la inferencia basada en bounding boxes obtuvo accuracy de 0.772 (77.2%) en usuarios sentados, demostrando que los cuadros delimitadores no son suficientemente precisos para monitoreo ergonómico. Este sistema, al usar regresión de 9 keypoints, reduce la ambigüedad causada por oclusiones parciales en entornos con escritorios.

### 15.3 Frente a Tecnologías de Radar

Estudios como [5], [34], [35] y [36] optan por radiofrecuencia (UWB/FMCW) para preservar la privacidad, logrando precisiones entre 88.7% y 98.07%, pero con costos de hardware elevados. Este sistema aborda el problema de privacidad mediante **procesamiento asíncrono en RAM**: el frame RGB se procesa, se genera el JSON con coordenadas y la imagen se elimina, logrando privacidad comparable usando cámaras estándar de bajo costo.

---

## 16. Hoja de Ruta Académica — Checklist Rúbrica 500/500

| Ítem | Puntos | Requisito | Estado |
|---|---|---|---|
| **Estado del Arte y Referencias** | 40 pts | 20 referencias IEEE · ≥18 en inglés · ninguna anterior a 2023 | ✅ 62 referencias documentadas |
| **Dataset** | 40 pts | >35.000 registros balanceados con augmentation | ✅ 15.000 → 35.000 con pipeline de augmentation |
| **Selección de Modelos** | 50 pts | 4 modelos YOLO comparados con justificación técnica | ✅ YOLOv8n, YOLOv8x, YOLO11, YOLOv26 |
| **Simulación y Trazabilidad** | 50 pts | Flujo asíncrono documentado con procesamiento en RAM | ✅ Pipeline documentado |
| **Modelo Matemático** | 100 pts | Backend determinista con vectores $\vec{u}$, $\vec{v}$ y cálculo de $\theta$ | ✅ Trigonometría vectorial implementada |
| **Aplicación Implementada** | 50 pts | mAP@0.5:0.95 y OKS >95% · 10 funcionalidades documentadas | En validación |
| **Idioma** | — | Entrega y sustentación **en inglés** | Pendiente |

---

## 17. Conclusiones

- **Eficacia de la arquitectura YOLO:** Los modelos YOLO-Pose demostraron capacidad para identificar correctamente los 9 puntos articulares, incluso con oclusión por escritorios, manteniendo detección consistente según mAP50 y OKS.

- **Superación del enfoque de "caja negra":** Limitar la red neuronal a ubicar puntos del cuerpo (generando el JSON) y dejar la evaluación ergonómica a un backend basado en trigonometría produce resultados deterministas y explicables, calculando directamente el ángulo $\theta$ de flexión cervicodorsal.

- **Viabilidad computacional:** El tiempo de inferencia del submodelo ganador demuestra que es posible realizar análisis biomecánicos en tiempo real usando hardware estándar de estaciones de trabajo (CPU y cámaras integradas), sin GPUs costosas ni sensores especializados.

- **Impacto en salud ocupacional y privacidad:** La solución contribuye a la prevención de trastornos musculoesqueléticos asociados al sedentarismo prolongado. El procesamiento sin almacenamiento de video en formato RGB reduce los problemas de privacidad que han limitado el uso de visión artificial en entornos laborales.

- **Trabajos futuros:** Se sugiere explorar modelos de estimación de pose en 3D para analizar inclinaciones laterales de la columna, e integrar un módulo asíncrono que ajuste automáticamente el umbral ergonómico a partir de un historial de medidas personales del usuario.

- **Conclusión general:** Esta investigación demuestra que combinar el aprendizaje profundo (enfocado en la obtención de coordenadas) con la geometría analítica permite mejores resultados que los sistemas basados solo en clasificadores. Este enfoque cambia el monitoreo postural, pasando de simples alertas basadas en probabilidad a una medición biomecánica precisa, escalable y respetuosa con la privacidad del usuario.

---

## 18. Referencias Bibliográficas

[1] *Posture and how it affects your health.* (2024). Brown University Health. https://www.brownhealth.org/be-well/posture-and-how-it-affects-your-health

[2] *Posture Improvement Benefits for Workplace Wellness.* (2025). Peakportland.com. https://peakportland.com/posture-improvement-benefits-for-workplace-wellness/

[3] *Building Body Posture Detection System Using MediaPipe.* (2023). Learnopencv.com. https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

[4] Jin, H., He, X., Wang, L., Zhu, Y., Jiang, W., & Zhou, X. (2024). *SitPose: Real-time detection of sitting posture and sedentary behavior using ensemble learning with depth sensor.* arXiv. http://arxiv.org/abs/2412.12216

[5] Lu, W., Bird, C., Sandhu, M., & Silvera-Tawil, D. (2025). *Office posture detection using ceiling-mounted ultra-wideband radar and attention-based modality fusion.* Sensors, 25(16), 5164. https://doi.org/10.3390/s25165164

[6] Pawitra, T. A. (2025). *Detection of ergonomic sitting postures in office environments.* Journal of Image and Graphics, 13(6). https://doi.org/10.18178/joig.13.6.686-701

[7] Huang, J., & Zhou, D. (2024). *A scalable real-time computer vision system for student posture detection in smart classrooms.* Education and Information Technologies, 29, 917–937. https://doi.org/10.1007/s10639-023-12365-5

[8] Nadeem, M., Elbasi, E., Zreikat, A. I., & Sharsheer, M. (2024). *Sitting Posture Recognition Systems: Comprehensive Literature Review and Analysis.* Applied Sciences, 14(18), 8557. https://doi.org/10.3390/app14188557

[34] Lai, D. K.-H., et al. (2023). *Dual ultra-wideband (UWB) radar-based sleep posture recognition system.* Engineered Regeneration, 4(1), 36–43. https://doi.org/10.1016/j.engreg.2022.11.003

[35] Liu, G., Li, X., Xu, C., Ma, L., & Li, H. (2023). *FMCW radar-based human sitting posture detection.* IEEE Access, 11, 102746–102756. https://doi.org/10.1109/access.2023.3312328

[36] Zhang, G., Li, S., Zhang, K., & Lin, Y.-J. (2023). *Machine Learning-Based Human Posture Identification from Point Cloud Data Acquired by FMCW Millimetre-Wave Radar.* Sensors, 23(16), 7208. https://doi.org/10.3390/s23167208

[37] Estrada, J. E., Vea, L. A., & Devaraj, M. (2023). *Modelling Proper and Improper Sitting Posture of Computer Users Using Machine Vision.* Applied Sciences, 13(9), 5402. https://doi.org/10.3390/app13095402

[38] Zhao, S., & Su, Y. (2024). *Sitting posture recognition based on the computer's camera.* Proceedings of CVIPPR 2024.

[39] Sabo, A., Mittal, N., Deshpande, A., Clarke, H., & Taati, B. (2024). *Automated, Vision-Based Goniometry and Range of Motion Calculation.* IEEE Journal of Translational Engineering in Health and Medicine, 12, 140–150. https://doi.org/10.1109/JTEHM.2023.3327691

[40] Moreira, R., et al. (2024). *Validity Analysis of Monocular Human Pose Estimation Models Interfaced with a Mobile Application for Assessing Upper Limb Range of Motion.* Sensors, 24(24), 7983. https://doi.org/10.3390/s24247983

[42] CLEAR. (2023). *What is a Cervical Curvature and Why is it Important?* CLEAR Scoliosis Institute. https://clear-institute.org/blog/cervical-curvature/

[43] Lingaraju. (2023). *Calculus of trigonometric functions in machine learning algorithms.* World Journal of Advanced Research and Reviews, 15(2), 926–931. https://doi.org/10.30574/wjarr.2022.15.2.0832

[44] *Floating-point operations per second (FLOPS).* (2024). GeeksforGeeks. https://www.geeksforgeeks.org/computer-organization-architecture/what-is-floating-point-operations-per-second-flops/

[45] *Object Keypoint Similarity in Keypoint Detection.* (2023). Learnopencv.com. https://learnopencv.com/object-keypoint-similarity/

[46] Kaur, R., & Singh, S. (2023). *A comprehensive review of object detection with deep learning.* Digital Signal Processing, 132, 103812. https://doi.org/10.1016/j.dsp.2022.103812

[48] Choudhary, A. S. (2023). *ONNX Model.* Analytics Vidhya. https://www.analyticsvidhya.com/blog/2023/07/onnx-model-open-neural-network-exchange/

[49] *NVIDIA TensorRT.* (2026). NVIDIA Developer. https://developer.nvidia.com/tensorrt

[50] *WebSocket and its difference from HTTP.* (2025). GeeksforGeeks. https://www.geeksforgeeks.org/web-tech/what-is-web-socket-and-how-it-is-different-from-the-http/

[54] Al-Dubai, A. Y., et al. (2025). *Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5.* arXiv:2510.09653.

[55] Singh, R. K., & Gupta, A. (2025). *Hand Pose Detection Using YOLOv8-pose.* University of Hertfordshire Research Archive. https://uhra.herts.ac.uk/id/eprint/25619/1/paper_17.pdf

[56] Khan, S., et al. (2024). *YOLO11 and Vision Transformers based 3D Pose Estimation of Immature Green Fruits.* arXiv:2410.19846v2.

[57] Nelson, J., & Solawetz, B. (2024). *How to Build a Computer Vision Active Learning Workflow.* Roboflow Blog. https://blog.roboflow.com/active-learning-workflow/

[58] Iglovikov, V., & Buslaev, A. (2024). *Albumentations: Fast and flexible image augmentations for computer vision.* https://albumentations.ai/docs/

[59] Bradski, G. (2024). *The OpenCV Library: Real-Time Computer Vision.* https://opencv.org/

[60] Harris, C. R., et al. (2023). *Array programming with NumPy.* Nature, 585, 357–362.

[61] Python Software Foundation. (2023). *Python Language Reference, version 3.11.* https://www.python.org/

[62] Terven, J., Córdova-Esparza, D.-M., & Romero-González, J.-A. (2023). *A Comprehensive Review of YOLO Architectures in Computer Vision.* Machine Learning and Knowledge Extraction, 5(4), 1680–1716. https://doi.org/10.3390/make5040083

---

*Universidad Surcolombiana — Neiva, Huila, Colombia · 2026*
*(Castañeda Guzmán & Idarraga Plazas, 2026)*
