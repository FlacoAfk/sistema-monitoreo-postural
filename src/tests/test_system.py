"""
Pruebas exhaustivas del sistema de monitoreo postural.
Valida: topología de keypoints, cálculo trigonométrico, clasificación, alertas,
imports, y coherencia entre módulos.
"""
import sys
import math
import time

# ── 1. Test de imports ───────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Imports y dependencias")
print("=" * 70)

try:
    from src.core.posture_analyzer import PostureAnalyzer, PostureStatus, PostureResult
    print("  ✅ posture_analyzer importado correctamente")
except Exception as e:
    print(f"  ❌ ERROR importando posture_analyzer: {e}")
    sys.exit(1)

try:
    from src.inference.inference_engine import (
        KEYPOINT_NAMES,
        CRITICAL_KEYPOINT_INDICES,
        SKELETON_CONNECTIONS,
        COLORS_BGR,
        COLOR_SKELETON,
        COLOR_ANGLE_LINE,
        KeypointResult,
        draw_pose_overlay,
    )
    print("  ✅ inference_engine importado correctamente")
except Exception as e:
    print(f"  ❌ ERROR importando inference_engine: {e}")
    sys.exit(1)

errors = []

# ── 2. Test de topología de keypoints ────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 2: Topología de keypoints (nombres y índices críticos)")
print("=" * 70)

# Verificar que los nombres coinciden con la topología verificada
expected_names = {
    0:     "K0_HeadBack",
    1: "K1_NeckBack",
    2: "K2_ShoulderTop",
    3: "K3_BackBorde",
    4: "K4_HipsBack",
    5: "K5_NeckMid",
    6: "K6_Jaw",
    7: "K7_Chin",
    8: "K8_ShoulderBack",
}

for idx, expected in expected_names.items():
    actual = KEYPOINT_NAMES[idx]
    if actual == expected:
        print(f"  ✅ K{idx}: {actual}")
    else:
        msg = f"  ❌ K{idx}: esperado '{expected}', obtenido '{actual}'"
        print(msg)
        errors.append(msg)

# Verificar índices críticos
expected_critical = [0, 1, 8]
if CRITICAL_KEYPOINT_INDICES == expected_critical:
    print(f"  ✅ Índices críticos: {CRITICAL_KEYPOINT_INDICES}")
else:
    msg = f"  ❌ Índices críticos: esperado {expected_critical}, obtenido {CRITICAL_KEYPOINT_INDICES}"
    print(msg)
    errors.append(msg)

# Verificar que el esqueleto NO contiene conexiones a K6/K7 como críticos
angle_connections = [(0, 1), (1, 8)]  # K0→K1 y K1→K8
for conn in angle_connections:
    if conn in SKELETON_CONNECTIONS:
        print(f"  ✅ Conexión angular {conn} presente en SKELETON_CONNECTIONS")
    else:
        msg = f"  ❌ Conexión angular {conn} NO encontrada en SKELETON_CONNECTIONS"
        print(msg)
        errors.append(msg)

# ── Helper: build 9 keypoints with given x,y,conf for posterior chain ─────────
def _make_kps(
    k0: tuple = (300, 50, 0.9), k1: tuple = (300, 100, 0.9),
    k2: tuple = (300, 150, 0.9), k3: tuple = (300, 300, 0.9),
    k4: tuple = (300, 400, 0.9), k5: tuple = (300, 175, 0.9),
    k6: tuple = (300, 125, 0.9), k7: tuple = (300, 135, 0.9),
    k8: tuple = (300, 200, 0.9),
) -> list:
    pts = [[0, 0, 0] for _ in range(9)]
    for i, val in enumerate([k0, k1, k2, k3, k4, k5, k6, k7, k8]):
        pts[i] = list(val)
    return pts


# ── 3. Test del CPI (Combined Posture Index) ────────────────────────────────
print("\n" + "=" * 70)
print("TEST 3: CPI — Combined Posture Index (lumbar + curvature)")
print("=" * 70)

analyzer = PostureAnalyzer()

# Caso 1: Postura perfecta (espalda recta)
# K1, K8, K3, K4 colineales en X=300 → ∠K8-K3-K4 ≈ 180° → CPI ≈ 0
kps_perfect = _make_kps(
    k0=(300, 50, 0.9), k1=(300, 100, 0.9), k2=(300, 150, 0.9),
    k3=(300, 300, 0.9), k4=(300, 400, 0.9), k5=(300, 175, 0.9),
    k6=(300, 125, 0.9), k7=(300, 135, 0.9), k8=(300, 200, 0.9),
)

result = analyzer.analyze(kps_perfect, detected=True, frame_id=1)
print(f"  Postura recta vertical: CPI = {result.cpi}, ∠lumbar = {result.lumbar_angle_deg}°, estado = {result.status.value}")
if result.cpi <= 35.0 and result.status == PostureStatus.CORRECTO:
    print(f"  ✅ Postura perfecta detectada correctamente (CPI ≤ 35)")
else:
    msg = f"  ❌ Postura perfecta: esperado CPI ≤ 35 CORRECTO, obtenido CPI={result.cpi} {result.status.value}"
    print(msg)
    errors.append(msg)

# Caso 2: Postura con curvatura leve (hombros adelantados)
# K8 se desplaza adelante; K3, K4, K1 en línea vertical
# → CPI entre 35-50 → ALERTA_LEVE
kps_leve = _make_kps(
    k0=(300, 50, 0.9), k1=(300, 100, 0.9), k2=(300, 150, 0.9),
    k3=(300, 300, 0.9), k4=(300, 400, 0.9), k5=(300, 175, 0.9),
    k6=(300, 125, 0.9), k7=(300, 135, 0.9), k8=(270, 200, 0.9),
)

result_leve = analyzer.analyze(kps_leve, detected=True, frame_id=2)
print(f"\n  Curvatura leve: CPI = {result_leve.cpi}, ∠lumbar = {result_leve.lumbar_angle_deg}°, estado = {result_leve.status.value}")
if 35.0 < result_leve.cpi <= 50.0 and result_leve.status == PostureStatus.ALERTA_LEVE:
    print(f"  ✅ Curvatura leve detectada (35 < CPI ≤ 50) → ALERTA_LEVE")
else:
    msg = f"  ❌ Curvatura leve: esperado 35 < CPI ≤ 50 ALERTA_LEVE, obtenido CPI={result_leve.cpi} {result_leve.status.value}"
    print(msg)
    errors.append(msg)

# Caso 3: Postura con curvatura crítica (hombros muy adelantados)
# K8 mucho más desplazado; K3, K4, K1 en línea vertical
# → CPI > 50 → ALERTA_CRÍTICA
kps_critica = _make_kps(
    k0=(300, 50, 0.9), k1=(300, 100, 0.9), k2=(300, 150, 0.9),
    k3=(300, 300, 0.9), k4=(300, 400, 0.9), k5=(300, 175, 0.9),
    k6=(300, 125, 0.9), k7=(300, 135, 0.9), k8=(250, 200, 0.9),
)

result_crit = analyzer.analyze(kps_critica, detected=True, frame_id=3)
print(f"\n  Curvatura crítica: CPI = {result_crit.cpi}, ∠lumbar = {result_crit.lumbar_angle_deg}°, estado = {result_crit.status.value}")
if result_crit.cpi > 50.0 and result_crit.status == PostureStatus.ALERTA_CRITICA:
    print(f"  ✅ Curvatura crítica detectada (CPI > 50) → ALERTA_CRÍTICA")
else:
    msg = f"  ❌ Curvatura crítica: esperado CPI > 50 ALERTA_CRÍTICA, obtenido CPI={result_crit.cpi} {result_crit.status.value}"
    print(msg)
    errors.append(msg)

# Caso 4: Sin detección (keypoints vacíos)
result_none = analyzer.analyze([], detected=False, frame_id=4)
if result_none.status == PostureStatus.NO_DETECTADO:
    print(f"\n  ✅ Sin detección → NO DETECTADO")
else:
    msg = f"  ❌ Sin detección: esperado NO DETECTADO, obtenido {result_none.status.value}"
    print(msg)
    errors.append(msg)

# Caso 5: Keypoint crítico (K1) con baja confianza
# El CPI requiere K1, K3, K4, K8 con conf ≥ 0.1
kps_low_conf = _make_kps(
    k0=(300, 50, 0.9), k1=(300, 100, 0.05),  # K1 con confianza muy baja
    k3=(300, 300, 0.9), k8=(300, 200, 0.9),
)
for i in range(9):
    if kps_low_conf[i][2] == 0:  # Rellenar keypoints no configurados
        kps_low_conf[i] = [300.0, 250.0, 0.5]

result_low = analyzer.analyze(kps_low_conf, detected=True, frame_id=5)
if result_low.status == PostureStatus.NO_DETECTADO:
    print(f"  ✅ Baja confianza en K1 → NO DETECTADO")
else:
    msg = f"  ❌ Baja confianza: esperado NO DETECTADO, obtenido {result_low.status.value}"
    print(msg)
    errors.append(msg)

# ── 4. Test del sistema de alertas temporales ────────────────────────────────
print("\n" + "=" * 70)
print("TEST 4: Sistema de alertas (acumulación de tiempo)")
print("=" * 70)

alert_analyzer = PostureAnalyzer()
# Mal postura: hombros adelantados → CPI > 50 → ALERTA_CRÍTICA
kps_bad = _make_kps(
    k0=(300, 50, 0.9), k1=(300, 100, 0.9), k2=(300, 150, 0.9),
    k3=(300, 300, 0.9), k4=(300, 400, 0.9), k5=(300, 175, 0.9),
    k6=(300, 125, 0.9), k7=(300, 135, 0.9), k8=(250, 200, 0.9),
)

t0 = time.time()
r1 = alert_analyzer.analyze(kps_bad, detected=True, timestamp=t0, frame_id=10)
print(f"  Frame 1 (t=0s): acumulado={r1.bad_posture_accumulated_s}s, alerta={r1.needs_alert}")
if r1.bad_posture_accumulated_s == 0.0 and not r1.needs_alert:
    print(f"  ✅ Inicio correcto (0s acumulados, sin alerta)")
else:
    msg = f"  ❌ Inicio: esperado 0s sin alerta"
    print(msg)
    errors.append(msg)

# Simular 31 segundos después
r2 = alert_analyzer.analyze(kps_bad, detected=True, timestamp=t0 + 31, frame_id=11)
print(f"\n  Frame 2 (t=31s): acumulado={r2.bad_posture_accumulated_s}s, alerta={r2.needs_alert}")
if r2.bad_posture_accumulated_s > 30.0 and r2.needs_alert:
    print(f"  ✅ Alerta disparada correctamente después de 31s")
else:
    msg = f"  ❌ Alerta: esperado >30s con alerta=True, obtenido {r2.bad_posture_accumulated_s}s alerta={r2.needs_alert}"
    print(msg)
    errors.append(msg)

# Simular corrección de postura (columna recta → CPI ≈ 0 → CORRECTO)
kps_good = _make_kps(
    k0=(300, 50, 0.9), k1=(300, 100, 0.9), k2=(300, 150, 0.9),
    k3=(300, 300, 0.9), k4=(300, 400, 0.9), k5=(300, 175, 0.9),
    k6=(300, 125, 0.9), k7=(300, 135, 0.9), k8=(300, 200, 0.9),
)

r3 = alert_analyzer.analyze(kps_good, detected=True, timestamp=t0 + 35, frame_id=12)
print(f"\n  Frame 3 (t=35s, postura corregida): acumulado={r3.bad_posture_accumulated_s}s, estado={r3.status.value}")
if r3.bad_posture_accumulated_s == 0.0 and r3.status == PostureStatus.CORRECTO:
    print(f"  ✅ Reset de contador al corregir postura")
else:
    msg = f"  ❌ Reset: esperado 0s CORRECTO"
    print(msg)
    errors.append(msg)

# ── 5. Test de coherencia posture_analyzer ↔ inference_engine ────────────────
print("\n" + "=" * 70)
print("TEST 5: Coherencia entre módulos")
print("=" * 70)

# El pivote en posture_analyzer debe ser K1 (índice 1)
# Verificación: crear keypoints donde K6/K7 son basura y K1/K3/K4/K8 son buenos
analyzer_coherence = PostureAnalyzer()
kps_coherence = _make_kps(
    k0=(300, 50, 0.9), k1=(300, 100, 0.9), k2=(300, 150, 0.9),
    k3=(300, 300, 0.9), k4=(300, 400, 0.9), k5=(300, 175, 0.9),
    k6=(999, 999, 0.9),   # K6 BASURA — NO debe afectar
    k7=(888, 888, 0.9),   # K7 BASURA — NO debe afectar
    k8=(300, 200, 0.9),
)

result_c = analyzer_coherence.analyze(kps_coherence, detected=True, frame_id=20)
print(f"  K1/K3/K4/K8 correctos, K6/K7 basura: CPI = {result_c.cpi}")
if result_c.cpi <= 35.0:
    print(f"  ✅ El sistema usa K1/K3/K4/K8 (ignora K6/K7 correctamente)")
else:
    msg = f"  ❌ ¡El sistema todavía lee K6 o K7! CPI={result_c.cpi} debería ser ~0"
    print(msg)
    errors.append(msg)

# Verificar que los nombres en inference_engine y el pivote en posture_analyzer son coherentes
pivote_name = KEYPOINT_NAMES[1]
if "NeckBack" in pivote_name or "Cervical" in pivote_name:
    print(f"  ✅ El pivote (K1) se llama '{pivote_name}' → coherente con cuello/cervical")
else:
    msg = f"  ❌ Nombre del pivote K1: '{pivote_name}' no contiene 'NeckBack' ni 'Cervical'"
    print(msg)
    errors.append(msg)

dorsal_name = KEYPOINT_NAMES[3]
if "Back" in dorsal_name or "Borde" in dorsal_name:
    print(f"  ✅ El extremo dorsal (K3) se llama '{dorsal_name}' → coherente")
else:
    msg = f"  ❌ Nombre del dorsal K3: '{dorsal_name}' no contiene 'Back' ni 'Borde'"
    print(msg)
    errors.append(msg)

# ── 6. Test de KeypointResult ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 6: KeypointResult (serialización y propiedades)")
print("=" * 70)

kps_full = [[float(i*30), float(i*20), 0.9] for i in range(9)]
kr = KeypointResult(
    timestamp=time.time(),
    frame_id=100,
    detected=True,
    num_people=1,
    keypoints=kps_full,
)

if kr.has_valid_pose:
    print("  ✅ has_valid_pose = True para 9 keypoints con conf > 0")
else:
    msg = "  ❌ has_valid_pose debería ser True"
    print(msg)
    errors.append(msg)

coords = kr.get_kp_coords(1)
if coords == (30.0, 20.0, 0.9):
    print(f"  ✅ get_kp_coords(1) = {coords}")
else:
    msg = f"  ❌ get_kp_coords(1): esperado (30.0, 20.0, 0.9), obtenido {coords}"
    print(msg)
    errors.append(msg)

d = kr.to_dict()
if d["keypoints"][0]["name"] == "K0_HeadBack":
    print(f"  ✅ Serialización JSON usa nombre correcto: '{d['keypoints'][0]['name']}'")
else:
    msg = f"  ❌ Serialización: nombre K0 = '{d['keypoints'][0]['name']}'"
    print(msg)
    errors.append(msg)

j = kr.to_json()
if '"K1_NeckBack"' in j:
    print(f"  ✅ to_json() contiene 'K1_NeckBack'")
else:
    msg = f"  ❌ to_json() no contiene 'K1_NeckBack'"
    print(msg)
    errors.append(msg)

# ── 7. Test de Gradio launch (solo importación, no abre server) ──────────────
print("\n" + "=" * 70)
print("TEST 7: Import de app.py (sin lanzar servidor)")
print("=" * 70)

try:
    # Solo importar para verificar que no hay errores de sintaxis o imports rotos
    import importlib
    spec = importlib.util.spec_from_file_location("app", "src/ui/app.py")
    # No ejecutamos porque lanzaría Gradio, solo verificamos que se puede parsear
    import ast
    with open("src/ui/app.py", "r", encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)
    print("  ✅ app.py parsea correctamente (sin errores de sintaxis)")

    # Verificar que app.py referencia K3 y K8 (lumbar angle) en lugar de K6 y K7
    if "keypoints[3]" in source and "keypoints[8]" in source:
        print("  ✅ app.py usa keypoints[3] y keypoints[8] para el ángulo lumbar")
    else:
        msg = "  ❌ app.py NO referencia keypoints[3] / keypoints[8] para el ángulo"
        print(msg)
        errors.append(msg)

    if "keypoints[6]" in source or "keypoints[7]" in source:
        # Verificar si es para el ángulo o solo para dibujar puntos normales
        import re
        # Buscar si K6/K7 se usan en el contexto del ángulo
        angle_section = source[source.find("# Dibujar líneas del ángulo"):]
        if "keypoints[6]" in angle_section or "keypoints[7]" in angle_section:
            msg = "  ❌ app.py TODAVÍA usa keypoints[6]/[7] en la sección del ángulo"
            print(msg)
            errors.append(msg)
        else:
            print("  ✅ app.py no usa K6/K7 en la sección del ángulo")
    else:
        print("  ✅ app.py no referencia keypoints[6] ni keypoints[7]")

except Exception as e:
    msg = f"  ❌ Error parseando app.py: {e}"
    print(msg)
    errors.append(msg)


# ── RESUMEN ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN DE PRUEBAS")
print("=" * 70)

if not errors:
    print("\n  🎉 TODAS LAS PRUEBAS PASARON (0 errores)")
    print("  El sistema está listo para uso en producción.\n")
else:
    print(f"\n  ⚠️  {len(errors)} ERRORES ENCONTRADOS:\n")
    for i, e in enumerate(errors, 1):
        print(f"    {i}. {e}")
    print()

sys.exit(len(errors))
