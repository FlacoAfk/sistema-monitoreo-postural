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
    from posture_analyzer import PostureAnalyzer, PostureStatus, PostureResult
    print("  ✅ posture_analyzer importado correctamente")
except Exception as e:
    print(f"  ❌ ERROR importando posture_analyzer: {e}")
    sys.exit(1)

try:
    from inference_engine import (
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
    0: "K0_Occipital",
    1: "K1_CervicalC7",
    2: "K2_Acromion",
    3: "K3_BordeDorsal",
    4: "K4_Cadera",
    5: "K5_CervicalMedia",
    6: "K6_Mandibula",
    7: "K7_Menton",
    8: "K8_Escapula",
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
expected_critical = [0, 1, 3]
if CRITICAL_KEYPOINT_INDICES == expected_critical:
    print(f"  ✅ Índices críticos: {CRITICAL_KEYPOINT_INDICES}")
else:
    msg = f"  ❌ Índices críticos: esperado {expected_critical}, obtenido {CRITICAL_KEYPOINT_INDICES}"
    print(msg)
    errors.append(msg)

# Verificar que el esqueleto NO contiene conexiones a K6/K7 como críticos
angle_connections = [(0, 1), (1, 3)]  # K0→K1 y K1→K3
for conn in angle_connections:
    if conn in SKELETON_CONNECTIONS:
        print(f"  ✅ Conexión angular {conn} presente en SKELETON_CONNECTIONS")
    else:
        msg = f"  ❌ Conexión angular {conn} NO encontrada en SKELETON_CONNECTIONS"
        print(msg)
        errors.append(msg)

# ── 3. Test del cálculo trigonométrico ───────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 3: Cálculo trigonométrico del ángulo cervicodorsal")
print("=" * 70)

analyzer = PostureAnalyzer()

# Caso 1: Postura perfecta (ángulo ~0°)
# K0 (occipital) directamente arriba de K1 (C7), K3 (dorsal) directamente abajo
# Vectores u y v apuntando en la misma dirección → θ ≈ 180° → α ≈ 0°
kps_perfect = [[0]*3 for _ in range(9)]
kps_perfect[0] = [300.0, 100.0, 0.9]  # K0 Occipital (arriba)
kps_perfect[1] = [300.0, 200.0, 0.9]  # K1 C7 (pivote, medio)
kps_perfect[3] = [300.0, 350.0, 0.9]  # K3 BordeDorsal (abajo)
# Los demás keypoints con confianza 0.5 para no afectar
for i in [2, 4, 5, 6, 7, 8]:
    kps_perfect[i] = [300.0, 250.0, 0.5]

result = analyzer.analyze(kps_perfect, detected=True, frame_id=1)
print(f"  Postura recta vertical: α = {result.angle_deg}°, estado = {result.status.value}")
if result.angle_deg < 5.0 and result.status == PostureStatus.CORRECTO:
    print(f"  ✅ Postura perfecta detectada correctamente (α < 5°)")
else:
    msg = f"  ❌ Postura perfecta: esperado α < 5° CORRECTO, obtenido α={result.angle_deg}° {result.status.value}"
    print(msg)
    errors.append(msg)

# Caso 2: Postura con flexión leve (~20°)
analyzer_2 = PostureAnalyzer()
kps_leve = [[0]*3 for _ in range(9)]
kps_leve[0] = [260.0, 110.0, 0.9]   # K0 Occipital (ligeramente adelante)
kps_leve[1] = [300.0, 200.0, 0.9]   # K1 C7 (pivote)
kps_leve[3] = [300.0, 350.0, 0.9]   # K3 BordeDorsal (abajo)
for i in [2, 4, 5, 6, 7, 8]:
    kps_leve[i] = [300.0, 250.0, 0.5]

result_leve = analyzer_2.analyze(kps_leve, detected=True, frame_id=2)
print(f"\n  Flexión leve: α = {result_leve.angle_deg}°, estado = {result_leve.status.value}")
if 10.0 < result_leve.angle_deg < 30.0:
    print(f"  ✅ Flexión leve detectada (10° < α < 30°)")
else:
    msg = f"  ❌ Flexión leve: obtenido α={result_leve.angle_deg}°"
    print(msg)
    errors.append(msg)

# Caso 3: Postura con flexión crítica (~40°)
analyzer_3 = PostureAnalyzer()
kps_critica = [[0]*3 for _ in range(9)]
kps_critica[0] = [200.0, 120.0, 0.9]  # K0 Occipital (MUY adelante)
kps_critica[1] = [300.0, 200.0, 0.9]  # K1 C7 (pivote)
kps_critica[3] = [300.0, 350.0, 0.9]  # K3 BordeDorsal (abajo)
for i in [2, 4, 5, 6, 7, 8]:
    kps_critica[i] = [300.0, 250.0, 0.5]

result_crit = analyzer_3.analyze(kps_critica, detected=True, frame_id=3)
print(f"\n  Flexión crítica: α = {result_crit.angle_deg}°, estado = {result_crit.status.value}")
if result_crit.angle_deg > 25.0 and result_crit.status == PostureStatus.ALERTA_CRITICA:
    print(f"  ✅ Flexión crítica detectada correctamente (α > 25°)")
else:
    msg = f"  ❌ Flexión crítica: esperado α > 25° ALERTA CRÍTICA, obtenido α={result_crit.angle_deg}° {result_crit.status.value}"
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

# Caso 5: Keypoints críticos con baja confianza
kps_low_conf = [[0]*3 for _ in range(9)]
kps_low_conf[0] = [300.0, 100.0, 0.05]  # Confianza muy baja
kps_low_conf[1] = [300.0, 200.0, 0.9]
kps_low_conf[3] = [300.0, 350.0, 0.9]
for i in [2, 4, 5, 6, 7, 8]:
    kps_low_conf[i] = [300.0, 250.0, 0.5]

result_low = analyzer.analyze(kps_low_conf, detected=True, frame_id=5)
if result_low.status == PostureStatus.NO_DETECTADO:
    print(f"  ✅ Baja confianza en K0 → NO DETECTADO")
else:
    msg = f"  ❌ Baja confianza: esperado NO DETECTADO, obtenido {result_low.status.value}"
    print(msg)
    errors.append(msg)

# ── 4. Test del sistema de alertas temporales ────────────────────────────────
print("\n" + "=" * 70)
print("TEST 4: Sistema de alertas (acumulación de tiempo)")
print("=" * 70)

alert_analyzer = PostureAnalyzer()
kps_bad = [[0]*3 for _ in range(9)]
kps_bad[0] = [200.0, 120.0, 0.9]
kps_bad[1] = [300.0, 200.0, 0.9]
kps_bad[3] = [300.0, 350.0, 0.9]
for i in [2, 4, 5, 6, 7, 8]:
    kps_bad[i] = [300.0, 250.0, 0.5]

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

# Simular corrección de postura
kps_good = [[0]*3 for _ in range(9)]
kps_good[0] = [300.0, 100.0, 0.9]
kps_good[1] = [300.0, 200.0, 0.9]
kps_good[3] = [300.0, 350.0, 0.9]
for i in [2, 4, 5, 6, 7, 8]:
    kps_good[i] = [300.0, 250.0, 0.5]

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
# Verificación: crear keypoints donde K6/K7 son basura y K1/K3 son buenos
analyzer_coherence = PostureAnalyzer()
kps_coherence = [[0]*3 for _ in range(9)]
kps_coherence[0] = [300.0, 100.0, 0.9]  # K0 Occipital
kps_coherence[1] = [300.0, 200.0, 0.9]  # K1 C7 (PIVOTE REAL)
kps_coherence[3] = [300.0, 350.0, 0.9]  # K3 BordeDorsal
# K6 y K7 con datos BASURA para verificar que NO se usan
kps_coherence[6] = [999.0, 999.0, 0.9]  # K6 Mandíbula (NO debe afectar)
kps_coherence[7] = [999.0, 999.0, 0.9]  # K7 Mentón (NO debe afectar)
for i in [2, 4, 5, 8]:
    kps_coherence[i] = [300.0, 250.0, 0.5]

result_c = analyzer_coherence.analyze(kps_coherence, detected=True, frame_id=20)
print(f"  K1/K3 correctos, K6/K7 basura: α = {result_c.angle_deg}°")
if result_c.angle_deg < 5.0:
    print(f"  ✅ El sistema usa K1/K3 (ignora K6/K7 correctamente)")
else:
    msg = f"  ❌ ¡El sistema todavía lee K6 o K7! α={result_c.angle_deg}° debería ser ~0°"
    print(msg)
    errors.append(msg)

# Verificar que los nombres en inference_engine y el pivote en posture_analyzer son coherentes
pivote_name = KEYPOINT_NAMES[1]
if "C7" in pivote_name or "Cervical" in pivote_name:
    print(f"  ✅ El pivote (K1) se llama '{pivote_name}' → coherente con C7")
else:
    msg = f"  ❌ Nombre del pivote K1: '{pivote_name}' no contiene 'C7' ni 'Cervical'"
    print(msg)
    errors.append(msg)

dorsal_name = KEYPOINT_NAMES[3]
if "Dorsal" in dorsal_name or "Back" in dorsal_name:
    print(f"  ✅ El extremo dorsal (K3) se llama '{dorsal_name}' → coherente")
else:
    msg = f"  ❌ Nombre del dorsal K3: '{dorsal_name}' no contiene 'Dorsal' ni 'Back'"
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
if d["keypoints"][0]["name"] == "K0_Occipital":
    print(f"  ✅ Serialización JSON usa nombre correcto: '{d['keypoints'][0]['name']}'")
else:
    msg = f"  ❌ Serialización: nombre K0 = '{d['keypoints'][0]['name']}'"
    print(msg)
    errors.append(msg)

j = kr.to_json()
if '"K1_CervicalC7"' in j:
    print(f"  ✅ to_json() contiene 'K1_CervicalC7'")
else:
    msg = f"  ❌ to_json() no contiene 'K1_CervicalC7'"
    print(msg)
    errors.append(msg)

# ── 7. Test de Gradio launch (solo importación, no abre server) ──────────────
print("\n" + "=" * 70)
print("TEST 7: Import de app.py (sin lanzar servidor)")
print("=" * 70)

try:
    # Solo importar para verificar que no hay errores de sintaxis o imports rotos
    import importlib
    spec = importlib.util.spec_from_file_location("app", "app.py")
    # No ejecutamos porque lanzaría Gradio, solo verificamos que se puede parsear
    import ast
    with open("app.py", "r", encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)
    print("  ✅ app.py parsea correctamente (sin errores de sintaxis)")

    # Verificar que app.py referencia K1 y K3 en lugar de K6 y K7 para el ángulo
    if "keypoints[1]" in source and "keypoints[3]" in source:
        print("  ✅ app.py usa keypoints[1] y keypoints[3] para el ángulo")
    else:
        msg = "  ❌ app.py NO referencia keypoints[1] / keypoints[3]"
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
