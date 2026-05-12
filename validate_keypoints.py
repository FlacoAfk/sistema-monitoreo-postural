"""
Extrae keypoints de las 6 imágenes de prueba con los 4 modelos YOLO-Pose.
Salida: keypoints crudos + visualización overlay + JSON.
"""
from pathlib import Path
import json, sys, time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
IMG_DIR = BASE / "imagenes de pruebas del modelo matematico"
MODELS_CFG = {
    "yolov8n": BASE / "models" / "yolov8n.pt",
    "yolov5n": BASE / "models" / "yolov5n.pt",
    "yolov26n": BASE / "models" / "yolov26n.pt",
    "yolov11n": BASE / "models" / "yolov11n.pt",
}

KP_NAMES = [
    "K0_HeadBack", "K1_NeckBack(C7)", "K2_ShoulderTop", "K3_BackBorde",
    "K4_HipsBack", "K5_NeckMid", "K6_Jaw", "K7_Chin", "K8_ShoulderBack"
]

def load_model(path, device):
    model = YOLO(str(path))
    model.to(device)
    # warmup
    dummy = np.zeros((640,640,3), dtype=np.uint8)
    model(dummy, verbose=False)
    return model

def run_one(model, img):
    preds = model(img, verbose=False, conf=0.2, imgsz=640)
    if not preds or preds[0].keypoints is None:
        return None
    kp = preds[0].keypoints
    data = kp.data.cpu().numpy()
    if data.shape[0] == 0:
        return None
    confs = data[:, :, 2]
    best = int(np.argmax(confs.mean(axis=1)))
    return data[best]

def draw_overlay(img, kps, out_path):
    out = img.copy()
    h, w = out.shape[:2]
    # Skeleton posterior chain
    connections = [(0,1), (1,8), (8,3), (3,4)]
    for a,b in connections:
        if kps[a][2] > 0.1 and kps[b][2] > 0.1:
            pa = (int(kps[a][0]), int(kps[a][1]))
            pb = (int(kps[b][0]), int(kps[b][1]))
            cv2.line(out, pa, pb, (255,200,0), 2, cv2.LINE_AA)
    # Cervicodorsal angle lines (K0-K1-K8 naranja)
    if kps[0][2] > 0.1 and kps[1][2] > 0.1 and kps[8][2] > 0.1:
        p1 = (int(kps[1][0]), int(kps[1][1]))
        p0 = (int(kps[0][0]), int(kps[0][1]))
        p8 = (int(kps[8][0]), int(kps[8][1]))
        cv2.line(out, p1, p0, (0,165,255), 2, cv2.LINE_AA)
        cv2.line(out, p1, p8, (0,165,255), 2, cv2.LINE_AA)
        u = np.array([kps[0][0]-kps[1][0], kps[0][1]-kps[1][1]])
        v = np.array([kps[8][0]-kps[1][0], kps[8][1]-kps[1][1]])
        cos_a = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
        alpha = np.degrees(np.arccos(np.clip(cos_a,-1,1)))
        cv2.putText(out, f"a={alpha:.1f}", (p1[0]+15, p1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2, cv2.LINE_AA)
    # All keypoints
    colors = [(255,0,0),(0,165,255),(0,255,0),(0,200,200),(128,128,128),
              (200,200,0),(200,0,200),(0,0,255),(128,0,128)]
    for i, kp in enumerate(kps):
        if kp[2] > 0.1:
            cx, cy = int(kp[0]), int(kp[1])
            cv2.circle(out, (cx,cy), 5, colors[i], -1)
            cv2.putText(out, KP_NAMES[i], (cx+8, cy-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[i], 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), out)

def calc_angle(kps, a, b, c):
    u = np.array([kps[a][0]-kps[b][0], kps[a][1]-kps[b][1]])
    v = np.array([kps[c][0]-kps[b][0], kps[c][1]-kps[b][1]])
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1 or nv < 1:
        return -1
    cos_a = np.dot(u,v)/(nu*nv)
    return float(np.degrees(np.arccos(np.clip(cos_a,-1,1))))

def calc_spine_deviation(kps):
    """Distancia perpendicular de K0 a línea K1→K4 (px)"""
    p1 = np.array([kps[1][0], kps[1][1]])
    p4 = np.array([kps[4][0], kps[4][1]])
    p0 = np.array([kps[0][0], kps[0][1]])
    spine_vec = p4 - p1
    spine_len = np.linalg.norm(spine_vec)
    if spine_len < 10:
        return -1
    return float(abs(np.cross(spine_vec, p0-p1)) / spine_len)

def calc_spine_curvature(kps):
    """Curvatura = cuánto se desvía K8 de la línea K1→K4 (px)"""
    p1 = np.array([kps[1][0], kps[1][1]])
    p4 = np.array([kps[4][0], kps[4][1]])
    p8 = np.array([kps[8][0], kps[8][1]])
    spine_vec = p4 - p1
    spine_len = np.linalg.norm(spine_vec)
    if spine_len < 10:
        return -1
    return float(abs(np.cross(spine_vec, p8-p1)) / spine_len)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    OUT_DIR = BASE / "posture_monitor" / "keypoint_output"
    OUT_DIR.mkdir(exist_ok=True)
    
    results = {}
    
    # Cargar cada modelo una vez
    loaded_models = {}
    for mname, mpath in MODELS_CFG.items():
        if mpath.exists():
            print(f"Cargando {mname}...")
            loaded_models[mname] = load_model(mpath, device)
        else:
            print(f"Modelo no encontrado: {mname}")
    
    for posture in ["Recto", "Semi-encorvado", "Encorvado"]:
        img_dir = IMG_DIR / posture
        for img_file in sorted(img_dir.glob("*.png")):
            img_name = img_file.stem
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"No se pudo leer: {img_file}")
                continue
            
            print(f"\n{'='*70}")
            print(f"POSTURA: {posture} | IMAGEN: {img_name}")
            print(f"{'='*70}")
            
            for model_name, model in loaded_models.items():
                kps = run_one(model, img)
                
                if kps is None:
                    print(f"  {model_name}: SIN DETECCIÓN")
                    continue
                
                a_cerv = calc_angle(kps, 0, 1, 8)   # cervicodorsal ∠K0-K1-K8
                a_tor  = calc_angle(kps, 1, 8, 3)   # torácico ∠K1-K8-K3
                a_lumb = calc_angle(kps, 8, 3, 4)   # lumbar ∠K8-K3-K4
                a_full = calc_angle(kps, 0, 1, 4)   # columna total ∠K0-K1-K4
                a_cerv_tor = calc_angle(kps, 0, 1, 3)  # ∠K0-K1-K3 (craneal→media)
                dev_head = calc_spine_deviation(kps)  # K0 ⊥ K1-K4
                curv = calc_spine_curvature(kps)       # K8 ⊥ K1-K4

                confs = [kps[i][2] for i in range(9)]
                avg_conf = sum(confs)/len(confs)
                detected = sum(1 for c in confs if c > 0.1)
                
                print(f"  {model_name}: det={detected}/9, conf={avg_conf:.3f}")
                print(f"    ∠K0-K1-K8  (cervicodorsal) = {a_cerv:.1f}°")
                print(f"    ∠K1-K8-K3  (torácico)      = {a_tor:.1f}°")
                print(f"    ∠K8-K3-K4  (lumbar)        = {a_lumb:.1f}°")
                print(f"    ∠K0-K1-K4  (columna total)  = {a_full:.1f}°")
                print(f"    ∠K0-K1-K3  (craneo-torax)   = {a_cerv_tor:.1f}°")
                print(f"    Desv. cabeza (px)           = {dev_head:.1f}")
                print(f"    Curvatura K8 (px)           = {curv:.1f}")
                print(f"    Keypoints:")
                for i, name in enumerate(KP_NAMES):
                    print(f"      {name}: x={kps[i][0]:6.1f} y={kps[i][1]:6.1f} c={kps[i][2]:.3f}")
                
                out_name = f"{posture}_{img_name}_{model_name}.png"
                draw_overlay(img.copy(), kps, OUT_DIR / out_name)
                
                key = f"{posture}/{img_name}/{model_name}"
                results[key] = {
                    "posture": posture, "model": model_name,
                    "detected": detected, "avg_conf": round(avg_conf,4),
                    "cervicodorsal_K0_K1_K8": round(a_cerv,1),
                    "toracico_K1_K8_K3": round(a_tor,1),
                    "lumbar_K8_K3_K4": round(a_lumb,1),
                    "columna_total_K0_K1_K4": round(a_full,1),
                    "craneo_torax_K0_K1_K3": round(a_cerv_tor,1),
                    "desviacion_cabeza_px": round(dev_head,1),
                    "curvatura_K8_px": round(curv,1),
                    "keypoints": [[round(float(x),1), round(float(y),1), round(float(c),3)] for x,y,c in kps]
                }
    
    # Guardar JSON
    json_path = OUT_DIR / "keypoints_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResultados guardados en: {json_path}")
    
    # ── Resumen comparativo ──
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO (mejor modelo por imagen)")
    print(f"{'='*70}")
    header = f"{'Postura/Imagen':<40} {'Modelo':<10} {'Cerv':>6} {'Tor':>6} {'Lumb':>6} {'Total':>6} {'Dev.px':>7} {'Curv':>6} {'Det':>4}"
    print(header)
    print("-"*95)
    for posture in ["Recto", "Semi-encorvado", "Encorvado"]:
        img_dir = IMG_DIR / posture
        for img_file in sorted(img_dir.glob("*.png")):
            img_name = img_file.stem
            best = None
            for model_name in loaded_models:
                key = f"{posture}/{img_name}/{model_name}"
                if key in results and results[key]["detected"] >= 5:
                    if best is None or results[key]["avg_conf"] > best["avg_conf"]:
                        best = results[key]
                        best["model"] = model_name
            if best:
                short_name = img_name.replace("Captura de pantalla 2026-05-08 ", "")[:20]
                label = f"{posture}/{short_name}"
                print(f"{label:<40} {best['model']:<10} {best['cervicodorsal_K0_K1_K8']:6.1f} {best['toracico_K1_K8_K3']:6.1f} {best['lumbar_K8_K3_K4']:6.1f} {best['columna_total_K0_K1_K4']:6.1f} {best['desviacion_cabeza_px']:7.1f} {best['curvatura_K8_px']:6.1f} {best['detected']:4}")
            else:
                print(f"{posture}/{img_name[:30]} -> NINGÚN MODELO DETECTÓ BIEN")

if __name__ == "__main__":
    main()
