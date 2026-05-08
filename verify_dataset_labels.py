import cv2
import os
import glob

dataset_path = r"D:\Documentos\entre\Desk Posture.v2i.yolov8_aug36k"
images_dir = os.path.join(dataset_path, "valid", "images")
labels_dir = os.path.join(dataset_path, "valid", "labels")

# Tomar las primeras 10 imágenes para revisar
image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))[:10]

output_dir = r"C:\Users\elkaw\Desktop\Modelos entrenados\imagenes_prueba_postura\verificacion_dataset_original"
os.makedirs(output_dir, exist_ok=True)

for img_path in image_paths:
    base_name = os.path.basename(img_path)
    label_name = base_name.replace(".jpg", ".txt")
    label_path = os.path.join(labels_dir, label_name)
    
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    h, w, _ = img.shape
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 23: # 1 class + 4 bbox + 9 keypoints * 2 (or 3)
                    # YOLOv8 format: class, x_c, y_c, width, height, px0, py0, [v0], px1, py1, [v1]...
                    kpts_raw = [float(x) for x in parts[5:]]
                    
                    # Detectar si tiene visibilidad (x, y, v) o solo (x, y)
                    items_per_kp = 3 if len(kpts_raw) % 3 == 0 else 2
                    num_kpts = len(kpts_raw) // items_per_kp
                    
                    for i in range(num_kpts):
                        idx = i * items_per_kp
                        kx = int(kpts_raw[idx] * w)
                        ky = int(kpts_raw[idx+1] * h)
                        
                        # Si tiene flag de visibilidad, revisarlo
                        if items_per_kp == 3:
                            v = kpts_raw[idx+2]
                            if v == 0: continue # No dibujarlo si está marcado como no visible/no etiquetado
                            
                        if kx > 0 and ky > 0:
                            # Dibujar punto rojo
                            cv2.circle(img, (kx, ky), 6, (0, 0, 255), -1)
                            # Dibujar texto amarillo K0, K1...
                            cv2.putText(img, f"K{i}", (kx + 10, ky + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
    out_path = os.path.join(output_dir, base_name)
    cv2.imwrite(out_path, img)
    print(f"Procesada: {base_name}")

print(f"\n¡Listo! Imágenes de validación guardadas en: {output_dir}")
