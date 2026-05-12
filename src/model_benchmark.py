"""
Component 4 — Comparador de Modelos (model_benchmark.py)

Carga los 4 modelos seleccionados secuencialmente, los evalúa sobre un
conjunto de imágenes de prueba y genera:
  - Tabla comparativa: confianza K2/K4/K7, tiempo de inferencia, OKS estimado
  - Reporte JSON con métricas detalladas
  - Gráfica de barras comparativa

Autor: Sistema de Monitoreo Postural — Universidad Surcolombiana 2026
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Sin GUI — solo exportar a archivo
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ── Configuración ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TEST_IMAGES_DIR = BASE_DIR / "imagenes_prueba_postura" / "imagenes_originales"
OUTPUT_DIR = BASE_DIR / "outputs"

# 4 modelos nano seleccionados (incluidos en el repo bajo models/)
SELECTED_MODELS: list[dict[str, Any]] = [
    {"name": "yolov5n",  "family": "YOLOv5",  "variant": "n", "weight": "yolov5n.pt"},
    {"name": "yolov8n",  "family": "YOLOv8",  "variant": "n", "weight": "yolov8n.pt"},
    {"name": "yolov11n", "family": "YOLO11",  "variant": "n", "weight": "yolov11n.pt"},
    {"name": "yolov26n", "family": "YOLOv26", "variant": "n", "weight": "yolov26n.pt"},
]

# Keypoints críticos para el ángulo cervicodorsal (mapeo Roboflow→YOLO):
#   K0 (Roboflow 0) = Cabeza, K6 (Roboflow 13) = Cervical C7 (pivote), K7 (Roboflow 14) = Escápula
CRITICAL_KP_NAMES = ["K0_Cabeza", "K6_CervicalC7", "K7_Escapula"]
CRITICAL_KP_INDICES = [0, 6, 7]


@dataclass
class ModelBenchmarkEntry:
    """Resultado de benchmark para un modelo."""

    model_name: str
    family: str
    variant: str
    total_images: int = 0
    detected_images: int = 0
    detection_rate_pct: float = 0.0
    avg_time_ms: float = 0.0
    total_time_ms: float = 0.0
    avg_conf_k2: float = 0.0
    avg_conf_k4: float = 0.0
    avg_conf_k7: float = 0.0
    avg_conf_all: float = 0.0
    composite_score: float = 0.0
    per_image: list[dict] = field(default_factory=list)


def run_benchmark(
    models: list[dict[str, Any]] | None = None,
    image_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[ModelBenchmarkEntry]:
    """
    Ejecuta benchmark completo sobre los 4 modelos seleccionados.

    Args:
        models: Lista de modelos a evaluar (usa SELECTED_MODELS por defecto).
        image_dir: Directorio con imágenes de prueba.
        output_dir: Directorio para guardar resultados.

    Returns:
        Lista de ModelBenchmarkEntry con resultados por modelo.
    """
    if models is None:
        models = SELECTED_MODELS
    if image_dir is None:
        image_dir = TEST_IMAGES_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtener imágenes de prueba
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if not image_paths:
        raise FileNotFoundError(f"No hay imágenes en {image_dir}")

    print(f"Benchmark: {len(models)} modelos × {len(image_paths)} imágenes")
    print(f"{'='*80}")

    results: list[ModelBenchmarkEntry] = []

    for model_cfg in models:
        weight_path = MODELS_DIR / model_cfg["weight"]

        if not weight_path.exists():
            print(f"  ⚠ SKIP {model_cfg['name']}: no {model_cfg['weight']}")
            continue

        print(f"\n  ▶ {model_cfg['name']} [{model_cfg['family']}-{model_cfg['variant']}]")
        model = YOLO(str(weight_path))

        entry = ModelBenchmarkEntry(
            model_name=model_cfg["name"],
            family=model_cfg["family"],
            variant=model_cfg["variant"],
            total_images=len(image_paths),
        )

        # Warmup
        _ = model(str(image_paths[0]), verbose=False)

        for img_path in image_paths:
            t0 = time.perf_counter()
            preds = model(str(img_path), verbose=False)
            t1 = time.perf_counter()
            elapsed = (t1 - t0) * 1000

            img_result: dict[str, Any] = {"image": img_path.name, "time_ms": round(elapsed, 2)}

            if not preds or preds[0].keypoints is None or preds[0].keypoints.data.shape[0] == 0:
                img_result["detected"] = False
                img_result["kp_conf"] = {}
                entry.per_image.append(img_result)
                continue

            kp_data = preds[0].keypoints.data.cpu().numpy()
            conf = preds[0].keypoints.conf.cpu().numpy() if preds[0].keypoints.conf is not None else kp_data[:, :, 2]

            if kp_data.shape[0] == 0:
                img_result["detected"] = False
                entry.per_image.append(img_result)
                continue

            entry.detected_images += 1
            best_person = int(np.argmax(conf.mean(axis=1)))
            kps = kp_data[best_person]  # [9, 3]

            confs_dict = {
                f"K{i}": round(float(kps[i, 2]), 4) for i in range(min(9, len(kps)))
            }
            img_result["detected"] = True
            img_result["kp_conf"] = confs_dict
            img_result["k0_conf"] = round(float(kps[0, 2]), 4)
            img_result["k6_conf"] = round(float(kps[6, 2]), 4)
            img_result["k7_conf"] = round(float(kps[7, 2]), 4)
            entry.per_image.append(img_result)

            print(f"    {img_path.name:35s} {elapsed:6.1f}ms  "
                  f"K0={kps[0,2]:.3f} K6={kps[6,2]:.3f} K7={kps[7,2]:.3f}")

        # Agregar métricas
        entry.detection_rate_pct = round(entry.detected_images / entry.total_images * 100, 1)

        detected = [r for r in entry.per_image if r.get("detected")]
        if detected:
            entry.avg_conf_k2 = round(np.mean([r.get("k0_conf", 0) for r in detected]), 4)
            entry.avg_conf_k4 = round(np.mean([r.get("k6_conf", 0) for r in detected]), 4)
            entry.avg_conf_k7 = round(np.mean([r.get("k7_conf", 0) for r in detected]), 4)
            entry.avg_conf_all = round((entry.avg_conf_k2 + entry.avg_conf_k4 + entry.avg_conf_k7) / 3, 4)

        times = [r["time_ms"] for r in entry.per_image]
        entry.avg_time_ms = round(np.mean(times), 2) if times else 0.0
        entry.total_time_ms = round(sum(times), 2)

        results.append(entry)

    # ── Guardar JSON ─────────────────────────────────────────────────────────
    json_path = output_dir / "model_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "tested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_count": len(image_paths),
            "models": [
                {
                    "model": e.model_name,
                    "family": e.family,
                    "variant": e.variant,
                    "detection_rate": e.detection_rate_pct,
                    "avg_time_ms": e.avg_time_ms,
                    "avg_conf_k2": e.avg_conf_k2,
                    "avg_conf_k4": e.avg_conf_k4,
                    "avg_conf_k7": e.avg_conf_k7,
                    "avg_conf_critical": e.avg_conf_all,
                    "composite_score": e.composite_score,
                    "per_image": e.per_image,
                }
                for e in results
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 JSON → {json_path}")

    # ── Generar gráfica ──────────────────────────────────────────────────────
    _generate_comparison_chart(results, output_dir)

    # Ordenar por score compuesto
    print(f"\n{'='*110}")
    print("RANKING COMPUESTO (Confianza K0+K6+K7 × 50% + Detección × 25% + Velocidad × 25%)")
    print(f"{'='*110}")
    print(f"{'Rank':>4} {'Modelo':<28} {'Familia':>8} {'Var':>4} {'Det%':>6} "
          f"{'ms':>7} {'K0':>7} {'K6':>7} {'K7':>7} {'C.avg':>7} {'SCORE':>7}")
    print("-" * 110)

    for e in results:
        speed_score = max(0, 1.0 - e.avg_time_ms / 100.0)
        detect_score = e.detection_rate_pct / 100.0
        conf_score = e.avg_conf_all
        composite = conf_score * 0.50 + detect_score * 0.25 + speed_score * 0.25
        e.composite_score = round(composite, 4)

    results.sort(key=lambda r: r.composite_score, reverse=True)

    for i, e in enumerate(results, 1):
        print(f"{i:>4} {e.model_name:<28} {e.family:>8} {e.variant:>4} {e.detection_rate_pct:>5.1f}% "
              f"{e.avg_time_ms:>6.1f} {e.avg_conf_k2:>7.4f} {e.avg_conf_k4:>7.4f} "
              f"{e.avg_conf_k7:>7.4f} {e.avg_conf_all:>7.4f} {e.composite_score:>7.4f}")

    # ── Selección TOP 4 con restricción de diversidad (máx 2 por familia) ────
    print(f"\n{'='*110}")
    print("TOP 4 SELECCIONADOS (diversidad: máx 2 por familia)")
    print(f"{'='*110}")

    selected = []
    family_count: dict[str, int] = {}
    for e in results:
        fam = e.family
        if fam not in family_count:
            family_count[fam] = 0
        if family_count[fam] < 2:
            selected.append(e)
            family_count[fam] += 1
            if len(selected) == 4:
                break

    for i, e in enumerate(selected, 1):
        print(f"  #{i} {e.model_name} ({e.family}-{e.variant})  "
              f"SCORE={e.composite_score:.4f}  K6={e.avg_conf_k4:.4f}  {e.avg_time_ms:.1f}ms  det={e.detection_rate_pct:.1f}%")

    return results


def _print_summary_table(results: list[ModelBenchmarkEntry]) -> None:
    """Imprime tabla comparativa en consola."""
    print(f"\n{'='*100}")
    print("TABLA COMPARATIVA — MODELOS SELECCIONADOS")
    print(f"{'='*100}")
    print(f"{'Modelo':<28} {'Familia':>8} {'Detect':>8} {'Tiempo':>8} "
          f"{'K0':>8} {'K6':>8} {'K7':>8} {'CritAvg':>8}")
    print("-" * 100)
    for e in results:
        print(f"{e.model_name:<28} {e.family:>8} {e.detected_images:>3}/{e.total_images:<3} "
              f"{e.avg_time_ms:>6.1f}ms {e.avg_conf_k2:>8.4f} {e.avg_conf_k4:>8.4f} "
              f"{e.avg_conf_k7:>8.4f} {e.avg_conf_all:>8.4f}")


def _generate_comparison_chart(results: list[ModelBenchmarkEntry], output_dir: Path) -> None:
    """
    Genera gráfica de barras: confianza por keypoint crítico + latencia.

    Args:
        results: Resultados del benchmark.
        output_dir: Directorio de salida.
    """
    if not results:
        return

    models_names = [r.model_name.replace("_pose_", "\n").replace("_b", " b").replace("_lr", " lr") for r in results]

    k2_vals = [r.avg_conf_k2 for r in results]
    k4_vals = [r.avg_conf_k4 for r in results]
    k7_vals = [r.avg_conf_k7 for r in results]
    times = [r.avg_time_ms for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfica 1: Confianza por keypoint crítico
    x = np.arange(len(results))
    width = 0.25
    bars1 = ax1.bar(x - width, k2_vals, width, label="K0 Cabeza", color="#2196F3")
    bars2 = ax1.bar(x, k4_vals, width, label="K6 Cervical C7 (pivote)", color="#F44336")
    bars3 = ax1.bar(x + width, k7_vals, width, label="K7 Escápula", color="#4CAF50")

    ax1.set_ylabel("Confianza promedio")
    ax1.set_title("Confianza de Keypoints Críticos por Modelo")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models_names, fontsize=8)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis="y", alpha=0.3)

    # Gráfica 2: Latencia
    bar_colors = ["#FF9800", "#2196F3", "#4CAF50", "#9C27B0"]
    bars = ax2.bar(x, times, color=bar_colors[:len(results)])
    ax2.set_ylabel("Tiempo de inferencia (ms)")
    ax2.set_title("Latencia por Modelo (menor = mejor)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_names, fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Anotar valores en barras
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{t:.1f}ms", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = output_dir / "model_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Gráfica → {chart_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_benchmark()
