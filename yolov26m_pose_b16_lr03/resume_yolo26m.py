""" Resume YOLO26m-pose from epoch 148 -> train 1 more epoch to reach 150
- cache=ram, workers=8, priority=HIGH (all apps closed, full PC power)
- val=False, plots=False for maximum speed
- adaptive batch: 8 -> 4 -> 2 (YOLO26m is 23.5M params, may not fit batch=16 on 6GB VRAM)
"""
from ultralytics import YOLO
import torch
import os
import time
import multiprocessing
import ctypes


def set_high_priority():
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, 0x00000080)
        print("[OK] Process priority = HIGH")
    except Exception as e:
        print(f"[WARN] Could not set priority: {e}")


def train_with_batch(last_pt: str, data_yaml: str, batch: int):
    model = YOLO(last_pt)
    return model.train(
        resume=True,
        batch=batch,
        workers=8,
        cache='ram',
        patience=0,
        amp=True,
        close_mosaic=10,
        deterministic=False,
        verbose=True,
        val=False,
        plots=False,
    )


def main():
    set_high_priority()

    LAST_PT = r"C:\Users\elkaw\Desktop\Vision artificial postura\Entrenamiento real\yolo26m_pose_b16_lr0003-2\weights\last.pt"
    DATA_YAML = r"D:\Documentos\entre\Desk Posture.v2i.yolov8_aug36k\data.yaml"

    assert os.path.isfile(LAST_PT), f"last.pt not found: {LAST_PT}"
    assert os.path.isfile(DATA_YAML), f"data.yaml not found: {DATA_YAML}"

    print(f"[OK] last.pt = {LAST_PT}")
    print(f"[OK] data.yaml = {DATA_YAML}")
    print(f"[OK] CUDA = {torch.cuda.get_device_name(0)}")
    print(f"[OK] VRAM = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
    print(f"[OK] PyTorch = {torch.__version__}")

    t0 = time.time()
    last_error = None

    for b in [8, 4, 2]:
        try:
            print(f"\n[TRY] batch={b}, workers=8, cache=ram, priority=HIGH")
            train_with_batch(LAST_PT, DATA_YAML, b)
            elapsed = time.time() - t0
            print(f"\n{'='*60}")
            print("TRAINING COMPLETE — epoch 150 reached")
            print(f"Total time: {elapsed/60:.1f} minutes")
            print(f"Last model: {LAST_PT}")
            print(f"{'='*60}")
            return
        except (RuntimeError, MemoryError) as e:
            msg = str(e).lower()
            if 'out of memory' in msg or 'cuda error' in msg or isinstance(e, MemoryError):
                last_error = e
                print(f"[FALLBACK] batch={b} failed ({e}). Trying smaller batch...")
                torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError(f"All batch sizes failed. Last error: {last_error}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
