FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU (not in requirements.txt — installed separately)
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Bake trained model weights into the image (~83 MB total).
# Epoch checkpoints and base pretrained models are excluded via .dockerignore.
COPY yolov11n_pose_b16_lr01/weights/last.pt  yolov11n_pose_b16_lr01/weights/last.pt
COPY yolov8s_pose_b8_lr05/weights/last.pt    yolov8s_pose_b8_lr05/weights/last.pt
COPY yolov5m_pose_b32_lr01/weights/last.pt   yolov5m_pose_b32_lr01/weights/last.pt
COPY yolov26n_pose_b128_lr05/weights/last.pt yolov26n_pose_b128_lr05/weights/last.pt

EXPOSE 7860 8765

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.ui.app"]
