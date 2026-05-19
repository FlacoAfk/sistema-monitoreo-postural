# posture_monitor — Sistema de Monitoreo Postural
# Universidad Surcolombiana, 2026
#
# Layered package structure:
#   src/core/       — Domain logic (PostureAnalyzer, CPI calculation)
#   src/inference/  — ML runtime (YOLO inference, keypoint extraction)
#   src/ui/         — Presentation (Gradio app, HTML/CSS/JS)
#   src/tools/      — Utilities (model_benchmark)
#   src/tests/      — Test suite
#   src/models/     — Benchmark outputs
