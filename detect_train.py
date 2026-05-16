# detect_train.py

import os
from ultralytics import YOLO

# ── Absolute paths — no more confusion ───────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
YOLO_YAML    = os.path.join(BASE_DIR, "yolo_dataset", "dataset.yaml")
DETECTOR_OUT = os.path.join(BASE_DIR, "outputs", "detector")
WEIGHTS_PATH = os.path.join(DETECTOR_OUT, "yolov8n_tubes", "weights", "best.pt")


def train_detector():
    print("=" * 60)
    print("Training YOLOv8 detector")
    print(f"YAML path : {YOLO_YAML}")
    print(f"Output dir: {DETECTOR_OUT}")
    print("=" * 60)

    if not os.path.exists(YOLO_YAML):
        print("ERROR: YOLO dataset not found. Run prepare_yolo.py first.")
        return None

    model = YOLO("yolov8n.pt")

    model.train(
        data     = YOLO_YAML,
        epochs   = 50,
        imgsz    = 640,
        batch    = 8,
        patience = 20,
        project  = DETECTOR_OUT,
        name     = "yolov8n_tubes",
        exist_ok = True,        # overwrite previous run instead of making yolov8n_tubes2
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        degrees  = 45.0,
        translate= 0.2,
        scale    = 0.5,
        fliplr   = 0.5,
        flipud   = 0.3,
        mosaic   = 0.5,
        mixup    = 0.1,
        save     = True,
        plots    = True,
        verbose  = True,
    )

    print()
    print("=" * 60)
    if os.path.exists(WEIGHTS_PATH):
        print(f"SUCCESS! Weights saved to:\n  {WEIGHTS_PATH}")
    else:
        print("WARNING: best.pt not found after training — check YOLO output above.")
    print("=" * 60)

    return WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else None


def validate_detector():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"ERROR: No weights found at {WEIGHTS_PATH}")
        return

    print("\nValidating detector...")
    model   = YOLO(WEIGHTS_PATH)
    metrics = model.val(data=YOLO_YAML)

    print()
    print("=" * 60)
    print("  Detection Metrics")
    print("=" * 60)
    print(f"  Precision : {metrics.box.mp:.3f}")
    print(f"  Recall    : {metrics.box.mr:.3f}")
    print(f"  mAP50     : {metrics.box.map50:.3f}")
    print(f"  mAP50-95  : {metrics.box.map:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    weights = train_detector()
    if weights:
        validate_detector()