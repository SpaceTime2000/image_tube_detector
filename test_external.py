# test_external.py
# ─────────────────────────────────────────────────────────────────────────────
# Run the full pipeline on external images with no ground truth.
# Just visual confirmation — draws detected tubes and predicted angles.
#
# Put your test images in:
#   ZeonSystems_Assignment1/test_images_external/
#
# Run:
#   python test_external.py
#
# Output saved to:
#   outputs/external_viz/
# ─────────────────────────────────────────────────────────────────────────────

import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from ultralytics import YOLO

import config
from dataset import crop_lid
from model import TubeOrientationModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEST_DIR      = os.path.join(BASE_DIR, "test_images_external")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "external_viz")
WEIGHTS_PATH  = os.path.join(BASE_DIR, "outputs", "detector",
                              "yolov8n_tubes", "weights", "best.pt")
ANGLE_CKPT    = os.path.join(BASE_DIR, "outputs", "best_model.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detector = YOLO(WEIGHTS_PATH)
    print(f"[test] Loaded detector from {WEIGHTS_PATH}")

    checkpoint = torch.load(ANGLE_CKPT, map_location=device)
    cfg = checkpoint.get('config', {})
    angle_model = TubeOrientationModel(
        backbone=cfg.get('backbone', config.BACKBONE),
        pretrained=False
    ).to(device)
    angle_model.load_state_dict(checkpoint['model_state'])
    angle_model.eval()
    print(f"[test] Loaded angle model from {ANGLE_CKPT}")

    return detector, angle_model, device


# ─────────────────────────────────────────────────────────────────────────────
# Predict angle for a single crop
# ─────────────────────────────────────────────────────────────────────────────

_normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)

def predict_angle(angle_model, img_pil, cx, cy, device):
    crop   = crop_lid(img_pil, cx, cy, config.CROP_SIZE)
    crop_t = TF.to_tensor(crop)
    crop_t = _normalize(crop_t).unsqueeze(0).to(device)
    with torch.no_grad():
        cossin = angle_model(crop_t)
        angle  = torch.atan2(cossin[0, 1], cossin[0, 0]).item()
        angle  = math.degrees(angle) % 360
    return angle


# ─────────────────────────────────────────────────────────────────────────────
# Process one image
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_path, detector, angle_model, device, conf_threshold=0.25):
    img_pil = Image.open(img_path).convert("RGB")

    # Detect
    results = detector(img_path, conf=conf_threshold, verbose=False)[0]

    detections = []
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cx   = (x1 + x2) / 2
            cy   = (y1 + y2) / 2
            angle = predict_angle(angle_model, img_pil, cx, cy, device)
            detections.append({
                'box':   (x1, y1, x2, y2),
                'cx':    cx,
                'cy':    cy,
                'conf':  conf,
                'angle': angle,
            })

    return img_pil, detections


# ─────────────────────────────────────────────────────────────────────────────
# Visualise
# ─────────────────────────────────────────────────────────────────────────────

def visualise(img_path, img_pil, detections, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    ax.imshow(img_pil)

    for det in detections:
        x1, y1, x2, y2 = det['box']
        cx, cy = det['cx'], det['cy']
        angle  = det['angle']
        conf   = det['conf']

        # Bounding box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='tomato', facecolor='none'
        )
        ax.add_patch(rect)

        # Angle arrow
        rad = math.radians(angle)
        ax.annotate(
            '', xy=(cx + 28*math.cos(rad), cy - 28*math.sin(rad)),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle='->', color='tomato', lw=2.5)
        )

        # Confidence + angle label
        ax.text(
            x1, y1 - 5,
            f'{conf:.2f} | {angle:.0f}°',
            color='tomato', fontsize=8, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, pad=2)
        )

    ax.set_title(
        f"{os.path.basename(img_path)}  |  "
        f"{len(detections)} tube(s) detected",
        fontsize=11
    )
    ax.axis('off')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    name     = os.path.splitext(os.path.basename(img_path))[0]
    savepath = os.path.join(save_dir, f"{name}_result.png")
    plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {savepath}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("External image evaluation")
    print("=" * 60)

    if not os.path.exists(TEST_DIR):
        print(f"ERROR: Test folder not found: {TEST_DIR}")
        print("Create it and add your images there.")
        return

    # Find all images
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.PNG', '*.JPG', '*.WEBP']:
        image_paths.extend(glob.glob(os.path.join(TEST_DIR, ext)))

    if not image_paths:
        print(f"No images found in {TEST_DIR}")
        return

    print(f"Found {len(image_paths)} image(s) in {TEST_DIR}")
    print()

    detector, angle_model, device = load_models()

    for img_path in image_paths:
        print(f"Processing: {os.path.basename(img_path)}")
        img_pil, detections = process_image(img_path, detector, angle_model, device, conf_threshold=0.1)

        print(f"  Detected {len(detections)} tube(s):")
        for i, det in enumerate(detections):
            print(f"    Tube {i+1}: angle={det['angle']:.1f}°  conf={det['conf']:.3f}")

        visualise(img_path, img_pil, detections, OUTPUT_DIR)
        print()

    print(f"All results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()