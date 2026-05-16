# Microcentrifuge Tube Detector

End-to-end system for detecting microcentrifuge tube positions and orientations from overhead RGB images.

## Results

| Metric | Score |
|--------|-------|
| Precision | 0.987 |
| Recall | 1.000 |
| F1 Score | 0.993 |
| Mean Angle Error | 4.47° |
| Median Angle Error | 3.96° |
| Within 15° | 100.0% |
| Within 30° | 100.0% |

Evaluated on 14 held-out validation images (76 tubes total).

---

## Architecture

Two-stage pipeline:

```
Full image (640×480)
      ↓
YOLOv8-nano (fine-tuned)  →  bounding boxes + centers
      ↓
Crop 64×64 around each center
      ↓
ResNet18 backbone (pretrained ImageNet)
      ↓
Rowdy KNN Head (Jagtap et al. 2022)
      ↓
(cos θ, sin θ)  →  angle via atan2
```

### Stage 1 — Detection (YOLOv8)
- Fine-tuned `yolov8n.pt` on 56 training images
- Heavy augmentation: rotation ±45°, HSV jitter, mosaic, mixup, flips
- Outputs bounding boxes → centers used to crop lids

### Stage 2 — Orientation (ResNet18 + Rowdy Head)
- Pretrained ResNet18 backbone extracts 512-dim feature vector
- Two Rowdy activation layers (Jagtap et al. 2022, Neurocomputing 2022)
- Rowdy activation: `φ(z) = α₁·ReLU(ω₁·z) + Σ αₖ·sin((k-1)·η·ωₖ·z)`
- Output: `(cos θ, sin θ)` — angle recovered via `atan2`
- Loss: `(cos_pred - cos_gt)² + (sin_pred - sin_gt)²` — handles 0°/360° wraparound

### Key Design Decisions
- **Predict (cos θ, sin θ) not raw angle** — MSE on raw angles fails at the 0°/360° boundary. The circular loss handles wraparound naturally.
- **Rowdy activations in the head** — sinusoidal fluctuations injected into FC layers help capture periodic/angular structure and overcome spectral bias.
- **Pretrained backbone** — with only 56 training images, ImageNet features are critical.
- **Split by image not by tube** — prevents data leakage (tubes from the same image could have correlated backgrounds).

---

## Setup

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow albumentations ultralytics
```

Place your data:
```
data/
├── images/          ← 70 PNG images
└── annotations.csv
```

---

## Run Order

```bash
# 1. Train orientation model
python train.py

# 2. Prepare YOLO dataset
python prepare_yolo.py

# 3. Train detector
python detect_train.py

# 4. Full end-to-end evaluation
python pipeline.py

# 5. (Optional) Test on external images
# Place images in test_images_external/
python test_external.py
```

---

## File Structure

```
├── config.py          — all hyperparameters
├── dataset.py         — data loading, augmentation, angle transforms
├── model.py           — ResNet18 backbone + Rowdy KNN head
├── train.py           — orientation model training loop
├── evaluate.py        — orientation-only metrics
├── prepare_yolo.py    — convert CSV annotations to YOLO format
├── detect_train.py    — YOLOv8 fine-tuning
├── pipeline.py        — full end-to-end pipeline + metrics
├── visualize.py       — visualisation utilities
└── test_external.py   — run on external images (no GT needed)
```

---

## Notes on Annotations

Through analysis of the CSV:
- `bbox_rotation` is the tilt of the annotated bounding rectangle (CCW, confirmed visually)
- `bbox_w/h` are the **screen enclosing rectangle** dimensions, not tight box dimensions
- Tight box dimensions can be recovered: `w_tight = (bbox_w·cos - bbox_h·sin) / (cos²-sin²)`
- `center_x/center_y` are always the true lid center — used directly for YOLO labels and cropping
- `angle_deg` is CCW from positive X-axis — verified empirically by visual inspection

---

## References

Jagtap, A.D., Shin, Y., Kawaguchi, K., Karniadakis, G.E. (2022).
*Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions.*
Neurocomputing, 468, 165–180.
