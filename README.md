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

---

## External Image Testing (No Ground Truth)

To assess generalization, the pipeline was tested on external images sourced from the web — different cameras, viewpoints, rack colors, and lighting conditions. No ground truth labels were available; results were evaluated qualitatively.

### Observations

**Detection generalizes reasonably well.** The model successfully detected tubes across orange, blue, and red racks, and handled different tube densities. The most impressive result was `reversible-microtube-racks-3_1.webp` where 26 tubes were detected across a crowded angled rack.

**Angle predictions are unreliable on non-overhead images.** The orientation model was trained exclusively on directly overhead images. On angled shots (product photos taken from the side), the predicted angles are not meaningful — the joint-to-tab direction changes with viewpoint. On near-overhead images the angles look plausible but cannot be verified without ground truth.

**Low confidence on external images.** Detection confidence scores (shown before the `|` in labels) are noticeably lower than on validation data (0.1–0.5 vs 0.6–0.75), indicating the model is less certain. This is expected distribution shift behavior.

**Some tubes missed entirely.** Images with tubes at the very edge of the frame, heavily occluded tubes, or very small tubes (far from camera) are often missed. The model was trained on tubes that occupied a consistent portion of the 640×480 frame.

### Example Results

| Image | Tubes in Image | Tubes Detected | Notes |
|-------|---------------|----------------|-------|
| 1.5mL-Minicentrifuge-Rack | 6 | 6 | Near-overhead, good detection |
| 71CtpBMMlaL (blue rack) | 8 | 4 | Missed orange-capped tubes (different appearance) |
| 71KTO+kuLpL (orange rack) | 2 | 3 | 1 FP, angled shot |
| reversible-microtube-racks-3 | ~40 | 26 | Best external result, crowded rack |
| reversible-microtube-racks-6 | ~12 | 6 | Only detected rightmost column |

### Key Takeaway

The system is well-calibrated for its training distribution (overhead lab images, specific rack types). Generalization to product photography and angled shots is limited — primarily a data diversity problem rather than an architectural one.
