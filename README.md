# Microcentrifuge Tube Detector

Two-stage pipeline for detecting microcentrifuge tube positions and orientations from overhead RGB images. YOLOv8-nano finds the tubes, a ResNet18 + Rowdy activation head predicts the angle.

## Results

| Metric | Score |
|--------|-------|
| Precision | 0.987 |
| Recall | 1.000 |
| F1 | 0.993 |
| Mean Angle Error | 4.47° |
| Median Angle Error | 3.96° |
| Within 15° | 100.0% |
| Within 30° | 100.0% |

14 held-out validation images, 76 tubes total. One false positive in ca8fd5c6-color.png — YOLO detected a circular region (probably a fingerprint smudge) at confidence 0.28, well below the 0.62–0.75 range of true positives in the same image.

---

## Architecture

```
Full image (640×480)
      ↓
YOLOv8-nano (fine-tuned on 56 images)
      ↓
64×64 crop around each detected center
      ↓
ResNet18 backbone (ImageNet pretrained) → 512-dim features
      ↓
Rowdy KNN Head (2 layers)
      ↓
(cos θ, sin θ) → atan2 → angle in [0°, 360°)
```

**Detection:** YOLOv8-nano fine-tuned with rotation ±45°, HSV jitter, mosaic, mixup, and flips. `center_x/center_y` from the CSV used directly as YOLO label centers — computing from `bbox_x + w/2` is wrong for rotated boxes.

**Orientation:** The model predicts `(cos θ, sin θ)` instead of a raw angle. MSE on raw angles breaks at the 0°/360° boundary — 1° and 359° are 2° apart physically but 358° apart numerically. The circular loss `(cos_pred - cos_gt)² + (sin_pred - sin_gt)²` handles this correctly.

The Rowdy activation (Jagtap et al., Neurocomputing 2022) injects trainable sinusoidal terms into each FC layer:

```
φ(z) = α₁·ReLU(ω₁·z) + Σ αₖ·sin((k-1)·η·ωₖ·z)
```

The motivation is that angle prediction has periodic structure, and the sinusoidal terms help avoid spectral bias in the FC head. All α and ω are trainable; initialized so the network starts as a plain ReLU network.

Train/val split is at the image level, not the tube level. Tubes from the same image share background and lighting — splitting by tube would leak that correlation into validation.

---

## Setup

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow albumentations ultralytics
```

```
data/
├── images/          ← 70 PNG images
└── annotations.csv
```

---

## Run Order

```bash
python train.py          # train orientation model
python prepare_yolo.py   # convert annotations to YOLO format
python detect_train.py   # fine-tune YOLOv8
python pipeline.py       # end-to-end evaluation

# optional: test on external images (no ground truth needed)
# add images to test_images_external/
python test_external.py
```

---

## Files

```
config.py          — hyperparameters
dataset.py         — loading, augmentation, angle transforms
model.py           — ResNet18 + Rowdy KNN head
train.py           — orientation training loop
evaluate.py        — orientation-only metrics
prepare_yolo.py    — CSV to YOLO format conversion
detect_train.py    — YOLOv8 fine-tuning
pipeline.py        — end-to-end pipeline and metrics
visualize.py       — draw predictions on images
test_external.py   — run on images without ground truth
```

---

## Annotation Notes

`bbox_w` and `bbox_h` in the CSV are the axis-aligned screen enclosure of the rotated box, not the tight box dimensions. The tight dimensions can be recovered:

```
w_tight = (bbox_w·|cos θ| - bbox_h·|sin θ|) / (cos²θ - sin²θ)
h_tight = (bbox_h·|cos θ| - bbox_w·|sin θ|) / (cos²θ - sin²θ)
```

`bbox_rotation` is counter-clockwise despite the dataset description saying clockwise — verified empirically by visual inspection of predictions. `center_x/center_y` are always the true lid center and are reliable even for rotated boxes.

---

## External Testing

Tested on 7 images from the web with no ground truth. Results evaluated qualitatively.

| Image | Tubes Present | Detected | Notes |
|-------|-------------|----------|-------|
| 1.5mL-Minicentrifuge-Rack | 6 | 6 | Near-overhead, clean detections |
| 71CtpBMMlaL (blue rack) | 8 | 4 | Missed orange-capped tubes entirely |
| 71KTO+kuLpL (orange rack) | 2 | 3 | 1 FP, angled product shot |
| microcentrifuge-tube-stand-96-well | 2 | 4 | 2 FPs, dark/blue tinted image |
| mtr-7 (orange rack, side view) | 15 | 6 | Angled shot, most tubes missed |
| reversible-microtube-racks-3 | 60 | 26 | Best result, crowded rack |
| reversible-microtube-racks-6 | 16 | 6 | Detected only partial column |

Detection confidence on external images runs 0.1–0.5 vs 0.6–0.75 on validation data. Angle predictions on angled shots aren't meaningful — the model learned joint-to-tab direction from a fixed overhead viewpoint, so perspective changes break orientation entirely. The main detection failure mode is appearance: the model missed orange-capped tubes in one image because they look nothing like the clear lids in training.

---

## Reference

Jagtap, A.D., Shin, Y., Kawaguchi, K., Karniadakis, G.E. (2022). Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions. *Neurocomputing*, 468, 165–180.
