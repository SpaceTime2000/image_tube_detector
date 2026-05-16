# pipeline.py

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from ultralytics import YOLO

import config
from dataset import (
    load_annotations, split_by_image,
    crop_lid, circular_angle_error
)
from model import TubeOrientationModel

# ── Absolute paths ────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "outputs", "detector",
                             "yolov8n_tubes", "weights", "best.pt")
# OBB model outputs oriented boxes — we extract center from obb.xywhr
ANGLE_CKPT   = os.path.join(BASE_DIR, "outputs", "best_model.pth")


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────

def load_detector():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Detector weights not found at:\n  {WEIGHTS_PATH}\n"
            "Run detect_train.py first."
        )
    model = YOLO(WEIGHTS_PATH)
    print(f"[pipeline] Loaded detector from {WEIGHTS_PATH}")
    return model


def load_angle_model(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(ANGLE_CKPT):
        raise FileNotFoundError(
            f"Angle model not found at:\n  {ANGLE_CKPT}\n"
            "Run train.py first."
        )
    checkpoint = torch.load(ANGLE_CKPT, map_location=device)
    cfg = checkpoint.get('config', {})
    model = TubeOrientationModel(
        backbone=cfg.get('backbone', config.BACKBONE),
        pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"[pipeline] Loaded angle model from {ANGLE_CKPT}")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# Rotated box corners
# ─────────────────────────────────────────────────────────────────────────────

def get_rotated_box_corners(cx, cy, w, h, angle_deg):
    """
    Compute 4 corners of a rotated bounding box.

    Args:
        cx, cy    : center of the box in pixels
        w, h      : width and height of the box
        angle_deg : rotation angle in degrees (clockwise, as in bbox_rotation)

    Returns:
        corners : np.array of shape (4, 2) — [top-left, top-right,
                                               bottom-right, bottom-left]
    """
    hw = w / 2
    hh = h / 2

    # Corners relative to center before rotation
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ])

    # Rotation matrix — bbox_rotation is clockwise so negate for standard math
    rad = math.radians(angle_deg)  # CCW confirmed
    R = np.array([
        [math.cos(rad), -math.sin(rad)],
        [math.sin(rad),  math.cos(rad)],
    ])

    rotated = (R @ corners.T).T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


# ─────────────────────────────────────────────────────────────────────────────
# Predict angle
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
# IoU
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Match predictions to GT
# ─────────────────────────────────────────────────────────────────────────────

def match_predictions_to_gt(pred_boxes, gt_boxes,
                             iou_threshold=config.IOU_THRESHOLD):
    matched_gt = set()
    matches    = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = iou_threshold
        best_gt  = None
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt  = gt_idx
        if best_gt is not None:
            matches.append((pred_idx, best_gt))
            matched_gt.add(best_gt)

    unmatched_preds = [i for i in range(len(pred_boxes))
                       if i not in {m[0] for m in matches}]
    unmatched_gts   = [i for i in range(len(gt_boxes))
                       if i not in {m[1] for m in matches}]
    return matches, unmatched_preds, unmatched_gts


# ─────────────────────────────────────────────────────────────────────────────
# Process one image
# ─────────────────────────────────────────────────────────────────────────────

def process_image(img_path, gt_rows, detector, angle_model, device,
                  conf_threshold=0.25):
    img_pil = Image.open(img_path).convert("RGB")

    # Step 1: detect with YOLO
    yolo_results = detector(img_path, conf=conf_threshold, verbose=False)[0]
    pred_boxes   = []
    pred_centers = []
    pred_confs   = []
    if yolo_results.boxes is not None and len(yolo_results.boxes) > 0:
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            pred_boxes.append((x1, y1, x2, y2))
            pred_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            pred_confs.append(conf)

    # Step 2: predict angle for each detection
    pred_angles = []
    for cx, cy in pred_centers:
        angle = predict_angle(angle_model, img_pil, cx, cy, device)
        pred_angles.append(angle)

    # Step 3: build GT — use center_x/center_y (correct even for rotated boxes)
    gt_boxes   = []
    gt_angles  = []
    gt_centers   = []
    gt_dims      = []   # tight box dims recovered from screen enclosure + rotation
    gt_rotations = []   # for drawing rotated box
    for _, row in gt_rows.iterrows():
        # IoU matching uses screen enclosure box
        x1 = row['bbox_x']
        y1 = row['bbox_y']
        x2 = row['bbox_x'] + row['bbox_w']
        y2 = row['bbox_y'] + row['bbox_h']
        gt_boxes.append((x1, y1, x2, y2))
        gt_angles.append(row['angle_deg'])
        # Visualisation: recover tight box dims from screen enclosure + rotation
        gt_centers.append((row['center_x'], row['center_y']))
        rad   = math.radians(row['bbox_rotation'])
        c, s  = abs(math.cos(rad)), abs(math.sin(rad))
        denom = c**2 - s**2
        if abs(denom) > 0.05:  # away from 45 degrees
            w_tight = (row['bbox_w']*c - row['bbox_h']*s) / denom
            h_tight = (row['bbox_h']*c - row['bbox_w']*s) / denom
        else:
            # Near 45 degrees — fall back to screen rect dims
            w_tight, h_tight = row['bbox_w'], row['bbox_h']
        gt_dims.append((w_tight, h_tight))
        gt_rotations.append(row['bbox_rotation'])

    # Step 4: match
    matches, fp_idxs, fn_idxs = match_predictions_to_gt(pred_boxes, gt_boxes)

    # Step 5: angle errors for matched pairs
    angle_errors = []
    for pred_idx, gt_idx in matches:
        err = circular_angle_error(pred_angles[pred_idx], gt_angles[gt_idx])
        angle_errors.append(err)

    return {
        'image':         os.path.basename(img_path),
        'n_gt':          len(gt_boxes),
        'n_pred':        len(pred_boxes),
        'n_tp':          len(matches),
        'n_fp':          len(fp_idxs),
        'n_fn':          len(fn_idxs),
        'angle_errors':  angle_errors,
        'pred_boxes':    pred_boxes,
        'pred_centers':  pred_centers,
        'pred_angles':   pred_angles,
        'pred_confs':    pred_confs,
        'gt_boxes':      gt_boxes,
        'gt_angles':     gt_angles,
        'gt_centers':    gt_centers,
        'gt_dims':       gt_dims,
        'gt_rotations':  gt_rotations,
        'matches':       matches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_pipeline_metrics(all_results):
    total_tp   = sum(r['n_tp'] for r in all_results)
    total_fp   = sum(r['n_fp'] for r in all_results)
    total_fn   = sum(r['n_fn'] for r in all_results)
    all_errors = [e for r in all_results for e in r['angle_errors']]

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_tp + total_fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    mae       = np.mean(all_errors)   if all_errors else float('nan')
    median    = np.median(all_errors) if all_errors else float('nan')
    w15       = np.mean([e < 15 for e in all_errors]) * 100 if all_errors else 0
    w30       = np.mean([e < 30 for e in all_errors]) * 100 if all_errors else 0

    return {
        'total_gt':       total_tp + total_fn,
        'total_pred':     total_tp + total_fp,
        'total_tp':       total_tp,
        'total_fp':       total_fp,
        'total_fn':       total_fn,
        'precision':      precision,
        'recall':         recall,
        'f1':             f1,
        'mae_deg':        mae,
        'median_err_deg': median,
        'within_15deg_%': w15,
        'within_30deg_%': w30,
    }


def print_pipeline_metrics(metrics):
    print()
    print("=" * 55)
    print("  End-to-End Pipeline Results")
    print("=" * 55)
    print(f"  GT tubes total       : {metrics['total_gt']}")
    print(f"  Predicted tubes      : {metrics['total_pred']}")
    print(f"  True Positives  (TP) : {metrics['total_tp']}")
    print(f"  False Positives (FP) : {metrics['total_fp']}")
    print(f"  False Negatives (FN) : {metrics['total_fn']}")
    print()
    print(f"  Precision            : {metrics['precision']:.3f}")
    print(f"  Recall               : {metrics['recall']:.3f}")
    print(f"  F1 Score             : {metrics['f1']:.3f}")
    print()
    print(f"  Mean Angle Error     : {metrics['mae_deg']:.2f}°")
    print(f"  Median Angle Error   : {metrics['median_err_deg']:.2f}°")
    print(f"  Within 15°           : {metrics['within_15deg_%']:.1f}%")
    print(f"  Within 30°           : {metrics['within_30deg_%']:.1f}%")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Visualise — GT boxes are drawn as proper rotated polygons
# ─────────────────────────────────────────────────────────────────────────────

def visualise_result(result, save_dir=None):
    img_path = os.path.join(config.DATA_DIR, result['image'])
    img_pil  = Image.open(img_path).convert("RGB")

    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    ax.imshow(img_pil)

    # ── GT boxes — proper rotated polygons ──────────────────────────────────
    for angle, center, dims, rot in zip(
            result['gt_angles'], result['gt_centers'],
            result['gt_dims'],   result['gt_rotations']):

        cx, cy = center
        w,  h  = dims

        corners = get_rotated_box_corners(cx, cy, w, h, rot)
        poly = Polygon(corners, linewidth=2, edgecolor='limegreen',
                       facecolor='none', linestyle='--')
        ax.add_patch(poly)

        # GT angle arrow from center
        rad = math.radians(angle)
        ax.annotate('', xy=(cx + 25*math.cos(rad), cy - 25*math.sin(rad)),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color='limegreen', lw=2))

    # ── Predicted boxes — axis-aligned (YOLO output) ──────────────────────────
    matched_pred_idxs = {m[0] for m in result['matches']}
    for i, (box, angle, conf) in enumerate(zip(
            result['pred_boxes'], result['pred_angles'], result['pred_confs'])):
        x1, y1, x2, y2 = box
        color = 'tomato' if i in matched_pred_idxs else 'yellow'
        label = 'TP' if i in matched_pred_idxs else 'FP'

        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        cx, cy = (x1+x2)/2, (y1+y2)/2
        rad = math.radians(angle)
        ax.annotate('', xy=(cx + 25*math.cos(rad), cy - 25*math.sin(rad)),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(x1, y1-4, f'{label} {conf:.2f}', color=color, fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # ── Angle errors for matched pairs ────────────────────────────────────────
    for pred_idx, gt_idx in result['matches']:
        err = circular_angle_error(result['pred_angles'][pred_idx],
                                   result['gt_angles'][gt_idx])
        b = result['pred_boxes'][pred_idx]
        cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        c = 'lime' if err < 15 else ('orange' if err < 30 else 'red')
        ax.text(cx+5, cy+5, f'{err:.0f}°', color=c, fontsize=9,
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(edgecolor='limegreen', facecolor='none', linestyle='--', label='GT (rotated box)'),
        Patch(edgecolor='tomato',    facecolor='none', label='Predicted TP'),
        Patch(edgecolor='yellow',    facecolor='none', label='Predicted FP'),
    ], loc='upper right', facecolor='black', labelcolor='white')

    ax.set_title(
        f"{result['image']}  |  "
        f"TP={result['n_tp']}  FP={result['n_fp']}  FN={result['n_fn']}"
    )
    ax.axis('off')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        name = os.path.splitext(result['image'])[0]
        path = os.path.join(save_dir, f"{name}_pipeline.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("Running full end-to-end pipeline")
    print("=" * 60)

    device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector            = load_detector()
    angle_model, device = load_angle_model(device)

    df = load_annotations()
    _, val_df = split_by_image(df)

    val_images = val_df['image'].unique()
    print(f"\n[pipeline] Processing {len(val_images)} validation images...")

    all_results = []
    viz_dir     = os.path.join(BASE_DIR, "outputs", "pipeline_viz")

    for img_name in val_images:
        img_path = os.path.join(config.DATA_DIR, img_name)
        gt_rows  = val_df[val_df['image'] == img_name]

        result = process_image(img_path, gt_rows, detector, angle_model, device)
        all_results.append(result)

        print(f"  {img_name:45s}  "
              f"GT={result['n_gt']}  Pred={result['n_pred']}  "
              f"TP={result['n_tp']}  FP={result['n_fp']}  FN={result['n_fn']}")

        visualise_result(result, save_dir=viz_dir)

    metrics = compute_pipeline_metrics(all_results)
    print_pipeline_metrics(metrics)
    print(f"\n[pipeline] Visualisations saved to {viz_dir}")
    return metrics, all_results


if __name__ == "__main__":
    run_pipeline()