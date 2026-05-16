# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Computes all metrics required by the assignment:
#   - Precision, Recall, F1  (detection quality)
#   - Mean Angle Error       (orientation quality)
#   - Per-image breakdown
#
# NOTE: This script operates at the FULL IMAGE level, not the crop level.
# It simulates a real pipeline: given a full image, find tubes + their angles.
#
# For detection we use a simple heuristic: since we know ground truth centers
# exist, we use the GT centers to crop (simulating a perfect detector) and
# only evaluate the orientation model. This isolates orientation quality.
#
# A "detected" tube = GT center within CENTER_DIST_THRESHOLD pixels of a
# predicted center. Since we're using GT crops here, every tube is "detected"
# and precision/recall reflect how well the angle matches.
#
# To evaluate a real end-to-end detector, swap crop_lid(GT center) with
# your actual detector output.
# ─────────────────────────────────────────────────────────────────────────────

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import config
from dataset import (
    load_annotations, split_by_image,
    TubeLidDataset, crop_lid,
    angle_to_cossin, cossin_to_angle, circular_angle_error
)
from model import TubeOrientationModel


# ─────────────────────────────────────────────────────────────────────────────
# Load trained model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get('config', {})

    model = TubeOrientationModel(
        backbone=cfg.get('backbone', config.BACKBONE),
        pretrained=False   # weights come from checkpoint
    ).to(device)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"[evaluate] Loaded model from {checkpoint_path}")
    print(f"           Trained for {checkpoint['epoch']} epochs")
    print(f"           Best val angle error: {checkpoint['val_angle_err']:.2f}°")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Predict angles for all tubes in a DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def predict_all(model, df, device):
    """
    For each row in df (one tube annotation), crop the lid and predict angle.

    Returns a copy of df with additional columns:
        pred_angle_deg   : model's predicted angle
        angle_error_deg  : circular angular error vs ground truth
    """
    from PIL import Image
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    results = []
    image_cache = {}

    model.eval()
    with torch.no_grad():
        for _, row in df.iterrows():
            # Load image (cached)
            img_path = os.path.join(config.DATA_DIR, row['image'])
            if row['image'] not in image_cache:
                image_cache[row['image']] = Image.open(img_path).convert("RGB")
            img = image_cache[row['image']]

            # Crop
            crop = crop_lid(img, row['center_x'], row['center_y'], config.CROP_SIZE)
            crop_t = TF.to_tensor(crop)
            crop_t = normalize(crop_t).unsqueeze(0).to(device)  # (1, 3, H, W)

            # Predict
            pred_cossin = model(crop_t)
            pred_angle  = torch.atan2(pred_cossin[0, 1],
                                       pred_cossin[0, 0]).item()
            pred_angle  = math.degrees(pred_angle) % 360

            # Error
            err = circular_angle_error(pred_angle, row['angle_deg'])

            results.append({
                **row.to_dict(),
                'pred_angle_deg':  pred_angle,
                'angle_error_deg': err,
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Compute detection metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results_df, angle_tolerance=config.ANGLE_TOLERANCE_DEG):
    """
    Given a results DataFrame (one row per tube, with pred_angle_deg and
    angle_error_deg columns), compute:

        Precision : fraction of predicted tubes with angle error < tolerance
        Recall    : fraction of GT tubes successfully predicted within tolerance
        F1        : harmonic mean of precision and recall
        MAE       : mean absolute circular angle error (degrees)
        Median AE : median circular angle error
        Within-15 : fraction of tubes with error < 15°
        Within-30 : fraction of tubes with error < 30°

    NOTE: Since we use GT crops (perfect detection), precision = recall here.
    In a real end-to-end system, some tubes would be missed (recall drops) or
    false detections would appear (precision drops).
    """
    n_gt    = len(results_df)
    errors  = results_df['angle_error_deg'].values

    # A tube is a "correct" prediction if angle error < tolerance
    n_correct = (errors < angle_tolerance).sum()

    precision = n_correct / n_gt   # = recall when using GT crops
    recall    = n_correct / n_gt
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    mae    = errors.mean()
    median = np.median(errors)
    w15    = (errors < 15).mean() * 100
    w30    = (errors < 30).mean() * 100

    metrics = {
        'n_tubes':        n_gt,
        'n_correct':      int(n_correct),
        'precision':      precision,
        'recall':         recall,
        'f1':             f1,
        'mae_deg':        mae,
        'median_err_deg': median,
        'within_15deg_%': w15,
        'within_30deg_%': w30,
    }
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Print a nice metrics table
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics(metrics, split_name="Validation"):
    print()
    print("=" * 50)
    print(f"  Results — {split_name} Set")
    print("=" * 50)
    print(f"  Total tubes           : {metrics['n_tubes']}")
    print(f"  Correct predictions   : {metrics['n_correct']}")
    print()
    print(f"  Precision             : {metrics['precision']:.3f}")
    print(f"  Recall                : {metrics['recall']:.3f}")
    print(f"  F1 Score              : {metrics['f1']:.3f}")
    print()
    print(f"  Mean Angle Error      : {metrics['mae_deg']:.2f}°")
    print(f"  Median Angle Error    : {metrics['median_err_deg']:.2f}°")
    print(f"  Within 15°            : {metrics['within_15deg_%']:.1f}%")
    print(f"  Within 30°            : {metrics['within_30deg_%']:.1f}%")
    print("=" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# Plot error distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_distribution(results_df, split_name="val"):
    errors = results_df['angle_error_deg'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(errors, bins=36, range=(0, 180),
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=15, color='red',    linestyle='--', label='15° threshold')
    axes[0].axvline(x=30, color='orange', linestyle='--', label='30° threshold')
    axes[0].axvline(x=np.mean(errors), color='green',
                    linestyle='-', label=f'Mean={np.mean(errors):.1f}°')
    axes[0].set_xlabel('Angle Error (degrees)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Angle Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative
    sorted_errors = np.sort(errors)
    cumulative    = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1].plot(sorted_errors, cumulative * 100, color='steelblue', linewidth=2)
    axes[1].axvline(x=15, color='red',    linestyle='--', label='15°')
    axes[1].axvline(x=30, color='orange', linestyle='--', label='30°')
    axes[1].set_xlabel('Angle Error (degrees)')
    axes[1].set_ylabel('Cumulative % of Tubes')
    axes[1].set_title('Cumulative Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 180)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, f'error_distribution_{split_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[evaluate] Saved error distribution plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-image breakdown
# ─────────────────────────────────────────────────────────────────────────────

def per_image_report(results_df):
    """Print per-image mean angle error — useful for spotting bad images."""
    print("\nPer-image mean angle error:")
    print("-" * 40)
    grouped = results_df.groupby('image')['angle_error_deg'].agg(['mean', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    for img, row in grouped.iterrows():
        flag = "  ← HIGH ERROR" if row['mean'] > 30 else ""
        print(f"  {img:40s}  MAE={row['mean']:5.1f}°  n={int(row['count'])}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')

    if not os.path.exists(checkpoint_path):
        print(f"[evaluate] ERROR: No checkpoint found at {checkpoint_path}")
        print("           Run train.py first.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(checkpoint_path, device)

    df = load_annotations()
    train_df, val_df = split_by_image(df)

    # Evaluate on validation set
    print("\n[evaluate] Running predictions on validation set...")
    val_results = predict_all(model, val_df, device)

    val_metrics = compute_metrics(val_results)
    print_metrics(val_metrics, "Validation")
    plot_error_distribution(val_results, "val")
    per_image_report(val_results)

    # Save results CSV
    results_path = os.path.join(config.OUTPUT_DIR, 'val_results.csv')
    val_results.to_csv(results_path, index=False)
    print(f"\n[evaluate] Full results saved → {results_path}")

    return val_metrics, val_results


if __name__ == "__main__":
    evaluate()
