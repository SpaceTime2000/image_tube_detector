# visualize.py
# ─────────────────────────────────────────────────────────────────────────────
# Draw predictions on full images so you can SEE what the model is doing.
# For each validation image, draws:
#   - Green arrow : ground truth angle (joint → tab direction)
#   - Red arrow   : predicted angle
#   - Circle      : lid center
#   - Text        : error in degrees
# ─────────────────────────────────────────────────────────────────────────────

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config
from dataset import load_annotations, split_by_image, crop_lid
from model import TubeOrientationModel


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get('config', {})
    model = TubeOrientationModel(
        backbone=cfg.get('backbone', config.BACKBONE),
        pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def predict_angle_for_crop(model, img_pil, cx, cy, device):
    """Predict angle for a single tube crop."""
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
    crop   = crop_lid(img_pil, cx, cy, config.CROP_SIZE)
    crop_t = TF.to_tensor(crop)
    crop_t = normalize(crop_t).unsqueeze(0).to(device)

    with torch.no_grad():
        cossin = model(crop_t)
        angle  = torch.atan2(cossin[0, 1], cossin[0, 0]).item()
        angle  = math.degrees(angle) % 360
    return angle


def draw_angle_arrow(ax, cx, cy, angle_deg, length=25, color='green', lw=2):
    """
    Draw an arrow from center (cx, cy) in direction of angle_deg.
    angle_deg=0 points right (+x), increases counter-clockwise.
    """
    rad = math.radians(angle_deg)
    dx  = length * math.cos(rad)
    dy  = -length * math.sin(rad)   # flip y because image y-axis points down
    ax.annotate(
        '', xy=(cx + dx, cy + dy), xytext=(cx, cy),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw)
    )


def visualize_image(model, img_pil, tubes_df, device, save_path=None):
    """
    Draw GT and predicted angles for all tubes in one image.

    tubes_df : subset of annotations for this specific image
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img_pil)

    for _, row in tubes_df.iterrows():
        cx, cy    = row['center_x'], row['center_y']
        gt_angle  = row['angle_deg']
        pred_angle = predict_angle_for_crop(model, img_pil, cx, cy, device)

        err = abs(pred_angle - gt_angle) % 360
        err = min(err, 360 - err)

        # Circle at center
        circle = plt.Circle((cx, cy), radius=8, color='white',
                              fill=True, alpha=0.8, linewidth=1.5, zorder=3)
        ax.add_patch(circle)

        # GT arrow (green)
        draw_angle_arrow(ax, cx, cy, gt_angle,
                         length=30, color='limegreen', lw=2.5)

        # Predicted arrow (red)
        draw_angle_arrow(ax, cx, cy, pred_angle,
                         length=30, color='tomato', lw=2.5)

        # Error text
        color = 'lime' if err < 15 else ('orange' if err < 30 else 'red')
        ax.text(cx + 10, cy - 10, f'{err:.0f}°',
                color=color, fontsize=9, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, pad=2))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='limegreen', lw=2, label='Ground Truth'),
        Line2D([0], [0], color='tomato',    lw=2, label='Predicted'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              facecolor='black', labelcolor='white')

    ax.set_title(os.path.basename(tubes_df.iloc[0]['image']), fontsize=11)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_crops(model, img_pil, tubes_df, device, save_path=None):
    """
    Show each individual crop with GT vs predicted angle side by side.
    Useful for debugging individual tube predictions.
    """
    n = len(tubes_df)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))
    if n == 1:
        axes = [axes]

    for i, (_, row) in enumerate(tubes_df.iterrows()):
        crop = crop_lid(img_pil, row['center_x'], row['center_y'], config.CROP_SIZE)
        pred = predict_angle_for_crop(model, img_pil,
                                       row['center_x'], row['center_y'], device)
        err  = abs(pred - row['angle_deg']) % 360
        err  = min(err, 360 - err)

        axes[i].imshow(crop)
        axes[i].set_title(
            f"GT: {row['angle_deg']:.0f}°\n"
            f"Pred: {pred:.0f}°\n"
            f"Err: {err:.0f}°",
            fontsize=8,
            color='lime' if err < 15 else ('orange' if err < 30 else 'red')
        )
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_all_val_images(n_images=None):
    """
    Run visualisation on all (or first n_images) validation images.
    Saves output to outputs/viz/
    """
    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print("[visualize] No checkpoint found. Run train.py first.")
        return

    viz_dir = os.path.join(config.OUTPUT_DIR, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = load_model(checkpoint_path, device)

    df = load_annotations()
    _, val_df = split_by_image(df)

    val_images = val_df['image'].unique()
    if n_images:
        val_images = val_images[:n_images]

    print(f"[visualize] Generating visualisations for {len(val_images)} images...")

    for img_name in val_images:
        img_path  = os.path.join(config.DATA_DIR, img_name)
        img_pil   = Image.open(img_path).convert("RGB")
        tubes_df  = val_df[val_df['image'] == img_name]

        # Full image view
        save_path = os.path.join(viz_dir, f"{os.path.splitext(img_name)[0]}_full.png")
        visualize_image(model, img_pil, tubes_df, device, save_path=save_path)

        # Individual crops
        save_path = os.path.join(viz_dir, f"{os.path.splitext(img_name)[0]}_crops.png")
        visualize_crops(model, img_pil, tubes_df, device, save_path=save_path)

        print(f"  Saved → {os.path.basename(save_path)}")

    print(f"\n[visualize] Done. All images saved to {viz_dir}/")


if __name__ == "__main__":
    visualize_all_val_images(n_images=5)  # Start with 5; remove limit for all
