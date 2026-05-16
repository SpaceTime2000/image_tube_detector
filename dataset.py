# dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# Handles:
#   1. Reading annotations.csv
#   2. Splitting images into train / val sets
#   3. Cropping individual tube lids from full images
#   4. Augmenting crops (with correct angle transforms)
#   5. Returning (crop_tensor, cos_theta, sin_theta) for training
# ─────────────────────────────────────────────────────────────────────────────

import os
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import config


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load and split annotations
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(csv_path=config.ANNOTATIONS_CSV):
    """
    Returns a DataFrame with columns:
        image, center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h,
        bbox_rotation, angle_deg
    """
    df = pd.read_csv(csv_path)
    print(f"[dataset] Loaded {len(df)} annotations across "
          f"{df['image'].nunique()} images.")
    return df


def split_by_image(df, val_split=config.VAL_SPLIT, seed=config.RANDOM_SEED):
    """
    Split at the IMAGE level (not annotation level).
    This prevents the same image appearing in both train and val,
    which would be data leakage.
    """
    all_images = df['image'].unique()
    np.random.seed(seed)
    np.random.shuffle(all_images)

    n_val = max(1, int(len(all_images) * val_split))
    val_images  = set(all_images[:n_val])
    train_images = set(all_images[n_val:])

    train_df = df[df['image'].isin(train_images)].reset_index(drop=True)
    val_df   = df[df['image'].isin(val_images)].reset_index(drop=True)

    print(f"[dataset] Train: {len(train_images)} images, {len(train_df)} tubes")
    print(f"[dataset] Val:   {len(val_images)} images, {len(val_df)} tubes")
    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Crop helper
# ─────────────────────────────────────────────────────────────────────────────

def crop_lid(image_pil, cx, cy, crop_size=config.CROP_SIZE):
    """
    Crops a square region of size crop_size × crop_size centred on (cx, cy).
    Pads with black if the crop goes outside the image boundary.

    Args:
        image_pil : PIL Image (full overhead image)
        cx, cy    : lid center in pixels
        crop_size : output crop size in pixels

    Returns:
        PIL Image of size (crop_size, crop_size)
    """
    half = crop_size // 2
    left   = int(cx) - half
    top    = int(cy) - half
    right  = left + crop_size
    bottom = top  + crop_size

    # Pad image so we can always crop even near edges
    pad = half + 10
    padded = TF.pad(image_pil, padding=pad, fill=0)

    # Adjust coords after padding
    left   += pad
    top    += pad
    right  += pad
    bottom += pad

    crop = padded.crop((left, top, right, bottom))
    return crop


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Angle utilities
# ─────────────────────────────────────────────────────────────────────────────

def angle_to_cossin(angle_deg):
    """Convert angle in degrees → (cos θ, sin θ) as floats."""
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)


def cossin_to_angle(cos_val, sin_val):
    """Convert (cos θ, sin θ) → angle in [0, 360)."""
    angle = math.degrees(math.atan2(sin_val, cos_val))
    return angle % 360


def circular_angle_error(pred_deg, gt_deg):
    """
    Shortest angular distance between two angles.
    Always in [0, 180].
    """
    diff = abs(pred_deg - gt_deg) % 360
    return min(diff, 360 - diff)


def rotate_angle(angle_deg, rotation_deg):
    """
    When we rotate an image by rotation_deg (counter-clockwise),
    the angle in the image transforms as:
        new_angle = (old_angle + rotation_deg) % 360

    This keeps the angle annotation correct after augmentation.
    """
    return (angle_deg + rotation_deg) % 360


def flip_horizontal_angle(angle_deg):
    """
    Horizontal flip transforms angle as:
        new_angle = (180 - angle_deg) % 360
    Because the x-axis flips, so the direction vector (cos θ, sin θ)
    becomes (-cos θ, sin θ), i.e. angle → 180° - angle.
    """
    return (180 - angle_deg) % 360


def flip_vertical_angle(angle_deg):
    """
    Vertical flip: (cos θ, sin θ) → (cos θ, -sin θ)
    i.e. angle → -angle = 360 - angle
    """
    return (360 - angle_deg) % 360


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class TubeLidDataset(Dataset):
    """
    Each item is one tube lid crop + its angle.

    __getitem__ returns:
        crop_tensor : FloatTensor of shape (3, CROP_SIZE, CROP_SIZE), normalised
        cos_theta   : float
        sin_theta   : float
        angle_deg   : float  (original, for evaluation)
    """

    def __init__(self, df, image_dir=config.DATA_DIR, augment=False):
        self.df        = df
        self.image_dir = image_dir
        self.augment   = augment

        # Standard ImageNet normalisation — used because we start from
        # a pretrained ResNet backbone
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )

        # Cache loaded images in memory (only 70 images, ~200MB max)
        self._image_cache = {}

    def _load_image(self, filename):
        if filename not in self._image_cache:
            path = os.path.join(self.image_dir, filename)
            img  = Image.open(path).convert("RGB")
            self._image_cache[filename] = img
        return self._image_cache[filename]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img       = self._load_image(row['image'])
        cx, cy    = row['center_x'], row['center_y']
        angle_deg = row['angle_deg']

        # ── Crop the lid region ───────────────────────────────────────────────
        crop = crop_lid(img, cx, cy, config.CROP_SIZE)

        # ── Augmentation (training only) ──────────────────────────────────────
        if self.augment:
            crop, angle_deg = self._augment(crop, angle_deg)

        # ── Convert to tensor ─────────────────────────────────────────────────
        crop_tensor = TF.to_tensor(crop)         # [3, H, W], values in [0,1]
        crop_tensor = self.normalize(crop_tensor) # ImageNet normalisation

        cos_t, sin_t = angle_to_cossin(angle_deg)

        return (
            crop_tensor,
            torch.tensor(cos_t, dtype=torch.float32),
            torch.tensor(sin_t, dtype=torch.float32),
            torch.tensor(angle_deg, dtype=torch.float32),
        )

    def _augment(self, crop_pil, angle_deg):
        """
        Apply random augmentations. IMPORTANT: each geometric transform
        must also update angle_deg accordingly.
        """

        # ── Random rotation ───────────────────────────────────────────────────
        # Rotate by a random angle; the lid angle rotates by the same amount.
        if np.random.rand() < 0.8:
            rot = np.random.uniform(0, 360)
            crop_pil  = TF.rotate(crop_pil, rot)   # PIL rotates CCW
            angle_deg = rotate_angle(angle_deg, rot)

        # ── Random horizontal flip ────────────────────────────────────────────
        if np.random.rand() < 0.5:
            crop_pil  = TF.hflip(crop_pil)
            angle_deg = flip_horizontal_angle(angle_deg)

        # ── Random vertical flip ──────────────────────────────────────────────
        if np.random.rand() < 0.5:
            crop_pil  = TF.vflip(crop_pil)
            angle_deg = flip_vertical_angle(angle_deg)

        # ── Colour jitter (does NOT affect angle) ─────────────────────────────
        if np.random.rand() < 0.7:
            jitter = T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.1
            )
            crop_pil = jitter(crop_pil)

        # ── Gaussian blur (does NOT affect angle) ─────────────────────────────
        if np.random.rand() < 0.3:
            crop_pil = crop_pil.filter(
                __import__('PIL').ImageFilter.GaussianBlur(radius=1)
            )

        return crop_pil, angle_deg


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(csv_path=config.ANNOTATIONS_CSV,
                     image_dir=config.DATA_DIR):
    df = load_annotations(csv_path)
    train_df, val_df = split_by_image(df)

    train_ds = TubeLidDataset(train_df, image_dir, augment=True)
    val_ds   = TubeLidDataset(val_df,   image_dir, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader, train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run this file directly to verify data loads
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Running dataset sanity check...")
    print("=" * 60)

    train_loader, val_loader, _, _ = make_dataloaders()

    # Check one batch
    batch = next(iter(train_loader))
    crops, cos_t, sin_t, angles = batch

    print(f"\nBatch shapes:")
    print(f"  crops  : {crops.shape}")       # should be [B, 3, 64, 64]
    print(f"  cos_t  : {cos_t.shape}")       # should be [B]
    print(f"  sin_t  : {sin_t.shape}")       # should be [B]
    print(f"  angles : {angles.shape}")      # should be [B]

    print(f"\nFirst 5 angles in batch: {angles[:5].tolist()}")
    print(f"cos²+sin² (should all be ~1.0): "
          f"{(cos_t**2 + sin_t**2)[:5].tolist()}")

    print(f"\nCrop pixel range: [{crops.min():.2f}, {crops.max():.2f}]")
    print("\n✓ Dataset looks good!")
