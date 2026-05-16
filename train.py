# train.py
# ─────────────────────────────────────────────────────────────────────────────
# Full training loop with:
#   - Learning rate annealing (as recommended in the paper for Rowdy nets)
#   - Per-epoch train + val loss logging
#   - Saving the best model checkpoint
#   - Plotting the loss curve at the end
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import config
from dataset import make_dataloaders
from model import TubeOrientationModel, orientation_loss


# ─────────────────────────────────────────────────────────────────────────────
# Device setup
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[train] Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple Silicon
        print("[train] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[train] Using CPU (training will be slow)")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    """
    Runs one full pass through the training set.
    Returns mean loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for crops, cos_gt, sin_gt, _ in loader:
        crops  = crops.to(device)
        cos_gt = cos_gt.to(device)
        sin_gt = sin_gt.to(device)

        optimizer.zero_grad()

        pred_cossin = model(crops)                          # (B, 2)
        loss        = orientation_loss(pred_cossin, cos_gt, sin_gt)

        loss.backward()

        # Gradient clipping — especially useful for Rowdy's sinusoidal params
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, loader, device):
    """
    Runs one full pass through the validation set without gradients.
    Returns mean loss and mean circular angle error (degrees).
    """
    model.eval()
    total_loss   = 0.0
    total_angle_err = 0.0
    n_batches    = 0
    n_samples    = 0

    with torch.no_grad():
        for crops, cos_gt, sin_gt, angle_gt in loader:
            crops    = crops.to(device)
            cos_gt   = cos_gt.to(device)
            sin_gt   = sin_gt.to(device)
            angle_gt = angle_gt.to(device)

            pred_cossin = model(crops)
            loss        = orientation_loss(pred_cossin, cos_gt, sin_gt)

            # Recover predicted angle
            pred_angle = torch.atan2(pred_cossin[:, 1],
                                     pred_cossin[:, 0])
            pred_angle = torch.rad2deg(pred_angle) % 360

            # Circular angle error per sample
            diff = torch.abs(pred_angle - angle_gt) % 360
            circular_err = torch.minimum(diff, 360 - diff)

            total_loss      += loss.item()
            total_angle_err += circular_err.sum().item()
            n_batches       += 1
            n_samples       += len(crops)

    mean_loss      = total_loss / n_batches
    mean_angle_err = total_angle_err / n_samples
    return mean_loss, mean_angle_err


# ─────────────────────────────────────────────────────────────────────────────
# Plot and save training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(train_losses, val_losses, val_angle_errs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label='Train Loss', color='steelblue')
    ax1.plot(epochs, val_losses,   label='Val Loss',   color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (cos-sin MSE)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_angle_errs, color='green')
    ax2.axhline(y=config.ANGLE_TOLERANCE_DEG, color='red',
                linestyle='--', label=f'{config.ANGLE_TOLERANCE_DEG}° tolerance')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Angle Error (degrees)')
    ax2.set_title('Validation Angle Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Saved training curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("Starting training")
    print("=" * 60)
    print(f"  Backbone  : {config.BACKBONE}")
    print(f"  Rowdy K   : {config.ROWDY_K}")
    print(f"  Rowdy η   : {config.ROWDY_N_INIT}")
    print(f"  Epochs    : {config.NUM_EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  LR        : {config.LEARNING_RATE}")
    print()

    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, _ = make_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TubeOrientationModel(
        backbone=config.BACKBONE,
        pretrained=config.PRETRAINED
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Two parameter groups:
    #   - backbone: smaller LR (it's pretrained, we don't want to destroy features)
    #   - head (Rowdy layers): full LR

    backbone_params = list(model.backbone.parameters())
    head_params     = list(model.head.parameters())

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': config.LEARNING_RATE * 0.1},
        {'params': head_params,     'lr': config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    # Reduce LR at epoch LR_ANNEAL_EPOCH and again at 3/4 of training.
    # The paper strongly recommends LR annealing for Rowdy networks (Section 4.2.1).
    milestones = [
        config.LR_ANNEAL_EPOCH,
        int(config.NUM_EPOCHS * 0.75)
    ]
    scheduler = MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=config.LR_ANNEAL_FACTOR
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss   = float('inf')
    best_angle_err  = float('inf')
    train_losses    = []
    val_losses      = []
    val_angle_errs  = []

    checkpoint_path = os.path.join(config.OUTPUT_DIR, 'best_model.pth')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t_start = time.time()

        train_loss              = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_angle_err = validate(model, val_loader, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_angle_errs.append(val_angle_err)

        elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[1]['lr']  # head LR

        print(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Angle Err: {val_angle_err:.1f}° | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model (judged by val angle error — that's what matters)
        if val_angle_err < best_angle_err:
            best_angle_err = val_angle_err
            best_val_loss  = val_loss
            torch.save({
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss':       val_loss,
                'val_angle_err':  val_angle_err,
                'config': {
                    'backbone':   config.BACKBONE,
                    'rowdy_K':    config.ROWDY_K,
                    'rowdy_n':    config.ROWDY_N_INIT,
                    'crop_size':  config.CROP_SIZE,
                }
            }, checkpoint_path)
            print(f"           ↑ New best! Saved checkpoint.")

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Training complete!")
    print(f"  Best val angle error : {best_angle_err:.2f}°")
    print(f"  Best val loss        : {best_val_loss:.4f}")
    print(f"  Checkpoint saved to  : {checkpoint_path}")
    print("=" * 60)

    plot_training_curves(train_losses, val_losses, val_angle_errs)

    return model, checkpoint_path


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
