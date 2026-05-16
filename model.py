# model.py
# ─────────────────────────────────────────────────────────────────────────────
# Architecture:
#   1. RowdyLayer    — single FC layer with Rowdy adaptive activation
#                      (Jagtap et al. 2022, Eq. 11)
#   2. RowdyHead     — stack of RowdyLayers for angle regression
#   3. TubeOrientationModel — pretrained ResNet18 backbone + RowdyHead
#                             outputs (cos θ, sin θ)
# ─────────────────────────────────────────────────────────────────────────────

import math
import torch
import torch.nn as nn
import torchvision.models as models

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rowdy Layer
# ─────────────────────────────────────────────────────────────────────────────

class RowdyLayer(nn.Module):
    """
    One fully-connected layer followed by a Rowdy adaptive activation.

    The activation is (Jagtap et al. 2022, Eq. 1 + Eq. 11):

        φ̃(z) = α_1 · φ_1(ω_1 · z)                   ← base term
              + α_2 · sin(1 · η · ω_2 · z)            ← harmonic 1
              + α_3 · sin(2 · η · ω_3 · z)            ← harmonic 2
              + ...
              + α_K · sin((K-1) · η · ω_K · z)        ← harmonic K-1

    Where:
        φ_1  = base activation (ReLU or tanh)
        η    = n_init, a fixed scaling factor
        α_k  = trainable amplitude weights
        ω_k  = trainable frequency weights
        K    = number of terms

    Initialisation follows Eq. (8) of the paper:
        α_1 = 1,  α_k = 0 for k ≥ 2   → starts as pure base activation
        ω_k = 1   for all k             → neutral scaling initially
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 K: int   = config.ROWDY_K,
                 n: float = config.ROWDY_N_INIT,
                 base: str = config.ROWDY_BASE,
                 dropout: float = 0.1):
        super().__init__()

        self.K = K
        self.n = n  # fixed scaling factor η

        # Standard linear transform
        self.linear  = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

        # Base activation φ_1
        if base == 'relu':
            self.phi1 = nn.ReLU()
        elif base == 'tanh':
            self.phi1 = nn.Tanh()
        else:
            raise ValueError(f"Unknown base activation: {base}. Use 'relu' or 'tanh'.")

        # Trainable amplitude parameters α = [α_1, ..., α_K]
        # Init: α_1=1, α_k=0 for k≥2 (Eq. 8 — starts identical to base AF)
        alpha_init = torch.zeros(K)
        alpha_init[0] = 1.0
        self.alpha = nn.Parameter(alpha_init)

        # Trainable frequency parameters ω = [ω_1, ..., ω_K]
        # Init: all 1.0
        self.omega = nn.Parameter(torch.ones(K))

        # Batch norm after activation helps stabilise Rowdy's oscillatory output
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch_size, in_features)
        returns : (batch_size, out_features)
        """
        z = self.linear(x)   # (B, out_features)

        # ── Base term: α_1 · φ_1(ω_1 · z) ───────────────────────────────────
        out = self.alpha[0] * self.phi1(self.omega[0] * z)

        # ── Sinusoidal harmonic terms ─────────────────────────────────────────
        # φ_k(x) = sin((k-1) · η · ω_k · x)   for k = 2, ..., K
        for k in range(1, self.K):
            harmonic_freq = k * self.n * self.omega[k]
            out = out + self.alpha[k] * torch.sin(harmonic_freq * z)

        out = self.dropout(out)
        out = self.bn(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rowdy Head (stack of RowdyLayers)
# ─────────────────────────────────────────────────────────────────────────────

class RowdyHead(nn.Module):
    """
    Two Rowdy layers followed by a linear output layer.
    Outputs 2 values: (cos θ, sin θ).

    Architecture:
        in_features → 256 [Rowdy] → 128 [Rowdy] → 2 [Linear]
    """

    def __init__(self, in_features: int):
        super().__init__()

        self.layers = nn.Sequential(
            RowdyLayer(in_features, 256, dropout=0.2),
            RowdyLayer(256,         128, dropout=0.1),
        )

        # Final output: 2 values (cos θ, sin θ)
        # No activation — we want unbounded output, then normalise
        self.output_layer = nn.Linear(128, 2)

        # Initialise output layer small — helps with early training stability
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, in_features)  — feature vector from backbone
        returns : (B, 2)            — [cos θ, sin θ], NOT normalised to unit circle
        """
        x = self.layers(x)
        x = self.output_layer(x)
        return x  # raw (cos, sin) — loss function handles the rest


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full Model
# ─────────────────────────────────────────────────────────────────────────────

class TubeOrientationModel(nn.Module):
    """
    Full pipeline:
        RGB crop (3 × 64 × 64)
            ↓
        ResNet18 backbone (pretrained, strips final FC)
            ↓
        512-dim feature vector
            ↓
        RowdyHead
            ↓
        (cos θ, sin θ)   — recover angle with atan2

    To get the actual angle:
        angle_deg = torch.atan2(out[:, 1], out[:, 0]) * 180 / pi % 360
    """

    def __init__(self,
                 backbone: str = config.BACKBONE,
                 pretrained: bool = config.PRETRAINED):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base    = models.resnet18(weights=weights)
            feature_dim = 512

            # Remove the final classification layer
            # Keep everything up to and including the adaptive avg pool
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        elif backbone == "lenet":
            # Small custom LeNet-style backbone for when you don't want pretrained
            # Less powerful but faster to train
            feature_dim   = 256
            self.backbone = _make_lenet_backbone()

        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'resnet18' or 'lenet'.")

        self.feature_dim = feature_dim

        # ── Rowdy KNN Head ────────────────────────────────────────────────────
        self.head = RowdyHead(in_features=feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, 3, CROP_SIZE, CROP_SIZE)
        returns : (B, 2)  — raw (cos θ, sin θ), not unit-normalised
        """
        features = self.backbone(x)          # (B, feature_dim, 1, 1) for ResNet
        features = features.flatten(start_dim=1)  # (B, feature_dim)
        cossin   = self.head(features)        # (B, 2)
        return cossin

    def predict_angle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: returns predicted angle in degrees [0, 360).
        """
        cossin = self.forward(x)
        angle  = torch.atan2(cossin[:, 1], cossin[:, 0])  # radians, [-π, π]
        angle  = torch.rad2deg(angle) % 360
        return angle


def _make_lenet_backbone():
    """
    Simple LeNet-style CNN backbone (no pretrained weights).
    Use this if you want a fully custom architecture.
    Output: (B, 256, 1, 1) before flattening.
    """
    return nn.Sequential(
        # Block 1
        nn.Conv2d(3, 32, kernel_size=5, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),      # 64 → 32

        # Block 2
        nn.Conv2d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),      # 32 → 16

        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),      # 16 → 8

        # Block 4
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), # 8 → 1×1
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────────────────────────────────────

def orientation_loss(pred_cossin: torch.Tensor,
                     gt_cos: torch.Tensor,
                     gt_sin: torch.Tensor) -> torch.Tensor:
    """
    Loss in (cos, sin) space.

    Why not just MSE on the angle?
    Because angles wrap around: MSE(359°, 1°) = huge, but they're only 2° apart.
    In (cos, sin) space this is handled naturally.

    Loss = mean( (cos_pred - cos_gt)² + (sin_pred - sin_gt)² )

    This is equivalent to:
        2 · mean(1 - cos(θ_pred - θ_gt))
    which is the Von Mises loss — the natural loss for circular regression.
    """
    cos_pred = pred_cossin[:, 0]
    sin_pred = pred_cossin[:, 1]

    loss = (cos_pred - gt_cos) ** 2 + (sin_pred - gt_sin) ** 2
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Running model sanity check...")
    print("=" * 60)

    model = TubeOrientationModel(backbone=config.BACKBONE,
                                  pretrained=False)  # False for quick test

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Forward pass test
    dummy_input = torch.randn(4, 3, config.CROP_SIZE, config.CROP_SIZE)
    output      = model(dummy_input)

    print(f"\nInput shape  : {dummy_input.shape}")
    print(f"Output shape : {output.shape}")   # should be [4, 2]

    # Loss test
    gt_cos = torch.tensor([1.0, 0.0, -1.0,  0.0])
    gt_sin = torch.tensor([0.0, 1.0,  0.0, -1.0])
    loss   = orientation_loss(output, gt_cos, gt_sin)
    print(f"\nLoss (random init): {loss.item():.4f}")

    # Angle prediction test
    angles = model.predict_angle(dummy_input)
    print(f"Predicted angles  : {angles.tolist()}")
    print("\n✓ Model looks good!")
