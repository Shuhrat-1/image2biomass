"""Model architectures for image-based biomass regression.

Two models matching the course arc:
- SimpleCNN: a small convnet trained from scratch (session 10) — the
  computer-vision baseline.
- PretrainedRegressor: transfer learning (session 12) with a ResNet18 or
  EfficientNet-B0 backbone and a 5-output regression head.

All models output 5 values in log1p space (consistent with the dataset and the
tabular baseline). Inversion (expm1) and clipping happen at metric time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

N_TARGETS = 5


# --------------------------------------------------------------------------- #
# From-scratch CNN (session 10)
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    """Small convnet for 224x224 RGB -> 5 regression outputs.

    Four conv blocks (conv-bn-relu-pool) then a global pool and an MLP head.
    Kept intentionally small: with only ~285 training images, a large network
    would overfit immediately.
    """

    def __init__(self, n_targets: int = N_TARGETS, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),     # 224 -> 112
            self._block(32, 64),    # 112 -> 56
            self._block(64, 128),   # 56 -> 28
            self._block(128, 256),  # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_targets),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Conv-BN-ReLU-MaxPool block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Transfer learning (session 12)
# --------------------------------------------------------------------------- #
class PretrainedRegressor(nn.Module):
    """Pretrained backbone + 5-output regression head.

    Parameters
    ----------
    backbone : str
        'resnet18' or 'efficientnet_b0'.
    pretrained : bool
        Load ImageNet weights.
    freeze_backbone : bool
        If True, freeze backbone (feature extraction); if False, fine-tune all.
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 freeze_backbone: bool = False, n_targets: int = N_TARGETS,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.net, in_features = self._build_backbone(backbone, pretrained)
        if freeze_backbone:
            self._freeze()
        self._attach_head(backbone, in_features, n_targets, dropout)

    @staticmethod
    def _build_backbone(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Instantiate backbone and return (model, head_in_features)."""
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        return net, in_features

    def _freeze(self) -> None:
        """Freeze all backbone parameters (head added later stays trainable)."""
        for p in self.net.parameters():
            p.requires_grad = False

    def _attach_head(self, backbone: str, in_features: int,
                     n_targets: int, dropout: float) -> None:
        """Replace the classifier with a regression head."""
        head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_targets),
        )
        if backbone == "resnet18":
            self.net.fc = head
        else:  # efficientnet_b0
            self.net.classifier = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(kind: str = "resnet18", **kwargs) -> nn.Module:
    """Build a model by name.

    kind: 'simple_cnn' | 'resnet18' | 'efficientnet_b0'.
    For pretrained backbones, kwargs pass through (pretrained, freeze_backbone).
    """
    if kind == "simple_cnn":
        return SimpleCNN(**{k: v for k, v in kwargs.items()
                            if k in {"n_targets", "dropout"}})
    if kind in {"resnet18", "efficientnet_b0"}:
        return PretrainedRegressor(backbone=kind, **kwargs)
    raise ValueError(f"unknown kind: {kind}")


def count_trainable(model: nn.Module) -> int:
    """Number of trainable parameters (useful to compare freeze vs fine-tune)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)