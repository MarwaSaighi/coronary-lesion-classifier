"""Frozen pretrained backbone + small MLP head.

The backbone's convolutional weights and BatchNorm affine parameters are frozen
and never seen by the optimizer. During training only the BatchNorm running
statistics adapt to the angiography domain; the MLP head is the only component
updated by gradient descent. Evaluation runs the whole model in `eval()` mode.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

from .config import IMAGENET_WEIGHTS

ARCHS = ("resnet18", "mobilenet_v2", "mobilenet_v3_small", "shufflenet_v2_x0_5", "squeezenet1_1")
_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureBackbone(nn.Module):
    """torchvision network without its classifier -> global-average-pooled vector."""

    def __init__(self, arch: str, net: nn.Module):
        super().__init__()
        self.arch, self.net = arch, net
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        m = self.net
        if self.arch in ("squeezenet1_1", "mobilenet_v2", "mobilenet_v3_small"):
            return m.features(x)
        if self.arch == "shufflenet_v2_x0_5":
            return m.conv5(m.stage4(m.stage3(m.stage2(m.maxpool(m.conv1(x))))))
        if self.arch.startswith("resnet"):
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return m.layer4(m.layer3(m.layer2(m.layer1(x))))
        raise ValueError(self.arch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.feature_map(x)).flatten(1)


def _strip_classifier(arch: str, net: nn.Module) -> None:
    if arch in ("squeezenet1_1", "mobilenet_v2"):
        net.classifier = nn.Identity()
    elif arch == "mobilenet_v3_small":
        net.classifier, net.avgpool = nn.Identity(), nn.Identity()
    else:
        net.fc = nn.Identity()


def load_radiologynet(net: nn.Module, path: str) -> None:
    """Load RadiologyNET ResNet weights (keys prefixed `model.`, head discarded)."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    state = {k.removeprefix("model."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, _ = net.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    if missing:
        raise RuntimeError(f"RadiologyNET backbone tensors missing: {missing[:5]}")


def build_backbone(arch: str, pretrain: str = "imagenet", weights_path: str | None = None) -> FeatureBackbone:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch}; choose from {ARCHS}")
    if pretrain == "imagenet":
        net = getattr(tvm, arch)(weights=IMAGENET_WEIGHTS[arch])
    elif pretrain == "radiologynet":
        if not arch.startswith("resnet"):
            raise ValueError("RadiologyNET weights exist only for ResNet backbones")
        net = getattr(tvm, arch)(weights=None)
        load_radiologynet(net, weights_path)
    elif pretrain == "none":
        net = getattr(tvm, arch)(weights=None)
    else:
        raise ValueError(pretrain)
    _strip_classifier(arch, net)
    return FeatureBackbone(arch, net)


class LesionClassifier(nn.Module):
    def __init__(self, arch: str, pretrain: str = "imagenet", weights_path: str | None = None,
                 hidden: int = 128, dropout: float = 0.3, seed: int = 0):
        super().__init__()
        self.arch, self.pretrain = arch, pretrain
        self.backbone = build_backbone(arch, pretrain, weights_path)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        with torch.no_grad():
            self.feature_dim = int(self.backbone(torch.zeros(1, 3, 224, 224)).shape[1])
        torch.manual_seed(seed)
        self.head = MLPHead(self.feature_dim, hidden, dropout)

    def train_mode(self) -> None:
        """Frozen backbone, but BatchNorm running statistics adapt to the data."""
        self.backbone.eval()
        for m in self.backbone.modules():
            if isinstance(m, _BN):
                m.train()
        self.head.train()

    def eval_mode(self) -> None:
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        return self.head(feats)

    def n_params(self) -> dict:
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.head.parameters())}


def p_lesion(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 0]
