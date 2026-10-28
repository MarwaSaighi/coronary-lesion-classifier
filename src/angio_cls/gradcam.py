"""Grad-CAM for the frozen-backbone classifier (used by the demo)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models import LesionClassifier


def gradcam(model: LesionClassifier, x: torch.Tensor, class_idx: int = 0) -> tuple[np.ndarray, float]:
    """Return (heatmap in [0,1] at input resolution, P(lesion)) for one image [1,3,H,W]."""
    model.eval_mode()
    fmap = model.backbone.feature_map(x).detach().requires_grad_(True)
    logits = model.head(model.backbone.gap(fmap).flatten(1))
    logits[0, class_idx].backward()
    weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.detach().numpy(), float(torch.softmax(logits.detach(), 1)[0, 0])
