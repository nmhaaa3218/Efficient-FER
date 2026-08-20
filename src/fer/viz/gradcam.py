from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def gradcam_heatmap(model: nn.Module, image: torch.Tensor, target_layer: nn.Module, class_idx: int | None = None) -> np.ndarray:
    """Grad-CAM heatmap for a single 1x48x48 image. Returns [H, W] float array in [0,1]."""
    model.eval()
    activations: dict = {}
    gradients: dict = {}

    def hook_fwd(module, inp, out):
        activations["value"] = out

    def hook_bwd(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    hf = target_layer.register_forward_hook(hook_fwd)
    hb = target_layer.register_full_backward_hook(hook_bwd)

    img = image.unsqueeze(0).clone().detach().requires_grad_(True)
    with torch.enable_grad():
        out = model(img)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1.0
        model.zero_grad()
        out.backward(gradient=one_hot)

    hf.remove()
    hb.remove()

    weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations["value"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
    return cam
