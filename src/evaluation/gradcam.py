"""Grad-CAM (Gradient-Weighted Class Activation Mapping) for Audio Spectrograms.

Project P.E.T.E.R. (Perceptual Evaluation & Transparent Emotion Research)
Affective Computing & Biosignals Pipeline

This module computes 2D spatial-temporal saliency heatmaps for EmotionCNN2D:
- Targets the final convolutional layer of the feature extraction backbone.
- Computes gradients of target/winning emotion logits with respect to feature maps.
- Pools gradients globally to weight feature channels.
- Applies ReLU rectification to isolate positive contributions to the decision.
- Symmetrically interpolates heatmaps to input spectrogram dimensions [80, T].
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("peter.evaluation.gradcam")


class GradCAM:
    """Gradient-weighted Class Activation Mapping for audio spectrogram CNNs."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        """Initializes GradCAM by registering forward and backward hooks on target layer.

        Args:
            model: PyTorch model instance (e.g., EmotionCNN2D).
            target_layer: Specific Conv2d layer to hook into (defaults to conv_blocks[12]).
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to the 4th convolutional layer (conv_blocks[12] in EmotionCNN2D)
            if hasattr(model, "conv_blocks") and len(model.conv_blocks) > 12:
                self.target_layer = model.conv_blocks[12]
            else:
                # Search for the last Conv2d layer in the model
                conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                if not conv_layers:
                    raise ValueError("No Conv2d layer found in the provided model.")
                self.target_layer = conv_layers[-1]
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Hook registration
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Captures feature activation maps from target layer during forward pass."""
        self.activations = output

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Captures gradients flowing back into target layer during backward pass."""
        self.gradients = grad_output[0]

    def generate_heatmap(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """Generates a normalized [80, T] Grad-CAM heatmap for an input spectrogram.

        Args:
            x: Input audio spectrogram tensor of shape [1, 1, 80, T].
            target_class: Target emotion class index (0-7). If None, winning class is used.

        Returns:
            Tuple of:
              - Normalized saliency heatmap tensor of shape [80, T] with values in [0, 1].
              - Class index evaluated (int).
              - Softmax confidence probability of the evaluated class (float).
        """
        if x.ndim == 3:
            # Expand [1, 80, T] -> [1, 1, 80, T]
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor input, got {x.shape}")

        device = next(self.model.parameters()).device
        x = x.to(device)

        with torch.enable_grad():
            self.model.zero_grad()

            # Forward pass
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)

            if target_class is None:
                target_class = int(torch.argmax(logits, dim=-1).item())

            class_prob = float(probs[0, target_class].item())
            score = logits[0, target_class]

            # Backward pass to obtain gradients with respect to activations
            score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients from target layer.")

        # Global average pooling across spatial/temporal dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of feature activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']

        # ReLU rectification: isolate features positively correlated with the class
        cam = F.relu(cam)

        # Min-max normalization
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Interpolate back to original spectrogram dimensions [80, T]
        target_h, target_w = x.shape[2], x.shape[3]
        heatmap = F.interpolate(
            cam,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        heatmap = heatmap.squeeze(0).squeeze(0)  # [80, T]
        return heatmap.detach().cpu(), target_class, class_prob

    def remove_hooks(self) -> None:
        """Removes registered PyTorch hooks to avoid memory leakage."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __del__(self) -> None:
        """Cleanup handles upon garbage collection."""
        try:
            self.remove_hooks()
        except Exception:
            pass
