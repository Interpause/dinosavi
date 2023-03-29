"""Visualization utilities."""

from pathlib import Path
from typing import List

import einops as E
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.nn.functional import softmax

__all__ = ["save_image", "label_to_image"]


def save_image(image: torch.Tensor, path: Path, palette: List[int] = None):
    """Save image with optional palette.

    Ensures parent directory exists before saving image.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    if palette:
        im = Image.fromarray(image.byte().numpy(), mode="P")
        im.putpalette(palette, "RGB")
    else:
        im = F.to_pil_image(image)
    im.save(path)


def label_to_image(
    label: torch.Tensor,
    colors: torch.Tensor,
    mode: str = "max",
    blend_temp: float = 0.5,
):
    """Convert label to color image.

    Args:
        label (torch.Tensor): NHW label.
        colors (torch.Tensor): NC colors.
        mode (str, optional): How to blend multiple classes together. Defaults to "max".
        blend_temp (float, optional): Softmax temperature when mode is ``blend``.

    Returns:
        torch.Tensor: CHW label as color image.
    """
    if mode == "max":
        lbl = label.argmax(dim=0)
        return E.rearrange(colors[lbl], "h w c -> c h w") / 255
    elif mode == "blend":
        lbl = softmax(label / blend_temp, dim=0)
        colors = colors[: len(lbl)].type_as(lbl)
        return E.einsum(lbl, colors, "n h w, n c -> c h w") / 255
