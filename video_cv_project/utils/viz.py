"""Visualization utilities."""

from functools import partial
from pathlib import Path
from typing import List

import einops as E
import torch
import torchvision.transforms.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image

__all__ = ["save_image", "label_to_image", "tb_viz_slots", "tb_hparams"]


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


def label_to_image(label: torch.Tensor, colors: torch.Tensor, mode: str = "max"):
    """Convert label to color image.

    Args:
        label (torch.Tensor): NHW label.
        colors (torch.Tensor): NC colors.
        mode (str, optional): How to blend multiple classes together. Defaults to "max".

    Returns:
        torch.Tensor: CHW label as color image.
    """
    if mode == "max":
        lbl = label.argmax(dim=0)
        return E.rearrange(colors[lbl], "h w c -> c h w") / 255
    elif mode == "blend":
        # lbl = softmax(label / blend_temp, dim=0)
        lbl = (label - label.min()) / (label.max() - label.min())
        colors = colors[: len(lbl)].type_as(lbl)
        return E.einsum(lbl, colors, "n h w, n c -> c h w") / 255


def tb_viz_slots(pats: torch.Tensor, attns: torch.Tensor):
    """Prepare visualization of Slot Attention for tensorboard.

    Args:
        pats (torch.Tensor): CHW patches.
        attns: (torch.Tensor): SN attention weights.

    Returns:
        tuple: Visualization info.
    """
    kwargs = dict(
        mat=E.rearrange(pats.detach().cpu(), "c h w -> (h w) c"),
        metadata=attns.detach().cpu().argmax(dim=0).tolist(),
    )
    return "add_embedding", kwargs


def tb_hparams(cfg: DictConfig):
    """Extract hyperparameters to log from the experiment config."""
    C = partial(OmegaConf.select, cfg, default=None)
    # This is quite hardcoded, but no choice.
    return dict(
        bs=C("data.batch_size"),
        nframes=C("data.dataset.frames_per_clip"),
        fps=C("data.dataset.frame_rate"),
        optim=C("train.optimizer._target_"),
        lr=C("train.optimizer.lr"),
        sched=C("train.scheduler._target_"),
    )
