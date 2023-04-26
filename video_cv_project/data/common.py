"""Shared dataloading functions."""

import logging
from typing import List, Sequence

import einops as E
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.nn.functional import one_hot

from video_cv_project.cfg import RGB

__all__ = ["load_images", "images_to_tensor", "labels_to_tensor", "find_label_classes"]

log = logging.getLogger(__name__)


def load_images(im_paths: Sequence[str]) -> List[Image.Image]:
    """Load images from path."""
    ims = []
    for path in im_paths:
        try:
            Image.open(path).verify()
            # Have to load again since `verify` closes the image.
            ims.append(Image.open(path))
        except Exception as e:
            log.warning(f"Skipping {path} due to {e}.")
    return ims


def images_to_tensor(ims: Sequence[Image.Image], mode: str = None):
    """Convert images to tensor."""
    ims = [im.convert(mode=mode) if mode else im for im in ims]
    return [F.to_tensor(im) for im in ims]


def labels_to_tensor(ims: Sequence[Image.Image]):
    """Palette-aware conversion of labels to TNHW tensor."""
    pal = ims[0].getpalette() if ims[0].mode == "P" else None
    lbls = torch.stack([torch.from_numpy(np.array(im)) for im in ims])

    # `lbls` is THW.
    if pal:
        # cls: torch.Tensor = E.rearrange(lbls.unique(), "n -> n 1 1")
        # lbls = cls == E.repeat(lbls, "t h w -> t n h w", n=len(cls))
        lbls = E.rearrange(one_hot(lbls), "t h w n -> t n h w")
        colors = E.rearrange(torch.tensor(pal), "(n c) -> n c", c=RGB)

    # `lbls` is THWC.
    else:
        colors = find_label_classes(lbls[0])
        lbls = E.repeat(lbls, "t h w c -> t h w n c", n=len(colors))
        lbls = E.reduce(lbls == colors, "t h w n c -> t n h w", "prod")

    return list(lbls.float()), colors


def find_label_classes(lbl: torch.Tensor) -> torch.Tensor:
    """Find unique label classes (colors).

    Args:
        lbl (torch.Tensor): HWC label map. Should be ``uint8`` with [0, 255] range.

    Returns:
        torch.Tensor: NC classes, where N is number of classes & C is the color.
    """
    return E.rearrange(lbl, "h w c -> (h w) c").unique(dim=0)
