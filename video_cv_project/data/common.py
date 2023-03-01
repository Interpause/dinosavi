"""Shared dataloading functions."""

import logging
from typing import List, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

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
    # Either THWC or THW depending on image mode.
    lbls = torch.stack([torch.from_numpy(np.array(im)) for im in ims])

    # `lbls` is THW, where values are the class.
    if pal:
        cls = lbls.unique()  # N unique classes.
        # THW -> TNHW one-hot label embeddings/bitmasks.
        lbls = lbls.unsqueeze(1).expand(-1, len(cls), -1, -1).eq(cls[:, None, None])
        colors = torch.tensor(pal).reshape(-1, 3)

    # `lbls` is THWC.
    else:
        colors = find_label_classes(lbls.transpose(0, 3))  # NC unique class colors.
        # THWC -> NTHWC -> THWNC
        lbls = lbls.expand(len(colors), -1, -1, -1, -1).permute(1, 2, 3, 0, 4)
        # THWNC == NC -> THWN -> TNHW
        lbls = lbls.eq(colors).all(4).permute(0, 3, 1, 2)

    return list(lbls.float()), colors


def find_label_classes(lbl: torch.Tensor) -> torch.Tensor:
    """Find unique label classes (colors).

    Args:
        lbl (torch.Tensor): CHW label map. Should be ``uint8`` with [0, 255] range.

    Returns:
        torch.Tensor: NC classes, where N is number of classes & C is the color.
    """
    return lbl.flatten(1).unique(dim=1).T
