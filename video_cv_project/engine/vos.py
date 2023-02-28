"""Code for applying model for Video Object Segmentation."""

from pathlib import Path
from typing import List

import torch
import torchvision.transforms.functional as F
from PIL import Image

__all__ = ["dump_vos_preds"]


# TODO: Load image from metadata.
# Prevents dataloader needing to store both normalized & original images, wasting memory.


def dump_vos_preds(
    save_dir: str,
    ims: torch.Tensor,
    lbls: torch.Tensor,
    lbl_cls: torch.Tensor,
    palette: List[int] | None = None,
    blend_name: str = "x_%d_blend.jpg",
    mask_name: str = "x_%d_mask.png",
):
    """Dump predictions for VOS.

    Endgoal is for this codebase to evaluate J&F without needing an external tool.

    Args:
        save_dir (str): Directory to save predictions to.
        ims (torch.Tensor): TCHW video frames.
        lbls (torch.Tensor): TNHW bitmasks for each class N.
        lbl_cls (torch.Tensor): Label colors.
        palette (List[int], optional): Color palette for label image.
        blend_name (str, optional): Filename format for blended image. Defaults to "x_%d_blend.jpg".
        mask_name (str, optional): Filename format for label image. Defaults to "x_%d_mask.png".
    """
    sz = ims.shape[-2:]
    out_dir = Path(save_dir)

    for t, (im, lbl) in enumerate(zip(ims, lbls)):
        # Resize labels to original size.
        # NOTE: Can squeeze extra contour accuracy by using a different interpolation method.
        lbl = F.resize(lbl, sz)

        # Argmax to get predicted class for each pixel.
        lbl = torch.argmax(lbl, dim=0)

        # Get colors for each class.
        color = lbl_cls[lbl].permute(2, 0, 1) / 255.0

        # Save label.
        save_image(lbl if palette else color, out_dir / f"{mask_name % t}", palette)

        # Save blended image for visualization.
        overlay = im * 0.5 + color * 0.5
        save_image(overlay, out_dir / f"{blend_name % t}")


def save_image(im: torch.Tensor, path: Path, palette: List[int] | None = None):
    """Converts image to the DAVIS color palette & saves it.

    Ensures parent directory exists before saving image.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    if palette:
        _im = Image.fromarray(im.to(torch.uint8).numpy(), mode="P")
        _im.putpalette(palette, "RGB")
    else:
        _im = F.to_pil_image(im)
    _im.save(path)
