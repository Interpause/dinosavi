"""Code for applying model for Video Object Segmentation."""

from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms.functional as F

from video_cv_project.data.common import images_to_tensor, load_images

from .common import save_image

__all__ = ["dump_vos_preds"]


def dump_vos_preds(
    save_dir: str,
    im_paths: Sequence[str],
    lbls: torch.Tensor,
    colors: torch.Tensor,
    has_palette: bool = False,
    blend_name: str = "x_%d_blend.jpg",
    mask_name: str = "x_%d_mask.png",
):
    """Dump predictions for VOS.

    Endgoal is for this codebase to evaluate J&F without needing an external tool.

    Args:
        save_dir (str): Directory to save predictions to.
        im_paths (Sequence[str]): Paths of images to blend predictions with.
        lbls (torch.Tensor): TNHW bitmasks for each class N.
        colors (torch.Tensor): Label colors.
        has_palette (bool, optional): Where to use a palette for label images. Defaults to False.
        blend_name (str, optional): Filename format for blended image. Defaults to "x_%d_blend.jpg".
        mask_name (str, optional): Filename format for label image. Defaults to "x_%d_mask.png".
    """
    out_dir = Path(save_dir)
    ims = images_to_tensor(load_images(im_paths), mode="RGB")
    sz = ims[0].shape[-2:]
    pal = colors.flatten().tolist()

    for t, (im, lbl) in enumerate(zip(ims, lbls)):
        # Resize labels to original size.
        # NOTE: Can squeeze extra contour accuracy by using a different interpolation method.
        lbl = F.resize(lbl, sz)

        # Argmax to get predicted class for each pixel.
        lbl = lbl.argmax(dim=0)

        # Get colors for each class.
        color_lbl = colors[lbl].permute(2, 0, 1) / 255.0

        # Save label.
        save_image(lbl if has_palette else color_lbl, out_dir / f"{mask_name % t}", pal)

        # Save blended image for visualization.
        overlay = im * 0.5 + color_lbl * 0.5
        save_image(overlay, out_dir / f"{blend_name % t}")
