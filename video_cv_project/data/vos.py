"""Video Object Segmentation dataloader."""

import logging
from math import ceil
from typing import Callable, Sequence, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .common import images_to_tensor, labels_to_tensor, load_images

__all__ = ["VOSDataset"]

log = logging.getLogger(__name__)


class VOSDataset(Dataset):
    """Video Object Segmentation dataset."""

    def __init__(
        self,
        im_dirs: Sequence[Tuple[str, Sequence[str]]] = [],
        lbl_dirs: Sequence[Tuple[str, Sequence[str]]] = [],
        im_size: int | Tuple[int, int] = -1,
        transform: Callable = lambda x: x,
        map_scale: int = 1,
        context_len: int = 20,
        is_frames: bool = True,
    ):
        """Base class for VOS datasets."""
        self.im_dirs = im_dirs
        self.lbl_dirs = lbl_dirs if len(lbl_dirs) > 0 else None
        self.im_size = (
            im_size
            if im_size == -1 or isinstance(im_size, tuple)
            else (im_size, im_size)
        )
        self.transform = transform
        self.map_scale = map_scale  # Downscale labels to encoder's latent map size.
        self.context_len = context_len
        # Must check first since workers are separate processes.
        self.has_palette = (
            Image.open(lbl_dirs[0][1][0]).mode == "P" if self.lbl_dirs else True
        )
        # TODO: Support loading videos directly?
        self.is_frames = is_frames
        assert is_frames, "Only loading videos split as frames is supported for now."

    def _repeat_context(self, items: Sequence):
        """Repeat items for context length."""
        return [items[0]] * self.context_len + list(items)

    # TODO: Either disk cache or LRU memory cache.
    def __getitem__(self, index: int):
        """Get video frames & labels.

        Args:
            index (int): Index of video.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                ims: TCHW video frames.
                lbls: TNHW one-hot label embeddings.
                colors: NC label classes where N is number of classes & C is the color.
                meta: Metadata. Useful to load original images for visualization.
        """
        im_dir, im_paths = self.im_dirs[index]
        meta = dict(im_dir=im_dir, im_paths=im_paths)

        ims = images_to_tensor(load_images(im_paths))
        if self.im_size != -1:
            ims = [F.resize(im, self.im_size, antialias=True) for im in ims]
        ims = self._repeat_context(ims)

        if self.lbl_dirs is not None:
            lbl_dir, lbl_paths = self.lbl_dirs[index]
            meta.update(lbl_dir=lbl_dir, lbl_paths=lbl_paths)

            lbls, colors = labels_to_tensor(load_images(lbl_paths))
            if self.im_size != -1:
                lbls = [F.resize(lbl, self.im_size, antialias=True) for lbl in lbls]

            lbls = torch.stack(lbls)  # TCHW
            h, w = lbls.shape[-2:]
            lbl_sz = (ceil(h / self.map_scale), ceil(w / self.map_scale))
            lbls = F.resize(lbls, lbl_sz, antialias=True)

            lbls = self._repeat_context(lbls)
            return self.transform(ims), torch.stack(lbls), colors, meta

        return self.transform(ims), None, torch.tensor(COLOR_PALETTE), meta

    def __len__(self):
        """Dataset length."""
        return len(self.im_dirs)


# COCO color palette seems okay. Needed for unsupervised task unless you like looking at blank masks.
COLOR_PALETTE = [
    (220, 20, 60),
    (119, 11, 32),
    (0, 0, 142),
    (0, 0, 230),
    (106, 0, 228),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 70),
    (0, 0, 192),
    (250, 170, 30),
    (100, 170, 30),
    (220, 220, 0),
    (175, 116, 175),
    (250, 0, 30),
    (165, 42, 42),
    (255, 77, 255),
    (0, 226, 252),
    (182, 182, 255),
    (0, 82, 0),
    (120, 166, 157),
    (110, 76, 0),
    (174, 57, 255),
    (199, 100, 0),
    (72, 0, 118),
    (255, 179, 240),
    (0, 125, 92),
    (209, 0, 151),
    (188, 208, 182),
    (0, 220, 176),
    (255, 99, 164),
    (92, 0, 73),
    (133, 129, 255),
    (78, 180, 255),
    (0, 228, 0),
    (174, 255, 243),
    (45, 89, 255),
    (134, 134, 103),
    (145, 148, 174),
    (255, 208, 186),
    (197, 226, 255),
    (171, 134, 1),
    (109, 63, 54),
    (207, 138, 255),
    (151, 0, 95),
    (9, 80, 61),
    (84, 105, 51),
    (74, 65, 105),
    (166, 196, 102),
    (208, 195, 210),
    (255, 109, 65),
    (0, 143, 149),
    (179, 0, 194),
    (209, 99, 106),
    (5, 121, 0),
    (227, 255, 205),
    (147, 186, 208),
    (153, 69, 1),
    (3, 95, 161),
    (163, 255, 0),
    (119, 0, 170),
    (0, 182, 199),
    (0, 165, 120),
    (183, 130, 88),
    (95, 32, 0),
    (130, 114, 135),
    (110, 129, 133),
    (166, 74, 118),
    (219, 142, 185),
    (79, 210, 114),
    (178, 90, 62),
    (65, 70, 15),
    (127, 167, 115),
    (59, 105, 106),
    (142, 108, 45),
    (196, 172, 0),
    (95, 54, 80),
    (128, 76, 255),
    (201, 57, 1),
    (246, 0, 122),
    (191, 162, 208),
]
