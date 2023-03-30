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
        self.lbl_dirs = lbl_dirs
        self.im_size = (
            im_size
            if im_size == -1 or isinstance(im_size, tuple)
            else (im_size, im_size)
        )
        self.transform = transform
        self.map_scale = map_scale  # Downscale labels to encoder's latent map size.
        self.context_len = context_len
        # Must check first since workers are separate processes.
        self.has_palette = Image.open(lbl_dirs[0][1][0]).mode == "P"
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
        """Get video frames & labels."""

        im_dir, im_paths = self.im_dirs[index]
        lbl_dir, lbl_paths = self.lbl_dirs[index]

        meta = dict(
            im_dir=im_dir, lbl_dir=lbl_dir, im_paths=im_paths, lbl_paths=lbl_paths
        )

        ims = images_to_tensor(load_images(im_paths))
        lbls, colors = labels_to_tensor(load_images(lbl_paths))

        if self.im_size != -1:
            for i, (im, lbl) in enumerate(zip(ims, lbls)):
                # Some interpolation methods only support PIL.
                ims[i] = F.resize(im, self.im_size, antialias=True)
                lbls[i] = F.resize(lbl, self.im_size, antialias=True)

        lbls = torch.stack(lbls)  # TCHW
        h, w = lbls.shape[-2:]
        lbl_sz = (ceil(h / self.map_scale), ceil(w / self.map_scale))
        lbls = F.resize(lbls, lbl_sz, antialias=True)

        ims = self._repeat_context(ims)
        lbls = self._repeat_context(lbls)
        return self.transform(ims), torch.stack(lbls), colors, meta

    def __len__(self):
        """Dataset length."""
        return len(self.im_dirs)
