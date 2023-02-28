"""Video Object Segmentation dataloader."""

import logging
from math import ceil
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset, default_collate

__all__ = ["VOSDataset", "vos_collate"]

log = logging.getLogger(__name__)


def load_images(im_paths: Sequence[str]):
    """Load images from path."""
    ims: List[Image.Image] = []
    for path in im_paths:
        try:
            Image.open(path).verify()
            ims.append(Image.open(path))
        except Exception as e:
            log.warning(f"Skipping {path} due to {e}.")
    return ims


def images_to_tensor(ims: Sequence[Image.Image], mode: str | None = "RGB"):
    """Convert images to tensor."""
    ims = [im.convert(mode=mode) if mode else im for im in ims]
    return [F.to_tensor(im) for im in ims]


def labels_to_tensor(lbls: Sequence[Image.Image]):
    """Palette-aware conversion of labels to TNHW tensor."""
    pal = lbls[0].getpalette() if lbls[0].mode == "P" else None
    # Either THWC or THW depending on image mode.
    _lbls = torch.stack([torch.from_numpy(np.array(lbl)) for lbl in lbls])

    # `_lbls` is THW, where values are the class.
    if pal:
        _cls: torch.Tensor = torch.unique(_lbls)  # N unique classes.
        # THW -> TNHW one-hot label embeddings/bitmasks.
        _lbls = _lbls[:, None, :, :].expand(-1, len(_cls), -1, -1)
        return (
            list(_lbls.eq(_cls[:, None, None])),
            torch.tensor(pal).reshape(-1, 3),
            pal,
        )

    # `_lbls` is THWC.
    else:
        _cls = find_label_classes(_lbls.transpose(0, 3))  # NC unique class colors.
        # THWC -> NTHWC -> THWNC
        _lbls = _lbls.expand(len(_cls), -1, -1, -1, -1).permute(1, 2, 3, 0, 4)
        # THWNC == NC -> THWN -> TNHW
        _lbls = torch.all(_lbls == _cls, dim=4).permute(0, 3, 1, 2)
        return list(_lbls), _cls, None


def find_label_classes(lbl: torch.Tensor) -> torch.Tensor:
    """Find unique label classes (colors).

    Args:
        lbl (torch.Tensor): CHW label map. Should be ``uint8`` with [0, 255] range.

    Returns:
        torch.Tensor: NC classes, where N is number of classes & C is the color.
    """
    return torch.unique(lbl.flatten(1), dim=1).T


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
        self.is_frames = is_frames
        self.palette: List[int] | None = None
        # TODO: Support loading videos directly?
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
                orig_ims: TCHW original video frames.
                tgts: TNHW one-hot label embeddings.
                lbl_cls: NC label classes where N is number of classes & C is the color.
                meta: Metadata. Useful to load original images for visualization.
        """
        """Get video frames & labels."""

        im_dir, im_paths = self.im_dirs[index]
        lbl_dir, lbl_paths = self.lbl_dirs[index]

        ims = load_images(im_paths)
        raw_lbls = load_images(lbl_paths)

        lbls, lbl_cls, _ = labels_to_tensor(raw_lbls)

        for i in range(len(ims)):
            im, lbl = ims[i], lbls[i]

            if self.im_size != -1:
                # Some interpolation methods only support PIL.
                im = F.resize(im, self.im_size)
                lbl = F.resize(
                    lbl, self.im_size, interpolation=T.InterpolationMode.NEAREST
                )  # Must use nearest to preserve labels.

            # NOTE: Original had option to convert to LAB colorspace here.

            ims[i], lbls[i] = F.to_tensor(im), lbl

        meta = dict(
            im_dir=im_dir,
            lbl_dir=lbl_dir,
            im_paths=self._repeat_context(im_paths),
            lbl_paths=self._repeat_context(lbl_paths),
        )

        lbls = torch.stack(lbls)  # TCHW
        lbl_sz = (
            ceil(lbls.shape[-2] / self.map_scale),
            ceil(lbls.shape[-1] / self.map_scale),
        )  # H, W
        # Should downscale after encoding in case regions overlap.
        lbls = F.resize(lbls.to(torch.float), lbl_sz)

        ims = self._repeat_context(ims)
        tgts = self._repeat_context(lbls)
        return self.transform(ims), torch.stack(ims), torch.stack(tgts), lbl_cls, meta

    def __len__(self):
        """Dataset length."""
        return len(self.im_dirs)


def vos_collate(batch):
    """Exclude metadata from being batched by dataloader."""
    metas = [b[-1] for b in batch]
    batch = [b[:-1] for b in batch]
    return (*default_collate(batch), metas)
