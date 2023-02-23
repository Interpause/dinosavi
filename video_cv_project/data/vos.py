"""Video Object Segmentation dataloader."""

import logging
from math import ceil
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from video_cv_project.cfg import RGB

__all__ = ["VOSDataset", "DAVISDataset"]

logger = logging.getLogger(__name__)

# TODO:
# - DavisDataset should process folder structure & metadata
# - VOSDataset gets the alr processed map of sorted video folders to sorted video frames


def load_images(im_paths: Sequence[str], mode: str = "RGB", to_tensor: bool = True):
    """Load images from path."""
    ims: List[torch.Tensor] = []
    for path in im_paths:
        im = Image.open(path)
        try:
            im = im.convert(mode=mode)
            ims.append(F.to_tensor(im) if to_tensor else im)
        except Exception as e:
            logger.warning(f"Skipping {path} due to {e}.")
    return ims


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
        map_scale: int = 8,
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
        self.map_scale = map_scale  # Downscale labels to encoder's latent map size.
        self.context_len = context_len
        self.is_frames = is_frames
        # TODO: Support loading videos directly?
        assert is_frames, "Only loading videos split as frames is supported for now."

    def repeat_context(self, items: Sequence):
        """Repeat items for context length."""
        return [items[0]] * self.context_len + list(items)

    # TODO: Either disk cache or LRU memory cache.
    def __getitem__(self, index: int):
        """Get video frames & labels.

        Args:
            index (int): Index of video.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                ims: TCHW video frames. Note they aren't normalized.
                tgts: TNHW one-hot label embeddings.
                lbl_cls: NC label classes where N is number of classes & C is the color.
                meta: Metadata. Useful to load original images for visualization.
        """
        """Get video frames & labels."""

        im_dir, im_paths = self.im_dirs[index]
        lbl_dir, lbl_paths = self.lbl_dirs[index]

        ims = load_images(im_paths, to_tensor=False)
        lbls = load_images(lbl_paths, to_tensor=True)

        for i in range(len(ims)):
            im, lbl = ims[i], lbls[i]

            if self.im_size != -1:
                im = F.resize(im, self.im_size)
                lbl = F.resize(
                    lbl, self.im_size, interpolation=T.InterpolationMode.NEAREST
                )  # Must use nearest to preserve label colors.

            # NOTE: Original had option to convert to LAB colorspace here.

            ims[i], lbls[i] = F.to_tensor(im), lbl

        meta = dict(
            im_dir=im_dir,
            lbl_dir=lbl_dir,
            im_paths=self.repeat_context(im_paths),
            lbl_paths=self.repeat_context(lbl_paths),
        )

        lbls = torch.stack(lbls)  # TCHW
        lbl_sz = (
            ceil(lbls.shape[-2] / self.map_scale),
            ceil(lbls.shape[-1] / self.map_scale),
        )  # H, W
        lbl_cls = find_label_classes((lbls[0] * 255).to(torch.uint8))  # NC
        N = lbl_cls.shape[0]  # Number of classes.

        # Target labels in the form of TNHW one-hot embeddings.
        tgts: List[torch.Tensor] = []
        for lbl in lbls:
            # This one-hot encoding method is both space & time efficient! Expanding
            # tensors doesn't copy data.
            # Should calculate one-hot before resizing since some regions might
            # overlap after downscale.

            # CHW -> NCHW -> HWNC
            lbl = (lbl * 255).to(torch.uint8).expand(N, -1, -1, -1).permute(2, 3, 0, 1)
            # HWNC == NC -> HWN -> NHW, where N is each class.
            tgt = torch.all(lbl == lbl_cls, dim=3).permute(2, 0, 1)
            # Resize to latent map size. Using smooth scaling is fine here.
            tgt = F.resize(tgt.to(torch.float32), lbl_sz)

            # TODO: Original has support for texture task here.

            tgts.append(tgt)

        ims = self.repeat_context(ims)
        tgts = self.repeat_context(tgts)

        return torch.stack(ims), torch.stack(tgts), lbl_cls, meta

    def __len__(self):
        """Dataset length."""
        return len(self.im_dirs)


class DAVISDataset(VOSDataset):
    """DAVIS Dataset."""

    def __init__(
        self,
        davis_dir: str,
        year: str = "2017",
        split: str = "val",
        quality: str = "480p",
        im_size: int | Tuple[int, int] = -1,
        map_scale: int = 8,
        context_len: int = 20,
    ):
        """Initialize DAVIS dataset."""
        im_dirs = []
        lbl_dirs = []

        root = Path(davis_dir).resolve()
        imageset_txt = root / "ImageSets" / year / f"{split}.txt"
        with imageset_txt.open() as f:
            videos = {l for s in f.readlines() if (l := s.strip()) != ""}

        for video in videos:
            lbl_dir = root / "Annotations" / quality / video
            im_dir = root / "JPEGImages" / quality / video
            im_paths = sorted(im_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            lbl_paths = sorted(lbl_dir.glob("*.png"), key=lambda p: int(p.stem))
            im_dirs.append((str(im_dir), [str(p) for p in im_paths]))
            lbl_dirs.append((str(lbl_dir), [str(p) for p in lbl_paths]))

        super().__init__(im_dirs, lbl_dirs, im_size, map_scale, context_len, True)
