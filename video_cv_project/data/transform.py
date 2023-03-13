"""Data transforms."""

from typing import Callable, List, Sequence, Tuple

import einops as E
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from video_cv_project.cfg import RGB_MEAN, RGB_STD

__all__ = [
    "create_train_pipeline",
    "create_pipeline",
    "MapTransform",
    "PatchSplitTransform",
    "PatchSplitJitterTransform",
]

# NOTE: Augmentations are random per frame so some don't make sense.


class MapTransform(nn.Module):
    """Map transforms over video or some other NCHW image tensor."""

    def __init__(self, *args: Callable):
        """Create MapTransform."""
        super(MapTransform, self).__init__()
        self.transforms = args

    def forward(self, ims: torch.Tensor):
        """Apply transforms to NCHW tensor.

        Args:
            ims (torch.Tensor): NCHW images.

        Returns:
            torch.Tensor: Transformed NCHW images.
        """
        t = T.Compose(self.transforms)
        return torch.stack([t(im) for im in ims])

    def __repr__(self):
        """Return string representation of class."""
        lines = "\n  ".join(l for t in self.transforms for l in repr(t).splitlines())
        return f"{self.__class__.__name__}(\n  {lines}\n)"


class PatchSplitTransform(nn.Module):
    """Split image into patches.

    Note that only tensors are supported.
    """

    def __init__(
        self, size: int | Tuple[int, int] = 64, stride: int | Tuple[int, int] = 32
    ):
        """Create PatchSplitTransform.

        Args:
            size (int | Tuple[int, int]): Patch size (H, W).
            stride (int | Tuple[int, int]): Patch stride (H, W).
        """
        super(PatchSplitTransform, self).__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    def forward(self, im: torch.Tensor):
        """Split CHW image into patches.

        Args:
            im (torch.Tensor): CHW image.

        Returns:
            torch.Tensor: NCHW image patches.
        """
        h, w = self.size
        x = F.unfold(im, self.size, stride=self.stride)
        return E.rearrange(x, "(c h w) n -> n c h w", h=h, w=w)

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(size={self.size}, stride={self.stride})"


class PatchSplitJitterTransform(nn.Module):
    """Split image into patches and apply spatial jitter."""

    def __init__(
        self,
        size: int | Tuple[int, int] = 64,
        stride: int | Tuple[int, int] = 32,
        scale: Tuple[float, float] = (0.7, 0.9),
        ratio: Tuple[float, float] = (0.8, 1.2),
    ):
        """Create PatchSplitWithJitterTransform.

        Args:
            size (int | Tuple[int, int]): Patch size (H, W).
            stride (int | Tuple[int, int]): Patch stride (H, W).
            scale (Tuple[float, float]): Bounds for crop size relative to image area.
            ratio (Tuple[float, float]): Bounds for random aspect ratio change.
        """
        super(PatchSplitJitterTransform, self).__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.scale = scale
        self.ratio = ratio
        self.splitter = PatchSplitTransform(size, stride)
        self.jitter = T.RandomResizedCrop(
            size, scale=scale, ratio=ratio, antialias=True
        )

    def forward(self, im: torch.Tensor):
        """Split CHW image into patches and apply spatial jitter.

        Args:
            im (torch.Tensor): CHW image.

        Returns:
            torch.Tensor: NCHW image patches.
        """
        pats = self.splitter(im)
        return torch.stack([self.jitter(p) for p in pats])

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(size={self.size}, stride={self.stride}, scale={self.scale}, ratio={self.ratio})"


class _ToTensor:
    """Unlike `T.ToTensor`, does not error if input is already a tensor."""

    def __init__(self):
        """Create _ToTensor."""
        self._t = T.ToTensor()

    def __call__(self, x):
        """Convert to tensor if necessary.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor.
        """
        return x if isinstance(x, torch.Tensor) else self._t(x)

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}()"


def create_pipeline(
    im_transforms: Sequence[Callable] = [],
    do_rgb_norm: bool = True,
    rgb_mean: Sequence[float] = RGB_MEAN,
    rgb_std: Sequence[float] = RGB_STD,
):
    """Create pipeline for training or inference.

    Intended for use with Hydra config system.
    """
    rgb_norm = T.Normalize(mean=rgb_mean, std=rgb_std)
    im: List[Callable] = [*im_transforms, _ToTensor()]
    im += [rgb_norm] if do_rgb_norm else []
    return MapTransform(*im)


def create_train_pipeline(im_size: Tuple[int, int] = (256, 256), **kwargs):
    """Create training pipeline."""
    im_transforms = [
        T.ToPILImage(),
        T.Resize(im_size, antialias=True),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
    ]

    return create_pipeline(im_transforms, **kwargs)
