"""Data transforms."""

from typing import Callable, List, Sequence, Tuple

import einops as E
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops.layers.torch import Rearrange

from video_cv_project.cfg import RGB_MEAN, RGB_STD

__all__ = [
    "create_train_pipeline",
    "create_pipeline",
    "MapTransform",
    "PatchSplitTransform",
]

# NOTE: Augmentations are random per frame so some don't make sense.
# TODO: Figure out how to fix augmentation per video. Might need albumentations.
# How to apply albumentation on video: https://albumentations.ai/docs/examples/example_multi_target/


class MapTransform:
    """Map transforms over video or some other NCHW image tensor."""

    def __init__(self, *args: Callable):
        """Create MapTransform."""
        self.transforms = args

    def __call__(self, ims: torch.Tensor):
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


class PatchSplitTransform:
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
        self.size = (size, size) if isinstance(size, int) else size
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    def __call__(self, im: torch.Tensor):
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
    patch_transforms: Sequence[Callable] = [],
    do_rgb_norm: bool = True,
    rgb_mean: Sequence[float] = RGB_MEAN,
    rgb_std: Sequence[float] = RGB_STD,
    do_patches: bool = False,
    patch_size: int | Tuple[int, int] = 64,
    patch_stride: int | Tuple[int, int] = 32,
):
    """Create pipeline for training or inference.

    Intended for use with Hydra config system.
    """
    rgb_norm = T.Normalize(mean=rgb_mean, std=rgb_std)
    im: List[Callable] = [*im_transforms, _ToTensor()]
    if do_patches:
        patch: List[Callable] = [*patch_transforms, _ToTensor()]
        patch += [rgb_norm] if do_rgb_norm else []
        im += [
            PatchSplitTransform(patch_size, patch_stride),
            MapTransform(*patch),
            Rearrange("n c h w -> (n c) h w"),
        ]
    else:
        im += [rgb_norm] if do_rgb_norm else []
    return MapTransform(*im)


def create_train_pipeline(
    im_size: Tuple[int, int] = (256, 256),
    patch_size: int | Tuple[int, int] = 64,
    patch_stride: int | Tuple[int, int] = 32,
):
    """Create training pipeline."""
    scaler = T.InterpolationMode.LANCZOS

    # Augmentation transforms before splitting image to patches.
    im_transforms = [
        T.ToPILImage(),
        T.Resize(im_size, interpolation=scaler),
        T.ToTensor(),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    ]

    # Augmentation transforms after splitting image to patches.
    patch_transforms = [
        T.ToPILImage(),
        # Spatial jitter from paper. NOTE: Upstream forgot to suppress aspect ratio changes.
        T.RandomResizedCrop(
            patch_size, scale=(0.8, 0.95), ratio=(0.9, 1.1), interpolation=scaler
        ),
        T.ToTensor(),
        # Cannot convert to PIL after normalization.
        T.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ]

    return MapTransform(
        *im_transforms,
        PatchSplitTransform(patch_size, patch_stride),
        MapTransform(*patch_transforms),
        Rearrange("n c h w -> (n c) h w"),
    )
