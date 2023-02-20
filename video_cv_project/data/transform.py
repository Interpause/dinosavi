"""TODO: Add module docstring."""

from typing import Callable, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from video_cv_project.cfg import RGB, RGB_MEAN, RGB_STD

__all__ = ["create_train_pipeline", "MapTransform", "PatchSplitTransform"]

# TODO: Albumentation superiority:
# How to apply albumentation on video: https://albumentations.ai/docs/examples/example_multi_target/


class MapTransform:
    """Map transform over video or some other NCHW image tensor."""

    def __init__(self, transform: Callable):
        """Create MapTransform."""
        self.transform = transform

    def __call__(self, ims: torch.Tensor):
        """Apply transform to NCHW tensor.

        Args:
            ims(torch.Tensor): NCHW images.

        Returns:
            torch.Tensor: Transformed NCHW images.
        """
        return torch.stack([self.transform(im) for im in ims])

    def __repr__(self):
        """Return string representation of class."""
        repr = self.__class__.__name__ + "("
        for t in str(self.transform).splitlines():
            repr += f"\n    {t}"
        repr += "\n)"
        return repr


class PatchSplitTransform:
    """Split image into patches.

    Note that only tensors are supported.
    """

    def __init__(
        self, size: int | Tuple[int, int] = 64, stride: int | Tuple[int, int] = 32
    ):
        """Create PatchSplitTransform.

        Args:
            size(int | Tuple[int, int]): Patch size (H, W).
            stride(int | Tuple[int, int]): Patch stride (H, W).
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.stride = (stride, stride) if isinstance(stride, int) else stride

    def __call__(self, im: torch.Tensor):
        """Split CHW image into patches.

        Args:
            im(torch.Tensor): CHW image.

        Returns:
            torch.Tensor: NCHW image patches.
        """
        x = F.unfold(im, self.size, stride=self.stride)
        x = x.unflatten(0, (RGB, *self.size))
        x = x.permute(3, 0, 1, 2)  # NCHW
        return x

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(size={self.size}, stride={self.stride})"


def create_train_pipeline(
    im_size: int = 256,
    patch_size: int = 64,
    patch_stride: int = 32,
):
    """Create training pipeline."""
    scaler = T.InterpolationMode.LANCZOS
    # NOTE: Below augmentations are randomized per video frame. So some augmentations don't make sense.
    # TODO: Figure out how to fix augmentation per video. Might need albumentations.
    # TODO: Use Hydra config system for deeper configurability?
    # Augmentation transforms before splitting image to patches.
    prepatch_augments = [
        # T.ToPILImage(),
        T.Resize((im_size, im_size), interpolation=scaler),
        # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    ]
    # Augmentation transforms after splitting image to patches.
    postpatch_augments = [
        T.ToPILImage(),
        # Spatial jitter from paper. NOTE: Upstream forgot to suppress aspect ratio changes.
        T.RandomResizedCrop(
            patch_size, scale=(0.8, 0.95), ratio=(1.0, 1.0), interpolation=scaler
        ),
        T.ToTensor(),
        # Cannot convert to PIL after normalization.
        T.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ]

    return MapTransform(
        T.Compose(
            [
                *prepatch_augments,
                # T.ToTensor(),  # Needed for patch split.
                PatchSplitTransform(patch_size, patch_stride),
                MapTransform(T.Compose([*postpatch_augments])),
            ]
        )
    )
