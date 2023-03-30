"""Data transforms."""

from typing import Callable, List, Sequence, Tuple

import einops as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import default_collate
from transformers import AutoImageProcessor, ViTModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from video_cv_project.cfg import RGB_MEAN, RGB_STD

__all__ = [
    "create_train_pipeline",
    "create_pipeline",
    "MapTransform",
    "PatchSplitTransform",
    "PatchAndJitter",
    "PatchAndViT",
    "HFTransform",
]

# NOTE: Augmentations are random per frame so some don't make sense.


class MapTransform(nn.Module):
    """Map transforms over video or some other NCHW image tensor."""

    def __init__(self, *args: Callable):
        """Create MapTransform."""
        super(MapTransform, self).__init__()
        self.transforms = args
        # Register modules so they show up.
        self._nn_transforms = nn.ModuleList(t for t in args if isinstance(t, nn.Module))

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


class PatchAndJitter(nn.Module):
    """Split video into patches and apply spatial jitter."""

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
        super(PatchAndJitter, self).__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.scale = scale
        self.ratio = ratio
        self.jitter = T.RandomResizedCrop(
            size, scale=scale, ratio=ratio, antialias=True
        )

    def forward(self, im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split TCHW images into patches and apply spatial jitter.

        Args:
            im (torch.Tensor): TCHW images.

        Returns:
            torch.Tensor:
                pats: TNCHW image patches.
                tgts: N target patch class ids.
        """
        h, w = self.size
        pats = F.unfold(im, self.size, stride=self.stride)
        pats = E.rearrange(pats, "t (c h w) n -> (t n) c h w", h=h, w=w)
        pats = [self.jitter(p) for p in pats]
        pats: torch.Tensor = E.rearrange(pats, "(t n) c h w -> t n c h w", t=im.shape[0])  # type: ignore
        # Keys are ids of start patches, values are ids of end patches.
        tgts = torch.arange(pats.shape[1])
        return pats, tgts

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(size={self.size}, stride={self.stride}, scale={self.scale}, ratio={self.ratio})"


class PatchAndViT(nn.Module):
    """Uses HuggingFace `ViTModel` to patch and encode video."""

    def __init__(self, name: str, batch_size: int = 1):
        """Create PatchAndViT.

        ``name`` is passed to `ViTModel.from_pretrained`, meaning all its tricks
        like loading from a local file is possible.

        Args:
            name (str, optional): Name of model to load. Defaults to None.
            batch_size (int, optional): batch size of encoder. Defaults to 1.
        """
        super(PatchAndViT, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.encoder: ViTModel = ViTModel.from_pretrained(name).cpu().eval()

    def __call__(self, ims):
        """Split TCHW images into patches and encode using ViT.

        Args:
            ims (torch.Tensor): TCHW images.

        Returns:
            torch.Tensor: TCHW latent patches.
        """
        B = self.batch_size
        h, w = np.array(ims.shape[-2:]) // self.encoder.config.patch_size
        with torch.inference_mode():
            bats = [
                dict(
                    self.encoder(
                        ims[t : t + B],
                        output_attentions=True,
                        output_hidden_states=True,
                        interpolate_pos_encoding=True,
                        return_dict=True,
                    )
                )
                for t in range(0, len(ims), B)
            ]
            output = BaseModelOutputWithPooling(**default_collate(bats))
            pats = output.last_hidden_state
            # As `default_collate` adds a batch dimension on top of the existing one...
            pats = E.rearrange(pats, "t b n c -> (t b) n c")
            pats = E.rearrange(pats[:, 1:], "t (h w) c -> t c h w", h=h, w=w)
        return pats

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(name={self.name})"


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


class HFTransform:
    """Wrapper over HuggingFace `AutoImageProcessor`."""

    # NOTE: `AutoImageProcessor` has smart detection of whether image is [0,1] or [0,255].
    # So don't have to worry about re-scaling the channels.

    def __init__(self, name=None, **kwargs):
        """Create HFTransform.

        If ``name`` is None, then `AutoImageProcessor` will not be initialized based
        on any particular model. ``name`` is passed to `AutoImageProcessor.from_pretrained`,
        meaning all its tricks like loading from a local file is possible.

        Args:
            name (str, optional): Name of model to load configuration on. Defaults to None.
        """
        # For string repr.
        self.kwargs = dict(kwargs)
        self.kwargs["name"] = name

        # See: https://github.com/huggingface/transformers/issues/22392
        self._workaround = not self.kwargs.get("do_resize", True)

        self._p = (
            AutoImageProcessor(**kwargs)
            if name is None
            else AutoImageProcessor.from_pretrained(name, **kwargs)
        )

    def __call__(self, x):
        """Process image.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Processed tensor.
        """
        # Since only CHW images given when used in `create_pipeline`, there's extra batch dimension.
        if self._workaround:
            x = x * 255
        return self._p(x, return_tensors="pt").pixel_values[0]

    def __repr__(self):
        """Return string representation of class."""
        # print(self._p)
        args = ", ".join(f'{k}="{v}"' for k, v in self.kwargs.items())
        return f"{self.__class__.__name__}({args})"


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
