"""Data transforms."""

from contextlib import ExitStack
from io import BytesIO
from multiprocessing import current_process, parent_process
from typing import Callable, Dict, List, Sequence, Tuple

import einops as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diskcache import Cache
from torch.utils.data import default_collate
from transformers import AutoImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from video_cv_project.cfg import CACHE_LAST_ATTNS, CACHE_PATCHES, RGB_MEAN, RGB_STD
from video_cv_project.models.encoders import ViTLastAttnModel
from video_cv_project.utils import hash_model, hash_tensor

__all__ = [
    "create_train_pipeline",
    "create_pipeline",
    "MapTransform",
    "PatchSplitTransform",
    "PatchAndJitter",
    "PatchAndViT",
    "TensorCache",
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

    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        compile: bool = False,
        cache_dir: str = None,
        device: torch.device = None,
    ):
        """Create PatchAndViT.

        ``name`` is passed to `ViTModel.from_pretrained`, meaning all its tricks
        like loading from a local file is possible.

        Note, if using ``compile``, the input size to the encoder should be the
        original input size of the `ViTModel` as specified by `ViTConfig` (i.e.,
        224x224). Interpolation may fail to compile.

        Note, a ``batch_size`` other than 1 might result in additional compiles
        if the last batch happens to be smaller.

        Args:
            name (str, optional): Name of model to load. Defaults to None.
            batch_size (int, optional): batch size of encoder. Defaults to 1.
            compile (bool, optional): Whether to use `torch.compile`. Defaults to False.
            cache_dir (str, optional): Path to cache folder.
            device (torch.device, optional): Device to use. Defaults to "cpu".
        """
        super(PatchAndViT, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.device = torch.device("cpu" if device is None else device)

        self.compile = compile
        self.compile_mode = None

        enc: ViTLastAttnModel = ViTLastAttnModel.from_pretrained(
            name, add_pooling_layer=False, torchscript=compile
        )
        self.enc = enc.to(device).eval().requires_grad_(False)
        self.cfg = self.enc.config

        self.cfg.return_dict = False
        # Both below can OOM if not careful.
        self.cfg.output_attentions = True
        # self.cfg.output_hidden_states = True

        self.cache = TensorCache(hash_model(enc), self.cache_dir)

    def _compile(self):
        """Cannot compile first when using multiprocessing. Must compile after fork."""
        if not self.compile or self.compile_mode is not None:
            return
        elif parent_process() is not None:
            self.compile_mode = "jit"
            # Compile doesn't work inside a daemonic child process.
            # See: https://github.com/pytorch/pytorch/issues/97992
            B, C, S = self.batch_size, self.cfg.num_channels, self.cfg.image_size
            x = torch.rand(B, C, S, S).to(self.device)
            with torch.no_grad():
                traced = torch.jit.trace(self.enc, x)
            self.enc = torch.jit.optimize_for_inference(traced)
        else:
            self.compile_mode = "dynamo"
            # `dynamic` doesn't work for ViTModel.
            # Positional encoding interpolation cannot compile either.
            self.enc: ViTModel = torch.compile(self.enc, mode="max-autotune")  # type: ignore
            self.enc.eval()  # Compile resets eval for some reason.

    def _get_vid_cache(
        self, vid: torch.Tensor
    ) -> Tuple[List[str | None], torch.Tensor | None, torch.Tensor | None]:
        """Cache video on frame-level."""
        hashes, pats_t, attns_t = [], [], []
        for im in vid:
            im_hash = hash_tensor(im)
            _, pats = self.cache.get_val(im_hash, CACHE_PATCHES)
            _, attns = self.cache.get_val(im_hash, CACHE_LAST_ATTNS)
            hashes.append(im_hash if None in (pats, attns) else None)
            pats_t.append(pats)
            attns_t.append(attns)
        miss = None in pats_t or None in attns_t
        pats_t = None if miss else torch.stack(pats_t)  # type: ignore
        attns_t = None if miss else torch.stack(attns_t)  # type: ignore
        return hashes, pats_t, attns_t

    def _put_vid_cache(
        self, hashes: List[str | None], pats_t: torch.Tensor, attns_t: torch.Tensor
    ):
        """Cache video on frame-level."""
        return self.cache.put_vid(hashes, pats_t, attns_t)

    @torch.inference_mode()
    def __call__(self, ims: torch.Tensor) -> torch.Tensor:
        """Split TCHW images into patches and encode using ViT.

        Args:
            ims (torch.Tensor): TCHW images.

        Returns:
            torch.Tensor: TCHW latent patches.
        """
        ori_device = ims.device
        key, pats_t, attns_t = self._get_vid_cache(ims)
        if pats_t is not None:
            return pats_t.to(ori_device)

        self._compile()
        ims = ims.to(self.device).requires_grad_(False)

        B = self.batch_size
        h, w = np.array(ims.shape[-2:]) // self.cfg.patch_size
        # print(ims[0:B].shape)

        bats = []
        for t in range(0, len(ims), B):
            im = ims[t : t + B]
            with ExitStack() as stack:
                # Disable inference mode as compile requires grad version counter.
                if self.compile_mode == "dynamo":
                    stack.enter_context(torch.inference_mode(False))
                    stack.enter_context(torch.no_grad())

                # Tuple of last_hidden_state, pooler_output, hidden_states, attentions.
                # All except last_hidden_state is optional, so the tuple length
                # varies depending on ViTConfig.
                o = (
                    # Interpolation completely unsupported for `torch.jit`.
                    self.enc(im)
                    if self.compile_mode == "jit"
                    else self.enc(im, interpolate_pos_encoding=True)
                )

            hiddens = o[0]
            attns = o[-1][-1]  # Keep only last layer attns.
            del o  # Save memory.

            for hid, attn in zip(hiddens, attns):
                bats.append(dict(last_hidden_state=hid, attentions=(attn,)))

        output = BaseModelOutputWithPooling(**default_collate(bats))
        pats = output.last_hidden_state
        pats = E.rearrange(pats[:, 1:], "t (h w) c -> t c h w", h=h, w=w)
        # TNPQ, where N is heads, P is each token, and Q are weights.
        attns = output.attentions[-1]
        # We only need the attention weights of the CLS token (token 0).
        attns = E.rearrange(attns[:, :, 0, 1:], "t n (h w) -> t n h w", h=h, w=w)
        self._put_vid_cache(key, pats, attns)
        return pats.to(ori_device)

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(model_hash={self.cache.model_hash}, name={self.name}, compile={self.compile}, cache_dir={self.cache_dir}, device={self.device})"


class TensorCache:
    """Checks if results are already cached."""

    def __init__(
        self,
        model_hash: str,
        cache_dir: str = None,
        attrs=[CACHE_PATCHES, CACHE_LAST_ATTNS],
    ):
        """Create CacheCheck."""
        self.model_hash = model_hash
        self.cache_dir = cache_dir
        self.cache = Cache(cache_dir)
        self.attrs = attrs

    @staticmethod
    def get_key(model_hash: str, tensor_hash: str, attr: str):
        """Get standardized cache key."""
        return f"{model_hash}/{attr}/{tensor_hash}"

    def put_vid(
        self, hashes: List[str | None], pats_t: torch.Tensor, attns_t: torch.Tensor
    ):
        """Cache video on frame-level."""
        for im_hash, pats, attns in zip(hashes, pats_t, attns_t):
            # If None, it is already in cache.
            if im_hash is None:
                continue
            self.put_val(im_hash, CACHE_PATCHES, pats)  # CHW
            self.put_val(im_hash, CACHE_LAST_ATTNS, attns)  # NHW

    def get_val(self, im_hash: str, attr: str) -> Tuple[str, torch.Tensor | None]:
        """Get associated key and value if already cached."""
        assert attr in self.attrs
        k = self.get_key(self.model_hash, im_hash, attr)
        v = self.cache.get(k, default=None, read=True)
        if v is not None:
            # print(f"{current_process().name} LOAD: {k}")
            v = torch.load(v, weights_only=True, map_location="cpu")
        return k, v

    def put_val(self, im_hash: str, attr: str, val: torch.Tensor):
        """Put tensor value into cache."""
        k = self.get_key(self.model_hash, im_hash, attr)
        # print(f"{current_process().name} SAVE: {k}")
        buf = BytesIO()
        val = val.detach().cpu().requires_grad_(False)
        torch.save(val, buf)
        buf.seek(0)
        self.cache.set(k, buf, read=True, tag=self.model_hash)
        return k

    def __call__(self, vid: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        """Check if frames of video already in cache.

        Args:
            vid (torch.Tensor): TCHW images.

        Returns:
            Dict[str, torch.Tensor | None]: Hashes and images not in cache.
        """
        results = {}
        for im in vid:
            im_hash = hash_tensor(im)
            found = all(
                self.get_key(self.model_hash, im_hash, k) in self.cache
                for k in self.attrs
            )
            if not found:
                results[im_hash] = im
        return results

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(model_hash={self.model_hash}, cache_dir={self.cache_dir}, keys={self.attrs})"


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
        args = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
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
