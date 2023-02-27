"""Code shared by most model applications."""

from functools import cache
from typing import List, Tuple

import torch
import torch.nn.functional as F

from video_cv_project.cfg import BEST_DEVICE

__all__ = ["calc_context_frame_idx", "create_spatial_mask", "batched_affinity"]


@cache
def calc_context_frame_idx(
    num_frames: int,
    context_len: int,
    extra_idx: Tuple[int, ...] = tuple(),
):
    """Calculate indexes of context frames for each frame.

    ``extra_idx`` is used to specify frames that should always be in context.
    ``num_frames`` should exclude prepended repeats of the initial frame. Due to
    limitations, the initial frame will always be included in the context for all
    frames.

    As this function is memoized, ``extra_idx`` must be specified as tuple.

    Args:
        num_frames (int): Number of frames in video.
        context_len (int): Number of past frames to use as context.
        extra_idx (Tuple[int, ...], optional): Extra frames to include in context. Defaults to [].

    Returns:
        torch.Tensor: TN indexes of context frames for each frame, where T is the current frame & N is each context frame.
    """
    context: List[torch.Tensor] = [torch.zeros(num_frames, 1, dtype=torch.long)]
    for t in extra_idx:
        assert 0 <= t < num_frames
        n = context_len + t + 1  # Account for prepended repeats.
        # `idx` specifies whether frame `t` should be included for each frame.
        idx = torch.full((num_frames, 1), n, dtype=torch.long)
        # The extra frame isn't included for frames before `t`. Due to limitations,
        # when `t` isn't included, frame 0 will be included.
        idx[:n] = 0
        context.append(idx)

    # Construct index of context frames for each frame.
    a = torch.arange(context_len).expand(num_frames, -1)
    b = torch.arange(num_frames).unsqueeze(1)
    context.append(a + b)
    return torch.cat(context, dim=1)


@cache
def create_spatial_mask(height: int, width: int, radius: float):
    """Create spatial mask by radius.

    Height & width is flattened when creating the mask. This function is memoized.

    Args:
        height (int): Height of latent image.
        width (int): Width of latent image.
        radius (float): Radius of mask.

    Returns:
        torch.Tensor: PQ bitmask, where the value is True if masked.
    """
    gy, gx = torch.meshgrid(
        torch.arange(height, dtype=torch.float),
        torch.arange(width, dtype=torch.float),
        indexing="ij",
    )
    # For each pixel, calculate distance to all other pixels.
    mask = (
        (gy[None, None, :, :] - gy[:, :, None, None]) ** 2
        + (gx[None, None, :, :] - gx[:, :, None, None]) ** 2
    ) ** 0.5  # XYXY
    # XYXY -> PQ, where P is each pixel & Q is each other pixel.
    mask = mask.flatten(0, 1).flatten(1, 2)
    # TODO: Consider "smoothing" the mask?
    return mask > radius


def batched_affinity(
    query: torch.Tensor,
    keys: torch.Tensor,
    mask: torch.Tensor,
    topk: int,
    temperature: float,
    batch_size: int,
    num_extra_frames: int = 1,
    device: torch.device = BEST_DEVICE,
):
    """Find indexes of top-k pixels and their affinities.

    Affinities are later used as weights during label propagation.

    Args:
        query (torch.Tensor): BCTQ query pixels.
        keys (torch.Tensor): BCTNP context pixels.
        mask (torch.Tensor): PQ bitmask, where the value is True if masked.
        topk (int): Top-k pixels to use for label propagation.
        temperature (float): Temperature for softmax.
        batch_size (int): Batch size for caculating affinities.
        num_extra_frames (int, optional): Extra frames always included in context.
        device (torch.device, optional): Device to use. Defaults to BEST_DEVICE.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            Is: BTkQ indexes of top-k context pixels for each query pixel.
            Ws: BTkQ affinities of top-k context pixels for each query pixel.
    """
    # Memory corruption seems to occur even at batch size of 4.
    # hardcode to 1, ~= 4GB VRAM usage.
    # TODO: Consider implementing original's pixel-level batching.
    b = 1
    T = keys.shape[2]
    # -1e10 penalty to hard mask pixels outside radius.
    mask = (mask * -1e10).to(device)

    Is, Ws = [], []
    for t in range(0, T, b):
        _k = keys[:, :, t : t + b].to(device)
        _q = query[:, :, t : t + b].to(device)

        # BTNPQ: P & Q are pixels for each frame, N is each context frame.
        A = torch.einsum("bctnp,bctq->btnpq", _k, _q)

        # Apply spatial mask, skipping frames always included in context.
        A[:, :, num_extra_frames:] += mask

        # BTNPQ -> BT(N*P)Q
        # Q are query pixels (current frame), N*P are context pixels.
        A = A.flatten(2, 3)

        # Indexes of top-k pixels and their affinities.
        weights, idx = torch.topk(A, topk, dim=2)
        weights = F.softmax(weights / temperature, dim=2)

        Is.append(idx.cpu())
        Ws.append(weights.cpu())

    return torch.cat(Is, dim=1), torch.cat(Ws, dim=1)
