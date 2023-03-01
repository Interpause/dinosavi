"""Code shared by most model applications."""

from functools import cache
from typing import List, Tuple

import torch
import torch.nn.functional as F

from video_cv_project.cfg import BEST_DEVICE

__all__ = [
    "calc_context_frame_idx",
    "create_spatial_mask",
    "batched_affinity",
    "propagate_labels",
]


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
        extra_idx (Tuple[int, ...], optional): Extra frames to include in context.

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
    # TODO: Consider implementing original's pixel-level batching.
    # Hardcode to 1, see below note on issue with `torch.topk`.
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
        # NOTE: https://github.com/pytorch/pytorch/issues/82569
        # Bug with `torch.topk` since 1.13.1.
        # Mitigation: Reduce size of affinity matrix by decreasing batch size,
        # context length, batching at pixel level, or using sparse matrix.
        weights, idx = torch.topk(A, topk, dim=2)
        weights = F.softmax(weights / temperature, dim=2)

        Is.append(idx.cpu())
        Ws.append(weights.cpu())

    return torch.cat(Is, dim=1), torch.cat(Ws, dim=1)


def propagate_labels(
    encoder: torch.nn.Module,
    ims: torch.Tensor,
    lbls: torch.Tensor,
    context_len: int = 20,
    topk: int = 10,
    radius: float = 12.0,
    temperature: float = 0.05,
    extra_idx: Tuple[int, ...] = tuple(),
    batch_size: int = 1,
    device: torch.device = BEST_DEVICE,
):
    """Propagate labels.

    Args:
        encoder (torch.nn.Module): Encoder used to extract image features.
        ims (torch.Tensor): BTCHW images.
        lbls (torch.Tensor): BTNHW bitmasks for each class N.
        context_len (int, optional): Number of context frames.
        topk (int, optional): Top-k pixels to use for label propagation.
        radius (float, optional): Radius of attention mask.
        temperature (float, optional): Temperature for softmax.
        extra_idx (Tuple[int, ...], optional): Extra frames always included in context.
        batch_size (int, optional): Batch size for encoding images & calculating affinities.
        device (torch.device, optional): Device to use. Defaults to BEST_DEVICE.

    Returns:
        torch.Tensor: BTNHW propagated labels.
    """
    b = batch_size
    T = ims.shape[1]
    # T includes prepended repeats of initial frame.
    num_frames = T - context_len

    # Mask labels for frames after the initial frame.
    lbls[:, context_len:] *= 0

    # BTCHW -> BCTHW
    ims = ims.transpose(1, 2)
    # TODO: Don't repeat encode for repeat frames.
    feats = torch.cat(
        [encoder(ims[:, :, t : t + b].to(device)).cpu() for t in range(0, T, b)], dim=2
    )
    feats = F.normalize(feats, p=2, dim=1)  # Euclidean norm.

    # TN, where N is the context frames for each time step T.
    key_idx = calc_context_frame_idx(num_frames, context_len, extra_idx)
    # BCTNHW context frames for each frame.
    keys = feats[:, :, key_idx]
    # BCTNHW -> BCTN(H*W) context features.
    keys = keys.flatten(-2)
    # Remove prepended repeats.
    query = feats[:, :, context_len:]
    # BCTHW -> BCT(H*W) query frames.
    query = query.flatten(-2)
    # PQ, where P is each pixel & Q is each other pixel.
    mask = create_spatial_mask(*feats.shape[-2:], radius)

    # BTkQ indexes, BTkQ weights, where k is the top-k context pixels.
    topk_idx, weights = batched_affinity(
        query,
        keys,
        mask,
        topk,
        temperature,
        batch_size,
        len(extra_idx) + 1,
        device=device,
    )

    batches = []
    # By right should only have 1 batch.
    for _lbls, Is, Ws in zip(lbls, topk_idx, weights):
        preds = []
        # Frame by frame propagation.
        for t in range(num_frames):
            # TNHW context labels, where T are context frames & N are label classes.
            ctx_lbls = _lbls[key_idx[t]].to(device)
            # TNHW -> -> NTHW -> N(T*H*W) context pixels.
            ctx_lbls = ctx_lbls.transpose(0, 1).flatten(1)
            # kQ weights for top-k context pixels.
            w = Ws[t].to(device)

            # Weighted sum of top-k context pixels.
            # NQ -> NHW predicted label classes for each query pixel.
            pred = (ctx_lbls[:, Is[t]] * w).sum(1).reshape(-1, *feats.shape[-2:])

            # TODO: Original used real labels for frame 0. Is there some reason?
            # Propagate labels.
            _lbls[context_len + t] = pred

            preds.append(pred.cpu())
        batches.append(torch.stack(preds, dim=0))
    return torch.stack(batches, dim=0)
