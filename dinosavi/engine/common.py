"""Code shared by most model applications."""

from functools import cache
from typing import Tuple

import einops as E
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from dinosavi.cfg import BEST_DEVICE

__all__ = [
    "infer_slot_labels",
    "calc_context_frame_idx",
    "create_spatial_mask",
    "batched_affinity",
    "propagate_labels",
]


@torch.inference_mode()
def infer_slot_labels(
    model,
    pats_t: torch.Tensor,
    num_slots: int = 7,
    num_iters: int = 1,
    ini_iters: int = 1,
    masks: torch.Tensor = None,
    method: str = "slot",
    track_method: str = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Infer labels using Slot Attention based model.

    Args:
        model (Any): Model.
        pats_t (torch.Tensor): TCHW latent pixels.
        num_slots (int, optional): Number of slots. Defaults to 7.
        num_iters (int, optional): Iterations for Slot Attention. Defaults to 1.
        ini_iters (int, optional): Iterations for first frame. Defaults to 1.
        masks (torch.Tensor, optional): TSN masks for each slot. Defaults to None.
        method (str, optional): Either "slot" or "alpha". Defaults to "slot".
        track_method (str, optional): Method used to rearrange predictions, either "overlap" or "slotsim". Defaults to None.
        temperature (float, optional): Temperature for `track_method`.

    Returns:
        torch.Tensor: TSHW slot predictions.
    """
    h, w = pats_t.shape[-2:]
    slots, iters = None, ini_iters

    pats_t = E.rearrange(pats_t, "t c h w -> t 1 c h w")
    preds, edges = [], []
    masks = None if masks is None else masks.unsqueeze(1)
    for i, p in enumerate(pats_t):
        prev = slots
        m = None if masks is None else masks[i]
        slots, attn = model.model(p, prev, num_slots, iters, m)

        if track_method == "slotsim" and prev is not None:
            a = F.normalize(prev[0], dim=1)
            b = F.normalize(slots[0], dim=1)
            edges.append(E.einsum(a, b, "s c, e c -> s e"))

        if method == "slot":
            preds.append(E.rearrange(attn, "1 s (h w) -> 1 s h w", h=h, w=w))
        elif method == "alpha":
            preds.append(model.get_masks(slots, (h, w)))
        else:
            assert False, f"Method not supported: {method}"

        # Reset first frame only things.
        iters = num_iters

    preds = torch.cat(preds, dim=0)

    if track_method == "overlap":
        t0, t1 = preds[:-1], preds[1:]
        t0 = E.repeat(t0, "t s h w -> t s e h w", e=num_slots)
        t1 = E.repeat(t1, "t s h w -> t e s h w", e=num_slots)

        # Below is IoU/Jaccard logically extended to non-binary masks.
        t01 = torch.stack([t0, t1])
        union = E.reduce(t01, "i t s e h w -> t s e h w", "max")
        union = E.reduce(union, "t s e h w -> t s e", "sum")
        intersect = E.reduce(t01, "i t s e h w -> t s e h w", "prod")
        intersect = E.reduce(intersect, "t s e h w -> t s e", "sum")
        edges = list(intersect / union)

        # RMS error approach.
        # edges = list(E.reduce((t1 - t0).square(), "t s e h w -> t s e", "sum").pow(-2))

    elif track_method == "slotsim":
        pass
    elif track_method is None:
        pass
    else:
        assert False, f"Unsupported track method: {track_method}"

    if track_method is not None:
        path: torch.Tensor = None  # type: ignore
        rearr = [preds[0]]
        for i, edge in enumerate(edges, start=1):
            edge = F.softmax(edge / temperature, dim=1)
            path = edge if path is None else path @ edge

            _, col = linear_sum_assignment(path, maximize=True)
            rearr.append(preds[i, col])

            # Greedy matching approach.
            # _, idx = path.flatten().topk(num_slots**2)
            # row, col = np.unravel_index(idx.numpy(force=True), path.shape)
            #
            # used = set()
            # pairs: list = [None] * num_slots
            # for s, e in zip(row, col):
            #     if pairs[s] is not None or e in used:
            #         continue
            #     pairs[s] = e
            #     used.add(e)
            #     if len(used) == num_slots:
            #         break
            # rearr.append(preds[i, pairs])

        return torch.stack(rearr, dim=0)
    return preds


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
    context = [torch.zeros(num_frames, 1)]
    for t in extra_idx:
        assert 0 <= t < num_frames
        n = context_len + t + 1  # Account for prepended repeats.
        # `idx` specifies whether frame `t` should be included for each frame.
        idx = torch.full((num_frames, 1), n)
        # The extra frame isn't included for frames before `t`. Due to limitations,
        # when `t` isn't included, frame 0 will be included.
        idx[:n] = 0
        context.append(idx)

    # Construct index of context frames for each frame.
    a = E.repeat(torch.arange(context_len), "n -> t n", t=num_frames)
    b = E.rearrange(torch.arange(num_frames), "t -> t 1")
    context.append(a + b)
    return torch.cat(context, dim=1).long()


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
    gy, gx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    # For each pixel, calculate distance to all other pixels.
    mask = (
        (gy[None, None, :, :] - gy[:, :, None, None]) ** 2
        + (gx[None, None, :, :] - gx[:, :, None, None]) ** 2
    ) ** 0.5
    # `p` refers to context pixels, `q` refers to query pixels.
    mask = E.rearrange(mask, "px py qx qy -> (px py) (qx qy)")
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
        query (torch.Tensor): TCQ query pixels.
        keys (torch.Tensor): TNCP context pixels.
        mask (torch.Tensor): PQ bitmask, where the value is True if masked.
        topk (int): Top-k pixels to use for label propagation.
        temperature (float): Temperature for softmax.
        batch_size (int): Batch size for caculating affinities.
        num_extra_frames (int, optional): Extra frames always included in context.
        device (torch.device, optional): Device to use. Defaults to BEST_DEVICE.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            Is: TkQ indexes of top-k context pixels for each query pixel.
            Ws: TkQ affinities of top-k context pixels for each query pixel.
    """
    # TODO: Consider implementing original's pixel-level batching.
    # Hardcode to 1, see below note on issue with `torch.topk`.
    b = 1  # batch_size
    # Penalty to hard mask pixels outside radius.
    mask = mask.float().to(device).masked_fill(mask, float("-inf"))

    Is, Ws = [], []
    for t in range(0, len(keys), b):
        bk = keys[t : t + b].to(device)
        bq = query[t : t + b].to(device)

        # TNPQ: P & Q are pixels for each frame, N is each context frame.
        A = E.einsum(bk, bq, "t n c p, t c q -> t n p q")

        # Apply spatial mask, skipping frames always included in context.
        A[:, num_extra_frames:] += mask

        # `q` are query pixels, `(n p)` are context pixels.
        A = E.rearrange(A, "t n p q -> t (n p) q")

        # Indexes of top-k pixels and their affinities.
        # NOTE: https://github.com/pytorch/pytorch/issues/82569
        # Must use PyTorch 2.0 for `torch.topk` to work with large matrices.
        weights, idx = A.topk(topk, dim=1)
        weights = F.softmax(weights / temperature, dim=1)

        Is.append(idx.cpu())
        Ws.append(weights.cpu())

    return torch.cat(Is), torch.cat(Ws)


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
        ims (torch.Tensor): TCHW images.
        lbls (torch.Tensor): TNHW bitmasks for each class N.
        context_len (int, optional): Number of context frames.
        topk (int, optional): Top-k pixels to use for label propagation.
        radius (float, optional): Radius of attention mask.
        temperature (float, optional): Temperature for softmax.
        extra_idx (Tuple[int, ...], optional): Extra frames always included in context.
        batch_size (int, optional): Batch size for encoding images & calculating affinities.
        device (torch.device, optional): Device to use. Defaults to BEST_DEVICE.

    Returns:
        torch.Tensor: TNHW propagated labels.
    """
    b = batch_size
    T = len(ims)
    # T includes prepended repeats of initial frame.
    num_frames = T - context_len

    # Mask labels for frames after the initial frame.
    lbls[context_len:] *= 0

    # TODO: Don't repeat encode for repeat frames.
    ims = E.rearrange(ims, "t c h w -> 1 t c h w")  # Add fake batch dim.
    bats = [encoder(ims[:, t : t + b].to(device)).cpu() for t in range(0, T, b)]
    feats = F.normalize(torch.cat(bats, dim=1), p=2, dim=2)[0]  # Euclidean norm.

    # TN, where N is the context frames for each time step T.
    key_idx = calc_context_frame_idx(num_frames, context_len, extra_idx)
    # Map of frame at `t` to context pixels `(h w)` for each context frame `n`.
    keys = E.rearrange(feats[key_idx], "t n c h w -> t n c (h w)")
    # `(h w)` query pixels for each frame `t`; Also remove prepended repeats.
    query = E.rearrange(feats[context_len:], "t c h w -> t c (h w)")
    # PQ mask, where P are context pixels & Q are query pixels.
    mask = create_spatial_mask(*feats.shape[-2:], radius)

    # TkQ indexes, TkQ weights, where k is the top-k context pixels.
    topk_idx, weights = batched_affinity(
        query,
        keys,
        mask,
        topk,
        temperature,
        batch_size,
        len(extra_idx) + 1,  # Add 1 as initial frame is always included.
        device=device,
    )

    _, w = feats.shape[-2:]
    preds = []
    # Frame by frame propagation.
    for t in range(num_frames):
        # NLHW context labels, where N are context frames & L are label classes.
        ctx = lbls[key_idx[t]].to(device)
        ctx = E.rearrange(ctx, "n l h w -> l (n h w)")
        # kQ weights for top-k context pixels.
        c = weights[t].to(device)

        # Weighted sum of top-k context pixels.
        pred = E.reduce(ctx[:, topk_idx[t]] * c, "l k (h w) -> l h w", "sum", w=w)

        # Propagate labels.
        lbls[context_len + t] = pred
        preds.append(pred.cpu())
    return torch.stack(preds)
