"""Code for applying model for Video Object Segmentation."""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from video_cv_project.cfg import BEST_DEVICE

from .common import batched_affinity, calc_context_frame_idx, create_spatial_mask

__all__ = ["vos_propagate"]


# TODO: Refactor this.
def vos_propagate(
    encoder: torch.nn.Module,
    save_dir: str,
    ims: torch.Tensor,
    orig_ims: torch.Tensor,
    lbls: torch.Tensor,
    lbl_cls: torch.Tensor,
    batch_size: int = 1,
    context_len: int = 20,
    radius: float = 12.0,
    temperature: float = 0.05,
    topk: int = 10,
    device: torch.device = BEST_DEVICE,
):
    """Propagate labels."""
    b = batch_size
    T = ims.shape[1]
    # T includes prepended repeats of initial frame.
    num_frames = T - context_len

    # BTCHW -> BCTHW
    ims = ims.transpose(1, 2)
    feats = torch.cat(
        [encoder(ims[:, :, t : t + b].to(device)).cpu() for t in range(0, T, b)], dim=2
    )
    feats = F.normalize(feats, p=2, dim=1)  # Euclidean norm.

    # TN, where N is the context frames for each time step T.
    key_idx = calc_context_frame_idx(num_frames, context_len)
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
        query, keys, mask, topk, temperature, batch_size, device=device
    )

    # Mask labels for frames after the initial frame.
    lbls[:, context_len:] *= 0

    # By right should only have 1 batch.
    for _b in range(ims.shape[0]):
        _lbl_cls, _lbls = lbl_cls[_b], lbls[_b]
        Is, Ws = topk_idx[_b], weights[_b]

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

            # Save predictions.
            # TODO: Load image at runtime from metadata so saving can be disabled.
            # Or else, image normalization can be done here instead of the dataset transform.
            # Meaning no need to waste memory for both `orig_ims` & `ims`.
            cur_img = orig_ims[0, t + context_len]
            dump_vos_preds(save_dir, f"{t:04d}", cur_img, pred.cpu(), _lbl_cls)


def dump_vos_preds(
    save_dir: str,
    name: str,
    img: torch.Tensor,
    pred: torch.Tensor,
    lbl_cls: torch.Tensor,
):
    """Dump predictions for VOS visualization."""
    sz = img.shape[-2:]

    # Resize labels to original size.
    # NOTE: Maybe using a different interpolation method here (or PIL) can squeeze contour accuracy.
    pred_dist = TF.resize(pred, sz)

    # Argmax to get predicted class for each pixel.
    pred_lbl = torch.argmax(pred_dist, dim=0)

    # Overlay predicted labels on original image.
    pred_lbl = lbl_cls[pred_lbl].permute(2, 0, 1) / 255.0
    labelled = img * 0.5 + pred_lbl * 0.5
    save_image(labelled, str(Path(save_dir) / f"{name}.jpg"))
