"""SlotModel evaluation script."""

import logging
from time import time
from typing import List

import einops as E
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.data import DAVISDataset, create_davis_dataloader
from video_cv_project.engine import Checkpointer, dump_vos_preds
from video_cv_project.models import SlotCPC, SlotModel
from video_cv_project.models.heads import AlphaSlotDecoder
from video_cv_project.utils import get_dirs, get_model_summary

log = logging.getLogger(__name__)

NUM_SLOTS = 16
NUM_ITERS = 3


def attn_weight_method(model: SlotModel, ims: torch.Tensor):
    """Use Slot Attention weights as labels."""
    slots = None
    ims = E.rearrange(ims, "t c h w -> t 1 c h w")
    weights: List[torch.Tensor] = []
    size_hw = (0, 0)
    for im in ims:
        # Number of slots = number of objects + 1 for background. Less fighting.
        # TBH maybe more slots better since the slot attention wasnt scale-trained properly.
        # Default DAVIS palette only has 22 colors to play with...
        slots, pats, attn = model(im, slots, num_slots=NUM_SLOTS, num_iters=NUM_ITERS)
        weights.append(attn)
        size_hw = pats.shape[-2:]
    h, w = size_hw
    preds = E.rearrange(weights, "t 1 s (h w) -> t s h w", h=h, w=w)  # type: ignore
    return preds


def alpha_mask_method(model: SlotCPC, ims: torch.Tensor):
    """Use AlphaSlotDecoder alpha masks as labels."""
    decoder: AlphaSlotDecoder = model.decoder  # type: ignore
    s = None
    pats_t = model._encode(ims[None])
    h, w = pats_t.shape[-2:]
    slots_t = torch.stack(
        [s := model.model.calc_slots(p, s, NUM_SLOTS, NUM_ITERS)[0] for p in pats_t]
    )
    preds = model.time_mlp(slots_t)
    preds = E.rearrange(preds, "t 1 s (p c) -> p t s c", p=model.num_preds)
    preds = decoder.get_alpha_masks(preds[0], (h, w))
    return preds


def eval(cfg: DictConfig):
    """Evaluate model."""
    assert cfg.resume is not None, "Must provide resume path in eval mode."
    root_dir, out_dir = get_dirs()

    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    log.info(f"Torch Device: {device}")

    log.debug("Create Model.")
    model = instantiate(cfg.model)
    checkpointer = Checkpointer(model=model)
    checkpointer.load(root_dir / cfg.resume)
    # TODO: What config values to overwrite?
    old_cfg = OmegaConf.create(checkpointer.cfg)
    log.debug(f"Ckpt Config:\n{old_cfg}")
    summary = get_model_summary(model.model, device=device, sizes=[(1, 3, 224, 224)])
    log.info(f"Model Summary for Input Shape {summary.input_size}:\n{summary}")

    log.debug("Create Eval Dataloader.")
    dataloader = create_davis_dataloader(cfg, 16)  # Labels not used so put anything.

    model.to(device).eval()

    dataset: DAVISDataset = dataloader.dataset  # type: ignore
    vid_names = dataset.videos
    has_palette = dataset.has_palette

    with torch.inference_mode():
        t_data, t_infer, t_save = time(), 0.0, 0.0
        for i, (ims, lbls, colors, meta) in enumerate(dataloader):
            B, T = ims.shape[:2]
            assert B == 1, "Video batch size must be 1."

            ims, lbls, colors, meta = ims[0], lbls[0], colors[0], meta[0]

            # Prepended frames are inferred on & contribute to run time.
            log.info(
                f"{i+1}/{len(dataloader)}: Processing {meta['im_dir']} with {T} frames."
            )
            log.debug(f"Data: {time() - t_data:.4f} s")

            save_dir = out_dir / "results"

            # Reset hidden state of encoder (if present).
            if hasattr(model, "reset"):
                log.debug("Reset Encoder.")
                model.reset()  # type: ignore

            t_infer = time()
            colors[0] = torch.Tensor([191, 128, 64])  # Temporary for visualization.
            preds_a = attn_weight_method(model.model, ims)
            preds_b = alpha_mask_method(model, ims)
            # Interleave the two predictions for visualization.
            preds = torch.stack([i for pair in zip(preds_a, preds_b) for i in pair])
            im_paths = [
                i for pair in zip(meta["im_paths"], meta["im_paths"]) for i in pair
            ]

            log.debug(f"Inference: {time() - t_infer:.4f} s")

            t_save = time()
            dump_vos_preds(
                save_dir,
                im_paths,
                preds,
                colors,
                has_palette=has_palette,
                blend_name=f"blends/{vid_names[i]}/%05d.jpg",
                mask_name=f"masks/{vid_names[i]}/%05d.png",
            )
            log.debug(f"Save: {time() - t_save:.4f} s")

            t_data = time()
