"""SlotModel evaluation script."""

import logging
from time import time
from typing import List

import einops as E
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.data import DAVISDataset, create_davis_dataloader
from video_cv_project.engine import Checkpointer, dump_vos_preds
from video_cv_project.models import SlotCPC, SlotModel
from video_cv_project.models.heads import AlphaSlotDecoder
from video_cv_project.utils import get_dirs, get_model_summary, tb_hparams

log = logging.getLogger(__name__)

TENSORBOARD_DIR = "."


def attn_weight_method(
    model: SlotCPC,
    pats_t: torch.Tensor,
    num_slots: int = 7,
    num_iters: int = 1,
    ini_iters: int = 1,
    lbl: torch.Tensor = None,
):
    """Use Slot Attention weights as labels."""
    model = model.model
    s = None
    h, w = pats_t.shape[-2:]
    pats_t = E.rearrange(pats_t, "t c h w -> t 1 c h w")
    weights: List[torch.Tensor] = []
    i = ini_iters
    m = None if lbl is None else E.rearrange(lbl, "s h w -> 1 s (h w)").bool()
    for p in pats_t:
        s, attn = model(p, s, num_slots, i, m)
        weights.append(attn)

        # Reset initial frame only things.
        i, m = num_iters, None
    preds = E.rearrange(weights, "t 1 s (h w) -> t s h w", h=h, w=w)  # type: ignore
    return preds


def alpha_mask_method(
    model: SlotCPC,
    pats_t: torch.Tensor,
    num_slots: int = 7,
    num_iters: int = 1,
    ini_iters: int = 1,
    lbl: torch.Tensor = None,
):
    """Use AlphaSlotDecoder alpha masks as labels."""
    s = None
    h, w = pats_t.shape[-2:]
    pats_t = E.rearrange(pats_t, "t c h w -> t 1 c h w")
    decoder: AlphaSlotDecoder = model.decoder  # type: ignore
    slots = []
    i = ini_iters
    m = None if lbl is None else E.rearrange(lbl, "s h w -> 1 s (h w)").bool()
    for p in pats_t:
        s, _ = model.model(p, s, num_slots, i, m)
        slots.append(s)

        # Reset initial frame only things.
        i, m = num_iters, None
    slots_t = torch.stack(slots)
    preds = model.time_mlp[0](slots_t)
    # TODO: add batching here.
    preds = E.rearrange(preds, "t 1 s c -> t s c")
    preds = decoder.get_alpha_masks(preds, (h, w))
    return preds


def tb_log_preds(writer: SummaryWriter, tag: str, preds: torch.Tensor):
    """Log the attention & alpha masks."""
    preds = E.rearrange(preds, "t s h w -> s t h w 1")
    # Normalize to [0, 1].
    preds = (preds - preds.min()) / (preds.max() - preds.min())
    for i, p in enumerate(preds):
        writer.add_images(f"{tag}/{i}", p, dataformats="NHWC")


def eval(cfg: DictConfig):
    """Evaluate model."""
    assert cfg.resume is not None, "Must provide resume path in eval mode."
    root_dir, out_dir = get_dirs()

    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    log.info(f"Torch Device: {device}")

    checkpointer = Checkpointer()
    checkpointer.load(root_dir / cfg.resume)
    # TODO: What config values to overwrite?
    old_cfg = OmegaConf.create(checkpointer.cfg)
    log.debug(f"Ckpt Config:\n{old_cfg.model}")

    log.debug("Create Model.")
    model = instantiate(old_cfg.model, _convert_="all")
    checkpointer.model = model
    checkpointer.reload()
    summary = get_model_summary(model, device=device)
    log.info(f"Model Summary for Input Shape {summary.input_size[0]}:\n{summary}")

    # Can override some stuff here.
    lock_on = cfg.lock_on
    output_mode = cfg.output
    num_slots = model.num_slots if cfg.num_slots is None else cfg.num_slots
    num_bslots = cfg.num_bg_slots
    num_cslots = cfg.slots_per_class
    use_bgfg = isinstance(num_slots, tuple)
    num_iters = model.num_iters if cfg.num_iters is None else cfg.num_iters
    ini_iters = model.ini_iters if cfg.ini_iters is None else cfg.ini_iters

    log.info(
        f"""Lock On: {lock_on}
BG FG Mode: {use_bgfg}
Output Mode: {output_mode}
Num Slots: {num_cslots if lock_on else num_slots}
Num Iters: {num_iters}
Ini Iters: {ini_iters}
        """
    )

    log.debug("Create Eval Dataloader.")
    dataloader = create_davis_dataloader(cfg, 1)

    log.debug("Create Encoder.")
    encoder = instantiate(cfg.data.transform.patch_func)
    log.info(f"Encoder:\n{encoder}")

    # Tensorboard Writer only used to log some visualizations.
    writer = SummaryWriter(log_dir=TENSORBOARD_DIR)
    # model.is_trace = True
    # writer.add_graph(model, torch.zeros(summary.input_size[0]).to(device))
    # model.is_trace = False
    # writer.add_hparams(tb_hparams(old_cfg), {})

    model.to(device).eval()

    dataset: DAVISDataset = dataloader.dataset  # type: ignore
    vid_names = dataset.videos
    has_palette = dataset.has_palette

    with torch.inference_mode():
        t_data, t_infer, t_save = time(), 0.0, 0.0
        for i, (ims, lbls, colors, meta) in enumerate(dataloader):
            T, save_dir = len(ims), out_dir / "results"
            # Prepended frames are inferred on & contribute to run time.
            log.info(
                f"{i+1}/{len(dataloader)}: Processing {meta['im_dir']} with {T} frames."
            )
            log.debug(f"Data: {time() - t_data:.4f} s")

            t_infer = time()
            pats_t, _ = encoder(ims)
            pats_t = pats_t.to(device)

            # If lock on is used, override number of slots to match labels.
            if lock_on:
                lbl = F.interpolate(lbls[:1], pats_t.shape[-2:])[0].to(device)
                # NOTE: Below assumes background class id is 0.
                bg_lbl = E.repeat(lbl[:1], "1 h w -> (1 s) h w", s=num_bslots)
                fg_lbl = E.repeat(lbl[1:], "n h w -> (n s) h w", s=num_cslots)
                lbl = torch.cat([bg_lbl, fg_lbl], dim=0)
                num_slots = (len(bg_lbl), len(fg_lbl)) if use_bgfg else len(lbl)
            else:
                lbl = None
                # Was black which is hard to see.
                colors[0] = torch.Tensor([191, 128, 64])

            if output_mode == "slot":
                method = attn_weight_method  # type: ignore
                log_label = "attn"
            elif output_mode == "alpha":
                method = alpha_mask_method  # type: ignore
                log_label = "alpha"
            else:
                assert False, f'Output mode "{output_mode}" unsupported!'
            preds = method(model, pats_t, num_slots, num_iters, ini_iters, lbl)

            # Map multiple slots back to specific class when using lock on.
            if lock_on:
                # Merge class slots by taking max.
                bg_preds = E.reduce(preds[:, :num_bslots], "t s h w -> t 1 h w", "max")
                fg_preds = E.reduce(
                    preds[:, num_bslots:], "t (n s) h w -> t n h w", "max", s=num_cslots
                )
                # Comment below out to see what each slot is doing.
                preds = torch.cat([bg_preds, fg_preds], dim=1)
                # Flip order of preds to allow palette to be assigned to foreground slots.
                # preds = preds[:, ::-1]

            log.debug(f"Inference: {time() - t_infer:.4f} s")

            t_save = time()
            tb_log_preds(writer, f"vid{i}-{log_label}", preds)
            dump_vos_preds(
                save_dir,
                meta["im_paths"],
                preds.cpu(),
                colors,
                has_palette=has_palette,
                blend_name=f"blends/{vid_names[i]}/%05d.jpg",
                mask_name=f"masks/{vid_names[i]}/%05d.png",
            )
            log.debug(f"Save: {time() - t_save:.4f} s")

            t_data = time()
