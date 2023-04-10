"""SlotModel evaluation script."""

import logging
from time import time
from typing import List

import einops as E
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.data import DAVISDataset, create_davis_dataloader
from video_cv_project.engine import Checkpointer, dump_vos_preds
from video_cv_project.models import SlotCPC, SlotModel
from video_cv_project.models.heads import AlphaSlotDecoder
from video_cv_project.utils import get_dirs, get_model_summary, tb_hparams

log = logging.getLogger(__name__)

TENSORBOARD_DIR = "."


def attn_weight_method(
    model: SlotModel,
    pats_t: torch.Tensor,
    num_slots: int = 7,
    num_iters: int = 1,
    ini_iters: int = 1,
):
    """Use Slot Attention weights as labels."""
    s = None
    h, w = pats_t.shape[-2:]
    pats_t = E.rearrange(pats_t, "t c h w -> t 1 c h w")
    weights: List[torch.Tensor] = []
    i = ini_iters
    for p in pats_t:
        s, attn = model.forward(p, s, num_slots, i)
        weights.append(attn)
        i = num_iters
    preds = E.rearrange(weights, "t 1 s (h w) -> t s h w", h=h, w=w)  # type: ignore
    return preds


def alpha_mask_method(
    model: SlotCPC,
    pats_t: torch.Tensor,
    num_slots: int = 7,
    num_iters: int = 1,
    ini_iters: int = 1,
):
    """Use AlphaSlotDecoder alpha masks as labels."""
    s = None
    h, w = pats_t.shape[-2:]
    pats_t = E.rearrange(pats_t, "t c h w -> t 1 c h w")
    decoder: AlphaSlotDecoder = model.decoder  # type: ignore
    slots = []
    i = ini_iters
    for p in pats_t:
        s, _ = model.model.forward(p, s, num_slots, i)
        slots.append(s)
        i = num_iters
    slots_t = torch.stack(slots)
    preds = model.time_mlp[0](slots_t)
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
    num_slots = model.num_slots if cfg.num_slots is None else cfg.num_slots
    num_iters = model.num_iters if cfg.num_iters is None else cfg.num_iters
    ini_iters = model.ini_iters if cfg.ini_iters is None else cfg.ini_iters
    output_mode = cfg.output
    log.info(
        f"Num Slots: {num_slots}\nNum Iters: {num_iters}\nIni Iters: {ini_iters}\nOutput Mode: {output_mode}"
    )

    log.debug("Create Eval Dataloader.")
    dataloader = create_davis_dataloader(cfg, 16)  # Labels not used so put anything.

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
        for i, (ims, _, colors, meta) in enumerate(dataloader):
            T = len(ims)

            # Prepended frames are inferred on & contribute to run time.
            log.info(
                f"{i+1}/{len(dataloader)}: Processing {meta['im_dir']} with {T} frames."
            )
            log.debug(f"Data: {time() - t_data:.4f} s")

            save_dir = out_dir / "results"

            colors[0] = torch.Tensor([191, 128, 64])  # Temporary for visualization.

            t_infer = time()
            pats_t = encoder(ims).to(device)
            if output_mode == "slot":
                preds = attn_weight_method(
                    model.model, pats_t, num_slots, num_iters, ini_iters
                )
                log_label = "attn"
            elif output_mode == "alpha":
                preds = alpha_mask_method(
                    model, pats_t, num_slots, num_iters, ini_iters
                )
                log_label = "alpha"
            else:
                assert False, f'Output mode "{output_mode}" unsupported!'
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
