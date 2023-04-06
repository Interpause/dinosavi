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

NUM_SLOTS = 7
INI_ITERS = 1
NUM_ITERS = 1
TENSORBOARD_DIR = "."


def attn_weight_method(model: SlotModel, pats_t: torch.Tensor):
    """Use Slot Attention weights as labels."""
    slots = None
    pats_t = E.rearrange(pats_t, "t c h w -> t 1 c h w")
    weights: List[torch.Tensor] = []
    size_hw = pats_t.shape[-2:]
    i = INI_ITERS
    for pats in pats_t:
        slots, attn = model.forward(pats, slots, num_slots=NUM_SLOTS, num_iters=i)
        weights.append(attn)
        i = NUM_ITERS
    h, w = size_hw
    preds = E.rearrange(weights, "t 1 s (h w) -> t s h w", h=h, w=w)  # type: ignore
    return preds


def alpha_mask_method(model: SlotCPC, pats_t: torch.Tensor):
    """Use AlphaSlotDecoder alpha masks as labels."""
    decoder: AlphaSlotDecoder = model.decoder  # type: ignore
    s = None
    h, w = pats_t.shape[-2:]
    slots = []
    i = INI_ITERS
    for p in pats_t:
        s, _ = model.model.forward(p, s, NUM_SLOTS, i)
        slots.append(s)
        i = NUM_ITERS
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

    log.debug("Create Model.")
    model = instantiate(cfg.model, _convert_="all")
    checkpointer = Checkpointer(model=model)
    checkpointer.load(root_dir / cfg.resume)
    # TODO: What config values to overwrite?
    old_cfg = OmegaConf.create(checkpointer.cfg)
    log.debug(f"Ckpt Config:\n{old_cfg}")
    summary = get_model_summary(model, device=device)
    log.info(f"Model Summary for Input Shape {summary.input_size[0]}:\n{summary}")

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
            preds_a = attn_weight_method(model.model, pats_t)
            # preds_b = alpha_mask_method(model, pats_t)
            # tb_log_preds(writer, f"vid{i}-alpha", preds_b)
            # Interleave the two predictions for visualization.
            # preds = torch.stack([i for pair in zip(preds_a, preds_b) for i in pair])
            # im_paths = [
            #     i for pair in zip(meta["im_paths"], meta["im_paths"]) for i in pair
            # ]
            log.debug(f"Inference: {time() - t_infer:.4f} s")
            im_paths = meta["im_paths"]
            preds = preds_a

            t_save = time()
            tb_log_preds(writer, f"vid{i}-attn", preds_a)
            dump_vos_preds(
                save_dir,
                im_paths,
                preds.cpu(),
                colors,
                has_palette=has_palette,
                blend_name=f"blends/{vid_names[i]}/%05d.jpg",
                mask_name=f"masks/{vid_names[i]}/%05d.png",
            )
            log.debug(f"Save: {time() - t_save:.4f} s")

            t_data = time()
