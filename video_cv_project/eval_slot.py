"""SlotModel evaluation script."""

import logging
from time import time

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.data import DAVISDataset, create_davis_dataloader
from video_cv_project.engine import Checkpointer, dump_vos_preds, infer_slot_labels
from video_cv_project.utils import (
    calc_lock_on_masks,
    get_dirs,
    get_model_summary,
    preds_from_lock_on,
    tb_hparams,
    tb_log_preds,
)

log = logging.getLogger(__name__)

TENSORBOARD_DIR = "."


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
    log.info(f"Ckpt Config:\n{OmegaConf.to_object(old_cfg.model)}")

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
    use_bgfg = not isinstance(num_slots, int)
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
                lbl = F.interpolate(lbls[:1], pats_t.shape[-2:]).to(device)
                lbl, num_slots = calc_lock_on_masks(lbl, num_bslots, num_cslots)
                num_slots = num_slots if use_bgfg else sum(num_slots)
            else:
                lbl = None
                # Was black which is hard to see.
                colors[0] = torch.Tensor([191, 128, 64])

            preds = infer_slot_labels(
                model, pats_t, num_slots, num_iters, ini_iters, lbl, output_mode
            )

            # Map multiple slots back to specific class when using lock on.
            if lock_on:
                # Comment below out to see what each slot is doing.
                preds = preds_from_lock_on(preds, num_bslots, num_cslots)
                # Flip order of preds to allow palette to be assigned to foreground slots.
                # preds = preds[:, ::-1]

            log.debug(f"Inference: {time() - t_infer:.4f} s")

            t_save = time()
            tb_log_preds(writer, f"vid{i}-{output_mode}", preds)
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
