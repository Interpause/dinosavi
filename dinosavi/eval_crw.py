"""Model evaluation script."""

import logging
from time import time

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dinosavi.cfg import BEST_DEVICE
from dinosavi.data import create_eval_dataloader
from dinosavi.engine import Checkpointer, dump_vos_preds, propagate_labels
from dinosavi.models import CRW
from dinosavi.utils import delete_layers, get_dirs, get_model_summary

log = logging.getLogger(__name__)


def eval(cfg: DictConfig):
    """Evaluate model."""
    assert cfg.resume is not None, "Must provide resume path in eval mode."
    root_dir, out_dir = get_dirs()

    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    log.info(f"Torch Device: {device}")
    context_len = cfg.eval.context_len
    log.info(f"Context Length: {context_len}")

    log.debug("Create Model.")
    model: CRW = instantiate(cfg.model, _convert_="all")
    delete_layers(model, ["head"])
    encoder = model.encoder
    summary = get_model_summary(encoder, device=device)
    log.info(f"Model Summary for Input Shape {summary.input_size[0]}:\n{summary}")
    log.info(f"Model scale: {model.map_scale}")

    log.debug("Create Eval Dataloader.")
    dataloader = create_eval_dataloader(cfg, model.map_scale)

    checkpointer = Checkpointer(model=model)
    resume_ckpt = root_dir / cfg.resume
    checkpointer.load(resume_ckpt)
    # TODO: What config values to overwrite?
    old_cfg = OmegaConf.create(checkpointer.cfg)
    log.info(f"Ckpt Config:\n{OmegaConf.to_object(old_cfg)}")

    encoder.to(device).eval()

    dataset = dataloader.dataset
    vid_names = dataset.videos  # type: ignore
    has_palette = dataset.has_palette  # type: ignore

    with torch.inference_mode():
        t_data, t_infer, t_save = time(), 0.0, 0.0
        for i, (ims, lbls, colors, meta) in enumerate(dataloader):
            T = len(ims)

            # Prepended frames are inferred on & contribute to run time.
            log.info(
                f"{i+1}/{len(dataloader)}: Processing {meta['im_dir']} with {T} frames."
            )
            log.debug(f"Data: {time() - t_data:.4f} s")

            save_dir = out_dir / "results"

            # Reset hidden state of encoder (if present).
            if hasattr(encoder, "reset"):
                log.debug("Reset Encoder.")
                encoder.reset()  # type: ignore

            t_infer = time()
            preds = propagate_labels(
                encoder,
                ims,
                lbls,
                context_len=context_len,
                topk=cfg.eval.topk,
                radius=cfg.eval.radius,
                temperature=cfg.eval.temperature,
                extra_idx=tuple(cfg.eval.extra_idx),
                batch_size=cfg.data.batch_size,
                device=device,
            )
            log.debug(f"Inference: {time() - t_infer:.4f} s")

            t_save = time()
            dump_vos_preds(
                save_dir,
                meta["im_paths"],
                preds,
                colors,
                has_palette=has_palette,
                blend_name=f"blends/{vid_names[i]}/%05d.jpg",
                mask_name=f"masks/{vid_names[i]}/%05d.png",
            )
            log.debug(f"Save: {time() - t_save:.4f} s")

            t_data = time()
