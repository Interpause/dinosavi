"""Model evaluation script."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.checkpointer import Checkpointer
from video_cv_project.data import create_davis_dataloader
from video_cv_project.engine import dump_vos_preds, propagate_labels
from video_cv_project.models import CRW
from video_cv_project.utils import get_dirs, perf_hack

log = logging.getLogger(__name__)

CKPT_FOLDER = "weights"
CKPT_EXT = ".ckpt"
LATEST_NAME = f"latest{CKPT_EXT}"
MODEL_NAME = f"epoch%d{CKPT_EXT}"
SAMPLE_INPUT = [1, 3, 8, 320, 320]


def eval(cfg: DictConfig):
    """Evaluate model."""
    assert cfg.resume is not None, "Must provide resume path in eval mode."

    root_dir, out_dir = get_dirs()
    ckpt_dir = out_dir / CKPT_FOLDER
    ckpt_dir.mkdir(exist_ok=False)  # Error if exists to prevent model overwrite.

    perf_hack()
    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    log.info(f"Torch Device: {device}")
    context_len = cfg.eval.context_len
    log.info(f"Context Length: {context_len}")

    log.debug("Create Model.")
    model: CRW = instantiate(cfg.model)
    encoder = model.encoder
    log.debug("Create Eval Dataloader.")
    dataloader = create_davis_dataloader(cfg, model.map_scale)

    checkpointer = Checkpointer(model=model)
    resume_ckpt = root_dir / cfg.resume
    checkpointer.load(resume_ckpt)
    old_cfg = OmegaConf.create(checkpointer.cfg)
    log.debug(f"Ckpt Config:\n{old_cfg}")

    # TODO: What config values to overwrite?

    model_summary = summary(
        encoder, SAMPLE_INPUT, verbose=0, col_width=20, device=device
    )
    model_summary.formatting.layer_name_width = 30
    log.info(f"Model Summary for Input Shape {SAMPLE_INPUT}:\n{model_summary}")
    log.info(f"Model scale: {model.map_scale}")

    encoder.to(device).eval()

    vid_names = dataloader.dataset.videos
    palette = dataloader.dataset.palette

    with torch.inference_mode():
        for i, (ims, orig_ims, lbls, lbl_cls, meta) in enumerate(dataloader):
            B, T = ims.shape[:2]
            assert B == 1, "Video batch size must be 1."

            # Prepended frames are inferred on & contribute to run time.
            log.info(
                f"{i+1}/{len(dataloader)}: Processing {meta[0]['im_dir']} with {T} frames."
            )

            save_dir = out_dir / "results"

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

            dump_vos_preds(
                save_dir,
                orig_ims[0, context_len:],
                preds[0],
                lbl_cls[0],
                palette=palette,
                blend_name=f"blends/{vid_names[i]}/%05d.jpg",
                mask_name=f"masks/{vid_names[i]}/%05d.png",
            )
