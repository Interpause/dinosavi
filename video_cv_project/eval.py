"""Model evaluation script."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.checkpointer import Checkpointer
from video_cv_project.data.davis import create_davis_dataloader
from video_cv_project.models import CRW
from video_cv_project.utils import get_dirs

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

    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    log.info(f"Torch Device: {device}")

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

    with torch.inference_mode():
        for idx, (ims, lbls, lbl_cls, meta) in enumerate(dataloader):
            print(ims.shape, lbls.shape, lbl_cls.shape)
            assert False
            ims = ims.to(device)
            lbls = lbls.to(device)
            lbl_cls = lbl_cls.to(device)
