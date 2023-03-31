"""Script to cache the dataset."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.data import create_kinetics_dataloader
from video_cv_project.engine import Trainer
from video_cv_project.models.encoders.video_wrapper import VideoWrapper

log = logging.getLogger(__name__)


def cache(cfg: DictConfig):
    """Cache dataset."""
    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    epochs = cfg.train.epochs
    log_every = cfg.train.log_every
    save_every = cfg.train.save_every

    log.info(f"Torch Device: {device}")
    log.info(f"Epochs: {epochs}")

    log.debug("Create Model.")
    model = instantiate(cfg.model, device=device, _convert_="all")
    model = VideoWrapper(model).eval()

    log.debug("Create Train Dataloader.")
    dataloader = create_kinetics_dataloader(cfg)
    with open_dict(cfg):
        cfg.total_steps = len(dataloader) * epochs
    log.info(f"Total Steps: {cfg.total_steps}")

    trainer = Trainer(
        dataloader, epochs, logger=log, log_every=log_every, save_every=save_every
    )

    log.info(f"Run through and cache dataset.")
    with torch.inference_mode():
        for i, n, video in trainer:
            video = video.to(device)
            model(video)
