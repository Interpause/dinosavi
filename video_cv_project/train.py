"""TODO: Add module docstring."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from video_cv_project.checkpointer import Checkpointer
from video_cv_project.data import create_kinetics400_dataloader
from video_cv_project.utils import get_dirs, iter_pbar

log = logging.getLogger(__name__)

CKPT_FOLDER = "weights"
CKPT_EXT = ".ckpt"
LATEST_NAME = f"latest{CKPT_EXT}"
MODEL_NAME = f"epoch%d{CKPT_EXT}"


def train(cfg: DictConfig):
    """Train model."""
    root_dir, out_dir = get_dirs()
    ckpt_dir = out_dir / CKPT_FOLDER
    ckpt_dir.mkdir(exist_ok=False)  # Error if exists to prevent model overwrite.

    transform = instantiate(cfg.train_transform)
    dataloader = create_kinetics400_dataloader(transform)
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer)

    checkpointer = Checkpointer(model, optimizer, scheduler, 0, dict(cfg))

    # TODO: Should load config from yaml log, not ckpt + do this in main() instead.
    if cfg.resume:
        checkpointer.load(root_dir / cfg.resume / CKPT_FOLDER / LATEST_NAME)
        cfg = OmegaConf.create(checkpointer.cfg)
        log.info(f"Resume train from epoch {checkpointer.epoch}.")
        log.debug(f"Resume Config:\n{cfg}")

    device = torch.device(cfg.device)
    model.to(device).train()

    with iter_pbar:
        log.info(f"Start training for {cfg.epochs} epochs.")
        epochtask = iter_pbar.add_task("Epoch", total=cfg.epochs, status="")

        for i in range(cfg.epochs):
            log.info(f"Epoch: {i + 1}/{cfg.epochs}")
            itertask = iter_pbar.add_task("Iteration", total=len(dataloader), status="")

            for batch in dataloader:
                batch = batch.to(device)

                _, loss, diag = model(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_pbar.update(
                    itertask,
                    advance=1,
                    status=f"Loss: {loss:.6g}, LR: {optimizer.param_groups[0]['lr']:.3g}",
                )

            scheduler.step()
            checkpointer.epoch += 1
            checkpointer.save(ckpt_dir / (MODEL_NAME % checkpointer.epoch))
            checkpointer.save(ckpt_dir / LATEST_NAME)

            iter_pbar.update(epochtask, advance=1)
