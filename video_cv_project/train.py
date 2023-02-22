"""TODO: Add module docstring."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from video_cv_project.checkpointer import Checkpointer
from video_cv_project.data import create_kinetics400_dataloader
from video_cv_project.utils import get_dirs, iter_pbar

log = logging.getLogger(__name__)

CKPT_FOLDER = "weights"
CKPT_EXT = ".ckpt"
LATEST_NAME = f"latest{CKPT_EXT}"
MODEL_NAME = f"epoch%d{CKPT_EXT}"
SAMPLE_INPUT = [1, 8, 147, 64, 64]
LOG_EVERY = 10

# TODO: Do smth with debug data like visualize.
# TODO: Tensorboard. Other per iteration logging (maybe epoch logging too?) should
# be handled by custom iterator.
# TODO: More metadata about input mode like patch size, number of patches, shape, etc.
# TODO: Distributed training wait who am i kidding.


def train(cfg: DictConfig):
    """Train model."""
    root_dir, out_dir = get_dirs()
    ckpt_dir = out_dir / CKPT_FOLDER
    ckpt_dir.mkdir(exist_ok=False)  # Error if exists to prevent model overwrite.

    log.debug("Create train pipeline.")
    transform = instantiate(cfg.transform.pipeline)
    log.info(f"Pipeline:\n{transform}")
    log.debug("Create train dataloader.")
    dataloader = create_kinetics400_dataloader(transform)
    log.debug("Create model.")
    model = instantiate(cfg.model)
    log.debug("Create optimizer.")
    optimizer = instantiate(cfg.optimizer, model.parameters())
    log.debug("Create scheduler.")
    scheduler = instantiate(cfg.scheduler, optimizer)

    checkpointer = Checkpointer(model, optimizer, scheduler, 0, dict(cfg))

    # TODO: Should load config from yaml log, not ckpt + do this in main() instead.
    if cfg.resume:
        resume_ckpt = root_dir / cfg.resume / CKPT_FOLDER / LATEST_NAME
        checkpointer.load(resume_ckpt)
        cfg = OmegaConf.create(checkpointer.cfg)
        log.info(f"Resume train from epoch {checkpointer.epoch}.")
        log.debug(f"Resume Config:\n{cfg}")

    model_summary = summary(model, SAMPLE_INPUT, verbose=0, col_width=20)
    model_summary.formatting.layer_name_width = 30
    log.info(f"Model Summary for Input Shape {SAMPLE_INPUT}:\n{model_summary}")

    device = torch.device(cfg.device)
    model.to(device).train()

    with iter_pbar:
        log.info(f"Start training for {cfg.epochs} epochs.")
        epochtask = iter_pbar.add_task("Epoch", total=cfg.epochs, status="")

        for i in range(cfg.epochs):
            log.info(f"Epoch: {i+1}/{cfg.epochs}")
            itertask = iter_pbar.add_task("Iteration", total=len(dataloader), status="")

            for n, batch in enumerate(dataloader, start=1):
                batch = batch.to(device)

                _, loss, debug = model(batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                status = f"Loss: {loss:.6g}, LR: {optimizer.param_groups[0]['lr']:.3g}"
                iter_pbar.update(itertask, advance=1, status=status)

                if n % LOG_EVERY == 0 or n == len(dataloader):
                    log.info(f"Iteration: {n}/{len(dataloader)}, {status}")

            scheduler.step()
            checkpointer.epoch += 1
            checkpointer.save(ckpt_dir / (MODEL_NAME % checkpointer.epoch))
            checkpointer.save(ckpt_dir / LATEST_NAME)

            iter_pbar.update(epochtask, advance=1)
