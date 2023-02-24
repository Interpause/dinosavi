"""Model training script."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.checkpointer import Checkpointer
from video_cv_project.data import create_kinetics_dataloader
from video_cv_project.utils import get_dirs, iter_pbar

log = logging.getLogger(__name__)

CKPT_FOLDER = "weights"
CKPT_EXT = ".ckpt"
LATEST_NAME = f"latest{CKPT_EXT}"
MODEL_NAME = f"epoch%d{CKPT_EXT}"
SAMPLE_INPUT = [1, 8, 147, 64, 64]

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

    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    epochs = cfg.train.epochs
    log_every = cfg.train.log_every

    log.info(f"Torch Device: {device}")
    log.info(f"Epochs: {epochs}")

    log.debug("Create Model.")
    model = instantiate(cfg.model)
    log.debug("Create Optimizer.")
    optimizer = instantiate(cfg.train.optimizer, model.parameters())
    log.debug("Create Scheduler.")
    scheduler = instantiate(cfg.train.scheduler, optimizer)
    log.debug("Create Train Dataloader.")
    dataloader = create_kinetics_dataloader(cfg)

    checkpointer = Checkpointer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=OmegaConf.to_container(cfg),  # type: ignore
    )

    # TODO: What should be preserved/overwritten when resuming train?
    # Or some system to specify that?
    if cfg.resume:
        resume_ckpt = root_dir / cfg.resume
        checkpointer.load(resume_ckpt)
        cfg = OmegaConf.create(checkpointer.cfg)
        log.info(f"Resume train from epoch {checkpointer.epoch}.")
        log.debug(f"Ckpt Config:\n{cfg}")

    model_summary = summary(model, SAMPLE_INPUT, verbose=0, col_width=20, device=device)
    model_summary.formatting.layer_name_width = 30
    log.info(f"Model Summary for Input Shape {SAMPLE_INPUT}:\n{model_summary}")

    model.to(device).train()

    with iter_pbar:
        log.info(f"Start training for {epochs} epochs.")
        epochtask = iter_pbar.add_task("Epoch", total=epochs, status="")

        for i in range(epochs):
            log.info(f"Epoch: {i+1}/{epochs}")
            itertask = iter_pbar.add_task("Iteration", total=len(dataloader), status="")

            for n, video in enumerate(dataloader, start=1):
                _, loss, debug = model(video.to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                status = f"Loss: {loss:.6g}, LR: {optimizer.param_groups[0]['lr']:.3g}"
                iter_pbar.update(itertask, advance=1, status=status)

                if n % log_every == 0 or n == len(dataloader):
                    log.info(f"Iteration: {n}/{len(dataloader)}, {status}")

            scheduler.step()
            checkpointer.epoch += 1
            checkpointer.save(ckpt_dir / (MODEL_NAME % checkpointer.epoch))
            checkpointer.save(ckpt_dir / LATEST_NAME)

            iter_pbar.update(epochtask, advance=1)
