"""Model training script."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.data import create_kinetics_dataloader
from video_cv_project.engine import Checkpointer, Trainer
from video_cv_project.utils import get_dirs, perf_hack

log = logging.getLogger(__name__)

CKPT_FOLDER = "weights"
MODEL_NAME = f"epoch%d_%d.ckpt"
SAMPLE_INPUT = [1, 8, 147, 64, 64]

# TODO: Do smth with debug data like visualize.
# TODO: More metadata about input mode like patch size, number of patches, shape, etc.
# TODO: Distributed training wait who am i kidding.


def train(cfg: DictConfig):
    """Train model."""
    root_dir, out_dir = get_dirs()
    ckpt_dir = out_dir / CKPT_FOLDER
    ckpt_dir.mkdir(exist_ok=False)  # Error if exists to prevent model overwrite.

    perf_hack()
    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    epochs = cfg.train.epochs
    log_every = cfg.train.log_every
    save_every = cfg.train.save_every

    log.info(f"Torch Device: {device}")
    log.info(f"Epochs: {epochs}")

    log.debug("Create Model.")
    model = instantiate(cfg.model)
    model_summary = summary(model, SAMPLE_INPUT, verbose=0, col_width=20, device=device)
    model_summary.formatting.layer_name_width = 30
    log.info(f"Model Summary for Input Shape {SAMPLE_INPUT}:\n{model_summary}")

    log.debug("Create Train Dataloader.")
    dataloader = create_kinetics_dataloader(cfg)

    log.debug("Create Optimizer.")
    optimizer = instantiate(cfg.train.optimizer, model.parameters())
    log.info(f"Optimizer:\n{optimizer}")

    log.debug("Create Scheduler.")
    if hasattr(cfg.train.scheduler, "milestones"):
        total = len(dataloader) * epochs
        cfg.train.scheduler.milestones = [
            int(total * m) for m in cfg.train.scheduler.milestones
        ]
    scheduler = instantiate(cfg.train.scheduler, optimizer)
    log.info(f"Scheduler:\n{scheduler.state_dict()}")

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

    model.to(device).train()

    trainer = Trainer(
        dataloader,
        epochs,
        logger=log,
        log_every=log_every,
        save_func=lambda i, n: checkpointer.save(
            ckpt_dir / (MODEL_NAME % (checkpointer.epoch, n))
        ),
        save_every=save_every,
    )

    ini_epoch = checkpointer.epoch
    log.info(f"Start training for {epochs} epochs.")
    for i, n, video in trainer:
        _, loss, debug = model(video.to(device))

        trainer.update(loss=float(loss), lr=float(scheduler.get_last_lr()[0]))
        checkpointer.epoch = ini_epoch + i

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
