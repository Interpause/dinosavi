"""Model training script."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from video_cv_project.cfg import BEST_DEVICE
from video_cv_project.data import create_kinetics_dataloader
from video_cv_project.engine import Checkpointer, Trainer
from video_cv_project.utils import get_dirs, get_model_summary

log = logging.getLogger(__name__)

CKPT_FOLDER = "weights"
MODEL_NAME = f"epoch%d_%d.ckpt"


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
    save_every = cfg.train.save_every

    log.info(f"Torch Device: {device}")
    log.info(f"Epochs: {epochs}")

    log.debug("Create Model.")
    model = instantiate(cfg.model)
    summary = get_model_summary(model, device=device)
    log.info(f"Model Summary for Input Shape {summary.input_size}:\n{summary}")

    log.debug("Create Train Dataloader.")
    dataloader = create_kinetics_dataloader(cfg)
    with open_dict(cfg):
        cfg.total_steps = len(dataloader) * epochs
    log.info(f"Total Steps: {cfg.total_steps}")

    log.debug("Create Optimizer.")
    optimizer = instantiate(cfg.train.optimizer, model.parameters())
    log.info(f"Optimizer:\n{optimizer.state_dict()}")

    log.debug("Create Scheduler.")
    if hasattr(cfg.train.scheduler, "milestones"):
        cfg.train.scheduler.milestones = [
            int(cfg.total_steps * m) for m in cfg.train.scheduler.milestones
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

        # TODO: Investigate impact of throwing away optimizer and scheduler state.
        checkpointer.load(resume_ckpt, ignore_keys=["optimizer", "lr_scheduler"])
        old_cfg = OmegaConf.create(checkpointer.cfg)
        log.info(f"Resume train from epoch {checkpointer.epoch}.")
        log.debug(f"Ckpt Config:\n{old_cfg}")

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
    with torch.autograd.set_detect_anomaly(True):
        for i, n, data in trainer:
            video, target = data if isinstance(data, tuple) else (data, None)
            video = video.to(device)

            if target is None:
                loss, debug = model(video)
            else:
                target = target.to(device)
                loss, debug = model(video, target)

            trainer.update(
                loss=float(loss), lr=float(scheduler.get_last_lr()[0]), **debug
            )
            checkpointer.epoch = ini_epoch + i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
