"""Model training script."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from dinosavi.cfg import BEST_DEVICE
from dinosavi.data import create_kinetics_dataloader
from dinosavi.engine import Checkpointer, ExpRunner
from dinosavi.utils import get_dirs, get_model_summary, tb_hparams

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
    dryrun = cfg.train.dryrun
    clip_norm = cfg.train.clip_gradnorm

    log.info(f"Torch Device: {device}")
    log.info(f"Epochs: {epochs}")

    if dryrun:
        log.warning("Dry run mode! Model will be not used.")

    log.debug("Create Model.")
    log.info(f"Model Config:\n{OmegaConf.to_object(cfg.model)}")
    model = instantiate(cfg.model, _convert_="all")
    summary = get_model_summary(model, device=device)
    log.info(f"Model Summary for Input Shape {summary.input_size[0]}:\n{summary}")

    log.debug("Create Train Dataloader.")
    dataloader = create_kinetics_dataloader(cfg)
    with open_dict(cfg):
        cfg.total_steps = len(dataloader) * epochs
    log.info(f"Total Steps: {cfg.total_steps}")

    log.debug("Create Optimizer.")
    optimizer = instantiate(cfg.train.optimizer, model.parameters(), _convert_="all")
    log.info(f"Optimizer:\n{optimizer.state_dict()}")

    log.debug("Create Scheduler.")
    if hasattr(cfg.train.scheduler, "milestones"):
        cfg.train.scheduler.milestones = [
            int(cfg.total_steps * m) for m in cfg.train.scheduler.milestones
        ]
    scheduler = instantiate(cfg.train.scheduler, optimizer, _convert_="all")
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
        log.info(f"Ckpt Config:\n{OmegaConf.to_object(old_cfg)}")

    runner = ExpRunner(
        dataloader,
        epochs,
        logger=log,
        log_every=log_every,
        save_func=lambda i, n: checkpointer.save(
            ckpt_dir / (MODEL_NAME % (checkpointer.epoch, n))
        ),
        save_every=save_every,
    )
    # model.is_trace = True
    # runner.tbwriter.add_graph(model.model, torch.zeros(1, 3, 224, 224).to(device))
    # model.is_trace = False
    # runner.tbwriter.add_hparams(tb_hparams(cfg), {})

    model.to(device).train()

    ini_epoch = checkpointer.epoch
    log.info(f"Start training for {epochs} epochs.")
    for i, n, data in runner:
        if dryrun:
            continue

        # `default_collate` turns tuples into lists for some reason.
        data = data if isinstance(data, list) or isinstance(data, tuple) else (data,)
        data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in data)

        loss, debug = model(*data)

        runner.update(lr=float(scheduler.get_last_lr()[0]), **debug)
        checkpointer.epoch = ini_epoch + i

        optimizer.zero_grad()
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
