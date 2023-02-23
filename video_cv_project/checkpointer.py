"""Manages loading & saving model state."""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

log = logging.getLogger(__name__)


@dataclass
class Checkpointer:
    """Manage state.

    All state is optional because some parts may not be needed. For example, the
    ``optimizer`` isn't needed for inference. The `reload` method also allows the
    state for parts to be loaded later without needing to load from disk again.
    This is used in inference mode to construct the model from the checkpoint rather
    than relying on the user specifying the exact same model config.
    """

    model: nn.Module | None = None
    optimizer: Optimizer | None = None
    scheduler: _LRScheduler | None = None
    epoch: int = 0
    cfg: dict = field(default_factory=dict)
    quiet: bool = False

    prev_ckpt: dict = field(default_factory=dict)

    def save(self, path):
        """Save state."""
        if not self.quiet:
            log.info(f"Saving checkpoint: {path}")
        ckpt = dict(
            model=self.model.state_dict() if self.model else None,
            optimizer=self.optimizer.state_dict() if self.optimizer else None,
            lr_scheduler=self.scheduler.state_dict() if self.scheduler else None,
            epoch=self.epoch,
            cfg=self.cfg,
        )
        torch.save(ckpt)
        self.prev_ckpt = ckpt

    def _load(self, ckpt):
        """Actually load state."""
        if self.model and ckpt["model"]:
            self.model.load_state_dict(ckpt["model"])
        if self.optimizer and ckpt["optimizer"]:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and ckpt["lr_scheduler"]:
            self.scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.epoch = ckpt["epoch"]
        self.cfg.clear()
        self.cfg.update(ckpt["cfg"])

    def load(self, path):
        """Load state in place."""
        if not self.quiet:
            log.info(f"Loading checkpoint: {path}")
        ckpt = torch.load(path)
        self._load(ckpt)
        self.prev_ckpt = ckpt

    def reload(self):
        """Reload previous checkpoint."""
        self._load(self.prev_ckpt)
