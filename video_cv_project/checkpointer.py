"""TODO: Add module docstring."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class Checkpointer:
    """Manage training state."""

    model: nn.Module
    optimizer: Optimizer
    scheduler: _LRScheduler
    epoch: int
    cfg: dict

    def save(self, path):
        """Save state."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "cfg": self.cfg,
            },
            path,
        )

    def load(self, path):
        """Load state in place."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epoch = checkpoint["epoch"]
        self.cfg.clear()
        self.cfg.update(checkpoint["cfg"])
