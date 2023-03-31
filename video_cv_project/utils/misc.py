"""Misc utilities."""

import random

import numpy as np
import torch

__all__ = ["perf_hack", "seed_rand", "seed_data"]


def seed_rand(seed: int = 42):
    """Seed random generators.

    Args:
        seed (int, optional): Seed. Defaults to 42.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _seed_dl(worker_id):
    """See `seed_data` below."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_data(seed: int = 42):
    """Return requirements to seed dataloader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return dict(worker_init_fn=_seed_dl, generator=g)


def perf_hack(deterministic=False):
    """Pytorch performance hacks."""
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(deterministic)

    # Seed random here, why not?
    seed_rand()

    # Fix for running out of File Descriptors.
    torch.multiprocessing.set_sharing_strategy("file_system")
