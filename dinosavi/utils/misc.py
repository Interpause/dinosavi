"""Misc utilities."""

import random

import numpy as np
import torch
import torch.nn as nn
from xxhash import xxh3_64_hexdigest as hexdigest

__all__ = ["perf_hack", "seed_rand", "seed_data", "hash_tensor", "hash_model"]


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
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    # Seed random here, why not?
    seed_rand()

    # Fix for running out of File Descriptors.
    torch.multiprocessing.set_sharing_strategy("file_system")


def hash_tensor(x: torch.Tensor) -> str:
    """Returns deterministic hexdigest of tensor."""
    # Ops used here are to minimize copies.
    is_float = torch.is_floating_point(x)
    # Using `x.numpy(force=True).data` is faster than `bytes(x.flatten().byte())`.
    x: np.ndarray = x.numpy(force=True)
    # At risk of collision, decrease precision due to floating point error.
    if is_float:
        x = np.interp(x, (x.min(), x.max()), (0, 255)).astype(np.uint8, order="C")
    # Standardize to contiguous array for deterministic hash.
    x = np.asarray(x, order="C")
    return hexdigest(x.data, seed=0)


def hash_model(m: nn.Module) -> str:
    """Returns deterministic hexdigest of model based on weights."""
    return hexdigest(
        "".join(f"{k}{hash_tensor(v)}" for k, v in sorted(m.state_dict().items())),
        seed=0,
    )
