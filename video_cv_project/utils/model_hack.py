"""Utility functions for modifying models."""

import random
from typing import Callable, Sequence, TypeVar

import numpy as np
import torch
import torch.nn as nn

from video_cv_project.cfg import BEST_DEVICE

__all__ = [
    "find_layers",
    "delete_layers",
    "override_forward",
    "infer_outdim",
    "perf_hack",
    "seed_rand",
]

Self = TypeVar("Self")


def seed_rand(seed: int = 42):
    """Seed random generators.

    Args:
        seed (int, optional): Seed. Defaults to 42.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def perf_hack():
    """Pytorch performance hacks."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Seed random here, why not?
    seed_rand()

    # Fix for running out of File Descriptors.
    torch.multiprocessing.set_sharing_strategy("file_system")


def find_layers(model: nn.Module, names: Sequence[str]):
    """Find and return layers by name from model.

    Args:
        model (torch.nn.Module): Model to search.
        names (Sequence[str]): Names of layers to find.

    Returns:
        list: Layers found.
    """
    return [getattr(model, l) for l in names if hasattr(model, l)]


def delete_layers(model: nn.Module, names: Sequence[str]):
    """Convert layers to `torch.nn.Identity` by name in model in place.

    Args:
        model (torch.nn.Module): Model to modify.
        names (Sequence[str]): Names of layers to delete.
    """
    for l in names:
        if hasattr(model, l):
            setattr(model, l, nn.Identity())


def override_forward(
    model: nn.Module, func: Callable[[Self, torch.Tensor], torch.Tensor]
):
    """Override forward pass of model in place.

    Args:
        model (torch.nn.Module): Model to modify.
        func (Callable[[Self, torch.Tensor], torch.Tensor]): New forward pass.
    """
    bound_method = func.__get__(model, model.__class__)
    setattr(model, "forward", bound_method)


@torch.inference_mode()
def infer_outdim(
    model: nn.Module, indim: Sequence[int], device=BEST_DEVICE
) -> torch.Size:
    """Infer output dimension of model.

    Args:
        model (torch.nn.Module): Model to check.
        indim (Sequence[int]): Input dimension.
        device (torch.device, optional): Device to use. Defaults to best available.

    Returns:
        torch.Size: Output dimension.
    """
    x = torch.zeros(indim).to(device)
    model.to(device).eval()
    return model(x).shape
