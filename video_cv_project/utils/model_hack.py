"""TODO: Add module docstring."""

from typing import Callable, List, Tuple, TypeVar

import torch
import torch.nn as nn

from video_cv_project.cfg import BEST_DEVICE

__all__ = ["find_layers", "delete_layers", "override_forward", "infer_outdim"]

Self = TypeVar("Self")


def find_layers(model: nn.Module, names: List[str]):
    """Find and return layers by name from model.

    Args:
        model (torch.nn.Module): Model to search.
        names (List[str]): Names of layers to find.

    Returns:
        list: Layers found.
    """
    return [getattr(model, l) for l in names if hasattr(model, l)]


def delete_layers(model: nn.Module, names: List[str]):
    """Convert layers to `torch.nn.Identity` by name in model in place.

    Args:
        model (torch.nn.Module): Model to modify.
        names (List[str]): Names of layers to delete.
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


def infer_outdim(model: nn.Module, indim: Tuple[int], device=BEST_DEVICE):
    """Infer output dimension of model.

    Args:
        model (torch.nn.Module): Model to check.
        indim (Tuple[int]): Input dimension.
        device (torch.device, optional): Device to use. Defaults to best available.

    Returns:
        torch.Size: Output dimension.
    """
    x = torch.zeros(indim).to(device)
    model.to(device).eval()

    with torch.inference_mode():
        y: torch.Tensor = model(x)

    return y.shape
