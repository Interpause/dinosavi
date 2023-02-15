"""TODO: Add module docstring."""

from typing import Callable, List, TypeVar

Self = TypeVar("Self")

import torch
from torch import nn

__all__ = ["find_layers", "delete_layers", "override_forward"]


def find_layers(model, names: List[str]):
    """Find and return layers by name from model.

    Args:
        model (Any): Model to search.
        names (List[str]): Names of layers to find.
    """
    return [getattr(model, l) for l in names if hasattr(model, l)]


def delete_layers(model, names: List[str]):
    """Convert layers to `torch.nn.Identity` by name in model in place.

    Args:
        model (Any): Model to modify.
        names (List[str]): Names of layers to delete.
    """
    for l in names:
        if hasattr(model, l):
            setattr(model, l, nn.Identity())


def override_forward(model, func: Callable[[Self, torch.Tensor], torch.Tensor]):
    """Override forward pass of model in place.

    Args:
        model (Any): Model to modify.
        func (Callable[[Self, torch.Tensor], torch.Tensor]): New forward pass.
    """
    bound_method = func.__get__(model, model.__class__)
    setattr(model, "forward", bound_method)
