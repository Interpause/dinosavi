"""Positional Encoding utilities."""

from functools import cache
from math import ceil
from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = ["gen_2d_pe", "interpolate_2d_pe"]


@cache
def gen_2d_pe(size: Tuple[int, int], type: str = "linear", sine_dim: int = 4):
    """Memoized function to create 2D positional encodings.

    Two types are supported: ``linear`` and ``sine``. For sinusoidal positional
    encodings, `sine_dim` can be specific for the wanted number of dims.

    ``linear`` is better if the model is intended to be invariant to resolution.
    Otherwise, ``sine`` supposedly is the better positional encoding type, though
    whether it can be effectively interpolated is another matter.

    Args:
        size (Tuple[int, int]): (H, W) of the positional encodings.
        type (str, optional): Type of positional encodings.
        sine_dim (int, optional): Number of dims for sinusoidal positional encodings.

    Returns:
        torch.Tensor: CHW positional encodings.
    """
    h, w = size
    if type == "linear":
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij"
        )
        return torch.stack((gx, gy))
    elif type == "sine":
        assert sine_dim % 2 == 0
        N = 10000  # Value by convention.
        D = ceil(sine_dim / 4) * 2
        freq = torch.pow(N, -torch.arange(0, D, 2) / D)

        x = torch.einsum("i,j->ij", freq, torch.arange(w)).unsqueeze(1)
        y = torch.einsum("i,j->ij", freq, torch.arange(h)).unsqueeze(-1)

        embed = torch.empty(D * 2, h, w)
        embed[:D:2] = x.sin()
        embed[1:D:2] = x.cos()
        embed[D : 2 * D : 2] = y.sin()
        embed[D + 1 : 2 * D : 2] = y.cos()
        return embed

    assert False, f"`{type}` not supported! Only `linear` and `sine` supported."


@cache
def interpolate_2d_pe(embed: torch.Tensor, size: Tuple[int, int]):
    """Memoized function to interpolate positional encodings to new size.

    Args:
        embed (torch.Tensor): CHW positional encodings.
        size (Tuple[int, int]): (H, W) new size.

    Returns:
        torch.Tensor: CHW positional encodings.
    """
    if embed.shape[-2:] == size:
        return embed
    # Yes, this seems messed up, but this is how ViT's resize encodings.
    new: torch.Tensor = F.interpolate(embed[None], size=size, mode="bicubic")
    return new[0]
