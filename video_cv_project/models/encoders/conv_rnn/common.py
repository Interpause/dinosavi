"""Convolutional RNNs."""

from typing import List, Tuple, Type

import einops as E
import torch
import torch.nn as nn

__all__ = ["ConvRNNCellBase", "ConvRNN"]


class ConvRNNCellBase(nn.Module):
    """Base class for ConvRNN cells."""

    def __init__(self, in_dim: int, hid_dim: int, size: int | Tuple[int, int]):
        """Initialize ConvRNN cell.

        Args:
            in_dim (int): Number of input channels.
            hid_dim (int): Number of hidden channels.
            size (int | Tuple[int, int]): Kernel size.
        """
        super(ConvRNNCellBase, self).__init__()

        self.size = (size, size) if isinstance(size, int) else size
        self.pad = self.size[0] // 2, self.size[1] // 2  # Ensure same output size.

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BCHW input.
            h (torch.Tensor): BCHW previous hidden state.

        Returns:
            torch.Tensor: BCHW new hidden state.
        """
        raise NotImplementedError


class ConvRNN(nn.Module):
    """Convolutional RNN that can use various cells."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        size: int | Tuple[int, int],
        depth: int,
        cell: Type[ConvRNNCellBase] = ConvRNNCellBase,
    ):
        """Initialize ConvGRU.

        Args:
            in_dim (int): Number of input channels.
            hid_dim (int): Number of hidden channels.
            size (int | Tuple[int, int]): Kernel size.
            depth (int): Number of layers.
            cell (Type[ConvRNNCellBase], optional): Cell to use.
        """
        super(ConvRNN, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.depth = depth

        self.cells = nn.ModuleList(cell(in_dim, hid_dim, size) for _ in range(depth))

    def forward(
        self, x: torch.Tensor, h: List[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.

        Args:
            x (torch.Tensor): BTCHW input.
            h (List[torch.Tensor], optional): List of BCHW previous hidden states.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                y: BTCHW output.
                h: List of BCHW new hidden states.
        """
        if h is None:
            # NOTE: Following concepts from Slot Attention, learning a distribution
            # for the initial hidden state might be interesting.
            h = [
                torch.zeros(len(x), self.hid_dim, *x.shape[-2:]).to(x.device)
                for _ in self.cells
            ]

        layer_in = list(E.rearrange(x, "b t c h w -> t b c h w"))
        layers_out = []
        for cell, cur in zip(self.cells, h):
            # Feed intermediate hidden states to next layer.
            layer_in = [cur := cell(t, cur) for t in layer_in]
            layers_out.append(cur)

        return E.rearrange(layer_in, "t b c h w -> b t c h w"), layers_out  # type: ignore
