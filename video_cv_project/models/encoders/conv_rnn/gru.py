"""Convolutional GRU."""

from typing import List, Tuple

import einops as E
import torch
import torch.nn as nn

from video_cv_project.cfg import BEST_DEVICE, RGB
from video_cv_project.utils import infer_outdim

from .common import ConvRNN, ConvRNNCellBase

__all__ = ["ConvGRUCell", "ConvGRU", "EncoderWithGRU"]


class ConvGRUCell(ConvRNNCellBase):
    """ConvGRU cell."""

    def __init__(self, in_dim: int, hid_dim: int, size: int | Tuple[int, int]):
        """Initialize ConvGRUCell.

        Args:
            in_dim (int): Number of input channels.
            hid_dim (int): Number of hidden channels.
            size (int | Tuple[int, int]): Kernel size.
        """
        super(ConvGRUCell, self).__init__(in_dim, hid_dim, size)

        self.conv_gates = nn.Conv2d(
            in_dim + hid_dim, 2 * hid_dim, self.size, padding="same"
        )
        self.conv_candidate = nn.Conv2d(
            in_dim + hid_dim, hid_dim, self.size, padding="same"
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """Forward pass.

        Args:
            x (torch.Tensor): BCHW input.
            h (torch.Tensor): BCHW previous hidden state.

        Returns:
            torch.Tensor: BCHW new hidden state.
        """
        cat = torch.cat([x, h], dim=1)
        y: torch.Tensor = self.conv_gates(cat).sigmoid()
        reset, update = E.rearrange(y, "b (cat c) h w -> cat b c h w", cat=2)
        cat = torch.cat([x, reset * h], dim=1)
        y = self.conv_candidate(cat).tanh()
        h = (1 - update) * h + update * y
        return h


class ConvGRU(ConvRNN):
    """Convolutional GRU.

    This only exists for convenience since configuring ConvRNN to use ConvGRUCell
    isn't easy with Hydra.
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        size: int | Tuple[int, int],
        depth: int,
    ):
        """Initialize ConvGRU.

        Args:
            in_dim (int): Number of input channels.
            hid_dim (int): Number of hidden channels.
            size (int | Tuple[int, int]): Kernel size.
            depth (int): Number of layers.
        """
        super(ConvGRU, self).__init__(in_dim, hid_dim, size, depth, cell=ConvGRUCell)


class EncoderWithGRU(nn.Module):
    """Adds GRU-based memory layer to encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        hid_dim: int = 256,
        size: int | Tuple[int, int] = 3,
        depth: int = 1,
        reset: bool = True,
        device: torch.device = BEST_DEVICE,
    ):
        """Initialize encoder with RNN.

        Args:
            encoder (torch.nn.Module): Encoder to wrap.
            hid_dim (int, optional): Number of hidden channels. Defaults to 256.
            size (int | Tuple[int, int], optional): Kernel size. Defaults to 3.
            depth (int, optional): Number of layers. Defaults to 1.
            reset (bool, optional): Whether to reset hidden state when input shape changes.
            device (torch.device, optional): Device to use. Defaults to BEST_DEVICE.
        """
        super(EncoderWithGRU, self).__init__()

        self.encoder = encoder

        # Arbitrary BTCHW input is fine.
        sz = 256
        enc_dim = infer_outdim(encoder, (1, 1, RGB, sz, sz), device=device)
        self.rnn = ConvGRU(enc_dim[2], hid_dim, size, depth)

        self.hidden: List[torch.Tensor] | None = None
        self.reset_hidden = reset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BTCHW input video.

        Returns:
            torch.Tensor: BTCHW output temporal memory maps.
        """
        x = self.encoder(x)
        if self.training or (
            self.hidden is not None
            and self.reset_hidden
            and x[:, 0, 0].shape != self.hidden[0][:, 0].shape
        ):
            # Both training & evaluation code keeps the shape constant, but below
            # serves as check anyways.
            # print("EncoderWithGRU: Resetting hidden state.")
            self.reset()
        y, self.hidden = self.rnn(x, self.hidden)
        return y

    def reset(self):
        """Reset hidden state."""
        self.hidden = None
