"""Modified Spatial Broadcast Decoder (SBD) head that takes into account all slots."""

from typing import Tuple

import einops as E
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D

from .cnn import CNN

__all__ = ["CrossSlotDecoder", "SlotDecoder", "AlphaSlotDecoder"]


class SpatialBroadcast(nn.Module):
    """Performs spatial broadcast.

    Note, instead of using Linear Positional Encoding, we used Sinusoidal Positional
    Encoding. Not sure what the impacts are.
    """

    def __init__(self, size: Tuple[int, int], pe_dim: int = 4):
        """Initialize Spatial Broadcast.

        Args:
            size (Tuple[int, int]): Size (H, W) of the feature map to broadcast to.
            pe_dim (int, optional): Size of the positional encodings.
        """
        super(SpatialBroadcast, self).__init__()

        self.pe = PositionalEncoding2D(pe_dim)
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform spatial broadcast.

        Args:
            x (torch.Tensor): BC latent feature to broadcast.

        Returns:
            torch.Tensor: BCHW broadcasted feature map.
        """
        h, w = self.size
        x = E.repeat(x, "b c -> b h w c", h=h, w=w)
        enc = self.pe(x)
        x = torch.cat((enc, x), dim=3)
        # Official implementation uses Linear Positional Encoding.
        # gx, gy = torch.meshgrid(
        #     torch.linspace(-1, 1, w), torch.linspace(-1, 1, h), indexing="ij"
        # )
        # gx = E.repeat(gx, "h w -> b h w 1", b=len(x))
        # gy = E.repeat(gy, "h w -> b h w 1", b=len(x))
        # x = torch.cat((gx, gy, x), dim=3)
        return E.rearrange(x, "b h w c -> b c h w")


class SlotDecoder(nn.Module):
    """Spatial Broadcast Slot Decoder."""

    def __init__(
        self,
        slot_dim: int,
        size: Tuple[int, int, int],
        depth: int = 3,
        kernel_size: int | Tuple[int, int] = 5,
        hid_dim: int = 256,
        pe_dim: int = 4,
    ):
        """Initialize Spatial Broadcast Slot Decoder.

        Args:
            slot_dim (int): Size of each slot.
            size (Tuple[int, int, int]): Size (C, H, W) of the feature map to broadcast to.
            depth (int, optional): Number of layers in the decoder.
            kernel_size (int | Tuple[int, int], optional): Kernel size of the decoder.
            hid_dim (int, optional): Hidden dimension of the decoder.
            pe_dim (int, optional): Size of the positional encodings.
        """
        super(SlotDecoder, self).__init__()

        self.size = size
        self.kernel_size = kernel_size

        self.broadcast = SpatialBroadcast(size[-2:], pe_dim)

        dims = [slot_dim + pe_dim] + [hid_dim] * (depth - 1) + [size[0]]
        self.conv = CNN(dims, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BSC slots.

        Returns:
            torch.Tensor: BCHW decoded feature map.
        """
        raise NotImplementedError


class AlphaSlotDecoder(SlotDecoder):
    """Use "alpha" channel to blend decoded slots maps together."""

    def __init__(self, slot_dim, size, *args, **kwargs):
        """Refer to `video_cv_project.models.heads.SlotDecoder`."""
        size = (size[0] + 1, *size[-2:])
        super(AlphaSlotDecoder, self).__init__(slot_dim, size, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BSC slots.

        Returns:
            torch.Tensor: BCHW decoded feature map.
        """
        B = len(x)
        x = E.rearrange(x, "b s c -> (b s) c")
        x = self.broadcast(x)
        x = self.conv(x)
        x = E.rearrange(x, "(b s) c h w -> b c h w s", b=B)
        # Transparencies for each slot S is the last channel.
        alpha = F.softmax(x[:, -1:], dim=-1)
        return E.reduce(x[:, :-1] * alpha, "b c h w s -> b c h w", "sum")


class CrossSlotDecoder(SlotDecoder):
    """Cross-Attention across Slots Spatial Broadcast Decoder."""

    def __init__(self, *args, **kwargs):
        """Refer to `video_cv_project.models.heads.SlotDecoder`."""
        super(CrossSlotDecoder, self).__init__(*args, **kwargs)

        # TODO: Cross-attention implementation.
        # After cross-attention, should the dim be slot_dim or hid_dim or something else?
        self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BSC slots.

        Returns:
            torch.Tensor: BCHW decoded feature map.
        """
        # BSC -> BC
        x = self.attn(x)
        # BC -> BCHW
        x = self.broadcast(x)
        return self.conv(x)
