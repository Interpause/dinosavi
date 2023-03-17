"""Modified Spatial Broadcast Decoder (SBD) head that takes into account all slots."""

from typing import Tuple

import einops as E
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding2D

__all__ = ["CrossSlotSBDecoder"]


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
            torch.Tensor: BHWC broadcasted feature map.
        """
        h, w = self.size
        x = E.repeat(x, "b c -> b h w c", h=h, w=w)
        enc = self.pe(x)
        return torch.concat((enc, x), dim=3)


class CrossSlotSBDecoder(nn.Module):
    """Cross-Attention across Slots Spatial Broadcast Decoder."""

    # TODO: hid_dim & kernel_size can be customized per layer similar to torchvision.ops.MLP.
    # TODO: Should we add some form of normalization, activation & dropout? Other implementations don't seem to do it.
    def __init__(
        self,
        slot_dim: int,
        size: Tuple[int, int, int],
        depth: int = 3,
        kernel_size: int | Tuple[int, int] = 4,
        hid_dim: int = 256,
        pe_dim: int = 4,
    ):
        """Initialize Cross-Attention Spatial Broadcast Decoder.

        Args:
            slot_dim (int): Size of each slot.
            size (Tuple[int, int, int]): Size (H, W, C) of the feature map to broadcast to.
            depth (int, optional): Number of layers in the decoder.
            kernel_size (int | Tuple[int, int], optional): Kernel size of the decoder.
            hid_dim (int, optional): Hidden dimension of the decoder.
            pe_dim (int, optional): Size of the positional encodings.
        """
        super(CrossSlotSBDecoder, self).__init__()

        self.size = size
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.pad = self.kernel_size[0] // 2, self.kernel_size[1] // 2

        self.broadcast = SpatialBroadcast(size[:2], pe_dim)

        # TODO: Cross-attention implementation.
        # After cross-attention, should the dim be slot_dim or hid_dim or something else?
        self.attn = nn.Identity()

        dims = [slot_dim + pe_dim] + [hid_dim] * (depth - 1) + [size[-1]]
        self.conv = nn.Sequential(
            *(
                nn.Conv2d(i, o, kernel_size, padding=self.pad)
                for i, o in zip(dims, dims[1:])
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BSC slots.

        Returns:
            torch.Tensor: BHWC decoded feature map.
        """
        # BSC -> BC
        x = self.attn(x)
        # BC -> BHWC
        x = self.broadcast(x)
        return self.conv(x)
