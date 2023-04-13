"""Modified Spatial Broadcast Decoder (SBD) head that takes into account all slots."""

from typing import Tuple

import einops as E
import torch
import torch.nn as nn
import torch.nn.functional as F

from video_cv_project.utils import gen_2d_pe, interpolate_2d_pe

from .cnn import CNN

__all__ = ["SlotDecoder", "AlphaSlotDecoder"]


class SpatialBroadcast(nn.Module):
    """Performs spatial broadcast."""

    def __init__(
        self,
        pe_type: str = "linear",
        pe_size: Tuple[int, int] = (14, 14),
        pe_dim: int = 4,
    ):
        """Initialize Spatial Broadcast.

        Args:
            pe_type (str, optional): Type of positional encodings to use.
            pe_size (Tuple[int, int], optional): Original positional encoding size. Ignored if `pe_type` is ``linear``.
            pe_dim (int, optional): Size of the positional encodings. Ignored if `pe_type` is ``linear``.
        """
        super(SpatialBroadcast, self).__init__()

        pe_size = tuple(pe_size)  # Config system might pass unhashable list.
        if pe_type == "learnt":
            self.pe = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(pe_dim, *pe_size))
            )
        else:
            # Helps to throw error if ``pe_size`` is changed after training.
            self.register_buffer(
                "pe", gen_2d_pe(pe_size, type=pe_type, sine_dim=pe_dim)
            )

        self.pe_type = pe_type
        self.pe_size = pe_size
        self.pe_dim = 2 if pe_type == "linear" else pe_dim

    def forward(self, x: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Perform spatial broadcast.

        Args:
            x (torch.Tensor): BC latent feature to broadcast.
            sz (Tuple[int, int]): Size (H, W) of the feature map to broadcast to.

        Returns:
            torch.Tensor: BCHW broadcasted feature map.
        """
        h, w = sz
        x = E.repeat(x, "b c -> b h w c", h=h, w=w)
        # Interpolate positional encodings if needed.
        pe = gen_2d_pe((h, w)).type_as(x) if self.pe_type == "linear" else self.pe
        if self.pe_type != "linear":
            pe = interpolate_2d_pe(pe, (h, w))
        pe = E.repeat(pe, "c h w -> b h w c", b=len(x))
        x = torch.cat((pe, x), dim=3)
        return E.rearrange(x, "b h w c -> b c h w")


class SlotDecoder(nn.Module):
    """Spatial Broadcast Slot Decoder."""

    def __init__(
        self,
        slot_dim: int,
        out_dim: int,
        depth: int = 3,
        kernel_size: int | Tuple[int, int] = 5,
        hid_dim: int = 256,
        pe_type: str = "linear",
        pe_size: Tuple[int, int] = (14, 14),
        pe_dim: int = 4,
    ):
        """Initialize Spatial Broadcast Slot Decoder.

        Args:
            slot_dim (int): Size of each slot.
            out_dim (int): Channel dimension of feature map to broadcast to.
            depth (int, optional): Number of layers in the decoder.
            kernel_size (int | Tuple[int, int], optional): Kernel size of the decoder.
            hid_dim (int, optional): Hidden dimension of the decoder.
            pe_type (str, optional): Type of positional encodings to use.
            pe_size (Tuple[int, int], optional): Original positional encoding size. Ignored if `pe_type` is ``linear``.
            pe_dim (int, optional): Size of the positional encodings. Ignored if `pe_type` is ``linear``.
        """
        super(SlotDecoder, self).__init__()

        self.out_dim = out_dim
        self.kernel_size = kernel_size

        self.broadcast = SpatialBroadcast(pe_type, pe_size, pe_dim)

        dims = [slot_dim + self.broadcast.pe_dim] + [hid_dim] * (depth - 1) + [out_dim]
        self.conv = CNN(dims, kernel_size)

    def forward(self, x: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BSC slots.
            sz (Tuple[int, int]): Size (H, W) of the feature map to broadcast to.

        Returns:
            torch.Tensor: BCHW decoded feature map.
        """
        raise NotImplementedError


class AlphaSlotDecoder(SlotDecoder):
    """Uses convolutions and "alpha" channel to blend decoded slots maps together."""

    def __init__(self, slot_dim, out_dim, *args, **kwargs):
        """Refer to `video_cv_project.models.heads.SlotDecoder`."""
        super(AlphaSlotDecoder, self).__init__(slot_dim, out_dim + 1, *args, **kwargs)

    def get_alpha_masks(self, x: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Inference route.

        Args:
            x (torch.Tensor): BSC slots.
            sz (Tuple[int, int]): Size (H, W) of the feature map to broadcast to.

        Returns:
            torch.Tensor: BSHW alpha masks for each slot.
        """
        return self.__call__(x, sz, ret_alpha=True)

    def forward(
        self, x: torch.Tensor, sz: Tuple[int, int], ret_alpha: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BSC slots.
            sz (Tuple[int, int]): Size (H, W) of the feature map to broadcast to.
            ret_alpha (bool, optional): Return BSHW alpha masks instead of features.

        Returns:
            torch.Tensor: BCHW decoded feature map.
        """
        B = len(x)
        x = E.rearrange(x, "b s c -> (b s) c")
        x = self.broadcast(x, sz)
        x = self.conv(x)
        x = E.rearrange(x, "(b s) c h w -> b c h w s", b=B)
        # Transparencies for each slot S is the last channel.
        alpha = F.softmax(x[:, -1:], dim=-1)
        if ret_alpha:
            return E.rearrange(alpha, "b 1 h w s -> b s h w")
        return E.reduce(x[:, :-1] * alpha, "b c h w s -> b c h w", "sum")


# TODO: Transformer-based decoder from DINOSAUR.
