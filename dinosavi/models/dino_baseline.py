"""DINOSAUR-inspired baseline model."""

from typing import Tuple

import einops as E
import torch
import torch.nn as nn

from dinosavi.models.heads import SlotDecoder
from dinosavi.utils import mse_loss, tb_viz_slots

from .slot_model import SlotModel

__all__ = ["DINOSAUR"]


class DINOSAUR(nn.Module):
    """SlotModel trainer."""

    def __init__(
        self,
        model: SlotModel,
        decoder: SlotDecoder,
        num_slots: Tuple[int, int] | int = 5,
        num_iters: int = 3,
    ):
        """Initialize DINOSAUR.

        Args:
            model (SlotModel): Slot model.
            decoder (SlotDecoder): Decodes slot to features.
            num_slots (Tuple[int, int] | int, optional): Number of slots to create. Defaults to 7.
                If tuple, should be (background, foreground).
            num_iters (int, optional): Number of iterations for slots to bind. Defaults to 3.
        """
        super(DINOSAUR, self).__init__()

        assert model.use_bgfg_groups == False

        self.model = model
        self.decoder = decoder

        self.layernorm = nn.LayerNorm(model.feat_dim)
        self.num_slots = num_slots
        self.num_iters = num_iters

        self.is_trace = False

    def _pred_slots(self, pats: torch.Tensor):
        """Predict slots for each frame."""
        T = len(pats)
        pats = E.rearrange(pats, "t b c h w -> (t b) c h w")
        slots_t, attn_t = self.model(pats, None, self.num_slots, self.num_iters)
        slots_t = E.rearrange(slots_t, "(t b) s c -> t b s c", t=T)
        attn_t = E.rearrange(attn_t, "(t b) s n -> t b s n", t=T)
        return slots_t, attn_t

    def _pred_feats(self, slots: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Predict features from slots."""
        slots = E.rearrange(slots, "t b s c -> (t b) s c")
        return self.decoder(slots, sz)

    def get_masks(self, slots: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Decode slots to get decoder masks.

        Args:
            slots (torch.Tensor): BSC slots.
            sz (Tuple[int, int]): Size (H, W) of mask.

        Returns:
            torch.Tensor: BSHW masks.
        """
        return self.decoder.get_masks(slots, sz)

    def forward(
        self, vid: torch.Tensor, cls_attns: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            vid (torch.Tensor): BTCHW image patches.
            cls_attns (torch.Tensor, optional): BTNHW CLS token attention weights.

        Returns:
            Tuple[torch.Tensor, dict]: Loss, metrics.
        """
        pats_t = E.rearrange(vid, "b t c h w -> t b c h w")
        # Shuffle frames.
        pats_t = pats_t[torch.randperm(len(pats_t))]

        slots_t, attn_t = self._pred_slots(pats_t)

        T = len(slots_t)
        h, w = pats_t.shape[-2:]
        x = self._pred_feats(slots_t, (h, w))
        # Flatten every pixel in batch together for loss.
        x = E.rearrange(x, "(t b) c h w -> 1 t (b h w) c", t=T)
        x = self.layernorm(x)  # Uniformity with ViT.
        y = E.rearrange(pats_t, "t b c h w -> 1 t (b h w) c")

        loss, debug = mse_loss(x, y)
        # NOTE: When logs are flushed to disk, this takes significant time to write.
        # debug["slot_attn"] = tb_viz_slots(pats_t[-1, -1], attn_t[-1, -1])
        if self.is_trace:
            return loss  # type: ignore
        return loss, debug
