"""Slot Attention implementation in PyTorch based on official TensorFlow implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SlotAttention"]

# TODO: It is possible to use multi-head attention.
# See: https://github.com/google-research/slot-attention-video/blob/main/savi/modules/attention.py


class SlotAttention(nn.Module):
    """Slot Attention Module."""

    def __init__(
        self, in_feats: int, num_slots: int, num_iters: int, slot_dim: int, hid_dim: int
    ):
        """Initialize Slot Attention Module.

        Args:
            in_feats (int): Number of input features.
            num_slots (int): Number of slots.
            num_iters (int): Number of iterations for slots to bind.
            slot_dim (int): Size of representations in each slot.
            hid_dim (int): Size of hidden layer in MLP.
        """
        super(SlotAttention, self).__init__()

        self.num_slots = num_slots
        self.num_iters = num_iters
        self.in_feats = in_feats
        self.slot_dim = slot_dim
        self.hid_dim = hid_dim

        self.norm_in = nn.LayerNorm(in_feats)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Parameters for slot initialization distribution.
        self.slots_mu = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, 1, slot_dim))
        )
        self.slots_logvar = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, 1, slot_dim))
        )

        # Slots act as query to find representable features.
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        # Input features are both the key and value.
        self.project_kv = nn.Linear(in_feats, 2 * slot_dim, bias=False)

        # Slot updaters.
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, slot_dim)
        )

    def _init_slots(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize slots.

        This function can be overwritten for a more intelligent initialization based
        on the input features.

        Args:
            x (torch.Tensor): BNC input features.

        Returns:
            torch.Tensor: BSC slots.
        """
        return self.slots_mu + self.slots_logvar.exp() * torch.randn(
            len(x), self.num_slots, self.slot_dim, device=x.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BNC input features.

        Returns:
            torch.Tensor: BSC slots.
        """
        x = self.norm_in(x)
        k, v = self.project_kv(x).split(self.slot_dim, dim=2)

        slots = self._init_slots(x)

        # Multiple rounds of attention for slots to bind.
        for _ in range(self.num_iters):
            q = self.project_q(self.norm_slots(slots))
            attn = F.scaled_dot_product_attention(q, k, v)
            slots = self.gru(attn.flatten(0, 1), slots.flatten(0, 1)).view_as(slots)
            slots += self.mlp(self.norm_mlp(slots))  # Residual.

        return slots
