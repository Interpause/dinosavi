"""Slot Attention implementation in PyTorch based on official TensorFlow implementation."""

import einops as E
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class SlotAttention(nn.Module):
    """Slot Attention Module."""

    def __init__(
        self, in_feats: int, num_slots: int, num_iters: int, slot_dim: int, hid_dim: int
    ):
        """Initialize Slot Attention Module.

        Args:
            in_feats (int): Number of input features.
            num_slots (int): Number of slots.
            num_iters (int): Number of iterations for slots to lock on.
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
        self.project_k = nn.Linear(in_feats, slot_dim, bias=False)
        self.project_v = nn.Linear(in_feats, slot_dim, bias=False)

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
        k = self.project_k(x)
        v = self.project_v(x)

        slots = self._init_slots(x)

        # Multiple rounds of attention for slots to lock on.
        for _ in range(self.num_iters):
            prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)
            logits = E.einsum(k, q, "b n c, b s c -> b n s")
            attn = F.softmax(logits / self.slot_dim**0.5, dim=2)

            # Weighted mean.
            attn /= attn.sum(dim=1, keepdim=True) + EPS
            updates = E.einsum(attn, v, "b n s, b n c -> b s c")

            # Slot update.
            slots = self.gru(updates.flatten(0, 1), prev.flatten(0, 1)).view_as(prev)
            # Residual.
            slots += self.mlp(self.norm_mlp(slots))

        return slots
