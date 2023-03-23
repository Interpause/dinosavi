"""Slot Attention implementation in PyTorch based on official TensorFlow implementation.

Some modifications were made, namely:

- Can pass forward previous slots to the next time step.
- Dynamic number of slots between calls.
- Uses `F.scaled_dot_product_attention` for more performance.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SlotAttention"]

# TODO: It is possible to use multi-head attention.


def _slot_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverted, scaled dot product attention with weighted mean.

    See `torch.nn.functional.scaled_dot_product_attention` for documentation. Instead
    of calculating attention weights over the keys, it is done over the queries.
    Furthermore, instead of doing a weighted sum, a weighted mean is done instead
    to get the final output.

    Args:
        q (torch.Tensor): *LE query.
        k (torch.Tensor): *SE key.
        v (torch.Tensor): *SE value.
        mask (torch.Tensor, optional): *LS attention mask.
        dropout_p (float, optional): Dropout probability.
        eps (float, optional): Epsilon value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: *LE attention output, *LS attention weights.
    """
    m = (
        torch.tensor(0)
        if mask is None
        else mask.masked_fill(mask.logical_not(), float("-inf"))
        if mask.dtype == torch.bool
        else mask
    ).type_as(q)

    w = weight = F.softmax(q @ k.mT / (q.size(-1) ** 0.5) + m, dim=-2)
    w = w / (w.sum(dim=-1, keepdim=True) + eps)
    w = F.dropout(w, dropout_p)
    return w @ v, weight


class SlotAttention(nn.Module):
    """Slot Attention Module."""

    def __init__(
        self,
        in_feats: int,
        slot_dim: int = 64,
        hid_dim: int = 128,
    ):
        """Initialize Slot Attention Module.

        Args:
            in_feats (int): Number of input features.
            slot_dim (int, optional): Size of representations in each slot.
            hid_dim (int, optional): Size of hidden layer in MLP.
        """
        super(SlotAttention, self).__init__()

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

    def _init_slots(self, x: torch.Tensor, num_slots: int) -> torch.Tensor:
        """Initialize slots.

        This function can be overwritten for a more intelligent initialization based
        on the input features.

        Args:
            x (torch.Tensor): BNC input features.
            num_slots (int): Number of slots to create.

        Returns:
            torch.Tensor: BSC slots.
        """
        return self.slots_mu + self.slots_logvar.exp() * torch.randn(
            len(x), num_slots, self.slot_dim
        ).type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: int = 7,
        num_iters: int = 3,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BNC input features.
            slots (torch.Tensor, optional): BSC slots from previous time step.
            num_slots (int, optional): Number of slots to create. Ignored if ``slots`` is provided.
            num_iters (int, optional): Number of iterations for slots to bind.

        Returns:
            torch.Tensor: BSC slots.
        """
        slots = self._init_slots(x, num_slots) if slots is None else slots
        x = self.norm_in(x)
        k, v = self.project_kv(x).split(self.slot_dim, dim=2)

        # Multiple rounds of attention for slots to bind.
        for _ in range(num_iters):
            q = self.project_q(self.norm_slots(slots))
            # TODO: Use weights to visualize slot bindings.
            attn, weights = _slot_attention(q, k, v)
            slots = self.gru(attn.flatten(0, 1), slots.flatten(0, 1)).view_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))  # Residual.

        return slots
