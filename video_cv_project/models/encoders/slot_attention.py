"""Slot Attention implementation in PyTorch based on official TensorFlow implementation.

Some modifications were made, namely:

- Can pass forward previous slots to the next time step.
- Dynamic number of slots between calls.
- Uses `F.scaled_dot_product_attention` for more performance.
"""

from typing import Tuple, Sequence

import torch
import torch.nn as nn

from video_cv_project.utils import inverted_scaled_mean_attention

__all__ = ["SlotAttention", "GroupSlotAttention"]


# TODO: It is possible to use multi-head attention.
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
        self.project_k = nn.Linear(in_feats, slot_dim, bias=False)
        self.project_v = nn.Linear(in_feats, slot_dim, bias=False)

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
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): BNC input features.
            slots (torch.Tensor, optional): BSC slots from previous time step.
            num_slots (int, optional): Number of slots to create. Ignored if ``slots`` is provided.
            num_iters (int, optional): Number of iterations for slots to bind.
            mask (torch.Tensor, optional): BSN attention mask, where True indicates the element should partake in attention.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: BSC slots, BSN attention weights.
        """
        slots = self._init_slots(x, num_slots) if slots is None else slots
        x = self.norm_in(x)
        k, v = self.project_k(x), self.project_v(x)

        # Multiple rounds of attention for slots to bind.
        for _ in range(num_iters):
            q = self.project_q(self.norm_slots(slots))
            attn, weights = inverted_scaled_mean_attention(q, k, v, mask=mask)
            slots = self.gru(attn.flatten(0, 1), slots.flatten(0, 1)).view_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))  # Residual.

        return slots, weights


class GroupSlotAttention(SlotAttention):
    """Slot Attention with slot groups."""

    def __init__(
        self,
        in_feats: int,
        slot_dim: int = 128,
        hid_dim: int = 512,
        groups: int = 2,
    ):
        """Initialize Slot Attention Module with slot groups.

        Args:
            in_feats (int): Number of input features.
            slot_dim (int, optional): Size of representations in each slot.
            hid_dim (int, optional): Size of hidden layer in MLP.
            groups (int, optional): Number of slot groups.
        """
        super(GroupSlotAttention, self).__init__(in_feats, slot_dim, hid_dim)
        self.groups = groups

        # Parameters for groups are added to slot initialization distribution.
        self.groups_mu = nn.ParameterList(
            torch.zeros(1, 1, slot_dim) for _ in range(groups)
        )
        self.groups_logvar = nn.ParameterList(
            torch.zeros(1, 1, slot_dim) for _ in range(groups)
        )

    def _init_group_slots(
        self, x: torch.Tensor, num_slots: int, group: int
    ) -> torch.Tensor:
        """Initialize slots from group.

        Args:
            x (torch.Tensor): BNC input features.
            num_slots (int): Number of slots to create.
            group (int): Which slot group to initialize.

        Returns:
            torch.Tensor: BSC slots.
        """
        mu = self.slots_mu + self.groups_mu[group]
        logvar = self.slots_logvar + self.groups_logvar[group]
        return mu + logvar.exp() * torch.randn(
            len(x), num_slots, self.slot_dim
        ).type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: int = 10,
        num_iters: int = 1,
        mask: torch.Tensor = None,
        slots_per_group: Sequence[int] = (3, 7),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x (torch.Tensor): BNC input features.
            slots (torch.Tensor, optional): BSC slots from previous time step.
            num_slots (int, optional): Use `slots_per_group` instead. Ignored for GroupSlotAttention.
            num_iters (int, optional): Number of iterations for slots to bind.
            mask (torch.Tensor, optional): BSN attention mask, where True indicates the element should partake in attention.
            slots_per_group (Sequence[int], optional): Number of slots per group.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: BSC slots, BSN attention weights.
        """
        assert len(slots_per_group) == self.groups

        slots = (
            torch.cat(
                [
                    self._init_group_slots(x, n, g)
                    for g, n in enumerate(slots_per_group)
                ],
                dim=1,
            )
            if slots is None
            else slots
        )

        return super(GroupSlotAttention, self).forward(
            x, slots=slots, num_slots=num_slots, num_iters=num_iters, mask=mask
        )
