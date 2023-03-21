"""Main model contribution."""

from typing import List, Tuple

import einops as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding2D
from torchvision.ops import MLP
from transformers import ViTModel

from video_cv_project.models.encoders import SlotAttention
from video_cv_project.models.heads import SlotDecoder

__all__ = ["SlotModel", "SlotCPC"]


class SlotPredictor(nn.Module):
    """Propagate slots temporally."""

    def __init__(self, slot_dim: int = 512, num_heads: int = 4):
        """Initialize SlotPredictor."""
        super(SlotPredictor, self).__init__()

        self.attn = nn.MultiheadAttention(slot_dim, num_heads, batch_first=True)
        self.mlp = nn.Linear(slot_dim, slot_dim)
        self.ln_attn = nn.LayerNorm(slot_dim)
        self.ln_mlp = nn.LayerNorm(slot_dim)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            slots (torch.Tensor): BSC slots.

        Returns:
            torch.Tensor: BSC slots.
        """
        slots = self.ln_attn(slots + self.attn(slots, slots, slots)[0])
        slots = self.ln_mlp(slots + self.mlp(slots).relu())
        return slots


class SlotModel(nn.Module):
    """This is the part used both during training & evaluation."""

    def __init__(
        self,
        encoder: ViTModel,
        pe_dim: int = 4,
        hid_dim: int = 512,
        slot_dim: int = 512,
        slot_hid_dim: int = 768,
        slot_predict_heads: int = 4,
    ):
        """Initialize SlotModel."""
        super(SlotModel, self).__init__()

        self.encoder = encoder
        self.encoder.requires_grad_(False)  # Freeze encoder.

        self.attn = SlotAttention(
            in_feats=hid_dim,
            slot_dim=slot_dim,
            hid_dim=slot_hid_dim,
        )
        self.predictor = SlotPredictor(slot_dim=slot_dim, num_heads=slot_predict_heads)

        self.pe = PositionalEncoding2D(pe_dim)
        enc_chns = self.encoder.config.hidden_size + pe_dim
        self.pat_mlp = nn.Linear(enc_chns, hid_dim)

        self.pe_dim = pe_dim
        self.hid_dim = hid_dim
        self.slot_dim = slot_dim
        self.slot_hid_dim = slot_hid_dim
        self.feat_dim = encoder.config.hidden_size

    def forward(
        self,
        im: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: int = 7,
        num_iters: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            im (torch.Tensor): BCHW image tensor.
            slots (torch.Tensor, optional): BSC slots.
            num_slots (int, optional): Number of slots to create.
            num_iters (int, optional): Number of iterations to run.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: BSC slots, BCHW features.
        """
        h, w = np.array(im.shape[-2:]) // self.encoder.config.patch_size

        # Encoder image to get features for each patch.
        y = self.encoder(im, interpolate_pos_encoding=True)
        cls, pats = y.last_hidden_state[:, 0], y.last_hidden_state[:, 1:]

        # Concat positional encodings to the patches.
        pats = E.rearrange(pats, "b (h w) c -> b h w c", h=h, w=w)
        enc = self.pe(pats)
        x = torch.cat((enc, pats), dim=-1)

        # Run MLP after concating position encodings.
        x = E.rearrange(x, "b h w c -> b (h w) c")
        # self.pat_mlp might be problematic? Need LayerNorm? Activation? Residual?
        x = self.pat_mlp(x)

        # Update slots temporally.
        if slots is not None:
            slots = self.predictor(slots)

        # Bind slots to features.
        slots = self.attn(x, slots=slots, num_slots=num_slots, num_iters=num_iters)

        return slots, E.rearrange(pats, "b h w c -> b c h w")


class SlotCPC(nn.Module):
    """SlotCPC trainer."""

    def __init__(
        self,
        model: SlotModel,
        decoder: SlotDecoder,
        time_steps: int = 2,
        num_slots: int = 7,
        num_iters: int = 2,
    ):
        """Initialize SlotCPC."""
        super(SlotCPC, self).__init__()

        self.model = model
        self.decoder = decoder
        self.time_steps = time_steps
        self.time_mlp = nn.ModuleList(
            nn.Linear(model.slot_dim, model.slot_dim) for _ in range(time_steps + 1)
        )
        self.layernorm = nn.LayerNorm(model.feat_dim)
        self.num_slots = num_slots
        self.num_iters = num_iters

    def forward(self, vid: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            vid (torch.Tensor): BTCHW image tensor.

        Returns:
            Tuple[torch.Tensor, dict]: Loss, metrics.
        """
        vid = E.rearrange(vid, " b t c h w -> t b c h w")

        slots_t = []
        pats_t = []
        cur_slots = None
        for im in vid:
            cur_slots, cur_pats = self.model(
                im, slots=cur_slots, num_slots=self.num_slots, num_iters=self.num_iters
            )
            slots_t.append(cur_slots)
            pats_t.append(cur_pats)
        slots_t = torch.stack(slots_t)  # TBSC
        pats_t = torch.stack(pats_t)  # TBCHW

        # TODO: If doing palindrome, reset cur_slots to None & iterate vid in reverse.

        losses: List[list | torch.Tensor] = [[] for _ in self.time_mlp]
        for t in range(len(vid) - self.time_steps):
            # Timesteps k to predict (t+k).
            steps = list(range(t, t + len(self.time_mlp)))
            # print(steps)
            slots = slots_t[t]
            x = torch.stack([self.decoder(mlp(slots)) for mlp in self.time_mlp])
            # print(x.shape)
            # Every pixel in every batch, flattened together for contrastive loss.
            # Is this actually correct...?
            x = E.rearrange(x, "t b c h w -> t (b h w) c")
            # Layer norm predictions for uniformity with ViT.
            x = self.layernorm(x)

            pats = pats_t[steps]
            y = E.rearrange(pats, "t b c h w -> t (b h w) c")

            # Cosine similarity.
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            logits_t = E.einsum(x, y, "t x c, t y c -> t x y")

            # Should have 1-1 correspondence between x and y.
            labels = torch.arange(logits_t.shape[-1]).to(logits_t.device)

            for i, logits in enumerate(logits_t):
                # Use softmax here?
                losses[i].append(F.cross_entropy(logits, labels))  # type: ignore

        debug = {}
        for i in range(len(losses)):
            loss = losses[i] = torch.stack(losses[i]).mean()  # type: ignore
            debug[f"loss/t+{i}"] = float(loss)

        return torch.stack(losses).mean(), debug  # type: ignore
