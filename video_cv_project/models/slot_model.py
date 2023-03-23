"""Main model contribution."""

from typing import Tuple

import einops as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from transformers import ViTModel

from video_cv_project.models.encoders import SlotAttention
from video_cv_project.models.heads import SlotDecoder
from video_cv_project.utils import create_crw_target as create_target

__all__ = ["SlotModel", "SlotCPC"]

# TODO: Test if DINO/ViT embeddings retain positional information.
EXTRA_PE = False


class SlotPredictor(nn.Module):
    """Propagate slots temporally.

    This is what SAVi did, but we might mess with this later.
    """

    def __init__(self, slot_dim: int = 512, num_heads: int = 4):
        """Initialize SlotPredictor.

        Args:
            slot_dim (int, optional): Size of each slot. Defaults to 512.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
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
    """This is the part used during both training & evaluation."""

    def __init__(
        self,
        encoder: ViTModel,
        pe_dim: int = 4,
        slot_dim: int = 512,
        slot_hid_dim: int = 768,
        slot_predict_heads: int = 4,
    ):
        """Initialize SlotModel.

        Args:
            encoder (ViTModel): Model used to encode frames.
            pe_dim (int, optional): Size of positional encoding. Defaults to 4.
            slot_dim (int, optional): Size of each slot. Defaults to 512.
            slot_hid_dim (int, optional): Size of hidden layer in `SlotAttention`. Defaults to 768.
            slot_predict_heads (int, optional): Number of attention heads in `SlotPredictor`. Defaults to 4.
        """
        super(SlotModel, self).__init__()

        self.encoder = encoder
        self.encoder.requires_grad_(False)  # Freeze encoder.

        feat_dim = encoder.config.hidden_size
        enc_chns = feat_dim + pe_dim

        self.attn = SlotAttention(
            in_feats=feat_dim,
            slot_dim=slot_dim,
            hid_dim=slot_hid_dim,
        )
        self.predictor = SlotPredictor(slot_dim=slot_dim, num_heads=slot_predict_heads)

        self.pe = PositionalEncodingPermute2D(pe_dim)
        self.pat_mlp = nn.Linear(enc_chns, feat_dim)

        self.pe_dim = pe_dim
        self.slot_dim = slot_dim
        self.slot_hid_dim = slot_hid_dim
        self.feat_dim = feat_dim

    def _encode(self, im: torch.Tensor) -> torch.Tensor:
        """Encode image to get features for each patch."""
        h, w = np.array(im.shape[-2:]) // self.encoder.config.patch_size
        y = self.encoder(im, interpolate_pos_encoding=True)
        pats = y.last_hidden_state[:, 1:]
        return E.rearrange(pats, "b (h w) c -> b c h w", h=h, w=w)

    def _proc_feats(self, pats: torch.Tensor) -> torch.Tensor:
        """Process features.

        Typically, positional encodings are concatenated to the features.
        """
        if EXTRA_PE:
            enc = self.pe(pats)
            x = torch.cat((enc, pats), dim=1)
            x = E.rearrange(x, "b c h w -> b (h w) c")
            # Run MLP after concating position encodings.
            # Slot prenorms input, no normalization needed here.
            x = self.pat_mlp(x)
        else:
            x = E.rearrange(pats, "b c h w -> b (h w) c")
        return x

    def calc_slots(
        self,
        pats: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: int = 7,
        num_iters: int = 3,
    ) -> torch.Tensor:
        """Calculate slots.

        Args:
            pats (torch.Tensor): BCHW features.
            slots (torch.Tensor, optional): BSC slots.
            num_slots (int, optional): Number of slots to create.
            num_iters (int, optional): Number of iterations to run.

        Returns:
            torch.Tensor: BSC slots.
        """
        x = self._proc_feats(pats)

        # Update slots temporally.
        if slots is not None:
            slots = self.predictor(slots)

        # Bind slots to features.
        slots = self.attn(x, slots=slots, num_slots=num_slots, num_iters=num_iters)
        return slots

    def forward(
        self,
        im: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: int = 7,
        num_iters: int = 3,
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
        pats = self._encode(im)
        slots = self.calc_slots(pats, slots, num_slots, num_iters)
        return slots, pats


class SlotCPC(nn.Module):
    """SlotCPC trainer."""

    def __init__(
        self,
        model: SlotModel,
        decoder: SlotDecoder,
        time_steps: int = 2,
        num_slots: int = 7,
        num_iters: int = 3,
    ):
        """Initialize SlotCPC.

        Args:
            model (SlotModel): Slot model.
            decoder (SlotDecoder): Decodes slot to features.
            time_steps (int, optional): Up to which time step to predict to. Defaults to 2.
            num_slots (int, optional): Number of slots to create. Defaults to 7.
            num_iters (int, optional): Number of iterations for slots to bind. Defaults to 3.
        """
        super(SlotCPC, self).__init__()

        self.model = model
        self.decoder = decoder
        self.time_steps = time_steps
        self.num_preds = time_steps + 1  # Include t+0.
        self.time_mlp = nn.Linear(model.slot_dim, self.num_preds * model.slot_dim)
        self.layernorm = nn.LayerNorm(model.feat_dim)
        self.num_slots = num_slots
        self.num_iters = num_iters

    def _encode(self, vid: torch.Tensor) -> torch.Tensor:
        """Encode video into feature patches."""
        B = len(vid)
        vid = E.rearrange(vid, "b t c h w -> (b t) c h w")
        pats = self.model._encode(vid)
        return E.rearrange(pats, "(b t) c h w -> t b c h w", b=B)

    def _prop_slots(self, pats: torch.Tensor) -> torch.Tensor:
        """Propagate slots forward in time."""
        # TODO: If doing palindrome, reset cur_slots to None & iterate vid in reverse.
        slots = None
        slots_t = [
            slots := self.model.calc_slots(p, slots, self.num_slots, self.num_iters)
            for p in pats[: -self.time_steps]
        ]
        return torch.stack(slots_t)  # TBSC

    def _pred_feats(self, slots: torch.Tensor) -> torch.Tensor:
        """Predict future features from slots."""
        preds = self.time_mlp(slots)
        preds = E.rearrange(preds, "t b s (p c) -> (p t b) s c", p=self.num_preds)
        return self.decoder(preds)  # (PTB)CHW

    def _calc_loss(self, x: torch.Tensor, y: torch.Tensor):
        """Calculate InfoNCE loss."""
        # Cosine similarity.
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        logits_p = E.einsum(x, y, "p t x c, p t y c -> p t x y")
        # `einops` doesn't support rearrange in einsum yet.
        logits_p = E.rearrange(logits_p, "p t x y -> p (t x) y")

        # Labels for 1-1 correspondence between x and y.
        # TODO: This is actually wrong, nearby patches should be similar, so the
        # labels should be soft.
        labels = create_target(x.shape[1], x.shape[2], x.device)

        # Calculate loss for each prediction (t+0, t+1, t+2, ...) separately for
        # metric logging.
        debug = {}
        losses = []
        for i, logits in enumerate(logits_p):
            loss = F.cross_entropy(logits, labels)
            losses.append(loss)
            debug[f"loss/t+{i}"] = float(loss)
        return torch.stack(losses).mean(), debug

    def forward(self, vid: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            vid (torch.Tensor): BTCHW image tensor.

        Returns:
            Tuple[torch.Tensor, dict]: Loss, metrics.
        """
        pats_t = self._encode(vid)
        slots_t = self._prop_slots(pats_t)

        # Predict future time steps simultaneously.
        T, P = len(slots_t), self.num_preds
        x = self._pred_feats(slots_t)
        # Flatten every pixel in batch together for InfoNCE.
        x = E.rearrange(x, "(p t b) c h w -> p t (b h w) c", t=T, p=P)
        x = self.layernorm(x)  # Uniformity with ViT.

        # Get corresponding time steps for each prediction.
        idx = torch.arange(P).expand(T, -1) + torch.arange(T).view(-1, 1)
        y = pats_t[idx]
        y = E.rearrange(y, "t p b c h w -> p t (b h w) c")

        return self._calc_loss(x, y)
