"""Main model contribution."""

from typing import Tuple

import einops as E
import numpy as np
import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute2D
from transformers import ViTModel

from video_cv_project.models.encoders import SlotAttention
from video_cv_project.models.heads import SlotDecoder
from video_cv_project.utils import infoNCE_loss, vicreg_loss

__all__ = ["SlotModel", "SlotCPC"]

# TODO: Test if ViT embeddings retain positional information.
# Might be dependent on pretraining method.
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate slots.

        Args:
            pats (torch.Tensor): BCHW features.
            slots (torch.Tensor, optional): BSC slots.
            num_slots (int, optional): Number of slots to create.
            num_iters (int, optional): Number of iterations to run.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: BSC slots, BSN attention weights.
        """
        x = self._proc_feats(pats)

        # Update slots temporally.
        if slots is not None:
            slots = self.predictor(slots)

        # Bind slots to features.
        slots, weights = self.attn(
            x, slots=slots, num_slots=num_slots, num_iters=num_iters
        )
        return slots, weights

    def forward(
        self,
        im: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: int = 7,
        num_iters: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            im (torch.Tensor): BCHW image tensor.
            slots (torch.Tensor, optional): BSC slots.
            num_slots (int, optional): Number of slots to create.
            num_iters (int, optional): Number of iterations to run.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: BSC slots, BCHW features, BSN attention weights.
        """
        pats = self._encode(im)
        slots, weights = self.calc_slots(pats, slots, num_slots, num_iters)
        return slots, pats, weights


class SlotCPC(nn.Module):
    """SlotCPC trainer."""

    def __init__(
        self,
        model: SlotModel,
        decoder: SlotDecoder,
        freeze_encoder: bool = True,
        time_steps: int = 2,
        num_slots: int = 7,
        num_iters: int = 3,
        ini_iters: int = None,
    ):
        """Initialize SlotCPC.

        Args:
            model (SlotModel): Slot model.
            decoder (SlotDecoder): Decodes slot to features.
            freeze_encoder(bool, optional): Whether to freeze the encoder.
            time_steps (int, optional): Up to which time step to predict to. Defaults to 2.
            num_slots (int, optional): Number of slots to create. Defaults to 7.
            num_iters (int, optional): Number of iterations for slots to bind. Defaults to 3.
            ini_iters (int, optional): Number of iterations for slots to bind when first initialized. Defaults to `num_iters`.
        """
        super(SlotCPC, self).__init__()

        self.model = model
        self.decoder = decoder
        self.time_steps = time_steps
        self.num_preds = time_steps + 1  # Include t+0.
        # TODO: Split this back up unless you can find a way to prune it. Might be used for inference after all.
        self.time_mlp = nn.Linear(model.slot_dim, self.num_preds * model.slot_dim)
        # self.time_mlp = nn.ModuleList(
        #     nn.Linear(model.slot_dim, self.num_preds) for _ in range(self.num_preds)
        # )
        self.layernorm = nn.LayerNorm(model.feat_dim)
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.ini_iters = num_iters if ini_iters is None else ini_iters

        self.encoder_frozen = freeze_encoder

    @property
    def encoder_frozen(self):
        """`encoder_frozen` getter."""
        return self._frozen_enc

    @property.setter
    def encoder_frozen(self, v: bool):
        """`encoder_frozen` setter."""
        self.model.encoder.requires_grad_(v)
        self._frozen_enc = v

    def _encode(self, vid: torch.Tensor) -> torch.Tensor:
        """Encode video into feature patches."""
        B = len(vid)
        vid = E.rearrange(vid, "b t c h w -> (b t) c h w")
        pats = self.model._encode(vid)
        return E.rearrange(pats, "(b t) c h w -> t b c h w", b=B)

    def _prop_slots(self, pats: torch.Tensor) -> torch.Tensor:
        """Propagate slots forward in time."""
        # TODO: If doing palindrome, reset cur_slots to None & iterate vid in reverse.
        s, slots_t = None, []
        T, i = len(pats) - self.time_steps, self.ini_iters
        for p in pats[:T]:
            # TODO: Might intermediate attention masks be useful for something?
            s, _ = self.model.calc_slots(p, s, self.num_slots, i)
            slots_t.append(s)
            i = self.num_iters
        return torch.stack(slots_t)  # TBSC

    def _pred_feats(self, slots: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Predict future features from slots."""
        preds = self.time_mlp(slots)
        preds = E.rearrange(preds, "t b s (p c) -> (p t b) s c", p=self.num_preds)
        return self.decoder(preds, sz)  # (PTB)CHW

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
        h, w = pats_t.shape[-2:]
        x = self._pred_feats(slots_t, (h, w))
        # Flatten every pixel in batch together for InfoNCE.
        x = E.rearrange(x, "(p t b) c h w -> p t (b h w) c", t=T, p=P)
        x = self.layernorm(x)  # Uniformity with ViT.

        # Get corresponding time steps for each prediction.
        idx = torch.arange(P).expand(T, -1) + torch.arange(T).view(-1, 1)
        y = pats_t[idx]
        y = E.rearrange(y, "t p b c h w -> p t (b h w) c")

        # return infoNCE_loss(x, y)
        return vicreg_loss(x, y, enc_frozen=self.encoder_frozen)
