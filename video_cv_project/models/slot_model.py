"""Main model contribution."""

from typing import Tuple, Sequence

import einops as E
import torch
import torch.nn as nn
from transformers import ViTConfig

from video_cv_project.models.encoders import GroupSlotAttention
from video_cv_project.models.heads import SlotDecoder
from video_cv_project.utils import gen_2d_pe, mse_loss, tb_viz_slots

__all__ = ["SlotModel", "SlotCPC"]


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
        enc_cfg: ViTConfig,
        add_pe: bool = False,
        slot_dim: int = 512,
        slot_hid_dim: int = 768,
        slot_predict_heads: int = 4,
    ):
        """Initialize SlotModel.

        `add_pe` is not necessary as `ViTModel` seems to retain positional info
        even after several transformer layers.

        Args:
            enc_cfg (ViTConfig): Config of ViT used to encode frames.
            add_pe (bool, optional): Add extra positional encoding to patches.
            slot_dim (int, optional): Size of each slot. Defaults to 512.
            slot_hid_dim (int, optional): Size of hidden layer in `SlotAttention`. Defaults to 768.
            slot_predict_heads (int, optional): Number of attention heads in `SlotPredictor`. Defaults to 4.
        """
        super(SlotModel, self).__init__()

        self.enc_cfg = enc_cfg
        feat_dim = enc_cfg.hidden_size

        # TODO: Use GroupSlotAttention to handle foreground/background.
        # - Refer to ignore.ipynb on how to use DINO attn to get background mask.
        # - Convert background mask to background & foreground bitmasks.
        # - Pass bitmasks to GroupSlotAttention.
        # - Figure out how to expose the config for slots per group.

        self.attn = GroupSlotAttention(
            in_feats=feat_dim, slot_dim=slot_dim, hid_dim=slot_hid_dim, groups=0
        )
        self.predictor = SlotPredictor(slot_dim=slot_dim, num_heads=slot_predict_heads)
        self.pat_mlp = nn.Linear(feat_dim + 2, feat_dim)  # Linear pos enc so add 2.

        self.add_pe = add_pe
        self.slot_dim = slot_dim
        self.slot_hid_dim = slot_hid_dim
        self.feat_dim = feat_dim

    def _proc_feats(self, pats: torch.Tensor) -> torch.Tensor:
        """Process features."""
        if self.add_pe:
            enc = gen_2d_pe(tuple(pats.shape[-2:])).type_as(pats)
            x = torch.cat((enc, pats), dim=1)
            x = E.rearrange(x, "b c h w -> b (h w) c")
            # Run MLP after concating position encodings.
            # Slot prenorms input, no normalization needed here.
            x = self.pat_mlp(x)
        else:
            x = E.rearrange(pats, "b c h w -> b (h w) c")
        return x

    def forward(
        self,
        pats: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: int = 7,
        num_iters: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

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


class SlotCPC(nn.Module):
    """SlotCPC trainer."""

    def __init__(
        self,
        model: SlotModel,
        decoder: SlotDecoder,
        time_steps: Sequence[int] | int = 2,
        num_slots: int = 7,
        num_iters: int = 3,
        ini_iters: int = None,
    ):
        """Initialize SlotCPC.

        Args:
            model (SlotModel): Slot model.
            decoder (SlotDecoder): Decodes slot to features.
            time_steps (Sequence[int] | int, optional): Up to which time step to predict to. Defaults to 2.
            num_slots (int, optional): Number of slots to create. Defaults to 7.
            num_iters (int, optional): Number of iterations for slots to bind. Defaults to 3.
            ini_iters (int, optional): Number of iterations for slots to bind when first initialized. Defaults to `num_iters`.
        """
        super(SlotCPC, self).__init__()

        self.model = model
        self.decoder = decoder
        self.time_steps = (
            list(range(time_steps + 1)) if isinstance(time_steps, int) else time_steps
        )
        self.time_mlp = nn.ModuleList(
            nn.Linear(model.slot_dim, model.slot_dim)
            for _ in range(len(self.time_steps))
        )
        self.layernorm = nn.LayerNorm(model.feat_dim)
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.ini_iters = num_iters if ini_iters is None else ini_iters

        self.is_trace = False

    def _prop_slots(self, pats: torch.Tensor):
        """Propagate slots forward in time."""
        # TODO: If doing palindrome, reset cur_slots to None & iterate vid in reverse.
        s, slots_t, attn_t = None, [], []
        T, i = len(pats) - max(self.time_steps), self.ini_iters
        for p in pats[:T]:
            # TODO: Might intermediate attention masks be useful for something?
            s, w = self.model.forward(p, s, self.num_slots, i)
            slots_t.append(s)
            attn_t.append(w)
            i = self.num_iters
        return torch.stack(slots_t), torch.stack(attn_t)

    def _pred_feats(self, slots: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Predict future features from slots."""
        preds_p = [l(slots) for l in self.time_mlp]
        preds = E.rearrange(preds_p, "p t b s c -> (p t b) s c")
        return self.decoder(preds, sz)  # (PTB)CHW

    def forward(
        self, vid: torch.Tensor, cls_attns: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Forward pass.

        Args:
            vid (torch.Tensor): BTCHW image patches.
            cls_attns (torch.Tensor): BTNHW CLS token attention weights.

        Returns:
            Tuple[torch.Tensor, dict]: Loss, metrics.
        """
        pats_t = E.rearrange(vid, "b t c h w -> t b c h w")
        slots_t, attn_t = self._prop_slots(pats_t)

        # Predict future time steps simultaneously.
        T, P = len(slots_t), len(self.time_steps)
        h, w = pats_t.shape[-2:]
        x = self._pred_feats(slots_t, (h, w))
        # Flatten every pixel in batch together for loss.
        x = E.rearrange(x, "(p t b) c h w -> p t (b h w) c", t=T, p=P)
        x = self.layernorm(x)  # Uniformity with ViT.

        # Get corresponding time steps for each prediction.
        idx = torch.tensor(self.time_steps).expand(T, -1) + torch.arange(T).view(-1, 1)
        y = pats_t[idx]
        y = E.rearrange(y, "t p b c h w -> p t (b h w) c")

        loss, debug = mse_loss(x, y)
        # NOTE: When logs are flushed to disk, this takes significant time to write.
        # debug["slot_attn"] = tb_viz_slots(pats_t[-1, -1], attn_t[-1, -1])
        if self.is_trace:
            return loss  # type: ignore
        return loss, debug
