"""Main model contribution."""

from functools import partial
from typing import Sequence, Tuple

import einops as E
import torch
import torch.nn as nn
from transformers import ViTConfig

from video_cv_project.models.encoders import GroupSlotAttention
from video_cv_project.models.heads import SlotDecoder
from video_cv_project.utils import bg_from_attn, calc_slot_masks, mse_loss, tb_viz_slots

__all__ = ["SlotModel", "SlotTrainer", "SAViSlotPredictor"]


class SAViSlotPredictor(nn.Module):
    """Propagate slots over time.

    This is what SAVi did, but we might mess with this later.
    """

    def __init__(self, slot_dim: int = 512, num_heads: int = 4):
        """Initialize SAViSlotPredictor.

        Args:
            slot_dim (int, optional): Size of each slot. Defaults to 512.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
        """
        super(SAViSlotPredictor, self).__init__()

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
        slot_predictor: nn.Module = None,
        slot_dim: int = 512,
        slot_hid_dim: int = 768,
        use_bgfg_groups: bool = False,
    ):
        """Initialize SlotModel.

        Args:
            enc_cfg (ViTConfig): Config of ViT used to encode frames.
            slot_predictor (nn.Module, optional): Propagates slots to the next frame.
            slot_dim (int, optional): Size of each slot. Defaults to 512.
            slot_hid_dim (int, optional): Size of hidden layer in `SlotAttention`. Defaults to 768.
            use_bgfg_groups (bool, optional): Use separate set of slots for background versus foreground.
        """
        super(SlotModel, self).__init__()

        self.enc_cfg = enc_cfg
        feat_dim = enc_cfg.hidden_size

        self.attn = GroupSlotAttention(
            in_feats=feat_dim,
            slot_dim=slot_dim,
            hid_dim=slot_hid_dim,
            num_groups=2 if use_bgfg_groups else 0,
        )
        self.slot_predictor = slot_predictor

        self.use_bgfg_groups = use_bgfg_groups
        self.slot_dim = slot_dim
        self.slot_hid_dim = slot_hid_dim
        self.feat_dim = feat_dim

    def forward(
        self,
        pats: torch.Tensor,
        slots: torch.Tensor = None,
        num_slots: Sequence[int] | int = 7,
        num_iters: int = 3,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            pats (torch.Tensor): BCHW features.
            slots (torch.Tensor, optional): BSC slots.
            num_slots (Sequence[int] | int, optional): Number of slots to create.
            num_iters (int, optional): Number of iterations to run.
            mask (torch.Tensor, optional): BSN attention bitmask, where True indicates the element should partake in attention.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: BSC slots, BSN attention weights.
        """
        x = E.rearrange(pats, "b c h w -> b (h w) c")

        # Update slots temporally.
        if slots is not None:
            # NOTE: Set `slot_predictor` to `nn.Identity` instead of None if you want to propagate slots.
            slots = None if self.slot_predictor is None else self.slot_predictor(slots)

        # Bind slots to features.
        slots, weights = self.attn(x, slots, num_slots, num_iters, mask)
        return slots, weights


class SlotTrainer(nn.Module):
    """SlotModel trainer."""

    def __init__(
        self,
        model: SlotModel,
        decoder: SlotDecoder,
        time_steps: Sequence[int] | int = 2,
        num_slots: Tuple[int, int] | int = 5,
        num_iters: int = 3,
        ini_iters: int = None,
        use_bgfg_groups: bool = False,
        bgfg_strategy: str = "initial",
        time_decoder: str = "linear",
    ):
        """Initialize SlotTrainer.

        Args:
            model (SlotModel): Slot model.
            decoder (SlotDecoder): Decodes slot to features.
            time_steps (Sequence[int] | int, optional): Up to which time step to predict to. Defaults to 2.
            num_slots (Tuple[int, int] | int, optional): Number of slots to create. Defaults to 7.
                If tuple, should be (background, foreground).
            num_iters (int, optional): Number of iterations for slots to bind. Defaults to 3.
            ini_iters (int, optional): Number of iterations for slots to bind when first initialized. Defaults to `num_iters`.
            use_bgfg_groups (bool, optional): Use separate set of slots for background versus foreground.
            bgfg_strategy (str, optional): Either "initial" or "always".
            time_decoder (str, optional): Either "linear" or "attn".
        """
        super(SlotTrainer, self).__init__()

        self.use_bgfg_groups = use_bgfg_groups
        self.bgfg_strategy = bgfg_strategy
        assert model.use_bgfg_groups == self.use_bgfg_groups
        assert self.use_bgfg_groups or isinstance(num_slots, int)

        self.model = model
        self.decoder = decoder
        self.time_steps = (
            list(range(time_steps + 1)) if isinstance(time_steps, int) else time_steps
        )

        self.time_dec_type = time_decoder
        if time_decoder == "linear":
            layer: partial = partial(nn.Linear, model.slot_dim, model.slot_dim)
        elif time_decoder == "attn":
            # TODO: Config for number of heads here.
            layer = partial(nn.MultiheadAttention, model.slot_dim, 4, batch_first=True)
        elif time_decoder is None:
            layer = nn.Identity
        else:
            assert False, f"Unsupported time decoder: {time_decoder}"
        self.time_decoder = nn.ModuleList(layer() for _ in range(len(self.time_steps)))

        self.layernorm = nn.LayerNorm(model.feat_dim)
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.ini_iters = num_iters if ini_iters is None else ini_iters

        self.is_trace = False

    def _calc_slot_masks(self, attn: torch.Tensor) -> torch.Tensor:
        """Calculate per slot bitmasks from attention weights."""
        bg, fg = (
            [self.num_slots] * 2 if isinstance(self.num_slots, int) else self.num_slots
        )
        mask = bg_from_attn(attn)
        masks = calc_slot_masks(mask, bg, fg, self.bgfg_strategy)
        return masks

    def _prop_slots(self, pats: torch.Tensor, masks: torch.Tensor = None):
        """Propagate slots forward in time."""
        # TODO: If doing palindrome, reset cur_slots to None & iterate vid in reverse.
        s, slots_t, attn_t = None, [], []
        T, i = len(pats) - max(self.time_steps), self.ini_iters

        if self.use_bgfg_groups and masks is not None:
            smasks = list(self._calc_slot_masks(masks[:T]))
        else:
            smasks = [None] * T

        for p, m in zip(pats[:T], smasks[:T]):
            s, w = self.model(p, s, self.num_slots, i, m)
            slots_t.append(s)
            attn_t.append(w)
            i = self.num_iters
        return torch.stack(slots_t), torch.stack(attn_t)

    def _pred_feats(self, slots: torch.Tensor, sz: Tuple[int, int]) -> torch.Tensor:
        """Predict future features from slots."""
        slots = E.rearrange(slots, "t b s c -> (t b) s c")
        if self.time_dec_type == "attn":
            preds_p = [l(slots, slots, slots)[0] for l in self.time_decoder]
        else:
            preds_p = [l(slots) for l in self.time_decoder]
        preds = E.rearrange(preds_p, "p b s c -> (p b) s c")
        return self.decoder(preds, sz)  # (PTB)CHW

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
        if cls_attns is not None:
            cls_attns = E.rearrange(cls_attns, "b t n h w -> t b n h w")
        slots_t, attn_t = self._prop_slots(pats_t, cls_attns)

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
