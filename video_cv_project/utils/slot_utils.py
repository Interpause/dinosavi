"""Utilities for the Slot Attention-based models."""

from typing import Tuple

import einops as E
import torch
import torch.nn.functional as F

from .crw_utils import create_crw_target as create_target

__all__ = [
    "inverted_scaled_mean_attention",
    "bg_from_attn",
    "calc_slot_masks",
    "calc_lock_on_masks",
    "preds_from_lock_on",
    "infoNCE_loss",
    "vicreg_loss",
    "mse_loss",
]


def inverted_scaled_mean_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    eps: float = 1e-6,
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
    m = torch.tensor(0) if mask is None else mask
    if m.dtype == torch.bool:
        m = torch.zeros(m.shape).to(q.device).masked_fill(~m, float("-inf"))
    m = m.type_as(q)

    w = weight = F.softmax(q @ k.mT / (q.size(-1) ** 0.5) + m, dim=-2)
    w = w / (w.sum(dim=-1, keepdim=True) + eps)
    w = F.dropout(w, dropout_p)
    return w @ v, weight


def bg_from_attn(attn: torch.Tensor) -> torch.Tensor:
    """Return background bitmask, where True indicates background.

    Inspired by FOUND paper: https://arxiv.org/abs/2212.07834

    Args:
        attn (torch.Tensor): *NHW attentions, where N is each attention head.

    Returns:
        torch.Tensor: *HW background bitmask.
    """
    h, w = attn.shape[-2:]
    attn = E.rearrange(attn, "... n h w -> ... n (h w)")
    mean = E.reduce(attn, "... n p -> ... 1 1", "mean")
    sums = E.reduce(attn > mean, "... n p -> ... n", "sum").clamp(min=1)
    total = E.reduce(sums, "... n -> ... 1", "sum")
    weights = (total / sums).log()
    attn = E.einsum(attn, weights, "... n p, ... n -> ... n p")
    attn = E.reduce(attn, "... n (h w) -> ... h w", "mean", h=h, w=w)
    return attn < E.reduce(attn, "... h w -> ... 1 1", "mean")


def calc_slot_masks(
    mask: torch.Tensor, bg: int, fg: int, strategy: str = "always"
) -> torch.Tensor:
    """Calculate per slot bitmasks from background bitmask.

    Args:
        mask (torch.Tensor): TBHW background bitmask, where True indicates background.
        bg (int): Number of background slots.
        fg (int): Number of foreground slots.
        strategy (str, optional): Either "initial" or "always". Defaults to "always".

    Returns:
        torch.Tensor: TBSN slot bitmasks.
    """
    masks = torch.cat(
        [
            E.repeat(mask, "t b h w -> t b s (h w)", s=bg),
            E.repeat(~mask, "t b h w -> t b s (h w)", s=fg),
        ],
        dim=-2,
    )
    if strategy == "initial":
        masks[1:] = True
    elif strategy == "always":
        pass
    else:
        assert False, f"Invalid strategy: {strategy}"
    return masks


def calc_lock_on_masks(
    bg: torch.Tensor,
    num_bg: int,
    fg: torch.Tensor,
    num_fg: int,
    extra: torch.Tensor,
    num_extra: int,
):
    """Calculate per slot bitmasks from first frame labels.

    Args:
        bg (torch.Tensor): *HW background label.
        num_bg (int): Number of slots assigned to background.
        fg (torch.Tensor): *CHW class labels.
        num_fg (int): Number of slots assigned per class.
        extra (torch.Tensor): *HW extra foreground label.
        num_extra (int): Number of slots to assign to foreground objects altogether.

    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: *SN slot bitmasks, tuple of number of (background, foreground) slots.
    """
    bg_lbl = E.repeat(bg, "... h w -> ... s (h w)", s=num_bg)
    fg_lbl = E.repeat(fg, "... c h w -> ... (c s) (h w)", s=num_fg)
    extra_lbl = E.repeat(extra, "... h w -> ... s (h w)", s=num_extra)
    lbl = torch.cat([bg_lbl, extra_lbl, fg_lbl], dim=-2).bool()
    return lbl, (bg_lbl.shape[-2], extra_lbl.shape[-2] + fg_lbl.shape[-2])


def preds_from_lock_on(preds: torch.Tensor, num_bg: int, num_fg: int, num_extra: int):
    """Merge each class prediction from multiple slot predictions.

    Args:
        preds (torch.Tensor): *SHW predictions.
        num_bg (int): Number of slots assigned to background.
        num_fg (int): Number of slots assigned per class.
        num_extra (int): Number of slots to assign to foreground objects altogether.

    Returns:
        torch.Tensor: *CHW class predictions.
    """
    num_bg = num_bg + num_extra
    # Merge class slots by taking max.
    bg_preds = E.reduce(preds[..., :num_bg, :, :], "... s h w -> ... 1 h w", "max")
    fg_preds = E.reduce(
        preds[..., num_bg:, :, :], "... (n s) h w -> ... n h w", "max", s=num_fg
    )
    return torch.cat([bg_preds, fg_preds], dim=-3)


def infoNCE_loss(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """Modified InfoNCE loss for patch features.

    Modified to accept PTNC features, where P is the predicted time step (i.e.,
    t+0, t+1, t+2, ...), and T is the current time step (t=0, 1, 2, ...).

    Args:
        x (torch.Tensor): PTNC Predicted features.
        y (torch.Tensor): PTNC Target features.

    Returns:
        Tuple[torch.Tensor, dict]: Loss, metrics.
    """
    # Cosine similarity.
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    logits_p = E.einsum(x, y, "p t x c, p t y c -> p t x y")
    # `einops` doesn't support rearrange in einsum yet.
    logits_p = E.rearrange(logits_p, "p t x y -> p (t x) y")

    # Labels for 1-1 correspondence between x and y.
    # This is actually wrong. Nearby patches should be similar, so the labels
    # should be soft.
    labels = create_target(x.shape[1], x.shape[2], x.device)

    # Calculate loss for each prediction (t+0, t+1, t+2, ...) separately for
    # metric logging.
    debug = {"loss": 0.0}
    losses = []
    for i, logits in enumerate(logits_p):
        loss = F.cross_entropy(logits, labels)
        losses.append(loss)
        debug[f"loss/t+{i}"] = float(loss)

    loss = torch.stack(losses).mean()
    debug["loss"] = float(loss)
    return loss, debug


def vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    w_inv: float = 25.0,
    w_var: float = 25.0,
    w_cov: float = 1.0,
    enc_frozen=True,
    eps=1e-8,
) -> Tuple[torch.Tensor, dict]:
    """Modified VICReg loss for patch features.

    Modified to accept PTNC features, where P is the predicted time step (i.e.,
    t+0, t+1, t+2, ...), and T is the current time step (t=0, 1, 2, ...).

    Note, the coefficient for Variance loss is 0 by default because for our model
    there is no risk of collapse. This is because the encoder is frozen. Trying to
    enforce variance when matching the encoder output might be counterproductive.

    Actually, the Covariance loss is also counterproductive when the encoder is
    frozen. This means VICReg becomes just MSE loss...

    Args:
        x (torch.Tensor): PTNC Predicted features.
        y (torch.Tensor): PTNC Target features.
        w_inv (float, optional): Invariance loss weight.
        w_var (float, optional): Variance loss weight.
        w_cov (float, optional): Covariance loss weight.
        enc_frozen (bool, optional): Whether the encoder is frozen.
        eps (float, optional): Epsilon value.

    Returns:
        Tuple[torch.Tensor, dict]: Loss, metrics.
    """
    # N is (B*H*W).
    x = E.rearrange(x, "p t n c -> p (t n) c")
    y = E.rearrange(y, "p t n c -> p (t n) c")

    debug = {"loss": 0.0}
    losses = []
    for i, (a, b) in enumerate(zip(x, y)):
        N, C = a.shape

        # Invariance loss.
        loss_inv = F.mse_loss(a, b)

        if not enc_frozen:
            # Variance loss.
            # Theoretically, should be done with (P*T*N), not just (T*N).
            std_a = (a.var(dim=0) + eps) ** 0.5
            std_b = (b.var(dim=0) + eps) ** 0.5
            loss_var = (F.relu(1 - std_a).mean() + F.relu(1 - std_b).mean()) / 2

            # Covariance loss.
            _a = a - a.mean(dim=0)
            _b = b - b.mean(dim=0)
            cov_a = (_a.T @ _a / (N - 1)) ** 2
            cov_b = (_b.T @ _b / (N - 1)) ** 2
            loss_cov = (
                (cov_a.sum() - cov_a.diagonal().sum())
                + (cov_b.sum() - cov_b.diagonal().sum())
            ) / (2 * C)
        else:
            loss_var = torch.tensor(0).type_as(a)
            loss_cov = torch.tensor(0).type_as(a)

        # TODO: Log down each individual loss.
        loss = w_inv * loss_inv + w_var * loss_var + w_cov * loss_cov

        losses.append(loss)
        debug[f"loss/t+{i}"] = float(loss)

    loss = torch.stack(losses).mean()
    debug["loss"] = float(loss)
    return loss, debug


def mse_loss(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """Mean squared error loss.

    Modified to accept PTNC features, where P is the predicted time step (i.e.,
    t+0, t+1, t+2, ...), and T is the current time step (t=0, 1, 2, ...).

    Args:
        x (torch.Tensor): PTNC Predicted features.
        y (torch.Tensor): PTNC Target features.

    Returns:
        Tuple[torch.Tensor, dict]: Loss, metrics.
    """
    # N is (B*H*W).
    x = E.rearrange(x, "p t n c -> p (t n) c")
    y = E.rearrange(y, "p t n c -> p (t n) c")

    debug = {"loss": 0.0}
    losses = []
    for i, (a, b) in enumerate(zip(x, y)):
        loss = F.mse_loss(a, b)
        losses.append(loss)
        debug[f"loss/t+{i}"] = float(loss)

    loss = torch.stack(losses).mean()
    debug["loss"] = float(loss)
    return loss, debug
