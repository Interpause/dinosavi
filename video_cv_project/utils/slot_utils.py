"""Utilities for the Slot Attention-based models."""

from typing import Tuple

import einops as E
import torch
import torch.nn.functional as F

from .crw_utils import create_crw_target as create_target

__all__ = ["inverted_scaled_mean_attention", "infoNCE_loss"]


def inverted_scaled_mean_attention(
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
    debug = {}
    losses = []
    for i, logits in enumerate(logits_p):
        loss = F.cross_entropy(logits, labels)
        losses.append(loss)
        debug[f"loss/t+{i}"] = float(loss)
    return torch.stack(losses).mean(), debug
