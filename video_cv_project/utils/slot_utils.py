"""Utilities for the Slot Attention-based models."""

from functools import cache
from math import ceil
from typing import Tuple

import einops as E
import torch
import torch.nn.functional as F

from .crw_utils import create_crw_target as create_target

__all__ = [
    "inverted_scaled_mean_attention",
    "gen_2d_pe",
    "interpolate_2d_pe",
    "infoNCE_loss",
    "vicreg_loss",
]


@cache
def gen_2d_pe(size: Tuple[int, int], type: str = "linear", sine_dim: int = 4):
    """Memoized function to create 2D positional encodings.

    Two types are supported: ``linear`` and ``sine``. For sinusoidal positional
    encodings, `sine_dim` can be specific for the wanted number of dims.

    Args:
        size (Tuple[int, int]): (H, W) of the positional encodings.
        type (str, optional): Type of positional encodings.
        sine_dim (int, optional): Number of dims for sinusoidal positional encodings.

    Returns:
        torch.Tensor: CHW positional encodings.
    """
    h, w = size
    if type == "linear":
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, w), torch.linspace(-1, 1, h), indexing="ij"
        )
        return torch.stack((gx, gy))
    elif type == "sine":
        assert sine_dim % 2 == 0
        N = 10000  # Value by convention.
        D = ceil(sine_dim / 4) * 2
        freq = torch.pow(N, -torch.arange(0, D, 2) / D)

        x = torch.einsum("i,j->ij", freq, torch.arange(w)).unsqueeze(1)
        y = torch.einsum("i,j->ij", freq, torch.arange(h)).unsqueeze(-1)

        embed = torch.empty(D * 2, h, w)
        embed[:D:2] = x.sin()
        embed[1:D:2] = x.cos()
        embed[D : 2 * D : 2] = y.sin()
        embed[D + 1 : 2 * D : 2] = y.cos()
        return embed

    assert False, f"`{type}` not supported! Only `linear` and `sine` supported."


@cache
def interpolate_2d_pe(embed: torch.Tensor, size: Tuple[int, int]):
    """Memoized function to interpolate positional encodings to new size.

    Args:
        embed (torch.Tensor): CHW positional encodings.
        size (Tuple[int, int]): (H, W) new size.

    Returns:
        torch.Tensor: CHW positional encodings.
    """
    if embed.shape[-2:] == size:
        return embed
    # Yes, this seems messed up, but this is how ViT's resize encodings.
    new: torch.Tensor = F.interpolate(embed[None], size=size, mode="bicubic")
    return new[0]


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

    debug = {}
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
    return torch.stack(losses).mean(), debug
