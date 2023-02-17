"""Utility functions used by the CRW model."""

import torch
import torch.nn.functional as F

__all__ = ["calc_affinity", "zero_out_diag", "calc_markov"]


def zero_out_diag(x: torch.Tensor):
    """Zero out the diagonal of a *XY tensor, where X=Y.

    Args:
        x (torch.Tensor): *XY tensor.

    Returns:
        torch.Tensor: Tensor with diagonal zeroed out.
    """
    mask = (1 - torch.eye(x.shape[-1])).to(x.device)
    return torch.einsum("...xy,xy->...xy", x, mask)


def calc_affinity(feats: torch.Tensor) -> torch.Tensor:
    """Calculate affinity matrices between each node and its neighbors for each time step.

    The output is BTNM node affinities where N are nodes at t+0, M are nodes at
    t+1, and T is from t=0 to t-1.

    Affinity is actually cosine similarity.

    Args:
        feats (torch.Tensor): BCTN node features.

    Returns:
        torch.Tensor: BTNM node affinity matrices.
    """
    t0, t1 = feats[:, :, :-1], feats[:, :, 1:]
    A = torch.einsum("bctn,bctm->btnm", t0, t1)  # From t=0 to t-1.
    return A


def calc_markov(
    affinity: torch.Tensor,
    temperature: float = 0.07,
    dropout: float = 0.0,
    do_dropout=False,
    zero_diag: bool = False,
) -> torch.Tensor:
    """Calculate Markov matrix from affinity matrix.

    Args:
        affinity (torch.Tensor): BNM node affinity matrix.
        temperature (float, optional): Temperature for softmax. Defaults to 0.07.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        do_dropout (bool, optional): Whether to apply dropout to the affinity matrix.
        Defaults to False.
        zero_diag (bool, optional): Whether to zero the diagonal of the affinity
        matrix. Defaults to False.

    Returns:
        torch.Tensor: BNM Markov matrix.
    """
    # NOTE: zero_diag is unused upstream, but it could be useful.
    if zero_diag:
        affinity = zero_out_diag(affinity)
    affinity = F.dropout(affinity, p=dropout, training=do_dropout)
    return F.softmax(affinity / temperature, dim=-1)
