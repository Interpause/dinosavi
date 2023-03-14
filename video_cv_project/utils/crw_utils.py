"""Utility functions used by the CRW model."""

from functools import cache

import einops as E
import torch
import torch.nn.functional as F

from video_cv_project.cfg import PENALTY

__all__ = ["calc_affinity", "zero_out_diag", "calc_markov", "create_crw_target"]


def zero_out_diag(x: torch.Tensor):
    """Zero out the diagonal of a *XY tensor, where X=Y.

    Args:
        x (torch.Tensor): *XY tensor.

    Returns:
        torch.Tensor: Tensor with diagonal zeroed out.
    """
    mask = (1 - torch.eye(x.shape[-1])).to(x.device)
    return x * mask


def calc_affinity(feats: torch.Tensor):
    """Calculate affinity matrices between each node and its neighbors for each time step.

    The output is BTNM node affinities where N are nodes at t+0, M are nodes at
    t+1, and T is from t=0 to t-1.

    Affinity is actually cosine similarity.

    Args:
        feats (torch.Tensor): BTNC node features.

    Returns:
        torch.Tensor: BTNM node affinity matrices.
    """
    t0, t1 = feats[:, :-1], feats[:, 1:]
    return E.einsum(t0, t1, "b t n c, b t m c -> b t n m")  # From t=0 to t-1.


def calc_markov(
    affinity: torch.Tensor,
    temperature: float = 0.07,
    dropout: float = 0.0,
    do_dropout: bool = False,
    zero_diag: bool = False,
):
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
    if do_dropout:
        affinity[torch.rand_like(affinity) < dropout] = PENALTY
    return F.softmax(affinity / temperature, dim=2)


@cache
def create_crw_target(batch_size: int, num_nodes: int, device: torch.device):
    """Create palindrome target for contrastive loss.

    Class ids are assigned to each node. For example, if there are 25 patches,
    they will be labelled from 0 to 24. This is repeated for each batch, allowing
    cross-entropy loss to be calculated for the entire batch at once.

    This function is memoized.

    Args:
        batch_size (int): Batch size.
        num_nodes (int): Number of nodes.
        device (torch.device): Device to use.

    Returns:
        torch.Tensor: Suitable tensor of class ids.
    """
    return torch.arange(num_nodes).to(device).repeat(batch_size)
