"""Fully-connected head."""

import torch
import torch.nn as nn

__all__ = ["FCHead"]


class FCHead(nn.Module):
    """Fully-connected head."""

    def __init__(self, in_feats: int, out_feats: int, depth: int):
        """Fully-connected head.

        Args:
            in_feats (int): Number of input features.
            out_feats (int): Number of output features.
            depth (int): Number of Linear layers to use.
        """
        super(FCHead, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.depth = depth

        dims = [in_feats] * depth + [out_feats]
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for l in self.layers[:-1]:
            x = l(x).relu()
        return self.layers[-1](x)  # Skip activation on last layer.
