"""Adapted from https://github.com/ajabri/videowalk/blob/047f3f40135a4b1be2f837793b89c3dbfe7a6683/code/utils/__init__.py#L273."""

import torch
from torch import nn


class VideoWrapper(nn.Module):
    """Wrapper that flattens temporal dimension into batch dimension.

    The wrapper converts NCTHW input to (N*T)CHW
    """

    def __init__(self, model: nn.Module):
        """Create wrapper that flattens temporal dimension into batch dimension.

        Args:
            model (torch.nn.Module): Model to wrap.
        """
        super(VideoWrapper, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x (torch.Tensor): NCTHW input.

        Returns:
            torch.Tensor: output.
        """
        N, C, T, H, W = x.shape
        # NCTHW -> NTCHW -> (N*T)CHW
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        m = self.model(x)

        #
        return m.view(N, T, *m.shape[-3:]).permute(0, 2, 1, 3, 4)
