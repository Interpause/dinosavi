"""Wrapper module that flattens temporal dimension into batch dimension."""

import einops as E
import torch
import torch.nn as nn


class VideoWrapper(nn.Module):
    """Wrapper that flattens temporal dimension into batch dimension.

    The wrapper converts BCTHW input to (B*T)CHW to feed to standard image models.
    Afterwards, the latent maps output by the image model is converted back to BCTHW.
    """

    def __init__(self, model: nn.Module):
        """Create wrapper that flattens temporal dimension into batch dimension.

        Args:
            model (torch.nn.Module): Model to wrap.
        """
        super(VideoWrapper, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): BCTHW input video.

        Returns:
            torch.Tensor: BCTHW output temporal latent maps.
        """
        B = len(x)
        x = E.rearrange(x, "b c t h w -> (b t) c h w")
        y: torch.Tensor = self.model(x)
        return E.rearrange(y, "(b t) c h w -> b c t h w", b=B)
