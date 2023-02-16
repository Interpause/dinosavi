"""TODO: Add module docstring."""

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
        B = x.shape[0]
        # BCTHW -> BTCHW -> (B*T)CHW
        x = x.transpose(1, 2).flatten(0, 1)
        y: torch.Tensor = self.model(x)  # (B*T)CHW latent maps

        # (B*T)CHW -> BTCHW -> BCTHW
        return y.unflatten(0, (B, -1)).transpose(1, 2)
