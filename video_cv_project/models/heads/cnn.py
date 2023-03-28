"""Basic stacked CNN, why is this not in torchvision?"""

from typing import Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["CNN"]


class CNN(nn.Module):
    """Fairly limited CNN."""

    # TODO: Residual. But not sure which version of residual to use...
    def __init__(
        self,
        channels: Sequence[int],
        size: int | Tuple[int, int],
        activation=nn.ReLU,
        norm=None,
        dropout: float = 0.0,
        padding: int | tuple[int, int] | str = "same",
        bias: bool = True,
    ):
        """Initialize CNN.

        Args:
            channels (Sequence[int]): CNN channels where the 0th element in the input channels and the last is the output channels.
            size (int | Tuple[int, int]): Size of the kernel.
            activation (nn.Module, optional): Activation function to use.
            norm (nn.Module, optional): Normalization function to use.
            dropout (float, optional): Dropout to use.
            padding (int | tuple[int, int] | str, optional): Padding to use.
            bias (bool, optional): Whether to use bias.
        """
        super(CNN, self).__init__()
        assert (
            norm is not nn.BatchNorm2d or dropout == 0.0
        ), "Should not use dropout with batch normalization."
        layers = []
        for i, o in zip(channels[:-2], channels[1:-1]):
            layers += [
                (nn.Conv2d(i, o, size, padding=padding, bias=bias)),
                activation(),
            ]
            if dropout > 0.0:
                layers += [nn.Dropout2d(dropout)]
            if norm is not None:
                layers += [norm(o)]
        layers += [
            nn.Conv2d(channels[-2], channels[-1], size, padding=padding, bias=bias)
        ]
        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.cnn(x)
