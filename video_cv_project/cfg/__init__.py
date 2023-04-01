"""Some constants used throughout the codebase."""

import torch

__all__ = [
    "RGB",
    "RGB_MEAN",
    "RGB_STD",
    "BEST_DEVICE",
    "PENALTY",
    "CACHE_PATCHES",
    "CACHE_LAST_ATTNS",
]

RGB = 3
# NOTE: Both PyTorch & PIL use RGB order so assume everything is RGB.

RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD = (0.2023, 0.1994, 0.2010)

BEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PENALTY = -1e8

CACHE_PATCHES = "pat"
CACHE_LAST_ATTNS = "attn/-1"
