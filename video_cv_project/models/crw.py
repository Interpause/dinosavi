"""Adapted from https://github.com/ajabri/videowalk/blob/047f3f40135a4b1be2f837793b89c3dbfe7a6683/code/model.py."""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from video_cv_project.cfg import BEST_DEVICE, RGB
from video_cv_project.models.heads import FCHead
from video_cv_project.utils import infer_outdim

EPS = 1e-20


class CRW(nn.Module):
    """Contrastive Random Walk model.

    TODO: How to cite paper in a docstring? https://ajabri.github.io/videowalk/
    """

    def __init__(
        self,
        encoder: nn.Module,
        edge_dropout: float = 0.0,
        feat_dropout: float = 0.0,
        temperature: float = 0.07,
        head_depth: int = 1,
        device=BEST_DEVICE,
    ):
        super(CRW, self).__init__()

        # Arbitrary BCTHW input is fine.
        _sz = 256
        _enc_dim = infer_outdim(encoder, (1, RGB, 1, _sz, _sz), device=device)[1]
        self.enc_channels: int = _enc_dim[1]  # Encoder output channels.
        self.map_scale: int = _sz // _enc_dim[-1]  # Latent map scale compared to input.

        self.encoder = encoder
        self.head = FCHead(self.enc_channels, 128, head_depth)  # TODO: why 128?

        self.edge_drop = nn.Dropout(edge_dropout)
        self.feat_drop = nn.Dropout(feat_dropout)

        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self._loss_targets = {}

    def _embed_nodes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed nodes (image patches)."""
        B = x.shape[0]
        # BNCTHW -> (B*N)CTHW
        maps: torch.Tensor = self.encoder(x.flatten(0, 1))
        maps = self.feat_drop(maps)

        # Use image latent map as nodes.
        if x.shape[1] == 1:
            # (B*N)CTHW -> (B*N)HWCT -> (B*N*H*W)CT -> (B*N)CT
            feats = maps.permute(0, 3, 4, 1, 2).flatten(0, 2)
            # Feature map shared by all nodes.
            # (B*N)CT -> (B*N)CTHW
            maps = feats[..., None, None]

        # Each node has its own latent map.
        else:
            # Pool latent maps: (B*N)CTHW -> (B*N)CT
            feats = maps.sum((-1, -2)) / np.prod(maps.shape[-2:])

        feats: torch.Tensor = self.head(feats.transpose(-1, -2)).transpose(-1, -2)
        # (B*N)CT -> BNCT -> BCTN
        feats = feats.unflatten(0, (B, -1)).permute(0, 2, 3, 1)
        # (B*N)CTHW -> BNCTHW
        maps = maps.unflatten(0, (B, -1))

        return feats, maps

    def forward(self, x: torch.Tensor, just_feats: bool = False):
        # Input is BT(N*C)HW where:
        #   - N=1: Batch of images.
        #   - N>1: Batch of image patches.

        # BT(N*C)HW -> B(N*C)THW -> BNCTHW
        x = x.transpose(1, 2).unflatten(1, (-1, RGB))
        feats, maps = self._embed_nodes(x)
        B, C, T, N = feats.shape

        return x
