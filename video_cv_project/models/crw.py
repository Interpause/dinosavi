"""Adapted from https://github.com/ajabri/videowalk/blob/047f3f40135a4b1be2f837793b89c3dbfe7a6683/code/model.py."""

from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from video_cv_project.cfg import BEST_DEVICE, RGB
from video_cv_project.models.heads import FCHead
from video_cv_project.utils import calc_affinity, calc_markov, infer_outdim


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
        _enc_dim = infer_outdim(encoder, (1, RGB, 1, _sz, _sz), device=device)
        self.enc_channels = _enc_dim[1]  # Encoder output channels.
        self.map_scale = _sz // _enc_dim[-1]  # Downscale factor of latent map.

        self.encoder = encoder
        self.head = FCHead(self.enc_channels, 128, head_depth)  # TODO: why 128?

        self.edge_dropout = edge_dropout
        self.feat_dropout = feat_dropout

        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self._loss_targets = {}

    def _embed_nodes(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed image or image patches using encoder to get node features.

        Whether the input is a list of images or image patches is detected by whether
        the number of patches N is 1 or greater than 1.

        Args:
            x (torch.Tensor): BNCTHW images or image patches.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                feats: BCTN node features.
                maps: BNCTHW latent maps of nodes.
        """
        B, N = x.shape[:2]
        # BNCTHW -> (B*N)CTHW
        maps: torch.Tensor = self.encoder(x.flatten(0, 1))
        maps = F.dropout(maps, p=self.feat_dropout, training=self.training)

        # Use image latent map as nodes.
        if N == 1:
            # (B*N)CTHW -> (B*N)HWCT -> (B*N*H*W)CT -> (B*N)CT
            feats = maps.permute(0, 3, 4, 1, 2).flatten(0, 2)
            # Feature map shared by all nodes: (B*N)CT -> (B*N)CTHW
            maps = feats.reshape(*feats.shape, 1, 1)

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

        # TODO: return feats at this stage if just_feats is True.

        # Compute walks (why do they call it walks not paths) between nodes.
        walks = {}
        calc_stoch = partial(
            calc_markov,
            temperature=self.temperature,
            dropout=self.edge_dropout,
            do_dropout=self.training,
        )
        # BTNM, where N are nodes at t+0, M are nodes at t+1, and T is from t=0 to t-1.
        As = calc_affinity(feats)
        forward = [calc_stoch(As[:, t]) for t in range(T - 1)]
        back = [calc_stoch(As[:, t].transpose(-1, -2)) for t in range(T - 1)]

        return x
