"""Contrastive Random Walk Model.

Adapted from https://github.com/ajabri/videowalk/blob/047f3f40135a4b1be2f837793b89c3dbfe7a6683/code/model.py.
"""

from functools import partial
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from video_cv_project.cfg import BEST_DEVICE, EPS, RGB
from video_cv_project.models.heads import FCHead
from video_cv_project.utils import (
    calc_affinity,
    calc_markov,
    create_crw_target,
    infer_outdim,
)

__all__ = ["CRW"]


class CRW(nn.Module):
    """Contrastive Random Walk model.

    TODO: How to cite paper in a docstring? https://ajabri.github.io/videowalk/
    NOTE: While it is called a model, Contrastive Random Walk is more similar in
    nature to a training technique than a model. In essence, this class is more
    of a trainer that wraps over the encoder being trained. Which is why loss calculation
    is done within the model itself, and in downstream tasks, only the encoder is
    used.

    Attributes:
        encoder (torch.nn.Module): Encoder for image patches.
        head (torch.nn.Module): Head to get node features from encoder latent map.
        edge_dropout (float): Dropout applied to edges in walk.
        feat_dropout (float): Dropout applied to latent map from encoder.
        temperature (float): Temperature of softmax.
        enc_channels (int): Number of channels in latent map from encoder.
        map_scale (int): Downscale factor of latent map.
    """

    def __init__(
        self,
        encoder: nn.Module,
        edge_dropout: float = 0.0,
        feat_dropout: float = 0.0,
        temperature: float = 0.07,
        head_depth: int = 1,
        num_feats: int = 128,
        device=BEST_DEVICE,
    ):
        """Create Contrastive Random Walk model.

        Args:
            encoder (torch.nn.Module): Model to use for encoding image patches.
            edge_dropout (float, optional): Dropout applied to edges in walk. Defaults to 0.0.
            feat_dropout (float, optional): Dropout applied to latent map from encoder. Defaults to 0.0.
            temperature (float, optional): Temperature of softmax. Defaults to 0.07.
            head_depth (int, optional): Number of layers for FC head. Defaults to 1.
            num_feats (int, optional): Embedding dimension of FC head. Defaults to 128.
            device (torch.device, optional): Which device to use. Defaults to BEST_DEVICE.
        """
        super(CRW, self).__init__()

        # Arbitrary BCTHW input is fine.
        _sz = 256
        _enc_dim = infer_outdim(encoder, (1, RGB, 1, _sz, _sz), device=device)
        self.enc_channels = _enc_dim[1]  # Encoder output channels.
        self.map_scale = _sz // _enc_dim[-1]  # Downscale factor of latent map.

        self.encoder = encoder
        self.head = FCHead(self.enc_channels, num_feats, head_depth)

        self.edge_dropout = edge_dropout
        self.feat_dropout = feat_dropout

        self.temperature = temperature

        self.to(device)

    def _embed_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image or image patches using encoder to get node features.

        Whether the input is a list of images or image patches is detected by whether
        the number of patches N is 1 or greater than 1.

        Args:
            x (torch.Tensor): BNCTHW images or image patches.

        Returns:
            torch.Tensor: BCTN node features.
        """
        B, N = x.shape[:2]
        feats: torch.Tensor
        # BNCTHW -> (B*N)CTHW
        maps: torch.Tensor = self.encoder(x.flatten(0, 1))
        maps = F.dropout(maps, p=self.feat_dropout, training=self.training)

        # Use image latent map as nodes.
        if N == 1:
            # (B*N)CTHW -> (B*N)HWCT -> (B*N*H*W)CT -> (B*N)CT
            feats = maps.permute(0, 3, 4, 1, 2).flatten(0, 2)

        # Each node has its own latent map.
        else:
            # Pool latent maps: (B*N)CTHW -> (B*N)CT
            feats = maps.sum((-1, -2)) / np.prod(maps.shape[-2:])

        feats = self.head(feats.transpose(-1, -2)).transpose(-1, -2)
        # (B*N)CT -> BNCT -> BCTN
        return feats.unflatten(0, (B, -1)).permute(0, 2, 3, 1)

    def _compute_walks(self, feats: torch.Tensor):
        """Compute walks between nodes.

        Args:
            feats (torch.Tensor): BCTN node features.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]: Map of sub-cycle palindromes
            to Markov matrices and target class ids.

            The Markov matrices are BNM and contain the total transition probability
            from all initial nodes to all final nodes for that palindrome.
        """
        walks: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        T = feats.shape[2]

        calc_stoch = partial(
            calc_markov,
            temperature=self.temperature,
            dropout=self.edge_dropout,
            do_dropout=self.training,
        )

        # BTNM affinity matrices for transitions from all N to all M, where N are
        # nodes at t+0, M are nodes at t+1, and T is from t=0 to t-1.
        As = calc_affinity(feats)

        # List of BNM Markov matrices at each time step for transitions from all
        # N to all M in both directions (left & right).
        right = tuple(calc_stoch(As[:, t]) for t in range(T - 1))
        left = tuple(calc_stoch(As[:, t].transpose(-1, -2)) for t in range(T - 1))

        # TODO: Can limit min length of sub-cycle palindromes for better training?
        # Include all sub-cycle palindromes.
        for i in range(1, T - 1):
            # List of BNM Markov matrices forming a palindrome, e.g., a->b->a.
            # NOTE: Original seems to have a bug, a->b->a is skipped. Instead it
            # starts at a->b->c->b->a. I have opted to keep it in for now.
            edges = right[: i + 1] + left[i::-1]
            path = edges[0]
            for e in edges[1:]:
                path @= e

            # NOTE: I removed walking to the left since it was marked as "bug" in
            # the original's argument.py.
            B, N, _ = path.shape
            walks[f"cyc r{i}"] = (path, create_crw_target(B, N, path.device))

        return walks

    def _calc_loss(self, walks: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """Calculate cross-entropy loss.

        For every sub-cycle palindrome, cross-entropy loss is calculated between
        the BNM Markov matrix and the target class ids.

        Args:
            walks (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): See `self._compute_walks`.

        Returns:
            Tuple[torch.Tensor, dict]: Loss and debug info.
        """
        losses = []
        debug = {}

        for name, (path, target) in walks.items():
            # BNM -> (B*N)M
            logits = torch.log(path + EPS).flatten(0, 1)
            loss = F.cross_entropy(logits, target)
            losses.append(loss)

            # TODO: Adding logits to debug might be useful for visualization.
            debug[name] = {
                "loss": float(loss),
                "accuracy": float((logits.argmax(-1) == target).float().mean()),
                "patches": path.shape[1],
            }

        return torch.stack(losses).mean(), debug

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x (torch.Tensor): BT(N*C)HW input images or image patches.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, dict]: BCTN node features, loss, and debug info.
        """
        # Input is BT(N*C)HW where:
        #   - N=1: Batch of images.
        #   - N>1: Batch of image patches.

        # BT(N*C)HW -> B(N*C)THW -> BNCTHW
        x = x.transpose(1, 2).unflatten(1, (-1, RGB))
        feats = self._embed_nodes(x)
        walks = self._compute_walks(feats)
        loss, debug = self._calc_loss(walks)
        return feats, loss, debug
