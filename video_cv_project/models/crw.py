"""Contrastive Random Walk Model.

Adapted from https://github.com/ajabri/videowalk/blob/047f3f40135a4b1be2f837793b89c3dbfe7a6683/code/model.py.
"""

from functools import partial
from typing import Dict

import einops as E
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from video_cv_project.cfg import BEST_DEVICE, RGB
from video_cv_project.utils import (
    calc_affinity,
    calc_markov,
    create_crw_target,
    infer_outdim,
)

from .heads import FCHead

__all__ = ["CRW"]


class CRW(nn.Module):
    """Contrastive Random Walk model.

    Based on: https://ajabri.github.io/videowalk/
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
        temperature: float = 0.05,
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

        # Arbitrary BTCHW input is fine.
        sz = 256
        enc_dim = infer_outdim(encoder, (1, 1, RGB, sz, sz), device=device)
        self.enc_channels = enc_dim[2]  # Encoder output channels.
        self.map_scale = sz // enc_dim[-1]  # Downscale factor of latent map.

        self.encoder = encoder
        self.head = FCHead(self.enc_channels, num_feats, head_depth)

        self.edge_dropout = edge_dropout
        self.feat_dropout = feat_dropout

        self.temperature = temperature

        self.to(device)
        self.is_trace = False

    def _embed_nodes(self, x: torch.Tensor) -> torch.Tensor:
        """Embed image or image patches using encoder to get node features.

        Whether the input is a list of images or image patches is detected by whether
        the number of patches N is 1 or greater than 1.

        Args:
            x (torch.Tensor): BTNCHW images or image patches.

        Returns:
            torch.Tensor: BTNC node features.
        """
        B, _, N = x.shape[:3]
        x = E.rearrange(x, "b t n c h w -> (b n) t c h w")
        maps: torch.Tensor = self.encoder(x)
        # I don't see how dropout helps. Original had dropout here but didn't use it.
        # maps = F.dropout(maps, p=self.feat_dropout, training=self.training)

        # Use image latent map as nodes.
        if N == 1:
            feats = E.rearrange(maps, "b t c h w -> (b h w) t c")

        # Each node has its own latent map.
        else:
            # Pool latent maps for each patch.
            feats = E.reduce(maps, "b t c h w -> b t c", "mean")

        feats = self.head(feats)
        feats = F.normalize(feats, p=2, dim=2)
        return E.rearrange(feats, "(b n) t c -> b t n c", b=B)

    def _compute_walks(self, feats: torch.Tensor):
        """Compute walks between nodes.

        Args:
            feats (torch.Tensor): BTNC node features.

        Returns:
            Dict[str, torch.Tensor]: Map of sub-cycle palindromes to Markov matrices.

            The Markov matrices are BNM and contain the total transition probability
            from all initial nodes to all final nodes for that palindrome.
        """
        walks: Dict[str, torch.Tensor] = {}

        T = feats.shape[1]

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
        left = tuple(calc_stoch(As[:, t].mT) for t in range(T - 1))

        # Include all sub-cycle palindromes.
        for i in range(T - 1):
            # List of BNM Markov matrices forming a palindrome, e.g., a->b->a.
            # NOTE: Original seems to have a bug, a->b->a is skipped. Instead it
            # starts at a->b->c->b->a. I have opted to remove it.
            # Author kept the bug because it was always there:
            # https://github.com/ajabri/videowalk/issues/21
            edges = right[: i + 1] + left[i::-1]
            path = edges[0]
            for e in edges[1:]:
                path = path @ e

            # NOTE: I removed walking to the left, see:
            # https://github.com/ajabri/videowalk/issues/36
            walks[f"cyc_r{i}"] = path

        return walks

    def _calc_loss(
        self,
        walks: Dict[str, torch.Tensor],
        tgts: torch.Tensor = None,
    ):
        """Calculate cross-entropy loss.

        For every sub-cycle palindrome, cross-entropy loss is calculated between
        the BNM Markov matrix and the target class ids. If target class ids aren't
        given, then assume palindrome, in which the target class ids are the same
        as the initial node ids.

        Args:
            walks (Dict[str, torch.Tensor]): See `self._compute_walks`.
            tgts (torch.Tensor, optional): BN target class ids. Defaults to None.

        Returns:
            Tuple[torch.Tensor, dict]: Loss and debug info.
        """
        losses = []
        debug = {"loss": 0.0, "acc": 0.0}

        # If no target is given, assume palindrome.
        _i = [*walks.values()][0]
        B, N, _ = _i.shape
        target = create_crw_target(B, N, _i.device) if tgts is None else tgts.flatten()

        for name, path in walks.items():
            logits = E.rearrange(path, "b n m -> (b n) m")
            # NOTE: Fixed incorrect cross entropy here:
            # https://github.com/ajabri/videowalk/issues/29
            # NOTE: Below can use `label_smoothing` kwarg to smooth target.
            # See: https://arxiv.org/abs/1906.02629
            # Ideally should smooth target spatially rather than uniformly however.
            loss = F.cross_entropy(logits, target)
            losses.append(loss)

            # TODO: Adding logits to debug might be useful for visualization.
            debug[f"loss/{name}"] = float(loss)
            debug[f"acc/{name}"] = float(logits.argmax(-1).eq(target).float().mean())

        loss = torch.stack(losses).mean()
        debug["acc"] = float(np.mean([v for k, v in debug.items() if "acc" in k]))
        debug["loss"] = float(loss)
        return loss, debug

    def forward(self, x: torch.Tensor, tgts: torch.Tensor = None):
        """Forward pass.

        Args:
            x (torch.Tensor): BTNCHW input patches or images (when N=1).
            tgts (torch.Tensor, optional): BN target patch class ids. Defaults to None.

        Returns:
            Tuple[torch.Tensor, dict]: Loss, metrics.
        """
        # TODO: Add patches to debug info for visualization.
        feats = self._embed_nodes(x)
        walks = self._compute_walks(feats)
        loss, debug = self._calc_loss(walks, tgts)
        if self.is_trace:
            return loss  # type: ignore
        return loss, debug
