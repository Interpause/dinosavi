"""Script to cache the dataset."""

import logging
from typing import Dict

import einops as E
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

from video_cv_project.cfg import BEST_DEVICE, CACHE_LAST_ATTNS, CACHE_PATCHES
from video_cv_project.data import create_kinetics_dataloader
from video_cv_project.engine import Trainer
from video_cv_project.utils import hash_model

log = logging.getLogger(__name__)


def cache(cfg: DictConfig):
    """Cache dataset."""
    device = torch.device(cfg.device if cfg.device else BEST_DEVICE)
    epochs = cfg.epochs
    compile = cfg.compile
    log_every = cfg.log_every
    batch_size = cfg.batch_size

    log.info(f"Torch Device: {device}")
    log.info(f"Epochs: {epochs}")
    log.info(f"Compile: {compile}")

    log.debug("Create Model.")
    model = instantiate(cfg.model, _convert_="all")
    model_hash = hash_model(model)
    with open_dict(cfg):
        cfg.vid_cache.model_hash = model_hash
    log.info(f"ViTModel Hash: {model_hash}")
    model = model.to(device).eval().requires_grad_(False)
    if compile:
        model = torch.compile(model, mode="default").eval()  # type: ignore

    cache = instantiate(cfg.vid_cache, _convert_="all")
    log.info(f"Tensor Cache: {cache}")

    log.debug("Create Train Dataloader.")
    dataloader = create_kinetics_dataloader(cfg)
    with open_dict(cfg):
        cfg.total_steps = len(dataloader) * epochs
    log.info(f"Total Steps: {cfg.total_steps}")

    trainer = Trainer(
        dataloader, epochs, logger=log, log_every=log_every, save_every=-1
    )

    log.info(f"Run through and cache dataset.")
    # Use Dict to avoid duplicate clips (due to high `num_clips_per_video`) and
    # otherwise same images (like complete black).
    queue: Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for i, n, video in trainer:
            queue.update(video)

            while len(queue) >= batch_size:
                hashes, ims = [], []

                for _ in range(batch_size):
                    im_hash, im = queue.popitem()
                    hashes.append(im_hash)
                    ims.append(im)

                batch = torch.stack(ims).to(device)
                h, w = np.array(batch.shape[-2:]) // model.config.patch_size

                # Tuple of last_hidden_state, pooler_output, hidden_states, attentions.
                # All except last_hidden_state is optional, so the tuple length
                # varies depending on ViTConfig.
                with torch.inference_mode(mode=not compile):
                    o = model(batch, interpolate_pos_encoding=True)

                pats_t = o[0]
                attns_t = o[-1][-1]

                pats_t = E.rearrange(pats_t[:, 1:], "t (h w) c -> t c h w", h=h, w=w)
                # Get attention weights for the CLS token (token 0).
                attns_t = E.rearrange(
                    attns_t[:, :, 0, 1:], "t n (h w) -> t n h w", h=h, w=w
                )

                for im_hash, pats, attns in zip(hashes, pats_t, attns_t):
                    cache.put_val(im_hash, CACHE_PATCHES, pats)
                    cache.put_val(im_hash, CACHE_LAST_ATTNS, attns)
