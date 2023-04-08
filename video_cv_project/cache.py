"""Script to cache the dataset."""

import logging
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from multiprocessing import BoundedSemaphore
from typing import Dict

import einops as E
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from transformers import ViTModel

from video_cv_project.cfg import BEST_DEVICE, CACHE_LAST_ATTNS, CACHE_PATCHES
from video_cv_project.data import create_kinetics_dataloader
from video_cv_project.engine import Trainer
from video_cv_project.utils import hash_model

log = logging.getLogger(__name__)


@torch.no_grad()
def max_compile(model: ViTModel, batch_size: int, device: torch.device) -> ViTModel:
    """Compile model."""
    log.debug("Compile Model.")
    model.to(device).eval()
    B, C, S = batch_size, model.config.num_channels, model.config.image_size
    x1 = torch.rand(B, C, S, S).to(device)
    x2 = torch.rand(B, C, S, S).to(device)
    before = model(x2)[0]
    model = torch.jit.trace(model, x1, check_trace=False)
    model = torch.jit.optimize_for_inference(model)
    # model = torch.compile(model, mode="max-autotune", fullgraph=True)  # type: ignore
    after = model(x2)[0]
    log.info(f"Compile Max Diff: {abs(after-before).max()}")
    return model


@torch.inference_mode()
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
    model: ViTModel = instantiate(cfg.model, _convert_="all")
    patch_size = model.config.patch_size
    model_hash = hash_model(model)
    with open_dict(cfg):
        cfg.vid_cache.model_hash = model_hash
    log.info(f"ViTModel Hash: {model_hash}")

    model = max_compile(model, batch_size, device) if compile else model
    model.to(device).eval()

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

    num_writers = cfg.data.num_workers  # type(cache).SETTINGS["shards"]
    semaphore = BoundedSemaphore(num_writers)
    log.info(f"Dataset Shards/Num Concurrent Writers: {num_writers}")

    def done(future):
        nonlocal semaphore
        semaphore.release()
        future.result()

    log.info(f"Run through and cache dataset.")
    # Use Dict to avoid duplicate clips (due to high `num_clips_per_video`) and
    # otherwise same images (like complete black).
    queue: Dict[str, torch.Tensor] = {}
    with PoolExecutor(max_workers=num_writers) as pool:
        for i, n, video in trainer:
            for v in video:
                queue.update(v)

            while len(queue) >= batch_size:
                hashes, ims = [], []

                for _ in range(batch_size):
                    im_hash, im = queue.popitem()
                    hashes.append(im_hash)
                    ims.append(im)

                batch = torch.stack(ims).to(device)
                h, w = np.array(batch.shape[-2:]) // patch_size

                # Tuple of last_hidden_state, pooler_output, hidden_states, attentions.
                # All except last_hidden_state is optional, so the tuple length
                # varies depending on ViTConfig.
                o = model(batch)

                pats_t = o[0]
                attns_t = o[-1][-1]

                pats_t = E.rearrange(pats_t[:, 1:], "t (h w) c -> t c h w", h=h, w=w)
                # Get attention weights for the CLS token (token 0).
                attns_t = E.rearrange(
                    attns_t[:, :, 0, 1:], "t n (h w) -> t n h w", h=h, w=w
                )

                # Prevent memory leak.
                if not semaphore.acquire(block=False):
                    log.debug("Waiting for write pool to free up...")
                    semaphore.acquire(block=True)
                    log.debug("Semaphore acquired.")
                future = pool.submit(
                    cache.put_vid,
                    hashes,
                    {CACHE_PATCHES: pats_t.cpu(), CACHE_LAST_ATTNS: attns_t.cpu()},
                )
                future.add_done_callback(done)
