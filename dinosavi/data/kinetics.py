"""Kinetics dataloader for unsupervised training."""

import logging

import torch
import torchvision.transforms as T
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import Kinetics

from dinosavi.utils import seed_data

from .transform import create_train_pipeline

__all__ = ["create_kinetics_dataloader"]

log = logging.getLogger(__name__)


def collate(batch):
    """`torchvision.datasets.video_utils.VideoClips` returns metadata along with video tensor. Select video tensor & stack into batch."""
    # Special treatment when not batched (used in cache mode).
    if not isinstance(batch, list):
        return [batch[0]]

    # See https://github.com/pytorch/vision/blob/707457050620e1f70ab1b187dad81cc36a7f9180/torchvision/datasets/video_utils.py#L289.
    batch = [c[0] for c in batch]

    # Special treatment for batched cache mode.
    if isinstance(batch[0], dict):
        return batch
    # `default_collate` searches for and stacks tensors while preserving data structure.
    return default_collate(batch)


def _get_transform(transform, patcher=None):
    """Get transform function for `Kinetics` dataset."""
    if patcher:
        return T.Compose([transform, patcher])
    return transform


def create_kinetics_dataloader(cfg: DictConfig) -> DataLoader:
    """Create dataloader for Kinetics dataset."""
    meta = None
    if cfg.data.cache_path:
        try:
            cache: Kinetics = torch.load(cfg.data.cache_path)
            meta = dict(
                video_paths=cache.video_clips.video_paths,
                video_fps=cache.video_clips.video_fps,
                video_pts=cache.video_clips.video_pts,
            )
            log.info(f"Dataset Cache: {cfg.data.cache_path}")
        except FileNotFoundError:
            pass

    transform = (
        instantiate(cfg.data.transform.pipeline, _convert_="all")
        if cfg.data.transform.pipeline
        else create_train_pipeline()
    )
    patch_func = (
        instantiate(cfg.data.transform.patch_func, _convert_="all")
        if cfg.data.transform.patch_func
        else None
    )
    log.info(f"Pipeline:\n{transform}")
    log.info(f"Patch Function: {patch_func}")

    try:
        dataset: Kinetics = instantiate(
            cfg.data.dataset, _precomputed_metadata=meta, download=True, _convert_="all"
        )
    except:
        dataset = instantiate(
            cfg.data.dataset, _precomputed_metadata=meta, _convert_="all"
        )

    torch.save(dataset, cfg.data.cache_path)
    # Don't save transform into cache else loading may fail.
    dataset.transform = _get_transform(transform, patch_func)

    log.info(f"Total Videos: {dataset.video_clips.num_videos()}")
    log.info(f"Total Clips: {len(dataset)}")

    sampler = instantiate(
        cfg.data.sampler, video_clips=dataset.video_clips, _convert_="all"
    )

    dataloader = instantiate(
        cfg.data.dataloader,
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate,
        _convert_="all",
        **seed_data(),
    )
    return dataloader
