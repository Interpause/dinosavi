"""Kinetics dataloader for unsupervised training."""

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import Kinetics

from .transform import create_train_pipeline

__all__ = ["create_kinetics_dataloader"]

log = logging.getLogger(__name__)


def collate(batch):
    """`torchvision.datasets.video_utils.VideoClips` returns metadata along with video tensor. Select video tensor & stack into batch."""
    # See https://github.com/pytorch/vision/blob/707457050620e1f70ab1b187dad81cc36a7f9180/torchvision/datasets/video_utils.py#L289.
    batch = [c[0] for c in batch]
    # `default_collate` searches for and stacks tensors while preserving data structure.
    return default_collate(batch)


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
        instantiate(cfg.data.transform.pipeline)
        if cfg.data.transform
        else create_train_pipeline()
    )
    log.info(f"Pipeline:\n{transform}")

    try:
        dataset: Kinetics = instantiate(
            cfg.data.dataset, _precomputed_metadata=meta, download=True
        )
    except:
        dataset = instantiate(cfg.data.dataset, _precomputed_metadata=meta)

    torch.save(dataset, cfg.data.cache_path)
    # Don't save transform into cache else loading may fail.
    dataset.transform = transform

    log.info(f"Total Videos: {dataset.video_clips.num_videos()}")
    log.info(f"Total Clips: {len(dataset)}")

    sampler = instantiate(cfg.data.sampler, video_clips=dataset.video_clips)

    dataloader = instantiate(
        cfg.data.dataloader,
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate,
    )
    return dataloader
