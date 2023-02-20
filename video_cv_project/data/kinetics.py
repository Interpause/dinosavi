"""TODO: Add module docstring."""

import logging

import torch
import torchvision
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets.samplers import RandomClipSampler

from video_cv_project.data.transform import create_train_pipeline

__all__ = ["create_kinetics400_dataloader"]

log = logging.getLogger(__name__)


def collate(batch):
    """`torchvision.datasets.video_utils.VideoClips` returns metadata along with video tensor. Select video tensor & stack into batch."""
    # See https://github.com/pytorch/vision/blob/707457050620e1f70ab1b187dad81cc36a7f9180/torchvision/datasets/video_utils.py#L289.
    batch = [c[0] for c in batch]
    # `default_collate` searches for and stacks tensors while preserving data structure.
    return default_collate(batch)


def create_kinetics400_dataloader(transform=None):
    """Create dataloader for Kinetics400 dataset."""
    cache_path = "datasets/kinetics400.pt"
    dataset_path = "datasets/kinetics400/"
    clips_per_video = 4

    transform = transform if transform else create_train_pipeline()
    rng = torch.manual_seed(42)

    try:
        dataset = torch.load(cache_path)
        cached_metadata = dict(
            video_paths=dataset.video_clips.video_paths,
            video_fps=dataset.video_clips.video_fps,
            video_pts=dataset.video_clips.video_pts,
        )
    except FileNotFoundError:
        dataset = None
        cached_metadata = None

    kwargs = dict(
        root=dataset_path,
        frames_per_clip=4,
        num_classes="400",
        split="val",
        frame_rate=8,
        step_between_clips=8,
        download=True,
        transform=transform,
        num_workers=16,
        num_download_workers=16,
        output_format="TCHW",
        _precomputed_metadata=cached_metadata,
    )
    try:
        dataset = torchvision.datasets.Kinetics(**kwargs)
    except:
        dataset = torchvision.datasets.Kinetics(**{**kwargs, "download": False})

    torch.save(dataset, cache_path)

    # Carry over for limiting dataset size.
    subset_idx = list(
        torch.randperm(dataset.video_clips.num_videos(), generator=rng)[:5000]
    )

    sampler = RandomClipSampler(dataset.video_clips.subset(subset_idx), clips_per_video)
    dataloader = DataLoader(
        dataset,
        batch_size=6,
        sampler=sampler,
        num_workers=16,
        collate_fn=collate,
        pin_memory=True,
        generator=rng,
    )

    log.info(f"Total videos: {dataset.video_clips.num_videos()}")
    log.info(f"Total clips: {len(dataset)}")
    log.info(f"Filtered clips: {len(dataloader) * dataloader.batch_size}")

    return dataloader
