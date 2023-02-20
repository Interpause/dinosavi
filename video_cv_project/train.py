"""TODO: Add module docstring."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from video_cv_project.data import create_kinetics400_dataloader
from video_cv_project.models import CRW


@hydra.main(version_base="1.3", config_path="cfg", config_name="train")
def train(cfg: DictConfig):
    """Train model."""
    transform = instantiate(cfg.train_transform)
    dataloader = create_kinetics400_dataloader(transform)
    model: CRW = instantiate(cfg.model)
    print(transform, model)


if __name__ == "__main__":
    train()
