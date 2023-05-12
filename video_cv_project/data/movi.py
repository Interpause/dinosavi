"""MOVi dataloader."""

import logging
from pathlib import Path
from typing import Callable, Tuple

from .vos import VOSDataset

__all__ = ["MOViDataset"]

log = logging.getLogger(__name__)


class MOViDataset(VOSDataset):
    """MOVi Dataset."""

    def __init__(
        self,
        root: str,
        name: str = "movi_a",
        split: str = "validation",
        im_size: int | Tuple[int, int] = -1,
        transform: Callable = lambda x: x,
        map_scale: int = 1,
        context_len: int = 20,
    ):
        """Initialize MOVi dataset."""
        im_dirs = []
        lbl_dirs = []

        r = Path(root).resolve() / name / split
        self.videos = [p.name for p in (r / "rgb").glob("*") if p.is_dir()]

        for video in self.videos:
            lbl_dir = r / "seg" / video
            im_dir = r / "rgb" / video
            im_paths = sorted(im_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            lbl_paths = sorted(lbl_dir.glob("*.png"), key=lambda p: int(p.stem))
            im_dirs.append((str(im_dir), [str(p) for p in im_paths]))
            if len(lbl_paths) > 0:
                lbl_dirs.append((str(lbl_dir), [str(p) for p in lbl_paths]))

        super().__init__(
            im_dirs, lbl_dirs, im_size, transform, map_scale, context_len, True
        )
