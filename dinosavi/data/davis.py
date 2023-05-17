"""DAVIS dataloader."""

import logging
from pathlib import Path
from typing import Callable, Tuple

from .vos import VOSDataset

__all__ = ["DAVISDataset"]

log = logging.getLogger(__name__)


class DAVISDataset(VOSDataset):
    """DAVIS Dataset."""

    def __init__(
        self,
        root: str,
        year: str = "2017",
        split: str = "val",
        quality: str = "480p",
        im_size: int | Tuple[int, int] = -1,
        transform: Callable = lambda x: x,
        map_scale: int = 1,
        context_len: int = 20,
    ):
        """Initialize DAVIS dataset."""
        im_dirs = []
        lbl_dirs = []

        r = Path(root).resolve()
        imageset_txt = r / "ImageSets" / year / f"{split}.txt"
        with imageset_txt.open() as f:
            videos = [l for s in f.readlines() if (l := s.strip()) != ""]
            self.videos = videos

        for video in videos:
            lbl_dir = r / "Annotations" / quality / video
            im_dir = r / "JPEGImages" / quality / video
            im_paths = sorted(im_dir.glob("*.jpg"), key=lambda p: int(p.stem))
            lbl_paths = sorted(lbl_dir.glob("*.png"), key=lambda p: int(p.stem))
            im_dirs.append((str(im_dir), [str(p) for p in im_paths]))
            if len(lbl_paths) > 0:
                lbl_dirs.append((str(lbl_dir), [str(p) for p in lbl_paths]))

        super().__init__(
            im_dirs, lbl_dirs, im_size, transform, map_scale, context_len, True
        )
