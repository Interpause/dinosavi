"""Data Caching."""


from io import BytesIO
from multiprocessing import current_process
from typing import Dict, List, Tuple

import torch
from diskcache import Cache

from video_cv_project.cfg import CACHE_LAST_ATTNS, CACHE_PATCHES
from video_cv_project.utils import hash_tensor

__all__ = ["TensorCache"]


class TensorCache:
    """Checks if results are already cached."""

    def __init__(
        self,
        model_hash: str,
        cache_dir: str = None,
        attrs=[CACHE_PATCHES, CACHE_LAST_ATTNS],
    ):
        """Create CacheCheck."""
        self.model_hash = model_hash
        self.cache_dir = cache_dir
        self.cache = Cache(cache_dir)
        self.attrs = attrs

    @staticmethod
    def get_key(model_hash: str, tensor_hash: str, attr: str):
        """Get standardized cache key."""
        return f"{model_hash}/{attr}/{tensor_hash}"

    def get_val(self, im_hash: str, attr: str) -> Tuple[str, torch.Tensor | None]:
        """Get associated key and value if already cached."""
        assert attr in self.attrs
        k = self.get_key(self.model_hash, im_hash, attr)
        v = self.cache.get(k, default=None, read=True)
        if v is not None:
            # print(f"{current_process().name} LOAD: {k}")
            v = torch.load(v, weights_only=True, map_location="cpu")
        return k, v

    def put_val(self, im_hash: str, attr: str, val: torch.Tensor):
        """Put tensor value into cache."""
        k = self.get_key(self.model_hash, im_hash, attr)
        # print(f"{current_process().name} SAVE: {k}")
        buf = BytesIO()
        val = val.detach().cpu().requires_grad_(False)
        torch.save(val, buf)
        buf.seek(0)
        self.cache.set(k, buf, read=True, tag=self.model_hash)
        return k

    def get_vid(
        self, vid: torch.Tensor
    ) -> Tuple[List[str | None], torch.Tensor | None, torch.Tensor | None]:
        """Cache video on frame-level."""
        hashes, pats_t, attns_t = [], [], []
        for im in vid:
            im_hash = hash_tensor(im)
            _, pats = self.get_val(im_hash, CACHE_PATCHES)
            _, attns = self.get_val(im_hash, CACHE_LAST_ATTNS)
            hashes.append(im_hash if None in (pats, attns) else None)
            pats_t.append(pats)
            attns_t.append(attns)
        miss = None in pats_t or None in attns_t
        pats_t = None if miss else torch.stack(pats_t)  # type: ignore
        attns_t = None if miss else torch.stack(attns_t)  # type: ignore
        return hashes, pats_t, attns_t

    def put_vid(
        self, hashes: List[str | None], pats_t: torch.Tensor, attns_t: torch.Tensor
    ):
        """Cache video on frame-level."""
        for im_hash, pats, attns in zip(hashes, pats_t, attns_t):
            # If None, it is already in cache.
            if im_hash is None:
                continue
            self.put_val(im_hash, CACHE_PATCHES, pats)  # CHW
            self.put_val(im_hash, CACHE_LAST_ATTNS, attns)  # NHW

    def __call__(self, vid: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        """Check if frames of video already in cache.

        Args:
            vid (torch.Tensor): TCHW images.

        Returns:
            Dict[str, torch.Tensor | None]: Hashes and images not in cache.
        """
        results = {}
        for im in vid:
            im_hash = hash_tensor(im)
            found = all(
                self.get_key(self.model_hash, im_hash, k) in self.cache
                for k in self.attrs
            )
            if not found:
                results[im_hash] = im
        return results

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(model_hash={self.model_hash}, cache_dir={self.cache_dir}, keys={self.attrs})"
