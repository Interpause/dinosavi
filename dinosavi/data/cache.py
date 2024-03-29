"""Data Caching."""


from io import BytesIO
from lzma import PRESET_EXTREME, compress, decompress
from multiprocessing import current_process
from pathlib import Path
from pickle import HIGHEST_PROTOCOL
from typing import Dict, List, Tuple

import torch
from diskcache import UNKNOWN, Cache, Disk, FanoutCache
from torch.utils.data import default_collate

from dinosavi.cfg import CACHE_LAST_ATTNS, CACHE_PATCHES
from dinosavi.utils import hash_tensor

__all__ = ["TensorCache"]


class TorchDisk(Disk):
    """Use `torch.save` to serialize & load tensors."""

    def store(self, value, read, key=UNKNOWN):
        """Override."""
        # Pickle: 2.12 ms ± 53 µs
        # Torch (zipfile): 3.46 ms ± 20.5 µs
        # Torch (legacy): 265 µs ± 2.86 µs
        # Legacy format much smaller for single tensors.
        if not read and isinstance(value, torch.Tensor):
            buf = BytesIO()
            # Halving reduces the size by over 700 times even before compression. IDK why.
            torch.save(
                value.half(),
                buf,
                pickle_protocol=self.pickle_protocol,
                _use_new_zipfile_serialization=False,
            )
            # XZ has compression ratio of 1.1 after halving.
            # Surprisingly, -0e and -9e have the same size, suggesting the data
            # is compressible in a very specific manner. -0e is much faster though.
            value = compress(buf.getbuffer(), preset=0 | PRESET_EXTREME)
        return super(TorchDisk, self).store(value, read, key)

    def fetch(self, mode, filename, value, read):
        """Override."""
        ori = data = super(TorchDisk, self).fetch(mode, filename, value, read)
        if not read and isinstance(data, bytes):
            try:
                data = decompress(data)
                data = torch.load(BytesIO(data), map_location="cpu").float()
            except:
                return ori
        return data


class TensorCache:
    """Checks if results are already cached."""

    # https://grantjenks.com/docs/diskcache/api.html?highlight=default_settings#diskcache.diskcache.DEFAULT_SETTINGS
    SETTINGS = dict(
        # shards=16,
        disk=TorchDisk,
        statistics=0,
        tag_index=0,
        eviction_policy="none",
        # size_limit=1099511627776,  # 1 TiB; `eviction_policy` of "none" ignores this.
        cull_limit=0,
        # disk_min_file_size=77070336,  # 9408 KiB
        disk_pickle_protocol=HIGHEST_PROTOCOL,
    )

    VID_LEVEL_CACHE = True

    def __init__(
        self,
        model_hash: str,
        cache_dir: Path | str | None = None,
        attrs=[CACHE_PATCHES, CACHE_LAST_ATTNS],
    ):
        """Create CacheCheck."""
        self.model_hash = model_hash
        self.cache_dir = Path(".cache" if cache_dir is None else cache_dir) / model_hash
        self.attrs = attrs

        # self.cache = {
        #     attr: FanoutCache(self.cache_dir / attr, **self.SETTINGS) for attr in attrs
        # }
        self.cache = {
            attr: Cache(self.cache_dir / attr, **self.SETTINGS) for attr in attrs
        }

    def get_val(self, im_hash: str, attr: str) -> torch.Tensor | None:
        """Get associated key and value if already cached."""
        assert attr in self.attrs
        v = self.cache[attr].get(im_hash, default=None, retry=True)
        # if v is not None:
        #     print(f"{current_process().name} LOAD: {attr}/{im_hash}")
        return v

    def put_val(self, im_hash: str, attr: str, val: torch.Tensor):
        """Put tensor value into cache."""
        # print(f"{current_process().name} SAVE: {attr}/{im_hash}")
        val = val.detach().cpu().requires_grad_(False)
        self.cache[attr].set(im_hash, val, retry=True)

    def get_vid(
        self, vid: torch.Tensor
    ) -> Tuple[List[str], Dict[str, torch.Tensor] | None]:
        """Cache video on frame-level."""
        if self.VID_LEVEL_CACHE:
            vid_hash = hash_tensor(vid)
            out = {a: self.get_val(vid_hash, a) for a in self.attrs}
            miss = None in out.values()
            return [vid_hash], None if miss else out  # type: ignore

        hashes, out_t, miss = [], [], False
        for im in vid:
            im_hash = hash_tensor(im)
            out = {a: self.get_val(im_hash, a) for a in self.attrs}
            hashes.append(im_hash)
            out_t.append(out)
            if None in out.values():
                miss = True
        return hashes, None if miss else default_collate(out_t)

    def put_vid(self, hashes: List[str], data: Dict[str, torch.Tensor]):
        """Cache video on frame-level."""
        for k, v in data.items():
            for i, im_hash in enumerate(hashes):
                if all(im_hash in c for c in self.cache.values()):
                    continue
                self.put_val(im_hash, k, v[i])

    def __call__(self, vid: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        """Check if frames of video already in cache.

        Args:
            vid (torch.Tensor): TCHW images.

        Returns:
            Dict[str, torch.Tensor | None]: Hashes and images not in cache.
        """
        results: dict = {}

        if self.VID_LEVEL_CACHE:
            vid_hash = hash_tensor(vid)
            found = all(vid_hash in self.cache[attr] for attr in self.attrs)
            if not found:
                results[vid_hash] = vid
            return results

        for im in vid:
            im_hash = hash_tensor(im)
            found = all(im_hash in self.cache[attr] for attr in self.attrs)
            # Below can be used to verify integrity.
            # found = all(
            #     self.cache[attr].get(im_hash) is not None for attr in self.attrs
            # )
            if not found:
                results[im_hash] = im
        return results

    def __repr__(self):
        """Return string representation of class."""
        return f"{self.__class__.__name__}(model_hash={self.model_hash}, cache_dir={self.cache_dir}, keys={self.attrs})"
