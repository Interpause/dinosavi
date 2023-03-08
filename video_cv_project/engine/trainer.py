"""Handles iterating dataloader and logging."""

import logging
import signal
from multiprocessing import parent_process

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from video_cv_project.utils import confirm_ask, iter_pbar

__all__ = ["Trainer"]

log = logging.getLogger(__name__)

TENSORBOARD_DIR = "tensorboard"


class Trainer:
    """Handles iterating dataloader and logging."""

    def __init__(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        pbar=iter_pbar,
        logger=log,
        log_every: int = 20,
        save_func=lambda i, n: None,
        save_every: int = 500,
    ):
        """Initialize Trainer."""
        self.dataloader = dataloader
        self.epochs = epochs
        self.pbar = pbar
        self.logger = logger
        self.log_every = log_every
        self.save_func = save_func
        self.save_every = save_every

        self._etask = pbar.add_task("Epoch", total=epochs, status="")
        self._itask = pbar.add_task("Iteration", total=len(dataloader), status="")
        self._stat: dict = dict(epoch=0, iteration=0)
        self._tbwriter = SummaryWriter(log_dir=TENSORBOARD_DIR)

    def __iter__(self):
        """Iterate over dataloader.

        Yields:
            Tuple[int, int, torch.Tensor]: Epoch, iteration, and batch.
        """
        is_interrupted = False

        def set_interrupted(sig, frame):
            if parent_process() is not None:
                return
            nonlocal is_interrupted
            is_interrupted = True
            log.debug("Interrupt Point:", stack_info=frame, stacklevel=2)
            if confirm_ask(
                "Interrupt immediately (or wait for current batch)?",
                default=False,
                pbar=self.pbar,
            ):
                self.save_func(self._stat["epoch"], self._stat["iteration"])
                raise KeyboardInterrupt

        orig_handler = signal.signal(signal.SIGINT, set_interrupted)

        try:
            self.pbar.start()
            for i in range(1, self.epochs + 1):
                self.pbar.reset(self._itask)
                for n, data in enumerate(self.dataloader, start=1):
                    yield i, n, data

                    self.update(epoch=i, iteration=n)
                    self.pbar.advance(self._itask)
                    self._log()

                    if n % self.save_every == 0:
                        self.save_func(i, n)

                    if is_interrupted:
                        self.save_func(i, n)
                        if not confirm_ask(
                            "Continue training?", default=True, pbar=self.pbar
                        ):
                            raise KeyboardInterrupt
                        is_interrupted = False

                self.save_func(i, n)
                self.pbar.advance(self._etask)

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        # Save on unexpected errors.
        except Exception as e:
            self.save_func(self._stat["epoch"], self._stat["iteration"])
            raise e
        finally:
            signal.signal(signal.SIGINT, orig_handler)
            self.pbar.stop()
            self._tbwriter.close()

    def update(self, **kwargs):
        """Update training metrics & other status."""
        self._stat.update(kwargs)

    def _log(self):
        i = self._stat["epoch"]
        n = self._stat["iteration"]

        self._tbwriter.add_scalars(
            "train", self._stat, (i - 1) * len(self.dataloader) + (n - 1)
        )

        stats = []
        for k, v in self._stat.items():
            if k in {"epoch", "iteration"}:
                continue
            # Add other formats as needed.
            if isinstance(v, float):
                stats.append(f"{k}: {v:.6g}")
            else:
                stats.append(f"{k}: {v}")
        stat_str = ", ".join(stats)
        self.pbar.update(self._itask, status=stat_str)

        if n % self.log_every == 0 or n == len(self.dataloader):
            self._tbwriter.flush()
            self.logger.info(
                f"epoch: {i}/{self.epochs}, iteration: {n}/{len(self.dataloader)}, {stat_str}"
            )
