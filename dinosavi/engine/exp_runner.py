"""Handles iterating dataloader and logging."""

import logging
import signal
from multiprocessing import parent_process
from time import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dinosavi.utils import confirm_ask, iter_pbar

__all__ = ["ExpRunner"]

log = logging.getLogger(__name__)

TENSORBOARD_DIR = "."


class ExpRunner:
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
        """Initialize ExpRunner."""
        self.dataloader = dataloader
        self.epochs = epochs
        self.pbar = pbar
        self.logger = logger
        self.log_every = log_every if log_every > 0 else float("inf")
        self.save_func = save_func
        self.save_every = save_every if save_every > 0 else float("inf")

        self._etask = pbar.add_task("Epoch", total=epochs, status="")
        self._itask = pbar.add_task("Iteration", total=len(dataloader), status="")
        self._stat: dict = dict(epoch=0, iteration=0)
        self.tbwriter = SummaryWriter(log_dir=TENSORBOARD_DIR)

    def __iter__(self):
        """Iterate over dataloader.

        Yields:
            Tuple[int, int, torch.Tensor]: Epoch, iteration, and batch.
        """
        is_interrupted = False
        orig_handler = signal.getsignal(signal.SIGINT)

        def set_interrupted(sig, frame):
            if parent_process() is not None:
                orig_handler(sig, frame)
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

        # I verified that `__iter__` is only called in the main process already.
        # So signal handler being registered for child processes is Python's behavior.
        signal.signal(signal.SIGINT, set_interrupted)

        try:
            for i in range(1, self.epochs + 1):
                self.pbar.reset(self._itask)
                data_t, infer_t = 0.0, 0.0
                t_data, t_infer = time(), 0.0
                for n, data in enumerate(self.dataloader, start=1):
                    self.pbar.start()
                    data_t = time() - t_data

                    t_infer = time()
                    yield i, n, data
                    infer_t = time() - t_infer

                    self.update(epoch=i, iteration=n, data_t=data_t, infer_t=infer_t)
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

                    t_data = time()

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
            self.tbwriter.close()

    def update(self, **kwargs):
        """Update training metrics & other status."""
        self._stat.update(kwargs)

    def _log(self):
        i = self._stat["epoch"]
        n = self._stat["iteration"]
        step = (i - 1) * len(self.dataloader) + (n - 1)

        stats = []
        for k, v in self._stat.items():
            # Workaround for special visualizations to log.
            if type(v) == tuple and tuple(map(type, v)) == (str, dict):
                if n % self.log_every == 0:
                    cmd, kwargs = v
                    func = getattr(self.tbwriter, cmd)
                    func(tag=k, global_step=step, **kwargs)
                continue

            self.tbwriter.add_scalar(k, v, step)
            if k in {"epoch", "iteration"}:
                continue
            # Add other formats as needed.
            if isinstance(v, float):
                stats.append(f"{k}: {v:.4g}")
            else:
                stats.append(f"{k}: {v}")
        stat_str = ", ".join(stats)
        self.pbar.update(self._itask, status=f"{stat_str[:50]}")

        if n % self.log_every == 0 or n == len(self.dataloader):
            self.tbwriter.flush()
            self.logger.info(
                f"epoch: {i}/{self.epochs}, iteration: {n}/{len(self.dataloader)}, {stat_str}"
            )
