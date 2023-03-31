"""Main entrypoint."""

import logging
import multiprocessing
import warnings

import hydra
from omegaconf import DictConfig

from video_cv_project.utils.logging import get_dirs

log = logging.getLogger("video_cv_project")

# warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base="1.3", config_path="cfg", config_name="main")
def main(cfg: DictConfig):
    """Entrypoint for everything."""
    root_dir, out_dir = get_dirs()

    log.info(f"Launch Mode: {cfg.mode}")
    log.info(f"Root Directory: {root_dir}")
    log.info(f"Output Directory: {out_dir}")
    log.debug(f"Full Config:\n{cfg}")

    try:
        if cfg.mode == "train":
            from video_cv_project.train import train
            from video_cv_project.utils import perf_hack

            perf_hack()
            train(cfg)
        elif cfg.mode == "eval":
            from video_cv_project.eval import eval
            from video_cv_project.utils import perf_hack

            perf_hack()
            eval(cfg)
        elif cfg.mode == "eval-slot":
            from video_cv_project.eval_slot import eval
            from video_cv_project.utils import perf_hack

            perf_hack()
            eval(cfg)
        elif cfg.mode == "cache":
            from video_cv_project.cache import cache
            from video_cv_project.utils import perf_hack

            perf_hack()
            cache(cfg)
        elif cfg.mode == "help":
            log.critical("Add `--help` for help message.")
        else:
            log.critical(f"Unsupported mode: {cfg.mode}")
    except KeyboardInterrupt:
        log.warning("Exiting due to Keyboard Interrupt.")
    except Exception as e:
        log.critical("Exiting due to error.", exc_info=e)
        raise e


if __name__ == "__main__":
    # Support CUDA tensors in multiprocessing/safer anyways.
    if "forkserver" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("forkserver")
    else:
        multiprocessing.set_start_method("spawn")
    multiprocessing.freeze_support()
    main()
