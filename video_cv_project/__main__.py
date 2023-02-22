"""TODO: Add module docstring."""

import logging
import warnings

import hydra
from omegaconf import DictConfig

from video_cv_project.utils import get_dirs

log = logging.getLogger("video_cv_project")

warnings.filterwarnings("ignore", category=UserWarning)


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

            train(cfg)
        elif cfg.mode == "help":
            log.critical("Add `--help` for help message.")
    except KeyboardInterrupt:
        log.warning("Exiting due to Keyboard Interrupt.")
    except Exception as e:
        log.critical("Exiting due to error.", exc_info=e)
        raise e


if __name__ == "__main__":
    main()