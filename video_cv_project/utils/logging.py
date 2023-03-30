"""Utility functions for logging."""

import json
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm

__all__ = ["pretty", "get_dirs", "iter_pbar", "confirm_ask", "get_model_summary"]


class TaskSpeed(ProgressColumn):
    def render(self, task):
        return f"[yellow]{( task.speed or 0.0 ):.3g} it/s"


def pretty(obj):
    """Pretty format object using JSON."""
    return json.dumps(obj, sort_keys=True, indent=2)


def get_dirs():
    """Get root and output directory."""
    root_dir = Path(HydraConfig.get().runtime.cwd)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    return root_dir, out_dir


iter_pbar = Progress(
    SpinnerColumn("moon"),
    "{task.description}",
    BarColumn(),
    "[green]{task.completed}/{task.total}",
    TaskSpeed(),
    TimeRemainingColumn(elapsed_when_finished=True),
    "{task.fields[status]}",
    transient=True,
)


def confirm_ask(*args, pbar=iter_pbar, **kwargs):
    """Workaround to stop prompt from getting overwritten by progress bar."""
    pbar.stop()
    res = Confirm.ask(*args, **kwargs)
    pbar.start()
    return res


SIZES = (
    (1, 8, 384, 28, 28),
    (1, 8, 49, 3, 64, 64),
    (1, 8, 3, 224, 224),
)


def get_model_summary(model, sizes=SIZES, device=None):
    """Get model summary."""
    from torchinfo import summary

    for size in sizes:
        try:
            model_summary = summary(model, size, verbose=0, col_width=20, device=device)
        except Exception as e:
            # raise e
            continue
        model_summary.formatting.layer_name_width = 30
        return model_summary
    assert False, "No supported size!"
