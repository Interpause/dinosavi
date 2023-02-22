"""TODO: Add module docstring."""

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

__all__ = ["pretty", "get_dirs", "iter_pbar"]


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
)