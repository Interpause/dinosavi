# video-cv-project

> Come up with name later

**Note**: <https://hydra.cc/> is used as the config system, see <https://hydra.cc/docs/advanced/override_grammar/basic/> for CLI overrides.

## Installation

```sh
poetry install
```

## Training

```sh
python -m video_cv_project mode=train hparam=... exp_name=...
```

## Evaluation

```sh
python -m video_cv_project mode=eval resume=... exp_name=...
```

## Packaging

- See <https://python-poetry.org/docs/pyproject/#extras> for adding `pip install package[extra]` support to built package.

## Style Guide

- We use [Black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort).
