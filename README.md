# DINOSAVi

> Self-supervised learning of Video Object Segmentation using DINOSAUR and SAVi

**Presentation Slides**: <https://1drv.ms/p/s!AgE9E4ZerfvahsdfaYUe8caynzF0iw>

![slot-s256d-line-t01-tline-ini5](https://github.com/Interpause/dinosavi/assets/42513874/625d61a7-6f5f-4f43-aa4d-f11d812eef43)
> VOS shown above is from `slot-s256d-line-t01-tline-ini5` model

Models: <https://1drv.ms/f/s!AgE9E4Zerfvahsd8TTjyDmHvikwY6g>

**Note**: <https://hydra.cc/> is used to manage configuration, CLI and experiment running. See <https://hydra.cc/docs/advanced/override_grammar/basic/> for CLI override grammar. Jump to [Codebase Notes](#codebase-notes) for more info.

## Installation

```sh
# (Optional) Use conda for virtual environment instead. Poetry creates venv by default.
conda create -n dinosavi python=3.10
poetry install
```

## Training

```sh
python -m dinosavi mode=train hparam={hparam_file_name} exp_name={experiment_name}
```

## Evaluation

```sh
python -m dinosavi mode=eval resume={.ckpt_to_load} exp_name={results_name} device={cpu_or_cuda}
```

## Codebase Notes

- [Hydra](https://hydra.cc/) (built on [OmegaConf](https://omegaconf.readthedocs.io/)) is used for configuration management, CLI and experiment running.
  - For CLI options, you can refer to [`dinosavi/cfg/main.yaml`](dinosavi/cfg/main.yaml) and [`dinosavi/cfg/mode`](dinosavi/cfg/mode/).
  - Hyperparameter config files are stored in [`dinosavi/cfg/hparam`](dinosavi/cfg/hparam/).
    - [OmegaConf Variable Interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation) is heavily used to link hyperparameter values into relevant config files.
  - Many parts of the code (i.e., models, datasets, trainers) are implemented as config files using [Hydra's Instantiate API](https://hydra.cc/docs/advanced/instantiate_objects/overview/).
- We use [Black](https://github.com/psf/black) for formatting, [isort](https://github.com/PyCQA/isort) for import sorting, and Google-style docstrings.
- Anything to do with [Contrastive Random Walk](https://ajabri.github.io/videowalk/) (CRW) is leftover legacy code from earlier experiments.
