"""ResNet encoder.

Attributes:
    SUPPORTED_RESNET_VARIANTS (dict): Map of supported resnet variants to their weights.
"""

import logging
from types import EllipsisType
from typing import Sequence

import torch.nn as nn
from torchvision.models import get_model
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

from video_cv_project.utils import delete_layers, find_layers, override_forward

from .video_wrapper import VideoWrapper

__all__ = ["SUPPORTED_RESNET_VARIANTS", "create_resnet_encoder"]

log = logging.getLogger(__name__)

# TODO: check modifications work on all resnet variants.
# Weights are fixed for reproducibility.
SUPPORTED_RESNET_VARIANTS = {
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": ResNet34_Weights.IMAGENET1K_V1,
    "resnet50": ResNet50_Weights.IMAGENET1K_V2,
}


def create_resnet_encoder(
    name: str = "resnet18",
    weights: str | EllipsisType | None = ...,
    del_layers: Sequence[str] = [],
    **kwargs,
) -> nn.Module:
    """Creates ResNet model and modifies it to act as an encoder.

    If ``weights`` is ``...``, default weights for the model will be used. Else
    if ``weights`` is ``None``, no weights will be loaded.

    `torchvision.models.get_model` is used to create the model. It passes ``weights``
    and ``**kwargs`` to the respective model builder. For example, `torchvision.models.resnet18`.

    Args:
        name (str, optional): ResNet model to use. Defaults to "resnet18".
        weights (str | Ellipsis | None, optional): Model weights to use. Defaults to ``...``.
        del_layers (Sequence[str], optional): Additional layers to delete. Defaults to [].
        **kwargs: Additional arguments to pass to `torchvision.models.get_model`.

    Returns:
        torch.nn.Module: ResNet encoder.
    """
    if name not in SUPPORTED_RESNET_VARIANTS:
        log.warning(
            f"{name} isn't explicitly supported. Pretrained weights may change depending on torchvision `DEFAULT` version."
        )

    weights = SUPPORTED_RESNET_VARIANTS.get(name, "DEFAULT") if weights is ... else weights
    # TODO: When using empty model, init parameters are random (should not be for reproducibility).
    model = get_model(name, weights=weights, **kwargs)
    # print(weights, next(model.parameters()).data.sum())

    # Reduce stride of layer3 & layer4 to 1.
    for layer in find_layers(model, ("layer3", "layer4")):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                m.stride = tuple(1 for _ in m.stride)

    delete_layers(model, [*del_layers, "fc", "avgpool"])

    # As flatten op between fc & avgpool cannot be disabled, override forward pass.
    def _forward(self, x):
        # Copied from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    override_forward(model, _forward)

    return VideoWrapper(model)
