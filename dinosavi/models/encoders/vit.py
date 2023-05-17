"""Modified HuggingFace ViTModel and ViTEncoder to return only last layer's attention."""

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vit.modeling_vit import (
    ViTConfig,
    ViTEmbeddings,
    ViTEncoder,
    ViTModel,
    ViTPooler,
)

__all__ = ["ViTLastAttnModel"]


class ViTLastAttnEncoder(ViTEncoder):
    """Refer to HuggingFace documentation."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        """Refer to HuggingFace documentation."""
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attentions = () if output_attentions else None

        last_attns = None
        for i, layer_module in enumerate(self.layer):
            output_last_layer_attns = i + 1 == len(self.layer) and output_attentions
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # return module(*inputs, output_attentions)
                        return module(*inputs, output_last_layer_attns)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                # layer_outputs = layer_module(
                #     hidden_states, layer_head_mask, output_attentions
                # )
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, output_last_layer_attns
                )

            hidden_states = layer_outputs[0]

            # if output_attentions:
            #     all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if output_last_layer_attns:
                last_attns = layer_outputs[1]

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        all_hidden_states = None
        all_self_attentions = (last_attns,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTLastAttnModel(ViTModel):
    """Refer to HuggingFace documentation."""

    def __init__(
        self,
        config: ViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        """Refer to HuggingFace documentation."""
        # I hate Object-Oriented Programming.
        super(ViTModel, self).__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        # self.encoder = ViTEncoder(config)
        self.encoder = ViTLastAttnEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()
