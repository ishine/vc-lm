import torch
import math
import random
from torch import nn
from typing import List, Optional, Union, Tuple

import pytorch_lightning as pl

from whisper.model import AudioEncoder

from vc_lm.models.base import VCLMPretrainedModel, VCLMConfig

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)

from vc_lm.models.bart.modeling_bart import BartLearnedPositionalEmbedding, BartEncoderLayer, _expand_mask
from vc_lm.datamodules.datasets.ar_dataset import ARDataset

class ContentEncoder(VCLMPretrainedModel):
    def __init__(self, config: VCLMConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = 1500

        self.embed_positions = BartLearnedPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.content_layer_num)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
                inputs: torch.FloatTensor = None,
                attention_mask: Optional[torch.Tensor] = None):
        embed_pos = self.embed_positions(inputs)
        embed_pos = embed_pos.to(inputs.device)

        hidden_states = inputs + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs.dtype)

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attention=None)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        None,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=None,
                        output_attentions=None,
                    )

                hidden_states = layer_outputs[0]
        return hidden_states


class WhisperEncoder(VCLMPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: VCLMConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        checkpoint = torch.load(config.encoder_model_path)
        self.audio_encoder = AudioEncoder(n_mels=checkpoint['dims']['n_mels'],
                                          n_ctx=checkpoint['dims']['n_audio_ctx'],
                                          n_state=checkpoint['dims']['n_audio_state'],
                                          n_head=checkpoint['dims']['n_audio_head'],
                                          n_layer=checkpoint['dims']['n_audio_layer'])
        if config.content_layer_num >= 0:
            self.content_encoder = ContentEncoder(config)
        else:
            self.content_encoder = None
        self.load_pretrained_whisper_params()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def load_pretrained_whisper_params(self):
        checkpoint = torch.load(self.config.encoder_model_path)
        self.audio_encoder.load_state_dict(checkpoint['model_state_dict'])

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.audio_encoder.forward(input_ids)

        if self.content_encoder is not None:
            hidden_states = self.content_encoder(hidden_states, attention_mask=attention_mask)

        if not return_dict:
            return tuple(v for v in [hidden_states, hidden_states, attention_mask] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=hidden_states, attentions=attention_mask
        )

    def freeze(self, only_whisper=False) -> None:
        r"""
        Freeze all params for inference.

        Example::

            model = MyLightningModule(...)
            model.freeze()

        """
        if only_whisper:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
        else:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters for training.

        .. code-block:: python

            model = MyLightningModule(...)
            model.unfreeze()
        """
        for param in self.parameters():
            param.requires_grad = True

        self.train()
