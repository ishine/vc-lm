import math
import torch
import random

from torch import nn

from typing import Optional, Union, List, Tuple
from vc_lm.models.bart.modeling_bart import BartLearnedPositionalEmbedding, BartDecoderLayer, _make_causal_mask, _expand_mask, BaseModelOutputWithPastAndCrossAttentions

from transformers.utils import logging

from vc_lm.models.base import VCLMConfig, VCLMPretrainedModel
from vc_lm.models.misc import StageAdaLN
from vc_lm.models.decoders.layers import NARStageDecoderLayer

logger = logging.get_logger(__name__)


class AccumulateMultiStageEmbedding(nn.Module):
    def __init__(self, embed_tokens: nn.Embedding,
                 q_size: int = 1024):
        """AccumulateMultiStageEmbedding"""
        super().__init__()
        self.embed_tokens = embed_tokens
        self.q_size = q_size

    def forward(self, multistage_code: torch.LongTensor):
        """
        Args:
            multistage_code: (batch_size, stage_num, seq_len)
        Return:
            multistage_code_emb: (batch_size, seq_len, dim)
        """
        stage_id = torch.arange(0, multistage_code.shape[1])[None, ..., None]
        multistage_code = stage_id * self.q_size + multistage_code
        # (batch_size, stage_num, seq_len, dim)
        multistage_code = self.embed_tokens(multistage_code)
        return torch.sum(multistage_code, dim=1)


class NARDecoder(VCLMPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: VCLMConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.style_positions = BartLearnedPositionalEmbedding(config.style_length, config.d_model)
        self.stage_embed = nn.Embedding(config.num_q, config.d_model)
        self.accumulate_multistage_embedding_layer = AccumulateMultiStageEmbedding(self.embed_tokens,
                                                                                   q_size=config.q_size)
        self.register_buffer('style_mask', torch.ones((1, config.style_length), dtype=torch.int64))

        self.layers = nn.ModuleList([NARStageDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = StageAdaLN(nn.LayerNorm(config.d_model),
                                              num_stage=config.n_q - 1)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds,
                                        past_key_values_length=None):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_code: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        style_code: torch.LongTensor = None,
        nar_stage: torch.LongTensor = None,
    ):
        r"""
        Args:
            input_code: (`torch.LongTensor` of shape `(batch_size, num_stage, sequence_length)`)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            style_code: (`torch.LongTensor` of shape `(batch_size, num_q, style_len)`)
            nar_stage: (`torch.LongTensor` of shape `(batch_size)`)
        """
        batch_size = input_code.shape[0]
        # get input_code_embeds
        # (batch_size, seq_len, dim)
        input_code_embeds = self.accumulate_multistage_embedding_layer(input_code)
        # (batch_size, style_len, dim)
        style_code_embeds = self.accumulate_multistage_embedding_layer(style_code)
        # (batch_size, style_len + seq_len, dim)
        inputs_embeds = torch.cat([style_code_embeds, input_code_embeds], 1) * self.embed_scale
        # pad style_mask: attention_mask (batch_size, style_len + seq_len)
        attention_mask = torch.cat([self.style_mask.expand(batch_size, -1), attention_mask], 1)
        input_shape = attention_mask.shape

        attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        # embed positions
        style_positions = self.style_positions(style_code[:, 0, :], past_key_values_length=None)
        input_code_positions = self.embed_positions(input_code[:, 0, :], past_key_values_length=None)
        positions = torch.cat([style_positions, input_code_positions], 1)
        positions = positions.to(inputs_embeds.device)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states, nar_stage)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, None, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=None,
                    cross_attn_layer_head_mask=None,
                    past_key_value=past_key_value,
                    output_attentions=None,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
        # extract code hidden_states
        return hidden_states[:, self.config.style_length:]
