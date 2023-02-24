import torch
from torch import nn
import warnings
from typing import List, Optional

from vc_lm.models.base import  VCLMPretrainedModel, VCLMConfig
from vc_lm.models.encoders.whisper_encoder import WhisperEncoder
from vc_lm.models.decoders.nar_decoder import NARDecoder


class NARModel(VCLMPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: VCLMConfig):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = WhisperEncoder(config)
        self.encoder.freeze(only_whisper=True)
        self.decoder = NARDecoder(config, self.shared)
        self.num_q = config.n_q
        self.q_size = config.q_size
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        style_code: torch.LongTensor = None,
        nar_stage: torch.LongTensor = None):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids,
                                           attention_mask=attention_mask)
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # （batch_size, target_len, dims）
        decoder_outputs = self.decoder(
            input_code=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            style_code=style_code,
            nar_stage=nar_stage
        )
        # (num_q, q_size, dim)
        reshaped_emb = torch.reshape(self.shared.weight[0:self.q_size * self.num_q], (self.num_q, self.q_size, -1))
        # (batch_size, q_size, dim)
        predicted_emb = reshaped_emb[nar_stage + 1]
        # (batch_size, target_len, q_size)
        logits = torch.einsum('btd,bqd->btq', decoder_outputs, predicted_emb)

        return decoder_outputs, logits
