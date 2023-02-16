import torch

import pytorch_lightning as pl

from whisper.model import AudioEncoder

class WhisperEncoder(pl.LightningModule):
    """
    Whisper audio encoder.
    """
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.audio_encoder = AudioEncoder(n_mels, n_ctx, n_state, n_head, n_layer)

    def forward(self, x: torch.Tensor):
        return self.audio_encoder.forward(x)

    @staticmethod
    def from_pretrained(model_path):
        checkpoint = torch.load(model_path)
        whisper_encoder = WhisperEncoder(n_mels=checkpoint['dims']['n_mels'],
                                         n_ctx=checkpoint['dims']['n_audio_ctx'],
                                         n_state=checkpoint['dims']['n_audio_state'],
                                         n_head=checkpoint['dims']['n_audio_head'],
                                         n_layer=checkpoint['dims']['n_audio_layer'])
        whisper_encoder.audio_encoder.load_state_dict(checkpoint['model_state_dict'])
        return whisper_encoder
