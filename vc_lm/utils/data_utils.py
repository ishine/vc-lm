from torch.nn import functional as F

import numpy as np
from typing import Any

from encodec.utils import convert_audio

import torchaudio
import torch

from whisper.audio import log_mel_spectrogram


def pad_or_trim(array, length: int, axis: int = -1,
                pad_value=0):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes],
                          value=pad_value)
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths,
                           constant_values=pad_value)

    return array


def get_code(audio: str,
             device: str,
             model: Any):
    wav, sr = torchaudio.load(audio)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    wav = wav.cuda(device)
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    code = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    return code.cpu().numpy().astype(np.int16)[0]


def get_mel_spectrogram(audio):
    mel = log_mel_spectrogram(audio)
    return mel.numpy()
