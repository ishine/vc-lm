import math
import numpy as np
import torch

from streaming import StreamingDataset

from vc_lm.utils.data_utils import pad_or_trim

class NARDataset(StreamingDataset):
    _MEL_SAMPLE_RATE = 100
    _CODE_SAMPLE_RATE = 75
    _NUM_Q = 8
    _EOS_ID = 1024
    _MAX_MEL_AUDIO_TIME = 30
    def __init__(self,
                 local,
                 remote=None,
                 max_audio_time=24,
                 style_audio_time=3,
                 shuffle=False):
        super().__init__(local, remote, shuffle=shuffle)
        self.max_audio_time = max_audio_time
        self.style_audio_time = 3
        self.max_mel_len = int(self._MEL_SAMPLE_RATE * self._MAX_MEL_AUDIO_TIME)
        self.max_code_len = int(self._CODE_SAMPLE_RATE * self.max_audio_time)
        self.max_content_len = math.ceil(self.max_mel_len/2)
        self.style_code_len = int(self._CODE_SAMPLE_RATE * style_audio_time)

    def __getitem__(self, idx: int):
        """
        return:
            {
                "mel": (80, max_mel_len),
                "content_mask": (max_content_len,),
                "input_code": (8, code_len),
                "code_mask": (max_code_len,)
            }
        """
        obj = super().__getitem__(idx)
        mel_len = obj['mel'].shape[1]
        # (max_content_len,)
        content_mask = torch.lt(torch.arange(0, self.max_content_len), math.ceil(mel_len//2)).type(torch.long)
        # （80， max_mel_len）
        mel = pad_or_trim(obj['mel'], self.max_mel_len)
        # （_NUM_Q, code_len）
        input_code = obj['code'].astype(np.int64)
        code_len = input_code.shape[1]
        # (max_code_len,)
        code_mask = torch.lt(torch.arange(0, self.max_code_len),
                             code_len).type(torch.long)
        # style_code (8, style_code_len)
        style_start_pos = torch.randint(0, input_code.shape[1] - self.style_code_len, ())
        style_code = input_code[:, style_start_pos:style_start_pos+self.style_code_len]

        # pad input_code, output_code
        input_code = pad_or_trim(input_code, self.max_code_len)
        return {
            'mel': mel,
            'content_mask': content_mask,
            'input_code': input_code,
            'code_mask': code_mask,
            'style_code': style_code
        }
