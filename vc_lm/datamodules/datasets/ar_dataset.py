import math
import numpy as np
import torch

import webdataset as wds

from vc_lm.utils.data_utils import pad_or_trim

class ARDataset(object):
    _MEL_SAMPLE_RATE = 100
    _CODE_SAMPLE_RATE = 75
    _NUM_Q = 8
    _EOS_ID = 1024 * 8
    _PAD_ID = 1024 * 8 + 1
    _MAX_MEL_AUDIO_TIME = 30

    def __init__(self,
                 local,
                 remote=None,
                 pattern=None,
                 max_audio_time=24,
                 shuffle=False,
                 shuffle_buffer=2000):
        self.local_path = local
        self.max_audio_time = max_audio_time
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.max_mel_len = int(self._MEL_SAMPLE_RATE * self._MAX_MEL_AUDIO_TIME)
        self.max_code_len = int(self._CODE_SAMPLE_RATE * self.max_audio_time)
        self.max_content_len = math.ceil(self.max_mel_len/2)
        self.pattern = pattern

    def process_record(self, record):
        obj = record['data.pyd']
        mel_len = obj['mel'].shape[1]
        # (max_content_len,)
        content_mask = torch.lt(torch.arange(0, self.max_content_len), math.ceil(mel_len//2)).type(torch.long)
        # （80， max_mel_len）
        mel = pad_or_trim(obj['mel'], self.max_mel_len)
        # （_NUM_Q, code_len）
        input_code = obj['code'].astype(np.int64)
        output_code = np.concatenate([input_code, np.ones([self._NUM_Q, 1], dtype=np.int64) * self._EOS_ID], axis=1)
        code_len = input_code.shape[1]
        # (max_code_len,)
        code_mask = torch.lt(torch.arange(0, self.max_code_len),
                             code_len).type(torch.long)
        # pad input_code, output_code
        input_code = pad_or_trim(input_code[0], self.max_code_len, pad_value=self._PAD_ID)
        output_code = pad_or_trim(output_code[0][1:], self.max_code_len,
                                  pad_value=self._PAD_ID)
        return {
            'mel': torch.tensor(mel),
            'content_mask': content_mask,
            'input_code': torch.tensor(input_code),
            'output_code': torch.tensor(output_code),
            'code_mask': code_mask
        }

    def get_dataset(self):
        dataset = wds.WebDataset(self.local_path + '/' + self.pattern,
                                 nodesplitter=wds.split_by_node)
        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer)
        dataset = dataset.decode().map(self.process_record)
        return dataset