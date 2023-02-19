import json
import unittest
import torch

from vc_lm.models.encoders.whisper_encoder import WhisperEncoder
from vc_lm.models.base import VCLMConfig


class TestWhisperEncoder(unittest.TestCase):
    def setUp(self) -> None:
        config = VCLMConfig(**json.load(open("configs/ar_model.json")))
        self.model = WhisperEncoder(config)
        self.model.cuda()

    def test_whisper_encoder(self):
        mels = torch.rand([2, 80, 3000]).cuda()
        with torch.inference_mode():
            content_feats = self.model(mels)
        self.assertEqual(list(content_feats.shape), [2, 1500, 1024])


if __name__ == '__main__':
    unittest.main()
