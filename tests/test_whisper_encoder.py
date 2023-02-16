import unittest
import torch

from vc_lm.models.modules.encoders.whisper_encoder import WhisperEncoder


class TestWhisperEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.model = WhisperEncoder.from_pretrained('/root/autodl-tmp/pretrained-models/whisper/small-encoder.pt')
        self.model.cuda()

    def test_whisper_encoder(self):
        mels = torch.rand([2, 80, 3000]).cuda()
        with torch.inference_mode():
            content_feats = self.model(mels)
        self.assertEqual(list(content_feats.shape), [2, 1500, 768])


if __name__ == '__main__':
    unittest.main()
