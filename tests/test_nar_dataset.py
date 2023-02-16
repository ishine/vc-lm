import unittest

from vc_lm.datamodules.datasets.nar_dataset import NARDataset

class TestARDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = NARDataset('/root/autodl-tmp/data/vc-lm-outputs/train')

    def test_nar_dataset(self):
        item = next(iter(self.dataset))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
