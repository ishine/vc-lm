import unittest

from vc_lm.datamodules.datasets.ar_dataset import ARDataset

class TestARDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = ARDataset('/root/autodl-tmp/data/vc-lm-outputs/train')

    def test_ar_dataset(self):
        item = next(iter(self.dataset))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()