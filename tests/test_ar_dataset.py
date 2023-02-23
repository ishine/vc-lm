import unittest

from vc_lm.datamodules.datasets.ar_dataset import ARDataset

class TestARDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = ARDataset('/home/jiangxinghua/data/vc-lm/wds/train')

    def test_ar_dataset(self):
        item = next(iter(self.dataset.get_dataset()))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()