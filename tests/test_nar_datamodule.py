import unittest

from vc_lm.datamodules.nar_datamodule import NARDataModule


class TestArDataModule(unittest.TestCase):
    def setUp(self) -> None:
        self.data_module = NARDataModule('/root/autodl-tmp/data/vc-lm-outputs',
                                         batch_size=64)

    def test_ar_datamodule(self):
        self.data_module.prepare_data()
        self.data_module.setup()
        assert self.data_module.train_dataloader() and self.data_module.val_dataloader() and self.data_module.test_dataloader()
        item = next(iter(self.data_module.train_dataloader()))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()