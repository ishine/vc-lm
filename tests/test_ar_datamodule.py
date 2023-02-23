import unittest

from vc_lm.datamodules.ar_datamodule import ARDataModule

from tqdm.auto import tqdm

class TestArDataModule(unittest.TestCase):
    def setUp(self) -> None:
        self.data_module = ARDataModule('/home/jiangxinghua/data/vc-lm/audios-dataset-small',
                                        batch_size=2,
                                        max_audio_time=24, num_workers=2, train_dataset_size=1000, train_pattern="shard-{000000..000012}.tar",
                                        val_dataset_size=300, val_pattern="shard-{000000..000009}.tar")

    def test_ar_datamodule(self):
        self.data_module.prepare_data()
        self.data_module.setup()
        assert self.data_module.train_dataloader() is not None and self.data_module.val_dataloader() is not None and self.data_module.test_dataloader() is not None
        item = next(iter(self.data_module.train_dataloader()))
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()