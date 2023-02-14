import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from vc_lm.datamodules.datasets.ar_dataset import ARDataset

class ARDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 64,
                 max_audio_time: float = 30,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_audio_time = max_audio_time
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`."""
        if not self.data_train or not self.data_val or not self.data_test:
            self.data_train = ARDataset(os.path.join(self.data_dir, 'train'),
                                        max_audio_time=self.max_audio_time, shuffle=True)
            self.data_val = ARDataset(os.path.join(self.data_dir, 'val'),
                                      max_audio_time=self.max_audio_time)
            self.data_test = ARDataset(os.path.join(self.data_dir, 'test'),
                                       max_audio_time=self.max_audio_time)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )