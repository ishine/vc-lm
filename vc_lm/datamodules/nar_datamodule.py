import os
import random
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from vc_lm.datamodules.datasets.nar_dataset import NARDataset

import webdataset as wds
from webdataset.utils import pytorch_worker_info

def nar_collate_fn(x):
    nar_stage = random.randint(0, NARDataset._NUM_Q - 2)
    for idx, item in enumerate(x):
        item['nar_stage'] = torch.tensor(nar_stage)
        item['output_code'] = item['input_code'][item['nar_stage'] + 1]
        item['input_code'] = item['input_code'][0:item['nar_stage'] + 1]
    y = default_collate(x)
    if '__key__' in y:
        del y['__key__']
    return y


class NARDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 64,
                 max_audio_time: float = 24,
                 style_audio_time: float = 3,
                 num_workers: int = 0,
                 train_dataset_size: int = -1,
                 val_dataset_size: int = 200,
                 train_pattern: str = None,
                 val_pattern: str = None,
                 pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_audio_time = max_audio_time
        self.style_audio_time = style_audio_time
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_dataset_size = train_dataset_size
        self.val_dataset_size = val_dataset_size
        self.train_pattern = train_pattern
        self.val_pattern = val_pattern

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`."""
        if not self.data_train or not self.data_val or not self.data_test:
            self.data_train = NARDataset(os.path.join(self.data_dir, 'train'),
                                         pattern=self.train_pattern,
                                         max_audio_time=self.max_audio_time,
                                         style_audio_time=self.style_audio_time,
                                         shuffle=True).get_dataset()
            self.data_val = NARDataset(os.path.join(self.data_dir, 'val'),
                                       pattern=self.val_pattern,
                                       max_audio_time=self.max_audio_time,
                                       style_audio_time=self.style_audio_time).get_dataset()
            self.data_test = NARDataset(os.path.join(self.data_dir, 'test'),
                                        pattern=self.val_pattern,
                                        max_audio_time=self.max_audio_time,
                                        style_audio_time=self.style_audio_time).get_dataset()

    def train_dataloader(self):
        return self.get_dataloader(self.data_train, self.train_dataset_size)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val, self.val_dataset_size)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test, self.val_dataset_size)

    def get_dataloader(self, dataset, dataset_size):
        # batch
        dataset = dataset.batched(self.batch_size,
                                  collation_fn=nar_collate_fn, partial=False)
        _, world_size, _, _ = pytorch_worker_info()
        number_of_batches = int(dataset_size // (world_size * self.batch_size))
        loader = wds.WebLoader(dataset,
                               batch_size=None,
                               shuffle=False, num_workers=self.num_workers).with_length(number_of_batches).with_epoch(number_of_batches)
        loader = loader.repeat(20).slice(number_of_batches)
        return loader
