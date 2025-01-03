import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from torch_uncertainty.datamodules import TUDataModule

from datasets.park_speech import ParkSpeechDataset

class ParkSpeechDataModule(TUDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        test_ids_path: Optional[str] = None,
        dev_ids_path: Optional[str] = None,
    ):
        super().__init__(root, batch_size, val_split, num_workers, pin_memory, persistent_workers)
        self.csv_path = Path(root) / "wavlm_fox_features.csv"
        self.dataset = ParkSpeechDataset(csv_path=self.csv_path)

        self.test_ids = self._load_ids(test_ids_path)
        self.dev_ids = self._load_ids(dev_ids_path)
        self.train_ids = self._load_train_ids()
        self.train = ParkSpeechDataset(csv_path=self.csv_path, ids=self.train_ids)
        self.val = ParkSpeechDataset(csv_path=self.csv_path, ids=self.dev_ids)
        self.test = ParkSpeechDataset(csv_path=self.csv_path, ids=self.test_ids)

        self.num_classes = 1
        self.num_features = self.train_dataloader().dataset.features.shape[1]

    def _load_ids(self, path: Optional[str]):
        if path and Path(path).exists():
            with open(path, "r") as f:
                return set(line.strip() for line in f)
        return None

    def _load_train_ids(self):
        all_ids = set(self.dataset.ids)
        return all_ids - self.test_ids - self.dev_ids

    def setup(self, stage: Optional[str] = None):
        print(f"from setup: train size: {len(self.train)}, val size: {len(self.val)}, test size: {len(self.test)}")
        print(f"from setup: num_features: {self.num_features}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers
        )
