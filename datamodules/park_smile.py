import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from torch_uncertainty.datamodules import TUDataModule

from datasets.park_smile import ParkSmileDataset

class ParkSmileDataModule(TUDataModule):
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
        self.csv_path = Path(root) / "facial_dataset.csv"
        self.test_ids = self._load_ids(test_ids_path)
        self.dev_ids = self._load_ids(dev_ids_path)
        self.train_ids = self._load_train_ids()
        self.setup()
        self.num_classes = 1
        self.num_features = self.train_dataloader().dataset.features.shape[1]

    def _load_ids(self, path: Optional[str]):
        if path and Path(path).exists():
            with open(path, "r") as f:
                return set(line.strip() for line in f)
        return None
    
    def _load_train_ids(self):
        data = pd.read_csv(self.csv_path)
        all_ids = set(data['ID'])
        return all_ids - self.test_ids - self.dev_ids

    def setup(self, stage: Optional[str] = None):
        features, labels, ids = self._load_data()

        print(f'size of features: {features.shape}')
        print(f'size of labels: {labels.shape}')
        print(f'size of ids: {len(ids)}')
        print(f'size of train_ids: {len(self.train_ids)}')
        print(f'size of dev_ids: {len(self.dev_ids)}')
        print(f'size of test_ids: {len(self.test_ids)}')

        train_mask = ids.isin(self.train_ids)
        dev_mask = ids.isin(self.dev_ids)
        test_mask = ids.isin(self.test_ids)
        train_features = features[train_mask]
        train_labels = labels[train_mask]
        dev_features = features[dev_mask]
        dev_labels = labels[dev_mask]
        test_features = features[test_mask]
        test_labels = labels[test_mask]

        self.train = ParkSmileDataset(features=train_features, labels=train_labels)
        self.val = ParkSmileDataset(features=dev_features, labels=dev_labels)
        self.test = ParkSmileDataset(features=test_features, labels=test_labels)
        

    def _load_data(self):
        dataset = ParkSmileDataset(csv_path=self.csv_path)
        return dataset.features, dataset.labels, dataset.ids

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
