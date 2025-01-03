from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from torch_uncertainty.datamodules import TUDataModule

from datasets.park_finger_tapping import ParkFingerTappingDataset

class ParkFingerTappingDataModule(TUDataModule):
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
        self.csv_path = Path(root) / "features_demography_diagnosis_Nov22_2023.csv"
        self.dataset = ParkFingerTappingDataset(csv_path=self.csv_path)

        self.test_ids = self._load_ids(test_ids_path)
        self.dev_ids = self._load_ids(dev_ids_path)
        self.train_ids = self._load_train_ids()
        
        self.train = ParkFingerTappingDataset(csv_path=self.csv_path, ids=self.train_ids)
        self.val = ParkFingerTappingDataset(csv_path=self.csv_path, ids=self.dev_ids)
        self.test = ParkFingerTappingDataset(csv_path=self.csv_path, ids=self.test_ids)
        
        print(f"all_ids: {self.dataset.ids.size} - test_ids: {self.test.ids.size} - dev_ids: {self.val.ids.size} - train_ids: {self.train.ids.size}")

        self.num_classes = 1
        self.num_features = len(self.train_dataloader().dataset.features[0])

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
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
