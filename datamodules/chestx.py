import os
import json
from typing import Optional
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_uncertainty.datamodules import TUDataModule

from datasets.chestx import ChestXDataset

class ChestXDataModule(TUDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):

        super().__init__(root, batch_size, None, num_workers, pin_memory, persistent_workers)
        self.root = root
        train_root = os.path.join(self.root, 'train')
        val_root = os.path.join(self.root, 'val')
        test_root = os.path.join(self.root, 'test')

        self.train = ChestXDataset(root=train_root, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)), transforms.ToTensor()]))
        self.val = ChestXDataset(root=val_root, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
        self.test = ChestXDataset(root=test_root, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))

        self.num_classes = self.train.nb_classes
        self.num_channels = 1

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
