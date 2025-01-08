from typing import Optional
from torch.utils.data import DataLoader
from torchvision import transforms
from torch_uncertainty.datamodules import TUDataModule

from datasets.chaoyang import ChaoyangDataset

class ChaoyangDataModule(TUDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 1,
        val_split: float = 0.2,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__(root, batch_size, val_split, num_workers, pin_memory, persistent_workers)
        self.root = root

        self.train = ChaoyangDataset(root=self.root, json_name="train.json", train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)), transforms.ToTensor()]))
        self.test = ChaoyangDataset(root=self.root, json_name="test.json", train=False, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))

        self.num_classes = self.train.nb_classes
        self.num_channels = 3

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
