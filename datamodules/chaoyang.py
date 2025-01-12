import os
import json
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

        self.split(val_split)

        self.train = ChaoyangDataset(root=self.root, json_name="temp_train.json", transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)), transforms.ToTensor()]))
        self.val = ChaoyangDataset(root=self.root, json_name="temp_val.json", transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
        self.test = ChaoyangDataset(root=self.root, json_name="test.json", transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))

        self.clean_split()

        self.num_classes = self.train.nb_classes
        self.num_channels = 3
    
    def split(self, val_split: float):
        with open(os.path.join(self.root, "train.json"), 'r') as f:
            load_list = json.load(f)
            train_list = load_list[:int(len(load_list)*(1-val_split))]
            val_list = load_list[int(len(load_list)*(1-val_split)):]
        with open(os.path.join(self.root, "temp_train.json"), 'w') as f:
            json.dump(train_list, f)
        with open(os.path.join(self.root, "temp_val.json"), 'w') as f:
            json.dump(val_list, f)
        
    
    def clean_split(self):
        os.remove(os.path.join(self.root, "temp_train.json"))
        os.remove(os.path.join(self.root, "temp_val.json"))


    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
