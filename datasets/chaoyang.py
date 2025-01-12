from PIL import Image
import os
import json
from torch.utils.data import Dataset

class ChaoyangDataset(Dataset):
    def __init__(self, root, json_name=None, path_list=None, label_list=None, transform=None):
        self.imgs = []
        self.labels = []
        if json_name:
            json_path = os.path.join(root,json_name)
            with open(json_path,'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(root,load_list[i]["name"])
                    self.imgs.append(img_path)
                    self.labels.append(load_list[i]["label"])
        if (path_list and label_list):
            self.imgs = path_list
            self.labels = label_list
        self.transform = transform
        self.dataset='chaoyang'
        self.nb_classes=4

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)