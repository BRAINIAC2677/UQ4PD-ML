import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXDataset(Dataset):
    def __init__(self, root, transform=None):
        self.imgs = []
        self.labels = []
        for i in os.listdir(os.path.join(root, 'NORMAL')):
            self.imgs.append(os.path.join(root, 'NORMAL', i))
            self.labels.append(0)
        for i in os.listdir(os.path.join(root, 'PNEUMONIA')):
            self.imgs.append(os.path.join(root, 'PNEUMONIA', i))
            self.labels.append(1)
        self.transform = transform
        self.dataset='chestx'
        self.nb_classes=2


    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        # open the image file in one channel
        img = Image.open(img).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


    def __len__(self):
        return len(self.imgs)