# from PIL import Image
# import os
# import json
# from torch.utils.data import Dataset

# class ChaoyangDataset(Dataset):
#     def __init__(self, root, json_name=None, path_list=None, label_list=None, transform=None):
#         self.imgs = []
#         self.labels = []
#         if json_name:
#             json_path = os.path.join(root,json_name)
#             with open(json_path,'r') as f:
#                 load_list = json.load(f)
#                 for i in range(len(load_list)):
#                     img_path = os.path.join(root,load_list[i]["name"])
#                     self.imgs.append(img_path)
#                     self.labels.append(load_list[i]["label"])
#         if (path_list and label_list):
#             self.imgs = path_list
#             self.labels = label_list
#         self.transform = transform
#         self.dataset='chaoyang'
#         self.nb_classes=4

#     def __getitem__(self, index):
#         img, target = self.imgs[index], self.labels[index]
#         img = Image.open(img)
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target

#     def __len__(self):
#         return len(self.imgs)



from PIL import Image
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ChaoyangDataset(Dataset):
    def __init__(self, root, json_name=None, path_list=None, label_list=None, transform=None, balance=True):
        """
        Chaoyang Dataset with optional dataset balancing through oversampling.
        
        Args:
            root (str): Root directory of the dataset.
            json_name (str, optional): JSON file containing image paths and labels.
            path_list (list, optional): List of image paths.
            label_list (list, optional): List of labels corresponding to image paths.
            transform (callable, optional): Transformations to apply to images.
            balance (bool, optional): Whether to balance the dataset by oversampling.
        """
        self.imgs = []
        self.labels = []

        # Load data from JSON file or provided lists
        if json_name:
            json_path = os.path.join(root, json_name)
            with open(json_path, 'r') as f:
                load_list = json.load(f)
                for entry in load_list:
                    img_path = os.path.join(root, entry["name"])
                    self.imgs.append(img_path)
                    self.labels.append(entry["label"])
        elif path_list and label_list:
            self.imgs = path_list
            self.labels = label_list

        self.data = list(zip(self.imgs, self.labels))
        self.transform = transform
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        self.dataset = 'chaoyang'
        self.nb_classes = 4

        # Balance dataset if required
        if balance:
            self._balance_dataset()

    def _balance_dataset(self):
        """
        Balances the dataset by oversampling undersampled classes using data augmentation.
        """
        # Group data by class
        class_data = {i: [] for i in range(self.nb_classes)}
        for img_path, label in self.data:
            class_data[label].append((img_path, label))

        # Find the maximum class size
        max_class_size = max(len(samples) for samples in class_data.values())

        # Augment data for undersampled classes
        for label, samples in class_data.items():
            augmented_data = []
            while len(samples) + len(augmented_data) < max_class_size:
                for img_path, _ in samples:
                    img = Image.open(img_path)

                    # Apply augmentation
                    augmented_img = self.augment(img)

                    # Add augmented sample to the dataset
                    augmented_data.append((augmented_img, label))
                    if len(samples) + len(augmented_data) >= max_class_size:
                        break
            self.data.extend(augmented_data)

    def __getitem__(self, index):
        """
        Fetch an item and apply transformations.
        """
        img_path, target = self.data[index]

        # Handle augmented images (Image object) differently
        if isinstance(img_path, str):
            img = Image.open(img_path)
        else:
            img = img_path  # Already an augmented Image object

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
