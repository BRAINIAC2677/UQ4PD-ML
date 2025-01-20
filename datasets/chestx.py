# import os
# from PIL import Image
# from torch.utils.data import Dataset


# class ChestXDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.imgs = []
#         self.labels = []
#         for i in os.listdir(os.path.join(root, 'NORMAL')):
#             self.imgs.append(os.path.join(root, 'NORMAL', i))
#             self.labels.append(0)
#         for i in os.listdir(os.path.join(root, 'PNEUMONIA')):
#             self.imgs.append(os.path.join(root, 'PNEUMONIA', i))
#             self.labels.append(1)
#         self.transform = transform
#         self.dataset='chestx'
#         self.nb_classes=2


#     def __getitem__(self, index):
#         img, target = self.imgs[index], self.labels[index]
#         # open the image file in one channel
#         img = Image.open(img).convert('L')
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, target


#     def __len__(self):
#         return len(self.imgs)
    


import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ChestXDataset(Dataset):
    def __init__(self, root, transform=None, balance=True):
        """
        Custom dataset for Chest X-ray images with dataset balancing.
        
        Args:
            root (str): Root directory of the dataset.
            transform (callable, optional): Transform to apply to the images.
            augment (callable, optional): Augmentation transform for the undersampled class.
            balance (bool): Whether to balance the dataset.
        """
        self.imgs = []
        self.labels = []

        normal_dir = os.path.join(root, 'NORMAL')
        pneumonia_dir = os.path.join(root, 'PNEUMONIA')

        for i in os.listdir(normal_dir):
            self.imgs.append(os.path.join(normal_dir, i))
            self.labels.append(0)
        for i in os.listdir(pneumonia_dir):
            self.imgs.append(os.path.join(pneumonia_dir, i))
            self.labels.append(1)

        # Combine image paths and labels for easier balancing
        self.data = list(zip(self.imgs, self.labels))
        self.transform = transform
        self.augment = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(10),
                        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    ])
        self.dataset = 'chestx'
        self.nb_classes = 2

        # Balance the dataset if needed
        if balance:
            self._balance_dataset()

    def _balance_dataset(self):
        """
        Balances the dataset by oversampling the undersampled class using data augmentation.
        """
        # Split data by class
        normal_data = [item for item in self.data if item[1] == 0]
        pneumonia_data = [item for item in self.data if item[1] == 1]

        # Identify the undersampled class
        undersampled_data, oversampled_data = (
            (normal_data, pneumonia_data)
            if len(normal_data) < len(pneumonia_data)
            else (pneumonia_data, normal_data)
        )

        # Augment the undersampled class to balance the dataset
        augmented_data = []
        while len(undersampled_data) + len(augmented_data) < len(oversampled_data):
            for img_path, label in undersampled_data:
                img = Image.open(img_path)
                
                # Check if grayscale and convert to 3 channels if necessary
                if img.mode == 'L':
                    img = np.stack([np.array(img)] * 3, axis=-1)  # Convert grayscale to RGB
                    img = Image.fromarray(img.astype('uint8'), 'RGB')

                # Apply augmentation
                if self.augment:
                    img = self.augment(img)

                # Save augmented data
                augmented_data.append((img, label))
                if len(undersampled_data) + len(augmented_data) >= len(oversampled_data):
                    break

        # Add augmented data to the dataset
        self.data.extend(augmented_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Fetch an item and apply transformations.
        """
        img_path, target = self.data[index]

        # If the sample is an augmented image, it will be an Image object
        if isinstance(img_path, str):
            img = Image.open(img_path)

            # Check if grayscale and convert to 3 channels if necessary
            if img.mode == 'L':
                img = np.stack([np.array(img)] * 3, axis=-1)  # Convert grayscale to RGB
                img = Image.fromarray(img.astype('uint8'), 'RGB')
        else:
            img = img_path  # Already an augmented Image object

        # Apply transformations (normalization, resizing, etc.)
        if self.transform:
            img = self.transform(img)

        return img, target


