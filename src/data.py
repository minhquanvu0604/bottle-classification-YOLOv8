import os
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd

class WDYWTDClassificationDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        try:
            image = read_image(img_path)
        except FileNotFoundError:
            logging.error(f"Image not found: {img_path}")
            return None
        except Exception as e:
            logging.error(f"Error reading the image {img_path}: {e}")
            return None        
        label = self.img_labels.iloc[idx, 1:].to_numpy()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
