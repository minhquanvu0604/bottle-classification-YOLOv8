import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

from PIL import Image
import os

class WDYWTDClassificationDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1:].to_numpy()
        if self.transform:
            image = self.transform(image)
        return image, label

# Set up transformations and dataset
transform = transforms.Compose([
    transforms.Resize((416, 416)),  # Resize to the input size expected by YOLOv8
    transforms.ToTensor(),          # Convert images to tensor
])
dataset = WDYWTDClassificationDataset('annotations.csv', 'path/to/images', transform=transform)

# DataLoader for batching operations
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Usage in a training loop
for images, labels in dataloader:
    # Here you would pass images and labels to your model
    pass
