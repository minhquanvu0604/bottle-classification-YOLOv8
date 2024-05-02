import os
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd

class ClassDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transform=None):
        self.img_labels = dataframe
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


def get_data_loader(annotations_file, img_dir, batch_size=2) -> tuple[DataLoader, DataLoader]:
    labels_df = pd.read_csv(annotations_file)
    train_df = labels_df[labels_df['dataset'] == 'train']
    val_df = labels_df[labels_df['dataset'] == 'val']
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.0)
    ])

    train_path = os.path.join(img_dir, 'train')
    val_path = os.path.join(img_dir, 'val')

    train_dataset = ClassDataset(dataframe=train_df, img_dir=train_path, transform=transform)
    val_dataset = ClassDataset(dataframe=val_df, img_dir=val_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    
    
    