import os
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class ClassDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, target_transform=None):
        self.img_labels = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        extension = ".jpg"
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + extension)

        try:
            image = read_image(img_path)
        except FileNotFoundError:
            logging.error(f"Image not found: {img_path}")
            return None
        except Exception as e:
            logging.error(f"Error reading the image {img_path}: {e}")
            return None        
        
        label = self.img_labels.iloc[idx, 1] # Typically an integer or a categorical encoding 

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def get_data_loader(annotations_file, img_dir, batch_size=1):
    labels_df = pd.read_csv(annotations_file)
    train_df = labels_df[labels_df['dataset'] == 'train']
    val_df = labels_df[labels_df['dataset'] == 'val']
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: x / 255.0)
    ])
    transform = None

    train_path = os.path.join(img_dir, 'train')
    val_path = os.path.join(img_dir, 'val')

    train_dataset = ClassDataset(dataframe=train_df, img_dir=train_path, transform=transform)
    val_dataset = ClassDataset(dataframe=val_df, img_dir=val_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def display_sample(data_loader : DataLoader):
    features, labels = next(iter(data_loader))

    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    # In the case of image tensors like [channels, height, width] coming directly from a DataLoader, 
    # using squeeze() generally doesn't change the tensor since there shouldn't be any singleton dimensions. 
    # However, if the tensor has a shape like [1, channels, height, width] or [channels, height, width, 1], 
    # then squeeze() would reduce it to [channels, height, width].
    
    img = features[0].squeeze() 
    img = img.permute(1, 2, 0)
    label = labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()


# Testing
if __name__ == "__main__":

    label_path = os.path.join("..", "data", "labels.csv")
    data_path = os.path.join("..", "data")
    train_loader, val_loader = get_data_loader(annotations_file=label_path, img_dir=data_path)
    
    display_sample(train_loader)
    
    
    