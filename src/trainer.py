import os

from ultralytics.models.yolo.classify.train import ClassificationTrainer
from dataset import get_data_loader


class CustomTrainer(ClassificationTrainer):

    LABEL_PATH = os.path.join("..", "data", "labels.csv")
    IMG_DIR = os.path.join("..", "data")

    def __init__(self):
        # First, initialize the base class with its necessary parameters
        super().__init__()

        self.train_data_loader, self.val_data_loader = get_data_loader(annotations_file=CustomTrainer.LABEL_PATH, img_dir=CustomTrainer.IMG_DIR, batch_size=1)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode == "train":
            return self.train_data_loader
        else:
            return self.val_data_loader
        
    def get_dataset(self):
        return None, None