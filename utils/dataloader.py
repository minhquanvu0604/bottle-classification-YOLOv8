import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
import numpy as np

# Define the path to the test dataset
test_dir = '/root/aifr/bottle-classification-YOLOv8/data/test'

# Define transformations for the test dataset
test_transforms = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize images to the size expected by the model
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
])

# Create a dataset
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

# Create a dataloader
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Load the model
model = YOLO('/root/aifr/bottle-classification-YOLOv8/src/runs/classify/train/weights/best.pt')

# Ensure the model is in evaluation mode
model.eval()

# Initialize lists to hold ground truth labels and model predictions
true_labels = []
pred_labels = []

# Get the class-to-index mapping
class_to_idx = test_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Disable gradient calculation for inference
with torch.no_grad():
    for images, labels in test_loader:
        # Predict with the model
        results = model(images)

        # Extract predictions and append them to the lists
        for i, r in enumerate(results):
            predicted_class_index = torch.argmax(r.probs).item()
            true_labels.append(labels[i].item())
            pred_labels.append(predicted_class_index)

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Calculate and print the classification report
class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
report = classification_report(true_labels, pred_labels, target_names=class_names)
print(report)

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
print(conf_matrix)
