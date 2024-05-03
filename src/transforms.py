from dataclasses import dataclass, field
from torchvision import transforms


train_transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.Lambda(lambda x: x / 255.0)
])
train_transforms_gpt = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

simple_transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.Lambda(lambda x: x / 255.0)
])