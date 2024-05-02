from dataclasses import dataclass, field
from torchvision import transforms

@dataclass
class TransformManager:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, image):
        if self.transform:
            image = self.transform(image)
        return image

basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(lambda x: x / 255.0)
])