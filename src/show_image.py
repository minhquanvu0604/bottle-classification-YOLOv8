from PIL import Image
import matplotlib.pyplot as plt

# Path to your image file
image_path = '/root/aifr/bottle-classification-YOLOv8/src/bus.jpg'

# Open an image file
with Image.open(image_path) as img:
    # Display image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
