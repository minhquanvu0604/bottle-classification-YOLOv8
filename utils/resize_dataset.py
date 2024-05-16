import os
from PIL import Image

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, ((size - x) // 2, (size - y) // 2))
    return new_im

def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # Create the corresponding directory structure in the output directory
        rel_path = os.path.relpath(root, input_dir)
        dest_path = os.path.join(output_dir, rel_path)
        os.makedirs(dest_path, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(dest_path, file)
                
                with Image.open(input_file_path) as im:
                    square_im = make_square(im)
                    square_im.save(output_file_path)

# Define the input and output directories
input_base_dir = '/root/aifr/bottle-classification-YOLOv8/data_original'
output_base_dir = '/root/aifr/bottle-classification-YOLOv8/data'

# Process the train and val directories
for sub_dir in ['train', 'val']:
    process_directory(os.path.join(input_base_dir, sub_dir), os.path.join(output_base_dir, sub_dir))

print("Processing completed successfully.")
