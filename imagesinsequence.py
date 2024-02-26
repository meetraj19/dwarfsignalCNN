import os
import shutil

# Define the source and destination directories
source_dir = '/home/thales1/Dwarfsignal/images'
dest_dir = '/home/thales1/Dwarfsignal/reorganized/images'

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Get all image files from the source directory
image_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]

# Sort the files based on their initial number
sorted_images = sorted(image_files, key=lambda x: int(x.split('_')[0]))

# Move the files to the destination directory in sorted order
for image in sorted_images:
    shutil.move(os.path.join(source_dir, image), os.path.join(dest_dir, image))

print("Images have been sorted and moved successfully.")
