import os
import shutil

# Define the source and destination directories
source_dir = '/home/thales1/Dwarfsignal/labels'
dest_dir = '/home/thales1/Dwarfsignal/reorganized/labels'

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Get all label files from the source directory
label_files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]

# Sort the files based on their initial number
sorted_labels = sorted(label_files, key=lambda x: int(x.split('_')[0]))

# Move the files to the destination directory in sorted order
for label in sorted_labels:
    shutil.move(os.path.join(source_dir, label), os.path.join(dest_dir, label))

print("Labels have been sorted and moved successfully.")
