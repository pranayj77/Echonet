import os
import random
import shutil

# Set the source and destination directories
src_dir = '/athena/sablab/scratch/bcl2004/datasets/echonet/EchoNet-Dynamic/Videos/'
dst_dir = '/athena/sablab/scratch/prj4005/Echonet/dynamic/videos'

# Get a list of all the files in the source directory
files = os.listdir(src_dir)

# Select 50 random files from the list
selected_files = random.sample(files, 50)

# Copy the selected files to the destination directory
for file in selected_files:
    src_path = os.path.join(src_dir, file)
    dst_path = os.path.join(dst_dir, file)
    shutil.copy(src_path, dst_path)
