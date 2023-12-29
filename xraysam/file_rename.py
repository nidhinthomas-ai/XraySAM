
import os
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Rename .png files in a specified directory.')
parser.add_argument('directory', type=str, help='The path to the directory containing .png files')
args = parser.parse_args()

# Use the directory from the command line argument
directory = args.directory

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a PNG image and doesn't already have "_mask" before ".png"
    if filename.endswith(".png") and not filename.endswith("_mask.png"):
        # Construct the new file name
        new_name = filename.replace(".png", "_mask.png")
        
        # Construct the full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")

print("All applicable PNG files have been renamed.")
