import pandas as pd
import os
import tifffile

# Define the directory where the parquet files are located
data_dir = os.getcwd()  # Current working directory

# Define the directory where you want to save the TIFF files
output_dir = os.getcwd()

# Function to convert parquet files to TIFF files
def convert_parquet_to_tiff(parquet_file, output_dir):
    # Load parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)
    
    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        # Load image and label data
        image_data = row['tif']
        label_data = row['label_tif']
        
        # Extract filename without extension
        filename = os.path.splitext(row['filename'])[0]
        
        # Save image and label data as TIFF files
        image_path = os.path.join(output_dir, f"{filename}.tif")
        label_path = os.path.join(output_dir, f"{filename}_mask.tif")
        tifffile.imwrite(image_path, image_data)
        tifffile.imwrite(label_path, label_data)

# Iterate over train parquet files
for i in range(6):
    parquet_file = os.path.join(data_dir, f"train-0000{i}-of-00006.parquet")
    convert_parquet_to_tiff(parquet_file, output_dir)

# Convert validation parquet file
val_parquet_file = os.path.join(data_dir, "val-00000-of-00001.parquet")
convert_parquet_to_tiff(val_parquet_file, output_dir)
