"""
Docstring for dataset_inspection

KEYS: 
ans : Label (0 to 9) Each integer corresponds to one of the 10 Galaxy10 morphology classes
dec : (Declination) Astronomical coordinate (like latitude in the sky)
images : Image
pxscale : Pixel scale (arcseconds per pixel)
ra : (Right Ascension) Astronomical coordinate
redshift : Distance-related astrophysical value



Name images 
Shape: (17736, 256, 256, 3)
Type: uint8

Name ans 
Type Integer (unsigned), 8-bit, little-endian 
Shape 17736


1. IMG need to be normalized
2. There is one label per image

Data science stack:
Python: numpy pandas matplotlib seaborn
"""
import numpy as np
import h5py

file_path = "./data/Galaxy10_DECals.h5"

with h5py.File(file_path, 'r') as f:
    
    images_dataset = f['images']
    # View attributes of the images
    print(f"\nDataset shape: {images_dataset.shape}")
    print(f"Dataset data type: {images_dataset.dtype}")
    print(f"Dataset attributes: {list(images_dataset.attrs.keys())}") #