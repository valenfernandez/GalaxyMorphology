# src/testing.py

from src.data.loader import load_galaxy10
from src.data.preprocessing import preprocess_images
from src.data.split import split_data
from src.utils.visualization import show_random_images

import numpy as np

def main():

     #if there is no preprocessed images already:
    print("=== Loading data ===")
    images, labels = load_galaxy10()

    print("Images:", images.shape, images.dtype)
    print("Labels:", labels.shape)

    print("\n=== Preprocessing ===")
    images = preprocess_images(images)

    print("Processed images:", images.shape, images.dtype)
    print("Pixel range:", images.min(), images.max())

    # if not simply load the existing images and lables
    print("\n=== Splitting ===")
    X_train, X_test, y_train, y_test = split_data(images, labels)

    print("Train:", X_train.shape, y_train.shape)
    print("Test:", X_test.shape, y_test.shape)

    print("\n=== Class distribution check ===")
    print("Train:", np.bincount(y_train))
    print("Test:", np.bincount(y_test))

    show_random_images(X_train, y_train)

if __name__ == "__main__":
    main()
