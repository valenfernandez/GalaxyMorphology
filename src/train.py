# src/train.py

from src.data.loader import load_galaxy10
from src.data.preprocessing import normalize_images, resize_images
from src.data.split import split_data
from src.utils.visualization import show_random_images

def main():
    images, labels = load_galaxy10()

    images = normalize_images(images)
    images = resize_images(images)

    X_train, X_val, y_train, y_val = split_data(images, labels)

    show_random_images(X_train, y_train)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

if __name__ == "__main__":
    main()
