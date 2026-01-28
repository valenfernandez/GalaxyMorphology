import h5py
import numpy as np

from src.config import DATA_PATH


def load_galaxy10():

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    with h5py.File(DATA_PATH, "r") as f:
        images = np.array(f["images"])
        labels = np.array(f["ans"])
    return images, labels