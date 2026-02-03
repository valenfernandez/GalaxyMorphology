from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Galaxy10_DECals.h5"
MODEL_PATH = "models/galaxy_cnn_v1"

IMAGE_SIZE = (128, 128)
NUM_CLASSES = 10
RANDOM_SEED = 42 # fixed integer seed
TEST_SIZE = 0.2 #20%

BATCH_SIZE = 32

EPOCHS = 15

GALAXY10_CLASSES = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar-shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge"
} # this can be moved to a class_mapping.py