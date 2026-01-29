#train 80% test 20%

from sklearn.model_selection import train_test_split #shuffles and splits
from src.config import TEST_SIZE, RANDOM_SEED

def split_data(images, labels):
    return train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        stratify=labels, #keeps class proportions
        random_state=RANDOM_SEED
    ) # x_train, X_test, y_train, y_test