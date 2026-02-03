#DEFINES THE MODEL ARCHITECTURE

import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMAGE_SIZE, NUM_CLASSES


def build_cnn_model():
    """
    Builds and returns a CNN model for galaxy morphology classification.
    """

    model = models.Sequential()  # straight pipeline
   
    # Input layer + Conv Block 1
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv Block 2
    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv Block 3
    model.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu"
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    # Classification Head
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return model
