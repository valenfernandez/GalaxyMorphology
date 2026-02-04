#DEFINES THE MODEL ARCHITECTURE

import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMAGE_SIZE, NUM_CLASSES
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

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
            activation=None, #raw conv outputs for batch normalization
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv Block 2
    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation=None
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv Block 3
    model.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation=None
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))


    # Classification Head
    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return model


def build_cnn_model_v2():
    """
    Transfer learning model using EfficientNetB0 as feature extractor.
    """

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )

    # Freeze pretrained backbone
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        layers.Lambda(preprocess_input),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    return model