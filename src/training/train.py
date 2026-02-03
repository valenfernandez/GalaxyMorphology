#training loop

import tensorflow as tf

from src.training.model import build_cnn_model
from src.data.loader import load_galaxy10
from src.data.preprocessing import normalize_images, resize_images
from src.data.split import split_data
from src.config import BATCH_SIZE, EPOCHS, MODEL_PATH


print("Loading data...")
images, labels = load_galaxy10()

images = resize_images(images)
images = normalize_images(images)

X_train, X_test, y_train, y_test = split_data(
    images, labels
)

model = build_cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")