from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.data.loader import load_galaxy10
from src.data.preprocessing import resize_images
from src.data.split import split_data


print("Loading data...")
images, labels = load_galaxy10()

images = resize_images(images)
# images = normalize_images(images) # when using a pretrained model we do not need to normalize by hand

X_train, X_test, y_train, y_test = split_data(
    images, labels
)

X_train = preprocess_input(X_train)
X_test  = preprocess_input(X_test)

print("Loading model...")
model = load_model("models/galaxy_cnn_v2.keras", custom_objects={"preprocess_input": preprocess_input})

base_model = None
for layer in model.layers:
    if isinstance(layer, EfficientNetB0):
        base_model = layer
        break

if base_model is None:
    raise ValueError("EfficientNet backbone not found in model")

base_model.trainable = False #freeze

# Unfreeze top N layers
FINE_TUNE_AT = len(base_model.layers) - 25

for layer in base_model.layers[FINE_TUNE_AT:]:
    layer.trainable = True


model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=[early_stop]
)


model.save("models/galaxy_cnn_v2_1_finetuned.keras")