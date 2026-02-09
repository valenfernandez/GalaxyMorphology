import matplotlib.pyplot as plt
import random
from src.config import GALAXY10_CLASSES

def show_random_images(images, labels, n=5):
    plt.figure(figsize=(15, 3))
    for i in range(n):
        idx = random.randint(0, len(images) - 1)
        plt.subplot(1, n, i + 1)
        plt.imshow(images[idx])
        plt.title(GALAXY10_CLASSES[labels[idx]])
        plt.axis("off")
    plt.show()



def plot_training_curves(history, model_name="Model"):
    """
    Plots training and validation accuracy and loss curves.

    :param history: Keras History object
    :param model_name: Name of the model (for titles)
    """

    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # --- Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["accuracy"], label="Train Accuracy")
    plt.plot(epochs, metrics["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy")
    plt.legend()
    plt.grid(True)

    # --- Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
