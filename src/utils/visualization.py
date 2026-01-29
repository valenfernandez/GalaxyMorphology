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
