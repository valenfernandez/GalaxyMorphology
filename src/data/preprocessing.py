import numpy as np
import cv2
from src.config import IMAGE_SIZE

def normalize_images(images):
    """
    This converts pixels from 0 to 255, to 0.0 to 1.0 to prepare for training (small, continuous inputs)
    
    :param images: images to normalize
    """
    return images.astype("float32") / 255.0



def resize_images(images, size=IMAGE_SIZE):
    '''
    This resizes images 256x256 → 128x128 to reduce memory for training
    
    '''

    #create a numpy array of zeros in the shape (num of images, image size defined in config, color channels (3))
    resized = np.zeros((images.shape[0], size[0], size[1], 3), dtype=np.float32)
    
    for i in range(images.shape[0]):
        resized[i] = cv2.resize(images[i], size)  #loops to every image and resizes it 
    return resized


def preprocess_images(images):

    images = resize_images(images)
    images = normalize_images(images)
    return images