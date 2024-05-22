"""
Custom function: Rotate
version: 0.0.1
update: 2024-03-06
"""
import numpy as np

__all__ = ['rotate_with_labels', 'random_rotate']


def rotate_with_labels(images, labels):
    """ Rotate the input PIL image with labels.
    Args:
        images (list): list of input PIL images.
        labels (np.array): labels for rotation.
    """
    results = []
    for img, label in zip(images, labels):
        if label == 1:
            img = img.rotate(90)
        elif label == 2:
            img = img.rotate(180)
        elif label == 3:
            img = img.rotate(270)
        results.append(img)
    return results


def random_rotate(image):
    """ Random rotate the input PIL image.
    Args:
        image (PIL.Image): input image.
    """
    label = np.random.randint(4)
    return rotate_with_labels([image], [label])[0]
