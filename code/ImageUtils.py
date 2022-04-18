import numpy as np
# import tensorflow.experimental.numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""


def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.
    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.
    Returns:
        image: An array of shape [32, 32, 3].
    """
    # transfer array shape of (3072, ) to (3, 32, 32)
    image = np.transpose(record.reshape((3, 32, 32)), [1, 2, 0])
    # print("image = ", image)
    # print("image shape = ", image.shape)
    if not training:
        image = preprocess_image(image, training)  # If any.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].
    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """

    mean = np.mean(image)
    std = np.std(image)
    image = np.divide(np.subtract(image, mean), std)
    # print("image = ", image)
    # print("image shape = ", image.shape)
    return image
