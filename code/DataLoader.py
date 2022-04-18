import os
import pickle
import tensorflow as tf
# import tensorflow.experimental.numpy as np
import numpy as np


"""This script implements the functions for reading data.
"""


def load_data(data_dir):
    """Load the CIFAR-10 dataset.
    Args:
        data_dir: A string. The directory where data batches
            are stored.
    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """
    # turning batches into np array for x_train, y_train, x_test, y_test
    train_data = []
    train_labels = []
    for batch in os.listdir(data_dir):
        path = os.path.join(data_dir, batch)
        if len(batch.split(".")) == 1:
            batch_f = open(path, "rb")
            batch = pickle.load(batch_f, encoding='bytes')
            train_data.append(batch[b'data'])  # load data bytes into train_data
            train_labels.append(batch[b'labels'])  # load label bytes into train_labels
    x_test = train_data.pop(-1)
    y_test_temp = train_labels.pop(-1)
    x_train = np.reshape(train_data, (50000, 3072))
    y_train_temp = np.reshape(train_labels, (50000,))
    y_train = np.zeros((50000, 10))
    y_test = np.zeros((10000, 10))

    # one-hot encoding the labels for easier classification process in mix-up training
    for i in range(len(y_train_temp)):
        y_labels = tf.keras.utils.to_categorical(y_train_temp[i], num_classes=10)
        y_train[i] = y_labels

    for i in range(len(y_test_temp)):
        y_labels = tf.keras.utils.to_categorical(y_test_temp[i], num_classes=10)
        y_test[i] = y_labels

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir, data_file='private_test_images_v3.npy'):
    """Load the images in private testing dataset.
    Args:
        data_dir: A string. The directory where the testing images
        are stored.
    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE

    if len(data_dir) > 4 and data_dir[-4:] == '.npy':
        test_file = data_dir
    else:
        test_file = os.path.join(data_dir, data_file)
    x_test = np.load(test_file).reshape((2000, 32, 32, 3))  # reshape (2000, 3072)
    # print("x_test shape = ", x_test.shape)
    # x_test = np.reshape(x_test, (1, 32, 32, 3))
    # print("x_test shape = ", x_test.shape)
    ### END CODE HERE

    return x_test
