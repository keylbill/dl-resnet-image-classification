### YOUR CODE HERE
import math
import random

import numpy as np
import tensorflow
# import tensorflow.experimental.numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ImageUtils import parse_record
from Network import MyNetwork

"""This script defines the training, validation and testing process.
"""


class MyModel(object):

    def __init__(self, parameters, configs):
        self.configs = configs
        self.net = MyNetwork(self.configs)
        self.model = self.net()
        self.parameters = parameters
        self.batch_size = configs["batch_size"]

    def model_setup(self, training):
        print('Setting up the network...')
        if training:
            print('Setting up for training...')
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(learning_rate=learning_rate_scheduler(0)), metrics=['accuracy'])
            # look into SGD+momentum
        else:
            print("Previous model found in 'trained_models' folder. Using that for training...")
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(learning_rate=learning_rate_scheduler(0)), metrics=['accuracy'])
            self.model.load_weights("../saved_models/model.h5")

    def train(self, x_train, y_train, max_epoch):
        print('Training...')
        self.model_setup(True)
        self.parameters.run(tensorflow.compat.v1.global_variables_initializer())

        sample_count = x_train.shape[0]
        batch_count = int(sample_count / self.batch_size)

        x_zeros = np.zeros((x_train.shape[0] + batch_count, 3072))
        for j in range(sample_count):
            x_zeros[j] = x_train[j]
        # print(x_zeros)

        y_zeros = np.zeros((y_train.shape[0] + batch_count, 10))
        for j in range(y_train.shape[0]):
            y_zeros[j] = y_train[j]
        # print(y_zeros)
        for i in range(batch_count):
            random_temp = np.random.beta(0.1, 0.1)
            randint_one = random.randint(i * self.configs["batch_size"], (i + 1) * self.configs["batch_size"] - 1)
            while True:
                randint_two = random.randint(i * self.configs["batch_size"], (i + 1) * self.configs["batch_size"] - 1)
                if randint_two != randint_one:
                    break
            x_zeros[x_train.shape[0] + i] = x_train[randint_one] * random_temp + x_train[randint_two] * (
                        1 - random_temp)
            y_zeros[y_train.shape[0] + i] = y_train[randint_one] * random_temp + y_train[randint_two] * (
                        1 - random_temp)

        random_shuffle = np.random.permutation(sample_count + batch_count)
        curr_x_train = x_zeros[random_shuffle]
        y_train = y_zeros[random_shuffle]
        x_train = np.asarray([parse_record(i, True) for i in curr_x_train])
        print("Data parsing complete...")

        lr_scheduler = LearningRateScheduler(learning_rate_scheduler)
        filepath = '../saved_models/model.h5'
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='accuracy',
                                     verbose=1,
                                     save_best_only=True)
        callbacks = [lr_scheduler, checkpoint]
        generator = ImageDataGenerator(
            featurewise_center=True,
            samplewise_center=False,
            featurewise_std_normalization=True,
            samplewise_std_normalization=False,
            zca_whitening=False,
            zca_epsilon=1e-06,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=False,
            rescale=None,
            preprocessing_function=None,
            data_format=None,
            validation_split=0.2)
        generator.fit(x_train)
        self.model.fit(generator.flow(x_train, y_train, batch_size=self.batch_size),
                                 epochs=max_epoch, verbose=1, workers=4,
                                 callbacks=callbacks)

    def evaluate(self, x, y):
        print('Validation (or) Testing...')
        self.model_setup(False)
        self.parameters.run(tensorflow.compat.v1.global_variables_initializer())

        x = np.asarray([parse_record(i, False) for i in x])
        scores = self.model.evaluate(x, y, verbose=2)
        print('Test Loss: ', scores[0])
        print('Test Accuracy: ', scores[1])

    def predict_prob(self, x):
        print('Predicting the private data set...')

        self.model_setup(False)
        self.parameters.run(tensorflow.compat.v1.global_variables_initializer())
        # x = np.expand_dims(x, axis=0) # adding an extra dimension
        # x = np.expand_dims(x, axis=3)
        mean = np.mean(x)
        std = np.std(x)
        x = np.divide(np.subtract(x, mean), std)
        # x = np.expand_dims(x, axis=0) # adding an extra dimension
        # x = np.expand_dims(x, axis=3)
        scores = np.round(self.model.predict(x), 3)
        print('Predictions sample:', scores)
        print("Predictions have been saved to predictions.npy")
        return scores


def learning_rate_scheduler(epoch):
    lr = (1 + math.cos(math.pi * epoch / 200)) * 1e-3 / 2
    return lr
