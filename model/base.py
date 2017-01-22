#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.base
~~~~~~~~~~~~~~~

This module contains the base class to build and train models
for the Udacity driving simulator.
"""
import json
import os
import math

import cv2
import pandas as pd
import numpy as np
from numpy.random import randint

from keras import callbacks

from sklearn.model_selection import train_test_split

from .utils import (load_image, change_image_brightness,
                    translate_image, bool_flip_image)

# pylint: disable=C0103


class SaveModelToJson(callbacks.Callback):
    """
    Keras callback to save the json model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.json`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
    """
    def __init__(self, filepath, monitor='val_loss', verbose=0):
        super(SaveModelToJson, self).__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch, **logs)
        with open(filepath, 'w') as f:
            json.dump(self.model.to_json(), f)
        if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch, filepath))


class SteeringSimulatorBase(object):
    """The Steering base model"""

    df = None
    name = None
    model = None
    n_rows = None
    n_cols = None

    def __init__(self):
        filename = 'data/driving_log.csv'
        self.df = pd.read_csv(filename)
        self.reduce_zeros()
        self.build_model()

    def reduce_zeros(self):
        """ reduce the number of rows with steering angle zero"""
        zero = self.df[self.df.steering == 0]
        msk = np.random.rand(len(zero)) < 0.1
        zero = zero[msk]
        nonzero = self.df[self.df.steering != 0]
        self.df = zero.append(nonzero, ignore_index=True)

    def build_model(self):
        """
        This is the function that the other classes must implement
        """
        raise NotImplementedError()

    def preprocess_image(self, image):
        """
        Crop the top 1/4 of the image to remove the horizon
        and the bottom 25 pixels to remove the car’s hood.
        We will next rescale the image to a 64X64 square image.

        Returns an array.
        """
        new_size = (self.n_cols, self.n_rows)
        n_rows = image.shape[0]
        image = image[math.floor(n_rows / 5):n_rows - 25, :, :]
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return np.array(image)

    def training_preprocess_image(self, row):
        """
        Preprocessing images for the training process.
        It randomly chooses between the three available images
        (left, center, right), corrects the steering angle, and
        applies augmentations.
        """
        options = [
            [row.left.strip(), row.steering + .25],
            [row.center.strip(), row.steering],
            [row.right.strip(), row.steering - .25]
        ]
        img_path, y_steer = options[randint(len(options))]
        img = load_image(img_path)
        # apply augmentations
        img, y_steer, tr_x = translate_image(img, y_steer, 50)
        img, y_steer = bool_flip_image(img, y_steer)
        img = change_image_brightness(img)
        img = self.preprocess_image(img)
        return img, y_steer

    def training_data_generator(self, train_df, batch_size=256):
        """
        This is a generator to create a batch of new images and its
        steering angles. It's used for the training process.
        """
        while 1:
            X_batch = []
            y_batch = []
            rows = train_df.sample(batch_size)
            for _, row in rows.iterrows():
                x, y = self.training_preprocess_image(row)
                X_batch.append(x)
                y_batch.append(y)
            yield np.array(X_batch), np.array(y_batch)

    def validation_preprocess_image(self, row):
        """
        This function loads the center image from a pandas dataframe
        and preprocesses to fix the size.
        """
        img = load_image(row.center.strip())
        img = self.preprocess_image(img)
        return img

    def validation_data_generator(self, valid_df, batch_size=256):
        """
        This is a generator to create a batch of new images and its
        steering angles. It's used for validation.
        """
        while 1:
            X_batch = []
            y_batch = []
            rows = valid_df.sample(batch_size)
            for _, row in rows.iterrows():
                x = self.validation_preprocess_image(row)
                X_batch.append(x)
                y_batch.append(row.steering)
            yield np.array(X_batch), np.array(y_batch)

    def train_model(self, epochs=1, batch_size=256):
        """
        The training function.
        """

        # Shuffle and split the dataset into Training and Validation Dataframes
        # so the data doesn't mantains the order it was collected
        train_df, validation_df = train_test_split(self.df, test_size=0.2, random_state=12897)

        # initialize validation values
        idx_best = 0
        val_best = 10000

        print("Training model:{}, image size({}, {})".format(self.name, self.n_rows, self.n_cols))
        print(self.model.summary())

        BASEDIR = './out'
        if not os.path.exists(BASEDIR):
            os.makedirs(BASEDIR)

        # Save the model as json after each epoch.
        base_name = 'model_' + self.name + '_{epoch:02d}-{val_loss:.2f}'
        filename = os.path.join(BASEDIR, base_name + '.json')
        save_json = SaveModelToJson(filename, monitor='val_loss', verbose=1)

        # Save the model after each epoch.
        filename = os.path.join(BASEDIR, base_name + '.h5')
        save_best = callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=1, mode='min')

        self.model.fit_generator(
            self.training_data_generator(train_df, batch_size),
            samples_per_epoch=batch_size * 50,
            nb_epoch=epochs,
            validation_data=self.validation_data_generator(validation_df, batch_size),
            nb_val_samples=len(validation_df),
            callbacks=[save_json, save_best]
        )
