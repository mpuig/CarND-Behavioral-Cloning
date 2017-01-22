#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.custom
~~~~~~~~~~~~~~~

This module contains the primary objects that defines the Nvidia model.
"""
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (Dense, Dropout, Flatten, Lambda, Convolution2D,
                          ELU, MaxPooling2D)


from .base import SteeringSimulatorBase


class CustomModel(SteeringSimulatorBase):
    """Build a keras model to drive a simulator based on Nvidia paper."""

    def __init__(self):
        self.name = 'custom'
        self.n_rows = 64
        self.n_cols = 64

        super(CustomModel, self).__init__()

    def build_model(self):
        """Build the custom model."""
        input_shape = (self.n_rows, self.n_cols, 3)
        filter_size = 3

        pool_size = (2, 2)
        model = Sequential()
        model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))
        model.add(Convolution2D(3, 1, 1, border_mode='valid', name='conv0', init='he_normal'))
        model.add(ELU())
        model.add(Convolution2D(32, filter_size, filter_size,
                                border_mode='valid', name='conv1', init='he_normal'))
        model.add(ELU())
        model.add(Convolution2D(32, filter_size, filter_size,
                                border_mode='valid', name='conv2', init='he_normal'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.5))
        model.add(Convolution2D(64, filter_size, filter_size,
                                border_mode='valid', name='conv3', init='he_normal'))
        model.add(ELU())
        model.add(Convolution2D(64, filter_size, filter_size,
                                border_mode='valid', name='conv4', init='he_normal'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.5))
        model.add(Convolution2D(128, filter_size, filter_size,
                                border_mode='valid', name='conv5', init='he_normal'))
        model.add(ELU())
        model.add(Convolution2D(128, filter_size, filter_size,
                                border_mode='valid', name='conv6', init='he_normal'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512, name='hidden1', init='he_normal'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(64, name='hidden2', init='he_normal'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(16, name='hidden3', init='he_normal'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(1, name='output', init='he_normal'))

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='mse')

        self.model = model

