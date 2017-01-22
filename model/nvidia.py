#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.nvidia
~~~~~~~~~~~~~~~

Model based on Nvidia paper https://arxiv.org/pdf/1604.07316v1.pdf
"""
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (Dense, Dropout, Flatten, Lambda, Convolution2D)


from .base import SteeringSimulatorBase


class NvidiaModel(SteeringSimulatorBase):
    """Build a keras model to drive a simulator based on Nvidia paper."""

    def __init__(self):
        self.name = 'nvidia'
        self.n_rows = 66
        self.n_cols = 220

        super(NvidiaModel, self).__init__()

    def build_model(self):
        """Build the nvidia model."""
        dropout = 0.5

        input_shape = (self.n_rows, self.n_cols, 3)
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
        model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Dropout(dropout))
        model.add(Convolution2D(64, 3, 3, activation='elu'))
        model.add(Convolution2D(64, 3, 3, activation='elu'))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(1164, activation='elu'))
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
        model.compile(optimizer=adam, loss='mse')

        self.model = model


def main():
    nvidia = NvidiaModel()
    nvidia.train_model(epochs=5, batch_size=256)

if __name__ == '__main__':
    main()
