#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.carputer
~~~~~~~~~~~~~~~

Model based on Nvidia paper https://arxiv.org/pdf/1604.07316v1.pdf
"""
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (Dense, Dropout, Flatten, Lambda, Convolution2D, Activation, MaxPooling2D)


from .base import SteeringSimulatorBase


class CarputerModel(SteeringSimulatorBase):
    """
    To do this we'll use a 3 layer convolution network with one fully connected layer.
    This model is based off of Otavio's Carputer but does not produce a throttle
    value output, does not use past steering values as input into the model,
    and uses one less convolution layer.
    https://github.com/otaviogood/carputer/blob/master/NeuralNet/convnetshared1.py
    """

    def __init__(self):
        self.name = 'carputer'
        self.n_rows = 120
        self.n_cols = 160

        super(CarputerModel, self).__init__()

    def build_model(self):
        """Build the carputer model."""
        dropout = 0.5

        input_shape = (self.n_rows, self.n_cols, 3)
        pool_size = (2, 2)
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
        model.add(Convolution2D(8, 8, 8, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Convolution2D(16, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(265))
        model.add(Activation('linear'))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        self.model = model


def main():
    carputer = CarputerModel()
    carputer.train_model(epochs=5, batch_size=256)

if __name__ == '__main__':
    main()
