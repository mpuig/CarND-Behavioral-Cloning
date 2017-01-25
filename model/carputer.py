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
        dropout = 0.2

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
        model.add(Dropout(dropout))
        model.add(Dense(1))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
        model.compile(optimizer=adam, loss='mse')

        self.model = model
