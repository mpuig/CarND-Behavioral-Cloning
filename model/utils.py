#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.base
~~~~~~~~~~~~~~~

This module contains the base class to build and train models
for the Udacity driving simulator.
"""
import cv2
import numpy as np
from numpy.random import uniform, randint


def load_image(filename):
    """ Read the filename and convert to RGB"""
    image = cv2.imread('data/' + filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def change_image_brightness(image):
    """
    Changing brightness to simulate day and night conditions.
    We will generate images with different brightness by
    first converting images to HSV, scaling the V channel
    (brightness) by a random number between .25 and 1.25
    and converting back to the RGB channel.
    """
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:, :, 2] = image[:, :, 2] * uniform(.25, 1.25)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


def translate_image(image, angle, translate_range):
    """
    Shifting left/right and up/down.
    Original idea by Vivek Yadav: https://goo.gl/HZe6wU

    Camera images are shifted to left or right to simulate
    the effect of the car at different positions in the lane.
    Random horizontal shifts are applied to simulate lane shifts,
    direction of upto 10 pixels, and applied angle change of .2 per pixel.
    """
    tr_x = translate_range * uniform() - translate_range / 2
    new_angle = angle + tr_x / translate_range * 2 * .2
    tr_y = 10 * uniform() - 10 / 2
    translation_matrix = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    rows, cols, _ = image.shape
    image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return image, new_angle, tr_x


def bool_flip_image(img, angle):
    """
    Randomly flipped images about the vertical midline
    to simulate driving in the opposite direction.
    """
    if randint(2) == 0:
        return cv2.flip(img, 1), -angle
    return img, angle
