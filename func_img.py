# func_img.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/20/2020
#
# contains helper functions for image manipulation


import cv2
import numpy as np


def crop_zero_padding(img, x_crop, y_crop, crop_size):
    """
    Crops an image to a given size from top-left point x_crop, y_crop.
    Pads with zeros if x_crop or y_crop is out of bounds.

    Args
        img: numpy array, the image to crop
        x_crop, y_crop: int x, y point, top-left of cropped image
        crop_size: int, size of resulting image (crop_size = width = height)

    Returns the resulting image as a numpy array
    """

    height, width, n_channels = img.shape
    img_padded = np.zeros([height + 2 * crop_size,
                           width + 2 * crop_size,
                           n_channels])
    img_padded[crop_size : crop_size + height,
               crop_size : crop_size + width,
               :] = img

    img_cropped = np.zeros([crop_size, crop_size, n_channels])
    img_cropped = img_padded[y_crop + crop_size : y_crop + 2 * crop_size,
                             x_crop + crop_size : x_crop + 2 * crop_size,
                             :]

    return img_cropped








