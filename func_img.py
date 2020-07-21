# func_img.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/20/2020
#
# contains helper functions for image manipulation


import cv2
import numpy as np


# creates base and target images cropped the same way, such that the object
# being tracked is in the center of the base image (and thus slightly off-
# center in the target image)
# args:
#       base_path - string, path to base image
#       target_path - string, path to target image
#       base_label - numpy array of the form:
#               [<x>, <y>, <obj_width>, <obj_height>]
#               taken from the groundtruth_rect.txt file
# return:
#       tuple of the form:
#               base_cropped, target_cropped, vx, vy
#       where
#               base_cropped - numpy array representing cropped base image
#               target_cropped - numpy array representing cropped target image
#               vx, vy - integer x and y motion of object from base to target
def crop_to_object(base_path, target_path, base_label):
    # TODO implement
    print("base_path: ", base_path)
    print("target_path: ", target_path)
    print("base_label: ", base_label)
    return 0, 0, 0, 0









