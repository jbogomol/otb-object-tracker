# train.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/28/2020
#
# Trains a convolutional neural network to guess frame-to-frame object motion
# in x and y direction.
# Formulated as a classification problem with a cross-entropy loss function
#
# Before running
# Run create_images.py to create necessary data and csv files containing
# ground truths.


import networks
import datasets
import func_file
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import cv2
import sys
import random


# on server or local computer
on_server = torch.cuda.is_available()

# which gpu
if on_server:
    torch.cuda.set_device(0)

# directory with images and results csv files
if on_server:
    datadir = "/home/datasets/data_jbogomol/OTB_data/results/"
else:
    datadir = "../OTB_data/results/"


# paths to csv files
csv_path_list = func_file.get_files_sorted(directory=datadir, extension=".csv")

# create pytorch datasets for train, validation, test. split by video






















