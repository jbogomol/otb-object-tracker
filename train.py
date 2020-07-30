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
import torchvision
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


# keep same random seed for replicable results
random_seed = 1
np.random.seed(random_seed)
torch.manual_seed(random_seed)


# paths to csv files
csv_path_list = func_file.get_files_sorted(directory=datadir, extension=".csv")
np.random.shuffle(csv_path_list)
 
# create pytorch datasets for train, validation, test. split by video
n_videos = len(csv_path_list)
n_videos_test = n_videos // 3
n_videos_validation = (n_videos - n_videos_test) // 6
n_videos_train = n_videos - n_videos_test - n_videos_validation

# initialize empty sets
train_set = torchvision.datasets.FakeData(size=0)
validation_set = torchvision.datasets.FakeData(size=0)
test_set = torchvision.datasets.FakeData(size=0)

# fill datasets video by video, looping through all csv paths
for i, csv_path in enumerate(csv_path_list):
    # get set for current csv path
    set_i = datasets.TrackingDataset(csv_path=csv_path)

    # determine where to put set
    if i < n_videos_train:
        train_set = torch.utils.data.dataset.ConcatDataset(
                        [train_set, set_i])
    elif i < n_videos_train + n_videos_validation:
        validation_set = torch.utils.data.dataset.ConcatDataset(
                        [validation_set, set_i])
    else:
        test_set = torch.utils.data.dataset.ConcatDataset(
                        [test_set, set_i])


# 
print("total videos: ", n_videos)
print("# training videos:   ", n_videos_train)
print("# validation videos: ", n_videos_validation)
print("# testing videos:    ", n_videos_test)
print("image pairs in train set:      ", len(train_set))
print("image pairs in validation set: ", len(validation_set))
print("image pairs in test set:       ", len(test_set), "\n")




















