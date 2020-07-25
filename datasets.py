# datasets.py
# 
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/23/2020
#
# Contains dataset classes for use in OTB tracker.
#
# Before use
# Must have a directory of images (all the same size) and
# results.csv file, where each line of results.csv takes the form
#       <path to base image>,<path to target image>,<vx>,<vy>\n


import torch
import numpy as np
import pandas as pd
import cv2


class TrackingDataset(torch.utils.data.Dataset):
    """
    Class for a generic single-object tracking dataset,
    extends the pytorch Dataset class.
    """
    def __init__(self, csv_path, transform=None):
        """
        Class constructor
        
        Args
            csv_path: string, path to results.csv file of format
                <path to base image>,<path to target image>,<vx>,<vy>\n
                where vx, vy are horizontal and vertical components of the
                object's motion from the base frame to the target frame
            transform: transformation function
                to apply on data upon __getitem__ call
        """

        self.labels = np.genfromtxt(
            fname=csv_path,
            delimiter=",",
            dtype=str)
        self.transform = transform

    def __getitem__(self, index):
        """
        Gets an item from the dataset at a specified index.

        Returns data packet of format [imgs, v] where
            imgs: pytorch tensor of images with the following shape
                ([batch_size, n_channels * 2, width, height])
                n_channels is multiplied by 2 because the base and target
                images are concatenated in the channels dimension
            v: list vx, vy
                where vx, vy represent horizontal and vertical components of
                the objects motion from the base frame to the target frame
        """

        # convert index/indexes to python list
        if torch.is_tensor(index):
            index = index.tolist()

        # get base and target image paths from labels
        base_path = self.labels[index, 0]
        target_path = self.labels[index, 1]
        
        # load base and target images into pytorch tensors
        base_img = torch.from_numpy(cv2.imread(base_path, -1))
        target_img = torch.from_numpy(cv2.imread(target_path, -1))

        # swap dimensions of tensors to match shape:
        #       ([batch_size, n_channels, width, height])
        base_img = base_img.permute(2, 0, 1)
        target_img = target_img.permute(2, 0, 1)

        # base and target concatenated in the channels dimension
        imgs = torch.cat((base_img, target_img), dim=0).float()

        # vector of object's motion from base to target frame
        v = self.labels[index, 2:].astype(float)

        # packet to return, transformed if dataset has a transform function
        data = [imgs, v]
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        """
        Returns the number of base/target pairs in dataset.
        """
        return len(self.labels)













