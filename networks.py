# networks.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/28/2020
#
# Contains network classes for use in the OTB tracker.


import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkClassifier(nn.Module)
    """
    Class representing the network to train.
    Layers are class attributes and called by the forward method.
    """

    def __init__(self, max_motion=32):
        """
        Class constructor, initialize network layers.
        """
        
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16,
                               kernel_size=3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=2)
        self.conv2_bn = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(num_features=64)
        self.fc1 = nn.Linear(in_features=64*31*31, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=42)

        self.max_motion = max_motion

    def forward(self, t):
        """
        Forward pass through the network.

        Args
            t: torch.Tensor, the input tensor to the network
                expected shape:
                    ([<batch_size>,<n_channels * 2>,<width>,<height>])
                n_channels is multiplied by 2 because the base and target
                images are concatenated in the channels dimension

        Return the network's rank-3 output torch.Tensor with the shape:
            ([<batch_size>,<2>,<2 * max_motion + 1>])
            Where batch size dimension indicates the data point's index within
            the batch,
            the second dimension indicates the direction of the motion, either
            x (index 0) or y (index 1),
            and the third dimension is the one-hot encoded prediction of the
            object's motion in the x or y dimension, from -max_motion to
            max_motion (inclusive).

            E.g.
            for a batch size of 64 and a maximum object motion of 10 pixels
            in any direction, the output tensor would be of shape:
                output.shape => torch.Size([64, 2, 21])
            The predicted motion of the 42nd data point in the x direction
            is given by
                argmax(output[41, 0, :])
            The same data point's motion in the y direction is given by
                argmax(output[41, 1, :])
        """

        # (1) convolutional layer
        t = self.conv1(t)
        t = self.conv1_bn(t)
        t = F.relu(t)

        # (2) convolutional layer
        t = self.conv2(t)
        t = self.conv2_bn(t)
        t = F.relu(t)

        # (3) convolutional layer
        t = self.conv3(t)
        t = self.conv3_bn(t)
        t = F.relu(t)

        # (4) fully connected layer
        t = t.reshape(-1, 64*31*31)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) fully connected layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        t = t.reshape(-1, 2, 2 * self.max_motion + 1)
        return t


























