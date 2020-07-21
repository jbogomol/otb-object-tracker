# create_images.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/20/2020
#
# Makes 2-frame pairs from the videos on otb_list where:
#     - Frame 1 (base) is cropped to a specified size, where the object being
#       tracked is in the center of the image.
#     - Frame 2 (target) is cropped the same as frame 1, where now the object
#       being tracked is slightly off center, assuming it has moved from frame
#       to frame.
# The base and target frames are selected using the following algorithm:
#     - For each frame t in the video:
#             - Frame t is the base frame.
#             - Frames t+1, t+2, t+3, t-1, t-2, t-3 are all used as target
#               frames.
#
# Usage
# > python create_images.py
#
# Before running
# Change global variables to correct file paths.
# Change otb_list to correct list of OTB videos being used.


import os
import cv2
import numpy as np
import torch
import func_file
import func_img


# on server or local computer
on_server = torch.cuda.is_available()

# OTB dataset directory
if on_server:
    imdir = "/home/datasets/OTB/"
else:
    imdir = "../OTB/"

# directory to store results in
if on_server:
    resultsdir = "/home/datasets/data_jbogomol/OTB_data/results/"
else:
    resultsdir = "../OTB_data/results/"

# list of OTB videos to use
# must correspond to the name of the folder within imdir
if on_server:
    otb_list = ["Basketball"]
else:
    otb_list = ["Basketball"]

# size to crop images to (square, height = width)
crop_size = 128


# empty results image directory
func_file.empty_folder(resultsdir)

# initialize results.csv file
csv_path = os.path.join(resultsdir, "results.csv")
csv = open(csv_path, "w+")

# loop through all videos on otb_list
for video in otb_list:
    print(video)

    # get lists of images and their labels
    img_list = func_file.get_files_sorted(
        directory=imdir + "Basketball/img/",
        extension=".jpg")
    labels = np.genfromtxt(
        fname=imdir + "Basketball/groundtruth_rect.txt",
        dtype="int",
        delimiter=",")
    num_imgs = len(img_list)
    
    # loop through all frames
    for i in range(num_imgs):
        # load base & target image paths
        base_path = img_list[i]
        base_label = labels[i]
        target_img_paths = [img_list[i+1]]
        target_labels = [labels[i+1]]

        x_obj, y_obj, width_obj, height_obj = base_label
        x_center = x_obj + width_obj//2
        y_center = y_obj + height_obj//2
        x_crop = x_center - crop_size//2
        y_crop = y_center - crop_size//2
        base_img = cv2.imread(base_path, -1)
        base_img_cropped = base_img[y_crop : y_crop + crop_size,
                                    x_crop : x_crop + crop_size]
        # TODO ^ probably some out of bounds case there, also possibly y_crop or x_crop is negative
        
        # save base image
        base_filename = "t" + str(i) + "_base.jpg"
        base_path_cropped = os.path.join(resultsdir, base_filename)
        cv2.imwrite(base_path_cropped, base_img_cropped)

        # for each target, create cropped img in resultsdir
        # and add line in results.csv
        for j in range(len(target_img_paths)):
            # get target path & label
            target_path = target_img_paths[j]
            target_label = target_labels[j]
            x_obj_t, y_obj_t, width_obj_t, height_obj_t = target_label

            # crop target
            target_img = cv2.imread(target_path, -1)
            target_img_cropped = target_img[y_crop : y_crop + crop_size,
                                            x_crop : x_crop + crop_size]
            # TODO again, probably an out of bounds case here
            
            # save target image
            target_filename = "t" + str(i) + "_target" + str(j) + ".jpg"
            target_path_cropped = os.path.join(resultsdir, target_filename)
            cv2.imwrite(target_path_cropped, target_img_cropped)
            
            # obtain object's motion from base to target (from object center)
            x_center_t = x_obj_t + width_obj_t//2
            y_center_t = y_obj_t + height_obj_t//2
            vx = x_center_t - x_center
            vy = y_center_t - y_center

            # display images
            print("t = " + str(i) + ", cropped images to:")
            print("y: " + str(y_crop)  + " to " + str(y_crop + crop_size))
            print("x: " + str(x_crop)  + " to " + str(x_crop + crop_size))
            print("vx, vy = ", vx, vy)
            
            csv.write(base_path_cropped + ","
                      + target_path_cropped + ","
                      + str(vx) + ","
                      + str(vy) + "\n")













