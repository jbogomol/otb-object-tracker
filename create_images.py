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
# Compiles results into csv files containing the image paths and ground truths.
#
# Usage
# > python create_images.py
#
# Before running
# Change global variables to correct file paths.


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
    imdir = "/home/datasets/OTB/train/"
else:
    imdir = "../OTB/train/"

# directory to store results in
if on_server:
    resultsdir = "/home/datasets/data_jbogomol/OTB_data/results/"
else:
    resultsdir = "../OTB_data/results/"

# size to crop images to (square, height = width)
crop_size = 256


# empty results image directory
func_file.empty_folder(resultsdir)

# list of OTB videos to use
otb_list = func_file.get_all_dirs_in(
    directory=imdir,
    exclude=[])

# loop through all videos on otb_list
for video in otb_list:
    print(video)

    # initialize results.csv file
    csv_path = os.path.join(resultsdir, video + "_results.csv")
    csv = open(csv_path, "w+")

    # get lists of images and their labels
    img_list = func_file.get_files_sorted(
        directory=imdir + video + "/img/",
        extension=".jpg")
    labels_path = os.path.join(imdir + video, "groundtruth_rect.txt")
    try:
        labels = np.genfromtxt(
            fname=labels_path,
            dtype="int",
            delimiter=",")
    except PermissionError:
        print("permission error reading " + labels_path)
        continue
    num_imgs = len(img_list)
    # delimiter could be tab also
    if labels.shape != (num_imgs, 4):
        labels = np.genfromtxt(
            fname=labels_path,
            dtype="int",
            delimiter="\t")
        if labels.shape != (num_imgs, 4):
            print("Err with parsing ground truths in video: " + video)
            continue
 
    # loop through all frames
    for i in range(num_imgs):
        # load base image path
        base_path = img_list[i]
        base_label = labels[i]

        # make list of all target paths for base at frame i
        target_img_paths = []
        target_labels = []

        # future frames:
        # i + 1
        if i < num_imgs - 1:
            target_img_paths.append(img_list[i+1])
            target_labels.append(labels[i+1])
        # i + 2
        if i < num_imgs - 2:
            target_img_paths.append(img_list[i+2])
            target_labels.append(labels[i+2])
        # i + 3
        if i < num_imgs - 3:
            target_img_paths.append(img_list[i+3])
            target_labels.append(labels[i+3])

        # past frames:
        # i - 1
        if i >= 1:
            target_img_paths.append(img_list[i-1])
            target_labels.append(labels[i-1])
        # i - 2
        if i >= 2:
            target_img_paths.append(img_list[i-2])
            target_labels.append(labels[i-2])
        # i - 3
        if i >= 3:
            target_img_paths.append(img_list[i-3])
            target_labels.append(labels[i-3])

        x_obj, y_obj, width_obj, height_obj = base_label
        x_center = x_obj + width_obj//2
        y_center = y_obj + height_obj//2
        x_crop = x_center - crop_size//2
        y_crop = y_center - crop_size//2
        base_img = cv2.imread(base_path, 1)
        base_img_cropped = func_img.crop_zero_padding(
            img=base_img,
            x_crop=x_crop,
            y_crop=y_crop,
            crop_size=crop_size)

        # save base image
        base_filename = video + "_t" + str(i) + "_base.jpg"
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
            target_img = cv2.imread(target_path, 1)
            target_img_cropped = func_img.crop_zero_padding(
                img=target_img,
                x_crop=x_crop,
                y_crop=y_crop,
                crop_size=crop_size)

            # save target image
            target_filename = video + "_t" + str(i) + "_target" + str(j) + ".jpg"
            target_path_cropped = os.path.join(resultsdir, target_filename)
            cv2.imwrite(target_path_cropped, target_img_cropped)
            
            # obtain object's motion from base to target (from object center)
            x_center_t = x_obj_t + width_obj_t//2
            y_center_t = y_obj_t + height_obj_t//2
            vx = x_center_t - x_center
            vy = y_center_t - y_center

            # add datapoint to csv file
            csv.write(base_path_cropped + ","
                      + target_path_cropped + ","
                      + str(vx) + ","
                      + str(vy) + "\n")













