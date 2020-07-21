# func_file.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/20/2020
#
# contains helper functions for managing files


import os
import numpy as np


# removes all files in a folder
# args:
#       directory - string, path to folder to empty
def empty_folder(directory):
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


# returns a numpy array of all files in a folder, sorted
# args:
#       directory - string, path to folder to empty
#       extension - string, file extension to grab
def get_files_sorted(directory, extension=None):
    files = []
    for filename in os.listdir(directory):
        if (extension == None) or (filename.endswith(extension)):
            path_to_file = os.path.join(directory, filename)
            files.append(path_to_file)
    files.sort()
    return np.array(files)












