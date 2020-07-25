# func_file.py
#
# @author       Jackson Bogomolny <jbogomol@andrew.cmu.edu>
# @date         07/20/2020
#
# contains helper functions for managing files


import os
import numpy as np


def empty_folder(directory):
    """
    Removes all files in a folder.

    Args
        directory: string, path to folder to empty
    """
    
    for filename in os.listdir(directory):
        os.remove(os.path.join(directory, filename))


def get_files_sorted(directory, extension=None):
    """
    Returns a numpy array of all files in a folder, sorted

    Args
        directory - string, path to folder to empty
        extension - (optional) string, file extension to grab
                    if not specified, grabs all files
    """

    files = []
    for filename in os.listdir(directory):
        if (extension == None) or (filename.endswith(extension)):
            path_to_file = os.path.join(directory, filename)
            files.append(path_to_file)
    files.sort()
    return np.array(files)












