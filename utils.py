##------------------------------------------------------------------##
## General Utils
##------------------------------------------------------------------##

import ntpath
from os.path import isdir


def flatten_list(l):
    """
    Flattens list with sublists.
    Note: Is NOT recursive.
    """
    return [item for sublist in l for item in sublist]


def path_leaf(path):
    """
    Extracts the filename from a path string

    """
    head, tail = ntpath.split(path)
    return tail


def path_head(path):
    """
    Extracts the directory from a path string

    """
    head, tail = ntpath.split(path)
    return head


def split_path_recursively(path):
    # Remove trailing / or \
    if path[-1] in ["/", "\\"] and len(path) > 1:
        path = path[:-1]
    tails = []
    prev_tail = "!"
    new_head, new_tail = ntpath.split(path)
    if new_tail != "":
        tails.append(new_tail)
    while prev_tail != new_tail:
        prev_tail = new_tail
        new_head, new_tail = ntpath.split(new_head)
        if new_tail != "":
            tails.append(new_tail)
    tails.reverse()
    return tails


def check_first_path_parts(path):
    # Might only work on unix
    path_parts = split_path_recursively(path)
    assert isdir("/" + path_parts[0] + "/" + path_parts[1]), \
        "first part of path was not found. Did you specify it for your system?"


def extract_sensor_colnames(df):
    return [cn for cn in df.columns if "S_" in cn]
