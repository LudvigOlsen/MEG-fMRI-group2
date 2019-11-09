##------------------------------------------------------------------##
## General Utils
##------------------------------------------------------------------##

import ntpath


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
