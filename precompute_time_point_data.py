"""
Split npy file by time point and save one data frame per time point
Will run for a fairly long time!
"""

from os.path import join
import pathlib
import numpy as np
import pandas as pd

from data_extraction import create_time_point_data_frames
from utils import check_first_path_parts

##------------------------------------------------------------------##
## Set Variables
##------------------------------------------------------------------##

# Group
GROUP_NAME = "group_1"

# Automatically create 'precomputed' folder
# NOTE: Set project path before enabling this
AUTO_CREATE_DIR = False

##------------------------------------------------------------------##
## Set Paths
##------------------------------------------------------------------##

# Paths
USER = "DarioUbuntu"

# Just add your profile below, so we only need to change the user locally
if USER == "LudvigMac":
    PROJECT_PATH = "/Users/ludvigolsen/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2/"
elif USER == "LudvigUbuntu":
    PROJECT_PATH = "/home/ludvigolsen/Development/python/MEG-fMRI-group2"
elif USER == "DarioUbuntu":
    PROJECT_PATH = "/home/darcusco/Documents/CogSci/Advanced Cognitive Neuroscience/Meg assinment/MEG-fMRI-group2"
# Stop if the first two parts of the project path were not found
check_first_path_parts(PROJECT_PATH)

# Data paths
DATA_PATH = join(PROJECT_PATH, "data/")
TRIALS_PATH = join(DATA_PATH, GROUP_NAME + "/pos_neg_img_trials.npy")
PRECOMPUTED_DIR_PATH = join(DATA_PATH, GROUP_NAME + "/precomputed")

# Create folder for precomputed files
if AUTO_CREATE_DIR:
    pathlib.Path(PRECOMPUTED_DIR_PATH).mkdir(parents=True, exist_ok=True)

##------------------------------------------------------------------##
## Load and save data
##------------------------------------------------------------------##

# Load trials data
trials = np.load(TRIALS_PATH)

# Create a list with one data frame per time point
# Names: S_0, S_1, S_2, ..., Trial, Time Point
# ... where S_* are the sensors
time_point_dfs = create_time_point_data_frames(trials)

# Save data frames to disk
[ts.to_csv(
    join(PRECOMPUTED_DIR_PATH, "time_point_{}.csv".format(i)))
    for i, ts in enumerate(time_point_dfs)]
