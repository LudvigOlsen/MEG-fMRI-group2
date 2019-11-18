"""
MEG Analysis - All time points for all participants analysis
Requires that you've run precompute_time_point_data for all groups first!
"""

from os.path import join
import pathlib
import numpy as np
import pandas as pd
import glob
from joblib import Parallel, delayed

from cross_validation import fold_trials, cross_validate_all_time_points
from models import logistic_regression_model, svm_model
from utils import path_head, path_leaf, check_first_path_parts, extract_sensor_colnames

##------------------------------------------------------------------##
## Set Variables
##------------------------------------------------------------------##

# Group
GROUP_NAMES = [
    "group_1",
    "group_3",
    "group_4",
    "group_5",
    "group_6",
    "group_7"
]

PARALLEL = True
CORES = 7  # CPU cores to utilize when PARALLEL is True
DEV_MODE = False  # Only uses the first 5 time points

# Automatically create result and 'precomputed' folders
# NOTE: Set project path before enabling this
AUTO_CREATE_DIRS = True

##------------------------------------------------------------------##
## Set Paths
##------------------------------------------------------------------##

# Set current user
USER = "JoeMac"

# Just add your profile below, so we only need to change the user locally
# F.i. if multiple people in a study groups is working on the code,
# it's annoying to have to find the path every time. Much easier
# to just change the user above.
if USER == "JoeMac":
    PROJECT_PATH = "/Users/joe/path/to/MEG-fMRI-group2/"
elif USER == "JoeUbuntu":
    PROJECT_PATH = "/home/joe/path/to/MEG-fMRI-group2"

# Stop if the first two parts of the project path were not found
check_first_path_parts(PROJECT_PATH)

# Data paths
DATA_PATH = join(PROJECT_PATH, "data/")

PRECOMPUTED_DIR_PATHS = [(group_name, join(DATA_PATH, group_name + "/precomputed/")) for group_name in GROUP_NAMES]
LABELS_PATHS = [(group_name, join(DATA_PATH, group_name + "/pos_neg_img_labels.npy")) for group_name in GROUP_NAMES]

SAVE_DATA_PATH = join(DATA_PATH, "multigroup/precomputed/")

# Create results folder for current model
# NOTE: Currently the other folders must be created manually
if AUTO_CREATE_DIRS:
    pathlib.Path(path_head(SAVE_DATA_PATH)).mkdir(parents=True, exist_ok=True)

##------------------------------------------------------------------##
## Load (and save) data
##------------------------------------------------------------------##

# Detect all the precomputed time point data frames
get_paths = lambda p: glob.glob(join(p, "time_point_*.csv"))
add_tp = lambda p: (int(path_leaf(p).split("_")[-1].split(".")[0]), p)

precomputed_df_paths = [(group_name, get_paths(path)) for group_name, path in PRECOMPUTED_DIR_PATHS]
precomputed_df_paths = [(group_name, sorted([add_tp(p) for p in paths],
                                            key=lambda x: int(x[0]))) \
                        for group_name, paths in precomputed_df_paths]

if DEV_MODE:
    precomputed_df_paths = [(group_name, paths[:5]) for group_name, paths in precomputed_df_paths]

# Num time points
num_time_points = len(precomputed_df_paths[0][1])
print("Number of time points: ", num_time_points)


# To save RAM, we do this for one at a time
def precompute_single_time_point(precomputed_df_paths, time_point):
    # Load the precomputed data frames for this time point

    # (group_name, tp, df)
    # assumes paths are sorted by time point and that time points are 0-N
    time_point_dfs = pd.concat([pd.read_csv(paths[time_point][1]).assign(group=lambda x: group_name) \
                                for group_name, paths in precomputed_df_paths])

    # Save data frames to disk
    time_point_dfs.to_csv(
        join(SAVE_DATA_PATH, "time_point_{}.csv".format(time_point))
    )


# Precompute all time points
Parallel(n_jobs=CORES)(delayed(precompute_single_time_point)(precomputed_df_paths, tp) \
                       for tp in range(num_time_points))


def load_to_pd(labels_path, group_name):
    labels = np.load(labels_path)
    num_labels = len(labels)
    return pd.DataFrame({"label": labels,
                         "group": [group_name] * num_labels,
                         "Trial": list(range(num_labels))})


# Load labels
labels = pd.concat([load_to_pd(labels_path, group_name) \
                    for group_name, labels_path in LABELS_PATHS])
# Save labels df to disk
labels.to_csv(join(SAVE_DATA_PATH, "labels.csv"))
