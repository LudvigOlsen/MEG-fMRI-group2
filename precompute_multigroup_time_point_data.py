"""
MEG Analysis - All time points for all participants analysis
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
    # "group_4",
    # "group_5",
    # "group_6",
    # "group_7"
]

# Leave-one-group-out Cross-validation
MODEL_FN = svm_model
SENSORS = ["all"]  # All sensors
MODEL_NAME = "svm_3"
PARALLEL = True
CORES = 7  # CPU cores to utilize when PARALLEL is True
DEV_MODE = True  # Only uses the first 5 time points

# Automatically create result and 'precomputed' folders
# NOTE: Set project path before enabling this
AUTO_CREATE_DIRS = True

##------------------------------------------------------------------##
## Set Paths
##------------------------------------------------------------------##

# Paths
USER = "LudvigMac"

# Just add your profile below, so we only need to change the user locally
if USER == "LudvigMac":
    PROJECT_PATH = "/Users/ludvigolsen/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2/"
elif USER == "LudvigUbuntu":
    PROJECT_PATH = "/home/ludvigolsen/Development/python/MEG-fMRI-group2"

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

# Load the precomputed data frames
if PARALLEL:
    load_dfs = lambda group_name, tp_paths: (group_name, [(tp, pd.read_csv(p)) for tp, p in tp_paths])
    time_point_dfs = Parallel(n_jobs=max(len(GROUP_NAMES), CORES))(delayed(load_dfs)(group_name, tp_paths) \
                                                                   for group_name, tp_paths in precomputed_df_paths)
else:
    time_point_dfs = [(group_name, [(tp, pd.read_csv(p)) for tp, p in paths]) \
                      for group_name, paths in precomputed_df_paths]

# Combine for each time frame
num_time_points = len(precomputed_df_paths[0][1])
print("Number of time points: ", num_time_points)

# Concat group dfs for each time point
if PARALLEL:
    concat_by_tp = lambda tp: (tp, pd.concat([tp_dfs[tp][1].assign(group=lambda x: group_name) \
                                              for group_name, tp_dfs in time_point_dfs]))
    time_point_dfs = Parallel(n_jobs=CORES)(delayed(concat_by_tp)(tp) \
                                            for tp in range(num_time_points))
else:
    time_point_dfs = [(tp, pd.concat([tp_dfs[tp][1].assign(group=lambda x: group_name) \
                                      for group_name, tp_dfs in time_point_dfs])) \
                      for tp in range(num_time_points)]

# Save data frames to disk
[ts.to_csv(
    join(SAVE_DATA_PATH, "time_point_{}.csv".format(tp)))
    for tp, ts in time_point_dfs]


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
