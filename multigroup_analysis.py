"""
MEG Analysis - All time points for all participants analysis
"""

from os.path import join
import pathlib
import numpy as np
import pandas as pd
import glob

from cross_validation import fold_trials, cross_validate_all_time_points_by_group
from models import logistic_regression_model, svm_model
from utils import path_head, path_leaf, check_first_path_parts, extract_sensor_colnames

##------------------------------------------------------------------##
## Set Variables
##------------------------------------------------------------------##

# Leave-one-group-out Cross-validation
MODEL_FN = svm_model
SENSORS = ["all"]  # All sensors
MODEL_NAME = "svm_3"
PARALLEL = False
CORES = 7  # CPU cores to utilize when PARALLEL is True
DEV_MODE = True  # Only uses the first 5 time points

# Group
GROUP_NAMES = [
    "group_1",
    "group_3",
    # "group_4",
    # "group_5",
    # "group_6",
    # "group_7"
]

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
DATA_PATH = join(PROJECT_PATH, "data/multigroup/")

LABELS_PATH = join(DATA_PATH, "precomputed/labels.csv")
PRECOMPUTED_DIR_PATH = join(DATA_PATH, "precomputed/")

# Result paths
RESULTS_PATH = join(PROJECT_PATH, "results/time_point_models/multigroups/")
SAVE_PREDS_PATH = join(RESULTS_PATH,
                       "predictions/" + MODEL_NAME + "/multigroup_" + MODEL_NAME +
                       "_predictions_at_all_time_points.csv")
SAVE_RESULTS_PATH = join(RESULTS_PATH, "results/" +
                         MODEL_NAME + "/multigroup_" + MODEL_NAME + "_results_at_all_time_points.csv")

# Create results folder for current model
# NOTE: Currently the other folders must be created manually
if AUTO_CREATE_DIRS:
    pathlib.Path(path_head(SAVE_PREDS_PATH)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_head(SAVE_RESULTS_PATH)).mkdir(parents=True, exist_ok=True)

##------------------------------------------------------------------##
## Load (and save) data
##------------------------------------------------------------------##

# Load labels
labels = pd.read_csv(LABELS_PATH)

# Detect all the precomputed time point data frames
get_paths = lambda p: glob.glob(join(p, "time_point_*.csv"))
add_tp = lambda p: (int(path_leaf(p).split("_")[-1].split(".")[0]), p)

precomputed_df_paths = get_paths(PRECOMPUTED_DIR_PATH)
precomputed_df_paths = sorted([add_tp(p) for p in precomputed_df_paths],
                              key=lambda x: int(x[0]))

if DEV_MODE:
    precomputed_df_paths = precomputed_df_paths[:5]

# Load the precomputed data frames
time_point_dfs = [(tp, pd.read_csv(path)) for tp, path in precomputed_df_paths]

# Combine for each time frame
num_time_points = len(time_point_dfs)
print("Number of time points: ", num_time_points)

##------------------------------------------------------------------##
## Running CV on all time points for a single participant
##------------------------------------------------------------------##

# Set sensor if "all"
if not isinstance(SENSORS, list):
    raise KeyError("SENSORS must be a list. For all sensors, specify as ['all'].")
if SENSORS[0] == "all":
    # Expects the same sensors in all time point dfs
    sensors = extract_sensor_colnames(time_point_dfs[0][1])  # Note: Very naive implementation
else:
    sensors = SENSORS
# TODO Check at the best timepoint which sensors are most important!

# Leave-one-group-out cross-validation

# # Cross-validate all time points
predictions, evaluations = cross_validate_all_time_points_by_group(time_point_dfs=time_point_dfs,
                                                                   y=labels,
                                                                   group_names=GROUP_NAMES,
                                                                   train_predict_fn=MODEL_FN,
                                                                   use_features=sensors,
                                                                   parallel=PARALLEL,
                                                                   cores=CORES)
# Save output to disk
evaluations.to_csv(SAVE_RESULTS_PATH)
predictions.to_csv(SAVE_PREDS_PATH)
