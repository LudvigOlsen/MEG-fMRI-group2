"""
MEG Analysis - All time point analysis (single participant)
"""

from os.path import join
import pathlib
import numpy as np
import pandas as pd
import glob

from cross_validation import fold_trials, cross_validate_all_time_points
from models import logistic_regression_model, svm_model
from utils import path_head, path_leaf, check_first_path_parts, extract_sensor_colnames

##------------------------------------------------------------------##
## Set Variables
##------------------------------------------------------------------##

# Cross-validation
NUM_FOLDS = 10
REPEATS = 3  # Run repeated cross-validation
MODEL_FN = svm_model
SENSORS = ["all"]  # All sensors
MODEL_NAME = "svm_3"
PARALLEL = True
CORES = 7  # CPU cores to utilize when PARALLEL is True
DEV_MODE = False  # Only uses the first 5 time points

# Group
GROUP_NAME = "group_5"

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
LABELS_PATH = join(DATA_PATH, GROUP_NAME + "/pos_neg_img_labels.npy")
PRECOMPUTED_DIR_PATH = join(DATA_PATH, GROUP_NAME + "/precomputed/")

# Result paths
RESULTS_PATH = join(PROJECT_PATH, "results/time_point_models/single/")
SAVE_PREDS_PATH = join(RESULTS_PATH, GROUP_NAME +
                       "/predictions/" + MODEL_NAME + "/" + GROUP_NAME + "_" + MODEL_NAME +
                       "_predictions_at_all_time_points.csv")
SAVE_RESULTS_PATH = join(RESULTS_PATH, GROUP_NAME + "/results/" +
                         MODEL_NAME + "/" + GROUP_NAME + "_" + MODEL_NAME + "_results_at_all_time_points.csv")

# Create results folder for current model
# NOTE: Currently the other folders must be created manually
if AUTO_CREATE_DIRS:
    pathlib.Path(path_head(SAVE_PREDS_PATH)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_head(SAVE_RESULTS_PATH)).mkdir(parents=True, exist_ok=True)

##------------------------------------------------------------------##
## Load (and save) data
##------------------------------------------------------------------##

# Load labels
labels = np.load(LABELS_PATH)

# Detect all the precomputed time point data frames
precomputed_df_paths = glob.glob(join(PRECOMPUTED_DIR_PATH, "time_point_*.csv"))
get_tp = lambda p: int(path_leaf(p).split("_")[-1].split(".")[0])
# Add time point info and sort by it
precomputed_df_paths = [(get_tp(p), p) for p in precomputed_df_paths]
precomputed_df_paths.sort(key=lambda x: int(x[0]))

if DEV_MODE:
    precomputed_df_paths = precomputed_df_paths[:5]

# Load the precomputed data frames
time_point_dfs = [(tp, pd.read_csv(path)) for tp, path in precomputed_df_paths]

##------------------------------------------------------------------##
## Running CV on all time points for a single participant
##------------------------------------------------------------------##

# Number of trials
num_trials = labels.shape[0]

# Set sensor if "all"
if not isinstance(SENSORS, list):
    raise KeyError("SENSORS must be a list. For all sensors, specify as ['all'].")
if SENSORS[0] == "all":
    sensors = extract_sensor_colnames(time_point_dfs[0][1])  # Note: Very naive implementation
else:
    sensors = SENSORS

# Create fold factor
folds = [fold_trials(num_trials, num_folds=NUM_FOLDS) for i in range(REPEATS)]

# Cross-validate all time points
predictions, evaluations = cross_validate_all_time_points(time_point_dfs=time_point_dfs,
                                                          y=labels,
                                                          trial_folds=folds,
                                                          train_predict_fn=MODEL_FN,
                                                          use_features=sensors,
                                                          parallel=PARALLEL,
                                                          cores=CORES)
# Save output to disk
evaluations.to_csv(SAVE_RESULTS_PATH)
predictions.to_csv(SAVE_PREDS_PATH)
