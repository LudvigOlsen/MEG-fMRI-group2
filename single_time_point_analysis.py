"""
MEG Analysis - Single time point analysis (single participant)
"""

import pathlib
from os.path import join

import numpy as np
import pandas as pd

from cross_validation import fold_trials, cross_validate_time_point
from evaluate import evaluate
from models import svm_model
from utils import path_head, check_first_path_parts, extract_sensor_colnames

##------------------------------------------------------------------##
## Set Variables
##------------------------------------------------------------------##

# Cross-validation
NUM_FOLDS = 10
MODEL_FN = svm_model
TIME_POINT = 601  # 501 is the stimuli start point
SENSORS = ["all"]  # All sensors
MODEL_NAME = "svm_2"

# Group
GROUP_NAME = "group_1"

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
SAVE_PREDS_PATH = join(RESULTS_PATH, GROUP_NAME + "/predictions/" +
                       MODEL_NAME + "/predictions_at_time_point_" +
                       str(TIME_POINT) + ".csv")
SAVE_RESULTS_PATH = join(RESULTS_PATH, GROUP_NAME + "/results/" +
                         MODEL_NAME + "/results_at_time_point_" +
                         str(TIME_POINT) + ".csv")

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

# Load the precomputed data frame for the time point
time_point_path = join(PRECOMPUTED_DIR_PATH, "time_point_{}.csv".format(TIME_POINT))
current_time_point_df = pd.read_csv(time_point_path)

##------------------------------------------------------------------##
## Running CV on one time point
##------------------------------------------------------------------##

# Number of trials
num_trials = labels.shape[0]

# Set sensor if "all"
if not isinstance(SENSORS, list):
    raise KeyError("SENSORS must be a list. For all sensors, specify as ['all'].")
if SENSORS[0] == "all":
    sensors = extract_sensor_colnames(current_time_point_df)  # Note: Very naive implementation
else:
    sensors = SENSORS

# Create fold factor
folds = fold_trials(num_trials, num_folds=NUM_FOLDS)

# Run cross-validation
# returns predictions as data frame
predictions = cross_validate_time_point(X=current_time_point_df, y=labels,
                                        trial_folds=folds, train_predict_fn=MODEL_FN,
                                        use_features=sensors)
predictions.to_csv(SAVE_PREDS_PATH)

# Evaluate predictions
eval = evaluate(list(predictions["Target"]),
                list(predictions["Predicted Class"]))
eval.to_csv(SAVE_RESULTS_PATH)

print("Results for {} at time point {}:".format(GROUP_NAME, TIME_POINT))
print(eval)
