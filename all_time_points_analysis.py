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
from utils import path_head, check_first_path_parts

##------------------------------------------------------------------##
## Set Variables
##------------------------------------------------------------------##

# Cross-validation
NUM_FOLDS = 10
REPEATS = 3  # Run repeated cross-validation
MODEL_FN = svm_model
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

# Paths
PROJECT_PATH = "/Users/ludvigolsen/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2/"

# Stop if the first two parts of the project path were not found
check_first_path_parts(PROJECT_PATH)

# Data paths
DATA_PATH = join(PROJECT_PATH, "data/")
LABELS_PATH = join(DATA_PATH, GROUP_NAME + "/pos_neg_img_labels.npy")
PRECOMPUTED_DIR_PATH = join(DATA_PATH, GROUP_NAME + "/precomputed/")

# Result paths
RESULTS_PATH = join(PROJECT_PATH, "results/time_point_models/single/")
SAVE_PREDS_PATH = join(RESULTS_PATH, GROUP_NAME +
                       "/predictions/" + MODEL_NAME + "/predictions_at_all_time_points.csv")
SAVE_RESULTS_PATH = join(RESULTS_PATH, GROUP_NAME + "/results/" +
                         MODEL_NAME + "/results_at_all_time_points.csv")

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
precomputed_df_paths = glob.glob(join(PRECOMPUTED_DIR_PATH, "time_point_*.csv"))[:3]

# Load the precomputed data frames
time_point_dfs = [pd.read_csv(path) for path in precomputed_df_paths]

##------------------------------------------------------------------##
## Running CV on all time points for a single participant
##------------------------------------------------------------------##

# Number of trials
num_trials = labels.shape[0]

# Set sensor if "all"
if not isinstance(SENSORS, list):
    raise KeyError("SENSORS must be a list. For all sensors, specify as ['all'].")
if SENSORS[0] == "all":
    sensors = ["S_" + str(i) for i in range(labels.shape[0])]
else:
    sensors = SENSORS

# Create fold factor
folds = [fold_trials(num_trials, num_folds=NUM_FOLDS) for i in range(REPEATS)]

# Cross-validate all time points
predictions, evaluations = cross_validate_all_time_points(time_point_dfs=time_point_dfs,
                                                          y=labels,
                                                          trial_folds=folds,
                                                          train_predict_fn=MODEL_FN,
                                                          use_features=sensors)
# Save output to disk
predictions.to_csv(SAVE_PREDS_PATH)
evaluations.to_csv(SAVE_RESULTS_PATH)
