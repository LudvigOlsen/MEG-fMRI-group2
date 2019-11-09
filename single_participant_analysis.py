"""
MEG Analysis
"""

from os.path import join
import pathlib
import numpy as np
import pandas as pd

from cross_validation import fold_trials, cross_validate_time_point
from data_extraction import create_time_point_data_frames
from evaluate import evaluate
from models import logistic_regression_model

##------------------------------------------------------------------##
## Set Variables
##------------------------------------------------------------------##

# Cross-validation
NUM_FOLDS = 10
MODEL_FN = logistic_regression_model
TIME_POINT = 701  # 501 is the stimuli start point
SENSORS = ["all"]  # All sensors
MODEL_NAME = "logistic_regression_1"

# Group
GROUP_NAME = "group_1"

# Automatically create result and 'precomputed' folders
# NOTE: Set project path before enabling this
# as it will otherwise fuck up everything in the whole world
# NOTE: Not tested!
AUTO_CREATE_DIRS = False

# Whether to start from the npy files or
# use the precomputed data frames (one per time point)
RUN_FROM_RAW = True
# Whether to save the computed data frames (one per time point)
# Only used when RUN_FROM_RAW is True
SAVE_TIME_POINT_DFS = True

##------------------------------------------------------------------##
## Set Paths
##------------------------------------------------------------------##

# Paths
PROJECT_PATH = "/Users/ludvigolsen/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2/"

# Data paths
DATA_PATH = join(PROJECT_PATH, "data/")
LABELS_PATH = join(DATA_PATH, GROUP_NAME + "/pos_neg_img_labels.npy")
PRECOMPUTED_DIR_PATH = join(DATA_PATH, GROUP_NAME + "/precomputed/")
# Only used if RUN_FROM_RAW is True
TRIALS_PATH = join(DATA_PATH, GROUP_NAME + "/pos_neg_img_trials.npy")

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
    pathlib.Path(SAVE_PREDS_PATH).mkdir(parents=True, exist_ok=True)
    pathlib.Path(SAVE_RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    if not RUN_FROM_RAW or (RUN_FROM_RAW and SAVE_RESULTS_PATH):
        pathlib.Path(PRECOMPUTED_DIR_PATH).mkdir(parents=True, exist_ok=True)

##------------------------------------------------------------------##
## Load (and save) data
##------------------------------------------------------------------##

# Load labels
labels = np.load(LABELS_PATH)

if RUN_FROM_RAW:

    trials = np.load(TRIALS_PATH)

    # Create a list with one data frame per time point
    # Names: S_0, S_1, S_2, ..., Trial, Time Point
    # ... where S_* are the sensors
    time_point_dfs = create_time_point_data_frames(trials)

    if SAVE_TIME_POINT_DFS:
        [ts.to_csv(
            join(PRECOMPUTED_DIR_PATH, "time_point_{}.csv".format(i)))
            for i, ts in enumerate(time_point_dfs)]

    current_time_point_df = time_point_dfs[TIME_POINT]

else:
    # Load the precomputed data frame for the time point
    time_point_path = join(PRECOMPUTED_DIR_PATH, "time_point_{}.csv".format(TIME_POINT))
    current_time_point_df = pd.read_csv(time_point_path)

    if SAVE_TIME_POINT_DFS:
        raise NotImplementedError("Can only save time point data frames when running from raw.")

##------------------------------------------------------------------##
## Running CV on one time point
##------------------------------------------------------------------##


# Number of trials
num_trials = labels.shape[0]

# Set sensor if "all"
if SENSORS[0] == "all":
    sensors = ["S_" + str(i) for i in range(labels.shape[0])]
else:
    sensors = SENSORS

# Extract fold IDs
folds = fold_trials(num_trials, num_folds=NUM_FOLDS)

# Run cross-validation
# returns predictions as data frame
predictions = cross_validate_time_point(current_time_point_df, labels, folds, MODEL_FN, sensors)
predictions.to_csv(SAVE_PREDS_PATH)

# Evaluate predictions
eval = evaluate(list(predictions["Target"]),
                list(predictions["Predicted Class"]))
eval.to_csv(SAVE_RESULTS_PATH)

print("Results for {} at time point {}:".format(GROUP_NAME, TIME_POINT))
print(eval)
