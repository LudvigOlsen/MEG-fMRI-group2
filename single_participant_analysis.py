## MEG Analysis
import numpy as np
from os.path import join

from cross_validation import fold_trials, cross_validate_time_point
from data_extraction import create_time_point_data_frames
from models import logistic_regression_model

# import tensorflow as tf

##------------------------------------------------------------------##
## Load data
##------------------------------------------------------------------##


data_path = "/Users/ludvigolsen/Documents/Aarhus/Cognitive Science/7. semester/Advanced Cognitive Neuroscience/Class tutorials/MEG_analysis/data/"
label_path = join(data_path, "group_1/pos_neg_img_labels.npy")
trials_path = join(data_path, "group_1/pos_neg_img_trials.npy")

labels = np.load(label_path)
trials = np.load(trials_path)

# TODO(ludvig) Precompute this
# Create a list with one data frame per time point
# Names: S_0, S_1, S_2, ..., Trial, Time Point
# ... where S_* are the sensors
time_point_dfs = create_time_point_data_frames(trials)

##------------------------------------------------------------------##
## Running CV on one time point
##------------------------------------------------------------------##

# Set these
NUM_FOLDS = 10
MODEL_FN = logistic_regression_model
TIME_POINT = 0
SENSORS = ["S_" + str(i) for i in range(trials.shape[1])]  # All sensors

# Number of trials
num_trials = trials.shape[0]

# Extract fold IDs
folds = fold_trials(num_trials, num_folds=NUM_FOLDS)

# Extract current data frame
current_data = time_point_dfs[TIME_POINT]

# Run cross-validation
# returns predictions as data frame
predictions = cross_validate_time_point(current_data, labels, folds, MODEL_FN, SENSORS)

print(predictions)
