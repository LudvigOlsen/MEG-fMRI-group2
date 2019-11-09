## MEG Analysis
from os.path import join

import numpy as np
# import tensorflow as tf

##------------------------------------------------------------------##
## Load data
##------------------------------------------------------------------##

data_path = "/Users/ludvigolsen/Documents/Aarhus/Cognitive Science/7. semester/Advanced Cognitive Neuroscience/Class tutorials/MEG_analysis/data/"
label_path = join(data_path, "group_1/pos_neg_img_labels.npy")
trials_path = join(data_path, "group_1/pos_neg_img_trials.npy")

labels = np.load(label_path)
trials = np.load(trials_path)



##------------------------------------------------------------------##
## Running CV on one time point
##------------------------------------------------------------------##

# cross_validate_time_point(ts_df, labs, folds, my_model_fn, ["S_0","S_1"])
