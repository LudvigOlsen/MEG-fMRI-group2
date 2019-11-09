import numpy as np
import pandas as pd

from utils import flatten_list


##------------------------------------------------------------------##
## Cross-Validation Utils
##------------------------------------------------------------------##

# For finding the folds of each trial
def fold_trials(num_trials, num_folds=10):
    fold_ids = np.array(list(range(num_folds)))
    n_reps = np.ceil(float(num_trials) / num_folds)
    n_excessive = int((num_folds * n_reps) - num_trials)
    smaller_folds = np.random.choice(fold_ids, n_excessive, replace=False)
    reps_per_fold = [n_reps - 1 if fid in smaller_folds else n_reps for fid in fold_ids]
    grouping_factor = np.repeat(fold_ids, reps_per_fold, axis=0)
    np.random.shuffle(grouping_factor)
    return grouping_factor


def cross_validate_time_point(X, y, trial_folds, train_predict_fn, use_features=None):
    folds = np.unique(trial_folds)
    target_collection = []
    predicted_probs_collection = []
    predicted_class_collection = []
    trial_id_collection = []
    fold_collection = []
    for fold in folds:
        train_indices = np.where(trial_folds != fold)[0]
        test_indices = np.where(trial_folds == fold)[0]
        if isinstance(X, pd.DataFrame):
            if use_features is not None:
                X = X.loc[:, use_features]
            X_train = np.asarray(X.iloc[train_indices])
            X_test = np.asarray(X.iloc[test_indices])
        else:
            X_train = X[train_indices]
            X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        # Fit model and predict test set
        predicted_probs, predicted_class = train_predict_fn(X_train=X_train, X_test=X_test, y_train=y_train)
        # Append to collections
        trial_id_collection.append(test_indices)
        target_collection.append(y_test)
        predicted_probs_collection.append(predicted_probs)
        predicted_class_collection.append(predicted_class)
        fold_collection.append([fold] * len(test_indices))
    return pd.DataFrame({"Fold": flatten_list(fold_collection),
                         "Trial": flatten_list(trial_id_collection),
                         "Target": flatten_list(target_collection),
                         "Predicted Probability": flatten_list(predicted_probs_collection),
                         "Predicted Class": flatten_list(predicted_class_collection)})
