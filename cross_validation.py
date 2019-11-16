import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils import flatten_list
from evaluate import evaluate


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


def cross_validate_all_time_points(time_point_dfs, y, trial_folds, train_predict_fn,
                                   use_features, parallel=False, cores=7):
    """
    Note: time_point_dfs should be list of tuples with time point and df, e.g. (0, df_0)
    """

    # Repeated CV
    repeats = len(trial_folds)

    def cv_single(time_point_df, time_point, rep):
        # Run cross-validation
        # returns predictions as data frame
        predictions = cross_validate_time_point(X=time_point_df, y=y, trial_folds=trial_folds[rep],
                                                train_predict_fn=train_predict_fn,
                                                use_features=use_features)
        predictions["Time Point"] = time_point
        predictions["Repetition"] = rep

        # Evaluate predictions
        eval = evaluate(list(predictions["Target"]),
                        list(predictions["Predicted Class"]))
        eval["Time Point"] = time_point
        eval["Repetition"] = rep
        print(eval)

        return predictions, eval

    if parallel:
        preds, evals = zip(*Parallel(n_jobs=cores)(delayed(cv_single)(df, tp, rep) \
                                                   for tp, df in time_point_dfs for rep in range(repeats)))
    else:
        preds, evals = zip(*[cv_single(df, tp, rep) for tp, df in time_point_dfs for rep in range(repeats)])

    all_predictions = pd.concat(preds)
    all_evaluations = pd.concat(evals)

    return all_predictions, all_evaluations


## Multigroup

def leave_one_group_out_cv_single_time_point(X, y, group_names, train_predict_fn, use_features=None):
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise KeyError("leave_one_group_out_cv expects X and y to be data frames.")
    groups = np.unique(group_names)
    target_collection = []
    predicted_probs_collection = []
    predicted_class_collection = []
    trial_id_collection = []
    group_id_collection = []
    for group in groups:
        # Subset X and y
        train_set = X[X['group'] != group]
        test_set = X[X['group'] == group]
        train_labels = y[y['group'] != group]
        test_labels = y[y['group'] == group]

        # Add trial info to collection
        trial_id_collection.append(list(test_set["Trial"]))

        # Extract sensors
        if use_features is not None:
            train_set = train_set.loc[:, use_features]
            test_set = test_set.loc[:, use_features]

        # Convert to numpy arrays
        X_train = np.asarray(train_set)
        X_test = np.asarray(test_set)
        y_train = np.asarray(train_labels["label"])
        y_test = np.asarray(test_labels["label"])

        # Fit model and predict test set
        predicted_probs, predicted_class = train_predict_fn(X_train=X_train, X_test=X_test, y_train=y_train)

        # Append to collections
        target_collection.append(y_test)
        predicted_probs_collection.append(predicted_probs)
        predicted_class_collection.append(predicted_class)
        group_id_collection.append([group] * len(predicted_class))
    return pd.DataFrame({"Group": flatten_list(group_id_collection),
                         "Trial": flatten_list(trial_id_collection),
                         "Target": flatten_list(target_collection),
                         "Predicted Probability": flatten_list(predicted_probs_collection),
                         "Predicted Class": flatten_list(predicted_class_collection)})


def cross_validate_all_time_points_by_group(time_point_dfs, y, group_names,
                                            train_predict_fn,
                                            use_features, parallel=False, cores=7):
    """
    Note: time_point_dfs should be list of tuples with group and a list of paths
    (group_name, [(tp_1, path_1),(tp_2, path_2),...])
    """

    def cv_single(time_point_df, time_point):
        # Run cross-validation
        # returns predictions as data frame
        predictions = leave_one_group_out_cv_single_time_point(
            X=time_point_df, y=y, group_names=group_names,
            train_predict_fn=train_predict_fn,
            use_features=use_features)
        predictions["Time Point"] = time_point

        # Evaluate predictions
        eval = evaluate(list(predictions["Target"]),
                        list(predictions["Predicted Class"]))
        eval["Time Point"] = time_point
        print(eval)

        return predictions, eval

    if parallel:
        preds, evals = zip(*Parallel(n_jobs=cores)(delayed(cv_single)(df, tp) \
                                                   for tp, df in time_point_dfs))
    else:
        preds, evals = zip(*[cv_single(df, tp) for tp, df in time_point_dfs])

    all_predictions = pd.concat(preds)
    all_evaluations = pd.concat(evals)

    return all_predictions, all_evaluations
