from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from preprocessing import preprocess


##------------------------------------------------------------------##
## Custom Model Functions
##------------------------------------------------------------------##

def logistic_regression_model(X_train, X_test, y_train):
    # Not doing anything atm. but we could f.i. use PCA
    X_train, X_test = preprocess(X_train, X_test, standardize=False,
                                 yeo_johnson=False, pca=False, binarize=False)
    clf = LogisticRegression(random_state=0,
                             solver='liblinear',
                             penalty='l2').fit(X_train, y_train)
    predicted_probs = clf.predict_proba(X_test)[:, 1]
    predicted_class = clf.predict(X_test)
    return predicted_probs, predicted_class


def svm_model(X_train, X_test, y_train):
    X_train, X_test = preprocess(X_train, X_test, standardize=True,
                                 yeo_johnson=False, pca=False, binarize=False)
    clf = LinearSVC(random_state=0, max_iter=100000).fit(X_train, y_train)
    # svm doesn't have probabilities, but cross_validate_time_point expects both
    # probs and classes from the output
    predicted_probs = predicted_class = clf.predict(X_test)
    return predicted_probs, predicted_class


def pca_svm_model(X_train, X_test, y_train):
    X_train, X_test = preprocess(X_train, X_test, standardize=True,
                                 yeo_johnson=False, pca=True, binarize=False)
    clf = LinearSVC(random_state=0, max_iter=100000).fit(X_train, y_train)
    # svm doesn't have probabilities, but cross_validate_time_point expects both
    # probs and classes from the output
    predicted_probs = predicted_class = clf.predict(X_test)
    return predicted_probs, predicted_class


def binarized_svm_model(X_train, X_test, y_train):
    X_train, X_test = preprocess(X_train, X_test, standardize=True,
                                 yeo_johnson=False, pca=False, binarize=True)
    clf = LinearSVC(random_state=0, max_iter=100000).fit(X_train, y_train)
    # svm doesn't have probabilities, but cross_validate_time_point expects both
    # probs and classes from the output
    predicted_probs = predicted_class = clf.predict(X_test)
    return predicted_probs, predicted_class


def yj_svm_model(X_train, X_test, y_train):
    X_train, X_test = preprocess(X_train, X_test, standardize=True,
                                 yeo_johnson=True, pca=False, binarize=False)
    clf = LinearSVC(random_state=0, max_iter=100000).fit(X_train, y_train)
    # svm doesn't have probabilities, but cross_validate_time_point expects both
    # probs and classes from the output
    predicted_probs = predicted_class = clf.predict(X_test)
    return predicted_probs, predicted_class
