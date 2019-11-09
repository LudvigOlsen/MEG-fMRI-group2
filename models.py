from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess


##------------------------------------------------------------------##
## Custom Model Functions
##------------------------------------------------------------------##

def logistic_regression_model(X_train, X_test, y_train):
  # Not doing anything atm. but we could f.i. use PCA
  X_train, X_test = preprocess(X_train, X_test, standardize=False,
                               yeo_johnson=False, pca=False, binarize=False)
  clf = LogisticRegression(random_state=0,
                           solver='lbfgs',
                           penalty='l2').fit(X_train, y_train)
  predicted_probs = clf.predict_proba(X_test)[:, 1]
  predicted_class = clf.predict(X_test)
  return predicted_probs, predicted_class
