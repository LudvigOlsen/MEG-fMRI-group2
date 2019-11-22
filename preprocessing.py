from sklearn.decomposition import PCA
from sklearn.preprocessing import Binarizer, PowerTransformer, StandardScaler


##------------------------------------------------------------------##
## Preprocessing Utils
##------------------------------------------------------------------##

def preprocess(X_train, X_test, standardize=True, yeo_johnson=False, pca=False, binarize=False):
    if (yeo_johnson):
        transformer = PowerTransformer(
            method='yeo-johnson', standardize=False).fit(X_train)
        X_train = transformer.transform(X_train)
        X_test = transformer.transform(X_test)
    if (pca):
        principal = PCA(n_components=25).fit(X_train)
        X_train = principal.transform(X_train)
        X_test = principal.transform(X_test)
    if (standardize):
        scaler = StandardScaler(copy=False).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    if (binarize):
        binarizer = Binarizer(threshold=0).fit(X_train)
        X_train = binarizer.transform(X_train)
        X_test = binarizer.transform(X_test)

    return X_train, X_test
