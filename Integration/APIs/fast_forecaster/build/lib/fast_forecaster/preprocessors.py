import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DropColumns(BaseEstimator, TransformerMixin):
    #Get imputed values, this package assumes preprocessed data after all
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.variables]
        return X

class TransformDFtoNDArray(BaseEstimator, TransformerMixin):
    #Get imputed values, this package assumes preprocessed data after all
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        X = X.values
        # ensure all data is float
        X = X.astype('float32')
        return X

class SplitInWindows(BaseEstimator, TransformerMixin):
    #Get imputed values, this package assumes preprocessed data after all
    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        X = np.array(np.split(X, len(X) / 1))
        return X


