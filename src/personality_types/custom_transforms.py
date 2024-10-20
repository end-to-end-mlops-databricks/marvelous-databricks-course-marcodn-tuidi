import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ForcedValuesTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.allowed_values = []
        self.inpute_value = "Unknown"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.where(np.isin(X, self.allowed_values), X, self.inpute_value)
        return X.reshape(-1, 1)
    
class GenderTransform(ForcedValuesTransform):
    def __init__(self):
        self.allowed_values = ["Male", "Female", "Unknown"]
        self.inpute_value = "Unknown"

class EducationTransform(ForcedValuesTransform):
    def __init__(self):
        self.allowed_values = [0, 1]
        self.inpute_value = 0
