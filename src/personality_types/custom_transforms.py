from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ForcedValuesTransform(BaseEstimator, TransformerMixin):
    """
    A transformer that forces the values in a column to be one of the allowed
    values.
    Any value not in the allowed list is replaced with the default input value.

    Attributes:
        allowed_values (list): List of values that are allowed in the column.
        impute_value (str or int): Value to replace any disallowed values.
    """

    def __init__(self) -> None:
        """
        Initializes the transformer with an empty list of allowed values and a
        default impute value of 'Unknown'.
        """
        self.allowed_values = []
        self.inpute_value = "Unknown"

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "ForcedValuesTransform":
        """
        Fit method required by scikit-learn, but no fitting occurs as this is a
        stateless transformer.

        Args:
            X (pd.DataFrame): Input data.
            y (pd.Series, optional): Target data (unused).

        Returns:
            ForcedValuesTransform: Returns self for consistency with
            scikit-learn transformers.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input array, replacing any value not in the allowed
        values with the impute value, and reshapes it to 2D.

        Args:
            X (np.ndarray): Input array with shape (n_samples,)
                or (n_samples, 1).

        Returns:
            np.ndarray: Transformed array with shape (n_samples, 1) where
            disallowed values are replaced.
        """
        X = np.where(np.isin(X, self.allowed_values), X, self.inpute_value)
        return X.reshape(-1, 1)


class GenderTransform(ForcedValuesTransform):
    """
    A transformer specifically for gender data, allowing only 'Male', 'Female'
    and 'Unknown'.
    Any other value is replaced with 'Unknown'.
    """

    def __init__(self):
        """
        Initializes the GenderTransform with the allowed gender values
        and replace value.
        """
        self.allowed_values = ["Male", "Female", "Unknown"]
        self.inpute_value = "Unknown"


class EducationTransform(ForcedValuesTransform):
    """
    A transformer specifically for education data, allowing only 0 and 1.
    Any other value is replaced with 0.
    """

    def __init__(self):
        """
        Initializes the EducationTransform with allowed education values
        and replace value.
        """
        self.allowed_values = [0, 1]
        self.inpute_value = 0
