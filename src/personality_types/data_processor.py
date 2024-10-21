import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from custom_transforms import GenderTransform, EducationTransform
import logging
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, Any

class DataProcessor:
    """
    A class responsible for loading, processing, and splitting data 
    for machine learning models.
    
    Attributes:
        df (pd.DataFrame): Loaded dataset as a pandas DataFrame.
        train (bool): A flag indicating whether the data is for training 
            (True) or inference (False).
        config (dict): A configuration dictionary containing feature names and 
            other processing parameters.
        X (Optional[pd.DataFrame]): The feature matrix.
        y (Optional[pd.Series]): The target vector.
        preprocessor (Optional[ColumnTransformer]): The preprocessor used for 
            data transformation.
    """

    def __init__(
            self, 
            data_path: str, 
            train: bool, 
            config: Dict[str, Any]
        ) -> None:
        """
        Initializes the DataProcessor with the data path, training flag, 
        and configuration.

        Args:
            data_path (str): Path to the CSV file containing the dataset.
            train (bool): A flag indicating whether it's training mode (True) 
                or inference mode (False).
            config (dict): Configuration dictionary containing feature names 
                and target variable details.
        """
        self.df = self.load_data(data_path)
        self.train = train
        self.config = config
        self.X = None
        self.y = None 
        self.preprocessor = None 

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Loads the dataset from the specified CSV file path.

        Args:
            data_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        return pd.read_csv(data_path)
    
    def create_target(self, target: str, raw_target: str) -> None:
        """
        Creates the target column in the dataframe by mapping raw target types 
        to groups.

        Args:
            target (str): The name of the new target column.
            raw_target (str): The column containing the raw personality types.
        """
        anlaysts = ["INTJ", "INTP", "ENTJ", "ENTP"]
        diplomats = ["INFJ", "INFP", "ENFJ", "ENFP"]
        sentinels = ["ISTJ", "ISFJ", "ESTJ", "ESFJ"]
        explorers = ["ISTP", "ISFP", "ESTP", "ESFP"]
        
        self.df[target] = (
            np.where(
                self.df[raw_target].isin(anlaysts), "Analyst",
                np.where(
                    self.df[raw_target].isin(diplomats), "Diplomat",
                    np.where(
                        self.df[raw_target].isin(sentinels), "Sentinel",
                        np.where(
                            self.df[raw_target].isin(explorers), "Explorer",
                            None
                        )
                    )
                )
            )
        )
    
    def preprocess_data(self) -> None:
        """
        Preprocesses the dataset by handling missing values, scaling numeric 
        features, and encoding categorical features.
        The preprocessor is built and stored as an attribute, and `X` and `y` 
        are set up based on the feature matrix and target.
        """
        num_features = self.config["num_features"]
        cat_features = self.config["cat_features"]
        train_features = num_features + cat_features

        # remove columns with missing raw target
        if self.train:
            raw_target = self.config["raw_target"]
            target = self.config["target"]

            self.create_target(target, raw_target)
            self.df = self.df.dropna(subset=[target])

            self.y = self.df[target]

        self.X = self.df[train_features]

        # train features preprocessing steps
        # numeric features
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # categorical features
        standard_categorical = list(
            set(self.config["cat_features"]) - set(["Gender", "Education"])
        )

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy="constant", fill_value="Unknown"
                )
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        gender_transformer = Pipeline(steps=[
            ("force_value", GenderTransform()),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        education_transform = Pipeline(steps=[
            ("force_value", EducationTransform())
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config["num_features"]),
                ("gender", gender_transformer, "Gender"),
                ("education", education_transform, "Education"),
                ("cat", categorical_transformer, standard_categorical)
            ]
        )
    
    def split_data(
            self, 
            test_size: Optional[float] = 0.2, 
            random_state: Optional[int]=42
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and testing sets based on the 
        specified test size and random state.

        Args:
            test_size (float): Proportion of the dataset to include in the test 
                split. Default is 0.2.
            random_state (int): Random seed for reproducibility. Default is 42.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The split 
                data as (X_train, X_test, y_train, y_test).
        """
        return train_test_split(
            self.X, 
            self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )
