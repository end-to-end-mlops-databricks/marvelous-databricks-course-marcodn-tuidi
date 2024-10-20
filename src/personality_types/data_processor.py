import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from custom_transforms import GenderTransform, EducationTransform
import logging
from sklearn.model_selection import train_test_split

class DataProcessor:

    def __init__(self, data_path, train, config):
        self.df = self.load_data(data_path)
        self.train = train
        self.config = config
        self.X = None
        self.y = None 
        self.preprocessor = None 

    def load_data(self, data_path):
        return pd.read_csv(data_path)
    
    def create_target(self, target, raw_target):
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
    
    def preprocess_data(self):
        num_features = self.config['num_features']
        cat_features = self.config['cat_features']
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
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # categorical features
        standard_categorical = list(
            set(self.config['cat_features']) - set(["Gender", "Education"])
        )

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy='constant', fill_value='Unknown'
                )
            ),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        gender_transformer = Pipeline(steps=[
            ('force_value', GenderTransform()),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        education_transform = Pipeline(steps=[
            ('force_value', EducationTransform())
        ])

        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['num_features']),
                ("gender", gender_transformer, "Gender"),
                ("education", education_transform, "Education"),
                ('cat', categorical_transformer, standard_categorical)
            ]
        )
    
    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(
            self.X, 
            self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y
        )

        

        

