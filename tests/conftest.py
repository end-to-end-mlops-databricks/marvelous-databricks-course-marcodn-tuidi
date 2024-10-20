import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src/personality_types")
    ))

import pandas as pd
import numpy as np
from src.personality_types.data_processor import DataProcessor



@pytest.fixture()
def test_data():
    test_data_df = pd.DataFrame(
        data={
            "Age": [10, 4, 34, 25, 56, 21],
            "Score": [np.nan, 5, 5, 8, 9, 8], 
            "Education": [np.nan, 4, 1, 1, 0, 1],
            "Gender": ["Male", None, "Male", "Female", "Male", "Female"],
            "Interest": ["Sport", "Art", "Unknown", "Sport", None, None],
            "raw_target": ["INTJ", "INFJ", "ISTJ", "ISTP", "INVALID", None]
        }
    )
    return test_data_df

@pytest.fixture
def test_config():
    test_config_dict = {
        "num_features": ["Age", "Score"],
        "cat_features": ["Gender", "Education", "Interest"],
        "raw_target": "raw_target",
        "target": "target"
    }
    return test_config_dict

@pytest.fixture
def data_processor_train(tmp_path, test_data, test_config):
    csv_path = tmp_path / "test_data.csv"
    test_data.to_csv(csv_path, index=False)
    return DataProcessor(csv_path, True, test_config)

@pytest.fixture
def x_transformed(data_processor_train):
    data_processor_train.preprocess_data()
    preprocessor = data_processor_train.preprocessor
    return preprocessor.fit_transform(data_processor_train.X)

@pytest.fixture()
def test_data_split():
    test_data_df = pd.DataFrame(
        data={
            "Age": [10, 4, 34, 25] * 4,
            "Score": [np.nan, 5, 5, 8] * 4, 
            "Education": [np.nan, 4, 1, 1] * 4,
            "Gender": ["Male", None, "Male", "Female"] * 4,
            "Interest": ["Sport", "Art", "Unknown", "Sport"] * 4,
            "raw_target": ["INTJ", "INFJ", "ISTJ", "ISTP"] * 4
        }
    )
    return test_data_df

@pytest.fixture
def data_processor_train_split(tmp_path, test_data_split, test_config):
    csv_path = tmp_path / "test_data_split.csv"
    test_data_split.to_csv(csv_path, index=False)
    return DataProcessor(csv_path, True, test_config)