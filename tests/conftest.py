import pytest
import pandas as pd
import numpy as np
from src.personality_types.data_processor import DataProcessor

@pytest.fixture
def test_data():
    test_data_df = pd.DataFrame(
        data={
            "age": [10, 4, 34, 25, 56, 21],
            "score": [4, 5, -6, 8, 9, 8],
            "education": [0, 0, 1, 1, 0, 1],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "interest": ["Sport", "Art", "Unknown", "Sport", None, None],
            "raw_target": ["INTJ", "INFJ", "ISTJ", "ISTP", "INVALID", None]
        }
    )
    return test_data_df

@pytest.fixture
def test_config():
    test_config_dict = {
        "num_features": ["age", "score"],
        "cat_features": ["gender", "education", "interest"],
        "raw_target": "raw_target",
        "target": "target"
    }
    return test_config_dict

@pytest.fixture
def data_processor_train(tmp_path, test_data, test_config):
    csv_path = tmp_path / "test_data.csv"
    test_data.to_csv(csv_path, index=False)
    return DataProcessor(csv_path, True, test_config)