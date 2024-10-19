import pytest
import numpy as np

def test_load_data(data_processor_train, test_data):
    assert data_processor_train.df.equals(test_data)

def test_create_target(data_processor_train):
    expected_target = [
        "Analyst", 
        "Diplomat", 
        "Sentinel", 
        "Explorer", 
        None, 
        None
    ]
    target = data_processor_train.config["target"]
    raw_target = data_processor_train.config["raw_target"]
    data_processor_train.create_target(target, raw_target)
    assert list(data_processor_train.df[target]).__eq__(expected_target)

def test_drop_na_target(data_processor_train):
    expected_length = 4
    data_processor_train.preprocess_data()
    result_length = len(data_processor_train.y)
    assert result_length == expected_length
