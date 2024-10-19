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
    print(list(data_processor_train.df[target]))
    print(expected_target)
    assert list(data_processor_train.df[target]).__eq__(expected_target)
