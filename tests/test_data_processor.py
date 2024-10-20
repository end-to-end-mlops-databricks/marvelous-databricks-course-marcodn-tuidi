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

def test_preprocessor_data(data_processor_train):
    data_processor_train.preprocess_data()
    expected_features = ["Age", "Score", "Gender", "Education", "Interest"]
    assert data_processor_train.preprocessor is not None
    assert data_processor_train.X.shape == (4, 5)
    assert data_processor_train.y.shape == (4, )
    assert list(data_processor_train.X.columns).__eq__(expected_features)

def test_preprocessed_data_shape(x_transformed):
    assert x_transformed.shape == (4, 9)


def test_preprocessor_transform_numeric(
        x_transformed,
        data_processor_train, 
        test_data
    ):
    median_score_value = data_processor_train.X["Score"].median()
    test_data.fillna({"Score": median_score_value}, inplace=True)
    scaler_mean = data_processor_train.X["Score"].mean()
    scaler_std_dev = data_processor_train.X["Score"].std()
    expected_score = (median_score_value - scaler_mean) / scaler_std_dev
    
    assert median_score_value == 5
    assert round(x_transformed[0, 1], 3) == round(expected_score, 3)

def test_split(data_processor_train_split):
    data_processor_train_split.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor_train_split.split_data(
        test_size=0.25, random_state=42)
    
    assert X_train.shape == (12, 5)
    assert X_test.shape == (4, 5)
    assert y_train.shape == (12, )
    assert y_test.shape == (4, )