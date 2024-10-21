import pytest
import numpy as np

def test_load_data(
        data_processor_train: pytest.FixtureRequest, 
        test_data: pytest.FixtureRequest
    ) -> None:
    """
    Test if the data is loaded correctly into the data processor.

    Args:
        data_processor_train (pytest.FixtureRequest): Instance of DataProcessor 
            initialized with training data.
        test_data (pytest.FixtureRequest): The test data used for comparison.

    Asserts:
        The DataFrame loaded in data_processor_train is equal to the test_data.
    """
    assert data_processor_train.df.equals(test_data)

def test_create_target(data_processor_train: pytest.FixtureRequest) -> None:
    """
    Test if the target column is correctly created based on the raw target.

    Args:
        data_processor_train (pytest.FixtureRequest): Instance of DataProcessor 
            initialized with training data.

    Asserts:
        The 'target' column created by the data processor matches the expected 
            target list.
    """
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

def test_drop_na_target(data_processor_train: pytest.FixtureRequest) -> None:
    """
    Test if rows with missing target values are correctly dropped during 
    preprocessing.

    Args:
        data_processor_train (pytest.FixtureRequest): Instance of DataProcessor 
            initialized with training data.

    Asserts:
        The length of the resulting target data (y) is correct after dropping 
            rows with missing targets.
    """
    expected_length = 4
    data_processor_train.preprocess_data()
    result_length = len(data_processor_train.y)
    assert result_length == expected_length

def test_preprocessor_data(
        data_processor_train: pytest.FixtureRequest
    ) -> None:
    """
    Test if the data is preprocessed correctly, ensuring the feature matrix and 
    target vector are properly shaped.

    Args:
        data_processor_train (pytest.FixtureRequest): Instance of DataProcessor 
            initialized with training data.

    Asserts:
        - Preprocessor is not None after preprocessing.
        - Feature matrix (X) has the correct shape (4, 5).
        - Target vector (y) has the correct shape (4,).
        - Column names of the feature matrix match the expected feature list.
    """
    data_processor_train.preprocess_data()
    expected_features = ["Age", "Score", "Gender", "Education", "Interest"]
    assert data_processor_train.preprocessor is not None
    assert data_processor_train.X.shape == (4, 5)
    assert data_processor_train.y.shape == (4, )
    assert list(data_processor_train.X.columns).__eq__(expected_features)

def test_preprocessed_data_shape(x_transformed: pytest.FixtureRequest) -> None:
    """
    Test if the transformed data has the expected shape.

    Args:
        x_transformed (pytest.FixtureRequest): Transformed feature matrix.

    Asserts:
        The shape of the transformed matrix is (4, 9).
    """
    assert x_transformed.shape == (4, 9)


def test_preprocessor_transform_numeric(
        x_transformed: pytest.FixtureRequest,
        data_processor_train: pytest.FixtureRequest, 
        test_data: pytest.FixtureRequest
    ) -> None:
    """
    Test if the numeric features are correctly transformed by the 
    preprocessor.

    In particular the median replace value and the scaler are tested.

    Args:
        x_transformed (npytest.FixtureRequest): Transformed feature matrix.
        data_processor_train (pytest.FixtureRequest): Instance of DataProcessor 
            for processing data.
        test_data (pytest.FixtureRequest): The fixture providing test data.

    Asserts:
        - Median value is correctly calculated.
        - Manually scaled value matches the expected standardized value.
    """
    median_score_value = data_processor_train.X["Score"].median()
    test_data.fillna({"Score": median_score_value}, inplace=True)
    scaler_mean = data_processor_train.X["Score"].mean()
    scaler_std_dev = data_processor_train.X["Score"].std()
    expected_score = (median_score_value - scaler_mean) / scaler_std_dev
    
    assert median_score_value == 5
    assert round(x_transformed[0, 1], 3) == round(expected_score, 3)

def test_split(data_processor_train_split: pytest.FixtureRequest) -> None:
    """
    Test if the data is correctly split into training and test sets.

    Args:
        data_processor_train_split (pytest.FixtureRequest): Instance of 
            DataProcessor initialized with split data.

    Asserts:
        - Training data shape is correct.
        - Test data shape is correct.
        - Target vector shape for training set is correct.
        - Target vector shape for testing set is correct.
    """
    data_processor_train_split.preprocess_data()
    X_train, X_test, y_train, y_test = data_processor_train_split.split_data(
        test_size=0.25, random_state=42)
    
    assert X_train.shape == (12, 5)
    assert X_test.shape == (4, 5)
    assert y_train.shape == (12, )
    assert y_test.shape == (4, )
