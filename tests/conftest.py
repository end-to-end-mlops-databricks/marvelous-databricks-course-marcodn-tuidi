import numpy as np
import pandas as pd
import pytest
from src.personality_types.config import ProjectConfig
from src.personality_types.data_processor import DataProcessor


@pytest.fixture()
def test_data() -> pd.DataFrame:
    """
    Fixture to create and return a sample Pandas DataFrame for testing.

    The DataFrame contains columns 'Age', 'Score', 'Education', 'Gender',
    'Interest', and 'raw_target' with some missing (NaN) values to simulate
    real-world data.

    Returns:
        pd.DataFrame: Test data DataFrame.
    """
    test_data_df = pd.DataFrame(
        data={
            "age": [10, 4, 34, 25, 56, 21],
            "score": [np.nan, 5, 5, 8, 9, 8],
            "education": [np.nan, 4, 1, 1, 0, 1],
            "gender": ["Male", None, "Male", "Female", "Male", "Female"],
            "interest": ["Sport", "Art", "Unknown", "Sport", None, None],
            "raw_target": ["INTJ", "INFJ", "ISTJ", "ISTP", "INVALID", None],
        }
    )
    return test_data_df


@pytest.fixture
def test_config() -> ProjectConfig:
    """
    Fixture to provide a sample configuration for data processing.

    Returns:
        ProjectConfig: Configuration dictionary containing numeric, categorical
        features and target information.
    """
    test_config_dict = {
        "num_features": ["age", "score"],
        "cat_features": ["gender", "education", "interest"],
        "raw_target": "raw_target",
        "target": "target",
        "catalog_name": "catalog_test",
        "schema_name": "schema_test",
        "parameters": {"learning_rate": 1},
    }
    return ProjectConfig.from_dict(test_config_dict)


@pytest.fixture
def data_processor_train(
    tmp_path: pytest.TempPathFactory,
    test_data: pd.DataFrame,
    test_config: ProjectConfig,
) -> DataProcessor:
    """
    Fixture to create and return a DataProcessor object for training data.

    Args:
        tmp_path (pytest.TempPathFactory): Temporary directory path.
        test_data (pd.DataFrame): Test data to be processed.
        test_config (ProjectConfig): Configuration for data processing.

    Returns:
        DataProcessor: Instance of DataProcessor initialized with test data
            and config.
    """
    csv_path = tmp_path / "test_data.csv"
    test_data.to_csv(csv_path, index=False)
    return DataProcessor(csv_path, True, test_config)


@pytest.fixture
def x_transformed(data_processor_train: DataProcessor) -> np.ndarray:
    """
    Fixture to return the transformed features after fitting the preprocessor.

    Args:
        data_processor_train (DataProcessor): Instance of DataProcessor for
            training data.

    Returns:
        np.ndarray: Transformed feature array.
    """
    data_processor_train.preprocess_data()
    preprocessor = data_processor_train.preprocessor
    return preprocessor.fit_transform(data_processor_train.X)


@pytest.fixture()
def test_data_split() -> pd.DataFrame:
    """
    Fixture to create and return a larger sample Pandas DataFrame for testing
    data splitting.

    The DataFrame contains repeated patterns of 'Age', 'Score', 'Education',
    'Gender', 'Interest', and 'raw_target' data.

    Returns:
        pd.DataFrame: Test data DataFrame with repeated rows.
    """
    test_data_df = pd.DataFrame(
        data={
            "age": [10, 4, 34, 25] * 4,
            "score": [np.nan, 5, 5, 8] * 4,
            "education": [np.nan, 4, 1, 1] * 4,
            "gender": ["Male", None, "Male", "Female"] * 4,
            "interest": ["Sport", "Art", "Unknown", "Sport"] * 4,
            "raw_target": ["INTJ", "INFJ", "ISTJ", "ISTP"] * 4,
        }
    )
    return test_data_df


@pytest.fixture
def data_processor_train_split(
    tmp_path: pytest.TempPathFactory,
    test_data_split: pd.DataFrame,
    test_config: ProjectConfig,
) -> DataProcessor:
    """
    Fixture to create and return a DataProcessor object for split test data.

    Args:
        tmp_path (pytest.TempPathFactory): Temporary directory path.
        test_data_split (pd.DataFrame): Split test data to be processed.
        test_config (ProjectConfig): Configuration for data processing.

    Returns:
        DataProcessor: Instance of DataProcessor initialized with split test
            data and config.
    """
    csv_path = tmp_path / "test_data_split.csv"
    test_data_split.to_csv(csv_path, index=False)
    return DataProcessor(csv_path, True, test_config)
