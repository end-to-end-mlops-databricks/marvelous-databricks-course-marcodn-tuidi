import pytest
from src.personality_types.custom_transforms import (
    EducationTransform,
    GenderTransform,
)


def test_gender_transfrom(test_data: pytest.FixtureRequest) -> None:
    """
    Test the GenderTransform custom transformation.

    This test checks if the GenderTransform correctly handles the 'Gender'
    column in the test data. Missing or unknown values are transformed to '
    Unknown', and other valid values like 'Male' and 'Female' are preserved.

    Args:
        test_data (pytest.FixtureRequest): The fixture providing test data
            containing a 'Gender' column.

    Asserts:
        The transformed result matches the expected output.
    """
    expected_result = ["Male", "Unknown", "Male", "Female", "Male", "Female"]
    transformer = GenderTransform()
    result = transformer.fit_transform(test_data["Gender"]).reshape(-1)
    assert result.__eq__(expected_result).all()


def test_education_transfrom(test_data: pytest.FixtureRequest) -> None:
    """
    Test the EducationTransform custom transformation.

    This test checks if the EducationTransform correctly processes the
    'Education' column in the test data. The transformer maps education levels
    into a numerical format where missing values are treated as zeros.

    Args:
        test_data (pytest.FixtureRequest): The fixture providing test data
            containing an 'Education' column.

    Asserts:
        The transformed result matches the expected output.
    """
    expected_result = [0, 0, 1, 1, 0, 1]
    transformer = EducationTransform()
    result = transformer.fit_transform(test_data["Education"]).reshape(-1)
    assert result.__eq__(expected_result).all()
