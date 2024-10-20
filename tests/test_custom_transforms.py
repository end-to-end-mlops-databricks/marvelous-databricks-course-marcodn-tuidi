import pytest
import numpy as np
from src.personality_types.custom_transforms import (
    GenderTransform, EducationTransform
)

def test_gender_transfrom(test_data):
    expected_result = ["Male", "Unknown", "Male", "Female", "Male", "Female"]
    transformer = GenderTransform()
    result = transformer.fit_transform(test_data['Gender']).reshape(-1)
    assert result.__eq__(expected_result).all()

def test_education_transfrom(test_data):
    expected_result = [0, 0, 1, 1, 0, 1]
    transformer = EducationTransform()
    result = transformer.fit_transform(test_data['Education']).reshape(-1)
    assert result.__eq__(expected_result).all()