import numpy as np
import pandas as pd
from src.personality_types.custom_transforms import GenderTransform, EducationTransform

expected_gender = [
        0,
        1,
        np.nan,
        1,
        5
    ]

df = pd.DataFrame(
    {"gender": expected_gender}
)

print(df)

transformer = EducationTransform()

transformed_data = transformer.fit_transform(df['gender'])

print(transformed_data)