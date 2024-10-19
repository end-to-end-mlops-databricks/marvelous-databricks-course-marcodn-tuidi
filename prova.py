import numpy as np
import pandas as pd

expected_target = [
        "Analyst", 
        "Diplomat", 
        "Sentinel", 
        "Explorer", 
        np.nan, 
        np.nan
    ]

df = pd.DataFrame(
    {"target": expected_target}
)

print(list(df["target"]).__eq__(expected_target))