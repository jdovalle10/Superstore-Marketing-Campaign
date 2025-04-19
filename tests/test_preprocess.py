import pandas as pd
import numpy as np
from src.preprocess import handle_missing_values

def test_handle_missing_values_drops_and_imputes():
    df = pd.DataFrame({
        "A": [np.nan, np.nan, np.nan, 2],  # 3/4 = 75% missing (above 70% threshold)
        "B": [1, np.nan, 3, 4],            # 1/4 = 25% missing (below threshold)
        "C": [10, 20, 30, 40],             # 0% missing
    })
    cleaned = handle_missing_values(df)
    # 'A' > 70% NA → dropped
    assert "A" not in cleaned.columns
    # 'B' < 70% NA → imputed
    assert "B" in cleaned.columns
    assert cleaned["B"].isna().sum() == 0  # No more NaN values
