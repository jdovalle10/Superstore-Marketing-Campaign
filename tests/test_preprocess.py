import pandas as pd
import numpy as np
from src.preprocess import handle_missing_values

def test_handle_missing_values_drops_and_imputes():
    df = pd.DataFrame({
        "A": [np.nan, np.nan, 1, 2],
        "B": [1, np.nan, 3, 4],
        "C": [10, 20, 30, 40],
    })
    cleaned = handle_missing_values(df)
    # 'A' > 70% NA → dropped
    assert "A" not in cleaned.columns
    # 'B' median of [1,3,4] is 3
    assert cleaned.loc[1, "B"] == 3
    # 'C' untouched
    assert list(cleaned["C"]) == [10, 20, 30, 40]
