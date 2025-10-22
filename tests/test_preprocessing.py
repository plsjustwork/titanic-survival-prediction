import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from titanic import preprocess_data


def test_no_missing_age_after_preprocess():
    raw = pd.read_csv("train.csv")
    X, y = preprocess_data(raw)
    assert X.isna().sum().sum() == 0, "Still contains NaNs"


def test_shape_consistency():
    raw = pd.read_csv("train.csv")
    X, y = preprocess_data(raw)
    assert len(X) == len(y), "X and y length mismatch"
