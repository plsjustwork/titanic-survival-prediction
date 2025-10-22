import pandas as pd
import sys
from pathlib import Path


def test_no_missing_age_after_preprocess():
    """Ensure no NaNs remain after preprocessing."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from titanic import preprocess_data  # noqa: E402

    raw = pd.read_csv("train.csv")
    X, y = preprocess_data(raw)
    assert X.isna().sum().sum() == 0, "Still contains NaNs"


def test_shape_consistency():
    """Ensure X and y have matching lengths after preprocessing."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from titanic import preprocess_data  # noqa: E402

    raw = pd.read_csv("train.csv")
    X, y = preprocess_data(raw)
    assert len(X) == len(y), "X and y length mismatch"
