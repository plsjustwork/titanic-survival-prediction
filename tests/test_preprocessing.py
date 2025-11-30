import pandas as pd
import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.functions.preprocessing import preprocess_df  # noqa: E402


def test_no_missing_after_preprocess():
    df = pd.read_csv(ROOT / "data" / "titanic.csv")

    X, y = preprocess_df(df, remove_outliers=False)

    assert X.isna().sum().sum() == 0, "Preprocessed features still contain NaNs"
    assert y.isna().sum() == 0, "Target contains NaNs"


def test_shape_consistency():
    df = pd.read_csv(ROOT / "data" / "titanic.csv")

    X, y = preprocess_df(df, remove_outliers=False)

    assert len(X) == len(y), f"Length mismatch: X={len(X)} vs y={len(y)}"
