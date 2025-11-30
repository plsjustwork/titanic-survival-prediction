import pandas as pd
import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.functions.preprocessing import preprocess_df  # noqa: E402


def test_no_missing_after_preprocess():
    """Ensure no NaNs remain after preprocessing."""
    df = pd.read_csv(ROOT / "data" / "titanic.csv")
    processed = preprocess_df(df)
    assert processed.isna().sum().sum() == 0, "Preprocessed dataframe still contains NaNs"


def test_shape_consistency():
    """Ensure Survived column still matches dataframe rows."""
    df = pd.read_csv(ROOT / "data" / "titanic.csv")
    processed = preprocess_df(df)
    assert "Survived" in processed.columns, "'Survived' column missing after preprocessing"
    assert len(processed) == processed["Survived"].shape[0], "Row count mismatch"