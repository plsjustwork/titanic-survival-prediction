import os
import pandas as pd

# ------------------------ 1️⃣ Load & Explore Data ------------------------


def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def explore_data(data):
    print(data.head())
    print(data.info())
    print(data.isnull().sum())
