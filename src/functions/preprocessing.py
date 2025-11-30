import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ------------------------ 2️⃣ Data Preprocessing ------------------------

def drop_unnecessary_columns(df):
    cols_to_drop = ["Name","Cabin","PassengerId"]
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns])

def impute_missing_values(df):
    # numerical simple imputation
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    return df

def remove_outliers(X, y):
    X = X.copy()
    y = y.copy()

    num_cols = X.select_dtypes(include=[np.number]).columns

    Q1 = X[num_cols].quantile(0.25)
    Q3 = X[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    # Boolean mask for rows WITHOUT outliers
    mask = ~((X[num_cols] < (Q1 - 1.5 * IQR)) | 
             (X[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

    # Return filtered X AND y
    return X[mask], y[mask]


def build_preprocessor(df):

    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns
    if "AgeBand" in numeric_features:
        numeric_features = numeric_features.drop("AgeBand")
        categorical_features = list(categorical_features) + ["AgeBand"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor

def preprocess_df(df):
    df = drop_unnecessary_columns(df)
    df = impute_missing_values(df)
    return df
