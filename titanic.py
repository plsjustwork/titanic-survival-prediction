# ================= Titanic Survival Prediction=================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# ------------------------ 1️⃣ Load & Explore Data ------------------------


def load_data(path="train.csv"):
    """Load Titanic dataset."""
    data = pd.read_csv(path)
    return data


def explore_data(data):
    print(data.head())
    print(data.info())
    print(data.isnull().sum())


# ------------------------ 2️⃣ Data Preprocessing ------------------------
def preprocess_data(data):
    """Clean and encode Titanic data."""
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)

    # Handle missing values using SimpleImputer
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Handle missing values
    data["Age"] = num_imputer.fit_transform(data[["Age"]])
    data["Embarked"] = cat_imputer.fit_transform(data[["Embarked"]]).ravel()

    # Encode categorical variables
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data = pd.get_dummies(data, columns=["Embarked"])

    # Drop PassengerId as it is not useful
    X = data.drop(["Survived", "PassengerId"], axis=1)
    y = data["Survived"]
    return X, y


# ------------------------ 5️⃣ Logistic Regression ------------------------
def train_logistic_regression(X_train, y_train, X_val, X_test, y_val, y_test):
    """Train and evaluate Logistic Regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=500, solver="liblinear")
    lr.fit(X_train_scaled, y_train)
    print(f"Validation accuracy (LR): {lr.score(X_val_scaled, y_val):.3f}")
    print(f"Hold-out test accuracy (LR): {lr.score(X_test_scaled, y_test):.3f}")
    y_pred = lr.predict(X_test_scaled)

    print("=== Logistic Regression Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Not Survived", "Survived"]).plot(
        cmap="Blues"
    )
    plt.title("Confusion Matrix - Logistic Regression")
    plt.savefig("outputs/confusion_lr.png", bbox_inches="tight")

    # Logistic Regression Coefficients (impact on survival)
    coefficients = pd.Series(lr.coef_[0], index=X_train.columns).sort_values(
        ascending=False
    )
    print("\nTop + Coefficients:\n", coefficients.head(3))
    print("Top - Coefficients:\n", coefficients.tail(3))
    return lr, coefficients


# ------------------------ 6️⃣ Random Forest ------------------------
def train_random_forest(X_train, y_train, X_val, X_test, y_val, y_test):
    """Train and evaluate Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)  # Trees do not need scaling
    print(f"Validation accuracy (RF): {rf.score(X_val, y_val):.3f}")
    print(f"Hold-out test accuracy (RF): {rf.score(X_test, y_test):.3f}")

    y_pred = rf.predict(X_test)

    print("\n=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Prepare DataFrame for visualization
    feature_importances = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": rf.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    # Plot Random Forest Feature Importances
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importances,
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Random Forest Feature Importances")
    plt.savefig("outputs/rf_feature_importance.png", bbox_inches="tight")
    return rf, feature_importances


def main():
    os.makedirs("outputs", exist_ok=True)
    data = load_data()
    explore_data(data)
    X, y = preprocess_data(data)
    # ---- Optional: Remove outliers (based on numeric columns) ----
    num_cols = X.select_dtypes(include=["number"]).columns
    Q1 = X[num_cols].quantile(0.25)
    Q3 = X[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    outlier_condition = ((X[num_cols] < (Q1 - 1.5 * IQR)) | (X[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    X_no_outliers = X[~outlier_condition]
    y_no_outliers = y.loc[X_no_outliers.index]

    print(f"Shape before removing outliers: {X.shape}")
    print(f"Shape after removing outliers: {X_no_outliers.shape}")

    # Replace X, y with filtered data
    X, y = X_no_outliers, y_no_outliers

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.176,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    # 0.85*0.176 ≈ 0.15  → final split 70 / 15 / 15

    lr, coef = train_logistic_regression(X_train, y_train, X_val, X_test, y_val, y_test)
    rf, feat_imp = train_random_forest(X_train, y_train, X_val, X_test, y_val, y_test)

    # ------------------------ 7️⃣ Compare LR Coefficients vs RF Importances ------------------------
    comparison = pd.DataFrame(
        {"LR Coefficients": lr.coef_[0], "RF Importances": rf.feature_importances_},
        index=X_train.columns,
        ).sort_values(by="RF Importances", ascending=False)

    print("\n=== Comparison: Logistic Regression vs Random Forest ===")
    print(comparison.round(3))

    cv_score = cross_val_score(rf, X_train, y_train, cv=5).mean()
    print(f"Cross-Validation Accuracy (RF, train fold): {cv_score:.3f}")


if __name__ == "__main__":
    main()

# ========================== End of Project ==========================
