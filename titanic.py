# ================= Titanic Survival Prediction=================

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split, validation_curve)
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
TEST_SIZE = 0.15
VAL_SIZE = 0.176
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
    """Train and evaluate TUNED Logistic Regression for fair comparison."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Hyper-parameter tuning (same 5-fold CV as Random-Forest)
    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]}
    lr_grid = GridSearchCV(
        LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=RANDOM_STATE
        ),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    lr_grid.fit(X_train_scaled, y_train)
    best_lr = lr_grid.best_estimator_

    print("LR best CV score :", f"{lr_grid.best_score_:.3f}")
    print("LR best params   :", lr_grid.best_params_)
    print(f"Tuned LR validation acc : {best_lr.score(X_val_scaled, y_val):.3f}")
    print(f"Tuned LR hold-out test acc: {best_lr.score(X_test_scaled, y_test):.3f}")

    y_pred = best_lr.predict(X_test_scaled)
    print("=== Logistic Regression (tuned) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion matrix – Plotly version & coefficients of the TUNED model
    cm = confusion_matrix(y_test, best_lr.predict(X_test_scaled))
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues'
    ))
    fig.update_layout(
        title='Confusion Matrix – Logistic Regression (hold-out test set)',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    fig.write_html('outputs/cm_lr_testset.html')

    coefficients = pd.Series(best_lr.coef_[0], index=X_train.columns).sort_values(
        ascending=False
    )
    print("\nTop + Coefficients:\n", coefficients.head(3))
    print("Top - Coefficients:\n", coefficients.tail(3))
    return best_lr, coefficients


# ------------------------ 6️⃣ Random Forest ------------------------
def train_random_forest(X_train, y_train, X_val, X_test, y_val, y_test):
    """Train and evaluate Random Forest with hyper-parameter tuning."""
    # 1. Grid-search on training fold
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 2, 4],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    gs = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("Best CV score:", f"{gs.best_score_:.3f}")
    print("Best params:", gs.best_params_)
    best_model = gs.best_estimator_

    train_score, val_score = validation_curve(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        X_train,
        y_train,
        param_name="max_depth",
        param_range=range(1, 11),
        cv=5,
    )
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 11), val_score.mean(1), label="val")
    plt.plot(range(1, 11), train_score.mean(1), label="train")
    plt.legend()
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.title("Validation Curve – max_depth")
    plt.savefig("outputs/val_curve_depth.png", bbox_inches="tight")

    # 2. Evaluate the tuned model
    print(f"Validation accuracy (RF): {best_model.score(X_val, y_val):.3f}")
    print(f"Hold-out test accuracy (RF): {best_model.score(X_test, y_test):.3f}")
    y_pred = best_model.predict(X_test)
    print("\n=== Random Forest (tuned) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 3. Confusion matrix – Plotly version
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues'
    ))
    fig.update_layout(
        title='Confusion Matrix – Random Forest (hold-out test set)',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    fig.write_html('outputs/cm_rf_testset.html')

    # 3. Feature importances of the tuned model
    feature_importances = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": best_model.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importances,
        hue="Feature",
        palette="viridis",
        legend=False,
    )
    plt.title("Random Forest Feature Importances (tuned)")
    plt.savefig("outputs/rf_feature_importance.png", bbox_inches="tight")
    return best_model, feature_importances


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

    outlier_condition = (
        (X[num_cols] < (Q1 - 1.5 * IQR)) | (X[num_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)
    X_no_outliers = X[~outlier_condition]
    y_no_outliers = y.loc[X_no_outliers.index]

    print(f"Shape before removing outliers: {X.shape}")
    print(f"Shape after removing outliers: {X_no_outliers.shape}")

    # Replace X, y with filtered data
    X, y = X_no_outliers, y_no_outliers

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )
    # 0.85*0.176 ≈ 0.15  → final split 70 / 15 / 15

    lr, coef = train_logistic_regression(X_train, y_train, X_val, X_test, y_val, y_test)
    rf, feat_imp = train_random_forest(X_train, y_train, X_val, X_test, y_val, y_test)

    # SHAP visualization for Random Forest
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_val)
    shap.summary_plot(shap_values, X_val, show=False)
    plt.savefig("outputs/shap_beeswarm.png", bbox_inches="tight")
    plt.close()

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
