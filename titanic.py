# ================= Titanic Survival Prediction=================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
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
    data = data.drop(['Name', 'Ticket','Cabin'], axis=1)

    # Handle missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # most common: 'S'

    # Encode categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'])

    # Drop PassengerId as it is not useful
    X = data.drop(['Survived', 'PassengerId'], axis=1)
    y = data['Survived']
    return X, y

# ------------------------ 5️⃣ Logistic Regression ------------------------
def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Logistic Regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=500, solver='liblinear')
    lr.fit(X_train_scaled, y_train)
    y_pred = lr.predict(X_test_scaled)

    print("=== Logistic Regression Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Not Survived', 'Survived']).plot(cmap='Blues')
    plt.title("Confusion Matrix - Logistic Regression")
    plt.savefig("outputs/confusion_lr.png", bbox_inches="tight")

    # Logistic Regression Coefficients (impact on survival)
    coefficients = pd.Series(lr.coef_[0], index=X_train.columns).sort_values(ascending=False)
    print("\nTop + Coefficients:\n", coefficients.head(3))
    print("Top - Coefficients:\n", coefficients.tail(3))
    return lr, coefficients

# ------------------------ 6️⃣ Random Forest ------------------------
def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)  # Trees do not need scaling
    y_pred = rf.predict(X_test)
    
    print("\n=== Random Forest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Prepare DataFrame for visualization
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plot Random Forest Feature Importances
    plt.figure(figsize=(8,6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
    plt.title("Random Forest Feature Importances")
    plt.savefig("outputs/rf_feature_importance.png", bbox_inches="tight")
    return rf, feature_importances

def main():
    os.makedirs("outputs", exist_ok=True)
    data = load_data()
    explore_data(data)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr, coef = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf, feat_imp = train_random_forest(X_train, y_train, X_test, y_test)

    # ------------------------ 7️⃣ Compare LR Coefficients vs RF Importances ------------------------
    comparison = pd.DataFrame({
        'LR Coefficients': lr.coef_[0],
        'RF Importances': rf.feature_importances_
    }, index=X_train.columns).sort_values(by='RF Importances', ascending=False)

    print("\n=== Comparison: Logistic Regression vs Random Forest ===")
    print(comparison.round(3))

    print("\nCross-Validation Accuracy (RF):",
          round(cross_val_score(rf, X, y, cv=5).mean(), 3))

if __name__ == "__main__":
    main()

# ========================== End of Project ==========================
