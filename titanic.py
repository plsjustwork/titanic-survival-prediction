# ================= Titanic Survival Prediction=================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------ 1️⃣ Load & Explore Data ------------------------
data = pd.read_csv("train.csv")

# Quick look at the dataset
print(data.head())
print(data.info())
print(data.isnull().sum())

# ------------------------ 2️⃣ Data Preprocessing ------------------------
# Drop irrelevant columns
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # most common: 'S'

# Encode categorical variables
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'])

# Drop PassengerId as it is not useful
X = data.drop(['Survived', 'PassengerId'], axis=1)
y = data['Survived']

# ------------------------ 3️⃣ Train/Test Split ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------ 4️⃣ Feature Scaling ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------ 5️⃣ Logistic Regression ------------------------
lr = LogisticRegression(max_iter=500, solver='liblinear')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("=== Logistic Regression Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Logistic Regression Coefficients (impact on survival)
coefficients = pd.Series(lr.coef_[0], index=X_train.columns)
coefficients_sorted = coefficients.sort_values(ascending=False)
print("Logistic Regression Coefficients (Descending):\n", coefficients_sorted)


# ------------------------ 6️⃣ Random Forest ------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)  # Trees do not need scaling
importances = rf.feature_importances_

# Prepare DataFrame for visualization
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot Random Forest Feature Importances
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title("Random Forest Feature Importances (Sorted)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("rf_feature_importance.png")

# ------------------------ 7️⃣ Compare LR Coefficients vs RF Importances ------------------------
comparison = pd.DataFrame({
    'LR Coefficients': lr.coef_[0],
    'RF Importances': rf.feature_importances_
}, index=X_train.columns)

# Sort by RF Importances for easier interpretation
comparison_sorted = comparison.sort_values(by='RF Importances', ascending=False)
print("=== Comparison: Logistic Regression vs Random Forest ===")
print(comparison_sorted)

# ------------------------ 8️⃣ Top Positive & Negative Factors ------------------------
# Logistic Regression: Positive coefficients increase survival chance, negative decrease
top_lr_positive = coefficients_sorted.head(3)
top_lr_negative = coefficients_sorted.tail(3)
print("\nTop 3 features increasing survival chance (Logistic Regression):")
print(top_lr_positive)
print("\nTop 3 features decreasing survival chance (Logistic Regression):")
print(top_lr_negative)

# Random Forest: Higher importance = stronger influence
top_rf_features = feature_importances.head(3)
bottom_rf_features = feature_importances.tail(3)
print("\nTop 3 most influential features (Random Forest):")
print(top_rf_features)
print("\n3 least influential features (Random Forest):")
print(bottom_rf_features)

# ========================== End of Project ==========================
