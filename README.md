![Python](https://img.shields.io/badge/python-3.12-blue)
# Titanic Survival Prediction

This project predicts passenger survival on the Titanic using **Logistic Regression** and **Random Forest Classifier**, showcasing data preprocessing, model evaluation, and feature importance visualization.

---

## 📊 Project Overview

The goal is to build a predictive model to determine which passengers survived the Titanic disaster based on passenger features.  
This project demonstrates:

- Data cleaning and preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Model training and evaluation
- Comparing model outputs and feature importance

---

## 🗂 Dataset

The dataset comes from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/data), containing features like:

- PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked with the Target variable being `Survived` (0 = did not survive, 1 = survived).

**Preprocessing steps:**

- Dropped irrelevant columns: `Name`, `Ticket`, `Cabin`, `PassengerId`
- Filled missing `Age` values with the median
- Filled missing `Embarked` values with the mode (`S`)
- Encoded `Sex` (male = 0, female = 1) and used one-hot encoding for `Embarked`
- Scaled features for Logistic Regression

---

## 🧠 Models Used

### Logistic Regression
- Used to analyze the impact of each feature on survival
- Positive coefficients indicate features increasing survival chances
- Negative coefficients indicate features decreasing survival chances

### Random Forest Classifier
- Provides feature importance to understand which features influence predictions the most
- Does not require feature scaling

---

## 📈 Evaluation Metrics

- **Accuracy**: measures overall model correctness
- **Confusion Matrix**: shows true positives, false positives, true negatives, false negatives
- **Classification Report**: precision, recall, f1-score

Example outputs:
 - Accuracy: 0.80
 - Confusion Matrix:
    [[89 16]
    [19 55]]
 -Classification Report:
  | Class | Precision | Recall | F1-Score | Support |
  |-------|-----------|--------|----------|---------|
  | 0     | 0.82      | 0.85   | 0.84     | 105     |
  | 1     | 0.78      | 0.74   | 0.76     | 74      |
  | **Accuracy** |       |        | **0.80** | 179     |
  | **Macro Avg** | 0.80 | 0.79   | 0.80     | 179     |
  | **Weighted Avg** | 0.80 | 0.80 | 0.80     | 179     |

---

## 🔍 Feature Importance

### Logistic Regression Coefficients
- Positive: increase survival chance  
- Negative: decrease survival chance  

Top 3 positive features:
- Sex (female)  
- Fare  
- Passenger Class (Pclass = 1)

Top 3 negative features:
- Pclass (3rd class)  
- Age  
- SibSp

### Random Forest Feature Importance

![Random Forest Feature Importance](rf_feature_importance.png)

- The figure shows the most and least influential features in descending order

---

## 📝 Conclusion
Logistic Regression highlights that being female, paying a higher fare, and traveling in first class increased survival chances, while age and traveling in larger families decreased it. Random Forest confirms these insights and shows which features are most important for predictions.

## 📂 File Structure

  titanic-survival/
  │
  ├── train.csv # Original Titanic dataset
  ├── titanic_survival.py # Python script with preprocessing, modeling, and evaluation
  ├── rf_feature_importance.png
  └── README.md
---

## 💻 How to Run

1. Clone the repository by pasting this code in bash:
  git clone https://github.com/yourusername/titanic-survival.git
  cd titanic-survival

2.Install dependencies needed to run this code by pasting this code in bash afterwards:
  pip install pandas scikit-learn seaborn matplotlib

3.Run the script using the bash:
  python titanic_survival.py
