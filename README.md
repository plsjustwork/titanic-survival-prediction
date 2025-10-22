# Titanic Survival Prediction
![Python](https://img.shields.io/badge/python-3.12-blue)
![CV](https://img.shields.io/badge/CV-0.845-blue.svg)
[![CI](https://github.com/plsjustwork/titanic-survival-prediction/workflows/CI/badge.svg)](https://github.com/plsjustwork/titanic-survival-prediction/actions)

Predict passenger survival with classical ML (Logistic Regression, Random-Forest) in a fully-reproducible pipeline powered by GitHub Actions. Includes data preprocessing, model evaluation, hyperparameter tuning, cross-validation, feature importance, and SHAP explainability. All outputs are saved automatically in `outputs/`.

---

## 📊 Project Overview

This project builds predictive models to determine Titanic passenger survival based on passenger features.
Updates from previous version include:

- Removal of outliers from numeric features
- Hyperparameter tuning for Logistic Regression and Random Forest
- Cross-validation for Random Forest
- SHAP summary plots for feature importance
- Validation curves for Random Forest depth
- Comparison of LR coefficients vs RF importanc
  
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
- Produces a confusion matrix visualization

### Random Forest Classifier
- Provides feature importance to understand which features influence predictions the most
- Does not require feature scaling
- Includes cross-validation for robustness
  
---

## 📈 Evaluation Metrics

- **Accuracy**: measures overall model correctness
- **Confusion Matrix**: shows true positives, false positives, true negatives, false negatives
- **Classification Report**: precision, recall, f1-score

Example outputs:
 - Accuracy: 0.81
 - Classification reports are printed in the console
 - Confusion Matrix (Logistic Regression):

![Confusion Matrix - Logistic Regression](outputs/confusion_lr.png)

---

## 🔍 Feature Importance

### Logistic Regression Coefficients
- Positive: increase survival chance  
- Negative: decrease survival chance 
- Top positive and negative features are printed in the console.

### Random Forest Feature Importance

![Random Forest Feature Importances](outputs/rf_feature_importance.png)

- Shows the most and least influential features

---

## 📝 Conclusion
Logistic Regression highlights that being female, paying a higher fare, and traveling in first class increased survival chances, while age and traveling in larger families decreased it. Random Forest confirms these insights and shows which features are most important for predictions.

## 📂 File Structure
```
titanic-survival-prediction/
│
├── train.csv, test.csv        # Kaggle data (DVC-tracked)
├── titanic.py                 # end-to-end script (pipeline)
├── outputs/                   # auto-generated plots & metrics (created by script)
├── tests/                     # pytest suite
└── requirements.txt
```
---

## 💻 How to Run

```bash
# 1. clone & enter
git clone https://github.com/plsjustwork/titanic-survival-prediction.git
cd titanic-survival-prediction

# 2. create environment
python -m venv .venv && source .venv/bin/activate  # Win: .venv\Scripts\activate
pip install -r requirements.txt

# 3.Run the script using the bash:
python titanic.py

# 4. (optional) run tests & linting
pytest
flake8
```
## 📌 Notes

- Random Forest cross-validation score is printed at the end of the script.
- The /outputs folder ensures all visualizations are saved and not overwritten in case you're running the code on local and didn't install the /outputs folder.
- Feel free to modify train_test_split or Random Forest hyperparameters for experimentation.
