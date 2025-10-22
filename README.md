# Titanic Survival Prediction
![Python](https://img.shields.io/badge/python-3.12-blue)
![CV](https://img.shields.io/badge/CV-0.845-blue.svg)
[![CI](https://github.com/plsjustwork/titanic-survival-prediction/workflows/CI/badge.svg)](https://github.com/plsjustwork/titanic-survival-prediction/actions)

Predict passenger survival with classical ML (Logistic Regression, Random-Forest, XGBoost) in a **fully-reproducible** pipeline powered by GitHub Actions, showcasing data preprocessing, model evaluation, and feature importance visualization.

---

## 📊 Project Overview

The goal is to build a predictive model to determine which passengers survived the Titanic disaster based on passenger features.  
This project demonstrates:

- Data loading, exploration, and cleaning
- Handling missing values
- Encoding categorical variables
- Feature scaling for Logistic Regression
- Train/test split with validation
- Random Forest classifier for predictions
- Model evaluation with:
    - Accuracy
    - Confusion Matrix
    - Classification Report
- SHAP values to explain feature importance
- Plots saved automatically in outputs/ folder
- Comparing model outputs and feature importance
- Cross-validation for Random Forest
  
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
├── .dvc/
│   ├── .gitignore
│   └── config
├── .github/
│   ├── workflows/
│   │    ├── ci.yml
├── .tests/
│   ├── __pycache__
│   │    ├── test_preprocessing.cpython-312-pytest-8.4.2.pyc
│   └── test_preprocessing.py
├── outputs/                     # Generated outputs
│   ├── confusion_lr.png
│   ├── rf_feature_importance.png 
│   ├── shap_beeswarm.png
│   ├── val_curve_depth.png
│   ├── cm_lr_testset.html
│   └── cm_rf_testset.html
├── .dvcignore
├── .gitignore
├── requirements.txt
├── setup.cfg
├── test.csv
├── train.csv                    # Original Titanic dataset
├── titanic_survival.py          # Python script with preprocessing, modeling, and evaluation
└── README.md
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

# 3. reproduce the entire pipeline
dvc repro                 # pulls data, trains, evaluates, writes outputs/

# 4.Run the script using the bash:
  python titanic_survival.py

# 5. (optional) run tests & linting
pytest
flake8
```
## 📌 Notes

- Random Forest cross-validation score is printed at the end of the script.
- The /outputs folder ensures all visualizations are saved and not overwritten in case you're running the code on local and didn't install the /outputs folder.
- Feel free to modify train_test_split or Random Forest hyperparameters for experimentation.
