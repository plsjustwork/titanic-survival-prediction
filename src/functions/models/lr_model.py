import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def train_logistic_regression(X_train, y_train):
    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l1", "l2"]}

    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    lr_grid.fit(X_train, y_train)

    return {"best_model": lr_grid.best_estimator_, "best_params": lr_grid.best_params_}
