import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 2, 4]
    }

    gs = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_

    return {
        "best_model": best_model,
        "best_params": gs.best_params_,
        "importances": best_model.feature_importances_
    }
