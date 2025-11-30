from src.loader.le_data import load_data, explore_data
from src.functions.preprocessing import (
    preprocess_df,
    build_preprocessor,
    remove_outliers,
)
from src.functions.models import lr_model, rf_model, evaluation
from src.functions.feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split


def main():

    print("\n🔍 Loading dataset...")
    df = load_data("data/titanic.csv")
    print(f"Dataset loaded with shape: {df.shape}")
    explore_data(df)
    print("\n🧹 Starting preprocessing...")
    df_clean = preprocess_df(df)
    df_clean = feature_engineering(df_clean)
    print(f"After preprocessing shape: {df_clean.shape}")

    print("\n📊 Splitting into train/validation/test...")

    X = df_clean.drop(columns=["Survived"])
    y = df_clean["Survived"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    X_train, y_train = remove_outliers(X_train, y_train)

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print("Number of features after encoding:", X_train_processed.shape[1])

    print("\n🤖 Training Logistic Regression model...")
    lr_results = lr_model.train_logistic_regression(X_train_processed, y_train)
    lr_model_trained = lr_results["best_model"]

    print("\n🌲 Training Random Forest model...")
    rf_results = rf_model.train_random_forest(X_train_processed, y_train)
    rf_model_trained = rf_results["best_model"]

    print("\n📉 Validation Results:")
    lr_val_results = evaluation.evaluate_model(     # noqa: F841
        lr_model_trained,
        X_val_processed,
        y_val,
        X_test_processed,
        y_test,
        model_name="LogisticRegression",
        save_plots=True,
    )

    rf_val_results = evaluation.evaluate_model(  # noqa: F841
        rf_model_trained,
        X_val_processed,
        y_val,
        X_test_processed,
        y_test,
        model_name="RandomForest",
        save_plots=True,
    )


if __name__ == "__main__":
    main()
