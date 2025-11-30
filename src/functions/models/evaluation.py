import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
def plot_confusion_matrix(cm, title, output_path=None):
    """Plot and optionally save a confusion matrix."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")

    plt.close()

def evaluate_model(model, X_val, y_val, X_test, y_test,model_name="Model", save_plots=False):
    """Evaluate any trained classifier (LR, RF, etc.)"""
    # -------------------- Validation evaluation --------------------
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    # ------------------------ Test evaluation -----------------------
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    test_report = classification_report(y_test, test_pred, output_dict=False)

    print(f"\n==================== {model_name} Evaluation ====================")
    print(f"Validation Accuracy : {val_acc:.3f}")
    print(f"Test Accuracy       : {test_acc:.3f}")
    print("\nClassification Report (Test Set):")
    print(test_report)
    print("\nConfusion Matrix (Test Set):")
    print(test_cm)

    # -------------------- Optional save of CM plots --------------------
    if save_plots:
        plot_confusion_matrix(
            test_cm,
            title=f"{model_name} – Confusion Matrix (Test Set)",
            output_path=f"outputs/{model_name.lower()}_cm.png"
        )

    # -------------------------- Return results -------------------------
    return {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "y_pred_val": val_pred,
        "y_pred_test": test_pred,
        "confusion_matrix": test_cm,
        "classification_report": test_report,
    }