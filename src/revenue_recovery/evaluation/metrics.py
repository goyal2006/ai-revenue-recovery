from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_proba,
) -> dict[str, Any]:
    """
    Evaluate a binary classification model.

    Parameters
    ----------
    y_true:
        True binary labels.

    y_pred:
        Predicted binary labels.

    y_proba:
        Predicted probability for the positive class.

    Returns
    -------
    dict
        Standard classification metrics and confusion matrix.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_proba,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }