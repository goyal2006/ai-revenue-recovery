from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.revenue_recovery.features.transform import build_recovery_dataset
from src.revenue_recovery.models.baseline import build_baseline_model


DATA_PATH = Path("data/synthetic/payment_events.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # 2. Build model-ready recovery dataset
    # ---------------------------------------------------------

    X, y = build_recovery_dataset(df)

    print("Recovery dataset:", X.shape)
    print("Target:", y.shape)
    print()

    # ---------------------------------------------------------
    # 3. Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("Train recovery rate:", round(y_train.mean(), 4))
    print("Test recovery rate:", round(y_test.mean(), 4))
    print()

    # ---------------------------------------------------------
    # 4. Build baseline pipeline
    # ---------------------------------------------------------

    model = build_baseline_model()

    # ---------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 6. Predictions
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 7. Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("BASELINE LOGISTIC REGRESSION")
    print("=" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print()

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("Classification Report")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()