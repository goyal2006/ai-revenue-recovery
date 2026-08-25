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
from src.revenue_recovery.models.random_forest import build_random_forest_model


DATA_PATH = Path("data/synthetic/payment_events.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    df = pd.read_csv(DATA_PATH)

    X, y = build_recovery_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_random_forest_model()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("RANDOM FOREST")
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