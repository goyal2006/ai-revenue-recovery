from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.revenue_recovery.features.transform import build_recovery_dataset
from src.revenue_recovery.models.baseline import build_baseline_model


DATA_PATH = Path("data/synthetic/payment_events.csv")

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


def main():
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X, y = build_recovery_dataset(df)

    # ---------------------------------------------------------
    # 2. Train baseline model on full recovery dataset
    # ---------------------------------------------------------

    model = build_baseline_model()

    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]

    # ---------------------------------------------------------
    # 3. Evaluate different decision thresholds
    # ---------------------------------------------------------

    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Accuracy':<12}"
        f"{'Targeted %':<12}"
    )

    print("-" * 80)

    for threshold in THRESHOLDS:

        predictions = probabilities >= threshold

        precision = precision_score(
            y,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        accuracy = accuracy_score(
            y,
            predictions,
        )

        targeted_rate = predictions.mean()

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
            f"{accuracy:<12.4f}"
            f"{targeted_rate:<12.4f}"
        )


if __name__ == "__main__":
    main()