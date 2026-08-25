from pathlib import Path

import pandas as pd

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
    # 1. Load recovery dataset
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    recovery_df = df[
        df["recovery_attempted"] == True
    ].copy()

    X, y = build_recovery_dataset(df)

    # ---------------------------------------------------------
    # 2. Train baseline model
    # ---------------------------------------------------------

    model = build_baseline_model()
    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]

    recovery_df["predicted_probability"] = probabilities

    # ---------------------------------------------------------
    # 3. Evaluate revenue at each threshold
    # ---------------------------------------------------------

    print("REVENUE-BASED THRESHOLD ANALYSIS")
    print("=" * 100)

    print(
        f"{'Threshold':<12}"
        f"{'Targeted %':<12}"
        f"{'Targeted':<12}"
        f"{'Actual Rec.':<12}"
        f"{'Recovery %':<12}"
        f"{'Revenue':<15}"
        f"{'Revenue/Target':<18}"
    )

    print("-" * 100)

    for threshold in THRESHOLDS:

        targeted = recovery_df[
            recovery_df["predicted_probability"] >= threshold
        ]

        targeted_count = len(targeted)

        if targeted_count == 0:
            continue

        recovered_count = targeted["recovered"].sum()

        recovery_rate = (
            recovered_count / targeted_count
        )

        total_revenue = targeted["recovered_amount"].sum()

        revenue_per_target = (
            total_revenue / targeted_count
        )

        targeted_percentage = (
            targeted_count / len(recovery_df)
        )

        print(
            f"{threshold:<12.2f}"
            f"{targeted_percentage:<12.3f}"
            f"{targeted_count:<12}"
            f"{recovered_count:<12}"
            f"{recovery_rate:<12.3f}"
            f"{total_revenue:<15.2f}"
            f"{revenue_per_target:<18.2f}"
        )


if __name__ == "__main__":
    main()