from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

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

RECOVERY_COSTS = [
    5,
    10,
    20,
    30,
]


def generate_oof_predictions(X, y):
    """
    Generate out-of-fold probability predictions.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    oof_probabilities = np.zeros(len(X))

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        model = clone(build_baseline_model())

        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        oof_probabilities[valid_idx] = model.predict_proba(
            X.iloc[valid_idx]
        )[:, 1]

        print(f"Completed fold {fold}")

    return oof_probabilities


def main():

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    recovery_df = df[
        df["recovery_attempted"] == True
    ].copy()

    X, y = build_recovery_dataset(df)

    # ---------------------------------------------------------
    # 2. Generate OOF predictions
    # ---------------------------------------------------------

    probabilities = generate_oof_predictions(X, y)

    recovery_df["oof_probability"] = probabilities

    # ---------------------------------------------------------
    # 3. Analyze each recovery cost
    # ---------------------------------------------------------

    for recovery_cost in RECOVERY_COSTS:

        print()
        print("=" * 110)
        print(
            f"RECOVERY ACTION COST: ₹{recovery_cost}"
        )
        print("=" * 110)

        print(
            f"{'Threshold':<12}"
            f"{'Targeted':<12}"
            f"{'Recovered':<12}"
            f"{'Revenue':<15}"
            f"{'Action Cost':<15}"
            f"{'Net Revenue':<15}"
            f"{'Net/Target':<15}"
        )

        print("-" * 110)

        results = []

        for threshold in THRESHOLDS:

            targeted = recovery_df[
                recovery_df["oof_probability"] >= threshold
            ]

            targeted_count = len(targeted)

            if targeted_count == 0:
                continue

            recovered_count = targeted["recovered"].sum()

            total_revenue = targeted[
                "recovered_amount"
            ].sum()

            total_action_cost = (
                targeted_count * recovery_cost
            )

            net_revenue = (
                total_revenue
                - total_action_cost
            )

            net_per_target = (
                net_revenue / targeted_count
            )

            results.append(
                {
                    "threshold": threshold,
                    "targeted": targeted_count,
                    "recovered": recovered_count,
                    "revenue": total_revenue,
                    "action_cost": total_action_cost,
                    "net_revenue": net_revenue,
                    "net_per_target": net_per_target,
                }
            )

            print(
                f"{threshold:<12.2f}"
                f"{targeted_count:<12}"
                f"{recovered_count:<12}"
                f"{total_revenue:<15.2f}"
                f"{total_action_cost:<15.2f}"
                f"{net_revenue:<15.2f}"
                f"{net_per_target:<15.2f}"
            )

        # -----------------------------------------------------
        # 4. Find best threshold
        # -----------------------------------------------------

        results_df = pd.DataFrame(results)

        best = results_df.loc[
            results_df["net_revenue"].idxmax()
        ]

        print()
        print(
            f"BEST THRESHOLD: {best['threshold']:.2f}"
        )

        print(
            f"Maximum net revenue: "
            f"₹{best['net_revenue']:.2f}"
        )

        print(
            f"Customers targeted: "
            f"{int(best['targeted'])}"
        )

        print(
            f"Recovered transactions: "
            f"{int(best['recovered'])}"
        )


if __name__ == "__main__":
    main()