from pathlib import Path

import pandas as pd

from src.revenue_recovery.features.transform import build_recovery_dataset


DATA_PATH = Path("data/synthetic/payment_events.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    X, y = build_recovery_dataset(df)

    analysis = X.copy()
    analysis["recovered"] = y.values

    numeric_features = [
        "transaction_amount",
        "attempt_number",
        "previous_successful_payments",
        "previous_failed_payments",
        "customer_tenure_days",
        "total_previous_payments",
        "historical_success_rate",
        "failure_history_ratio",
    ]

    print("NUMERICAL FEATURE CORRELATION WITH RECOVERY")
    print("=" * 55)

    correlations = (
        analysis[numeric_features + ["recovered"]]
        .corr(numeric_only=True)["recovered"]
        .drop("recovered")
        .sort_values(key=abs, ascending=False)
    )

    print(correlations)

    print()
    print("RECOVERY RATE BY FEATURE QUARTILES")
    print("=" * 55)

    for feature in numeric_features:
        analysis["bin"] = pd.qcut(
            analysis[feature],
            q=4,
            duplicates="drop",
        )

        rates = analysis.groupby(
            "bin",
            observed=True,
        )["recovered"].mean()

        print()
        print(feature)
        print(rates)


if __name__ == "__main__":
    main()