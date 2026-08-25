from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.revenue_recovery.features.transform import build_recovery_dataset
from src.revenue_recovery.models.random_forest import build_random_forest_model


DATA_PATH = Path("data/synthetic/payment_events.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def get_logical_feature(feature_name: str) -> str:
    """
    Convert transformed feature names back into their
    original logical feature groups.
    """

    if feature_name.startswith("numeric__"):
        return feature_name.replace("numeric__", "")

    if feature_name.startswith("categorical__failure_reason_"):
        return "failure_reason"

    if feature_name.startswith("categorical__payment_method_"):
        return "payment_method"

    return feature_name


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

    pipeline = build_random_forest_model()

    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "transformed_feature": feature_names,
            "importance": importances,
        }
    )

    importance_df["logical_feature"] = (
        importance_df["transformed_feature"]
        .apply(get_logical_feature)
    )

    aggregated = (
        importance_df
        .groupby("logical_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    aggregated["importance_percent"] = (
        aggregated["importance"] * 100
    )

    print("AGGREGATED RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 55)

    print(
        aggregated[
            ["logical_feature", "importance_percent"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()