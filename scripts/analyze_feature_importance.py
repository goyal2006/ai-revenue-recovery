from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.revenue_recovery.features.transform import build_recovery_dataset
from src.revenue_recovery.models.random_forest import build_random_forest_model


DATA_PATH = Path("data/synthetic/payment_events.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # 2. Build recovery dataset
    # ---------------------------------------------------------

    X, y = build_recovery_dataset(df)

    # ---------------------------------------------------------
    # 3. Same train/test split as baseline experiments
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------------------------------------------------------
    # 4. Train Random Forest
    # ---------------------------------------------------------

    pipeline = build_random_forest_model()

    pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Get fitted preprocessing pipeline
    # ---------------------------------------------------------

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # ---------------------------------------------------------
    # 6. Get feature names after preprocessing
    # ---------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()

    # ---------------------------------------------------------
    # 7. Get Random Forest feature importance
    # ---------------------------------------------------------

    importances = model.feature_importances_

    # ---------------------------------------------------------
    # 8. Build importance table
    # ---------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    # ---------------------------------------------------------
    # 9. Display results
    # ---------------------------------------------------------

    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 50)
    print(importance_df.to_string(index=False))

    print()
    print("Top 10 features")
    print("=" * 50)
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()