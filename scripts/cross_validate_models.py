from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.revenue_recovery.features.transform import build_recovery_dataset
from src.revenue_recovery.models.baseline import build_baseline_model
from src.revenue_recovery.models.random_forest import build_random_forest_model


DATA_PATH = Path("data/synthetic/payment_events.csv")
RANDOM_STATE = 42
N_SPLITS = 5


def main():
    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # 2. Build model-ready dataset
    # ---------------------------------------------------------

    X, y = build_recovery_dataset(df)

    print("Dataset:", X.shape)
    print("Recovery rate:", round(y.mean(), 4))
    print()

    # ---------------------------------------------------------
    # 3. Stratified K-Fold
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------
    # 4. Models
    # ---------------------------------------------------------

    models = {
        "Logistic Regression": build_baseline_model(),
        "Random Forest": build_random_forest_model(),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    # ---------------------------------------------------------
    # 5. Cross-validation
    # ---------------------------------------------------------

    for name, model in models.items():

        results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        print(name)
        print("=" * 50)

        for metric in scoring:
            scores = results[f"test_{metric}"]

            print(
                f"{metric:10s}: "
                f"{scores.mean():.4f} "
                f"+/- {scores.std():.4f}"
            )

        print()


if __name__ == "__main__":
    main()