from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.revenue_recovery.features.preprocessing import build_preprocessor


def build_baseline_model() -> Pipeline:
    """
    Build the baseline recovery prediction model.

    Preprocessing and Logistic Regression are combined into
    one scikit-learn pipeline.
    """

    preprocessor = build_preprocessor()

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline