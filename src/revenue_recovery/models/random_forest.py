from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.revenue_recovery.features.preprocessing import build_preprocessor


def build_random_forest_model() -> Pipeline:
    """
    Build the baseline Random Forest recovery prediction pipeline.
    """

    preprocessor = build_preprocessor()

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline