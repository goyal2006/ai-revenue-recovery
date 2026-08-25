

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "transaction_amount",
    "attempt_number",
    "previous_successful_payments",
    "previous_failed_payments",
    "customer_tenure_days",
    "total_previous_payments",
    "historical_success_rate",
    "failure_history_ratio",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "failure_reason",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline for recovery prediction.

    Numerical features:
        - Missing values are replaced with the median.
        - Features are standardized.

    Categorical features:
        - Missing values are replaced with the most frequent value.
        - Categories are one-hot encoded.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor