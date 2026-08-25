import numpy as np
import pandas as pd


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

TARGET = "recovered"


def build_recovery_dataset(df: pd.DataFrame):
    """
    Build the model-ready recovery dataset.

    Only transactions where recovery was attempted are included.

    Returns
    -------
    X : pandas.DataFrame
        Feature dataframe.
    y : pandas.Series
        Recovery outcome.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # 1. Keep only recovery candidates
    # ---------------------------------------------------------

    data = data[data["recovery_attempted"] == True].copy()

    # ---------------------------------------------------------
    # 2. Create historical payment features
    # ---------------------------------------------------------

    data["total_previous_payments"] = (
        data["previous_successful_payments"]
        + data["previous_failed_payments"]
    )

    data["historical_success_rate"] = np.where(
        data["total_previous_payments"] > 0,
        data["previous_successful_payments"]
        / data["total_previous_payments"],
        0.0,
    )

    data["failure_history_ratio"] = np.where(
        data["total_previous_payments"] > 0,
        data["previous_failed_payments"]
        / data["total_previous_payments"],
        0.0,
    )

    # ---------------------------------------------------------
    # 3. Select model features
    # ---------------------------------------------------------

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = data[feature_columns].copy()

    # ---------------------------------------------------------
    # 4. Extract target
    # ---------------------------------------------------------

    y = data[TARGET].copy()

    return X, y