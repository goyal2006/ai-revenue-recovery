import pandas as pd


def validate_payment_events(df: pd.DataFrame) -> None:
    """
    Validate the core invariants of the payment-event dataset.

    Raises:
        ValueError: If any data-quality rule is violated.
    """

    required_columns = {
        "customer_id",
        "transaction_id",
        "transaction_amount",
        "payment_method",
        "payment_status",
        "failure_reason",
        "attempt_number",
        "previous_successful_payments",
        "previous_failed_payments",
        "customer_tenure_days",
        "recovery_attempted",
        "recovery_action",
        "recovered",
        "recovered_amount",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction_id values found.")

    if (df["transaction_amount"] <= 0).any():
        raise ValueError(
            "transaction_amount must be greater than zero."
        )

    successful_with_recovery = (
        (df["payment_status"] == "success")
        & df["recovery_attempted"]
    )

    if successful_with_recovery.any():
        raise ValueError(
            "Successful payments cannot have recovery attempts."
        )

    recovered_without_attempt = (
        df["recovered"]
        & ~df["recovery_attempted"]
    )

    if recovered_without_attempt.any():
        raise ValueError(
            "A payment cannot be recovered without a recovery attempt."
        )

    recovered_amount_mismatch = (
        df["recovered"]
        & (
            df["recovered_amount"]
            != df["transaction_amount"]
        )
    )

    if recovered_amount_mismatch.any():
        raise ValueError(
            "Recovered rows must have recovered_amount "
            "equal to transaction_amount."
        )

    unrecovered_with_amount = (
        ~df["recovered"]
        & (df["recovered_amount"] != 0)
    )

    if unrecovered_with_amount.any():
        raise ValueError(
            "Unrecovered rows must have recovered_amount equal to 0."
        )