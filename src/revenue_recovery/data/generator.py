import numpy as np
import pandas as pd


def generate_payment_events(
    n_rows: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a reproducible synthetic payment-event dataset.
    """

    rng = np.random.default_rng(seed)

    customer_ids = [
        f"CUST_{i:05d}"
        for i in rng.integers(1, max(n_rows // 2, 2), size=n_rows)
    ]

    transaction_ids = [
        f"TXN_{i:06d}"
        for i in range(1, n_rows + 1)
    ]

    transaction_amount = np.round(
        rng.lognormal(mean=3.5, sigma=0.8, size=n_rows),
        2,
    )

    payment_method = rng.choice(
        ["card", "upi", "netbanking", "wallet"],
        size=n_rows,
        p=[0.45, 0.30, 0.15, 0.10],
    )

    payment_status = rng.choice(
        ["success", "failed"],
        size=n_rows,
        p=[0.85, 0.15],
    )

    failure_reason = np.where(
        payment_status == "failed",
        rng.choice(
            [
                "insufficient_funds",
                "card_declined",
                "expired_card",
                "network_error",
                "authentication_failed",
            ],
            size=n_rows,
        ),
        None,
    )

    attempt_number = rng.integers(1, 4, size=n_rows)

    previous_successful_payments = rng.poisson(
        lam=8,
        size=n_rows,
    )

    previous_failed_payments = rng.poisson(
        lam=1.5,
        size=n_rows,
    )

    customer_tenure_days = rng.integers(
        1,
        1500,
        size=n_rows,
    )

    recovery_attempted = (
        (payment_status == "failed")
        & (rng.random(n_rows) < 0.70)
    )

    recovery_action = np.where(
        recovery_attempted,
        rng.choice(
            [
                "retry_payment",
                "send_reminder",
                "update_payment_method",
                "offer_support",
            ],
            size=n_rows,
        ),
        None,
    )

    recovery_probability = np.full(n_rows, 0.45)

    recovery_probability += (
        previous_successful_payments * 0.01
    )

    recovery_probability -= (
        previous_failed_payments * 0.02
    )

    recovery_probability += (
        np.minimum(customer_tenure_days, 365) / 365 * 0.10
    )

    recovery_probability = np.where(
        failure_reason == "network_error",
        recovery_probability + 0.10,
        recovery_probability,
    )

    recovery_probability = np.where(
        failure_reason == "insufficient_funds",
        recovery_probability - 0.10,
        recovery_probability,
    )

    recovery_probability = np.clip(
        recovery_probability,
        0.05,
        0.95,
    )

    recovered = (
        recovery_attempted
        & (rng.random(n_rows) < recovery_probability)
    )

    recovered_amount = np.where(
        recovered,
        transaction_amount,
        0.0,
    )

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "transaction_id": transaction_ids,
            "transaction_amount": transaction_amount,
            "payment_method": payment_method,
            "payment_status": payment_status,
            "failure_reason": failure_reason,
            "attempt_number": attempt_number,
            "previous_successful_payments": previous_successful_payments,
            "previous_failed_payments": previous_failed_payments,
            "customer_tenure_days": customer_tenure_days,
            "recovery_attempted": recovery_attempted,
            "recovery_action": recovery_action,
            "recovered": recovered,
            "recovered_amount": recovered_amount,
        }
    )