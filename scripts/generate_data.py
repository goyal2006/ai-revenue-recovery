from pathlib import Path

from src.revenue_recovery.data.generator import generate_payment_events
from src.revenue_recovery.data.validation import validate_payment_events


def main() -> None:
    output_path = Path("data/synthetic/payment_events.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_payment_events(
        n_rows=10_000,
        seed=42,
    )

    validate_payment_events(df)

    df.to_csv(
        output_path,
        index=False,
    )

    print("Data validation: PASSED")
    print(f"Generated {len(df):,} payment events.")
    print(f"Saved to: {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()