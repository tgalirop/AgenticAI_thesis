"""Create leakage-safe PaySim features for the main experiment."""

from __future__ import annotations

import polars as pl


# The four balances are excluded from the main experiment following the PaySim
# methodological specification: simulated fraudulent transfers are cancelled, so
# post-transaction balances can expose how the simulator generated the label.
# isFlaggedFraud is itself the output of an existing fraud rule.  Account IDs are
# removed to prevent memorisation of individual accounts.
MAIN_EXPERIMENT_EXCLUDED_COLUMNS = (
    "isFlaggedFraud",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "nameOrig",
    "nameDest",
)


def add_features(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Add deterministic features that use only information available at scoring.

    PaySim's ``step`` starts at 1 and represents elapsed hours.  Subtracting one
    makes hour 1 map to hour-of-day 0 and day 0.  ``log1p`` reduces amount skew
    while remaining defined for zero-valued transactions.

    The merchant flag is derived before dropping ``nameDest``.  It uses only the
    documented account prefix and never target statistics, so it is leakage-safe.
    """

    required = {"step", "type", "amount", "nameDest"}
    columns = set(frame.collect_schema().names()) if isinstance(frame, pl.LazyFrame) else set(frame.columns)
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"Cannot create features; missing columns: {', '.join(missing)}")

    return frame.with_columns(
        ((pl.col("step") - 1) % 24).cast(pl.Int16).alias("hour"),
        ((pl.col("step") - 1) // 24).cast(pl.Int16).alias("day"),
        pl.col("amount").log1p().alias("log_amount"),
        (pl.col("type") == "TRANSFER").cast(pl.Int8).alias("is_transfer"),
        (pl.col("type") == "CASH_OUT").cast(pl.Int8).alias("is_cash_out"),
        pl.col("nameDest").str.starts_with("M").cast(pl.Int8).alias("is_merchant_destination"),
    )


def prepare_main_experiment_features(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """Create derived features and remove columns excluded from the main study."""

    featured = add_features(frame)
    # ``strict=False`` keeps this helper usable on already-sanitised frames while
    # the ingestion schema validator remains responsible for raw-data integrity.
    return featured.drop(MAIN_EXPERIMENT_EXCLUDED_COLUMNS, strict=False)
