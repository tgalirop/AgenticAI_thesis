"""Load and validate PaySim datasets efficiently with Polars."""

from __future__ import annotations

from pathlib import Path

import polars as pl


# These columns define the source-data contract.  Checking them at ingestion
# catches accidental use of another CSV or a damaged export before a long model
# run begins.  Extra columns are allowed so future derived datasets remain usable.
PAYSIM_REQUIRED_COLUMNS = frozenset(
    {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }
)


def scan_dataset(path: str | Path) -> pl.LazyFrame:
    """Lazily scan a CSV or Parquet dataset without loading it into memory.

    Lazy scanning is important for PaySim: query optimisation and streaming keep
    memory use bounded while processing more than six million rows.
    """

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(dataset_path, try_parse_dates=False)
    if suffix in {".parquet", ".pq"}:
        return pl.scan_parquet(dataset_path)
    raise ValueError(f"Unsupported dataset format '{suffix}': {dataset_path}")


def validate_paysim_schema(frame: pl.DataFrame | pl.LazyFrame) -> None:
    """Raise a descriptive error if required PaySim columns are absent."""

    # ``collect_schema`` reads metadata only for a LazyFrame; it does not collect
    # all 6.36 million observations.
    columns = set(frame.collect_schema().names()) if isinstance(frame, pl.LazyFrame) else set(frame.columns)
    missing = sorted(PAYSIM_REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"PaySim dataset is missing columns: {', '.join(missing)}")


def load_dataset(path: str | Path) -> pl.DataFrame:
    """Load a complete supported dataset after validating its source schema."""

    lazy_frame = scan_dataset(path)
    validate_paysim_schema(lazy_frame)
    return lazy_frame.collect(engine="streaming")
