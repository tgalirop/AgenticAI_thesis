"""Build the processed PaySim dataset and leakage-safe chronological partitions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from agenticai_thesis.config import PROJECT_ROOT, load_data_config
from agenticai_thesis.data.convert_to_parquet import convert_csv_to_parquet
from agenticai_thesis.data.dataset_split import create_dataset_splits


def _dataset_summary(path: Path) -> dict[str, int]:
    """Collect only the small set of statistics printed after a pipeline run."""

    # Lazy aggregation lets Polars use Parquet metadata and avoids loading every
    # column into memory merely to verify the generated artifact.
    return (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"),
            pl.col("step").min().alias("min_step"),
            pl.col("step").max().alias("max_step"),
            pl.col("isFraud").sum().alias("fraud_rows"),
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )


def main() -> None:
    """Execute the complete Phase-1 dataset preparation workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_data_config(args.config)
    started = time.perf_counter()

    print("1/2 Converting raw PaySim CSV to Parquet...")
    convert_csv_to_parquet(config.raw_csv_path, config.parquet_path)

    print("2/2 Creating leakage-safe chronological partitions...")
    training, test_dataset = create_dataset_splits(
        config.parquet_path,
        config.training_path,
        config.test_dataset_path,
        time_column=config.time_column,
        test_fraction=config.test_dataset_fraction,
    )

    for label, path in (("training", training), ("test_dataset", test_dataset)):
        print(f"{label}: {path.relative_to(PROJECT_ROOT)} {_dataset_summary(path)}")
    print(f"Pipeline completed in {time.perf_counter() - started:.2f} seconds.")


if __name__ == "__main__":
    main()
