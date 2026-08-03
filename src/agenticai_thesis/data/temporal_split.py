"""Create development and untouched temporal test datasets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import polars as pl

from agenticai_thesis.config import PROJECT_ROOT, load_data_config
from agenticai_thesis.data.features import prepare_main_experiment_features
from agenticai_thesis.data.load import scan_dataset, validate_paysim_schema


def determine_temporal_cutoff(
    frame: pl.DataFrame | pl.LazyFrame,
    time_column: str,
    test_fraction: float,
) -> int | float:
    """Return the first time value assigned to the temporal test partition.

    The split is based on distinct time steps rather than row positions.  Thus all
    transactions from one hour stay in the same partition and no future hour can
    leak into development.  ``ceil`` ensures the test partition is never smaller
    than the requested fraction in terms of represented time steps.
    """

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema_names = lazy.collect_schema().names()
    if time_column not in schema_names:
        raise ValueError(f"Time column not found: {time_column}")

    time_steps = (
        lazy.select(pl.col(time_column).drop_nulls().unique().sort())
        .collect(engine="streaming")
        .get_column(time_column)
        .to_list()
    )
    if len(time_steps) < 2:
        raise ValueError("At least two distinct time values are required for a temporal split")

    test_steps = math.ceil(len(time_steps) * test_fraction)
    # Both partitions must contain at least one complete time step.
    test_steps = min(max(test_steps, 1), len(time_steps) - 1)
    return time_steps[-test_steps]


def split_temporally(
    frame: pl.DataFrame,
    time_column: str = "step",
    test_fraction: float = 0.20,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split an in-memory frame chronologically; primarily used by tests."""

    cutoff = determine_temporal_cutoff(frame, time_column, test_fraction)
    development = frame.filter(pl.col(time_column) < cutoff)
    temporal_test = frame.filter(pl.col(time_column) >= cutoff)
    return development, temporal_test


def create_temporal_splits(
    parquet_path: str | Path,
    development_path: str | Path,
    temporal_test_path: str | Path,
    *,
    time_column: str = "step",
    test_fraction: float = 0.20,
) -> tuple[Path, Path]:
    """Feature-engineer PaySim and stream both chronological splits to Parquet."""

    source = scan_dataset(parquet_path)
    validate_paysim_schema(source)
    cutoff = determine_temporal_cutoff(source, time_column, test_fraction)
    prepared = prepare_main_experiment_features(source)

    development = Path(development_path)
    temporal_test = Path(temporal_test_path)
    development.parent.mkdir(parents=True, exist_ok=True)
    temporal_test.parent.mkdir(parents=True, exist_ok=True)

    # Identical feature logic is applied before both filters, but each output is
    # written independently.  Neither profiling nor model selection needs to read
    # the temporal holdout after this point.
    prepared.filter(pl.col(time_column) < cutoff).sink_parquet(
        development, compression="zstd", compression_level=3, statistics=True
    )
    prepared.filter(pl.col(time_column) >= cutoff).sink_parquet(
        temporal_test, compression="zstd", compression_level=3, statistics=True
    )
    return development, temporal_test


def main() -> None:
    """Create configured development and temporal-test Parquet datasets."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_data_config(args.config)
    development, temporal_test = create_temporal_splits(
        config.parquet_path,
        config.development_path,
        config.temporal_test_path,
        time_column=config.time_column,
        test_fraction=config.temporal_test_fraction,
    )
    # Repository-relative output remains readable in Windows terminals whose
    # legacy encoding cannot represent the Greek characters in the parent path.
    print(f"Development dataset created: {development.relative_to(PROJECT_ROOT)}")
    print(f"Temporal test dataset created: {temporal_test.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
