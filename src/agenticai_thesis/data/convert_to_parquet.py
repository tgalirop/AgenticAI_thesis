"""Convert the raw PaySim CSV dataset to compressed Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from agenticai_thesis.config import PROJECT_ROOT, load_data_config
from agenticai_thesis.data.load import scan_dataset, validate_paysim_schema


def convert_csv_to_parquet(csv_path: str | Path, parquet_path: str | Path) -> Path:
    """Convert PaySim to Parquet using Polars' streaming sink.

    Parquet is smaller and faster than CSV for repeated profiling and experiments.
    The streaming sink avoids materialising the full source dataset in RAM.
    """

    source = Path(csv_path)
    destination = Path(parquet_path)
    if source.resolve() == destination.resolve():
        raise ValueError("CSV source and Parquet destination must be different")

    lazy_frame = scan_dataset(source)
    validate_paysim_schema(lazy_frame)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Zstandard gives a useful storage reduction while retaining fast reads.
    # Statistics enable predicate pushdown in later temporal/filtering queries.
    lazy_frame.sink_parquet(
        destination,
        compression="zstd",
        compression_level=3,
        statistics=True,
        maintain_order=True,
    )
    return destination


def main() -> None:
    """Run the CSV-to-Parquet conversion from the configured project paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_data_config(args.config)
    output = convert_csv_to_parquet(config.raw_csv_path, config.parquet_path)
    # Print a repository-relative path.  Besides being easier to read, this avoids
    # legacy Windows terminals failing on Greek characters in the absolute path.
    print(f"Parquet dataset created: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
