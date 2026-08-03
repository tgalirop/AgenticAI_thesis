"""Generate a structured Data Quality Report without using an LLM.

The profiler is deliberately deterministic.  Its output becomes the compact,
auditable input that the Agent may inspect in Phase 2; the raw dataset itself is
never sent to a language model.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import psutil

from agenticai_thesis.config import PROJECT_ROOT, load_data_config
from agenticai_thesis.data.load import scan_dataset
from agenticai_thesis.utils.file_io import write_json_atomic


DEFAULT_PROFILE_PATH = PROJECT_ROOT / "reports/profiles/development_profile.json"
EXPECTED_TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")

# Polars types are grouped explicitly so the report logic remains readable and
# does not accidentally apply skewness or quantiles to identifiers/strings.
NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def _json_number(value: Any) -> int | float | None:
    """Convert Polars scalar values into strict, portable JSON numbers."""

    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        # JSON NaN/Infinity values are non-standard and break many downstream
        # validators, so undefined statistics are represented as null.
        return None
    if isinstance(value, (int, float)):
        return value
    return float(value)


def _column_aggregation_expressions(
    schema: pl.Schema,
) -> list[pl.Expr]:
    """Build one aggregation query for missingness, cardinality and statistics."""

    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        expressions.extend(
            [
                pl.col(name).null_count().alias(f"{name}__null_count"),
                pl.col(name).n_unique().alias(f"{name}__n_unique"),
            ]
        )
        if dtype in NUMERIC_DTYPES:
            expressions.extend(
                [
                    pl.col(name).min().alias(f"{name}__min"),
                    pl.col(name).max().alias(f"{name}__max"),
                    pl.col(name).mean().alias(f"{name}__mean"),
                    pl.col(name).median().alias(f"{name}__median"),
                    pl.col(name).std().alias(f"{name}__std"),
                    pl.col(name).skew().alias(f"{name}__skewness"),
                ]
            )
    return expressions


def _invalid_value_expressions(schema_names: Iterable[str]) -> dict[str, pl.Expr]:
    """Define PaySim domain checks only for columns that are present.

    These are transparent, research-defined checks—not learned rules.  Keeping
    their names in the report makes it possible to distinguish each failure type.
    """

    columns = set(schema_names)
    checks: dict[str, pl.Expr] = {}
    if "step" in columns:
        checks["step_non_positive"] = (pl.col("step") <= 0).sum()
    if "amount" in columns:
        checks["amount_negative"] = (pl.col("amount") < 0).sum()
    if "type" in columns:
        checks["unknown_transaction_type"] = (~pl.col("type").is_in(EXPECTED_TRANSACTION_TYPES)).sum()
    if "isFraud" in columns:
        checks["isFraud_not_binary"] = (~pl.col("isFraud").is_in([0, 1])).sum()
    if "hour" in columns:
        checks["hour_outside_0_23"] = (~pl.col("hour").is_between(0, 23)).sum()
    if "day" in columns:
        checks["day_negative"] = (pl.col("day") < 0).sum()
    for binary_feature in ("is_transfer", "is_cash_out", "is_merchant_destination"):
        if binary_feature in columns:
            checks[f"{binary_feature}_not_binary"] = (~pl.col(binary_feature).is_in([0, 1])).sum()
    return checks


def _consistency_expressions(schema_names: Iterable[str]) -> dict[str, pl.Expr]:
    """Define cross-column rules that must hold between derived features.

    Validity asks whether one value belongs to its permitted domain. Consistency
    asks whether two or more individually valid values agree with each other. The
    distinction is preserved explicitly because the thesis evaluates both quality
    dimensions independently.
    """

    columns = set(schema_names)
    checks: dict[str, pl.Expr] = {}
    if {"step", "hour"}.issubset(columns):
        expected_hour = ((pl.col("step") - 1) % 24).cast(pl.Int16)
        checks["hour_inconsistent_with_step"] = (pl.col("hour") != expected_hour).sum()
    if {"step", "day"}.issubset(columns):
        expected_day = ((pl.col("step") - 1) // 24).cast(pl.Int16)
        checks["day_inconsistent_with_step"] = (pl.col("day") != expected_day).sum()
    if {"amount", "log_amount"}.issubset(columns):
        # A small tolerance avoids treating harmless floating-point rounding as a
        # genuine inconsistency between amount and its log1p transformation.
        checks["log_amount_inconsistent_with_amount"] = (
            (pl.col("log_amount") - pl.col("amount").log1p()).abs() > 1e-10
        ).sum()
    if {"type", "is_transfer"}.issubset(columns):
        checks["is_transfer_inconsistent_with_type"] = (
            pl.col("is_transfer") != (pl.col("type") == "TRANSFER").cast(pl.Int8)
        ).sum()
    if {"type", "is_cash_out"}.issubset(columns):
        checks["is_cash_out_inconsistent_with_type"] = (
            pl.col("is_cash_out") != (pl.col("type") == "CASH_OUT").cast(pl.Int8)
        ).sum()
    return checks


def _value_distribution(
    lazy_frame: pl.LazyFrame,
    column: str,
    row_count: int,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic counts and rates for a categorical column."""

    distribution = (
        lazy_frame.group_by(column)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    if limit is not None:
        distribution = distribution.head(limit)

    rows = distribution.collect(engine="streaming").to_dicts()
    for row in rows:
        row["rate"] = row["count"] / row_count if row_count else None
    return rows


def profile_dataset(
    dataset_path: str | Path,
    *,
    target_column: str = "isFraud",
    time_column: str = "step",
) -> dict[str, Any]:
    """Profile a Parquet/CSV dataset and return a JSON-serializable report.

    The function accepts arbitrary paths for unit testing and reuse.  The public
    CLI below intentionally supplies only the configured development path, which
    protects the temporal test set from accidental exploratory inspection.
    """

    source_path = Path(dataset_path)
    started_at = datetime.now(UTC)
    timer_started = time.perf_counter()
    process = psutil.Process()
    rss_before = process.memory_info().rss

    lazy_frame = scan_dataset(source_path)
    schema = lazy_frame.collect_schema()
    schema_names = schema.names()
    if target_column not in schema_names:
        raise ValueError(f"Target column not found: {target_column}")
    if time_column not in schema_names:
        raise ValueError(f"Time column not found: {time_column}")

    # A single aggregate query calculates the row count, full-row uniqueness and
    # all per-column statistics.  This minimizes repeated scans of the 6.2M-row
    # development file.
    aggregate_expressions = [
        pl.len().alias("__rows"),
        pl.struct(pl.all()).n_unique().alias("__unique_rows"),
        *_column_aggregation_expressions(schema),
    ]
    invalid_checks = _invalid_value_expressions(schema_names)
    aggregate_expressions.extend(expression.alias(name) for name, expression in invalid_checks.items())
    consistency_checks = _consistency_expressions(schema_names)
    aggregate_expressions.extend(
        expression.alias(name) for name, expression in consistency_checks.items()
    )
    aggregate = lazy_frame.select(aggregate_expressions).collect(engine="streaming").to_dicts()[0]

    row_count = int(aggregate["__rows"])
    unique_rows = int(aggregate["__unique_rows"])
    columns_report: dict[str, dict[str, Any]] = {}
    for name, dtype in schema.items():
        null_count = int(aggregate[f"{name}__null_count"])
        column_report: dict[str, Any] = {
            "dtype": str(dtype),
            "null_count": null_count,
            "null_rate": null_count / row_count if row_count else None,
            "non_null_count": row_count - null_count,
            "n_unique": int(aggregate[f"{name}__n_unique"]),
        }
        if dtype in NUMERIC_DTYPES:
            column_report["statistics"] = {
                statistic: _json_number(aggregate[f"{name}__{statistic}"])
                for statistic in ("min", "max", "mean", "median", "std", "skewness")
            }
        columns_report[name] = column_report

    class_distribution = _value_distribution(lazy_frame, target_column, row_count)
    categorical_distributions = {
        name: _value_distribution(lazy_frame, name, row_count, limit=20)
        for name, dtype in schema.items()
        if dtype == pl.String
    }

    invalid_values = {name: int(aggregate[name]) for name in invalid_checks}
    consistency_values = {name: int(aggregate[name]) for name in consistency_checks}
    rss_after = process.memory_info().rss
    elapsed = time.perf_counter() - timer_started

    # Class imbalance is a faithful property of fraud data.  It is surfaced as a
    # modeling risk rather than incorrectly counted as invalid data.
    minority_rate = min((entry["rate"] for entry in class_distribution), default=None)
    modeling_risks = {
        "class_imbalance": {
            "detected": minority_rate is not None and minority_rate < 0.01,
            "minority_class_rate": minority_rate,
            "interpretation": "Modeling risk; not automatically a data-quality error.",
        }
    }

    return {
        "report_version": "1.0",
        "generated_at_utc": started_at.isoformat(),
        "dataset": {
            "name": source_path.name,
            "path": str(source_path),
            "file_size_bytes": source_path.stat().st_size,
            "format": source_path.suffix.lower().lstrip("."),
        },
        "dimensions": {"rows": row_count, "columns": len(schema)},
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "duplicates": {
            "duplicate_rows": row_count - unique_rows,
            "duplicate_rate": (row_count - unique_rows) / row_count if row_count else None,
            "scope": "Exact duplicates across the columns present in the profiled dataset.",
            "interpretation": (
                "Because direct account identifiers and balance columns are excluded from the "
                "main modeling dataset, equal feature rows are not necessarily duplicate source transactions."
            ),
        },
        "columns": columns_report,
        "class_distribution": {"column": target_column, "values": class_distribution},
        "categorical_distributions": categorical_distributions,
        "invalid_values": {
            "checks": invalid_values,
            "total_failures": sum(invalid_values.values()),
        },
        "consistency": {
            "checks": consistency_values,
            "total_failures": sum(consistency_values.values()),
        },
        "time_range": {
            "column": time_column,
            "min": columns_report[time_column].get("statistics", {}).get("min"),
            "max": columns_report[time_column].get("statistics", {}).get("max"),
        },
        "modeling_risks": modeling_risks,
        "resource_usage": {
            "execution_time_seconds": elapsed,
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
            "process_rss_change_bytes": rss_after - rss_before,
        },
    }


def generate_profile(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    target_column: str = "isFraud",
    time_column: str = "step",
) -> Path:
    """Profile a dataset and atomically persist its strict JSON report."""

    report = profile_dataset(dataset_path, target_column=target_column, time_column=time_column)
    return write_json_atomic(report, output_path)


def main() -> None:
    """Profile only the configured development set and write its JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default=str(DEFAULT_PROFILE_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()
    config = load_data_config(args.config)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output = generate_profile(
        config.development_path,
        output_path,
        target_column=config.target_column,
        time_column=config.time_column,
    )
    print(f"Development profile created: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
