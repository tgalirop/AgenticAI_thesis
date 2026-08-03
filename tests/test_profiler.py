"""Tests for the standalone data profiler."""

import json
import math
from pathlib import Path

import polars as pl

from agenticai_thesis.quality.profiler import generate_profile, profile_dataset


def _write_profile_fixture(path: Path) -> Path:
    """Create a tiny dataset containing known quality defects."""

    frame = pl.DataFrame(
        {
            "step": [1, 2, 2, 2],
            "type": ["PAYMENT", "TRANSFER", "TRANSFER", "UNKNOWN"],
            "amount": [10.0, 100.0, 100.0, -5.0],
            "isFraud": [0, 1, 1, 3],
            "hour": [0, 1, 1, 25],
            "day": [0, 0, 0, -1],
            "log_amount": [math.log1p(10.0), math.log1p(100.0), math.log1p(100.0), None],
            "is_transfer": [0, 1, 1, 0],
            "is_cash_out": [0, 0, 0, 2],
            "is_merchant_destination": [1, 0, 0, 0],
        }
    )
    frame.write_parquet(path)
    return path


def test_profile_reports_dimensions_missingness_duplicates_and_classes(tmp_path: Path) -> None:
    dataset = _write_profile_fixture(tmp_path / "development.parquet")
    report = profile_dataset(dataset)

    assert report["dimensions"] == {"rows": 4, "columns": 10}
    assert report["duplicates"]["duplicate_rows"] == 1
    assert report["duplicates"]["duplicate_rate"] == 0.25
    assert report["columns"]["log_amount"]["null_count"] == 1
    assert report["columns"]["log_amount"]["null_rate"] == 0.25
    assert report["time_range"] == {"column": "step", "min": 1, "max": 2}

    class_counts = {
        entry["isFraud"]: entry["count"]
        for entry in report["class_distribution"]["values"]
    }
    assert class_counts == {0: 1, 1: 2, 3: 1}


def test_profile_reports_named_domain_validation_failures(tmp_path: Path) -> None:
    dataset = _write_profile_fixture(tmp_path / "development.parquet")
    report = profile_dataset(dataset)
    checks = report["invalid_values"]["checks"]

    assert checks["amount_negative"] == 1
    assert checks["unknown_transaction_type"] == 1
    assert checks["isFraud_not_binary"] == 1
    assert checks["hour_outside_0_23"] == 1
    assert checks["day_negative"] == 1
    assert checks["is_cash_out_not_binary"] == 1
    assert report["invalid_values"]["total_failures"] == 6
    # The invalid binary flag is also inconsistent with its source transaction
    # type, demonstrating the distinction between validity and consistency.
    assert report["consistency"]["checks"]["is_cash_out_inconsistent_with_type"] == 1
    # The last row also carries an invalid hour and day relative to step=2.
    assert report["consistency"]["total_failures"] == 3


def test_generated_report_is_strict_json_and_matches_returned_schema(tmp_path: Path) -> None:
    dataset = _write_profile_fixture(tmp_path / "development.parquet")
    output = tmp_path / "reports" / "profile.json"

    returned_path = generate_profile(dataset, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert returned_path == output
    assert loaded["report_version"] == "1.0"
    assert loaded["dataset"]["name"] == "development.parquet"
    assert loaded["resource_usage"]["execution_time_seconds"] >= 0
