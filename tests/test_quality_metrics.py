"""Tests for the object-oriented Data Quality Evaluator."""

import json
from pathlib import Path

import pytest

from agenticai_thesis.quality.quality_metrics import (
    DataQualityEvaluator,
    DataQualityWeights,
)


def _quality_report() -> dict:
    """Return a small report whose expected dimension scores are exact."""

    return {
        "dimensions": {"rows": 10, "columns": 2},
        "columns": {
            "a": {"null_count": 1},
            "b": {"null_count": 1},
        },
        "invalid_values": {
            "checks": {"rule_a": 1, "rule_b": 0},
            "total_failures": 1,
        },
        "consistency": {
            "checks": {"relationship_a_b": 2},
            "total_failures": 2,
        },
        "duplicates": {"duplicate_rows": 1},
    }


def test_evaluator_calculates_dimensions_and_weighted_score() -> None:
    evaluator = DataQualityEvaluator()
    result = evaluator.evaluate(_quality_report())

    assert result.completeness == pytest.approx(0.90)
    assert result.validity == pytest.approx(0.95)
    assert result.consistency == pytest.approx(0.80)
    assert result.uniqueness == pytest.approx(0.90)
    assert result.data_quality_score == pytest.approx(0.8875)
    assert result.to_dict()["missing_cells"] == 2


def test_custom_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        DataQualityWeights(0.30, 0.25, 0.25, 0.25)


def test_evaluator_reads_json_report_from_file(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_quality_report()), encoding="utf-8")
    result = DataQualityEvaluator().evaluate_file(path)
    assert result.duplicate_rows == 1


def test_impossible_failure_counts_are_rejected() -> None:
    report = _quality_report()
    report["duplicates"]["duplicate_rows"] = 11
    with pytest.raises(ValueError, match="exceed"):
        DataQualityEvaluator().evaluate(report)
