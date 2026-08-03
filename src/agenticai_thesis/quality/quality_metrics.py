"""Object-oriented evaluation of the thesis data-quality dimensions.

The evaluator consumes the deterministic profiler report rather than rescanning
the dataset. This keeps the quality calculation cheap, testable and reusable by
both the conventional experiment and future LangGraph Agent nodes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from agenticai_thesis.config import PROJECT_ROOT
from agenticai_thesis.utils.file_io import write_json_atomic


DEFAULT_PROFILE_PATH = PROJECT_ROOT / "reports/profiles/development_profile.json"
DEFAULT_QUALITY_PATH = PROJECT_ROOT / "reports/profiles/development_quality.json"


@dataclass(frozen=True)
class DataQualityWeights:
    """Fixed weights used by the composite Data Quality Score."""

    completeness: float = 0.30
    validity: float = 0.25
    consistency: float = 0.25
    uniqueness: float = 0.20

    def __post_init__(self) -> None:
        """Reject negative weights and totals that would distort the score."""

        values = asdict(self).values()
        if any(value < 0.0 for value in values):
            raise ValueError("Data-quality weights cannot be negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Data-quality weights must sum to 1.0")


@dataclass(frozen=True)
class DataQualityResult:
    """Immutable and serializable output of one quality evaluation."""

    completeness: float
    validity: float
    consistency: float
    uniqueness: float
    data_quality_score: float
    total_cells: int
    missing_cells: int
    validity_checks: int
    validity_failures: int
    consistency_checks: int
    consistency_failures: int
    duplicate_rows: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a plain mapping suitable for JSON and Agent state storage."""

        return asdict(self)


class DataQualityEvaluator:
    """Calculate transparent quality scores from a profiler report.

    The class is stateless apart from its injected weights, making the same
    evaluator safe to reuse across conventional, degraded and Agentic datasets.
    """

    def __init__(self, weights: DataQualityWeights | None = None) -> None:
        self._weights = weights or DataQualityWeights()

    @property
    def weights(self) -> DataQualityWeights:
        """Expose the immutable weights for experiment logging and auditing."""

        return self._weights

    @staticmethod
    def _bounded_score(failures: int, opportunities: int) -> float:
        """Convert a failure count to a score in [0, 1].

        An absent denominator is treated as a perfect score only when there are no
        failures. This supports narrow test fixtures while preventing impossible
        failure counts from being silently accepted.
        """

        if failures < 0 or opportunities < 0:
            raise ValueError("Quality counts cannot be negative")
        if opportunities == 0:
            if failures:
                raise ValueError("Failures cannot exist without evaluation opportunities")
            return 1.0
        if failures > opportunities:
            raise ValueError("Failures cannot exceed evaluation opportunities")
        return 1.0 - failures / opportunities

    def evaluate(self, report: Mapping[str, Any]) -> DataQualityResult:
        """Validate a profiler report and calculate all four quality dimensions."""

        try:
            rows = int(report["dimensions"]["rows"])
            columns = int(report["dimensions"]["columns"])
            columns_report = report["columns"]
            invalid_section = report["invalid_values"]
            consistency_section = report["consistency"]
            duplicate_rows = int(report["duplicates"]["duplicate_rows"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Profiler report does not match the required schema") from error

        if rows < 0 or columns < 0:
            raise ValueError("Dataset dimensions cannot be negative")
        total_cells = rows * columns
        missing_cells = sum(int(details["null_count"]) for details in columns_report.values())

        invalid_checks = invalid_section.get("checks", {})
        validity_failures = int(invalid_section["total_failures"])
        # Every named validity rule evaluates at most one value per row.
        validity_opportunities = rows * len(invalid_checks)

        consistency_checks_map = consistency_section.get("checks", {})
        consistency_failures = int(consistency_section["total_failures"])
        consistency_opportunities = rows * len(consistency_checks_map)

        completeness = self._bounded_score(missing_cells, total_cells)
        validity = self._bounded_score(validity_failures, validity_opportunities)
        consistency = self._bounded_score(consistency_failures, consistency_opportunities)
        uniqueness = self._bounded_score(duplicate_rows, rows)
        weighted_score = (
            self._weights.completeness * completeness
            + self._weights.validity * validity
            + self._weights.consistency * consistency
            + self._weights.uniqueness * uniqueness
        )

        return DataQualityResult(
            completeness=completeness,
            validity=validity,
            consistency=consistency,
            uniqueness=uniqueness,
            data_quality_score=weighted_score,
            total_cells=total_cells,
            missing_cells=missing_cells,
            validity_checks=len(invalid_checks),
            validity_failures=validity_failures,
            consistency_checks=len(consistency_checks_map),
            consistency_failures=consistency_failures,
            duplicate_rows=duplicate_rows,
        )

    def evaluate_file(self, report_path: str | Path) -> DataQualityResult:
        """Load a UTF-8 JSON profiler report and evaluate it."""

        path = Path(report_path)
        if not path.is_file():
            raise FileNotFoundError(f"Profiler report not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        return self.evaluate(report)


def main() -> None:
    """Evaluate the development profile and persist Agent-ready quality metrics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=str(DEFAULT_PROFILE_PATH.relative_to(PROJECT_ROOT)),
        help="Profiler JSON path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_QUALITY_PATH.relative_to(PROJECT_ROOT)),
        help="Destination JSON path, relative to the repository root by default.",
    )
    args = parser.parse_args()
    profile_path = Path(args.profile)
    output_path = Path(args.output)
    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    evaluator = DataQualityEvaluator()
    result = evaluator.evaluate_file(profile_path)
    write_json_atomic(
        {
            "metric_version": "1.0",
            "source_profile": str(profile_path.relative_to(PROJECT_ROOT)),
            "weights": asdict(evaluator.weights),
            "result": result.to_dict(),
        },
        output_path,
    )
    print(f"Data-quality metrics created: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
