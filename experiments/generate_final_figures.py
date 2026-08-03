"""Generate all final thesis figures from checked experiment artifacts."""

from agenticai_thesis.config import PROJECT_ROOT
from agenticai_thesis.reporting import ThesisFigureGenerator


def main() -> None:
    """Create reproducible, publication-ready PNG files."""

    outputs = ThesisFigureGenerator(
        temporal_metrics_path=PROJECT_ROOT / "reports/metrics/temporal_holdout_results.csv",
        run_summary_path=PROJECT_ROOT / "logs/agentic/run_summary.json",
        output_dir=PROJECT_ROOT / "figures/final_comparisons",
    ).generate_all()
    for output in outputs:
        print(output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
