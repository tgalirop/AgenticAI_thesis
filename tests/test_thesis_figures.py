"""Tests for reproducible thesis-figure generation."""

import json

import pandas as pd

from agenticai_thesis.reporting import ThesisFigureGenerator


def test_generator_creates_all_five_nonempty_figures(tmp_path) -> None:
    metrics = pd.DataFrame(
        [
            {"pipeline": pipeline, "model": model, "pr_auc": 0.2, "precision": 0.1, "recall": 0.8}
            for model in ThesisFigureGenerator.MODEL_LABELS
            for pipeline in ThesisFigureGenerator.PIPELINE_LABELS
        ]
    )
    metrics_path = tmp_path / "metrics.csv"
    summary_path = tmp_path / "summary.json"
    metrics.to_csv(metrics_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "initial_data_quality_score": 0.998,
                "selected_data_quality_score": 0.999,
                "selected_quality_delta": 0.001,
                "selected_runtime_multiplier": 1.2,
            }
        ),
        encoding="utf-8",
    )

    outputs = ThesisFigureGenerator(metrics_path, summary_path, tmp_path / "figures").generate_all()

    assert len(outputs) == 5
    assert all(path.is_file() and path.stat().st_size > 1_000 for path in outputs)
