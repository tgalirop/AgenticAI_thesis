"""Tests for fair, shared-fold model benchmarking."""

import pandas as pd

from agenticai_thesis.modeling.benchmark import benchmark_models


def test_benchmark_uses_all_shared_folds_for_each_model() -> None:
    """Every model must produce one metric row for every identical CV split."""

    rows = 40
    sample = pd.DataFrame(
        {
            "type": ["PAYMENT", "TRANSFER"] * (rows // 2),
            "step": list(range(1, rows + 1)),
            "amount": [float(value + 1) for value in range(rows)],
            "hour": [value % 24 for value in range(rows)],
            "day": [value // 24 for value in range(rows)],
            "log_amount": [float(value) / 10 for value in range(rows)],
            "is_transfer": [0, 1] * (rows // 2),
            "is_cash_out": [0] * rows,
            "is_merchant_destination": [1, 0] * (rows // 2),
            "isFraud": [0, 1] * (rows // 2),
        }
    )
    result = benchmark_models(
        sample,
        model_names=["logistic_regression", "decision_tree"],
        model_parameters={"logistic_regression": {"max_iter": 100}},
        target_column="isFraud",
        folds=2,
        repeats=2,
        random_seed=42,
        threshold=0.5,
    )

    counts = result.fold_metrics.groupby("model").size().to_dict()
    assert counts == {"decision_tree": 4, "logistic_regression": 4}
    # Each of the 40 observations appears once per repeat and per model.
    prediction_counts = result.predictions.groupby("model").size().to_dict()
    assert prediction_counts == {"decision_tree": 80, "logistic_regression": 80}
