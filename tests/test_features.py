"""Tests for leakage-safe feature engineering."""

import math

import polars as pl

from agenticai_thesis.data.features import prepare_main_experiment_features


def test_features_have_expected_values_and_exclusions() -> None:
    """Derived values must be correct and leakage-prone columns must be removed."""

    source = pl.DataFrame(
        {
            "step": [1, 25],
            "type": ["TRANSFER", "CASH_OUT"],
            "amount": [0.0, 99.0],
            "nameOrig": ["C1", "C2"],
            "oldbalanceOrg": [10.0, 20.0],
            "newbalanceOrig": [10.0, 0.0],
            "nameDest": ["M1", "C3"],
            "oldbalanceDest": [0.0, 5.0],
            "newbalanceDest": [0.0, 104.0],
            "isFraud": [0, 1],
            "isFlaggedFraud": [0, 0],
        }
    )

    result = prepare_main_experiment_features(source)

    assert result.get_column("hour").to_list() == [0, 0]
    assert result.get_column("day").to_list() == [0, 1]
    assert result.get_column("is_transfer").to_list() == [1, 0]
    assert result.get_column("is_cash_out").to_list() == [0, 1]
    assert result.get_column("is_merchant_destination").to_list() == [1, 0]
    assert math.isclose(result[1, "log_amount"], math.log1p(99.0))

    excluded = {
        "isFlaggedFraud",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "nameOrig",
        "nameDest",
    }
    assert excluded.isdisjoint(result.columns)
    # The target is retained for supervised model training and evaluation.
    assert "isFraud" in result.columns


def test_feature_engineering_supports_lazy_frames() -> None:
    """The production pipeline must remain lazy until it writes Parquet."""

    source = pl.DataFrame(
        {"step": [1], "type": ["PAYMENT"], "amount": [10.0], "nameDest": ["M1"]}
    ).lazy()
    result = prepare_main_experiment_features(source).collect()
    assert result[0, "is_merchant_destination"] == 1

