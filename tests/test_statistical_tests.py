"""Tests for paired conventional-versus-Agentic statistical comparisons."""

import pandas as pd
import numpy as np
import pytest

from agenticai_thesis.modeling.statistical_tests import PairedPipelineComparator


def _results(offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "random_forest",
                "repeat": 1,
                "fold": fold,
                "pr_auc": 0.70 + fold / 100 + offset,
                "recall": 0.60 + offset,
                "precision": 0.65 + offset,
                "f1": 0.62 + offset,
            }
            for fold in range(1, 6)
        ]
    )


def test_comparator_returns_paired_deltas_and_corrected_p_values() -> None:
    result = PairedPipelineComparator(test_train_ratio=0.25).compare(
        _results(0.0), _results(0.02)
    )
    assert set(result["metric"]) == {"pr_auc", "recall", "precision", "f1"}
    assert (result["pairs"] == 5).all()
    assert np.allclose(result["mean_delta"], 0.02)
    assert result["p_value_holm"].between(0.0, 1.0).all()
    assert result["nb_p_value_holm"].between(0.0, 1.0).all()


def test_comparator_rejects_unpaired_fold_sets() -> None:
    agentic = _results(0.01).iloc[:-1]
    with pytest.raises(ValueError, match="same model/fold"):
        PairedPipelineComparator(test_train_ratio=0.25).compare(_results(0.0), agentic)


def _noisy_results(deltas: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [{"model": "random_forest", "repeat": r, "fold": f} for r in range(5) for f in range(5)]
    reference = pd.DataFrame(
        [
            {**key, "pr_auc": 0.70, "recall": 0.6, "precision": 0.6, "f1": 0.6,
             "train_rows": 400, "validation_rows": 100}
            for key in keys
        ]
    )
    candidate = reference.copy()
    candidate["pr_auc"] = candidate["pr_auc"] + deltas
    return reference, candidate


def test_nadeau_bengio_matches_formula_and_is_more_conservative_than_naive_t() -> None:
    rng = np.random.default_rng(3)
    deltas = 0.004 + rng.normal(0.0, 0.01, size=25)
    reference, candidate = _noisy_results(deltas)

    row = PairedPipelineComparator().compare(reference, candidate).set_index("metric").loc["pr_auc"]

    # Ratio derived from recorded fold sizes: 100 / 400.
    expected_t = deltas.mean() / np.sqrt((1 / 25 + 0.25) * deltas.var(ddof=1))
    naive_t = deltas.mean() / np.sqrt(deltas.var(ddof=1) / 25)
    assert row["nb_t_statistic"] == pytest.approx(expected_t)
    assert abs(row["nb_t_statistic"]) < abs(naive_t)


def test_nadeau_bengio_requires_fold_sizes_or_explicit_ratio() -> None:
    with pytest.raises(ValueError, match="test_train_ratio"):
        PairedPipelineComparator().compare(_results(0.0), _results(0.02))
