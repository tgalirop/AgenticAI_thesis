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
    result = PairedPipelineComparator().compare(_results(0.0), _results(0.02))
    assert set(result["metric"]) == {"pr_auc", "recall", "precision", "f1"}
    assert (result["pairs"] == 5).all()
    assert np.allclose(result["mean_delta"], 0.02)
    assert result["p_value_holm"].between(0.0, 1.0).all()


def test_comparator_rejects_unpaired_fold_sets() -> None:
    agentic = _results(0.01).iloc[:-1]
    with pytest.raises(ValueError, match="same model/fold"):
        PairedPipelineComparator().compare(_results(0.0), agentic)
