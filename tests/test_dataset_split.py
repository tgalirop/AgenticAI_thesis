"""Tests for the chronological training/test dataset split."""

import polars as pl
import pytest

from agenticai_thesis.data.dataset_split import determine_test_cutoff, split_chronologically


def test_split_keeps_complete_time_steps_and_future_in_test() -> None:
    """No time step may appear on both sides of the test boundary."""

    frame = pl.DataFrame(
        {
            "step": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "isFraud": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    training, test_dataset = split_chronologically(frame, test_fraction=0.40)

    assert training.get_column("step").unique().sort().to_list() == [1, 2, 3]
    assert test_dataset.get_column("step").unique().sort().to_list() == [4, 5]
    assert training.get_column("step").max() < test_dataset.get_column("step").min()
    assert training.height + test_dataset.height == frame.height


def test_cutoff_uses_distinct_steps_not_row_fraction() -> None:
    """A heavily populated hour must not move the chronological boundary."""

    frame = pl.DataFrame({"step": [1] * 100 + [2, 3, 4, 5]})
    assert determine_test_cutoff(frame, "step", 0.20) == 5


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_test_fraction_is_rejected(fraction: float) -> None:
    frame = pl.DataFrame({"step": [1, 2]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        determine_test_cutoff(frame, "step", fraction)

