"""Tests for the chronological development/holdout split."""

import polars as pl
import pytest

from agenticai_thesis.data.temporal_split import determine_temporal_cutoff, split_temporally


def test_split_keeps_complete_time_steps_and_future_in_test() -> None:
    """No time step may appear on both sides of the temporal boundary."""

    frame = pl.DataFrame(
        {
            "step": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "isFraud": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    development, temporal_test = split_temporally(frame, test_fraction=0.40)

    assert development.get_column("step").unique().sort().to_list() == [1, 2, 3]
    assert temporal_test.get_column("step").unique().sort().to_list() == [4, 5]
    assert development.get_column("step").max() < temporal_test.get_column("step").min()
    assert development.height + temporal_test.height == frame.height


def test_cutoff_uses_distinct_steps_not_row_fraction() -> None:
    """A heavily populated hour must not move the chronological boundary."""

    frame = pl.DataFrame({"step": [1] * 100 + [2, 3, 4, 5]})
    assert determine_temporal_cutoff(frame, "step", 0.20) == 5


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_test_fraction_is_rejected(fraction: float) -> None:
    frame = pl.DataFrame({"step": [1, 2]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        determine_temporal_cutoff(frame, "step", fraction)

