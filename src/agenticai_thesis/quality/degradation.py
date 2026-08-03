"""Reproducible controlled data degradation for the thesis experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ControlledDegradationConfig:
    """Missingness rates injected per development feature."""

    missing_rates: Mapping[str, float]
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.missing_rates:
            raise ValueError("At least one degradation column is required")
        for column, rate in self.missing_rates.items():
            if not column:
                raise ValueError("Degradation column names cannot be empty")
            if not 0.0 < rate < 1.0:
                raise ValueError("Every missing rate must be between zero and one")


@dataclass(frozen=True)
class ControlledDegradationResult:
    """Degraded copy plus compact, non-sensitive injection audit."""

    frame: pd.DataFrame
    affected_rows_by_column: dict[str, int]
    random_seed: int


class ControlledDataDegrader:
    """Inject deterministic MCAR missingness without modifying source data."""

    def __init__(
        self,
        config: ControlledDegradationConfig,
        *,
        protected_columns: frozenset[str] = frozenset(),
    ) -> None:
        overlap = set(config.missing_rates).intersection(protected_columns)
        if overlap:
            raise ValueError(
                "Protected columns cannot be degraded: " + ", ".join(sorted(overlap))
            )
        self._config = config

    def degrade(self, frame: pd.DataFrame) -> ControlledDegradationResult:
        """Return a deep copy with independently sampled missing positions."""

        missing = sorted(set(self._config.missing_rates).difference(frame.columns))
        if missing:
            raise ValueError("Degradation columns not found: " + ", ".join(missing))
        if frame.empty:
            raise ValueError("Cannot degrade an empty dataframe")

        degraded = frame.copy(deep=True)
        rng = np.random.default_rng(self._config.random_seed)
        affected: dict[str, int] = {}
        for column, rate in self._config.missing_rates.items():
            count = max(1, int(round(len(degraded) * rate)))
            positions = rng.choice(len(degraded), size=count, replace=False)
            column_index = degraded.columns.get_loc(column)
            degraded.iloc[positions, column_index] = np.nan
            affected[column] = count
        return ControlledDegradationResult(
            frame=degraded,
            affected_rows_by_column=affected,
            random_seed=self._config.random_seed,
        )
