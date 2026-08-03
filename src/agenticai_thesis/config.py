"""Configuration loading and validation utilities.

Keeping paths and experiment parameters in YAML files makes every run easier to
reproduce.  This module centralises YAML parsing and resolves data paths relative
to the repository root, rather than relative to the caller's current directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


# config.py lives in ``<repository>/src/agenticai_thesis``.  ``parents[2]`` is
# therefore the repository root.  Deriving the path here allows commands to be
# launched from another working directory without silently reading wrong files.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    """Validated paths and parameters required by the dataset pipeline."""

    raw_csv_path: Path
    parquet_path: Path
    development_path: Path
    temporal_test_path: Path
    target_column: str
    time_column: str
    temporal_test_fraction: float
    random_seed: int


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping and fail early when its top level is invalid."""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream)

    if not isinstance(content, Mapping):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    return dict(content)


def _project_path(value: object, field_name: str) -> Path:
    """Resolve one required YAML path relative to the repository root."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    """Load and validate the configuration used by all data commands."""

    raw = load_yaml(path)
    required = {
        "raw_csv_path",
        "parquet_path",
        "development_path",
        "temporal_test_path",
        "target_column",
        "time_column",
        "temporal_test_fraction",
        "random_seed",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing data configuration keys: {', '.join(missing)}")

    test_fraction = float(raw["temporal_test_fraction"])
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("'temporal_test_fraction' must be between 0 and 1")

    return DataConfig(
        raw_csv_path=_project_path(raw["raw_csv_path"], "raw_csv_path"),
        parquet_path=_project_path(raw["parquet_path"], "parquet_path"),
        development_path=_project_path(raw["development_path"], "development_path"),
        temporal_test_path=_project_path(raw["temporal_test_path"], "temporal_test_path"),
        target_column=str(raw["target_column"]),
        time_column=str(raw["time_column"]),
        temporal_test_fraction=test_fraction,
        random_seed=int(raw["random_seed"]),
    )
