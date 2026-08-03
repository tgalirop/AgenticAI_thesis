"""Tests for reproducible project configuration loading."""

from agenticai_thesis.config import PROJECT_ROOT, load_data_config


def test_data_config_resolves_repository_relative_paths() -> None:
    config = load_data_config()
    assert config.raw_csv_path == PROJECT_ROOT / "data/raw/paysim.csv"
    assert config.temporal_test_fraction == 0.20
    assert config.target_column == "isFraud"

