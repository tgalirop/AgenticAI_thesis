"""Tests for reproducible project configuration loading."""

from agenticai_thesis.config import PROJECT_ROOT, load_agent_config, load_data_config


def test_data_config_resolves_repository_relative_paths() -> None:
    config = load_data_config()
    assert config.raw_csv_path == PROJECT_ROOT / "data/raw/paysim.csv"
    assert config.temporal_test_fraction == 0.20
    assert config.target_column == "isFraud"


def test_agent_config_selects_local_open_weight_model() -> None:
    """The checked-in experiment defaults must use the audited Groq adapter."""

    config = load_agent_config()
    assert config.llm.provider == "groq"
    assert config.llm.model == "openai/gpt-oss-20b"
    assert config.llm.api_key_environment_variable == "GROQ_API_KEY"
    assert config.llm.temperature == 0.0
    assert config.structured_output_only is True
    assert config.allow_arbitrary_code_execution is False
    assert "execute_python" not in config.allowed_transformations
