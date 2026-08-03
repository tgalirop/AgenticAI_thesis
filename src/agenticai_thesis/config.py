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


@dataclass(frozen=True)
class LlmConfig:
    """Validated provider settings used by the Strategy Generator."""

    provider: str
    model: str
    base_url: str
    api_key_environment_variable: str
    timeout_seconds: float
    temperature: float
    reasoning_effort: str
    max_rate_limit_retries: int


@dataclass(frozen=True)
class AgentConfig:
    """Validated top-level safety and model settings for the Agent."""

    max_iterations: int
    structured_output_only: bool
    allow_arbitrary_code_execution: bool
    llm: LlmConfig
    allowed_transformations: tuple[str, ...]


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


def load_agent_config(path: str | Path = "configs/agent.yaml") -> AgentConfig:
    """Load the Agent configuration and enforce non-negotiable safety rules."""

    raw = load_yaml(path)
    llm_raw = raw.get("llm")
    if not isinstance(llm_raw, Mapping):
        raise ValueError("'llm' must be a YAML mapping")

    transformations = raw.get("allowed_transformations")
    if not isinstance(transformations, list) or not transformations:
        raise ValueError("'allowed_transformations' must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in transformations):
        raise ValueError("Every allowed transformation must be a non-empty string")
    if len(transformations) != len(set(transformations)):
        raise ValueError("Allowed transformations must not contain duplicates")

    max_iterations = int(raw.get("max_iterations", 0))
    if max_iterations < 1:
        raise ValueError("'max_iterations' must be positive")
    structured_output_only = raw.get("structured_output_only")
    arbitrary_code = raw.get("allow_arbitrary_code_execution")
    if structured_output_only is not True:
        raise ValueError("Agent requires structured_output_only: true")
    if arbitrary_code is not False:
        raise ValueError("Agent forbids arbitrary code execution")

    provider = llm_raw.get("provider")
    model = llm_raw.get("model")
    base_url = llm_raw.get("base_url")
    if provider != "groq":
        raise ValueError("The checked-in Agent configuration requires provider: groq")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("'llm.model' must be a non-empty string")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValueError("'llm.base_url' must be an HTTP(S) URL")

    api_key_variable = llm_raw.get("api_key_environment_variable")
    if api_key_variable != "GROQ_API_KEY":
        raise ValueError("Groq credentials must be read from GROQ_API_KEY")

    timeout_seconds = float(llm_raw.get("timeout_seconds", 120.0))
    temperature = float(llm_raw.get("temperature", 0.0))
    reasoning_effort = str(llm_raw.get("reasoning_effort", "medium"))
    max_rate_limit_retries = int(llm_raw.get("max_rate_limit_retries", 2))
    if timeout_seconds <= 0:
        raise ValueError("'llm.timeout_seconds' must be positive")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("'llm.temperature' must be between 0 and 2")
    if reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError("'llm.reasoning_effort' must be low, medium, or high")
    if max_rate_limit_retries < 0:
        raise ValueError("'llm.max_rate_limit_retries' cannot be negative")

    return AgentConfig(
        max_iterations=max_iterations,
        structured_output_only=structured_output_only,
        allow_arbitrary_code_execution=arbitrary_code,
        llm=LlmConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_environment_variable=api_key_variable,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_rate_limit_retries=max_rate_limit_retries,
        ),
        allowed_transformations=tuple(transformations),
    )
