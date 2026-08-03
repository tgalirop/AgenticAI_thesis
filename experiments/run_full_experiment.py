"""Run the reproducible conventional-versus-Agentic experiment end to end.

This module is intentionally a thin orchestration layer.  Each individual
experiment keeps its own single responsibility, while ``FullExperimentRunner``
coordinates their order, validates prerequisites and stops immediately when a
stage fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from agenticai_thesis.config import PROJECT_ROOT, load_data_config


class CommandExecutorProtocol(Protocol):
    """Boundary used to execute an experiment stage and to isolate it in tests."""

    def execute(self, command: Sequence[str], *, working_directory: Path) -> None:
        """Execute ``command`` or raise when the child process is unsuccessful."""


class SubprocessCommandExecutor:
    """Execute stages in fresh Python processes using the active interpreter."""

    def execute(self, command: Sequence[str], *, working_directory: Path) -> None:
        # check=True is important: a failed stage must never be followed by later
        # stages that could otherwise consume partial or stale artifacts.
        subprocess.run(list(command), cwd=working_directory, check=True)


@dataclass(frozen=True)
class ExperimentStage:
    """Immutable description of one independently executable experiment stage."""

    name: str
    module: str
    arguments: tuple[str, ...]

    def command(self, python_executable: str) -> tuple[str, ...]:
        """Build the interpreter-safe command for this stage."""

        return (python_executable, "-m", self.module, *self.arguments)


class FullExperimentRunner:
    """Coordinate dataset preparation, baselines and the Agentic experiment."""

    VALID_STAGES = ("data", "conventional", "agentic")

    def __init__(
        self,
        *,
        data_config_path: str,
        baseline_config_path: str,
        agent_config_path: str,
        executor: CommandExecutorProtocol | None = None,
        python_executable: str | None = None,
    ) -> None:
        self._data_config_path = data_config_path
        self._baseline_config_path = baseline_config_path
        self._agent_config_path = agent_config_path
        self._executor = executor or SubprocessCommandExecutor()
        self._python_executable = python_executable or sys.executable

    def run(self, selected_stages: Sequence[str] | None = None) -> None:
        """Run selected stages in their required scientific order.

        ``selected_stages`` is useful when the immutable data artifacts already
        exist and only the computationally expensive experiment must be repeated.
        Omitting it executes the complete workflow from the raw dataset onward.
        """

        # ``None`` means that the caller omitted the option and requests the
        # complete workflow.  An explicitly empty sequence is invalid input.
        stages = self.VALID_STAGES if selected_stages is None else tuple(selected_stages)
        self._validate_environment()
        self._validate_stage_selection(stages)
        self._validate_prerequisites(stages)

        started = time.perf_counter()
        stage_definitions = self._stage_definitions()
        for index, stage_name in enumerate(stages, start=1):
            stage = stage_definitions[stage_name]
            print(f"[{index}/{len(stages)}] Running {stage.name} stage...", flush=True)
            self._executor.execute(
                stage.command(self._python_executable),
                working_directory=PROJECT_ROOT,
            )

        print(
            f"Full experiment completed in {time.perf_counter() - started:.2f} seconds.",
            flush=True,
        )

    def _stage_definitions(self) -> dict[str, ExperimentStage]:
        """Return the canonical commands without duplicating experiment logic."""

        return {
            "data": ExperimentStage(
                name="dataset preparation",
                module="experiments.run_data_pipeline",
                arguments=("--config", self._data_config_path),
            ),
            "conventional": ExperimentStage(
                name="conventional benchmark",
                module="experiments.run_conventional",
                arguments=(
                    "--data-config",
                    self._data_config_path,
                    "--baseline-config",
                    self._baseline_config_path,
                ),
            ),
            "agentic": ExperimentStage(
                name="LangGraph Agentic benchmark",
                module="experiments.run_agentic",
                arguments=(
                    "--data-config",
                    self._data_config_path,
                    "--baseline-config",
                    self._baseline_config_path,
                    "--agent-config",
                    self._agent_config_path,
                ),
            ),
        }

    @staticmethod
    def _validate_environment() -> None:
        """Detect a stale installed package before it resolves paths incorrectly."""

        if not (PROJECT_ROOT / "pyproject.toml").is_file():
            raise RuntimeError(
                "The active interpreter is not loading this repository's src package. "
                "Set PYTHONPATH=src or install the project with 'python -m pip install -e .'."
            )

    def _validate_prerequisites(self, stages: Sequence[str]) -> None:
        """Fail early with actionable messages instead of obscure child errors."""

        config_paths = (
            self._data_config_path,
            self._baseline_config_path,
            self._agent_config_path,
        )
        missing_configs = [path for path in config_paths if not (PROJECT_ROOT / path).is_file()]
        if missing_configs:
            raise FileNotFoundError(f"Missing configuration files: {missing_configs}")

        data_config = load_data_config(self._data_config_path)
        if "data" in stages and not data_config.raw_csv_path.is_file():
            raise FileNotFoundError(
                "Raw PaySim CSV not found. Place it at "
                f"'{data_config.raw_csv_path}' or omit the data stage when processed "
                "partitions already exist."
            )

        # When dataset preparation is omitted, downstream stages require the
        # immutable development and temporal-test partitions to exist already.
        if "data" not in stages and any(stage in stages for stage in ("conventional", "agentic")):
            required = (data_config.development_path, data_config.temporal_test_path)
            missing_data = [str(path) for path in required if not path.is_file()]
            if missing_data:
                raise FileNotFoundError(f"Missing processed dataset artifacts: {missing_data}")

    @classmethod
    def _validate_stage_selection(cls, stages: Sequence[str]) -> None:
        """Reject empty, duplicated or out-of-order stage selections."""

        if not stages:
            raise ValueError("At least one experiment stage must be selected")
        unknown = [stage for stage in stages if stage not in cls.VALID_STAGES]
        if unknown:
            raise ValueError(f"Unknown experiment stages: {unknown}")
        if len(set(stages)) != len(stages):
            raise ValueError("Experiment stages must not be repeated")
        positions = [cls.VALID_STAGES.index(stage) for stage in stages]
        if positions != sorted(positions):
            raise ValueError("Experiment stages must follow data, conventional, agentic order")


def main() -> None:
    """Parse CLI options and execute the complete or selected workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=FullExperimentRunner.VALID_STAGES,
        help="Optional ordered subset; default: data conventional agentic",
    )
    args = parser.parse_args()
    FullExperimentRunner(
        data_config_path=args.data_config,
        baseline_config_path=args.baseline_config,
        agent_config_path=args.agent_config,
    ).run(args.stages)


if __name__ == "__main__":
    main()
