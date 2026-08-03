"""Tests for the fail-fast full-experiment orchestration layer."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from experiments.run_full_experiment import FullExperimentRunner


class RecordingExecutor:
    """Test double that records commands without running expensive experiments."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def execute(self, command: Sequence[str], *, working_directory: Path) -> None:
        self.calls.append((tuple(command), working_directory))


def _runner(executor: RecordingExecutor) -> FullExperimentRunner:
    return FullExperimentRunner(
        data_config_path="configs/data.yaml",
        baseline_config_path="configs/baseline.yaml",
        agent_config_path="configs/agent.yaml",
        executor=executor,
        python_executable="python-test",
    )


def test_selected_processed_data_stages_execute_in_scientific_order() -> None:
    executor = RecordingExecutor()

    _runner(executor).run(("conventional", "agentic"))

    modules = [command[2] for command, _ in executor.calls]
    assert modules == ["experiments.run_conventional", "experiments.run_agentic"]
    assert all(command[:2] == ("python-test", "-m") for command, _ in executor.calls)


@pytest.mark.parametrize(
    "stages, message",
    [
        ((), "At least one"),
        (("agentic", "conventional"), "must follow"),
        (("agentic", "agentic"), "must not be repeated"),
        (("unknown",), "Unknown"),
    ],
)
def test_invalid_stage_selection_fails_before_execution(
    stages: tuple[str, ...], message: str
) -> None:
    executor = RecordingExecutor()

    with pytest.raises(ValueError, match=message):
        _runner(executor).run(stages)

    assert executor.calls == []


def test_agentic_command_forwards_all_configuration_paths() -> None:
    executor = RecordingExecutor()

    _runner(executor).run(("agentic",))

    command, _ = executor.calls[0]
    assert command == (
        "python-test",
        "-m",
        "experiments.run_agentic",
        "--data-config",
        "configs/data.yaml",
        "--baseline-config",
        "configs/baseline.yaml",
        "--agent-config",
        "configs/agent.yaml",
    )
