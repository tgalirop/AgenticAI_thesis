"""Typed, object-oriented LangGraph orchestration for the Agentic AI loop.

LangGraph controls *when* domain services run; it does not implement profiling,
preprocessing, model fitting, or quality calculations.  Those responsibilities
remain in independently tested classes injected through narrow interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph

from agenticai_thesis.agentic.checkpointing import CheckpointStoreProtocol
from agenticai_thesis.agentic.domain import (
    DatasetContext,
    DatasetRole,
    TransformationPlan,
    ValidationResult,
)
from agenticai_thesis.agentic.feedback import (
    CandidateAssessment,
    FeedbackAction,
    FeedbackDecision,
    FeedbackPolicy,
)
from agenticai_thesis.agentic.prompts import StrategyPromptContext
from agenticai_thesis.agentic.state import (
    AgentRunStatus,
    AgentState,
    AgentStateManager,
    IterationRecord,
)
from agenticai_thesis.agentic.strategy_generator import StrategyGenerator
from agenticai_thesis.agentic.validator import TransformationPlanValidator


class AgentGraphState(TypedDict, total=False):
    """Values exchanged by LangGraph nodes during one complete Agent run.

    The durable ``agent_state`` remains an immutable Pydantic object.  The other
    fields are short-lived values for the current iteration and are overwritten
    on each retry instead of accumulating fitted estimators or raw observations.
    """

    agent_state: AgentState
    dataset_context: DatasetContext
    quality_metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    current_plan: TransformationPlan
    current_validation: ValidationResult
    current_assessment: CandidateAssessment
    current_feedback: FeedbackDecision
    current_errors: tuple[str, ...]


@runtime_checkable
class CandidateEvaluatorProtocol(Protocol):
    """Evaluate one valid plan with deterministic quality and ML services."""

    def evaluate(
        self,
        *,
        validation: ValidationResult,
        dataset_context: DatasetContext,
    ) -> CandidateAssessment:
        """Return compact benefit/cost evidence for the current plan."""


class CandidateEvaluationError(RuntimeError):
    """Expected evaluation failure that the feedback policy may reason about."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


@dataclass(frozen=True, slots=True)
class AgentGraphDependencies:
    """Complete dependency container required to assemble the graph."""

    strategy_generator: StrategyGenerator
    validator: TransformationPlanValidator
    candidate_evaluator: CandidateEvaluatorProtocol
    feedback_policy: FeedbackPolicy
    state_manager: AgentStateManager
    checkpoint_store: CheckpointStoreProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_evaluator, CandidateEvaluatorProtocol):
            raise TypeError("candidate_evaluator must satisfy CandidateEvaluatorProtocol")
        if not isinstance(self.checkpoint_store, CheckpointStoreProtocol):
            raise TypeError("checkpoint_store must satisfy CheckpointStoreProtocol")


class PrepareIterationNode:
    """Fail closed before exposing context to the Strategy Generator."""

    def __call__(self, graph_state: AgentGraphState) -> dict[str, Any]:
        state = graph_state["agent_state"]
        context = graph_state["dataset_context"]
        if state.status != AgentRunStatus.RUNNING:
            raise ValueError("Agent workflow requires a running AgentState")
        if context.role != DatasetRole.DEVELOPMENT:
            raise ValueError("Agent planning is permitted only on development data")
        if context.dataset_id != state.dataset_id:
            raise ValueError("AgentState and DatasetContext refer to different datasets")
        if state.current_iteration >= state.max_iterations:
            raise ValueError("AgentState has no remaining iterations")
        return {}


class GenerateStrategyNode:
    """Build compact evidence and request one typed transformation plan."""

    def __init__(self, generator: StrategyGenerator) -> None:
        self._generator = generator

    def __call__(self, graph_state: AgentGraphState) -> dict[str, Any]:
        state = graph_state["agent_state"]
        feedback_messages = tuple(
            recommendation
            for record in state.history
            for recommendation in record.feedback.recommendations
        )
        prompt_context = StrategyPromptContext(
            dataset=graph_state["dataset_context"],
            iteration=state.current_iteration + 1,
            quality_metrics=graph_state["quality_metrics"],
            baseline_metrics=graph_state["baseline_metrics"],
            previous_feedback=feedback_messages,
        )
        return {"current_plan": self._generator.generate(prompt_context)}


class ValidateStrategyNode:
    """Apply deterministic schema, leakage, allowlist, and parameter rules."""

    def __init__(self, validator: TransformationPlanValidator) -> None:
        self._validator = validator

    def __call__(self, graph_state: AgentGraphState) -> dict[str, Any]:
        validation = self._validator.validate(
            graph_state["current_plan"], graph_state["dataset_context"]
        )
        return {"current_validation": validation}


class InvalidStrategyAssessmentNode:
    """Represent validation failure as typed feedback rather than an exception."""

    def __call__(self, graph_state: AgentGraphState) -> dict[str, Any]:
        plan = graph_state["current_plan"]
        return {
            "current_assessment": CandidateAssessment(
                plan_id=plan.plan_id,
                iteration=plan.iteration,
                validation_valid=False,
                execution_status="success",
            ),
            "current_errors": (),
        }


class EvaluateCandidateNode:
    """Delegate valid-plan execution and evaluation to one injected service."""

    def __init__(self, evaluator: CandidateEvaluatorProtocol) -> None:
        self._evaluator = evaluator

    def __call__(self, graph_state: AgentGraphState) -> dict[str, Any]:
        plan = graph_state["current_plan"]
        try:
            assessment = self._evaluator.evaluate(
                validation=graph_state["current_validation"],
                dataset_context=graph_state["dataset_context"],
            )
            errors: tuple[str, ...] = ()
        except CandidateEvaluationError as exc:
            assessment = CandidateAssessment(
                plan_id=plan.plan_id,
                iteration=plan.iteration,
                validation_valid=True,
                execution_status="fatal_error" if exc.fatal else "recoverable_error",
            )
            errors = (f"{type(exc).__name__}: {exc}",)

        if assessment.plan_id != plan.plan_id or assessment.iteration != plan.iteration:
            raise ValueError("Candidate evaluator returned evidence for a different plan")
        return {"current_assessment": assessment, "current_errors": errors}


class DecideFeedbackNode:
    """Apply deterministic benefit, guardrail, retry, and stopping rules."""

    def __init__(self, policy: FeedbackPolicy) -> None:
        self._policy = policy

    def __call__(self, graph_state: AgentGraphState) -> dict[str, Any]:
        state = graph_state["agent_state"]
        prior_assessments = tuple(record.assessment for record in state.history)
        decision = self._policy.decide(
            graph_state["current_assessment"],
            state.baseline,
            prior_assessments,
        )
        return {"current_feedback": decision}


class RecordIterationNode:
    """Commit one complete audit record and atomically checkpoint the new state."""

    def __init__(
        self,
        state_manager: AgentStateManager,
        checkpoint_store: CheckpointStoreProtocol,
    ) -> None:
        self._state_manager = state_manager
        self._checkpoint_store = checkpoint_store

    def __call__(self, graph_state: AgentGraphState) -> dict[str, Any]:
        plan = graph_state["current_plan"]
        validation = graph_state["current_validation"]
        issues = tuple(f"{issue.code}: {issue.message}" for issue in validation.issues)
        record = IterationRecord(
            iteration=plan.iteration,
            plan=plan,
            validation=validation,
            assessment=graph_state["current_assessment"],
            feedback=graph_state["current_feedback"],
            warnings=issues,
            errors=graph_state.get("current_errors", ()),
        )
        updated = self._state_manager.append_iteration(graph_state["agent_state"], record)
        self._checkpoint_store.save(updated)
        return {"agent_state": updated}


class AgentGraphBuilder:
    """Assemble the workflow from small nodes with explicit routing."""

    def __init__(self, dependencies: AgentGraphDependencies) -> None:
        self._dependencies = dependencies

    def build(self) -> Any:
        """Compile and return a reusable LangGraph application."""

        graph = StateGraph(AgentGraphState)
        graph.add_node("prepare_iteration", PrepareIterationNode())
        graph.add_node(
            "generate_strategy", GenerateStrategyNode(self._dependencies.strategy_generator)
        )
        graph.add_node(
            "validate_strategy", ValidateStrategyNode(self._dependencies.validator)
        )
        graph.add_node("assess_invalid_strategy", InvalidStrategyAssessmentNode())
        graph.add_node(
            "evaluate_candidate", EvaluateCandidateNode(self._dependencies.candidate_evaluator)
        )
        graph.add_node(
            "decide_feedback", DecideFeedbackNode(self._dependencies.feedback_policy)
        )
        graph.add_node(
            "record_iteration",
            RecordIterationNode(
                self._dependencies.state_manager,
                self._dependencies.checkpoint_store,
            ),
        )

        graph.add_edge(START, "prepare_iteration")
        graph.add_edge("prepare_iteration", "generate_strategy")
        graph.add_edge("generate_strategy", "validate_strategy")
        graph.add_conditional_edges(
            "validate_strategy",
            self._route_validation,
            {
                "valid": "evaluate_candidate",
                "invalid": "assess_invalid_strategy",
            },
        )
        graph.add_edge("evaluate_candidate", "decide_feedback")
        graph.add_edge("assess_invalid_strategy", "decide_feedback")
        graph.add_edge("decide_feedback", "record_iteration")
        graph.add_conditional_edges(
            "record_iteration",
            self._route_after_record,
            {"retry": "prepare_iteration", "end": END},
        )
        return graph.compile()

    @staticmethod
    def _route_validation(graph_state: AgentGraphState) -> str:
        return "valid" if graph_state["current_validation"].is_valid else "invalid"

    @staticmethod
    def _route_after_record(graph_state: AgentGraphState) -> str:
        return (
            "retry"
            if graph_state["agent_state"].status == AgentRunStatus.RUNNING
            else "end"
        )


class AgentWorkflow:
    """Public facade that validates inputs and hides LangGraph implementation."""

    _NODES_PER_ITERATION = 6

    def __init__(self, dependencies: AgentGraphDependencies) -> None:
        self._application = AgentGraphBuilder(dependencies).build()

    def run(
        self,
        *,
        agent_state: AgentState,
        dataset_context: DatasetContext,
        quality_metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> AgentState:
        """Run until a deterministic terminal decision and return durable state."""

        initial: AgentGraphState = {
            "agent_state": agent_state,
            "dataset_context": dataset_context,
            "quality_metrics": dict(quality_metrics),
            "baseline_metrics": dict(baseline_metrics),
        }
        # The limit is derived from the domain maximum, with a small allowance
        # for START/END routing. It prevents accidental infinite graph cycles.
        recursion_limit = agent_state.max_iterations * self._NODES_PER_ITERATION + 4
        result = self._application.invoke(
            initial,
            config={"recursion_limit": recursion_limit},
        )
        final_state = result["agent_state"]
        if final_state.status != AgentRunStatus.COMPLETED:
            raise RuntimeError("Agent workflow ended without a terminal decision")
        return final_state
