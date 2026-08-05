"""Tests for production composition of plan, quality, and ML evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from agenticai_thesis.agentic.candidate_evaluation import (
    AgentCandidateEvaluator,
    PlanQualityEvaluation,
)
from agenticai_thesis.agentic.domain import (
    ColumnKind,
    DatasetContext,
    DatasetRole,
    TransformationAction,
    TransformationPlan,
)
from agenticai_thesis.agentic.evaluators import EvaluationConfig, MachineLearningEvaluator
from agenticai_thesis.agentic.execution import ModelPipelineBuilder, TransformationFactoryRegistry
from agenticai_thesis.agentic.executor import TransformationExecutor
from agenticai_thesis.agentic.transformations import TransformationRegistry
from agenticai_thesis.agentic.validator import TransformationPlanValidator
from agenticai_thesis.modeling.cross_validation import CrossValidationFoldProvider


class FixedQualityEvaluator:
    """Small deterministic quality service used only by this composition test."""

    def evaluate(self, *, validation: object, dataset_context: DatasetContext) -> PlanQualityEvaluation:
        assert validation is not None
        assert dataset_context.role == DatasetRole.DEVELOPMENT
        return PlanQualityEvaluation(data_quality_score=0.96, evaluation_seconds=0.02)


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    features = pd.DataFrame(
        {
            "amount": [10, 12, 15, 18, 20, 25, 90, 100, 110, 120, 130, 140],
        }
    )
    target = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    return features, target


def _context(role: DatasetRole = DatasetRole.DEVELOPMENT) -> DatasetContext:
    return DatasetContext(
        dataset_id="development-v1",
        role=role,
        column_types={"amount": ColumnKind.NUMERIC, "isFraud": ColumnKind.TARGET},
        target_column="isFraud",
        protected_columns=frozenset({"isFraud"}),
    )


def _validation() -> object:
    plan = TransformationPlan(
        plan_id="plan_001",
        dataset_id="development-v1",
        iteration=1,
        objective="Scale numeric fraud features.",
        actions=(
            TransformationAction(
                action_id="scale_amount",
                transformation="scale_numeric",
                columns=("amount",),
                parameters={"method": "robust"},
                rationale="Reduce sensitivity to extreme amounts.",
                model_scope="logistic_regression",
            ),
        ),
    )
    return TransformationPlanValidator(TransformationRegistry.default()).validate(plan, _context())


def _evaluator() -> AgentCandidateEvaluator:
    features, target = _data()
    folds = CrossValidationFoldProvider(folds=2, repeats=1, random_seed=42).create(target)
    return AgentCandidateEvaluator(
        executor=TransformationExecutor(
            ModelPipelineBuilder(TransformationFactoryRegistry.default())
        ),
        quality_evaluator=FixedQualityEvaluator(),
        ml_evaluator=MachineLearningEvaluator(EvaluationConfig()),
        estimators={
            "logistic_regression": LogisticRegression(max_iter=200, random_state=42)
        },
        features=features,
        target=target,
        folds=folds,
    )


def test_candidate_evaluator_composes_quality_and_fold_local_ml() -> None:
    assessment = _evaluator().evaluate(
        validation=_validation(),  # type: ignore[arg-type]
        dataset_context=_context(),
    )

    assert assessment.execution_status == "success"
    assert assessment.data_quality_score == pytest.approx(0.96)
    assert assessment.preprocessing_seconds >= 0.02
    assert assessment.peak_memory_bytes is not None
    assert assessment.peak_memory_bytes > 0
    assert assessment.memory_rss_start_bytes is not None
    assert assessment.memory_peak_increase_bytes is not None
    assert assessment.peak_memory_bytes >= assessment.memory_rss_start_bytes
    assert assessment.memory_peak_increase_bytes == (
        assessment.peak_memory_bytes - assessment.memory_rss_start_bytes
    )
    assert len(assessment.model_outcomes) == 1
    assert assessment.model_outcomes[0].status == "success"
    assert assessment.model_outcomes[0].primary_metric is not None


def test_candidate_evaluator_rejects_temporal_test_role() -> None:
    with pytest.raises(ValueError, match="development data"):
        _evaluator().evaluate(
            validation=_validation(),  # type: ignore[arg-type]
            dataset_context=_context(DatasetRole.TEMPORAL_TEST),
        )
